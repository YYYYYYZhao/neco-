import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
            self,
            nclass,
            in_channels,
            features=256,
            use_bn=False,
            out_channels=[256, 512, 1024, 1024]
    ):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv(path_1)
        return out, path_1


class ABGuidanceNetwork(nn.Module):
    def __init__(self, in_dim=384, mid_dim=32):
        super().__init__()
        # 1. 投影层 (384 -> 32)
        self.projects = nn.ModuleList([nn.Conv2d(in_dim, mid_dim, 1) for _ in range(4)])

        # 2. 级联融合卷积 (18x18 尺度)
        self.fuse_l43 = nn.Sequential(nn.Conv2d(mid_dim, mid_dim, 3, 1, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.fuse_l32 = nn.Sequential(nn.Conv2d(mid_dim, mid_dim, 3, 1, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.fuse_l21 = nn.Sequential(nn.Conv2d(mid_dim, mid_dim, 3, 1, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))

        # 3. 逐步上采样到 144
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(mid_dim, mid_dim, 4, 2, 1),  # 18 -> 36
            nn.BatchNorm2d(mid_dim), nn.ReLU(True),
            nn.ConvTranspose2d(mid_dim, mid_dim, 4, 2, 1),  # 36 -> 72
            nn.BatchNorm2d(mid_dim), nn.ReLU(True),
            nn.ConvTranspose2d(mid_dim, mid_dim, 4, 2, 1),  # 72 -> 144
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1)
        )
        self.ws = 9
        self.scale = math.sqrt(mid_dim)

    def _cascade_fusion(self, f_list):
        p1, p2, p3, p4 = [self.projects[i](f_list[i]) for i in range(4)]
        f43 = self.fuse_l43(p4 + p3)
        f432 = self.fuse_l32(f43 + p2)
        f_base = self.fuse_l21(f432 + p1)
        return self.upsample(f_base)

    def forward(self, featsA, featsB):
        # 得到两张 144 尺度的融合特征图
        fa = self._cascade_fusion(featsA)
        fb = self._cascade_fusion(featsB)

        N, C, H, W = fa.shape
        pad = self.ws // 2
        # 算 A 与 B 之间的 9x9 局部点积
        fb_padded = F.pad(fb, (pad, pad, pad, pad), mode='reflect')
        fb_unfold = F.unfold(fb_padded, kernel_size=self.ws).view(N, C, 81, H * W)
        fa_center = fa.view(N, C, H * W).unsqueeze(2)
        # ab_logits: (N, 81, HW) -> 描述 A 在 B 里的物理匹配得分
        ab_logits = (fa_center * fb_unfold).sum(dim=1) / self.scale
        return ab_logits

class DPT(nn.Module):
    def __init__(
            self,
            encoder_size='base',
            nclass=21,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False
    ):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }
        self.ab_guidance = ABGuidanceNetwork(in_dim=384, mid_dim=32)

        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)
        self.corr = Corr()
        self.representation = nn.Sequential(
            nn.Conv2d(64, 16, 1),
        )
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x1, x2, comp_drop=False, corr=False):
        h, w = x1.shape[2:]

        # 1. 提取原始多尺度特征
        featuress = self.backbone.get_intermediate_layers(
            torch.cat((x1, x2)),
            self.intermediate_layer_idx[self.encoder_size],
            return_class_token=False,
            reshape=True
        )  # (2B, C, H, W)
        featsA, featsB, features = [], [], []
        for feature in featuress:
            A, B = feature.chunk(2)
            featsA.append(A)
            featsB.append(B)
            features.append(A - B)  # 原有 A-B 逻辑
        ab_logits = self.ab_guidance(featsA, featsB)
        # 4. 原有的补差/Dropout 逻辑 (保持不动)
        if comp_drop:
            bs, dim = features[0].shape[:2]
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).cuda() * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            features = [feature * dropout_mask[..., None, None] for feature in features]
        out, delta = self.head(features)
        path = self.representation(delta)
        out_interp = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        if corr:
            outs = self.corr(delta, out, ab_logits)
            outs = F.interpolate(outs, size=(h, w), mode='bilinear', align_corners=True)
            return out_interp, outs, path
        return out_interp, path


class Corr(nn.Module):
    def __init__(self, window_size=9, delta_scale=1.0):
        super(Corr, self).__init__()
        self.window_size = window_size
        self.delta_scale = delta_scale
        self.conv1 = nn.Conv2d(64, 16, 1)
        self.conv2 = nn.Conv2d(64, 16, 1)

        # 【关键】引入可学习的温度系数，用于自动平衡距离和点积的量级
        self.gamma_d = nn.Parameter(torch.ones([]))  # 距离的权重
        self.gamma_s = nn.Parameter(torch.ones([]))  # 点积的权重

    def forward(self, delta, out, ab_logits):
        N, C, H, W = delta.shape
        ws = self.window_size
        pad = ws // 2

        # --- 权重 1: 基于距离的 delta 相似度 ---
        f1 = self.conv1(delta)
        f2 = self.conv2(delta)
        f2_padded = F.pad(f2, (pad, pad, pad, pad), mode='reflect')
        f2_unfold = F.unfold(f2_padded, kernel_size=ws).view(N, 16, 81, H * W)
        d_center = f1.view(N, 16, H * W).unsqueeze(2)

        # delta_logits: 负值，越接近 0 越相似
        delta_logits = ((d_center - f2_unfold) ** 2).sum(dim=1) * -1.0

        # --- 双权重共同决定 ---
        # 逻辑：Total = alpha * (-Distance^2) + beta * (DotProduct)
        # 加法在 Logit 空间对应概率空间的“且”逻辑
        combined_attn = F.softmax(
            self.gamma_d * delta_logits + self.gamma_s * ab_logits,
            dim=1
        )

        # --- 最终重组 ---
        out_detach = F.interpolate(out.detach(), size=(H, W), mode='bilinear', align_corners=True)
        out_padded = F.pad(out_detach, (pad, pad, pad, pad), mode='reflect')
        out_unfold = F.unfold(out_padded, kernel_size=ws).view(N, 2, 81, H * W)

        out_refine = (out_unfold * combined_attn.unsqueeze(1)).sum(dim=2)
        return out_refine.view(N, 2, H, W)
def compute_reco_loss(rep, label, mask, prob, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
    batch_size, num_feat, im_w_, im_h = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    valid_pixel = label * mask  # 保留有效像素

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)  # （B,H,W,C）

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []  # 每类所有有效像素特征
    seg_feat_hard_list = []  # 每类难例特征
    seg_num_list = []  # 每类像素数量
    seg_proto_list = []  # 每类原型（均值特征）
    for i in range(num_segments):  # 遍历i类 （变化/未变化）
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class遍历了当前类的有效像素 i是当前类 #[B, H, W]
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = prob[:, i, :, :]  # 当前类预测概率（B,H,W）
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries找到困难的向量

        seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))  # [1, C] 保存每个像素的原型！！！
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()])  # 保存该类所有有效像素特征[N_selected, C]
        seg_feat_hard_list.append(rep[rep_mask_hard])  # 保存该类难例特征[N_hard, C]
        seg_num_list.append(int(valid_pixel_seg.sum().item()))  # 保存该类有效像素数量（整数）

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)  # 类别原型进行拼接(变化检测就是[2,c])
        valid_seg = len(seg_num_list)  # 当前 batch 中有效类别数量
        seg_len = torch.arange(valid_seg)  # 类别索引

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:  # 判断是否有难例特征
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]),
                                             size=(num_queries,))  # 生成索引（取num_queries个向量的索引）
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]  # shape = [num_queries, C]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]],
                                                    dim=1)  # 计算当前类原型与其他类别原型的余弦相似度。
                proto_prob = torch.softmax(proto_sim / temp, dim=0)  # 计算类别之间相似度大小，相似度大，就多采样（这个对于二分类可以删去）
                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)  # 对二分类也是没用的
                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i + 1:] + seg_num_list[:i]  # 其他类别有效像素数量
                negative_index = negative_index_sampler(samp_num, negative_num_list)
                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i + 1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)
                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)
            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        return reco_loss / valid_seg


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j + 1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index


def label_onehot(inputs, num_segments):
    """
    将 label 转为 one-hot 编码，同时避免越界错误。

    inputs: [B, H, W] int64 Tensor, 可能含 -1 或超出类别范围的值
    num_segments: 类别数
    返回:
        outputs: [B, num_segments, H, W] one-hot Tensor
        valid_mask: [B, 1, H, W] 有效像素掩码 (1 表示有效, 0 表示无效)
    """
    batch_size, im_h, im_w = inputs.shape
    # 创建有效掩码
    valid_mask = (inputs >= 0) & (inputs < num_segments)

    # clamp 到合法范围，避免 scatter 越界
    inputs_clamped = torch.clamp(inputs, 0, num_segments - 1)

    # 初始化 one-hot
    outputs = torch.zeros(batch_size, num_segments, im_h, im_w, device=inputs.device, dtype=torch.float)
    outputs.scatter_(1, inputs_clamped.unsqueeze(1), 1.0)

    # 将无效像素置 0
    outputs *= valid_mask.unsqueeze(1).float()

    return outputs, valid_mask.unsqueeze(1)
