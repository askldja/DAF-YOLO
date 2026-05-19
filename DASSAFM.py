import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class DASSAFM(nn.Module):
    def __init__(self, c1, c3, c4, c5, use_refine=True, r=4):
        super().__init__()
        assert c1 == (c3 + c4 + c5)
        self.c3, self.c4, self.c5 = c3, c4, c5
        self.use_refine = use_refine

        self.proj4 = Conv(c4, c3, k=1)
        self.proj5 = Conv(c5, c3, k=1)

        # 密度特征提取：DW+PW替代两个标准Conv3×3
        self.density_stem = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1, groups=c3, bias=False),
            nn.Conv2d(c3, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU()
        )

        # 密度预测头
        self.density_pred = nn.Conv2d(c3, 1, 1)

        # 三尺度logit预测
        self.logit3 = nn.Conv2d(c3, 1, 1)
        self.logit4 = nn.Conv2d(c3, 1, 1)
        self.logit5 = nn.Conv2d(c3, 1, 1)

        # 密度偏移系数（可学习）
        self.k = nn.Parameter(torch.tensor(1.0))

        if use_refine:
            hidden = max(c3 // 16, 8)
            # 通道注意力：只用avg路
            self.ch_mlp = nn.Sequential(
                nn.Conv2d(c3, hidden, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(hidden, c3, 1, bias=False),
            )
            # 空间注意力：3×3替代7×7
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(c3, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            self.out_conv = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1, groups=c3, bias=False),  # DWConv
            nn.Conv2d(c3, c3, 1, bias=False),                    # PWConv
            nn.BatchNorm2d(c3),
            nn.SiLU()
        )

    def channel_refine(self, x):
        # 只用avg pool，去掉max pool路
        avg = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.ch_mlp(avg))
        return x * w

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 3
            p3, p4, p5 = x
        else:
            p3, p4, p5 = torch.split(
                x, [self.c3, self.c4, self.c5], dim=1
            )

        # 统一通道到c3
        p4p = self.proj4(p4)
        p5p = self.proj5(p5)

        # ✅ 上采样到p3的空间尺寸再相加
        H, W = p3.shape[2], p3.shape[3]
        p4p_up = F.interpolate(p4p, size=(H, W), mode='nearest')
        p5p_up = F.interpolate(p5p, size=(H, W), mode='nearest')

        # 密度图生成
        dens_feat = self.density_stem(p3 + p4p_up + p5p_up)
        D = torch.sigmoid(self.density_pred(dens_feat))

        # 三尺度logit
        l3 = self.logit3(p3)
        l4 = self.logit4(p4p_up)
        l5 = self.logit5(p5p_up)

        # 密度偏移
        bias = self.k * (D - 0.5)
        logits = torch.cat([l3 + bias, l4, l5 - bias], dim=1)
        w = torch.softmax(logits, dim=1)
        w3, w4, w5 = w[:, 0:1], w[:, 1:2], w[:, 2:3]

        # 密度感知软注意力加权融合
        fused = w3 * p3 + w4 * p4p_up + w5 * p5p_up

        if self.use_refine:
            fused = self.channel_refine(fused)
            fused = fused * self.spatial_attn(fused)
            enh_p3 = p3 + self.out_conv(fused)
        else:
            enh_p3 = p3 + fused

        return enh_p3

class DASSAFMBlock(nn.Module):
    def __init__(self, c1, c3, c4, c5, use_refine=True, r=4):
        super().__init__()
        self.m = DASSAFM(c1, c3, c4, c5,
                         use_refine=use_refine, r=r)

    def forward(self, x):
        return self.m(x)