import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        hidden = max(c // r, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class CrossScaleFusion(nn.Module):
    def __init__(self, c3, c4, c5, hid=96):   # ✅ 128→96
        super().__init__()
        self.p3_proj = Conv(c3, hid, 1)
        self.p4_proj = Conv(c4, hid, 1)
        self.p5_proj = Conv(c5, hid, 1)
        self.fuse = nn.Sequential(
            Conv(hid * 3, hid, 1),
            Conv(hid, hid, 3)                  # 保持标准Conv，和原始一致
        )
        self.out3 = Conv(hid, c3, 1)
        self.out4 = Conv(hid, c4, 1)
        self.out5 = Conv(hid, c5, 1)

    def forward(self, p3, p4, p5):
        p3h = self.p3_proj(p3)
        p4h = F.interpolate(self.p4_proj(p4), scale_factor=2, mode="nearest")
        p5h = F.interpolate(self.p5_proj(p5), scale_factor=4, mode="nearest")
        ctx = self.fuse(torch.cat([p3h, p4h, p5h], dim=1))

        H4, W4 = p4.shape[2], p4.shape[3]
        H5, W5 = p5.shape[2], p5.shape[3]

        p3 = p3 + self.out3(ctx)
        # ✅ 唯一修复：max_pool2d → adaptive_avg_pool2d，消除NaN
        p4 = p4 + self.out4(F.adaptive_avg_pool2d(ctx, (H4, W4)))
        p5 = p5 + self.out5(F.adaptive_avg_pool2d(ctx, (H5, W5)))
        return p3, p4, p5


class LSDDetect(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        assert len(ch) == 3

        c3, c4, c5 = ch
        self.fusion = CrossScaleFusion(c3, c4, c5, hid=96)  # ✅ 128→96

        reg_mid = [max(c // 2, 64) for c in ch]   # 不动
        cls_mid = [max(c // 2, 64) for c in ch]   # 不动

        self.reg_stem = nn.ModuleList(
            Conv(c, reg_mid[i], 1) for i, c in enumerate(ch)
        )
        self.cls_stem = nn.ModuleList(
            Conv(c, cls_mid[i], 1) for i, c in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DWConv(reg_mid[i], reg_mid[i], 3),  # ✅ 第一个Conv→DWConv
                Conv(reg_mid[i], reg_mid[i], 3),    # 第二个保留标准Conv
                nn.Conv2d(reg_mid[i], 4 * self.reg_max, 1)
            ) for i in range(self.nl)
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(cls_mid[i], cls_mid[i], 3),
                SEBlock(cls_mid[i]),
                nn.Conv2d(cls_mid[i], self.nc, 1)
            ) for i in range(self.nl)
        )

    def forward(self, x):
        if self.end2end:
            return self.forward_end2end(x)

        p3, p4, p5 = x
        p3, p4, p5 = self.fusion(p3, p4, p5)
        x = [p3, p4, p5]

        for i in range(self.nl):
            reg = self.cv2[i](self.reg_stem[i](x[i]))
            cls = self.cv3[i](self.cls_stem[i](x[i]))
            x[i] = torch.cat((reg, cls), 1)

        if self.training:
            return x

        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(
                5 / self.nc / (640 / s.item()) ** 2
            )