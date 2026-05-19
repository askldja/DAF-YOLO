import torch
import torch.nn as nn


# -----------------------------
# Basic branches
# -----------------------------

class LightweightMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CKS(nn.Module):
    """Channel-Spatial Co-Selection block"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn


class SeparableDilatedLargeKernelBranch(nn.Module):
    """Separable Dilated Large-Kernel Attention branch"""

    def __init__(self, channels, kernel_size=21, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding,
                              groups=channels, dilation=dilation)
        self.pointwise = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.pointwise(self.conv(x)))


# -----------------------------
# SDLKA Variants
# -----------------------------

class SDLKA_S(nn.Module):
    """Small SDLKA: only large-kernel branch"""

    def __init__(self, channels):
        super().__init__()
        self.lkb = SeparableDilatedLargeKernelBranch(channels)

    def forward(self, x):
        return self.lkb(x)


class SDLKA_M(nn.Module):
    """Medium SDLKA: large-kernel branch + lightweight MLP"""

    def __init__(self, channels):
        super().__init__()
        self.lkb = SeparableDilatedLargeKernelBranch(channels)
        self.mlp = LightweightMLP(channels)

    def forward(self, x):
        return self.lkb(x) + self.mlp(x)


class SDLKA_L(nn.Module):
    """Large SDLKA: large-kernel branch + CKS + MLP"""

    def __init__(self, channels):
        super().__init__()
        self.lkb = SeparableDilatedLargeKernelBranch(channels)
        self.cks = CKS(channels)
        self.mlp = LightweightMLP(channels)

    def forward(self, x):
        return self.cks(self.lkb(x)) + self.mlp(x)


# -----------------------------
# C3k2 block wrapper
# -----------------------------

class C3k2_SDLKA_L(nn.Module):
    def __init__(self, channels, shortcut=False):
        super().__init__()
        self.module = SDLKA_L(channels)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.module(x)
        return x + y if self.shortcut else y


class C3k2_SDLKA_M_Scaled(nn.Module):
    def __init__(self, channels, shortcut=False):
        super().__init__()
        self.module = SDLKA_M(channels)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.module(x)
        return x + y if self.shortcut else y


class C3k2_SDLKA_S(nn.Module):
    def __init__(self, channels, shortcut=False):
        super().__init__()
        self.module = SDLKA_S(channels)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.module(x)
        return x + y if self.shortcut else y