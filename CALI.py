import math
import numpy as np
from einops import rearrange
from basicsr.archs.arch_util import flow_warp

import torch
import torch.nn.functional as F
import torch.nn as nn


class EnhancedChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )
        self.dynamic_gate = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        gate = self.dynamic_gate(out)
        gate = gate.unsqueeze(-1).unsqueeze(-1) 

        gate_expanded = gate.expand_as(x)

        return x * gate_expanded

class CALI(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CALI, self).__init__()
        self.eca = EnhancedChannelAttention(in_channels, reduction_ratio)

    def forward(self, x):
        omega = 0.45
        x_1 = self.eca(x)
        x_2 = self.eca(x_1)
        return (1 - omega) * x + omega * x_2