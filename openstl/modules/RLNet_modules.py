import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.convnext import ConvNeXtBlock
from timm.models.mlp_mixer import MixerBlock
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
from timm.models.vision_transformer import Block as ViTBlock


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


# RLNet
def _get_valid_gn_groups(c, max_groups=8):
    g = min(max_groups, c)
    while g > 1:
        if c % g == 0:
            return g
        g -= 1
    return 1


class TemporalChannelBottleneck(nn.Module):
    def __init__(self,
                 C_in,  # 640
                 C_hid, # 128
                 T_in,
                 groups=8,
                 mid_ratio=0.5,
                 act_inplace=True):
        super().__init__()


        self.T_in = T_in
        self.C_per_t = C_in // T_in

        target_mid = max(int(C_hid * mid_ratio), T_in * 4)
        mid_per_t = math.ceil(target_mid / T_in)
        C_mid = T_in * mid_per_t

        # Step 1
        self.local_mix = nn.Sequential(
            nn.Conv2d(
                C_in,
                C_mid,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=T_in,
                bias=False
            ),
            nn.GroupNorm(_get_valid_gn_groups(C_mid, groups), C_mid),
            nn.LeakyReLU(0.2, inplace=act_inplace)
        )

        # Step 2
        self.global_mix = nn.Conv2d(
            C_mid,
            C_hid,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        x = self.local_mix(x)
        x = self.global_mix(x)
        return x


# BranchCat
class gInception_ST(nn.Module):
    """An Inception block for SimVP with cat fusion"""

    def __init__(self,
                 C_in,
                 C_hid,
                 C_out,
                 incep_ker=[3, 5, 7, 11],
                 groups=8,
                 use_tscb=False,
                 T_in=10,
                 tscb_mid_ratio=0.5,
                 act_inplace=True):
        super(gInception_ST, self).__init__()

        if use_tscb:
            self.conv1 = TemporalChannelBottleneck(
                C_in=C_in,
                C_hid=C_hid,
                T_in=T_in,
                groups=groups,
                mid_ratio=tscb_mid_ratio,
                act_inplace=act_inplace
            )
        else:
            self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        self.n_branch = len(incep_ker)
        branch_out = C_out // self.n_branch

        branch_channels = [branch_out] * self.n_branch
        branch_channels[-1] += C_out - branch_out * self.n_branch

        layers = []
        for ker, ch_out in zip(incep_ker, branch_channels):
            layers.append(
                GroupConv2d(
                    C_hid, ch_out,
                    kernel_size=ker,
                    stride=1,
                    padding=ker // 2,
                    groups=groups,
                    act_norm=True
                )
            )

        self.layers = nn.ModuleList(layers)

        self.fuse = nn.Conv2d(C_out, C_out, kernel_size=1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        ys = [layer(x) for layer in self.layers]
        y = torch.cat(ys, dim=1)
        y = self.fuse(y) + shortcut
        return y


