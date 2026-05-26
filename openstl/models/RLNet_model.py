import torch
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST)


class RLNet_Model(nn.Module):
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape  # [4,10,1,128,128]
        x = x_raw.view(B*T, C, H, W)  # [4*10,1,128,128]

        embed, skip = self.enc(x)  # [4*10,64,128,128]
        _, C_, H_, W_ = embed.shape  # [4*10,64,128,128]

        z = embed.view(B, T, C_, H_, W_)  # [4,10,64,128,128]
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


# RLNet
class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self,
                 channel_in,
                 channel_hid,
                 N2,
                 incep_ker=[3, 5, 7, 11],
                 groups=8,
                 T_in=10,
                 use_tscb=True,
                 tscb_mid_ratio=0.5,
                 **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        self.T_in = T_in

        enc_layers = [gInception_ST(
            channel_in,
            channel_hid // 2,
            channel_hid,
            incep_ker=incep_ker,
            groups=groups,
            use_tscb=use_tscb,
            T_in=T_in,
            tscb_mid_ratio=tscb_mid_ratio
        )]

        for i in range(1, N2 - 1):
            enc_layers.append(
                gInception_ST(
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups
                )
            )

        enc_layers.append(
            gInception_ST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups
            )
        )

        dec_layers = [
            gInception_ST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups
            )
        ]

        for i in range(1, N2 - 1):
            dec_layers.append(
                gInception_ST(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups
                )
            )

        dec_layers.append(
            gInception_ST(
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                incep_ker=incep_ker,
                groups=groups
            )
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
        
        y = z.reshape(B, T, C, H, W)
        return y
