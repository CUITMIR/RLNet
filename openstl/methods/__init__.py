# Copyright (c) CAIRI AI Lab. All rights reserved

from .convlstm import ConvLSTM
from .e3dlstm import E3DLSTM
from .mau import MAU
from .mim import MIM
from .phydnet import PhyDNet
from .predrnn import PredRNN
from .predrnnpp import PredRNNpp
from .predrnnv2 import PredRNNv2
from .RLNet import RLNet
from .tau import TAU
from .mmvp import MMVP
from .swinlstm import SwinLSTM_D, SwinLSTM_B
from .wast import WaST
from .earthformer import Earthformer
from .rainhcnet import RainHCNet

method_maps = {
    'convlstm': ConvLSTM,
    'e3dlstm': E3DLSTM,
    'mau': MAU,
    'mim': MIM,
    'phydnet': PhyDNet,
    'predrnn': PredRNN,
    'predrnnpp': PredRNNpp,
    'predrnnv2': PredRNNv2,
    'rlnet': RLNet,
    'tau': TAU,
    'mmvp': MMVP,
    'swinlstm_d': SwinLSTM_D,
    'swinlstm_b': SwinLSTM_B,
    'swinlstm': SwinLSTM_B,
    'wast': WaST,
    'earthformer': Earthformer,
    'rainhcnet': RainHCNet,
}

__all__ = [
    'method_maps', 'ConvLSTM', 'E3DLSTM', 'MAU', 'MIM',
    'PredRNN', 'PredRNNpp', 'PredRNNv2', 'PhyDNet', 'RLNet', 'TAU',
    "MMVP", 'SwinLSTM_D', 'SwinLSTM_B', 'WaST', 'Earthformer', 'RainHCNet'
]
