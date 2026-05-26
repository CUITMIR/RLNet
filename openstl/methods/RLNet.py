import torch
from openstl.models import SimVP_Model
from .base_method import Base_method
import torch.nn.functional as F
import torch.nn as nn


def _js_divergence(p, q, eps=1e-8):
    """
    p, q: [N, D], each row sums to 1
    return: scalar
    """
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1) +
        (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
    )
    return js.mean()


def _to_mass_prob(x, weight=None, pool=1, eps=1e-8):
    """
    x: [B, T-1, C, H, W], non-negative
    weight: same shape as x or broadcastable
    """
    if weight is not None:
        x = x * weight

    B, Tm1, C, H, W = x.shape
    x = x.reshape(B * Tm1, C, H, W)

    if pool > 1:
        x = F.max_pool2d(x, kernel_size=pool, stride=pool)

    x = x.flatten(1)           # [B*(T-1), D]
    x = x + eps                # avoid all-zero distribution
    x = x / x.sum(dim=-1, keepdim=True)
    return x

def radar_diff_div_reg(pred_y, batch_y, scales=(1, 2), eps=1e-8):
    B, T, C, H, W = pred_y.shape
    if T <= 1:
        return pred_y.new_tensor(0.0)

    d_pred = pred_y[:, 1:] - pred_y[:, :-1]
    d_true = batch_y[:, 1:] - batch_y[:, :-1]

    grow_pred = F.relu(d_pred)
    grow_true = F.relu(d_true)

    decay_pred = F.relu(-d_pred)
    decay_true = F.relu(-d_true)

    loss_grow = pred_y.new_tensor(0.0)
    loss_decay = pred_y.new_tensor(0.0)

    for s in scales:
        pg = _to_mass_prob(grow_pred, weight=None, pool=s, eps=eps)
        qg = _to_mass_prob(grow_true, weight=None, pool=s, eps=eps)

        pd = _to_mass_prob(decay_pred, weight=None, pool=s, eps=eps)
        qd = _to_mass_prob(decay_true, weight=None, pool=s, eps=eps)

        loss_grow = loss_grow + _js_divergence(pg, qg, eps=eps)
        loss_decay = loss_decay + _js_divergence(pd, qd, eps=eps)

    loss_grow = loss_grow / len(scales)
    loss_decay = loss_decay / len(scales)

    # return 0.5 * (loss_grow + loss_decay)
    # return 0.7*loss_grow + 0.3*loss_decay
    alpha = 0.5
    return alpha * loss_grow + (1 - alpha) * loss_decay


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        
        # loss_rddr = radar_diff_div_reg(pred_y, batch_y, scales=(1, 2))
        # loss = self.criterion(pred_y, batch_y) + 0.01*loss_rddr
        
        loss = self.criterion(pred_y, batch_y)
        
        # loss = self.criterion(pred_y, batch_y) + 0.01 * self.diff_div_reg(pred_y, batch_y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
