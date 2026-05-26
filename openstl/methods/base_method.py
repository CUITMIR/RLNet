import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
import torch.nn.functional as F
import os
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []
        self.val_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def on_validation_epoch_start(self):
        self.val_outputs = []
    
    
    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2: return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        # 原版: return loss_gap.mean()  # 在 D 维上又除了一次,导致量级被压低 D 倍
        return loss_gap.sum(dim=-1).mean()  # 标准 KL: 在分布维度求和,在样本维度求均值

    
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        
        loss_ddr = 0.01 * self.diff_div_reg(pred_y, batch_y)
        loss = self.criterion(pred_y, batch_y) + loss_ddr
        
        if batch_idx == 0:
            print(f"loss: {loss.item():.7f}, loss_ddr: {loss_ddr.item():.7f}")

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
    
        outputs = {
            'preds': pred_y.detach().cpu().numpy(),
            'trues': batch_y.detach().cpu().numpy()
        }
        self.val_outputs.append(outputs)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_outputs) == 0:
            return
    
        results_all = {}
        for k in self.val_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.val_outputs], axis=0)
    
        eval_res, eval_log = metric(
            results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std,
            metrics=self.metric_list,
            channel_names=self.channel_names,
            spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None)
        )
    
        if self.trainer.is_global_zero:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            train_loss = self.trainer.callback_metrics.get('train_loss')
            val_loss = self.trainer.callback_metrics.get('val_loss')
    
            if train_loss is not None and hasattr(train_loss, 'item'):
                train_loss = train_loss.item()
            if val_loss is not None and hasattr(val_loss, 'item'):
                val_loss = val_loss.item()
    
            msg = f"Epoch {self.current_epoch}: Lr: {lr:.7f}"
    
            if train_loss is not None:
                msg += f" | Train Loss: {train_loss:.7f}"
            if val_loss is not None:
                msg += f" | Vali Loss: {val_loss:.7f}"
    
            msg += f" | {eval_log}"
            print_log(msg)
    
        self.val_outputs = []

    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

