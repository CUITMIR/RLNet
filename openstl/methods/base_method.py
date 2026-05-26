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

    # def validation_step(self, batch, batch_idx):
    #     batch_x, batch_y = batch
    #     pred_y = self(batch_x, batch_y)
    #     loss = self.criterion(pred_y, batch_y)
    #     self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
    #     return loss
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
        
        # loss_rddr = radar_diff_div_reg(pred_y, batch_y, scales=(1, 2))
        # loss_ddr = 0.01 * loss_rddr
        # loss = self.criterion(pred_y, batch_y)+loss_ddr
        
        # loss = self.criterion(pred_y, batch_y)
        
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

#     def on_test_epoch_end(self):
#         results_all = {}
#         for k in self.test_outputs[0].keys():
#             results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
            
#         print(results_all['preds'].shape)
#         print(results_all['trues'].shape)
#         eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
#             self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
#             channel_names=self.channel_names, spatial_norm=self.spatial_norm,
#             threshold=self.hparams.get('metric_threshold', None))
#         print(self.metric_list)
        
#         results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

#         if self.trainer.is_global_zero:
#             print_log(eval_log)
#             folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

#             for np_data in ['metrics', 'inputs', 'trues', 'preds']:
#                 np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
#         return results_all
    
    
    
    #######################################################################################
    def _infer_diff_vis_tag(self):
        """
        自动从 ex_name / save_dir 中判断当前是 ddr 还是 rddr
        """
        texts = []
    
        try:
            if hasattr(self.hparams, 'get'):
                texts.append(str(self.hparams.get('ex_name', '')))
                texts.append(str(self.hparams.get('save_dir', '')))
            else:
                texts.append(str(getattr(self.hparams, 'ex_name', '')))
                texts.append(str(getattr(self.hparams, 'save_dir', '')))
        except Exception:
            pass
    
        s = ' '.join(texts).lower()
    
        # 注意先判断 rddr，再判断 ddr
        if 'rddr' in s:
            return 'rddr'
        if 'ddr' in s:
            return 'ddr'
        return 'pred'
    
    
    def _denorm_np_for_vis(self, x, mean, std):
        """
        x: [N,T,C,H,W]
        按 test_mean / test_std 反归一化，便于画真实物理差分图
        """
        x = x.astype(np.float32)
    
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
    
        if mean.ndim == 0:
            mean = mean.reshape(1, 1, 1, 1, 1)
        elif mean.ndim == 1:
            mean = mean.reshape(1, 1, -1, 1, 1)
    
        if std.ndim == 0:
            std = std.reshape(1, 1, 1, 1, 1)
        elif std.ndim == 1:
            std = std.reshape(1, 1, -1, 1, 1)
    
        return x * std + mean
    
    
    def _to_2d_signed_map(self, x):
        """
        x: [C,H,W] or [H,W]
        保留正负号的2D差分图
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            if x.shape[0] == 1:
                x = x[0]
            else:
                x = x.mean(axis=0)   # 多通道时做通道均值，保留正负号
        elif x.ndim != 2:
            raise ValueError(f'Unexpected diff map shape: {x.shape}')
        return x
    
    
    def _save_diff_grid(self, gt_maps, pred_maps, save_path, pred_tag='pred'):
        """
        保存两行图：
        第1行 GT diff
        第2行 当前模型 diff
        """
        Tm1 = len(gt_maps)
        vmax = max(
            max(np.abs(m).max() for m in gt_maps),
            max(np.abs(m).max() for m in pred_maps)
        ) + 1e-8
    
        fig, axes = plt.subplots(2, Tm1, figsize=(2.2 * Tm1, 4.8))
        if Tm1 == 1:
            axes = np.array(axes).reshape(2, 1)
    
        for i in range(Tm1):
            axes[0, i].imshow(gt_maps[i], cmap='seismic', vmin=-vmax, vmax=vmax)
            axes[0, i].set_title(f'Δt={i+1}', fontsize=9)
            axes[0, i].axis('off')
    
            axes[1, i].imshow(pred_maps[i], cmap='seismic', vmin=-vmax, vmax=vmax)
            axes[1, i].set_title(f'Δt={i+1}', fontsize=9)
            axes[1, i].axis('off')
    
        axes[0, 0].set_ylabel('GT', fontsize=10)
        axes[1, 0].set_ylabel(pred_tag.upper(), fontsize=10)
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
        plt.close()
    
    
    def _try_compose_ddr_rddr_compare(self, compare_dir):
        """
        如果 GT / DDR / RDDR 三份差分结果都在，则自动拼成三行图
        """
        compare_dir = Path(compare_dir)
        gt_file = compare_dir / 'gt_diff_maps.npz'
        ddr_file = compare_dir / 'ddr_diff_maps.npz'
        rddr_file = compare_dir / 'rddr_diff_maps.npz'
    
        if not (gt_file.exists() and ddr_file.exists() and rddr_file.exists()):
            return
    
        gt_maps = np.load(gt_file)['maps']
        ddr_maps = np.load(ddr_file)['maps']
        rddr_maps = np.load(rddr_file)['maps']
    
        Tm1 = gt_maps.shape[0]
        vmax = max(
            np.abs(gt_maps).max(),
            np.abs(ddr_maps).max(),
            np.abs(rddr_maps).max()
        ) + 1e-8
    
        fig, axes = plt.subplots(3, Tm1, figsize=(2.2 * Tm1, 6.8))
        if Tm1 == 1:
            axes = np.array(axes).reshape(3, 1)
    
        for i in range(Tm1):
            axes[0, i].imshow(gt_maps[i], cmap='seismic', vmin=-vmax, vmax=vmax)
            axes[0, i].set_title(f'Δt={i+1}', fontsize=9)
            axes[0, i].axis('off')
    
            axes[1, i].imshow(ddr_maps[i], cmap='seismic', vmin=-vmax, vmax=vmax)
            axes[1, i].axis('off')
    
            axes[2, i].imshow(rddr_maps[i], cmap='seismic', vmin=-vmax, vmax=vmax)
            axes[2, i].axis('off')
    
        axes[0, 0].set_ylabel('GT', fontsize=10)
        axes[1, 0].set_ylabel('DDR', fontsize=10)
        axes[2, 0].set_ylabel('RDDR', fontsize=10)
    
        plt.tight_layout()
        out_path = compare_dir / 'compare_gt_ddr_rddr_diff.png'
        plt.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close()
    
        print(f'[AutoDiffVis] saved compare figure to: {out_path}')
    
    
    def _save_auto_diff_visualization(self, preds, trues):
        """
        preds, trues: [N,T,C,H,W], numpy
        自动保存第一个样本的差分图
        """
        # 默认取第一个样本
        sample_idx = 0
    
        # 本次实验的标签：ddr / rddr / pred
        tag = self._infer_diff_vis_tag()
    
        # 当前实验自己的保存目录
        local_vis_dir = check_dir(osp.join(self.hparams.save_dir, 'saved', 'diff_vis'))
    
        # 跨实验共享目录（用于自动拼接 DDR vs RDDR）
        compare_root = check_dir(osp.join(osp.dirname(self.hparams.save_dir), 'diff_vis_compare'))
    
        # 取第一个样本
        pred_seq = preds[sample_idx]   # [T,C,H,W]
        true_seq = trues[sample_idx]   # [T,C,H,W]
    
        # 相邻帧差分
        diff_pred = pred_seq[1:] - pred_seq[:-1]   # [T-1,C,H,W]
        diff_true = true_seq[1:] - true_seq[:-1]   # [T-1,C,H,W]
    
        gt_maps = [self._to_2d_signed_map(diff_true[t]) for t in range(diff_true.shape[0])]
        pred_maps = [self._to_2d_signed_map(diff_pred[t]) for t in range(diff_pred.shape[0])]
    
        # 1) 保存当前模型的两行图
        local_path = osp.join(local_vis_dir, f'{tag}_diff_grid.png')
        self._save_diff_grid(gt_maps, pred_maps, local_path, pred_tag=tag)
        print(f'[AutoDiffVis] saved current figure to: {local_path}')
    
        # 2) 保存 npz，供后续自动拼接
        np.savez(osp.join(compare_root, 'gt_diff_maps.npz'), maps=np.stack(gt_maps, axis=0))
        np.savez(osp.join(compare_root, f'{tag}_diff_maps.npz'), maps=np.stack(pred_maps, axis=0))
    
        # 3) 如果 DDR 和 RDDR 都跑过，就自动拼出三行图
        self._try_compose_ddr_rddr_compare(compare_root)
        
    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
    
        print(results_all['preds'].shape)
        print(results_all['trues'].shape)
    
        eval_res, eval_log = metric(
            results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std,
            metrics=self.metric_list,
            channel_names=self.channel_names,
            spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None)
        )
        print(self.metric_list)
    
        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])
    
        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))
    
            # 原来的 npy 保存
            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
    
            # ========== 新增：差分图自动可视化 ==========
            preds_denorm = self._denorm_np_for_vis(
                results_all['preds'], self.hparams.test_mean, self.hparams.test_std
            )
            trues_denorm = self._denorm_np_for_vis(
                results_all['trues'], self.hparams.test_mean, self.hparams.test_std
            )
    
            self._save_auto_diff_visualization(preds_denorm, trues_denorm)
            # =========================================
    
        return results_all