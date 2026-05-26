import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1

def _threshold(x, y, t):
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p

def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


# def PSNR(pred, true, min_max_norm=True):
#     """Peak Signal-to-Noise Ratio.
#
#     Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#     """
#     mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
#     if mse == 0:
#         return float('inf')
#     else:
#         if min_max_norm:  # [0, 1] normalized by min and max
#             return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
#         else:
#             return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std

def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std

# 假设 pred 和 true 是你的预测值和真实值
# 这里简单示例，你需要替换为实际的数据
# pred = ...
# true = ...
# if 'psnr' in metrics:
#     psnr = 0
#     for b in range(pred.shape[0]):
#         for f in range(pred.shape[1]):
#             psnr += PSNR(pred[b, f], true[b, f])
#     eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])


def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def POD(hits, misses, eps=1e-6):
    """
    probability_of_detection
    Inputs:
    Outputs:
        pod = hits / (hits + misses) averaged over the T channels
        
    """
    pod = (hits + eps) / (hits + misses + eps)
    return np.mean(pod)

def SUCR(hits, fas, eps=1e-6):
    """
    success_rate
    Inputs:
    Outputs:
        sucr = hits / (hits + false_alarms) averaged over the D channels
    """
    sucr = (hits + eps) / (hits + fas + eps)
    return np.mean(sucr)

def prep_clf(sim, obs, threshold=0.1):
    # obs = np.asarray(obs.cpu().detach().numpy())
    # sim = np.asarray(sim.cpu().detach().numpy())
    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (sim == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (sim == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (sim == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (sim == 0))

    return hits, misses, falsealarms, correctnegatives

def CSI(sim, obs, threshold=0.1):

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)
    results = hits / (hits + misses + falsealarms + 1)

    return results

def sevir_metrics(pred, true, threshold):
    """
    calcaulate t, p, hits, fas, misses
    Inputs:
    pred: [N, T, C, L, L]
    true: [N, T, C, L, L]
    threshold: float
    """
    pred = pred.transpose(1, 0, 2, 3, 4)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)

def merge_tiles(data, num_tiles_per_image=4, split_size=2, tile_size=256):
    """
    Merge tiled data into full images.

    Args:
        data: Input array of shape (N, T, C, H, W)
        num_tiles_per_image: Number of tiles per full image (default 4 for 2x2)
        split_size: Grid size for merging (2 for 2x2)
        tile_size: Size of each tile (default 256)

    Returns:
        Merged array of shape (N_new, T, C, H_full, W_full)
    """
    # 计算完整图像数量
    num_images = data.shape[0] // num_tiles_per_image
    # 截断多余的分片（确保整除）
    data = data[:num_images * num_tiles_per_image]
    # 重塑为 (num_images, num_tiles_per_image, T, C, H, W)
    data = data.reshape(num_images, num_tiles_per_image, *data.shape[1:])
    # 合并分片：交换轴以对齐维度
    data = data.transpose(0, 2, 3, 1, 4, 5)  # (num_images, T, C, num_tiles_per_image, H, W)
    # 重塑为网格 (num_images, T, C, split_size, split_size, H, W)
    data = data.reshape(num_images, data.shape[1], data.shape[2], split_size, split_size, tile_size, tile_size)
    # 合并为完整图像
    merged = np.zeros((num_images, data.shape[1], data.shape[2], split_size * tile_size, split_size * tile_size))
    for i in range(split_size):
        for j in range(split_size):
            merged[..., i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] = \
                data[..., i, j, :, :]
    return merged

# def merge_tiles(data, num_tiles_per_image=16, split_size=4, tile_size=128):
#     """
#     Merge tiled data into full images.
    
#     Args:
#         data: Input array of shape (N, T, C, H, W)
#         num_tiles_per_image: Number of tiles per full image (default 16 for 4x4)
#         split_size: Grid size for merging (4 for 4x4)
#         tile_size: Size of each tile (default 128)
        
#     Returns:
#         Merged array of shape (N_new, T, C, H_full, W_full)
#     """
#     # 计算完整图像数量
#     num_images = data.shape[0] // num_tiles_per_image
#     # 截断多余的分片（确保整除）
#     data = data[:num_images * num_tiles_per_image]
#     # 重塑为 (num_images, num_tiles_per_image, T, C, H, W)
#     data = data.reshape(num_images, num_tiles_per_image, *data.shape[1:])
#     # 合并分片：交换轴以对齐维度
#     data = data.transpose(0, 2, 3, 1, 4, 5)  # (num_images, T, C, num_tiles_per_image, H, W)
#     # 重塑为网格 (num_images, T, C, split_size, split_size, H, W)
#     data = data.reshape(num_images, data.shape[1], data.shape[2], split_size, split_size, tile_size, tile_size)
#     # 合并为完整图像
#     merged = np.zeros((num_images, data.shape[1], data.shape[2], split_size * tile_size, split_size * tile_size))
#     for i in range(split_size):
#         for j in range(split_size):
#             merged[..., i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] = \
#                 data[..., i, j, :, :]
#     return merged


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        channel_names (list | None): The name of different channels.
        spatial_norm (bool): Weather to normalize the metric by HxW.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi', 'csi_hko', 'csi_imerg']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                       true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses) 
        
    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'snr' in metrics:
        snr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr += SNR(pred[b, f], true[b, f])
        eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        lpips = 0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred = pred.transpose(0, 1, 3, 4, 2)
        true = true.transpose(0, 1, 3, 4, 2)
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                lpips += cal_lpips(pred[b, f], true[b, f])
        eval_res['lpips'] = lpips / (pred.shape[0] * pred.shape[1])

        
    # pred = merge_tiles(pred)
    # true = merge_tiles(true)
    print(pred.shape)
    print(true.shape)
    if 'csi' in metrics:
        for threshold_dbz, threshold_pix in zip([10, 20, 30, 40, 50], [86, 106, 126, 146, 166]):
            csi = 0
            for b in range(pred.shape[0]):
                for f in range(pred.shape[1]):
                    t = CSI(pred[b, f], true[b, f], threshold=threshold_pix / 255)
                    csi += t
            eval_res[f'csi_sc_{threshold_dbz}dbz'] = csi / (pred.shape[0] * pred.shape[1])
       
    # if 'csi_hko' in metrics:
    #     for threshold_dbz, threshold_pix in zip([10, 20, 30], [73, 109, 146]):
    #         csi = 0
    #         if threshold_dbz == 20:
    #             csi_20dbz_step = [0] * 10
    #         for b in range(pred.shape[0]):
    #             for f in range(pred.shape[1]):
    #                 t = CSI(pred[b, f], true[b, f], threshold=threshold_pix / 255)
    #                 csi += t
    #                 if threshold_dbz == 20:
    #                     csi_20dbz_step[f] += t
    #         eval_res[f'csi_hko_{threshold_dbz}dbz'] = csi / (pred.shape[0] * pred.shape[1])
    #         if threshold_dbz == 20:
    #             for f in range(10):
    #                 eval_res[f'csi_hko_20dbz_step{f}'] = csi_20dbz_step[f] / pred.shape[0]
    if 'csi_hko' in metrics:
        for threshold_dbz, threshold_pix in zip([10, 20, 30, 40, 50], [73, 109, 146, 182, 219]):
            csi = 0
            for b in range(pred.shape[0]):
                for f in range(pred.shape[1]):
                    t = CSI(pred[b, f], true[b, f], threshold=threshold_pix / 255)
                    csi += t
            eval_res[f'csi_hko_{threshold_dbz}dbz'] = csi / (pred.shape[0] * pred.shape[1])
                    
    if 'csi_imerg' in metrics:
        for threshold_mmh, threshold_pix in zip([0.1, 0.5, 1, 2, 5], [0.15, 0.3247, 0.4, 0.4753, 0.5747]):
            csi = 0
            for b in range(pred.shape[0]):
                for f in range(pred.shape[1]):
                    t = CSI(pred[b, f], true[b, f], threshold=threshold_pix)
                    csi += t
            eval_res[f'csi_{threshold_mmh}mm/h'] = csi / (pred.shape[0] * pred.shape[1])


    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log
