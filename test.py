import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve2d

def ssim_metric(output, target):
    """
    计算序列平均SSIM，以及序列首帧的SSIM。
    Args:
        output: [B, seq_len, C, H, W]
        target: [B, seq_len, C, H, W]
    Returns:
        (avg_ssim_over_seq, first_frame_ssim)
    """
    batch_size, seq_len, _, height, width = output.shape
    ssim_values_sum = np.zeros(batch_size)
    ssim_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        # 首帧
        output_img = output[i, 0, 0, :, :].detach().cpu().numpy() * 255
        target_img = target[i, 0, 0, :, :].detach().cpu().numpy() * 255
        ssim_values_first[i] = ssim(target_img, output_img, data_range=255)

        # 整个序列
        ssim_seq_sum = 0
        for j in range(seq_len):
            out_j = output[i, j, 0, :, :].detach().cpu().numpy() * 255
            tgt_j = target[i, j, 0, :, :].detach().cpu().numpy() * 255
            ssim_seq_sum += ssim(tgt_j, out_j, data_range=255)
        ssim_values_sum[i] = ssim_seq_sum / seq_len

    return np.mean(ssim_values_sum), np.mean(ssim_values_first)

def mse_metric(output, target):
    """
    计算序列平均MSE，以及序列首帧MSE。
    """
    batch_size, seq_len, _, height, width = output.shape
    mse_values_sum = np.zeros(batch_size)
    mse_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        # 首帧
        out_first = output[i, 0, :, :, :].detach().cpu().numpy().flatten() * 255
        tgt_first = target[i, 0, :, :, :].detach().cpu().numpy().flatten() * 255
        mse_values_first[i] = mean_squared_error(tgt_first, out_first)

        # 整个序列
        mse_seq_sum = 0
        for j in range(seq_len):
            out_j = output[i, j, :, :, :].detach().cpu().numpy().flatten() * 255
            tgt_j = target[i, j, :, :, :].detach().cpu().numpy().flatten() * 255
            mse_seq_sum += mean_squared_error(tgt_j, out_j)
        mse_values_sum[i] = mse_seq_sum / seq_len

    return np.mean(mse_values_sum), np.mean(mse_values_first)

def psnr_metric(output, target):
    """
    计算序列平均PSNR，以及序列首帧PSNR。
    """
    batch_size, seq_len, _, height, width = output.shape
    psnr_values_sum = np.zeros(batch_size)
    psnr_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        # 首帧
        out_first = output[i, 0, :, :, :].detach().cpu().numpy() * 255
        tgt_first = target[i, 0, :, :, :].detach().cpu().numpy() * 255
        psnr_values_first[i] = cv2.PSNR(tgt_first, out_first)

        # 整个序列
        psnr_seq_sum = 0
        for j in range(seq_len):
            out_j = output[i, j, :, :, :].detach().cpu().numpy() * 255
            tgt_j = target[i, j, :, :, :].detach().cpu().numpy() * 255
            psnr_seq_sum += cv2.PSNR(tgt_j, out_j)
        psnr_values_sum[i] = psnr_seq_sum / seq_len

    return np.mean(psnr_values_sum), np.mean(psnr_values_first)

def sharpness_calculate(img1):
    """
    No-Reference Image Sharpness Assessment:
    基于最大梯度与梯度变动
    """
    F1 = np.array([[0, 0], [-1, 1]])
    F2 = F1.T

    from scipy.signal import convolve2d
    H1 = convolve2d(img1, F1, mode='valid')
    H2 = convolve2d(img1, F2, mode='valid')
    g = np.sqrt(H1 ** 2 + H2 ** 2)

    row, col = g.shape
    B = round(min(row, col) / 16)
    g_center = g[B + 1: -B, B + 1: -B]
    MaxG = np.max(g_center)
    MinG = np.min(g_center)
    CVG = (MaxG - MinG) / np.mean(g_center)
    re = MaxG ** 0.61 * CVG ** 0.39
    return re

def sharpness_metric(output):
    """
    计算序列平均sharpness，以及序列首帧sharpness。
    """
    batch_size, seq_len, _, height, width = output.shape
    sharpness_values_sum = np.zeros(batch_size)
    sharpness_values_first = np.zeros(batch_size)

    for i in range(batch_size):
        # 首帧
        out_first = output[i, 0, 0, :, :].detach().cpu().numpy() * 255
        sharpness_values_first[i] = sharpness_calculate(out_first)

        # 整个序列
        sharp_seq_sum = 0
        for j in range(seq_len):
            out_j = output[i, j, 0, :, :].detach().cpu().numpy() * 255
            sharp_seq_sum += sharpness_calculate(out_j)
        sharpness_values_sum[i] = sharp_seq_sum / seq_len

    return np.mean(sharpness_values_sum), np.mean(sharpness_values_first)


def test_model(model_name, model, test_loader, device, writer=None):
    """
    使用给定的 test_loader 对模型进行评估, 计算并打印平均的 ssim, mse, psnr, sharpness。

    Args:
        model_name (str): 模型名称 ('convlstm' 或 'sa_encode2decode'等)
        model (nn.Module): 已经训练好的 pytorch 模型
        test_loader (DataLoader): 测试数据集 Dataloader, 每次返回 (batch_data, batch_target)
        device (torch.device): 设备
        writer (SummaryWriter, optional): 如需tensorboard,可传.否则None
    """

    # 切换到评估模式
    model.eval()

    # 累加各项指标
    total_ssim = 0.0
    total_ssim_first = 0.0
    total_mse = 0.0
    total_mse_first = 0.0
    total_psnr = 0.0
    total_psnr_first = 0.0
    total_sharp = 0.0
    total_sharp_first = 0.0

    count = 0  # 统计batch数

    with torch.no_grad():
        for i, (batch_data, batch_target) in enumerate(test_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            # Forward
            if model_name in ['simple', 'encode2decode', 'encode2decode_unet']:
                # 仅需要 (batch_data)
                output = model(batch_data)
            else:   # 'sa_encode2decode' | 'sa_encode2decode_unet' | 'sa_encode2decode_gan'
                # 需要 (frames, target, mask_true=None, is_training=False)
                output = model(batch_data, batch_target, mask_true=None, is_training=False)

            # 指标计算
            ssim_val, ssim_first = ssim_metric(output, batch_target)
            mse_val, mse_first = mse_metric(output, batch_target)
            psnr_val, psnr_first = psnr_metric(output, batch_target)
            sharp_val, sharp_first = sharpness_metric(output)

            total_ssim += ssim_val
            total_ssim_first += ssim_first
            total_mse += mse_val
            total_mse_first += mse_first
            total_psnr += psnr_val
            total_psnr_first += psnr_first
            total_sharp += sharp_val
            total_sharp_first += sharp_first

            # 若用TensorBoard记录
            if writer is not None:
                writer.add_scalar('Test/SSIM', ssim_val, i)
                writer.add_scalar('Test/MSE', mse_val, i)
                writer.add_scalar('Test/PSNR', psnr_val, i)
                writer.add_scalar('Test/Sharpness', sharp_val, i)

            count += 1

    # 取平均
    mean_ssim = total_ssim / count
    mean_ssim_first = total_ssim_first / count
    mean_mse = total_mse / count
    mean_mse_first = total_mse_first / count
    mean_psnr = total_psnr / count
    mean_psnr_first = total_psnr_first / count
    mean_sharp = total_sharp / count
    mean_sharp_first = total_sharp_first / count

    # 打印结果
    print(f"\n[TEST] {model_name}:  total {count} batches")
    print(f"  SSIM: {mean_ssim:.4f}, first-frame={mean_ssim_first:.4f}")
    print(f"  MSE : {mean_mse:.4f}, first-frame={mean_mse_first:.4f}")
    print(f"  PSNR: {mean_psnr:.4f}, first-frame={mean_psnr_first:.4f}")
    print(f"  Sharpness: {mean_sharp:.4f}, first-frame={mean_sharp_first:.4f}\n")

    return {
        'ssim': mean_ssim,
        'ssim_first': mean_ssim_first,
        'mse': mean_mse,
        'mse_first': mean_mse_first,
        'psnr': mean_psnr,
        'psnr_first': mean_psnr_first,
        'sharp': mean_sharp,
        'sharp_first': mean_sharp_first
    }