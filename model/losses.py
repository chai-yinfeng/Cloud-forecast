import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

class generator_loss_function(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.loss_ssim = ssim

    def forward(self, gen_img: torch.Tensor, target: torch.Tensor, gen_D: torch.Tensor):
        """
        计算生成器的损失，结合 L1、L2 和 SSIM 损失，并加入对抗损失（gen_D 输出）。
        
        Args:
            gen_img (Tensor): 生成器输出的图像序列，形状为 [B, T, C, H, W]。
            target (Tensor): 目标图像序列，形状为 [B, T, C, H, W]。
            gen_D (Tensor): 判别器对生成图像的输出，形状为 [B, 1]。
        
        Returns:
            L_total (Tensor): 总损失，包括重建损失、SSIM 损失和对抗损失。
            L_rec (Tensor): 重建损失 (L1 + L2)。
            L_ssim (Tensor): SSIM 损失。
            L_adv (Tensor): 对抗损失。
        """
        L_rec = self.l1_loss(gen_img, target) + self.l2_loss(gen_img, target)

        output_np = gen_img.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        ssim_value = 0
        B, T, C, H, W = gen_img.shape
        for i in range(B):
            ssim_seq = 0
            for k in range(T):
                v1 = output_np[i, k, 0] * 255
                v2 = target_np[i, k, 0] * 255
                ssim_seq += self.loss_ssim(v1, v2, data_range=255)
            ssim_value += ssim_seq / T

        L_ssim = ssim_value / B
        L_adv = -torch.mean(gen_D)
        L_total = L_rec + 1e-2 * (1 - L_ssim) + 1e-4 * L_adv

        return L_total, L_rec, L_ssim, L_adv

# 再写一个给普通(不使用GAN)模型的损失函数，不含对抗损失
def sa_lstm_loss(output, target):
    loss_mae = nn.L1Loss()
    loss_mse = nn.MSELoss()
    loss_ssim = ssim

    L_rec = loss_mae(output, target) + loss_mse(output, target)

    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    ssim_value = 0
    for i in range(output_np.shape[0]):
        ssim_seq = 0
        for k in range(output_np.shape[1]):
            result = loss_ssim(output_np[i, k, 0, :, :] * 255, target_np[i, k, 0, :, :] * 255, data_range=255)
            ssim_seq += result
        ssim_value += ssim_seq / 6

    L_ssim = ssim_value / output_np.shape[0]

    return L_rec + 0.01 * (1 - L_ssim), L_rec, L_ssim