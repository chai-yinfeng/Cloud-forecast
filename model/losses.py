import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.metrics import structural_similarity as ssim

# 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, layers: List[str] = ['relu2_2', 'relu3_3'], use_gpu: bool = True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.use_gpu = use_gpu
        if use_gpu:
            vgg = vgg.cuda()
        # 按照层名拆分：relu2_2 对应索引 8，relu3_3 对应索引 16（可根据 torchvision 版本调整）
        self.slice1 = nn.Sequential(*vgg[:9])   # up to relu2_2
        self.slice2 = nn.Sequential(*vgg[9:17]) # up to relu3_3
        # 冻结参数
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: [B, 3, H, W], 归一化到 [0,1] 或 VGG 要求的范围
        """
        x1 = self.slice1(x)
        y1 = self.slice1(y)
        x2 = self.slice2(x)
        y2 = self.slice2(y)
        loss = F.l1_loss(x1, y1) + F.l1_loss(x2, y2)
        return loss


# 边缘损失
class EdgeLoss(nn.Module):
    def __init__(self, mode: str = 'sobel'):
        """
        mode: 'sobel' or 'laplacian'
        """
        super().__init__()
        if mode == 'sobel':
            # 定义 Sobel 滤波核
            sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
            sobel_y = sobel_x.t()
            kernel = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)  # [2,1,3,3]
            self.register_buffer('kernel', kernel)
            self.mode = 'sobel'
        elif mode == 'laplacian':
            lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
            self.register_buffer('kernel', lap.unsqueeze(0).unsqueeze(0))  # [1,1,3,3]
            self.mode = 'laplacian'
        else:
            raise ValueError("mode must be 'sobel' or 'laplacian'")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: [B, 1, H, W] or [B, C, H, W] (直接逐通道计算)
        """
        kb = self.kernel.to(x)
        pad = k.size(-1) // 2

        ex = F.conv2d(x, kb, padding=pad)
        ey = F.conv2d(y, kb, padding=pad)

        if self.mode == 'sobel':
            gx_x, gx_y = ex.split(1, dim=1)
            gy_x, gy_y = ey.split(1, dim=1)
            ex = torch.sqrt(gx_x.pow(2) + gx_y.pow(2) + 1e-6)
            ey = torch.sqrt(gy_x.pow(2) + gy_y.pow(2) + 1e-6)

        return F.l1_loss(ex, ey)


# gan的损失函数（暂时搁置，gan的性能较差）
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

# 再写一个给普通(不使用GAN)模型的损失函数，不含对抗损失，但加入感知损失和边缘增强
def sa_lstm_loss(output, target,
                 lambda_ssim: float = 1e-2,
                 lambda_perc: float = 1e-1,
                 lambda_edge: float = 1e-2,
                 use_perceptual: bool = True,
                 use_edge: bool = True):
    loss_mae = nn.L1Loss()
    loss_mse = nn.MSELoss()
    loss_ssim = ssim

    L_rec = loss_mae(output, target) + loss_mse(output, target)

    B, T, C, H, W = output.shape

    # SSIM
    ssim_value = 0
    for i in range(B):
        for k in range(T):
            ssim_value += loss_ssim(
                output.detach().cpu().numpy()[i, k, 0]*255,
                target.detach().cpu().numpy()[i, k, 0]*255,
                data_range=255
            )
    L_ssim = (ssim_value / (B*T))

    # Perception
    if use_perceptual:
        perc = PerceptualLoss().to(output.device)
        L_perc = perc(
            output[:, -1, :, :, :].repeat(1, 3, 1, 1), 
            target[:, -1, :, :, :].repeat(1, 3, 1, 1)
        )
    else:
        L_perc = 0

    # Edge
    if use_edge:
        edge = EdgeLoss().to(output.device)
        L_edge = 0
        for k in range(T):
            L_edge += edge(
                output[:, k:k+1, 0, :, :], 
                target[:, k:k+1, 0, :, :]
            )
        L_edge = L_edge / T
    else:
        L_edge = 0

    L_total = L_rec + lambda_ssim * (1 - L_ssim) + lambda_perc * L_perc + lambda_edge * L_edge

    return L_total, L_rec, L_ssim, L_perc, L_edge
