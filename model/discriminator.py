import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    3D卷积判别器，用于 GAN 训练。
    输入形状: [B, C, T, H, W] (或 [B, T, C, H, W] 后再 permute)
    这里只写成单通道 => Conv3d(in_channels=1,...)
    如果想支持RGB，需要改成 in_channels=3
    """
    def __init__(self):
        super().__init__()
        # [B, 1, T, H, W] --> [B, 16, T, H/2, W/2]
        self.lay1 = nn.Sequential(
            nn.Conv3d(1, 16, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.LeakyReLU(0.2)
        )
        # --> [B, 32, T, H/4, W/4]
        self.lay2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.LeakyReLU(0.2)
        )
        # --> [B, 64, T, H/8, W/8]
        self.lay3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), 1),
            nn.LeakyReLU(0.2)
        )
        # --> [B, 128, T, H/16, W/16]
        self.lay4 = nn.Sequential(
            nn.Conv3d(64, 128, (3, 3, 3), (1, 2, 2), 1),
            nn.LeakyReLU(0.2)
        )
        # --> [B, 256, T, H/32, W/32]
        self.lay5 = nn.Sequential(
            nn.Conv3d(128, 256, (3, 3, 3), (1, 2, 2), 1),
            nn.LeakyReLU(0.2)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dense1 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 输入张量，形状为 [B, C, T, H, W]。
        
        Returns:
            Tensor: 判别器的输出，形状为 [B, 1]，表示每个样本的真假概率。
        """
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.avgpool(x)
        dense_input = x.contiguous().view(x.size(0), -1)
        dense_output_1 = self.dense1(dense_input)

        return dense_output_1