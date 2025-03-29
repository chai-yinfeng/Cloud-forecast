import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

from model.base_cells import ConvLSTMCell

class UNetEncoder(nn.Module):
    """
    这里的编码器包含三层:
    1) 第一次卷积 (不改变分辨率)
    2) 下采样 1: stride=2
    3) 下采样 2: stride=2
    并在每个阶段保留特征图
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, in_channels, H, W]
        return: (x1, x2, x3)
            x1: [B, hidden_dim, H, W]
            x2: [B, hidden_dim, H/2, W/2]
            x3: [B, hidden_dim, H/4, W/4]
        """
        x1 = self.initial_conv(x)  # [H, W]
        x2 = self.down1(x1)        # [H/2, W/2]
        x3 = self.down2(x2)        # [H/4, W/4]
        return x1, x2, x3

class UNetDecoder(nn.Module):
    """
    解码器分两步上采样, 并把对应分辨率的Encoder特征拼接:
    1) up1 上采样 -> 与 x2 拼接 -> conv
    2) up2 上采样 -> 与 x1 拼接 -> conv
    最终恢复到原图分辨率后, 使用最后一层conv映射回输入通道数
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv_after_cat1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv_after_cat2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.out_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x3: torch.Tensor, x2: torch.Tensor, x1: torch.Tensor):
        """
        x3: 最小分辨率特征 [B, hidden_dim, H/4, W/4]
        x2: 中间分辨率特征 [B, hidden_dim, H/2, W/2]
        x1: 最大分辨率特征 [B, hidden_dim, H, W]
        """
        x2_dec = self.up1(x3)  # [B, hidden_dim, H/2, W/2]

        x2_cat = torch.cat([x2_dec, x2], dim=1)  # [B, 2*hidden_dim, H/2, W/2]
        x2_fused = self.conv_after_cat1(x2_cat)  # [B, hidden_dim, H/2, W/2]

        x1_dec = self.up2(x2_fused)  # [B, hidden_dim, H, W]

        x1_cat = torch.cat([x1_dec, x1], dim=1)  # [B, 2*hidden_dim, H, W]
        x1_fused = self.conv_after_cat2(x1_cat)  # [B, hidden_dim, H, W]

        out = self.out_conv(x1_fused)  # [B, output_dim, H, W]
        return out
    

class ConvLSTMEncode2DecodeUNet(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        使用UNetEncoder和UNetDecoder进行跳跃连接,
        在最底层分辨率上使用多层ConvLSTM对时间序列进行处理.
        
        Args:
            params (dict): 包含以下键值对的参数字典:
                - 'input_dim' (int): 输入图像的通道数.
                - 'batch_size' (int): 批次大小.
                - 'kernel_size' (int): ConvLSTMCell 卷积核大小.
                - 'img_size' (tuple): 原始图像尺寸 (height, width).
                - 'hidden_dim' (int): ConvLSTM 隐状态的通道数 & U-Net的主干通道数.
                - 'n_layers' (int): ConvLSTM 层数.
                - 'bias' (bool): 是否使用偏置项.
        """
        super().__init__()

        self.batch_size = params['batch_size']
        self.n_layers = params['n_layers']
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.img_size = params['img_size']  # (H, W)
        self.bias = params['bias']
        self.input_window_size = params['input_window_size']  # T_in
        self.output_window_size = params['output_window_size']  # T_out

        self.encoder = UNetEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)

        self.cells = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.n_layers):
            self.cells.append(ConvLSTMCell(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=params['kernel_size'],
                bias=self.bias
            ))

            self.bns.append(nn.LayerNorm([self.hidden_dim, self.img_size[0] // 4, self.img_size[1] // 4]))

        self.decoder = UNetDecoder(hidden_dim=self.hidden_dim, output_dim=self.input_dim)

    def forward(self, frames: torch.Tensor, hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        前向传播:
        frames: [B, T, C, H, W]
        hidden: 如果不提供, 则自动根据batch_size初始化.
        
        return:
            outputs: [B, T, C, H, W], 每个时间步都输出与输入分辨率相同的图像.
        """
        B, T_in, C, H, W = frames.shape
        seq_len = T_in + self.output_window_size
        if hidden is None:
            hidden = self._init_hidden(B)

        outputs = []
        x_prev = None
        for t in range(seq_len):
            if t < T_in:
                x_in = frames[:, t]
            else:
                x_in = x_prev
            x1, x2, x3 = self.encoder(x_in)  
            # x1: [B, hidden_dim, H,   W  ]
            # x2: [B, hidden_dim, H/2, W/2]
            # x3: [B, hidden_dim, H/4, W/4]

            for i, cell in enumerate(self.cells):
                x3, hidden[i] = cell(x3, hidden[i])
                x3 = self.bns[i](x3)

            out = self.decoder(x3, x2, x1)  # [B, input_dim, H, W]
            x_prev = out
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)   # [B, T, input_dim, H, W]
        return outputs[:, T_in:]

    def _init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        初始化每层 ConvLSTMCell 的 (h, c) 状态.
        这里的 (height, width) = (H/4, W/4), 因为编码器最终下采样了2次
        """
        height = self.img_size[0] // 4
        width = self.img_size[1] // 4
        hidden_states = []
        for cell in self.cells:
            hidden_states.append(cell.init_hidden(batch_size, (height, width)))
        return hidden_states
