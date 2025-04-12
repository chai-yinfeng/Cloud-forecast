import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

from model.base_cells import SA_ConvLSTMCell

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
            x1: [B, hidden_dim, H,   W]
            x2: [B, hidden_dim, H/2, W/2]
            x3: [B, hidden_dim, H/4, W/4]
        """
        x1 = self.initial_conv(x)  # [B, hidden_dim, H,   W]
        x2 = self.down1(x1)        # [B, hidden_dim, H/2, W/2]
        x3 = self.down2(x2)        # [B, hidden_dim, H/4, W/4]
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
        x1: 最大分辨率特征 [B, hidden_dim, H,   W  ]
        """
        x2_dec = self.up1(x3)  # [B, hidden_dim, H/2, W/2]
        x2_cat = torch.cat([x2_dec, x2], dim=1)  # [B, 2*hidden_dim, H/2, W/2]
        x2_fused = self.conv_after_cat1(x2_cat)  # [B, hidden_dim, H/2, W/2]

        x1_dec = self.up2(x2_fused)              # [B, hidden_dim, H, W]
        x1_cat = torch.cat([x1_dec, x1], dim=1)  # [B, 2*hidden_dim, H, W]
        x1_fused = self.conv_after_cat2(x1_cat)  # [B, hidden_dim, H, W]

        out = self.out_conv(x1_fused)            # [B, output_dim, H, W]
        return out

class SA_ConvLSTMEncode2DecodeUNet(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        使用 UNetEncoder + UNetDecoder 做跳跃连接,
        在最底层分辨率上使用多层 SA_ConvLSTMCell 对时间序列进行处理.

        Args:
            params (dict): 包含以下键值对的参数字典:
                - 'input_dim' (int): 输入图像的通道数.
                - 'batch_size' (int): 批次大小.
                - 'kernel_size' (int): SA_ConvLSTMCell 卷积核大小.
                - 'img_size' (tuple): 原始图像尺寸 (height, width).
                - 'hidden_dim' (int): LSTM隐状态通道 & Unet主干通道.
                - 'n_layers' (int): SA_ConvLSTM 层数.
                - 'bias' (bool): 是否使用偏置.
                - 'att_hidden_dim' (int): 自注意力中 Q,K 的投影维度.
                - 'input_window_size' (int): 输入序列长度 T_in
                - 'output_window_size' (int): 预测/输出序列长度 T_out
        """
        super().__init__()

        self.batch_size = params['batch_size']
        self.n_layers = params['n_layers']
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.img_size = params['img_size']  # (H, W)
        self.input_window_size = params['input_window_size']
        self.output_window_size = params['output_window_size']

        self.encoder = UNetEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)

        # 使用带自注意力的 SA_ConvLSTMCell
        self.cells = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(self.n_layers):
            self.cells.append(SA_ConvLSTMCell(params))
            self.bns.append(nn.LayerNorm([self.hidden_dim, self.img_size[0] // 4, self.img_size[1] // 4]))

        self.decoder = UNetDecoder(hidden_dim=self.hidden_dim, output_dim=self.input_dim)

    def forward(self, frames: torch.Tensor, target: torch.Tensor, mask_true: torch.Tensor, is_training: bool,
            hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        前向传播:
        frames: [B, T_in, C, H, W], 输入序列
        target: [B, T_out, C, H, W], 目标序列
        mask_true: [B, T_out, C, H, W], scheduled sampling 时, 1 表示使用 GT, 0 表示用预测
        is_training: bool, 标志是否训练模式 (决定是否拼接 frames+target 以及使用 mask_true)
        hidden: 不提供则自动初始化 -> 每层 (h, c, m), 都是 [B, hidden_dim, H/4, W/4]

        return:
            outputs: [B, T_out, C, H, W], 只返回最后 T_out 个时间步的输出
        """
        B, T_in, C, H, W = frames.shape

        seq_len = self.input_window_size + self.output_window_size

        if hidden is None:
            hidden = self._init_hidden(B)

        # 如果是训练, 把输入 + 目标帧合并起来(方便后面 index)
        # frames: [B, T_in+T_out, C, H, W]
        if is_training:
            frames = torch.cat([frames, target], dim=1)

        predict_temp_de = []
        out = None

        for t in range(seq_len):
            if is_training:
                # 前 T_in 帧，用 frames[:, t] 直接是输入
                if t < self.input_window_size:
                    x_in = frames[:, t]  
                else:
                    # scheduled sampling: x = mask_true * gt + (1 - mask_true) * pred
                    # gt 是 frames[:, t], pred 是 out
                    # mask_true[:, t - T_in] 的形状是 [B, C, H, W]
                    x_in = mask_true[:, t - self.input_window_size] * frames[:, t] \
                           + (1 - mask_true[:, t - self.input_window_size]) * out
            else:
                # 测试/推理：前 T_in 帧用真实输入，之后都用上一步预测
                if t < self.input_window_size:
                    x_in = frames[:, t]
                else:
                    x_in = out

            # 1) Encoder => x1, x2, x3
            x1, x2, x3 = self.encoder(x_in) 
            # x3 => [B, hidden_dim, H/4, W/4]

            # 2) 底层多层 SA_ConvLSTMCell
            for i, cell in enumerate(self.cells):
                x3, hidden[i] = cell(x3, hidden[i])  # hidden[i]: (h, c, m)
                x3 = self.bns[i](x3)

            # 3) Decoder => 恢复到原分辨率
            out = self.decoder(x3, x2, x1)

            # 4) 把输出保存到列表 (out: [B, C, H, W])
            predict_temp_de.append(out)

        # predict_temp_de = [seq_len个, each (B,C,H,W)]
        predict_temp_de = torch.stack(predict_temp_de, dim=1)  # => [B, seq_len, C, H, W]

        # 只取后 T_out 帧 => [B, T_out, C, H, W]
        predict_temp_de = predict_temp_de[:, self.input_window_size :, :, :, :]

        return predict_temp_de

    def _init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        调用每个 SA_ConvLSTMCell 的 init_hidden,
        让 (h, c, m) 的 spatial 分辨率 = (H/4, W/4).
        """
        downsampled_size = (self.img_size[0] // 4, self.img_size[1] // 4)
        states = []
        for cell in self.cells:
            states.append(cell.init_hidden(batch_size, downsampled_size))
        return states
