import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

from model.base_cells import ConvLSTMCell

class ConvLSTMEncode2Decode(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        初始化编码器—解码器模型，结合 ConvLSTM 层对时空数据进行处理。
        对输入先下采样到 1/4 分辨率，在底层进行多层 ConvLSTM，再反卷积上采样回原分辨率。
        
        Args:
            params (dict): 包含以下键值对的参数字典：
                - 'input_dim' (int): 输入图像的通道数。
                - 'batch_size' (int): 批次大小。
                - 'kernel_size' (int): ConvLSTMCell 卷积核大小。
                - 'img_size' (tuple): 原始图像尺寸 (height, width)。
                - 'hidden_dim' (int): 编码器输出以及 ConvLSTM 隐状态的通道数。
                - 'n_layers' (int): ConvLSTM 层数。
                - 'bias' (bool): 是否使用偏置项。
        """
        super().__init__()

        self.batch_size = params['batch_size']
        self.n_layers = params['n_layers']
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.bias = params['bias']
        self.input_window_size = params['input_window_size']  # T_in
        self.output_window_size = params['output_window_size']  # T_out

        # 2 times downsampling, size of features map becomes 1/4
        self.encoded_size = (params['img_size'][0] // 4, params['img_size'][1] // 4)

        self.img_encode = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.img_decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=1, stride=1, padding=0)
        )

        self.cells = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(ConvLSTMCell(input_dim=self.hidden_dim,   # after encoder, channels = hidden_dim
                                           hidden_dim=self.hidden_dim,
                                           kernel_size=params['kernel_size'],
                                           bias=self.bias))
            self.bns.append(nn.LayerNorm([self.hidden_dim, self.encoded_size[0], self.encoded_size[1]]))

    def forward(self, frames: torch.Tensor, hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] =None) -> torch.Tensor:
        """
        前向传播处理时空数据帧。
        
        Args:
            frames (torch.Tensor): 输入序列，形状为 [batch_size, time_steps, channels, height, width]。
            hidden (list, optional): 每层 ConvLSTMCell 的初始隐藏状态列表。如果未提供，则自动初始化。
        
        Returns:
            torch.Tensor: 模型输出，形状为 [batch_size, time_steps, input_dim, height, width]。
        """
        B, T_in, C, H, W = frames.shape
        seq_len = T_in + self.output_window_size
        if hidden is None:
            hidden = self._init_hidden(batch_size=B)

        outputs = []
        x_prev = None
        for t in range(seq_len):
            if t < T_in:
                x_in = frames[:, t]    # [B, input_dim, H, W]
            else:
                x_in = x_prev

            enc = self.img_encode(x_in)  # [B, hidden_dim, H/4, W/4]

            for i, cell in enumerate(self.cells):
                enc, hidden[i] = cell(enc, hidden[i])
                enc = self.bns[i](enc)

            dec = self.img_decode(enc)
            x_prev = dec
            outputs.append(dec)

        outputs = torch.stack(outputs, dim=1)   # [B, hidden_dim, H/4, W/4]
        return outputs[:, T_in:]    # => [B, T_out, C, H, W]
    
    def _init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        初始化每层 ConvLSTMCell 的隐藏状态和记忆状态。
        
        Args:
            batch_size (int): 批次大小。
        
        Returns:
            list: 每层状态的列表，每个状态为 (h0, c0)。
        """
        hidden_states = []
        for i in range(self.n_layers):
            hidden_states.append(self.cells[i].init_hidden(batch_size, self.encoded_size))
        return hidden_states