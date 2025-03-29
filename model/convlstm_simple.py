import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

from model.base_cells import ConvLSTMCell

class ConvLSTMSimple(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> Any:
        """
        在输入图像原分辨率上直接进行多层ConvLSTM处理，不做上下采样和转置卷积。

        Args:
            params (dict): 需要包含以下键:
                - 'input_dim' (int): 输入张量的通道数
                - 'hidden_dim' (int): ConvLSTM每层隐状态的通道数
                - 'n_layers' (int): 堆叠的ConvLSTM层数
                - 'kernel_size' (int): 卷积核大小(此处假设为正方形, 若需要长方形可传入tuple)
                - 'bias' (bool): 卷积是否使用偏置
                - 'img_size' (tuple): (H, W), 输入图像的空间尺寸
                - 'batch_size' (int): 批大小(主要用于初始化隐藏状态)
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
        
        # input_dim --> hidden_dim
        self.in_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias
        )

        self.cells = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(
                ConvLSTMCell(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    kernel_size=params['kernel_size'],
                    bias=self.bias
                )
            )
            self.bns.append(nn.LayerNorm([self.hidden_dim, self.img_size[0], self.img_size[1]]))

        # hidden_dim --> input_dim (output_dim)
        self.out_conv = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.input_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias
        )

    def forward(self, frames: torch.Tensor, hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Args:
            frames (torch.Tensor): [B, T, C, H, W]
            hidden (list, optional): 每层ConvLSTMCell的 (h, c) 状态, 如果不传则自动初始化为 0

        Returns:
            outputs (torch.Tensor): [B, T, C, H, W], 与输入相同分辨率和通道数
        """
        B, T_in, C, H, W = frames.shape
        seq_len = T_in + self.output_window_size

        if hidden is None:
            hidden = self._init_hidden(batch_size=B)

        outputs = []
        x_prev = None  # 保存上一时刻输出
        for t in range(seq_len):
            if t < T_in:    # 前 T_in 步用真实输入
                x_in = frames[:, t]            # [B, C, H, W]
            else:           # 后 T_out 步用前一步的预测
                x_in = x_prev

            x_in = self.in_conv(x_in)         # [B, hidden_dim, H, W]

            for i, cell in enumerate(self.cells):
                x_in, hidden[i] = cell(x_in, hidden[i])
                x_in = self.bns[i](x_in)

            x_out = self.out_conv(x_in)  # [B, input_dim, H, W]
            x_prev = x_out  # 保存给下一个时刻使用
            outputs.append(x_out)
        
        outputs = torch.stack(outputs, dim=1)  # [B, T, input_dim, H, W]
        pred_outputs = outputs[:, T_in:, :, :, :]

        return pred_outputs
    
    def _init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        初始化每层ConvLSTM的隐藏状态(h, c).
        由于不下采样，img_size用原图尺寸.
        """
        hidden_states = []
        for i in range(self.n_layers):
            hidden_states.append(self.cells[i].init_hidden(batch_size, self.img_size))
        return hidden_states