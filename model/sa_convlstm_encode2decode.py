import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

from model.base_cells import SA_ConvLSTMCell

class SA_ConvLSTMEncode2Decode(nn.Module):
    """
    基于 self-attention 的 ConvLSTM(seq2seq思路) + 下采样 + 上采样 的生成器
    与前面的 ConvLSTMEncode2Decode 思路类似，但中间用 SA_ConvLSTMCell
    使用 scheduled sampling 避免长序列训练难度

    Args:
        params (dict):
            - 'batch_size': int
            - 'img_size': tuple(int, int), 原图尺寸 (H, W)
            - 'n_layers': int, 堆叠的 SA_ConvLSTMCell 数量
            - 'input_window_size': int, 输入帧数
            - 'output_window_size': int, 输出帧数 (即预测的时序长度)
            - 'input_dim': int, 输入通道 (例如=1或3)
            - 'hidden_dim': int, 下采样后特征通道, 也是 SA_ConvLSTMCell 的通道
            - 'att_hidden_dim': int, 注意力中 Q,K 的投影维度
            - 'kernel_size': int
            - 'padding': int
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.n_layers = params['n_layers']
        self.input_window_size = params['input_window_size']
        self.output_window_size = params['output_window_size']  # predicted frame number
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']

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

        # SA_ConvLSTMCell + LayerNorm
        self.cells = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(self.n_layers):
            self.cells.append(SA_ConvLSTMCell(params))
            self.bns.append(nn.LayerNorm((self.hidden_dim, self.img_size[0] // 4, self.img_size[1] // 4)))

        # extra decoder-point for prediction (or output of self.img_decode)
        self.decoder_predict = nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, frames: torch.Tensor, target: torch.Tensor, mask_true: torch.Tensor, is_training: bool, 
                hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Inputs:
            frames: [B, T_in, C, H, W], 输入序列
            target: [B, T_out, C, H, W], 目标序列
            mask_true: [B, T_out, C, H, W], scheduled sampling 时, 1 表示使用 GT, 0 表示用预测
            is_training: bool, 标志是否训练模式 (决定是否拼接 frames+target 以及使用 mask_true)
            hidden: 可选的初始状态 (list of (h, c, m)), 若不提供则初始化为 0

        Returns:
            predict_temp_de: [B, T_out, C, H, W], 生成的预测帧
        """
        if hidden is None:
            hidden = self._init_hidden(batch_size=frames.size(0), img_size=self.img_size)

        # 考虑输入帧和输出帧， 如果算力有限可以写死 range 的大小
        seq_len = self.input_window_size + self.output_window_size

        # In training section, merge frames and target
        if is_training:
            frames = torch.cat([frames, target], dim=1)  # [B, T_in + T_out, C, H, W]

        predict_temp_de = []
        for t in range(seq_len):
            if is_training:
                if t < self.input_window_size:
                    x = frames[:, t]
                else:
                    # scheduled sampling: x = p*gt + (1-p)*out
                    x = mask_true[:, t - self.input_window_size] * frames[:, t] + (1 - mask_true[:, t - self.input_window_size]) * out
            else:
                if t < self.input_window_size:
                    x = frames[:, t]
                else:
                    x = out  # In testing section, use the last predicted frame

            # Down sampling
            x = self.img_encode(x)

            # Multi-layer SA_ConvLSTM
            for i, cell in enumerate(self.cells):
                x, hidden[i] = cell(x, hidden[i])  # out = [B, hidden_dim, H/4, W/4]
                x = self.bns[i](x)

            # Up sampling
            out = self.img_decode(x)
            predict_temp_de.append(out)

        # [B, seq_len, C, H, W]
        predict_temp_de = torch.stack(predict_temp_de, dim=1)

        # only take T_out frames => [B, T_out, C, H, W]
        predict_temp_de = predict_temp_de[:, self.input_window_size :, :, :, :]

        return predict_temp_de

    def _init_hidden(self, batch_size: int, img_size: Tuple[int, int]):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))
        return states