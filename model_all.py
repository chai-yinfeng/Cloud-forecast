# 该文件包含多个ConvLSTM及其变体的实现, 目前包括模块如下: 
# ConvLSTMSimple: 直接对原图像 (而非采样后的特征图) 进行 ConvLSTM 处理的网络, 作为 baseline 和参考
# ConvLSTMEncode2Decode: 包含上下采样的 ConvLSTM , 只在 U 形编解码器结构的底层特征上进行 ConvLSTM 处理,
# ConvLSTMEncode2DecodeUNet: 在 ConvLSTMEncode2Decode 的基础上增添跳跃连接, 提搞网络保持空间细节的能力
# SA_ConvLSTMEncode2Decode: 与 ConvLSTMEncode2Decode 结构相似, 但对下采样的特征图使用带有自注意力机制的 ConvLSTM 进行处理
# Discriminator & generator_loss_function: 判别器和损失函数, 用于 GAN 训练

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
from skimage.metrics import structural_similarity as ssim

###############################################################################
#                           基础 ConvLSTMCell (公用)                          #
###############################################################################

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool) -> Any:
        """
        初始化 ConvLSTMCell。
        
        Args:
            input_dim (int): 输入张量的通道数。
            hidden_dim (int): 隐状态的通道数。
            kernel_size (int): 卷积核大小。
            bias (bool): 是否在卷积层中使用偏置项。
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播计算。
        
        Args:
            input_tensor (torch.Tensor): 当前时间步的输入张量，形状为 (batch, channels, height, width)。
            cur_state (tuple): 包含当前隐藏状态和记忆状态的元组 (h_cur, c_cur)。
        
        Returns:
            Tuple: (h_next, (h_next, c_next))，分别为下一个隐藏状态及新的状态元组。
        """
        h_cur, c_cur = cur_state    # [batch_size, hidden_dim, height, width]

        combined = torch.cat([input_tensor, h_cur], dim=1)  # [batch_size, input_dim + hidden_dim, height, width]
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)
    
    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化隐藏状态和记忆状态，全零张量，尺寸与编码器输出匹配。
        
        Args:
            batch_size (int): 批次大小。
            image_size (Tuple[int, int]): 特征图的 (height, width)。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 隐藏状态和记忆状态。
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

###############################################################################
#                      1) ConvLSTMSimple (无下采样)                           #
###############################################################################

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

            x = self.in_conv(x)         # [B, hidden_dim, H, W]

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

###############################################################################
#         2) ConvLSTMEncode2Decode (下采样 -> ConvLSTM -> 上采样)            #
###############################################################################

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
    
###############################################################################
#          3) ConvLSTMEncode2DecodeUNet (UNet + ConvLSTM在最底层)            #
###############################################################################

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

###############################################################################
#                  自注意力+记忆结构 (SA_ConvLSTMCell)                       #
###############################################################################

class self_attention_memory_module(nn.Module):
    """
    对 (h, m) 进行自注意力融合，用于更新隐藏状态和记忆。
    
    Args:
        input_dim (int): 输入通道数 (例如 = hidden_dim).
        hidden_dim (int): 内部进行 Q, K 投影时使用的通道数。通常 < input_dim。
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # q, k, v for h
        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        # k2, v2 for m
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)

        # Concatinate attentioned Z_h, Z_m --> Z
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        # (Z, h) --> (mo, mg, mi)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, input_dim, H, W], 当前时刻由 ConvLSTM 得到的 h
        m: [B, input_dim, H, W], 记忆状态(上一时刻或初始化时全0)

        return: new_h, new_m
        """
        B, C, H, W = h.shape

        # (1). Compute Z_h
        # K_h, Q_h: [B, hidden_dim, H, W] --> [B, hidden_dim, H*W], [B, H*W, hidden_dim]
        K_h = self.layer_k(h).view(B, self.hidden_dim, H * W)
        Q_h = self.layer_q(h).view(B, self.hidden_dim, H * W).transpose(1, 2)

        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)    # [B, H*W, H*W]

        # V_h: [B, input_dim, H*W]
        V_h = self.layer_v(h).view(B, self.input_dim, H * W)
        # Z_h: [B, H*W, input_dim] --> [B, input_dim, H, W]
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(B, self.input_dim, H, W)

        # (2). Compute Z_m
        K_m = self.layer_k2(m).view(B, self.hidden_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)

        V_m = self.layer_v2(m).view(B, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_m = Z_m.transpose(1, 2).view(B, self.input_dim, H, W)

        # (3). Combine Z_h, Z_m, merge with h
        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)  # [B, 2*input_dim, H, W] --> [B, 2*input_dim, H, W]

        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)  # [B, input_dim, H, W]

        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m

class SA_ConvLSTMCell(nn.Module):
    """
    在常规ConvLSTMCell的计算 (h, c) 后，额外引入记忆状态 m。
    使用 self_attention_memory_module 对 (h, m) 做自注意力更新，生成新的 (h, m)。
    
    Args:
        params (dict): 
            - 'hidden_dim' (int): 隐状态通道数, 同时也是 x, h 的通道数
            - 'kernel_size' (int): 卷积核大小
            - 'padding' (int): 卷积补零
            - 'att_hidden_dim' (int): 自注意力中 Q,K 的投影维度
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.input_channels = params['hidden_dim']
        self.hidden_dim = params['hidden_dim']
        self.kernel_size = params['kernel_size']
        self.padding = params['padding']

        self.attention_layer = self_attention_memory_module(
            input_dim=params['hidden_dim'],
            hidden_dim=params['att_hidden_dim']
        )

        # (x + h) --> i, f, o, g
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim)
        )

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, hidden_dim, H, W], 当前输入 (下游已将输入通道映射到 hidden_dim)
            hidden: (h, c, m)，分别是:
                h: [B, hidden_dim, H, W]
                c: [B, hidden_dim, H, W]
                m: [B, hidden_dim, H, W]
        
        Returns:
            h_next: [B, hidden_dim, H, W]
            (h_next, c_next, m_next): 新的状态
        """
        h, c, m = hidden

        # ConvLSTM part
        combined = torch.cat([x, h], dim=1) # [B, hidden_dim*2, H, W]
        combined_conv = self.conv2d(combined)   # [B, 4*hidden_dim, H, W]
        i, f, o, g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        # Update h, m
        h_next, m_next = self.attention_layer(h_next, m)
        return h_next, (h_next, c_next, m_next)

    def init_hidden(self, batch_size: int, img_size: Tuple[int, int]):
        """
        h, c, m 全零初始化: [B, hidden_dim, H, W]
        """
        H, W = img_size
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W),
            torch.zeros(batch_size, self.hidden_dim, H, W),
            torch.zeros(batch_size, self.hidden_dim, H, W)
        )
    
###############################################################################
#                  SA_ConvLSTMEncode2Decode (自注意力 + ConvLSTM + 下采样)               #
###############################################################################
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


###############################################################################
#                                 Discriminator                               #
###############################################################################
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

###############################################################################
#                             损失函数 (GAN/SSIM等)                           #
###############################################################################
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


###############################################################################
#                             主要测试入口 (示例)                             #
###############################################################################

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 测试 ConvLSTMSimple
    params = {
        'input_dim': 1, 
        'batch_size': 8, 
        'kernel_size': 3, 
        'img_size': (128, 128), 
        'hidden_dim': 64,
        'n_layers': 4, 
        'bias': True
    }
    conv_model = ConvLSTMSimple(params)
    x1 = torch.rand(params['batch_size'], 6, params['input_dim'], params['img_size'][0], params['img_size'][1])
    output1 = conv_model(x1)
    print("output shape:", output1.shape)  # 预期形状: [8, 6, 1, 128, 128]

    u_conv_model = ConvLSTMEncode2Decode(params)
    x2 = torch.rand(params['batch_size'], 6, params['input_dim'], params['img_size'][0], params['img_size'][1])
    output2 = u_conv_model(x2)
    print("output shape:", output2.shape)

    unet_conv_model = ConvLSTMEncode2DecodeUNet(params)
    x3 = torch.rand(params['batch_size'], 6, params['input_dim'], params['img_size'][0], params['img_size'][1])
    output3 = unet_conv_model(x3)
    print("output shape:", output3.shape)

    # 2) 测试 SA_ConvLSTMEncode2Decode
    params_sa = {
        'batch_size': 4,
        'img_size': (128, 128),
        'n_layers': 2,
        'input_dim': 1,
        'hidden_dim': 32,
        'att_hidden_dim': 16,   # self_attention_memory_module
        'kernel_size': 3,
        'padding': 1,
        'input_window_size': 5,
        'output_window_size': 6
    }

    frames = torch.rand(params_sa['batch_size'], 5, 1, 128, 128)
    target = torch.rand(params_sa['batch_size'], 6, 1, 128, 128)
    mask_true = torch.ones_like(target)
    model_sa = SA_ConvLSTMEncode2Decode(params_sa)
    out_sa = model_sa(frames, target, mask_true, is_training=True)
    print("[Encode2Decode (SA)] out shape:", out_sa.shape)  # => [B, 6, 1, 128, 128]

    # 3) 测试判别器
    disc = Discriminator()
    # 假设总序列=11帧 => (5输入+6预测), 单通道 => [B, 1, T=11, H=128, W=128]
    fake_seq = torch.rand(params_sa['batch_size'], 1, 11, 128, 128)
    d_out = disc(fake_seq)
    print("[Discriminator] out shape:", d_out.shape)