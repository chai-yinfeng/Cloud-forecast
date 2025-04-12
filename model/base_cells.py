import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

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
        device = next(self.parameters()).device  # 或 self.conv2d[0].weight.device
        H, W = img_size
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        )