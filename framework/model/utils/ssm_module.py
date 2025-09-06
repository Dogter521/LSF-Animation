import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class CausalConv1d(nn.Module):
    """Causal 1D Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation,
            padding=(kernel_size - 1) * dilation, padding_mode='zeros')
    
    def forward(self, x):
        out = self.conv(x)
        # Remove future padding for causality
        return out[..., :x.shape[-1]]


class MambaSSM(nn.Module):
    """Mamba SSM: Combines State Space Models with Causal Convolutions."""
    def __init__(self, input_dim, num_layers, num_heads, intermediate_dim):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim

        self.layers = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(input_dim, intermediate_dim, kernel_size=3, dilation=2 ** i),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, input_dim)
            ) for i in range(num_layers)
        ])

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        """
        x: Input tensor of shape [B, T, D]
        """
        for layer in self.layers:
            # Apply causal convolution and residual connection
            residual = x
            x = layer(x.transpose(1, 2)).transpose(1, 2)  # [B, D, T] -> [B, T, D]
            x = x + residual

        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output
