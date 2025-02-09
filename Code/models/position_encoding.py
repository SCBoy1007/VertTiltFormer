import torch
import torch.nn as nn
import math
from typing import Tuple

class PositionalEncoding1D(nn.Module):
    """1D sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin/cos positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, D)
        
        Returns:
            Tensor of shape (B, N, D) with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 1536, max_w: int = 512):
        super().__init__()
        
        if d_model % 4 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model})")
        
        # Calculate feature map size without automatic downsampling
        feat_h = max_h  # Remove the //2 to match your input size
        feat_w = max_w
        
        pe = torch.zeros(feat_h, feat_w, d_model)
        d_model = int(d_model / 2)
        
        y_position = torch.arange(0, feat_h, dtype=torch.float).unsqueeze(1)
        x_position = torch.arange(0, feat_w, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe_h = torch.zeros(feat_h, 1, d_model)
        pe_h[:, 0, 0::2] = torch.sin(y_position * div_term)
        pe_h[:, 0, 1::2] = torch.cos(y_position * div_term)
        
        pe_w = torch.zeros(1, feat_w, d_model)
        pe_w[0, :, 0::2] = torch.sin(x_position * div_term)
        pe_w[0, :, 1::2] = torch.cos(x_position * div_term)
        
        pe[:, :, :d_model] = pe_h + pe_w
        pe[:, :, d_model:] = pe_h + pe_w
        
        pe = pe.reshape(1, feat_h * feat_w, d_model * 2)
        self.register_buffer('pe', pe)
        
        self.max_h = max_h
        self.max_w = max_w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize with normal distribution
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, D)
        
        Returns:
            Tensor of shape (B, N, D) with positional encoding added
        """
        return self.dropout(x + self.pos_embed[:, :x.size(1)])

class LearnedPositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for rectangular image-like inputs"""
    def __init__(
        self,
        d_model: int,
        max_h: int,  # 特征图实际高度
        max_w: int,  # 特征图实际宽度
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 存储特征图尺寸
        self.feat_h = max_h
        self.feat_w = max_w
        
        # 创建位置编码参数
        self.row_embed = nn.Parameter(torch.zeros(max_h, d_model // 2))
        self.col_embed = nn.Parameter(torch.zeros(max_w, d_model // 2))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize with normal distribution
        nn.init.normal_(self.row_embed, std=0.02)
        nn.init.normal_(self.col_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, H*W, D)
        
        Returns:
            Tensor of shape (B, H*W, D) with positional encoding added
        """
        # Create position encodings
        pos_embed = torch.cat([
            self.row_embed[:self.feat_h].unsqueeze(1).repeat(1, self.feat_w, 1),
            self.col_embed[:self.feat_w].unsqueeze(0).repeat(self.feat_h, 1, 1)
        ], dim=-1).reshape(1, self.feat_h * self.feat_w, -1)
        
        return self.dropout(x + pos_embed)
