import torch
import torch.nn as nn
from typing import Optional
from torch.utils.checkpoint import checkpoint

from .attention import MemoryEfficientAttention

class TransformerEncoderLayer(nn.Module):
    """Memory efficient transformer encoder layer"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        act_layer: nn.Module = nn.GELU,
        use_checkpointing: bool = True
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MemoryEfficientAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            chunk_size=128  # 使用分块attention
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.use_checkpointing = use_checkpointing
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention block
        x = x + self.attn(self.norm1(x), mask, key_padding_mask)
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            x.requires_grad_(True)
            return checkpoint(self._forward_impl, x, mask, key_padding_mask)
        return self._forward_impl(x, mask, key_padding_mask)

class TransformerEncoder(nn.Module):
    """Memory efficient transformer encoder"""
    def __init__(
        self,
        dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        act_layer: nn.Module = nn.GELU,
        use_checkpointing: bool = True
    ):
        super().__init__()
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                use_checkpointing=use_checkpointing
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.use_checkpointing = use_checkpointing
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Apply transformer layers with optional checkpointing
        for layer in self.layers:
            x = layer(x, mask, key_padding_mask)
        
        return self.norm(x)

class EfficientFeatureEncoder(nn.Module):
    """Memory efficient feature encoder with improved architecture"""
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        use_checkpointing: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Channel reduction to save memory
        self.channel_reduce = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, embed_dim),
            nn.GELU()
        )
        
        # Unified transformer for joint spatial-channel attention
        self.transformer = TransformerEncoder(
            dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            use_checkpointing=use_checkpointing
        )
        
        # Lightweight refinement
        self.refine = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(drop)
        )
        
        self.use_checkpointing = use_checkpointing
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Reshape and reduce channels
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.channel_reduce(x)  # (B, H*W, embed_dim)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Refinement
        x = self.refine(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            x.requires_grad_(True)
            return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)

# 为了向后兼容
FeatureEncoder = EfficientFeatureEncoder
