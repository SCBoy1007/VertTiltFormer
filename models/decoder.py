import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

from .attention import MemoryEfficientAttention, MemoryEfficientCrossAttention

class MemoryEfficientDecoderLayer(nn.Module):
    """Memory efficient transformer decoder layer"""
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
        self.self_attn = MemoryEfficientAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            chunk_size=128
        )
        
        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MemoryEfficientCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            chunk_size=128
        )
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
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
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
        
        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), context=memory, mask=memory_mask)
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            x.requires_grad_(True)
            memory.requires_grad_(True)
            return checkpoint(self._forward_impl, x, memory, tgt_mask, memory_mask)
        return self._forward_impl(x, memory, tgt_mask, memory_mask)

class EfficientTransformerDecoder(nn.Module):
    """Memory efficient transformer decoder"""
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
        
        self.layers = nn.ModuleList([
            MemoryEfficientDecoderLayer(
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
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)


class EfficientKeypointDecoder(nn.Module):
    """Memory efficient keypoint decoder with improved architecture"""

    def __init__(
            self,
            dim: int,
            num_keypoints: int,
            num_layers: int = 6,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            drop: float = 0.,
            temperature: float = 1.0,
            use_checkpointing: bool = True
    ):
        super().__init__()

        # Learnable queries with reduced dimension
        query_dim = max(dim // 2, 128)  # 减小query维度
        self.query_embed = nn.Parameter(torch.zeros(1, num_keypoints, query_dim))
        self.query_embed.requires_grad_(True)  # 确保需要梯度
        self.query_proj = nn.Linear(query_dim, dim)  # 投影到完整维度
        nn.init.normal_(self.query_embed, std=0.02)

        # 确保所有参数都需要梯度
        for param in self.query_proj.parameters():
            param.requires_grad_(True)

        # Transformer decoder
        self.decoder = EfficientTransformerDecoder(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            use_checkpointing=use_checkpointing
        )

        # 共享特征提取层
        self.shared_features = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(drop)
        )

        # 分离的x坐标预测头
        self.x_head = nn.Sequential(
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim // 4, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )

        # 分离的y坐标预测头
        self.y_head = nn.Sequential(
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim // 4, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )

        # 温度参数
        self.temperature = temperature

        # Lightweight position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )

        self.use_checkpointing = use_checkpointing

    def _forward_impl(
            self,
            memory: torch.Tensor,
            prev_keypoints: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = memory.shape[0]

        # Project queries to full dimension
        queries = self.query_embed.expand(B, -1, -1)
        queries.requires_grad_(True)  # 确保queries需要梯度
        queries = self.query_proj(queries)

        # Add position features if available
        if prev_keypoints is not None:
            pos_features = self.pos_embed(prev_keypoints)
            queries = queries + pos_features

        # Decode features
        decoded = self.decoder(queries, memory, memory_mask=memory_mask)
        decoded = decoded / self.temperature

        # 共享特征提取
        shared_features = self.shared_features(decoded)

        # 分别预测x和y坐标
        x_coords = self.x_head(shared_features)
        y_coords = self.y_head(shared_features)

        # 合并坐标 [B, num_keypoints, 2]
        keypoints = torch.cat([x_coords, y_coords], dim=-1)

        return keypoints

    def forward(
            self,
            memory: torch.Tensor,
            prev_keypoints: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            memory.requires_grad_(True)
            if prev_keypoints is not None:
                prev_keypoints.requires_grad_(True)
            return checkpoint(self._forward_impl, memory, prev_keypoints, memory_mask)
        return self._forward_impl(memory, prev_keypoints, memory_mask)

# 为了向后兼容
TransformerDecoderLayer = MemoryEfficientDecoderLayer
TransformerDecoder = EfficientTransformerDecoder
KeypointDecoder = EfficientKeypointDecoder
