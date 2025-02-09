import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

class MemoryEfficientAttention(nn.Module):
    """Memory efficient attention implementation"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        chunk_size: int = 128
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.chunk_size = chunk_size
        
        # 分离QKV投影以减少参数量
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def _chunk_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        """Compute attention scores in chunks to save memory"""
        B, H, N, D = q.shape
        S = k.size(2)
        
        out = torch.zeros_like(q)
        
        # 分块计算attention
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            for j in range(0, S, chunk_size):
                end_j = min(j + chunk_size, S)
                
                # 计算当前块的attention scores
                attn_chunk = torch.matmul(q[:, :, i:end_i], k[:, :, j:end_j].transpose(-2, -1)) * self.scale
                attn_chunk = attn_chunk.softmax(dim=-1)
                attn_chunk = self.attn_drop(attn_chunk)
                
                # 更新输出
                out[:, :, i:end_i] += torch.matmul(attn_chunk, v[:, :, j:end_j])
        
        return out
    
    def _attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for attention computation"""
        B, N, C = q.shape
        
        # 重塑为多头形式
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 分块计算attention
        attn_output = self._chunk_attention(q, k, v, self.chunk_size)
        
        # 重塑回原始维度
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask
            key_padding_mask: Optional key padding mask
        """
        # 投影Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 使用梯度检查点计算attention
        if self.training and torch.is_grad_enabled():
            # 确保输入需要梯度
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            out = checkpoint(self._attention_forward, q, k, v, mask)
        else:
            out = self._attention_forward(q, k, v, mask)
        
        # 最终投影
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out

class MemoryEfficientCrossAttention(nn.Module):
    """Memory efficient cross-attention implementation"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        chunk_size: int = 128
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.chunk_size = chunk_size
        
        # 分离投影
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def _chunk_cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        """Compute cross-attention scores in chunks"""
        B, H, N, D = q.shape
        S = k.size(2)
        
        out = torch.zeros_like(q)
        
        # 分块计算cross-attention
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            for j in range(0, S, chunk_size):
                end_j = min(j + chunk_size, S)
                
                # 计算当前块的attention scores
                attn_chunk = torch.matmul(q[:, :, i:end_i], k[:, :, j:end_j].transpose(-2, -1)) * self.scale
                attn_chunk = attn_chunk.softmax(dim=-1)
                attn_chunk = self.attn_drop(attn_chunk)
                
                # 更新输出
                out[:, :, i:end_i] += torch.matmul(attn_chunk, v[:, :, j:end_j])
        
        return out
    
    def _cross_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for cross-attention computation"""
        B, N, C = q.shape
        M = k.shape[1]
        
        # 重塑为多头形式
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 分块计算attention
        attn_output = self._chunk_cross_attention(q, k, v, self.chunk_size)
        
        # 重塑回原始维度
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor (B, N, C)
            context: Key/Value tensor (B, M, C)
            mask: Optional attention mask
        """
        # 投影Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        # 使用梯度检查点计算cross-attention
        if self.training and torch.is_grad_enabled():
            # 确保输入需要梯度
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            out = checkpoint(self._cross_attention_forward, q, k, v, mask)
        else:
            out = self._cross_attention_forward(q, k, v, mask)
        
        # 最终投影
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out

# 为了向后兼容，保留原始类名
MultiHeadAttention = MemoryEfficientAttention
CrossAttention = MemoryEfficientCrossAttention
