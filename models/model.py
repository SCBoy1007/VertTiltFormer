import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint
import gc

from .backbone import CNNBackbone, ResNetBackbone
from .encoder import TransformerEncoder
from .decoder import KeypointDecoder
from .position_encoding import PositionalEncoding2D, LearnedPositionalEncoding2D

class KeypointDetector(nn.Module):
    """Complete keypoint detection model combining CNN and Transformer"""
    def __init__(
        self,
        img_size: Tuple[int, int] = (768, 256),
        in_channels: int = 1,  # 修改为单通道灰度图
        backbone_type: str = 'cnn',
        embed_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        num_keypoints: int = 17,
        dropout: float = 0.1,
        pos_encoding_type: str = 'learned',
        use_checkpointing: bool = True
    ):
        super().__init__()
        
        # Save parameters
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_checkpointing = use_checkpointing
        
        # 1. CNN Backbone
        if backbone_type == 'cnn':
            self.backbone = CNNBackbone(
                in_channels=in_channels,
                base_channels=64,
                output_channels=embed_dim,
                use_checkpointing=use_checkpointing
            )
        elif backbone_type == 'resnet':
            self.backbone = ResNetBackbone(
                in_channels=in_channels,
                base_channels=64,
                output_channels=embed_dim,
                use_checkpointing=use_checkpointing
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # Calculate feature map size after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size[0], img_size[1])
            feat_map = self.backbone(dummy)
            feat_h, feat_w = feat_map.shape[-2:]
            del dummy, feat_map
            torch.cuda.empty_cache()
        
        # 2. Position Encoding
        if pos_encoding_type == 'learned':
            self.pos_encoding = LearnedPositionalEncoding2D(
                d_model=embed_dim,
                max_h=feat_h,  # 使用特征图的实际高度
                max_w=feat_w,  # 使用特征图的实际宽度
                dropout=dropout
            )
        else:
            self.pos_encoding = PositionalEncoding2D(
                d_model=embed_dim,
                max_h=feat_h,  # 使用特征图的实际高度
                max_w=feat_w   # 使用特征图的实际宽度
            )
        
        # Store feature dimensions for later use
        self.feat_h = feat_h
        self.feat_w = feat_w
        
        # 3. Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            dim=embed_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout
        )
        
        # 5. Keypoint Decoder
        self.keypoint_decoder = KeypointDecoder(
            dim=embed_dim,
            num_keypoints=num_keypoints,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for non-pretrained components"""
        def _init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.transformer_encoder.apply(_init_fn)
        self.keypoint_decoder.apply(_init_fn)
    
    def _forward_transformer_encoder(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return checkpoint(self.transformer_encoder, x)
        return self.transformer_encoder(x)
    
    def _forward_keypoint_decoder(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return checkpoint(self.keypoint_decoder, x)
        return self.keypoint_decoder(x)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            mask: Optional attention mask
        
        Returns:
            Predicted keypoint coordinates of shape (B, num_keypoints, 2)
        """
        # 1. Extract features using CNN backbone
        features = self.backbone(x)  # (B, embed_dim, H', W')
        B, C, H, W = features.shape
        
        # 2. Reshape features and add position encoding
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        
        # Verify feature dimensions match expected size
        if features.size(1) != self.feat_h * self.feat_w:
            raise ValueError(f"Feature map size mismatch. Expected {self.feat_h * self.feat_w} positions, got {features.size(1)}")
            
        features = self.pos_encoding(features)
        
        # 3. Process through transformer encoder (features already in correct shape B, H*W, D)
        memory = self._forward_transformer_encoder(features)
        
        # 5. Decode keypoints
        keypoints = self._forward_keypoint_decoder(memory)
        
        # Clear unnecessary tensors
        del features, memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return keypoints

def create_model(
    img_size: Tuple[int, int] = (768, 256),
    in_channels: int = 1,  # 修改为单通道灰度图
    backbone_type: str = 'cnn',
    embed_dim: int = 256,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.,
    num_keypoints: int = 17,
    dropout: float = 0.1,
    pos_encoding_type: str = 'learned',
    use_checkpointing: bool = True
) -> KeypointDetector:
    """Create a keypoint detection model with specified parameters"""
    model = KeypointDetector(
        img_size=img_size,
        in_channels=in_channels,
        backbone_type=backbone_type,
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_keypoints=num_keypoints,
        dropout=dropout,
        pos_encoding_type=pos_encoding_type,
        use_checkpointing=use_checkpointing
    )
    return model

if __name__ == "__main__":
    # Example usage
    model = create_model(
        img_size=(768, 256),  # 使用新的图像尺寸
        in_channels=1,        # 使用单通道
        backbone_type='resnet',
        embed_dim=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_keypoints=17,
        use_checkpointing=True
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 768, 256)  # 单通道输入
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (2, 17, 2)
