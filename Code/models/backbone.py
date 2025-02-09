import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint
import torchvision.models as models

class EarlyDownsample(nn.Module):
    """Light downsampling module with 2x reduction and enhanced feature extraction"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # 特征提取（无下采样）
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 轻量下采样（stride=2）
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路径
        identity = x
        
        # 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # 下采样
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Skip connection
        identity = self.skip(identity)
        
        # 合并
        x = x + identity
        x = self.relu2(x)
        
        return x

class CNNBackbone(nn.Module):
    """Enhanced CNN backbone with light downsampling and gradient checkpointing"""
    def __init__(
        self, 
        in_channels: int = 3, 
        base_channels: int = 64,
        output_channels: int = 256,
        pretrained: bool = True,
        use_checkpointing: bool = True
    ):
        super().__init__()
        
        # Early light downsampling (2x reduction)
        self.early_downsample = EarlyDownsample(in_channels, base_channels)
        
        # 使用预训练的ResNet34作为基础网络，但修改第一层以适应单通道输入
        resnet = models.resnet34(pretrained=pretrained)
        
        # 修改第一个卷积层以适应单通道输入
        if in_channels != 3:  # 如果不是3通道输入
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels, 
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            
            # 初始化新的卷积层
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            # 如果使用预训练模型，将原始权重的均值应用到新的卷积层
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, 0:1, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            
            resnet.conv1 = new_conv
        
        # 确保所有参数都需要梯度
        for param in resnet.parameters():
            param.requires_grad = True
        
        # 完全移除maxpool层，只保留需要的层
        conv1 = nn.Conv2d(base_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.requires_grad_(True)
        
        self.layer1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[block for block in resnet.layer1]
        )
        
        # 确保layer1中的所有参数都需要梯度
        for param in self.layer1.parameters():
            param.requires_grad = True
        
        # 修改所有block的stride为1并确保需要梯度
        self.layer2 = nn.Sequential(
            ResNetBlock(64, 128, stride=1),
            ResNetBlock(128, 128, stride=1)
        )
        for param in self.layer2.parameters():
            param.requires_grad = True
        
        self.layer3 = nn.Sequential(
            ResNetBlock(128, 256, stride=1),
            ResNetBlock(256, 256, stride=1)
        )
        for param in self.layer3.parameters():
            param.requires_grad = True
        
        # 特征调整层
        self.adjust = nn.Sequential(
            nn.Conv2d(256, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.use_checkpointing = use_checkpointing
        self._init_weights()
    
    def _init_weights(self):
        """初始化新添加层的权重"""
        for m in self.early_downsample.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        for m in self.adjust.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _forward_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            x.requires_grad_(True)
            return checkpoint(layer, x)
        return layer(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入需要梯度
        if self.training and torch.is_grad_enabled():
            x.requires_grad_(True)
            
        # 早期下采样
        x = self.early_downsample(x)
        
        # 使用梯度检查点的特征提取
        x = self._forward_layer(self.layer1, x)
        x = self._forward_layer(self.layer2, x)
        x = self._forward_layer(self.layer3, x)
        
        # 特征调整
        x = self.adjust(x)
        return x

class ResNetBlock(nn.Module):
    """
    改进的ResNet块，支持可变通道数和更灵活的降采样
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # 主分支卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 短路连接处理
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主分支计算
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 短路连接
        identity = self.shortcut(x)
        out += identity
        
        # 激活
        out = nn.functional.relu(out)
        return out

class ResNetBackbone(nn.Module):
    """
    重新设计的ResNet Backbone，专注于更有效的特征提取
    """
    def __init__(
        self,
        in_channels: int = 1,  # 默认单通道
        base_channels: int = 64,
        output_channels: int = 256,
        num_blocks: Tuple[int, ...] = (2, 2, 2),
        use_checkpointing: bool = True
    ):
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        
        # 初始卷积层，专门处理单通道输入
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 创建降采样阶段
        self.stage1 = self._make_stage(base_channels, base_channels*2, num_blocks[0], stride=2)
        self.stage2 = self._make_stage(base_channels*2, base_channels*4, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(base_channels*4, base_channels*8, num_blocks[2], stride=2)
        
        # 最终投影层
        self.proj = nn.Conv2d(base_channels*8, output_channels, kernel_size=1)
        
        self._init_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        创建包含多个ResNet块的阶段，并支持降采样
        """
        layers = []
        
        # 第一个块可能需要调整stride和通道数
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        
        # 后续块保持通道数不变
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """
        初始化网络权重，使用Kaiming初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始卷积
        x = self.initial_conv(x)
        
        # 通过阶段
        x = self.stage1(x)  # 降采样到原尺寸1/2
        x = self.stage2(x)  # 降采样到原尺寸1/4
        x = self.stage3(x)  # 降采样到原尺寸1/8
        
        # 最终投影
        x = self.proj(x)
        
        return x
