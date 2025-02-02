import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np

class KeypointOrderedLoss(nn.Module):
    """关键点检测损失函数，只包含坐标损失和顺序约束"""
    def __init__(
        self,
        use_smooth_l1: bool = True,
        smooth_l1_beta: float = 0.2,  # 对小误差敏感
        coord_weight: float = 1.0,    # 坐标损失权重
        order_weight: float = 1.0,    # 顺序约束权重，与坐标损失同等重要
        relative_weight: bool = True   # 是否使用自适应权重
    ):
        super().__init__()
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.coord_weight = coord_weight
        self.order_weight = order_weight
        self.relative_weight = relative_weight
    
    def compute_order_violation(self, keypoints: torch.Tensor) -> torch.Tensor:
        """计算关键点顺序违反程度（从上到下的顺序）"""
        y_coords = keypoints[..., 1]
        y_diff = y_coords[:, :-1] - y_coords[:, 1:]
        
        # 使用sigmoid来平滑过渡
        margin = 0.01  # 适应高分辨率的margin值
        violation = F.sigmoid(-y_diff / margin)  # 平滑的违反度量
        
        return violation.mean()
    
    def forward(
        self,
        pred_keypoints: torch.Tensor,
        target_keypoints: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失"""
        if isinstance(target_keypoints, list):
            target_keypoints = torch.stack(target_keypoints)
        
        # 检查输入范围
        if not (pred_keypoints.min() >= 0 and pred_keypoints.max() <= 1):
            raise ValueError("预测的关键点坐标必须在[0,1]范围内")
        if not (target_keypoints.min() >= 0 and target_keypoints.max() <= 1):
            raise ValueError("目标关键点坐标必须在[0,1]范围内")
        
        # 1. 坐标预测损失
        if self.use_smooth_l1:
            point_loss = F.smooth_l1_loss(
                pred_keypoints,
                target_keypoints,
                beta=self.smooth_l1_beta,
                reduction='none'
            )
        else:
            point_loss = F.mse_loss(
                pred_keypoints,
                target_keypoints,
                reduction='none'
            )
        
        point_loss = point_loss.sum(dim=-1)  # (B, N)
        
        # 计算自适应权重（基于点之间的距离）
        if self.relative_weight:
            with torch.no_grad():
                # 计算目标关键点之间的距离
                diff = target_keypoints[:, 1:] - target_keypoints[:, :-1]
                dist = torch.norm(diff, dim=-1)
                
                # 使用距离作为权重
                weights = torch.cat([
                    dist,
                    dist[:, -1:] # 最后一个点使用最后一段距离作为权重
                ], dim=1)
                
                # 归一化权重
                weights = F.softmax(weights, dim=1)
        else:
            weights = torch.ones_like(point_loss) / point_loss.size(1)
        
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        coord_loss = (point_loss * weights).sum(dim=1).mean()
        
        # 2. 顺序约束损失
        order_loss = self.compute_order_violation(pred_keypoints)
        
        # 总损失只包含坐标损失和顺序约束
        total_loss = (
            self.coord_weight * coord_loss +
            self.order_weight * order_loss
        )
        
        # 计算监控指标
        with torch.no_grad():
            pixel_dist = torch.norm(pred_keypoints - target_keypoints, dim=-1)
            mean_pixel_error = pixel_dist.mean()
            max_pixel_error = pixel_dist.max()
            
            pck_01 = (pixel_dist < 0.1).float().mean()
            pck_02 = (pixel_dist < 0.2).float().mean()
            
            y_coords = pred_keypoints[..., 1]
            y_diff = y_coords[:, :-1] - y_coords[:, 1:]
            num_violations = (y_diff < 0).sum().item()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'coord_loss': coord_loss.item(),
            'order_loss': order_loss.item(),
            'mean_pixel_error': mean_pixel_error.item(),
            'max_pixel_error': max_pixel_error.item(),
            'PCK@0.1': pck_01.item(),
            'PCK@0.2': pck_02.item(),
            'order_violations': num_violations
        }
        
        return total_loss, loss_dict
