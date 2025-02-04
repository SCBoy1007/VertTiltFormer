import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class KeypointLoss(nn.Module):
    """简化的关键点检测损失函数"""
    def __init__(
        self,
        use_smooth_l1: bool = False,    # 默认使用MSE损失
        smooth_l1_beta: float = 1.0,    # 默认不特别关注小误差
        relative_weight: bool = False    # 默认不使用自适应权重
    ):
        super().__init__()
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.relative_weight = relative_weight
    
    def forward(
        self,
        pred_keypoints: torch.Tensor,
        target_keypoints: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """计算损失"""
        if isinstance(target_keypoints, list):
            target_keypoints = torch.stack(target_keypoints)
        
        # 检查输入范围
        if not (pred_keypoints.min() >= 0 and pred_keypoints.max() <= 1):
            raise ValueError("预测的关键点坐标必须在[0,1]范围内")
        if not (target_keypoints.min() >= 0 and target_keypoints.max() <= 1):
            raise ValueError("目标关键点坐标必须在[0,1]范围内")
        
        # 计算坐标预测损失
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
        
        # 权重计算
        if self.relative_weight:
            with torch.no_grad():
                diff = target_keypoints[:, 1:] - target_keypoints[:, :-1]
                dist = torch.norm(diff, dim=-1)
                weights = torch.cat([
                    dist,
                    dist[:, -1:]
                ], dim=1)
                weights = F.softmax(weights, dim=1)
        else:
            weights = torch.ones_like(point_loss) / point_loss.size(1)
        
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        total_loss = (point_loss * weights).sum(dim=1).mean()
        
        # 计算监控指标
        with torch.no_grad():
            pixel_dist = torch.norm(pred_keypoints - target_keypoints, dim=-1)
            mean_pixel_error = pixel_dist.mean()
            max_pixel_error = pixel_dist.max()
            
            pck_01 = (pixel_dist < 0.1).float().mean()
            pck_02 = (pixel_dist < 0.2).float().mean()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'mean_pixel_error': mean_pixel_error.item(),
            'max_pixel_error': max_pixel_error.item(),
            'PCK@0.1': pck_01.item(),
            'PCK@0.2': pck_02.item(),
        }
        
        return total_loss, loss_dict