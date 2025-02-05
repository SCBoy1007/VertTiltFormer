import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class KeypointLoss(nn.Module):
    def __init__(
            self,
            use_smooth_l1: bool = False,
            smooth_l1_beta: float = 1.0,
            x_weight: float = 2.0,  # x坐标的权重
            y_weight: float = 1.0  # y坐标的权重
    ):
        super().__init__()
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.x_weight = x_weight
        self.y_weight = y_weight

    def forward(
            self,
            pred_keypoints: torch.Tensor,
            target_keypoints: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(target_keypoints, list):
            target_keypoints = torch.stack(target_keypoints)

        # 分离x和y坐标
        pred_x = pred_keypoints[..., 0]
        pred_y = pred_keypoints[..., 1]
        target_x = target_keypoints[..., 0]
        target_y = target_keypoints[..., 1]

        # 分别计算x和y的损失
        if self.use_smooth_l1:
            x_loss = F.smooth_l1_loss(
                pred_x, target_x,
                beta=self.smooth_l1_beta,
                reduction='none'
            )
            y_loss = F.smooth_l1_loss(
                pred_y, target_y,
                beta=self.smooth_l1_beta,
                reduction='none'
            )
        else:
            x_loss = F.mse_loss(pred_x, target_x, reduction='none')
            y_loss = F.mse_loss(pred_y, target_y, reduction='none')

        # 应用权重
        weighted_x_loss = x_loss * self.x_weight
        weighted_y_loss = y_loss * self.y_weight

        # 合并损失
        point_loss = weighted_x_loss + weighted_y_loss

        if mask is not None:
            point_loss = point_loss * mask

        total_loss = point_loss.mean()

        # 计算监控指标
        with torch.no_grad():
            mean_x_error = torch.abs(pred_x - target_x).mean()
            mean_y_error = torch.abs(pred_y - target_y).mean()
            max_x_error = torch.abs(pred_x - target_x).max()
            max_y_error = torch.abs(pred_y - target_y).max()
            mean_pixel_error = (mean_x_error + mean_y_error) / 2

        loss_dict = {
            'total_loss': total_loss.item(),
            'x_loss': weighted_x_loss.mean().item(),
            'y_loss': weighted_y_loss.mean().item(),
            'mean_x_error': mean_x_error.item(),
            'mean_y_error': mean_y_error.item(),
            'max_x_error': max_x_error.item(),
            'max_y_error': max_y_error.item(),
            'mean_pixel_error': mean_pixel_error.item(),  # 添加这一行
            'order_violations': 0,  # 添加这一行
            'PCK@0.1': 0  # 添加这一行
        }

        return total_loss, loss_dict