import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class KeypointLoss(nn.Module):
    def __init__(
        self,
        use_smooth_l1: bool = False,
        smooth_l1_beta: float = 1.0,
        x_weight: float = 0.5,
        y_weight: float = 0.2,
        angle_weight: float = 0.299,
        constraint_weight: float = 0.001
    ):
        """
        :param use_smooth_l1: 是否使用Smooth L1代替MSE
        :param smooth_l1_beta: Smooth L1的beta
        :param x_weight: x坐标损失的权重
        :param y_weight: y坐标损失的权重
        :param angle_weight: 角度损失的权重
        :param constraint_weight: 角度约束损失的权重
        """
        super().__init__()
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.x_weight = x_weight
        self.y_weight = y_weight
        self.angle_weight = angle_weight
        self.constraint_weight = constraint_weight

        # 每个锥体（或椎体）的约束阈值（单位：度），需根据自己数据合理调整
        self.angle_thresholds = torch.tensor([
            4.8506, 2.0919, 1.5026, 1.6009, 2.1762, 2.3260,
            2.1743, 2.0768, 1.9951, 2.0089, 1.9652, 2.1529,
            2.5862, 2.6576, 2.5778, 2.7211, 2.5210, 2.6854
        ])

    def calculate_directional_angles(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        计算序列关键点的方向角（相邻关键点之间）。
        假设 keypoints 的形状为 [B, N, 2]，其中 B 是batch数量，N是关键点个数，每个关键点包含(x, y)。
        这里的角度以度为单位，范围强制在 [-180, 180]。
        """
        epsilon = 1e-7
        # dx 的形状: [B, (N-1), 2]
        dx = keypoints[:, 1:] - keypoints[:, :-1]

        # denominator = -dx[..., 1]，避免 denominator == 0
        denominator = -dx[..., 1]
        denominator = torch.where(
            torch.abs(denominator) < epsilon,
            torch.ones_like(denominator) * epsilon * torch.sign(denominator),
            denominator
        )

        # angles = arctan2(dx[..., 0], denominator)，再转换为角度
        angles = torch.atan2(dx[..., 0], denominator)
        angles = torch.rad2deg(angles)

        # 将角度限定在 [-180, 180] 之间
        return torch.clamp(angles, min=-180.0, max=180.0)

    def get_angle_constraint_loss(
        self,
        tilt_angles: torch.Tensor,
        keypoints: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算椎体 tilt_angles 与关键点方向角之间的偏差约束损失。
        tilt_angles: [B, num_vertebrae]，范围应在 [-90, 90] 之间
        keypoints: [B, num_keypoints, 2]，每个关键点为 (x, y)
        mask: 可选的 [B, num_vertebrae] 用于屏蔽某些椎体（例如有缺失标注）。
        """
        # 如果输入本身出现NaN，这里直接返回0
        if torch.isnan(tilt_angles).any():
            return torch.tensor(0.0, device=tilt_angles.device)
        if torch.isnan(keypoints).any():
            return torch.tensor(0.0, device=keypoints.device)

        # 将 angle_thresholds 转移到当前设备
        self.angle_thresholds = self.angle_thresholds.to(tilt_angles.device)

        # clamp 到 [-90, 90] 避免极值
        tilt_angles = torch.clamp(tilt_angles, min=-45, max=45)

        # 计算关键点序列的方向角
        directional_angles = self.calculate_directional_angles(keypoints)

        batch_size, num_vertebrae = tilt_angles.shape
        # directional_angles 形状是 [B, N-1], 其中 N-1 = num_vertebrae - 1 (如果N=18，这里就是17)
        # 需要根据索引 i 分别与 tilt_angles 做对比
        constraint_losses = []

        for i in range(num_vertebrae):
            if i == 0:
                # 第1个椎体与第1段方向角做对比
                angle_diff = torch.abs(tilt_angles[:, i] - directional_angles[:, i])
            elif i == num_vertebrae - 1:
                # 最后1个椎体与最后1段方向角对比
                angle_diff = torch.abs(tilt_angles[:, i] - directional_angles[:, -1])
            else:
                # 中间的椎体与相邻两段方向角的平均对比
                avg_directional = (directional_angles[:, i - 1] + directional_angles[:, i]) / 2.0
                angle_diff = torch.abs(tilt_angles[:, i] - avg_directional)

            # 这里可以再加一层 clamp，避免极端大值
            angle_diff = torch.clamp(angle_diff, 0.0, 60)

            # 应用阈值约束
            threshold = self.angle_thresholds[i]
            constrained_diff = F.relu(angle_diff - threshold)  # 超过阈值的部分才计入损失
            constraint_losses.append(constrained_diff)

        constraint_loss = torch.stack(constraint_losses, dim=1)  # [B, num_vertebrae]

        if mask is not None:
            constraint_loss = constraint_loss * mask

        # 数值稳定性检查
        constraint_loss = torch.where(
            torch.isnan(constraint_loss),
            torch.zeros_like(constraint_loss),
            constraint_loss
        )

        return constraint_loss.mean()

    def forward(
        self,
        pred_keypoints: torch.Tensor,
        target_keypoints: torch.Tensor,
        pred_angles: torch.Tensor,
        target_angles: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        pred_keypoints: [B, N, 2]
        target_keypoints: [B, N, 2] 或 list[tensor, ...]
        pred_angles: [B, M] or [B, M, 1]
        target_angles: [B, M] or [B, M, 1]
        mask: [B, M], 用于屏蔽部分无效椎体（可选）

        返回 (total_loss, loss_dict)，其中 loss_dict 包含监控的指标。
        """
        # 如果 target_keypoints / target_angles 是 list，就 stack 到 batch 维度
        if isinstance(target_keypoints, list):
            target_keypoints = torch.stack(target_keypoints).to(pred_keypoints.device)
        if isinstance(target_angles, list):
            target_angles = torch.stack(target_angles).to(pred_angles.device)

        # NaN 检查
        if torch.isnan(pred_keypoints).any() or torch.isnan(target_keypoints).any():
            raise ValueError("NaN detected in keypoints (pred或target).")
        if torch.isnan(pred_angles).any() or torch.isnan(target_angles).any():
            raise ValueError("NaN detected in angles (pred或target).")

        # 如果 pred_angles / target_angles 形状是 [B, M, 1]，则 squeeze(-1)
        if len(pred_angles.shape) == 3:
            pred_angles = pred_angles.squeeze(-1)
        if len(target_angles.shape) == 3:
            target_angles = target_angles.squeeze(-1)

        # 统一设备
        target_keypoints = target_keypoints.to(pred_keypoints.device)
        target_angles = target_angles.to(pred_angles.device)

        # 强制 clamp 角度范围 [-90, 90]
        pred_angles = torch.clamp(pred_angles, min=-45.0, max=45.0)
        target_angles = torch.clamp(target_angles, min=-45.0, max=45.0)

        # 分离坐标
        pred_x = pred_keypoints[..., 0]
        pred_y = pred_keypoints[..., 1]
        target_x = target_keypoints[..., 0]
        target_y = target_keypoints[..., 1]

        # 这里如果 x,y 是归一化到[0,1]，那么某些可视化或角度计算需要对应地使用统一尺度
        # 如果你的角度是实际角度（-90度到90度），那么坐标是否也需要放缩到实际物理单位？根据业务需求自行斟酌。

        # 选择 Smooth L1 或 MSE
        if self.use_smooth_l1:
            x_loss = F.smooth_l1_loss(pred_x, target_x, beta=self.smooth_l1_beta, reduction='none')
            y_loss = F.smooth_l1_loss(pred_y, target_y, beta=self.smooth_l1_beta, reduction='none')
            angle_loss = F.smooth_l1_loss(pred_angles, target_angles, beta=self.smooth_l1_beta, reduction='none')
        else:
            x_loss = F.mse_loss(pred_x, target_x, reduction='none')
            y_loss = F.mse_loss(pred_y, target_y, reduction='none')
            angle_loss = F.mse_loss(pred_angles, target_angles, reduction='none')

        # 计算椎体的角度约束损失
        constraint_loss = self.get_angle_constraint_loss(pred_angles, pred_keypoints, mask)

        # 加权
        weighted_x_loss = x_loss * self.x_weight
        weighted_y_loss = y_loss * self.y_weight
        weighted_angle_loss = angle_loss * self.angle_weight
        weighted_constraint_loss = constraint_loss * self.constraint_weight

        # 全部求平均
        total_loss = (
            weighted_x_loss.mean() +
            weighted_y_loss.mean() +
            weighted_angle_loss.mean() +
            weighted_constraint_loss
        )

        # 检查最终loss
        if torch.isnan(total_loss):
            print("NaN detected in total loss!")
            print(f"x_loss: {weighted_x_loss.mean().item()}")
            print(f"y_loss: {weighted_y_loss.mean().item()}")
            print(f"angle_loss: {weighted_angle_loss.mean().item()}")
            print(f"constraint_loss: {weighted_constraint_loss.item()}")
            return torch.tensor(0.0, device=total_loss.device), {}

        # 计算辅助监控指标
        with torch.no_grad():
            mean_x_error = torch.abs(pred_x - target_x).mean()
            mean_y_error = torch.abs(pred_y - target_y).mean()
            max_x_error = torch.abs(pred_x - target_x).max()
            max_y_error = torch.abs(pred_y - target_y).max()
            mean_pixel_error = (mean_x_error + mean_y_error) / 2.0

            # 对角度的误差做度数统计
            mean_angle_error = torch.abs(pred_angles - target_angles).mean()
            max_angle_error = torch.abs(pred_angles - target_angles).max()
            # 由于这里 pred_angles 和 target_angles 就是度数范围 [-90, 90]，不需要再额外乘以 90
            # 如果你的网络中把角度归一化到 [-1,1]/[-0.5, 0.5] 等，就需要再乘回缩放系数
            mean_angle_error_degrees = mean_angle_error.item()
            max_angle_error_degrees = max_angle_error.item()

        loss_dict = {
            'total_loss': total_loss.item(),
            'x_loss': weighted_x_loss.mean().item(),
            'y_loss': weighted_y_loss.mean().item(),
            'angle_loss': weighted_angle_loss.mean().item(),
            'constraint_loss': weighted_constraint_loss.item(),
            'mean_x_error': mean_x_error.item(),
            'mean_y_error': mean_y_error.item(),
            'max_x_error': max_x_error.item(),
            'max_y_error': max_y_error.item(),
            'mean_pixel_error': mean_pixel_error.item(),
            'mean_angle_error': mean_angle_error_degrees,
            'max_angle_error': max_angle_error_degrees,
            'order_violations': 0,  # 这里你可以根据需要定义
            'PCK@0.1': 0           # 这里你可以根据需要定义
        }

        return total_loss, loss_dict