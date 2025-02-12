import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from typing import List, Tuple
import math


def extend_line_to_image_borders(point, angle_rad, image_shape):
    """计算线段与图像边界的交点"""
    height, width = image_shape[:2]
    x0, y0 = point

    # 计算斜率和截距
    if abs(math.cos(angle_rad)) < 1e-10:  # 近似垂直线
        return [(x0, 0), (x0, height)]

    slope = math.tan(angle_rad)
    b = y0 - slope * x0

    # 可能的交点
    intersections = []

    # 与左边界的交点
    x, y = 0, b
    if 0 <= y <= height:
        intersections.append((x, y))

    # 与右边界的交点
    x, y = width, slope * width + b
    if 0 <= y <= height:
        intersections.append((x, y))

    # 与上边界的交点
    if abs(slope) > 1e-10:  # 不是水平线
        x, y = -b / slope, 0
        if 0 <= x <= width:
            intersections.append((x, y))

    # 与下边界的交点
    if abs(slope) > 1e-10:  # 不是水平线
        x, y = (height - b) / slope, height
        if 0 <= x <= width:
            intersections.append((x, y))

    # 如果找到少于两个交点，返回原始点
    if len(intersections) < 2:
        return [(x0, y0), (x0, y0)]

    # 返回距离最远的两个交点
    intersections = sorted(intersections, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)
    return [intersections[0], intersections[-1]]


def draw_keypoints_on_image(
        image: Image.Image,
        keypoints: np.ndarray,
        angles: np.ndarray = None,
        radius: int = 3,
        color: Tuple[int, int, int] = (255, 0, 0),
        angle_color: Tuple[int, int, int] = (0, 0, 255),
        draw_order: bool = True,
        order_color: Tuple[int, int, int] = (0, 255, 0),
        violation_color: Tuple[int, int, int] = (255, 0, 0)
) -> Image.Image:
    """在图像上绘制关键点、连接线和倾斜角度"""
    draw = ImageDraw.Draw(image)

    # 绘制倾斜角度线
    if angles is not None:
        for (x, y), angle in zip(keypoints, angles):
            angle_rad = math.radians(angle)
            endpoints = extend_line_to_image_borders((x, y), angle_rad, image.size[::-1])
            draw.line(
                [(endpoints[0][0], endpoints[0][1]),
                 (endpoints[1][0], endpoints[1][1])],
                fill=angle_color,
                width=1
            )

    # 绘制关键点
    for x, y in keypoints:
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
            outline=color
        )

    # 绘制连接线
    if draw_order and len(keypoints) > 1:
        for i in range(len(keypoints) - 1):
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[i + 1]
            line_color = violation_color if y2 > y1 else order_color
            draw.line(
                [(x1, y1), (x2, y2)],
                fill=line_color,
                width=2
            )

    return image

def visualize_predictions(
        image: torch.Tensor,
        pred_keypoints: torch.Tensor,
        target_keypoints: torch.Tensor,
        transform_params: tuple,
        original_size: tuple,
        pred_angles: torch.Tensor = None,
        target_angles: torch.Tensor = None,
        save_path: str = None,
        fig_size: tuple = (20, 10)
) -> None:
    """可视化预测结果，包括关键点位置和倾斜角度"""
    # 转换图像为numpy数组
    img_np = image.permute(1, 2, 0).numpy()

    # 处理灰度图像
    if img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)

    # 归一化到0-255范围
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)

    # 创建PIL图像
    img_pil = Image.fromarray(img_np)

    # 还原图像到原始尺寸
    orig_w, orig_h = original_size
    img_pil = img_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # 还原关键点坐标
    (scale_w, scale_h), _, _, _ = transform_params

    def restore_keypoints(kpts):
        kpts = kpts.clone()
        # 从归一化坐标还原到目标尺寸
        kpts[:, 0] = kpts[:, 0] * 256  # 目标宽度
        kpts[:, 1] = kpts[:, 1] * 768  # 目标高度

        # 还原到原始图像尺寸
        kpts[:, 0] = kpts[:, 0] / scale_w
        kpts[:, 1] = kpts[:, 1] / scale_h
        return kpts.numpy()

    pred_kpts = restore_keypoints(pred_keypoints)
    target_kpts = restore_keypoints(target_keypoints)

    # 创建图像网格
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)

    # 转换角度从[-1, 1]到度数
    if pred_angles is not None:
        pred_angles = pred_angles.cpu().numpy() * 90  # 将[-1, 1]范围转换为[-90, 90]度
    if target_angles is not None:
        target_angles = target_angles.cpu().numpy() * 90

    # 绘制预测结果
    img_pred = img_pil.copy()
    img_pred = draw_keypoints_on_image(
        img_pred,
        pred_kpts,
        angles=pred_angles,  # 现在传入的是度数
        radius=5,
        color=(255, 0, 0),
        angle_color=(0, 0, 255),
        draw_order=True,
        order_color=(0, 255, 0),
        violation_color=(255, 0, 0)
    )
    ax1.imshow(img_pred)
    ax1.set_title('Predictions')
    ax1.axis('off')

    # 绘制目标关键点和角度
    img_target = img_pil.copy()
    img_target = draw_keypoints_on_image(
        img_target,
        target_kpts,
        angles=target_angles,  # 现在传入的是度数
        radius=5,
        color=(0, 255, 0),
        angle_color=(0, 0, 255),
        draw_order=True,
        order_color=(0, 255, 0)
    )
    ax2.imshow(img_target)
    ax2.set_title('Ground Truth')
    ax2.axis('off')

    # 添加额外信息
    plt.suptitle(f'Original Image Size: {original_size}')

    # 计算顺序违反数量
    num_violations = 0
    for i in range(len(pred_kpts) - 1):
        if pred_kpts[i + 1][1] > pred_kpts[i][1]:
            num_violations += 1

    if num_violations > 0:
        plt.figtext(
            0.02, 0.02,
            f'Order Violations: {num_violations}',
            color='red',
            fontsize=12
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def create_validation_visualization(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: str,
        save_dir: str,
        num_samples: int = 5,
        fig_size: tuple = (10, 5),
        show_angles: bool = True
):
    """为验证集创建可视化结果，随机选择样本"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 获取验证集的总样本数
    total_samples = len(val_loader.dataset)

    # 随机选择要可视化的样本索引
    selected_indices = np.random.choice(
        total_samples,
        min(num_samples, total_samples),
        replace=False
    )

    samples_visualized = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch_size = batch['images'].size(0)
            batch_start_idx = i * batch_size

            # 检查这个batch中是否有我们要可视化的样本
            batch_indices = np.where(
                (selected_indices >= batch_start_idx) &
                (selected_indices < batch_start_idx + batch_size)
            )[0]

            if len(batch_indices) == 0:
                continue

            images = batch['images'].to(device)
            keypoints = batch['keypoints']
            angles = batch.get('angles', None)  # 使用get方法安全地获取angles
            transform_params = batch['transform_params']
            original_sizes = batch['original_sizes']

            # 获取预测结果
            pred_keypoints, pred_angles = model(images)

            # 只为选中的样本创建可视化
            for idx in batch_indices:
                within_batch_idx = selected_indices[idx] - batch_start_idx
                save_path = os.path.join(save_dir, f'sample_{samples_visualized}.png')

                try:
                    visualize_predictions(
                        images[within_batch_idx].cpu(),
                        pred_keypoints[within_batch_idx].cpu(),
                        keypoints[within_batch_idx],
                        transform_params[within_batch_idx],
                        original_sizes[within_batch_idx],
                        pred_angles[within_batch_idx].cpu() if show_angles else None,
                        angles[within_batch_idx].cpu() if angles is not None and show_angles else None,
                        save_path,
                        fig_size=fig_size
                    )
                    samples_visualized += 1
                except Exception as e:
                    print(f"Error visualizing sample {samples_visualized}: {str(e)}")
                    print(f"Error details: {type(e).__name__}")
                finally:
                    plt.close()

            if samples_visualized >= num_samples:
                break