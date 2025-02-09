import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from typing import List, Tuple


def draw_keypoints_on_image(
        image: Image.Image,
        keypoints: np.ndarray,
        radius: int = 3,
        color: Tuple[int, int, int] = (255, 0, 0),
        draw_order: bool = True,
        order_color: Tuple[int, int, int] = (0, 255, 0),
        violation_color: Tuple[int, int, int] = (255, 0, 0)
) -> Image.Image:
    """
    在图像上绘制关键点和它们之间的连接

    Args:
        image: PIL图像
        keypoints: 关键点坐标数组 (N, 2)
        radius: 关键点圆圈的半径
        color: 关键点的颜色
        draw_order: 是否绘制关键点之间的连接线
        order_color: 正常顺序连接线的颜色
        violation_color: 违反顺序连接线的颜色

    Returns:
        绘制了关键点的图像
    """
    # 转换为PIL的Draw对象
    draw = ImageDraw.Draw(image)

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

            # 检查是否违反顺序（y坐标应该递减）
            line_color = violation_color if y2 > y1 else order_color

            # 绘制连接线
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
        save_path: str = None,
        fig_size: tuple = (20, 10)
) -> None:
    """
    可视化预测结果，支持RGB和灰度图像，并还原到原始图像尺寸
    """
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

    # 绘制预测结果
    img_pred = img_pil.copy()
    img_pred = draw_keypoints_on_image(
        img_pred,
        pred_kpts,
        radius=5,
        color=(255, 0, 0),
        draw_order=True,
        order_color=(0, 255, 0),
        violation_color=(255, 0, 0)
    )
    ax1.imshow(img_pred)
    ax1.set_title('Predictions')
    ax1.axis('off')

    # 绘制目标关键点
    img_target = img_pil.copy()
    img_target = draw_keypoints_on_image(
        img_target,
        target_kpts,
        radius=5,
        color=(0, 255, 0),
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
        fig_size: tuple = (10, 5)
):
    """
    为验证集创建可视化结果

    Args:
        model: 训练好的模型
        val_loader: 验证数据加载器
        device: 设备（'cuda'或'cpu'）
        save_dir: 保存目录
        num_samples: 要可视化的样本数量
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break

            images = batch['images'].to(device)
            keypoints = batch['keypoints']  # 保持关键点在CPU上
            transform_params = batch['transform_params']
            original_sizes = batch['original_sizes']

            # 获取预测结果
            pred_keypoints = model(images)

            # 为每个批次中的样本创建可视化
            for j in range(images.size(0)):
                save_path = os.path.join(save_dir, f'sample_{i}_{j}.png')

                try:
                    visualize_predictions(
                        images[j].cpu(),
                        pred_keypoints[j].cpu(),
                        keypoints[j],
                        transform_params[j],
                        original_sizes[j],
                        save_path,
                        fig_size=fig_size
                    )
                except Exception as e:
                    print(f"Error visualizing sample {i}_{j}: {str(e)}")
                finally:
                    plt.close()  # 确保图像被关闭，释放内存