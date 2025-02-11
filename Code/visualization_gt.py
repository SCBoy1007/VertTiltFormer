import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
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


def visualize_spine_angles(image_path, keypoints_path, angles_path):
    # 读取图像
    try:
        image = Image.open(image_path)
        image = np.array(image)
    except Exception as e:
        print(f"无法读取图像: {e}")
        return

    # 读取关键点坐标
    try:
        with open(keypoints_path, 'r') as f:
            lines = f.readlines()
        keypoints = []
        for line in lines:
            if ',' in line:
                x, y = map(float, line.strip().split(','))
                keypoints.append([x, y])
        keypoints = np.array(keypoints)
    except Exception as e:
        print(f"无法读取关键点文件: {e}")
        return

    # 读取角度数据
    try:
        with open(angles_path, 'r') as f:
            angles = [float(line.strip()) for line in f.readlines()]
    except Exception as e:
        print(f"无法读取角度文件: {e}")
        return

    # 创建图像显示
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)

    # 绘制关键点和连接线
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', marker='o', s=5, label='Vertebrae Centers')
    plt.plot(keypoints[:, 0], keypoints[:, 1], 'r-', linewidth=1, alpha=0.5)

    # 为每个点绘制倾斜角度线
    for point, angle in zip(keypoints, angles):
        # 将角度转换为弧度
        angle_rad = math.radians(angle)

        # 获取延长到图像边界的线段端点
        endpoints = extend_line_to_image_borders(point, angle_rad, image.shape)

        # 绘制角度线
        plt.plot([endpoints[0][0], endpoints[1][0]],
                 [endpoints[0][1], endpoints[1][1]],
                 'b-', linewidth=0.5, alpha=0.7)

    # 添加标签和标题
    plt.title('Spine Vertebrae Centers with Tilt Angles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # 调整布局
    plt.tight_layout()

    # 获取当前脚本的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 从原始图像路径中提取文件名
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 构建输出文件路径
    output_path = os.path.join(script_dir, f'visualization_angles_{image_name}.png')

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"可视化结果已保存到: {output_path}")


# 使用示例
image_path = r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\dataset\train\images\00002.png"
keypoints_path = r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\dataset\train\keypoints\00002.txt"
angles_path = r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\dataset\train\centerline_angles\00002.txt"

visualize_spine_angles(image_path, keypoints_path, angles_path)