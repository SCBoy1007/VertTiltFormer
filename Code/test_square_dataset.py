import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dataset import KeypointDataset

def visualize_processing(dataset, idx):
    """可视化数据处理过程"""
    # 获取原始图像和标注
    img_path, ann_path = dataset.samples[idx]
    original_image = Image.open(img_path).convert('RGB')
    original_keypoints = dataset.read_annotation(ann_path)
    
    # 获取处理后的数据
    sample = dataset[idx]
    processed_image = sample['image']
    processed_keypoints = sample['keypoints']
    transform_params = sample['transform_params']
    
    # 创建图像网格
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # 1. 显示原始图像和关键点
    axes[0].imshow(original_image)
    axes[0].scatter(original_keypoints[:, 0], original_keypoints[:, 1], c='red', s=30)
    axes[0].set_title(f'Original Image\nSize: {original_image.size}')
    
    # 2. 显示填充后的正方形图像（转换回PIL图像）
    processed_img_np = processed_image.permute(1, 2, 0).numpy()
    processed_img_np = (processed_img_np - processed_img_np.min()) / (processed_img_np.max() - processed_img_np.min())
    axes[1].imshow(processed_img_np)
    
    # 转换归一化坐标回像素坐标用于显示
    display_keypoints = processed_keypoints.clone()
    display_keypoints = display_keypoints * dataset.target_size
    axes[1].scatter(display_keypoints[:, 0], display_keypoints[:, 1], c='red', s=30)
    axes[1].set_title(f'Processed Square Image\nSize: {dataset.target_size}x{dataset.target_size}')
    
    # 3. 恢复坐标并显示
    recovered_keypoints = dataset.recover_coordinates(processed_keypoints, transform_params)
    axes[2].imshow(original_image)
    axes[2].scatter(recovered_keypoints[:, 0], recovered_keypoints[:, 1], c='red', s=30)
    axes[2].set_title('Recovered Coordinates\nCompare with Original')
    
    plt.tight_layout()
    plt.show()
    
    # 打印坐标信息
    print("\n=== Coordinate Comparison ===")
    print("Original Keypoints (first 3):")
    print(original_keypoints[:3])
    print("\nProcessed Keypoints (normalized, first 3):")
    print(processed_keypoints[:3])
    print("\nRecovered Keypoints (first 3):")
    print(recovered_keypoints[:3])
    
    # 计算恢复误差
    error = np.abs(original_keypoints - recovered_keypoints.numpy())
    mean_error = error.mean()
    max_error = error.max()
    print(f"\nMean recovery error: {mean_error:.4f} pixels")
    print(f"Max recovery error: {max_error:.4f} pixels")

if __name__ == "__main__":
    # 创建数据集
    dataset = KeypointDataset(
        img_dir=r"F:\RA-MED\Set Transformer\Data\TrialTrainingDataset\Xray+Json(HD)\Cleaned\Xray_Cleaned",
        annotation_dir=r"F:\RA-MED\Set Transformer\Data\TrialTrainingDataset\Generate Ground Truth\Data\GeometricCenters",
        target_size=1024
    )
    
    # 测试前3个样本
    for i in range(3):
        print(f"\nTesting sample {i+1}...")
        visualize_processing(dataset, i)
