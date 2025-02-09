import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import KeypointDataset, collate_fn
from models.model import create_model
from visualization import visualize_predictions

def load_model(checkpoint_path: str, model, device: str = 'cuda'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_model(
    model,
    test_loader,
    device: str,
    save_dir: str = None,
    num_visualize: int = 10
):
    """评估模型性能"""
    model.eval()
    
    # 评估指标
    metrics = {
        'mean_pixel_error': [],
        'PCK@0.1': [],
        'PCK@0.2': [],
        'order_violations': []
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            # 准备数据
            images = batch['images'].to(device)
            keypoints = [kpts.to(device) for kpts in batch['keypoints']]
            transform_params = batch['transform_params']
            original_sizes = batch['original_sizes']
            
            # 获取预测结果
            pred_keypoints = model(images)
            
            # 计算指标
            for j in range(len(pred_keypoints)):
                pred = pred_keypoints[j]
                target = keypoints[j]
                
                # 计算像素误差
                pixel_dist = torch.norm(pred - target, dim=-1)
                metrics['mean_pixel_error'].append(pixel_dist.mean().item())
                
                # 计算PCK
                metrics['PCK@0.1'].append((pixel_dist < 0.1).float().mean().item())
                metrics['PCK@0.2'].append((pixel_dist < 0.2).float().mean().item())
                
                # 计算顺序违反
                y_coords = pred[:, 1]
                y_diff = y_coords[:-1] - y_coords[1:]
                num_violations = (y_diff < 0).sum().item()
                metrics['order_violations'].append(num_violations)
            
            # 保存可视化结果
            if save_dir and i < num_visualize:
                os.makedirs(save_dir, exist_ok=True)
                for j in range(len(pred_keypoints)):
                    save_path = os.path.join(save_dir, f'test_sample_{i}_{j}.png')
                    visualize_predictions(
                        images[j].cpu(),
                        pred_keypoints[j].cpu(),
                        keypoints[j].cpu(),
                        transform_params[j],
                        original_sizes[j],
                        save_path
                    )
    
    # 计算平均指标
    results = {k: np.mean(v) for k, v in metrics.items()}
    
    print("\nEvaluation Results:")
    print(f"Mean Pixel Error: {results['mean_pixel_error']:.4f}")
    print(f"PCK@0.1: {results['PCK@0.1']:.4f}")
    print(f"PCK@0.2: {results['PCK@0.2']:.4f}")
    print(f"Average Order Violations: {results['order_violations']:.2f}")
    
    return results

def test_single_image(
    model,
    image_path: str,
    annotation_path: str,
    device: str = 'cuda',
    save_dir: str = None
):
    """测试单张图片"""
    # 创建数据集（只包含一张图片）
    dataset = KeypointDataset(
        img_dir=os.path.dirname(image_path),
        annotation_dir=os.path.dirname(annotation_path),
        train=False
    )
    
    # 获取数据
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    keypoints = sample['keypoints'].unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        pred_keypoints = model(image)
    
    # 可视化结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'single_test_result.png')
        visualize_predictions(
            image[0].cpu(),
            pred_keypoints[0].cpu(),
            keypoints[0].cpu(),
            sample['transform_params'],
            sample['original_size'],
            save_path
        )
    
    return pred_keypoints[0].cpu()

if __name__ == '__main__':
    # 测试参数
    test_params = {

        'checkpoint_path': r'I:\RA-MED\VertTiltFormer\keypoint_detection\Data\checkpoints_Feb5_3\best_model.pth',
        'test_dir': r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\data\train",
        'results_dir': r'F:\RA-MED\Set Transformer\keypoint_detection\test_results_test',
        'batch_size': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 创建模型
    model = create_model(
        img_size=(768, 256),    # 保持图像尺寸不变
        backbone_type='resnet', # 保持使用ResNet
        num_keypoints=18,
        embed_dim=256,          # 保持embedding维度
        num_encoder_layers=6,   # 减少transformer层数
        num_decoder_layers=6,   # 减少transformer层数
        num_heads=4,            # 减少attention heads
        mlp_ratio=4.,          # 保持MLP比率
        dropout=0.1,
        use_checkpointing=True # 关闭checkpointing以加快训练
    ).to(test_params['device'])
    # model = create_model(
    #     img_size=(768,256),  # 使用新的图像尺寸
    #     backbone_type='resnet',
    #     num_keypoints=18
    # ).to(test_params['device'])
    
    # 加载训练好的权重
    model = load_model(test_params['checkpoint_path'], model, test_params['device'])
    
    # 创建测试数据加载器
    test_dataset = KeypointDataset(
        img_dir=os.path.join(test_params['test_dir'], 'images'),
        annotation_dir=os.path.join(test_params['test_dir'], 'annotations'),
        train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_params['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 评估模型
    results = evaluate_model(
        model,
        test_loader,
        test_params['device'],
        test_params['results_dir']
    )
