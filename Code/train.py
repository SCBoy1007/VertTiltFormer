import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from PIL import Image, ImageOps
from dataset import KeypointDataset, collate_fn
from models.model import create_model
from models.loss import KeypointLoss
from visualization import create_validation_visualization

def clean_memory():
    """清理GPU和CPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mean(arr, metric_name=None):
    """安全计算平均值，处理空数组和特殊指标"""
    if len(arr) == 0:
        return 0.0

    # 转换为numpy数组以便处理
    arr = np.array(arr)

    # 移除无效值
    valid_mask = ~(np.isnan(arr) | np.isinf(arr))
    valid_values = arr[valid_mask]

    if len(valid_values) == 0:
        return 0.0

    return float(np.mean(valid_values))

def safe_add_metric(metrics_dict, key, value):
    if not np.isnan(value) and not np.isinf(value):
        metrics_dict[key].append(value)

def train_model(
    train_dir: str,
    val_dir: str,
    model_save_dir: str,
    num_epochs: int = 100,
    batch_size: int = 4,        # 减小batch size以节省显存
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,    # 添加梯度裁剪阈值
    warmup_epochs: int = 5,        # 添加warmup epochs
    device: str = 'cuda',
    num_keypoints: int = 18
):
    # 创建保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 创建数据集和数据加载器
    train_dataset = KeypointDataset(
        img_dir=os.path.join(train_dir, 'images'),
        annotation_dir=os.path.join(train_dir, 'annotations'),
        train=True,
        max_aug_samples=10
    )
    
    val_dataset = KeypointDataset(
        img_dir=os.path.join(val_dir, 'images'),
        annotation_dir=os.path.join(val_dir, 'annotations'),
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    
    # 创建模型（只减少transformer部分的复杂度）
    model = create_model(
        img_size=(768, 256),    # 保持图像尺寸不变
        backbone_type='resnet', # 保持使用ResNet
        num_keypoints=num_keypoints,
        embed_dim=256,          # 保持embedding维度
        num_encoder_layers=6,   # 减少transformer层数
        num_decoder_layers=6,   # 减少transformer层数
        num_heads=4,            # 减少attention heads
        mlp_ratio=4.,          # 保持MLP比率
        dropout=0.1,
        use_checkpointing=True # 关闭checkpointing以加快训练
    ).to(device)

    # 立即设置为训练模式并确保requires_grad
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,}")
    
    # 创建损失函数和优化器
    criterion = KeypointLoss(
        use_smooth_l1=False,
        smooth_l1_beta=1,        # 使用自适应权重
        x_weight = 0.8,  # x坐标的权重
        y_weight = 0.2  # y坐标的权重
    )
    
    # 过滤出需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(
        trainable_params,  # 只优化需要梯度的参数
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # 修改学习率调度器，添加warmup
    def get_lr_multiplier(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_multiplier
    )
    
    # 添加ReduceLROnPlateau作为第二个调度器
    reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # 创建梯度缩放器用于混合精度训练
    scaler = GradScaler()
    
    # TensorBoard
    writer = SummaryWriter(
        os.path.join(model_save_dir, 'logs'),
        max_queue=10,
        flush_secs=60
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_metrics = {
            'x_loss': [],
            'y_loss': [],
            'mean_x_error': [],
            'mean_y_error': [],
            'max_x_error': [],
            'max_y_error': [],
            'mean_pixel_error': [],
            'PCK@0.1': [],
            'order_violations': []
        }
        
        # 确保梯度被正确清零
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in pbar:
            # 清理不需要的缓存
            clean_memory()
            
            try:
                # 准备数据
                images = batch['images'].to(device, non_blocking=True)
                keypoints = [kpts.to(device, non_blocking=True) for kpts in batch['keypoints']]
                
                # 确保requires_grad在autocast之前设置
                images.requires_grad_(True)
                for param in model.parameters():
                    if not param.requires_grad:
                        param.requires_grad_(True)
                
                # 打印形状信息（仅在第一个batch）
                if batch_idx == 0:
                    print(f"\nInput shapes:")
                    print(f"- Images: {images.shape}")
                    print(f"- Keypoints: {[k.shape for k in keypoints]}\n")
                
                # 使用混合精度训练
                with autocast():
                    # 前向传播
                    pred_keypoints = model(images)
                    
                    # 计算损失
                    loss, metrics = criterion(pred_keypoints, keypoints)
                
                # 修改反向传播部分
                scaler.scale(loss).backward()
                
                # 添加梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # 记录损失和指标
                train_losses.append(loss.item())  # 恢复原始损失大小
                for k, v in metrics.items():
                    if k in train_metrics:
                        train_metrics[k].append(v)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'pixel_error': f'{metrics["mean_pixel_error"]:.6f}'
                })
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
            finally:
                # 清理显存
                if 'images' in locals(): del images
                if 'keypoints' in locals(): del keypoints
                if 'pred_keypoints' in locals(): del pred_keypoints
                if 'loss' in locals(): del loss
                clean_memory()
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_metrics = {
            'x_loss': [],
            'y_loss': [],
            'mean_x_error': [],
            'mean_y_error': [],
            'max_x_error': [],
            'max_y_error': [],
            'mean_pixel_error': [],
            'PCK@0.1': [],
            'order_violations': []
        }
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in pbar:
                # 清理缓存
                clean_memory()
                
                images = batch['images'].to(device, non_blocking=True)
                # images.requires_grad_(True)
                keypoints = [kpts.to(device, non_blocking=True) for kpts in batch['keypoints']]
                
                # 使用混合精度
                with autocast():
                    pred_keypoints = model(images)
                    loss, metrics = criterion(pred_keypoints, keypoints)
                
                val_losses.append(loss.item())
                for k, v in metrics.items():
                    if k in val_metrics:
                        safe_add_metric(val_metrics, k, v)

                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'x_error': f'{metrics["mean_x_error"]:.6f}',
                    'y_error': f'{metrics["mean_y_error"]:.6f}'
                })
                
                # 清理显存
                del images, keypoints, pred_keypoints, loss
                clean_memory()
        
        # 计算平均损失和指标
        train_loss = safe_mean(train_losses)
        val_loss = safe_mean(val_losses)
        # 更新学习率
        # scheduler.step()

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for k in train_metrics:
            train_mean = safe_mean(train_metrics[k], k)
            val_mean = safe_mean(val_metrics[k], k)
            if not np.isnan(train_mean) and not np.isnan(val_mean):
                writer.add_scalar(f'{k}/train', train_mean, epoch)
                writer.add_scalar(f'{k}/val', val_mean, epoch)
        
        # 验证集可视化
        if (epoch + 1) % 2 == 0:  # 每2个epoch
            vis_dir = os.path.join(model_save_dir, f'vis_epoch_{epoch+1}')
            create_validation_visualization(
                model,
                val_loader,
                device,
                vis_dir,
                num_samples=4
            )

            # 添加TensorBoard图像
            for img_path in os.listdir(vis_dir)[:2]:  # 添加2个样本
                try:
                    img = Image.open(os.path.join(vis_dir, img_path))

                    # 直接转换为numpy数组，不做resize
                    img = np.array(img)

                    # 确保图像是uint8类型且范围在0-255
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img * 255).astype(np.uint8)
                    # 如果图像是RGBA，转换为RGB
                    if len(img.shape) == 3 and img.shape[-1] == 4:
                        img = img[:, :, :3]
                    # 确保图像是3通道的
                    if len(img.shape) == 2:  # 如果是灰度图
                        img = np.stack([img] * 3, axis=-1)
                    # 转换为TensorBoard期望的格式 (C, H, W)
                    img = img.transpose(2, 0, 1)
                    writer.add_image(f'Predictions/{img_path}', img, epoch)
                except Exception as e:
                    print(f"Error adding image to TensorBoard: {str(e)}")
                    continue
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'reduce_lr_state_dict': reduce_lr.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(model_save_dir, 'best_model.pth'))
        
        # 保存checkpoint
        if (epoch + 1) % 10 == 0:  # 每10个epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 打印训练信息
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {safe_mean(train_losses):.6f}, Val Loss: {safe_mean(val_losses):.6f}')
        print(f'Train X Error: {safe_mean(train_metrics["mean_x_error"]):.6f}, '
              f'Val X Error: {safe_mean(val_metrics["mean_x_error"]):.6f}')
        print(f'Train Y Error: {safe_mean(train_metrics["mean_y_error"]):.6f}, '
              f'Val Y Error: {safe_mean(val_metrics["mean_y_error"]):.6f}')
        
        # 每个epoch结束后清理内存
        if epoch < warmup_epochs:
            scheduler.step()
        else:
            reduce_lr.step(val_loss)
        clean_memory()
    
    writer.close()
    return model

if __name__ == '__main__':
    # 设置CUDA内存分配器
    if torch.cuda.is_available():
        # 启用cudnn benchmark以优化性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # 设置内存分配器
        torch.cuda.empty_cache()
        # 设置memory_format
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 设置训练参数
    train_params = {
        'train_dir': r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\data\train",
        'val_dir': r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\data\train",
        'model_save_dir': r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\checkpoints",
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 2e-4,  
        'max_grad_norm': 1.0,    
        'warmup_epochs': 5,     
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_keypoints': 18
    }
    
    # 开始训练
    model = train_model(**train_params)
