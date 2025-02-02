import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class KeypointDataset(Dataset):
    """Dataset for keypoint detection"""
    def __init__(
        self,
        img_dir: str,
        annotation_dir: str,
        train: bool = True
    ):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.train = train
        
        # Get all valid image-annotation pairs
        self.samples = []
        for img_name in os.listdir(img_dir):
            if not img_name.endswith('.png'):
                continue
                
            base_name = os.path.splitext(img_name)[0]
            ann_path = os.path.join(annotation_dir, base_name + '.txt')
            
            if os.path.exists(ann_path):
                self.samples.append((
                    os.path.join(img_dir, img_name),
                    ann_path
                ))
    
    def read_annotation(self, file_path):
        """Read keypoint annotations from file"""
        keypoints = []
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip() or ';' in line:
                    continue
                try:
                    coord_str = line.strip()
                    nums = []
                    current_num = ""
                    for char in coord_str:
                        if char.isdigit() or char == '.':
                            current_num += char
                        elif current_num:
                            if current_num.count('.') <= 1:
                                nums.append(float(current_num))
                            current_num = ""
                    
                    if current_num and current_num.count('.') <= 1:
                        nums.append(float(current_num))
                    
                    if len(nums) >= 2:
                        keypoints.append([nums[0], nums[1]])
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}")
                    print(f"Error: {e}")
        return np.array(keypoints)
    
    def process_image_and_keypoints(self, image, keypoints):
        """Process image and keypoints to target size 768x256"""
        # 获取原始尺寸
        orig_w, orig_h = image.size
        
        # 目标尺寸
        target_h, target_w = 768, 256
        
        # 计算缩放比例
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        # 调整图像大小
        image = F.resize(image, (target_h, target_w))
        
        # 调整关键点坐标
        keypoints = keypoints.copy()
        keypoints[:, 0] = keypoints[:, 0] * scale_w  # x坐标使用宽度的缩放比例
        keypoints[:, 1] = keypoints[:, 1] * scale_h  # y坐标使用高度的缩放比例
        
        # 归一化关键点坐标到[0,1]范围
        keypoints[:, 0] = keypoints[:, 0] / target_w
        keypoints[:, 1] = keypoints[:, 1] / target_h
        
        return image, keypoints, ((scale_w, scale_h), orig_w, orig_h, 0)
    
    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        
        # 加载图像为灰度图
        image = Image.open(img_path).convert('L')  # 'L'表示灰度图
        keypoints = self.read_annotation(ann_path)
        
        # 处理图像和关键点
        image, keypoints, transform_params = self.process_image_and_keypoints(image, keypoints)
        (scale_w, scale_h), orig_w, orig_h, _ = transform_params
        
        # 转换为tensor并归一化到[-1,1]范围
        image = F.to_tensor(image)  # 会自动转换为[0,1]范围
        image = (image - 0.5) / 0.5  # 标准化到[-1,1]范围
        
        # 转换关键点为tensor
        keypoints = torch.from_numpy(keypoints).float()
        
        # 获取原始图像尺寸
        original_size = (orig_w, orig_h)
        
        return {
            'image': image,
            'keypoints': keypoints,
            'transform_params': transform_params,
            'image_id': os.path.basename(img_path),
            'original_size': original_size
        }
    
    def __len__(self):
        return len(self.samples)
    
    def recover_coordinates(self, pred_keypoints, transform_params):
        """恢复预测的关键点到原始图像坐标系"""
        (scale_w, scale_h), orig_w, orig_h, _ = transform_params
        
        # 从归一化坐标恢复到目标尺寸坐标
        pred_keypoints = pred_keypoints.clone()
        pred_keypoints[:, 0] = pred_keypoints[:, 0] * 256  # 宽度
        pred_keypoints[:, 1] = pred_keypoints[:, 1] * 768  # 高度
        
        # 恢复到原始图像尺寸
        pred_keypoints[:, 0] = pred_keypoints[:, 0] / scale_w
        pred_keypoints[:, 1] = pred_keypoints[:, 1] / scale_h
        
        return pred_keypoints

# 数据增强
class KeypointAugmentation:
    """关键点检测的数据增强"""
    def __init__(self):
        pass
    
    def __call__(self, image, keypoints):
        """应用数据增强到图像和关键点"""
        # 对于灰度图，只调整对比度
        if torch.rand(1) > 0.5:
            contrast_factor = 1.0 + torch.rand(1) * 0.2 - 0.1
            image = F.adjust_contrast(image, contrast_factor)
        
        return image, keypoints

def collate_fn(batch):
    """自定义的collate函数"""
    images = torch.stack([item['image'] for item in batch])
    keypoints = [item['keypoints'] for item in batch]
    transform_params = [item['transform_params'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'keypoints': keypoints,
        'transform_params': transform_params,
        'image_ids': image_ids,
        'original_sizes': original_sizes
    }
