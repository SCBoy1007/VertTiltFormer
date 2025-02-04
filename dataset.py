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
        # 数据增强
        augmentation = KeypointAugmentation()
    
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
        
        # 数据增强（仅在训练集上应用）
        if self.train:
            augmented_images, augmented_keypoints = self.augmentation(image, keypoints)
        else:
            augmented_images = [image]
            augmented_keypoints = [keypoints]
        
        # 转换为tensor并归一化到[-1,1]范围
        augmented_tensors = []
        for img, kp in zip(augmented_images, augmented_keypoints):
            img_tensor = F.to_tensor(img)  # 会自动转换为[0,1]范围
            img_tensor = (img_tensor - 0.5) / 0.5  # 标准化到[-1,1]范围
            kp_tensor = torch.from_numpy(kp).float()
            augmented_tensors.append({
                'image': img_tensor,
                'keypoints': kp_tensor,
                'transform_params': transform_params,
                'image_id': os.path.basename(img_path),
                'original_size': (orig_w, orig_h)
            })
        
        return augmented_tensors

    
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
        # 初始化增强后的图像和关键点列表
        augmented_images = [image]
        augmented_keypoints = [keypoints]
        
        # 旋转增强
        rotated_images, rotated_keypoints = self.rotate(image, keypoints)
        augmented_images.extend(rotated_images)
        augmented_keypoints.extend(rotated_keypoints)
        
        # 平移增强
        translated_images, translated_keypoints = self.translate(augmented_images, augmented_keypoints)
        augmented_images.extend(translated_images)
        augmented_keypoints.extend(translated_keypoints)
        
        # 镜像增强
        mirrored_images, mirrored_keypoints = self.mirror(augmented_images, augmented_keypoints)
        augmented_images.extend(mirrored_images)
        augmented_keypoints.extend(mirrored_keypoints)
        
        return augmented_images, augmented_keypoints

    def rotate(self, image, keypoints):
        """基于几何中心的旋转增强"""
        rotated_images = []
        rotated_keypoints = []
        
        # 逆时针旋转
        angle_ccw = torch.randint(1, 11, (1,)).item()  # 1-10 度
        rotated_image_ccw, rotated_kps_ccw = self._rotate_image_and_keypoints(image, keypoints, angle_ccw)
        rotated_images.append(rotated_image_ccw)
        rotated_keypoints.append(rotated_kps_ccw)
        
        # 顺时针旋转
        angle_cw = torch.randint(1, 11, (1,)).item()  # 1-10 度
        rotated_image_cw, rotated_kps_cw = self._rotate_image_and_keypoints(image, keypoints, -angle_cw)
        rotated_images.append(rotated_image_cw)
        rotated_keypoints.append(rotated_kps_cw)
        
        return rotated_images, rotated_keypoints

    def _rotate_image_and_keypoints(self, image, keypoints, angle):
        """旋转图像和关键点"""
        if len(keypoints) == 0:
            return image, keypoints
        
        # 计算几何中心点
        center_x, center_y = keypoints.mean(dim=0)
        
        # 旋转图像
        rotated_image = F.rotate(image, angle, center=(center_x, center_y))
        
        # 更新关键点坐标
        theta = np.radians(angle)
        cos = np.cos(theta)
        sin = np.sin(theta)
        
        keypoints_rotated = keypoints - torch.tensor([center_x, center_y])
        keypoints_rotated[:, 0] = keypoints_rotated[:, 0] * cos - keypoints_rotated[:, 1] * sin
        keypoints_rotated[:, 1] = keypoints_rotated[:, 0] * sin + keypoints_rotated[:, 1] * cos
        keypoints_rotated = keypoints_rotated + torch.tensor([center_x, center_y])
        
        # 确保关键点在范围内
        keypoints_rotated[:, 0] = keypoints_rotated[:, 0].clamp(0, 1)
        keypoints_rotated[:, 1] = keypoints_rotated[:, 1].clamp(0, 1)
        
        return rotated_image, keypoints_rotated

    def translate(self, images, keypoints_list):
        """仅进行左右平移增强"""
        translated_images = []
        translated_keypoints = []
        
        for image, keypoints in zip(images, keypoints_list):
            # 左移
            translated_left, keypoints_left = self._translate_image_and_keypoints(image, keypoints, direction='left')
            translated_images.append(translated_left)
            translated_keypoints.append(keypoints_left)
            
            # 右移
            translated_right, keypoints_right = self._translate_image_and_keypoints(image, keypoints, direction='right')
            translated_images.append(translated_right)
            translated_keypoints.append(keypoints_right)
        
        return translated_images, translated_keypoints

    def _translate_image_and_keypoints(self, image, keypoints, direction):
        """平移图像和关键点"""
        x_min, y_min = keypoints.min(dim=0).values
        x_max, y_max = keypoints.max(dim=0).values
        
        # 计算可平移范围
        dx_left = x_min / 3
        dx_right = (1 - x_max) / 3
        
        if direction == 'left':
            dx = dx_left
            dy = 0
        elif direction == 'right':
            dx = dx_right
            dy = 0
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # 将平移距离映射到像素范围
        target_w, target_h = 256, 768  # 归一化后的目标尺寸
        dx_pixels = int(dx * target_w)
        dy_pixels = int(dy * target_h)
        
        # 平移图像
        translated_image = F.affine(image, angle=0, translate=(-dx_pixels, -dy_pixels), scale=1.0, shear=0)
        
        # 更新关键点坐标
        keypoints[:, 0] = keypoints[:, 0] - dx
        keypoints[:, 1] = keypoints[:, 1] - dy
        
        # 确保关键点在范围内
        keypoints[:, 0] = keypoints[:, 0].clamp(0, 1)
        keypoints[:, 1] = keypoints[:, 1].clamp(0, 1)
        
        return translated_image, keypoints

    def mirror(self, images, keypoints_list):
        """水平镜像增强"""
        mirrored_images = []
        mirrored_keypoints = []
        
        for image, keypoints in zip(images, keypoints_list):
            # 镜像图像
            mirrored_image = F.hflip(image)
            # 镜像关键点
            mirrored_kps = keypoints.clone()
            mirrored_kps[:, 0] = 1 - mirrored_kps[:, 0]  # 只镜像x坐标
            mirrored_images.append(mirrored_image)
            mirrored_keypoints.append(mirrored_kps)
        
        return mirrored_images, mirrored_keypoints

        
def collate_fn(batch):
    """自定义的collate函数"""
    all_samples = []
    for sample in batch:
        all_samples.extend(sample)
    
    images = torch.stack([item['image'] for item in all_samples])
    keypoints = [item['keypoints'] for item in all_samples]
    transform_params = [item['transform_params'] for item in all_samples]
    image_ids = [item['image_id'] for item in all_samples]
    original_sizes = [item['original_size'] for item in all_samples]
    
    return {
        'images': images,
        'keypoints': keypoints,
        'transform_params': transform_params,
        'image_ids': image_ids,
        'original_sizes': original_sizes
    }
