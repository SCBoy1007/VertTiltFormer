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
        # 数据增强（注意：旋转操作已移除，仅保留平移和镜像增强）
        self.augmentation = KeypointAugmentation()

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

    def __getitem__(self, index):
        """根据索引返回数据样本"""
        # 获取当前样本的图片和标注路径
        img_path, ann_path = self.samples[index]

        # 读取图像为灰度图
        image = Image.open(img_path).convert('L')  # 'L' 模式表示灰度图

        # 读取关键点标注
        keypoints = self.read_annotation(ann_path)

        # 对图像和关键点进行预处理（调整到目标尺寸）
        image, keypoints, transform_params = self.process_image_and_keypoints(image, keypoints)
        (scale_w, scale_h), orig_w, orig_h, _ = transform_params

        # 如果是训练模式，进行数据增强
        if self.train:
            aug_images, aug_keypoints = self.augmentation(image, keypoints)
            # 随机选择两个增强样本用于训练，这是实现的关键！
            sample1_idx = np.random.randint(len(aug_images))
            sample2_idx = np.random.randint(len(aug_images))
            while sample2_idx == sample1_idx:  # 确保选择不同的样本
                sample2_idx = np.random.randint(len(aug_images))

            # 返回两个随机选择的增强样本
            img_tensor1 = F.to_tensor(aug_images[sample1_idx]).contiguous()
            img_tensor2 = F.to_tensor(aug_images[sample2_idx]).contiguous()

            kp_tensor1 = torch.tensor(aug_keypoints[sample1_idx], dtype=torch.float32).contiguous()
            kp_tensor2 = torch.tensor(aug_keypoints[sample2_idx], dtype=torch.float32).contiguous()

            # 构建返回的字典，包含两个样本
            sample = {
                'image': torch.stack([img_tensor1, img_tensor2]),  # [2, 1, H, W]
                'keypoints': torch.stack([kp_tensor1, kp_tensor2]),  # [2, num_keypoints, 2]
                'transform_params': transform_params,
                'image_id': os.path.basename(img_path),
                'original_size': (orig_w, orig_h)
            }
        else:
            # 验证模式下只返回原始图像
            img_tensor = F.to_tensor(image).contiguous()
            kp_tensor = torch.tensor(keypoints, dtype=torch.float32).contiguous()
            sample = {
                'image': img_tensor.unsqueeze(0),  # [1, 1, H, W]
                'keypoints': kp_tensor.unsqueeze(0),  # [1, num_keypoints, 2]
                'transform_params': transform_params,
                'image_id': os.path.basename(img_path),
                'original_size': (orig_w, orig_h)
            }

        return sample


    def __len__(self):
        return len(self.samples)





# 数据增强，已移除旋转增强，仅保留平移和镜像
class KeypointAugmentation:
    """关键点检测的数据增强（仅保留平移和镜像）"""
    def __init__(self):
        pass

    def _ensure_tensor(self, data):
        """确保数据是 torch.Tensor 类型"""
        if not isinstance(data, torch.Tensor):
            return torch.from_numpy(data).float()
        return data
    

    def __call__(self, image, keypoints):
        """应用数据增强到图像和关键点"""
        keypoints = self._ensure_tensor(keypoints)

        # 初始化增强后的图像和关键点列表
        augmented_images = [image]
        augmented_keypoints = [keypoints]
        
        # 平移增强
        translated_images, translated_keypoints = self.translate(augmented_images, augmented_keypoints)
        augmented_images.extend(translated_images)
        augmented_keypoints.extend(translated_keypoints)
        
        # 镜像增强
        mirrored_images, mirrored_keypoints = self.mirror(augmented_images, augmented_keypoints)
        augmented_images.extend(mirrored_images)
        augmented_keypoints.extend(mirrored_keypoints)
        
        return augmented_images, augmented_keypoints

    def translate(self, images, keypoints_list):
        """水平平移增强，生成左右两个方向的1/3和2/3位置的样本"""
        translated_images = []
        translated_keypoints = []

        for image, keypoints in zip(images, keypoints_list):
            # 生成1/3位置的平移样本（左右两个方向）
            translated_third_images, keypoints_third = self._translate_image_and_keypoints(
                image, keypoints, position='third')
            translated_images.extend(translated_third_images)
            translated_keypoints.extend(keypoints_third)

            # 生成2/3位置的平移样本（左右两个方向）
            translated_two_thirds_images, keypoints_two_thirds = self._translate_image_and_keypoints(
                image, keypoints, position='two_thirds')
            translated_images.extend(translated_two_thirds_images)
            translated_keypoints.extend(keypoints_two_thirds)

        return translated_images, translated_keypoints

    def _translate_image_and_keypoints(self, image, keypoints, position):
        """平移图像和关键点到指定位置（左右两个方向）"""
        keypoints = self._ensure_tensor(keypoints)
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.from_numpy(keypoints).float()

        # 计算当前关键点的边界
        x_min, y_min = keypoints.min(dim=0).values
        x_max, y_max = keypoints.max(dim=0).values

        # 计算关键点组的中心位置
        center_x = (x_min + x_max) / 2

        # 根据目标位置计算平移量
        if position == 'third':
            targets = [1 / 3, 2 / 3]  # 左右两个目标位置
        elif position == 'two_thirds':
            targets = [1 / 3, 2 / 3]  # 左右两个目标位置
        else:
            raise ValueError(f"Unknown position: {position}")

        translated_images = []
        translated_keypoints = []

        for target_x in targets:
            # 计算需要平移的距离
            dx = target_x - center_x

            # 将平移距离映射到像素范围
            target_w, target_h = 256, 768
            dx_pixels = int(dx * target_w)

            # 平移图像
            translated_image = F.affine(image, angle=0, translate=(-dx_pixels, 0), scale=1.0, shear=0)

            # 更新关键点坐标
            new_keypoints = keypoints.clone()
            new_keypoints[:, 0] = new_keypoints[:, 0] + dx

            # 确保关键点在有效范围内
            new_keypoints[:, 0] = new_keypoints[:, 0].clamp(0, 1)
            new_keypoints[:, 1] = new_keypoints[:, 1].clamp(0, 1)

            translated_images.append(translated_image)
            translated_keypoints.append(new_keypoints)

        return translated_images, translated_keypoints

    def mirror(self, images, keypoints_list):
        """水平镜像增强"""
        mirrored_images = []
        mirrored_keypoints = []
        
        for image, keypoints in zip(images, keypoints_list):
            keypoints = self._ensure_tensor(keypoints)
            # 镜像图像
            mirrored_image = F.hflip(image)
            # 镜像关键点（只镜像 x 坐标）
            mirrored_kps = keypoints.clone()
            mirrored_kps[:, 0] = 1 - mirrored_kps[:, 0]
            mirrored_images.append(mirrored_image)
            mirrored_keypoints.append(mirrored_kps)
        
        return mirrored_images, mirrored_keypoints


def collate_fn(batch):
    """自定义的 collate 函数"""
    all_samples = []
    for sample in batch:
        all_samples.append(sample)
    
    images = []
    for item in all_samples:
        img = item['image']
        if not img.is_contiguous():
            img = img.contiguous()
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        images.append(img)
    images = torch.stack(images).contiguous()
    
    keypoints = []
    for item in all_samples:
        kp = item['keypoints']
        if not kp.is_contiguous():
            kp = kp.contiguous()
        keypoints.append(kp.view(-1, 2))
    
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