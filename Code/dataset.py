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
            train: bool = True,
            max_aug_samples: int = 2  # 添加新参数
    ):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.train = train

        # 获取所有有效的图像-标注对
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

        # 初始化数据增强器，传入最大增强样本数量
        self.augmentation = KeypointAugmentation(max_aug_samples=max_aug_samples)

    def read_angles(self, file_path):
        """读取角度标注文件"""
        angles = []
        base_name = os.path.splitext(file_path)[0]
        angle_path = base_name.replace('keypoints', 'centerline_angles') + '.txt'

        try:
            with open(angle_path, 'r') as f:
                for line in f:
                    if line.strip():  # 忽略空行
                        angle = float(line.strip())
                        angles.append(angle)
        except Exception as e:
            print(f"Error reading angle file {angle_path}: {e}")
            return None

        return np.array(angles)

    def recover_angles(self, pred_angles):
        """从归一化范围恢复到原始角度范围"""
        # 从[-1, 1]恢复到[-90, 90]
        return pred_angles * 90.0
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

    def process_image_and_keypoints(self, image, keypoints, angles):
        """Process image, keypoints and angles to target size and normalized range"""
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
        keypoints[:, 0] = keypoints[:, 0] * scale_w
        keypoints[:, 1] = keypoints[:, 1] * scale_h

        # 归一化关键点坐标到[0,1]范围
        keypoints[:, 0] = keypoints[:, 0] / target_w
        keypoints[:, 1] = keypoints[:, 1] / target_h

        # 归一化角度到[-1,1]范围
        angles = angles.copy()
        angles = angles / 90.0  # 从[-90,90]归一化到[-1,1]

        return image, keypoints, angles, ((scale_w, scale_h), orig_w, orig_h, 0)

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
        original_idx = index // self.augmentation.max_aug_samples
        aug_idx = index % self.augmentation.max_aug_samples

        img_path, ann_path = self.samples[original_idx]
        image = Image.open(img_path).convert('L')

        # 读取关键点和角度
        keypoints = self.read_annotation(ann_path)
        angles = self.read_angles(ann_path)

        # 处理数据
        image, keypoints, angles, transform_params = self.process_image_and_keypoints(
            image, keypoints, angles)

        # 从transform_params中解包出原始尺寸
        (scale_w, scale_h), orig_w, orig_h, _ = transform_params

        # 数据增强
        if self.train:
            aug_images, aug_keypoints, aug_angles = self.augmentation(image, keypoints, angles)
            image = aug_images[aug_idx]
            keypoints = aug_keypoints[aug_idx]
            angles = aug_angles[aug_idx]

        # 转换为tensor
        img_tensor = F.to_tensor(image)
        kp_tensor = torch.tensor(keypoints, dtype=torch.float32)
        angles_tensor = torch.tensor(angles, dtype=torch.float32)

        return {
            'image': img_tensor,
            'keypoints': kp_tensor,
            'angles': angles_tensor,
            'transform_params': transform_params,
            'image_id': os.path.basename(img_path),
            'original_size': (orig_w, orig_h)  # 使用解包后的orig_w和orig_h
        }

    def __len__(self):
        if self.train:
            # 考虑数据增强后的总样本数
            return len(self.samples) * self.augmentation.max_aug_samples
        return len(self.samples)




# 数据增强，已移除旋转增强，仅保留平移和镜像
class KeypointAugmentation:
    """关键点检测的数据增强（仅保留平移和镜像）"""

    def __init__(self, max_aug_samples=10):
        self.max_aug_samples = min(max(1, max_aug_samples), 10)  # 确保在1-10之间

    def _ensure_tensor(self, data):
        """确保数据是 torch.Tensor 类型"""
        if not isinstance(data, torch.Tensor):
            return torch.from_numpy(data).float()
        return data

    def __call__(self, image, keypoints, angles):
        """应用数据增强到图像、关键点和角度"""
        keypoints = self._ensure_tensor(keypoints)
        angles = self._ensure_tensor(angles)

        all_augmented_images = [image]
        all_augmented_keypoints = [keypoints]
        all_augmented_angles = [angles]

        # 平移增强（角度保持不变）
        translated_images, translated_keypoints = self.translate([image], [keypoints])
        translated_angles = [angles.clone() for _ in range(len(translated_images))]

        all_augmented_images.extend(translated_images)
        all_augmented_keypoints.extend(translated_keypoints)
        all_augmented_angles.extend(translated_angles)

        # 镜像增强
        mirrored_images, mirrored_keypoints, mirrored_angles = self.mirror(
            all_augmented_images,
            all_augmented_keypoints,
            all_augmented_angles
        )

        all_augmented_images.extend(mirrored_images)
        all_augmented_keypoints.extend(mirrored_keypoints)
        all_augmented_angles.extend(mirrored_angles)

        # 选择需要的样本数
        if self.max_aug_samples < len(all_augmented_images):
            indices = [0] + list(np.random.choice(
                range(1, len(all_augmented_images)),
                size=self.max_aug_samples - 1,
                replace=False
            ))
            augmented_images = [all_augmented_images[i] for i in indices]
            augmented_keypoints = [all_augmented_keypoints[i] for i in indices]
            augmented_angles = [all_augmented_angles[i] for i in indices]
        else:
            augmented_images = all_augmented_images
            augmented_keypoints = all_augmented_keypoints
            augmented_angles = all_augmented_angles

        return augmented_images, augmented_keypoints, augmented_angles

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

    def mirror(self, images, keypoints_list, angles_list):
        """水平镜像增强"""
        mirrored_images = []
        mirrored_keypoints = []
        mirrored_angles = []

        for image, keypoints, angles in zip(images, keypoints_list, angles_list):
            keypoints = self._ensure_tensor(keypoints)
            angles = self._ensure_tensor(angles)

            # 镜像图像
            mirrored_image = F.hflip(image)
            # 镜像关键点
            mirrored_kps = keypoints.clone()
            mirrored_kps[:, 0] = 1 - mirrored_kps[:, 0]
            # 镜像角度（取反）
            mirrored_angs = -angles.clone()

            mirrored_images.append(mirrored_image)
            mirrored_keypoints.append(mirrored_kps)
            mirrored_angles.append(mirrored_angs)

        return mirrored_images, mirrored_keypoints, mirrored_angles


def collate_fn(batch):
    images = []
    keypoints = []
    angles = []
    transform_params = []
    image_ids = []
    original_sizes = []

    for sample in batch:
        img = sample['image']
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        images.append(img)

        keypoints.append(sample['keypoints'].view(-1, 2))
        angles.append(sample['angles'])
        transform_params.append(sample['transform_params'])
        image_ids.append(sample['image_id'])
        original_sizes.append(sample['original_size'])

    images = torch.stack(images)

    return {
        'images': images,
        'keypoints': keypoints,
        'angles': angles,
        'transform_params': transform_params,
        'image_ids': image_ids,
        'original_sizes': original_sizes
    }