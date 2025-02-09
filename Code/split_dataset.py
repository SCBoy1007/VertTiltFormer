import os
import shutil
import random

# 设置路径
original_image_dir = r"I:\RA-MED\VertTiltFormer\Data\TrialTrainingDataset\Xray+Json(HD)\Cleaned\Xray_Cleaned"
geometric_centers_dir = r"I:\RA-MED\VertTiltFormer\Data\TrialTrainingDataset\Generate Ground Truth\Data\GeometricCenters"
centerline_angle_dir = r"I:\RA-MED\VertTiltFormer\Data\TrialTrainingDataset\Generate Ground Truth\Data\CenterLine"
output_dir = r"I:\RA-MED\VertTiltFormer\keypoint_detection\Data\dataset"

# 创建输出文件夹结构
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# 训练集子文件夹
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "keypoints"), exist_ok=True)  # 关键点标注
os.makedirs(os.path.join(train_dir, "centerline_angles"), exist_ok=True)  # 中心线角度

# 测试集子文件夹
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "keypoints"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "centerline_angles"), exist_ok=True)

# 获取所有图片文件名（不带扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(original_image_dir) if f.endswith(".png")]
random.shuffle(image_files)

# 按10:1的比例划分训练集和测试集
split_ratio = 10
train_size = int(len(image_files) * (split_ratio / (split_ratio + 1)))
train_files = image_files[:train_size]
test_files = image_files[train_size:]


def copy_data(file_list, target_dir, is_train=True):
    missing_files = []
    for file in file_list:
        # 复制图片
        src_image = os.path.join(original_image_dir, file + ".png")
        dst_image = os.path.join(target_dir, "images", file + ".png")
        if os.path.exists(src_image):
            shutil.copy(src_image, dst_image)
        else:
            missing_files.append(("image", file))

        # 复制关键点标注
        src_keypoints = os.path.join(geometric_centers_dir, file + ".txt")
        dst_keypoints = os.path.join(target_dir, "keypoints", file + ".txt")
        if os.path.exists(src_keypoints):
            shutil.copy(src_keypoints, dst_keypoints)
        else:
            missing_files.append(("keypoints", file))

        # 复制中心线角度标注
        src_centerline = os.path.join(centerline_angle_dir, file + ".txt")
        dst_centerline = os.path.join(target_dir, "centerline_angles", file + ".txt")
        if os.path.exists(src_centerline):
            shutil.copy(src_centerline, dst_centerline)
        else:
            missing_files.append(("centerline_angles", file))

    return missing_files


# 复制训练集和测试集
print("正在复制训练集...")
train_missing = copy_data(train_files, train_dir, True)
print("正在复制测试集...")
test_missing = copy_data(test_files, test_dir, False)

# 输出结果
print("\n数据集划分完成！")
print(f"训练集大小: {len(train_files)}")
print(f"测试集大小: {len(test_files)}")

if train_missing:
    print("\n训练集中缺失的文件:")
    for file_type, file_name in train_missing:
        print(f"{file_type}: {file_name}")

if test_missing:
    print("\n测试集中缺失的文件:")
    for file_type, file_name in test_missing:
        print(f"{file_type}: {file_name}")

if not train_missing and not test_missing:
    print("\n所有文件的标注均已找到并复制。")