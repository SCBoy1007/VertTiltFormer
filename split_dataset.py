import os
import shutil
import random

# 设置路径
original_image_dir = r"F:\RA-MED\Set Transformer\Data\TrialTrainingDataset\Xray+Json(HD)\Cleaned\Xray_Processed"
original_annotation_dir = r"F:\RA-MED\Set Transformer\Data\TrialTrainingDataset\Generate Ground Truth\Data\GeometricCenters"
output_dir = r"F:\RA-MED\Set Transformer\keypoint_detection\data"

# 创建输出文件夹结构
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "annotations"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "annotations"), exist_ok=True)

# 获取所有图片文件名（不带扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(original_image_dir) if f.endswith(".png")]  # 假设图片是PNG格式
random.shuffle(image_files)  # 随机打乱文件列表

# 按10:1的比例划分训练集和测试集
split_ratio = 10  # 10:1
train_size = int(len(image_files) * (split_ratio / (split_ratio + 1)))
train_files = image_files[:train_size]
test_files = image_files[train_size:]

# 复制训练集文件
missing_annotations = []  # 记录缺失的标注文件
for file in train_files:
    # 复制图片
    src_image = os.path.join(original_image_dir, file + ".png")  # 假设图片是PNG格式
    dst_image = os.path.join(train_dir, "images", file + ".png")
    shutil.copy(src_image, dst_image)
    
    # 复制标注
    src_annotation = os.path.join(original_annotation_dir, file + ".txt")  # 修改为TXT格式
    if os.path.exists(src_annotation):
        dst_annotation = os.path.join(train_dir, "annotations", file + ".txt")
        shutil.copy(src_annotation, dst_annotation)
    else:
        missing_annotations.append(file)  # 记录缺失的标注文件

# 复制测试集文件
for file in test_files:
    # 复制图片
    src_image = os.path.join(original_image_dir, file + ".png")  # 假设图片是PNG格式
    dst_image = os.path.join(test_dir, "images", file + ".png")
    shutil.copy(src_image, dst_image)
    
    # 复制标注
    src_annotation = os.path.join(original_annotation_dir, file + ".txt")  # 修改为TXT格式
    if os.path.exists(src_annotation):
        dst_annotation = os.path.join(test_dir, "annotations", file + ".txt")
        shutil.copy(src_annotation, dst_annotation)
    else:
        missing_annotations.append(file)  # 记录缺失的标注文件

# 输出结果
print("数据集划分完成！")
print(f"训练集大小: {len(train_files)}")
print(f"测试集大小: {len(test_files)}")
if missing_annotations:
    print(f"以下文件的标注缺失: {missing_annotations}")
else:
    print("所有文件的标注均已找到并复制。")