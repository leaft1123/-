import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class EMNISTDataset(Dataset):
    """
    自定义 PyTorch 数据集，用于加载 .npy 数据并应用增强
    """
    def __init__(self, X_npy_path, y_npy_path, transform=None):
        # 加载我们之前清洗并去噪好的 .npy 文件
        self.X = np.load(X_npy_path)
        self.y = np.load(y_npy_path)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        # 归一化后的 numpy 数组需转为 0-255 的格式，才能被转换为 PIL 图像进行高级增强
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image, mode='L') # 'L' 表示灰度图

        # 如果定义了增强流水线，则应用增强
        if self.transform:
            image = self.transform(image)

        # PyTorch 的交叉熵损失函数要求 target 为 long 类型
        return image, torch.tensor(label, dtype=torch.long)

def get_data_loaders(processed_dir=None, batch_size=64):
    """
    构建包含数据增强的数据加载器
    """
    if processed_dir is None:
        # 获取当前 dataset_loader.py 所在的目录 (即 data 文件夹)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接出绝对路径: .../data/processed
        processed_dir = os.path.join(base_dir, 'processed')
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),                 # 随机旋转 ±15度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 随机上下左右平移 10%
        transforms.ToTensor(),                                 # 转换为 Tensor，并自动归一化到 0-1
    ])

    # 测试集绝不能增强！必须保持原样，以保证评估的客观性
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 实例化数据集 (使用 os.path.join 保证 Windows/Mac 路径符号绝对正确)
    train_dataset = EMNISTDataset(
        os.path.join(processed_dir, 'X_train.npy'), 
        os.path.join(processed_dir, 'y_train.npy'), 
        transform=train_transform
    )
    
    test_dataset = EMNISTDataset(
        os.path.join(processed_dir, 'X_test.npy'), 
        os.path.join(processed_dir, 'y_test.npy'), 
        transform=test_transform
    )

    # 封装为 DataLoader，方便后续按批次 (Batch) 投喂给 CNN
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 测试代码是否工作正常
if __name__ == "__main__":
    print("正在测试带有数据增强的数据加载器...")
    train_loader, test_loader = get_data_loaders()
    
    # 获取一个批次的数据看看
    images, labels = next(iter(train_loader))
    print(f"生成的批次图像形状 (Batch, Channels, H, W): {images.shape}") # 预期: [64, 1, 28, 28]
    print(f"生成的批次标签形状: {labels.shape}")                         # 预期: [64]
