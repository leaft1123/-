1. 模块概述
本项目使用 EMNIST Letters 数据集（26类字母，约10万样本）。本模块已完成从原始数据下载到深度学习可用格式（.npy）的全流程处理，包含以下工程优化：

自动化采集：支持Kaggle API 一键下载与校验。

数据清洗：剔除重复项及缺失值。

图像纠偏：修正了EMNIST 原始数据中的镜像翻转与90度旋转问题。

图像去噪：采用中值滤波(Median Filter)消除笔画噪点。

数据增强：提供动态旋转、平移接口，增强模型泛化能力。

2.目录结构
project_root/
├── data/
│   ├── raw/               # 原始 CSV 文件 (存放从 Kaggle 下载的原始数据)
│   ├── processed/         # 清洗后的 .npy 文件 (直接用于训练)
│   │   ├── X_train.npy    # 训练集图像 (88800, 28, 28)
│   │   ├── y_train.npy    # 训练集标签 (0-25)
│   │   ├── X_test.npy     # 测试集图像 (14800, 28, 28)
│   │   └── y_test.npy     # 测试集标签 (0-25)
│   ├── data_process.py    # 执行下载、清洗、去噪、保存的主脚本
│   └── dataset_loader.py  #  PyTorch 带有数据增强功能的数据加载器

3.环境准备
pip install pandas numpy torch torchvision scipy kaggle
将kaggle.json放置在用户目录的.kaggle文件夹下（{"username":"kaggle用户名","key":"生成的API Tokens"}）

4.数据特征说明：

尺寸：28 × 28 像素（已归一化至 0.0 - 1.0）。

标签：0-25（对应 A-Z），已完成 label - 1 的偏移处理。

增强策略：训练集开启了随机旋转（±15°）和平移（10%），测试集保持原始分布。