# AIhomework

本项目为 Python 人工智能程序设计实践课程项目，主题是基于 EMNIST Letters 数据集的手写英文字母识别。项目包含 CNN 深度学习模型、传统机器学习对比模型，以及二者的评估结果对比。

## 环境依赖

建议使用 Python 3.10 及以上版本。

安装依赖：

```powershell
pip install -r requirements.txt
```

主要依赖包括：

- `numpy`：数据读取与数组处理
- `pandas`：表格数据处理
- `matplotlib`：训练曲线、混淆矩阵和对比图绘制
- `scikit-learn`：传统模型训练与评估指标计算
- `torch`：PyTorch CNN 模型训练
- `tensorflow`：Keras CNN 模型训练

如只运行传统模型，可不安装 `torch` 和 `tensorflow`；如只查看已有输出结果，则无需重新训练模型。

## 项目结构

```text
AIhomework/
├── README.md
├── requirements.txt
├── cnn_deep_learning/
│   ├── code/
│   │   ├── train_emnist_letters_cnn.py
│   │   └── train_emnist_letters_torch.py
│   └── outputs/
│       ├── keras_cnn_outputs/
│       └── pytorch_cnn_outputs/
├── traditional_model/
│   ├── code/
│   │   └── train_emnist_letters_traditional.py
│   └── outputs/
│       └── sgd_logistic_regression/
├── comparison_evaluation/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── experiment_summary.md
│   └── 指标说明.md
├── dataset/
│   ├── data_process.py
│   ├── dataset_loader.py
│   ├── raw/
│   └── processed/
└── documents_and_reference/
    ├── task_and_reports/
    ├── reference_project/
    └── midterm_images/
```

各目录说明：

- `cnn_deep_learning/`：CNN 深度学习模型代码和输出结果。
- `traditional_model/`：传统机器学习模型代码和输出结果。
- `comparison_evaluation/`：CNN 与传统模型的对比结果、对比图和指标说明。
- `dataset/`：数据处理代码、原始数据和处理后的 `.npy` 数据文件。
- `documents_and_reference/`：任务书、报告、参考项目和中期图片等资料。

## 运行方式

运行 PyTorch CNN 模型：

```powershell
python cnn_deep_learning/code/train_emnist_letters_torch.py
```

运行 TensorFlow/Keras CNN 模型：

```powershell
python cnn_deep_learning/code/train_emnist_letters_cnn.py
```

运行传统机器学习对比模型：

```powershell
python traditional_model/code/train_emnist_letters_traditional.py
```

说明：模型训练脚本需要使用 `dataset/processed/` 中的 `X_train.npy`、`y_train.npy`、`X_test.npy`、`y_test.npy`。如果本地没有处理后的数据，需要先根据 `dataset/data_process.py` 生成处理数据。

## 当前主要结果

当前主要对比结果如下：

| 模型 | Accuracy | Macro F1 | Macro AUC | RMSE |
| --- | ---: | ---: | ---: | ---: |
| PyTorch CNN | 0.937885 | 0.937361 | 0.998301 | 1.979729 |
| 传统 Logistic Regression | 0.665096 | 0.667449 | 0.879965 | 5.809723 |

详细结果见：

- `comparison_evaluation/model_comparison.csv`
- `comparison_evaluation/model_comparison.png`
- `comparison_evaluation/experiment_summary.md`
- `comparison_evaluation/指标说明.md`

## 结果说明

CNN 模型利用卷积结构提取图像空间特征，传统模型将 28x28 图像展平成向量后进行分类。实验结果表明，CNN 在手写字母识别任务上的准确率和综合评价指标明显优于传统线性模型。
