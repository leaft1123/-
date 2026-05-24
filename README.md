# AIhomework

本项目为 Python 人工智能程序设计实践课程项目，主题是基于 EMNIST Letters 数据集的手写英文字母识别。

## 项目内容

项目主要完成以下内容：

- 使用 CNN 深度学习模型识别 26 类手写英文字母。
- 使用传统机器学习模型作为对比实验。
- 对不同模型的 Accuracy、F1-score、ROC-AUC、RMSE 等指标进行评估。
- 输出训练日志、分类报告、混淆矩阵和模型对比结果。

## 文件夹结构

- `cnn_deep_learning/`
  - `code/`：CNN 深度学习模型训练代码。
  - `outputs/keras_cnn_outputs/`：TensorFlow/Keras CNN 模型输出结果。
  - `outputs/pytorch_cnn_outputs/`：PyTorch CNN 模型输出结果，也是当前主要深度学习结果。
- `traditional_model/`
  - `code/`：传统机器学习模型代码。
  - `outputs/`：传统模型的评估结果和分类报告。
- `comparison_evaluation/`
  - CNN 模型与传统模型的对比表、对比图和结果说明。
  - `指标说明.md`：各项评估指标的中文解释。
- `dataset/`
  - EMNIST 原始数据和处理后的 `.npy` 数据文件。
- `documents_and_reference/`
  - 任务书、选题报告、中期报告、参考项目和中期处理图片等资料。

## 当前主要结果

当前主要对比结果如下：

- PyTorch CNN 准确率：`0.937885`
- 传统 Logistic Regression 准确率：`0.665096`

详细结果见：

- `comparison_evaluation/model_comparison.csv`
- `comparison_evaluation/experiment_summary.md`
- `comparison_evaluation/model_comparison.png`
- `comparison_evaluation/指标说明.md`

## 运行方式

运行 PyTorch CNN 模型：

```powershell
python cnn_deep_learning/code/train_emnist_letters_torch.py
```

运行传统机器学习对比模型：

```powershell
python traditional_model/code/train_emnist_letters_traditional.py
```

脚本会自动在 `dataset/` 文件夹下查找处理后的 EMNIST `.npy` 数据文件。

## 说明

本项目中 CNN 模型利用卷积结构提取图像空间特征，传统模型则将 28x28 图像展平成向量后进行分类。实验结果表明，CNN 在手写字母识别任务上明显优于传统线性模型。
