# PyTorch CNN GPU Training Summary

## Environment

- Framework: PyTorch 2.11.0+cu128
- Device: cuda
- CUDA available: True
- CUDA version: 12.8
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU

## Dataset

- Dataset: EMNIST Letters
- Train samples: 88798
- Test samples: 20800
- Classes: 26 uppercase letters (A-Z)

## Training Setup

- Epochs: 10
- Batch size: 512
- Validation split: 0.1
- Optimizer: Adam
- Loss: CrossEntropyLoss

## Test Metrics

- Accuracy: 0.937885
- Top-5 Accuracy: 0.996635
- Macro Precision: 0.939787
- Macro Recall: 0.937885
- Macro F1-score: 0.937361
- Weighted F1-score: 0.937361
- RMSE (label id): 1.979729
- Macro ROC-AUC (OvR): 0.998301
- Weighted ROC-AUC (OvR): 0.998301
