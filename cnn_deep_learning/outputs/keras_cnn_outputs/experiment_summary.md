# CNN Model Training and Evaluation Summary

## Dataset

- Dataset: EMNIST Letters
- Train samples: 88798
- Test samples: 20800
- Image size: 28 x 28 grayscale
- Classes: 26 uppercase letters (A-Z)

## Training Setup

- Model: CNN based on the reference HandWrittenTextRecognition architecture
- Epochs: 10
- Batch size: 128
- Validation split: 0.1
- Optimizer: Adam
- Loss: sparse categorical crossentropy

## Test Metrics

- Accuracy: 0.927260
- Top-5 Accuracy: 0.995048
- Macro Precision: 0.930406
- Macro Recall: 0.927260
- Macro F1-score: 0.926688
- Weighted F1-score: 0.926688
- RMSE (label id): 2.241919
- Macro ROC-AUC (OvR): 0.998066
- Weighted ROC-AUC (OvR): 0.998066

## Generated Files

- `training_log.csv`: epoch-level training log
- `training_curves.png`: loss and accuracy curves
- `test_metrics.txt`: final evaluation metrics
- `classification_report.txt`: per-class precision, recall, and F1
- `classification_report.csv`: tabular classification report
- `confusion_matrix.png`: confusion matrix visualization
- `confusion_matrix.csv`: raw confusion matrix values
- `predictions.csv`: test-set predictions and confidence values
- `model_summary.txt`: CNN architecture summary
- `emnist_letters_cnn_best.keras`: best validation model
- `emnist_letters_cnn_final.keras`: final saved model
