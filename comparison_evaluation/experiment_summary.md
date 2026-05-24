# Traditional Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 | Macro AUC | RMSE | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pytorch_cnn | 0.937885 | 0.937361 | 0.937361 | 0.998301 | 1.979729 | 0.00 |
| sgd_logistic_regression | 0.665096 | 0.667449 | 0.667449 | 0.879965 | 5.809723 | 12.21 |

The CNN uses local image structure through convolution, while the traditional baselines classify flattened 28x28 pixels. Higher CNN scores indicate that spatial feature learning is more suitable for EMNIST letter recognition.
