import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
VALIDATION_SPLIT = 0.1
CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]

tf.random.set_seed(SEED)
np.random.seed(SEED)


def find_processed_dir():
    root = Path(__file__).resolve().parent
    matches = list(root.rglob("processed"))
    for path in matches:
        needed = {"X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"}
        if needed.issubset({item.name for item in path.iterdir() if item.is_file()}):
            return path
    raise FileNotFoundError("Could not find a processed directory with EMNIST npy files.")


def load_data(processed_dir):
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_test, y_test


def build_cnn_model(num_classes, input_shape=(28, 28, 1)):
    """Reuse the reference project's CNN structure as closely as practical."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_datasets(X_train, y_train, X_test, y_test, batch_size, validation_split):
    val_size = int(len(X_train) * validation_split)
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train_main, y_train_main = X_train[val_size:], y_train[val_size:]

    augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.1, 0.1),
        ],
        name="augmentation",
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_main, y_train_main))
        .shuffle(buffer_size=len(X_train_main), seed=SEED)
        .batch(batch_size)
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


def plot_history(history, output_dir):
    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], label="train")
    plt.plot(epochs, history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], label="train")
    plt.plot(epochs, history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


def save_model_summary(model, output_dir):
    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))


def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=220)
    plt.close()


def save_predictions(y_true, y_pred, y_prob, output_dir):
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", "pred_label", "true_char", "pred_char", "confidence"])
        confidences = np.max(y_prob, axis=1)
        for true_label, pred_label, confidence in zip(y_true, y_pred, confidences):
            writer.writerow(
                [
                    int(true_label),
                    int(pred_label),
                    CLASS_NAMES[int(true_label)],
                    CLASS_NAMES[int(pred_label)],
                    f"{float(confidence):.6f}",
                ]
            )


def evaluate_model(model, test_ds, y_test, output_dir):
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    rmse = math.sqrt(float(np.mean((y_test - y_pred) ** 2)))

    y_test_bin = label_binarize(y_test, classes=np.arange(len(CLASS_NAMES)))
    try:
        auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_test_bin, y_prob, average="weighted", multi_class="ovr")
    except ValueError:
        auc_macro = float("nan")
        auc_weighted = float("nan")

    top5_accuracy = float(np.mean([true in np.argsort(prob)[-5:] for true, prob in zip(y_test, y_prob)]))
    report_text = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(CLASS_NAMES)))

    metrics = {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "rmse_label": rmse,
        "auc_macro_ovr": auc_macro,
        "auc_weighted_ovr": auc_weighted,
    }

    with open(output_dir / "test_metrics.txt", "w", encoding="utf-8") as file:
        for name, value in metrics.items():
            line = f"{name}: {value:.6f}"
            print(line)
            file.write(line + "\n")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report_text)

    with open(output_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, values in report_dict.items():
            if isinstance(values, dict):
                writer.writerow(
                    [
                        label,
                        values.get("precision", ""),
                        values.get("recall", ""),
                        values.get("f1-score", ""),
                        values.get("support", ""),
                    ]
                )

    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, output_dir)
    save_predictions(y_test, y_pred, y_prob, output_dir)

    return metrics


def save_experiment_summary(output_dir, args, dataset_info, metrics):
    summary_path = output_dir / "experiment_summary.md"
    lines = [
        "# CNN Model Training and Evaluation Summary",
        "",
        "## Dataset",
        "",
        f"- Dataset: EMNIST Letters",
        f"- Train samples: {dataset_info['train_samples']}",
        f"- Test samples: {dataset_info['test_samples']}",
        f"- Image size: 28 x 28 grayscale",
        f"- Classes: 26 uppercase letters (A-Z)",
        "",
        "## Training Setup",
        "",
        f"- Model: CNN based on the reference HandWrittenTextRecognition architecture",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Validation split: {args.validation_split}",
        f"- Optimizer: Adam",
        f"- Loss: sparse categorical crossentropy",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Top-5 Accuracy: {metrics['top5_accuracy']:.6f}",
        f"- Macro Precision: {metrics['precision_macro']:.6f}",
        f"- Macro Recall: {metrics['recall_macro']:.6f}",
        f"- Macro F1-score: {metrics['f1_macro']:.6f}",
        f"- Weighted F1-score: {metrics['f1_weighted']:.6f}",
        f"- RMSE (label id): {metrics['rmse_label']:.6f}",
        f"- Macro ROC-AUC (OvR): {metrics['auc_macro_ovr']:.6f}",
        f"- Weighted ROC-AUC (OvR): {metrics['auc_weighted_ovr']:.6f}",
        "",
        "## Generated Files",
        "",
        "- `training_log.csv`: epoch-level training log",
        "- `training_curves.png`: loss and accuracy curves",
        "- `test_metrics.txt`: final evaluation metrics",
        "- `classification_report.txt`: per-class precision, recall, and F1",
        "- `classification_report.csv`: tabular classification report",
        "- `confusion_matrix.png`: confusion matrix visualization",
        "- `confusion_matrix.csv`: raw confusion matrix values",
        "- `predictions.csv`: test-set predictions and confidence values",
        "- `model_summary.txt`: CNN architecture summary",
        "- `emnist_letters_cnn_best.keras`: best validation model",
        "- `emnist_letters_cnn_final.keras`: final saved model",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN on EMNIST Letters.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=VALIDATION_SPLIT,
        help="Fraction of training data used for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cnn_outputs",
        help="Directory for logs, figures, metrics, and model files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed_dir = find_processed_dir()
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    print(f"Using processed data from: {processed_dir}")
    X_train, y_train, X_test, y_test = load_data(processed_dir)
    num_classes = len(np.unique(y_train))
    dataset_info = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    train_ds, val_ds, test_ds = build_datasets(
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )

    model = build_cnn_model(num_classes=num_classes)
    save_model_summary(model, output_dir)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_dir / "emnist_letters_cnn_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(output_dir / "training_log.csv"),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history, output_dir)
    metrics = evaluate_model(model, test_ds, y_test, output_dir)
    save_experiment_summary(output_dir, args, dataset_info, metrics)
    model.save(output_dir / "emnist_letters_cnn_final.keras")
    print(f"Training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
