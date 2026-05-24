import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


SEED = 42
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 10
DEFAULT_VALIDATION_SPLIT = 0.1
CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]


class NpyEMNISTDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path).astype("float32")
        self.y = np.load(y_path).astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        image = torch.from_numpy(self.x[index]).unsqueeze(0)
        label = torch.tensor(self.y[index], dtype=torch.long)
        return image, label


class EMNISTLetterCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_processed_dir():
    root = Path(__file__).resolve().parent
    matches = list(root.rglob("processed"))
    for path in matches:
        needed = {"X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"}
        if needed.issubset({item.name for item in path.iterdir() if item.is_file()}):
            return path
    raise FileNotFoundError("Could not find processed EMNIST npy files.")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def build_loaders(processed_dir, batch_size, validation_split, seed, device):
    train_full = NpyEMNISTDataset(processed_dir / "X_train.npy", processed_dir / "y_train.npy")
    test_dataset = NpyEMNISTDataset(processed_dir / "X_test.npy", processed_dir / "y_test.npy")

    val_size = int(len(train_full) * validation_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size], generator=generator)

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, len(train_full), len(test_dataset)


def run_epoch(model, data_loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def predict(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_training_curves(rows, output_dir):
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in rows], label="train")
    plt.plot(epochs, [row["val_loss"] for row in rows], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    plt.plot(epochs, [row["val_accuracy"] for row in rows], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("PyTorch CNN Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=220)
    plt.close()


def evaluate_and_save(model, test_loader, device, output_dir):
    y_true, y_prob = predict(model, test_loader, device)
    y_pred = np.argmax(y_prob, axis=1)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "top5_accuracy": float(np.mean([true in np.argsort(prob)[-5:] for true, prob in zip(y_true, y_prob)])),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "rmse_label": math.sqrt(float(np.mean((y_true - y_pred) ** 2))),
        "auc_macro_ovr": roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"),
        "auc_weighted_ovr": roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr"),
    }

    with open(output_dir / "test_metrics.txt", "w", encoding="utf-8") as file:
        for name, value in metrics.items():
            line = f"{name}: {value:.6f}"
            print(line)
            file.write(line + "\n")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
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

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, output_dir)

    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", "pred_label", "true_char", "pred_char", "confidence"])
        for true_label, pred_label, prob in zip(y_true, y_pred, y_prob):
            writer.writerow(
                [
                    int(true_label),
                    int(pred_label),
                    CLASS_NAMES[int(true_label)],
                    CLASS_NAMES[int(pred_label)],
                    f"{float(np.max(prob)):.6f}",
                ]
            )

    return metrics


def save_summary(output_dir, args, train_samples, test_samples, metrics, device):
    lines = [
        "# PyTorch CNN GPU Training Summary",
        "",
        "## Environment",
        "",
        f"- Framework: PyTorch {torch.__version__}",
        f"- Device: {device}",
        f"- CUDA available: {torch.cuda.is_available()}",
        f"- CUDA version: {torch.version.cuda}",
        f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        "",
        "## Dataset",
        "",
        "- Dataset: EMNIST Letters",
        f"- Train samples: {train_samples}",
        f"- Test samples: {test_samples}",
        "- Classes: 26 uppercase letters (A-Z)",
        "",
        "## Training Setup",
        "",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Validation split: {args.validation_split}",
        "- Optimizer: Adam",
        "- Loss: CrossEntropyLoss",
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
    ]
    (output_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train EMNIST Letters CNN with PyTorch CUDA.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "torch_outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    processed_dir = find_processed_dir()
    device = get_device()
    train_loader, val_loader, test_loader, train_samples, test_samples = build_loaders(
        processed_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=SEED,
        device=device,
    )

    model = EMNISTLetterCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    (output_dir / "model_summary.txt").write_text(str(model), encoding="utf-8")

    best_val_accuracy = 0.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f} - train_accuracy: {train_accuracy:.4f} - "
            f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), output_dir / "emnist_letters_cnn_best.pt")

    with open(output_dir / "training_log.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(model.state_dict(), output_dir / "emnist_letters_cnn_final.pt")
    plot_training_curves(rows, output_dir)
    metrics = evaluate_and_save(model, test_loader, device, output_dir)
    save_summary(output_dir, args, train_samples, test_samples, metrics, device)
    print(f"Training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
