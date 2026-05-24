import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


SEED = 42
CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]


def find_processed_dir():
    project_root = Path(__file__).resolve().parents[2]
    search_roots = [project_root / "dataset", project_root]
    matches = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob("processed"))
    for path in matches:
        needed = {"X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"}
        if needed.issubset({item.name for item in path.iterdir() if item.is_file()}):
            return path
    raise FileNotFoundError("Could not find processed EMNIST npy files.")


def load_flattened_data(processed_dir):
    x_train = np.load(processed_dir / "X_train.npy").astype("float32")
    y_train = np.load(processed_dir / "y_train.npy").astype("int64")
    x_test = np.load(processed_dir / "X_test.npy").astype("float32")
    y_test = np.load(processed_dir / "y_test.npy").astype("int64")

    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    return x_train, y_train, x_test, y_test


def limit_training_data(x_train, y_train, max_train_samples):
    if max_train_samples is None or max_train_samples >= len(y_train):
        return x_train, y_train

    rng = np.random.default_rng(SEED)
    selected = []
    classes = np.unique(y_train)
    per_class = max(1, max_train_samples // len(classes))
    for label in classes:
        label_indices = np.flatnonzero(y_train == label)
        take = min(per_class, len(label_indices))
        selected.extend(rng.choice(label_indices, size=take, replace=False))

    selected = np.array(selected)
    rng.shuffle(selected)
    return x_train[selected], y_train[selected]


def build_models():
    return {
        "sgd_logistic_regression": make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=0.0001,
                max_iter=40,
                tol=1e-3,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
    }


def get_scores(model, x_test):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)
        if np.all(np.isfinite(probs)):
            return probs
    return None


def evaluate_model(name, model, x_test, y_test, output_dir):
    y_pred = model.predict(x_test)
    y_score = get_scores(model, x_test)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "rmse_label": math.sqrt(float(np.mean((y_test - y_pred) ** 2))),
    }

    if y_score is not None:
        y_test_bin = label_binarize(y_test, classes=np.arange(len(CLASS_NAMES)))
        metrics["top5_accuracy"] = float(
            np.mean([true in np.argsort(score)[-5:] for true, score in zip(y_test, y_score)])
        )
        metrics["auc_macro_ovr"] = roc_auc_score(
            y_test_bin, y_score, average="macro", multi_class="ovr"
        )
        metrics["auc_weighted_ovr"] = roc_auc_score(
            y_test_bin, y_score, average="weighted", multi_class="ovr"
        )
    else:
        metrics["top5_accuracy"] = np.nan
        metrics["auc_macro_ovr"] = np.nan
        metrics["auc_weighted_ovr"] = np.nan

    model_dir = output_dir / name
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / "test_metrics.txt", "w", encoding="utf-8") as file:
        for metric_name, value in metrics.items():
            file.write(f"{metric_name}: {value:.6f}\n")

    report_text = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (model_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    with open(model_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as file:
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

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(CLASS_NAMES)))
    np.savetxt(model_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    return metrics


def read_cnn_metrics(cnn_metrics_path):
    if not cnn_metrics_path.exists():
        return None
    metrics = {}
    for line in cnn_metrics_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        metrics[name.strip()] = float(value.strip())
    return metrics


def save_comparison(rows, output_dir):
    metric_names = [
        "accuracy",
        "top5_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "rmse_label",
        "auc_macro_ovr",
        "auc_weighted_ovr",
        "fit_seconds",
    ]

    with open(output_dir / "model_comparison.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Traditional Model Comparison",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Macro AUC | RMSE | Fit seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {accuracy:.6f} | {f1_macro:.6f} | {f1_weighted:.6f} | "
            "{auc_macro_ovr:.6f} | {rmse_label:.6f} | {fit_seconds:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The CNN uses local image structure through convolution, while the traditional "
            "baselines classify flattened 28x28 pixels. Higher CNN scores indicate that "
            "spatial feature learning is more suitable for EMNIST letter recognition.",
            "",
        ]
    )
    (output_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")

    plot_rows = [row for row in rows if not row["model"].lower().startswith("pytorch")]
    if any(row["model"].lower().startswith("pytorch") for row in rows):
        plot_rows = rows
    names = [row["model"] for row in plot_rows]
    accuracy = [row["accuracy"] for row in plot_rows]
    f1_macro = [row["f1_macro"] for row in plot_rows]

    x = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, f1_macro, width, label="Macro F1")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("CNN vs Traditional Models on EMNIST Letters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train traditional Scikit-learn baselines for EMNIST Letters."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional stratified training subset size for quick experiments.",
    )
    parser.add_argument(
        "--cnn-metrics",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "cnn_deep_learning"
        / "outputs"
        / "pytorch_cnn_outputs"
        / "test_metrics.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    processed_dir = find_processed_dir()
    x_train, y_train, x_test, y_test = load_flattened_data(processed_dir)
    x_train, y_train = limit_training_data(x_train, y_train, args.max_train_samples)

    rows = []
    cnn_metrics = read_cnn_metrics(args.cnn_metrics)
    if cnn_metrics:
        cnn_row = {"model": "pytorch_cnn", "fit_seconds": 0.0}
        cnn_row.update(cnn_metrics)
        rows.append(cnn_row)

    for name, model in build_models().items():
        print(f"Training {name}...")
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        metrics = evaluate_model(name, model, x_test, y_test, output_dir)
        row = {"model": name, "fit_seconds": fit_seconds}
        row.update(metrics)
        rows.append(row)
        print(f"{name} accuracy: {metrics['accuracy']:.6f}, macro F1: {metrics['f1_macro']:.6f}")

    save_comparison(rows, output_dir)
    print(f"Traditional model comparison saved to: {output_dir}")


if __name__ == "__main__":
    main()
