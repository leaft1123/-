from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
VALIDATION_SPLIT = 0.1

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


def build_datasets(X_train, y_train, X_test, y_test):
    val_size = int(len(X_train) * VALIDATION_SPLIT)
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
        .batch(BATCH_SIZE)
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
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


def main():
    processed_dir = find_processed_dir()
    output_dir = Path(__file__).resolve().parent / "cnn_outputs"
    output_dir.mkdir(exist_ok=True)

    print(f"Using processed data from: {processed_dir}")
    X_train, y_train, X_test, y_test = load_data(processed_dir)
    num_classes = len(np.unique(y_train))

    train_ds, val_ds, test_ds = build_datasets(X_train, y_train, X_test, y_test)

    model = build_cnn_model(num_classes=num_classes)
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
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    test_metrics = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names

    with open(output_dir / "test_metrics.txt", "w", encoding="utf-8") as file:
        for name, value in zip(metric_names, test_metrics):
            line = f"{name}: {value:.6f}"
            print(line)
            file.write(line + "\n")

    plot_history(history, output_dir)
    model.save(output_dir / "emnist_letters_cnn_final.keras")
    print(f"Training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
