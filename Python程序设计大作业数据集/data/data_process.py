import os
import struct

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
SOURCE_DIR = os.path.join(RAW_DIR, "emnist_source_files")

DATASET_ID = "crawford/emnist"

# Minimum expected file sizes in bytes.
CHECKLIST = {
    "emnist-letters-train.csv": 10 * 1024 * 1024,
    "emnist-letters-test.csv": 2 * 1024 * 1024,
}


def verify_local_data(data_dir):
    """Check whether the expected CSV files exist and are non-trivial."""
    if not os.path.exists(data_dir):
        return False

    for filename, min_size in CHECKLIST.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"[verify] missing file: {filename}")
            return False

        actual_size = os.path.getsize(file_path)
        if actual_size < min_size:
            print(f"[verify] suspicious file size: {filename} ({actual_size} bytes)")
            return False

    return True


def download_dataset():
    """Download and unzip the EMNIST dataset through the Kaggle API."""
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

    if verify_local_data(RAW_DIR):
        print(">>> Raw CSV files already look valid.")
        return

    print(f">>> Downloading dataset from Kaggle: {DATASET_ID}")
    try:
        import kaggle

        kaggle.api.dataset_download_files(DATASET_ID, path=RAW_DIR, unzip=True)
        print(">>> Download finished.")

        if not verify_local_data(RAW_DIR):
            raise RuntimeError("downloaded files did not pass validation")
    except Exception as exc:
        print(f"Fatal error: failed to download dataset. Details: {exc}")
        raise


def fix_orientation(images):
    """Fix the built-in EMNIST rotation/mirroring issue."""
    return np.array([np.fliplr(np.rot90(img, k=-1)) for img in images])


def denoise_and_normalize(images):
    """Apply median filtering and scale pixels to [0, 1]."""
    print("Applying median filter denoising...")
    denoised = np.array([median_filter_3x3(img) for img in images])
    print("Normalizing pixel values...")
    return denoised.astype("float32") / 255.0


def median_filter_3x3(image):
    """Small NumPy-only median filter to avoid extra SciPy dependency."""
    padded = np.pad(image, pad_width=1, mode="edge")
    windows = [
        padded[row : row + image.shape[0], col : col + image.shape[1]]
        for row in range(3)
        for col in range(3)
    ]
    stacked = np.stack(windows, axis=0)
    return np.median(stacked, axis=0).astype(image.dtype)


def transform_images(images):
    """Run the same image pipeline for both CSV and IDX sources."""
    fixed = fix_orientation(images)
    return denoise_and_normalize(fixed)


def clean_and_transform_csv(filename):
    """Load EMNIST CSV data, remove invalid rows, and transform images."""
    file_path = os.path.join(RAW_DIR, filename)
    print(f"--- Processing {filename} ---")

    df = pd.read_csv(file_path, header=None)
    df = df.dropna().drop_duplicates()

    labels = df.iloc[:, 0].to_numpy(dtype=np.int64) - 1
    images = df.iloc[:, 1:].to_numpy().reshape(-1, 28, 28)

    return transform_images(images), labels


def load_idx_dataset(image_filename, label_filename):
    """Load a complete EMNIST split from the original IDX files."""
    image_path = os.path.join(SOURCE_DIR, image_filename)
    label_path = os.path.join(SOURCE_DIR, label_filename)

    with open(label_path, "rb") as label_file:
        magic, num_labels = struct.unpack(">II", label_file.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic number: {magic}")
        labels = np.frombuffer(label_file.read(), dtype=np.uint8)

    with open(image_path, "rb") as image_file:
        magic, num_images, rows, cols = struct.unpack(">IIII", image_file.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image magic number: {magic}")
        images = np.frombuffer(image_file.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    if num_images != num_labels:
        raise ValueError(f"image count {num_images} does not match label count {num_labels}")

    return images, labels.astype(np.int64) - 1


def build_test_set():
    """
    Prefer the existing CSV workflow, but automatically fall back to the
    original IDX test files if the CSV test set is missing classes.
    """
    X_test, y_test = clean_and_transform_csv("emnist-letters-test.csv")
    unique_classes = np.unique(y_test)

    if len(unique_classes) == 26:
        print(">>> CSV test split contains all 26 classes.")
        return X_test, y_test

    print(
        f">>> CSV test split only contains {len(unique_classes)} classes; "
        "rebuilding the test set from IDX files."
    )
    raw_images, raw_labels = load_idx_dataset(
        "emnist-letters-test-images-idx3-ubyte",
        "emnist-letters-test-labels-idx1-ubyte",
    )
    X_test = transform_images(raw_images)
    y_test = raw_labels
    print(f">>> Rebuilt test split with {len(np.unique(y_test))} classes.")
    return X_test, y_test


if __name__ == "__main__":
    download_dataset()

    X_train, y_train = clean_and_transform_csv("emnist-letters-train.csv")
    X_test, y_test = build_test_set()

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print("\n" + "=" * 40)
    print("Data collection and cleaning complete.")
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    print(f"Output directory: {PROCESSED_DIR}")
    print("=" * 40)
