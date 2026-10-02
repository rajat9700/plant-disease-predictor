"""
Plant Disease Detection with Transfer Learning (MobileNetV2)
============================================================

The dataset is downloaded automatically from Kaggle via kagglehub
(abdallahalidev/plantvillage-dataset). The script locates the folder of
RGB images ("color") whose sub-folders are the class labels:

    <kagglehub cache>/.../color/
    ├── Tomato___healthy/
    ├── Tomato___Late_blight/
    └── Potato___Early_blight/ ...

To use your own local data instead, set LOCAL_DATA_DIR below
(e.g. "dataset/train") and the Kaggle download is skipped.

Requirements:
    pip install tensorflow pillow numpy matplotlib kagglehub

Kaggle credentials may be required for the first download
(see https://github.com/Kaggle/kagglehub#authenticate).

Run:
    python plant_disease_detection.py
"""

import json
import os
import random

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
IMAGE_VARIANT = "color"  # PlantVillage ships "color", "grayscale" and "segmented" versions
LOCAL_DATA_DIR = None  # e.g. os.path.join("dataset", "train") to skip the Kaggle download
IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS = 10
SEED = 123

MODEL_PATH = "best_plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"
HISTORY_PLOT_PATH = "training_history.png"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# --- "Not a leaf" rejection settings -----------------------------------
# The model is trained ONLY on leaf images, so it has no real concept of
# "not a leaf" -- it will always force its softmax output onto one of the
# known classes, sometimes with high confidence, even for a photo of a car
# or a face. Two heuristics mitigate that:
#   1. CONFIDENCE_THRESHOLD: reject predictions the model itself is unsure of.
#   2. USE_COLOR_HEURISTIC: a cheap PIL-only sanity check (no model call)
#      that the image contains vegetation-like colors before it is even
#      passed to the network.
# Neither is perfect. The robust fix is to retrain with an explicit
# "not_a_leaf" / background class built from non-plant images.
CONFIDENCE_THRESHOLD = 60.0  # percent; below this -> flagged as invalid
USE_COLOR_HEURISTIC = True
MIN_VEGETATION_RATIO = 0.15  # min fraction of green/yellow/brown pixels required
INVALID_LABEL = "Invalid (No leaf detected)"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Data pipeline
# ---------------------------------------------------------------------------
def _looks_like_class_root(path: str) -> bool:
    """True if `path` has >= 2 sub-folders that directly contain image files."""
    try:
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
    except OSError:
        return False

    folders_with_images = 0
    for sub in subdirs:
        if any(f.lower().endswith(VALID_EXTENSIONS) for f in os.listdir(sub)):
            folders_with_images += 1
    return folders_with_images >= 2


def find_class_root(base_path: str, preferred_name: str = IMAGE_VARIANT) -> str:
    """
    Search the downloaded Kaggle folder for the directory whose sub-folders are
    the class labels. Prefers a folder named `preferred_name` (e.g. "color"),
    otherwise falls back to the first directory that looks like a class root.
    """
    # Pass 1: folder explicitly named like the preferred image variant
    for root, dirs, _ in os.walk(base_path):
        dirs.sort()
        for d in dirs:
            candidate = os.path.join(root, d)
            if d.lower() == preferred_name.lower() and _looks_like_class_root(candidate):
                return candidate

    # Pass 2: any directory that looks like a class root
    for root, dirs, _ in os.walk(base_path):
        dirs.sort()
        if _looks_like_class_root(root):
            return root

    raise FileNotFoundError(
        f"Could not find class sub-folders with images inside '{base_path}'."
    )


def resolve_data_dir() -> str:
    """Return the directory to feed into image_dataset_from_directory."""
    if LOCAL_DATA_DIR:
        print(f"Using local dataset directory: {LOCAL_DATA_DIR}")
        return LOCAL_DATA_DIR

    # kagglehub caches downloads, so re-running the script won't download again
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Path to dataset files: {download_path}")

    data_dir = find_class_root(download_path)
    print(f"Using image directory: {data_dir}")
    return data_dir


def inspect_dataset_with_pil(data_dir: str, samples_per_class: int = 2) -> None:
    """Use PIL to count images per class and print the shape of a few samples."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. "
            "Expected structure: <data_dir>/<class_name>/<images>"
        )

    class_dirs = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    if not class_dirs:
        raise ValueError(f"No class sub-folders found inside '{data_dir}'.")

    print("=" * 70)
    print(f"Dataset inspection: {data_dir}")
    print("=" * 70)

    total_images = 0
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        files = [
            f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXTENSIONS)
        ]
        total_images += len(files)
        print(f"\nClass '{class_name}': {len(files)} images")

        for fname in files[:samples_per_class]:
            fpath = os.path.join(class_path, fname)
            try:
                with Image.open(fpath) as img:
                    width, height = img.size
                    channels = len(img.getbands())
                    print(
                        f"   - {fname}: (H={height}, W={width}, C={channels}), "
                        f"mode={img.mode}"
                    )
            except Exception as exc:  # corrupted file, etc.
                print(f"   - {fname}: could not be opened ({exc})")

    print(f"\nTotal classes: {len(class_dirs)} | Total images: {total_images}")
    print("=" * 70)


def build_augmentation_layers() -> tf.keras.Sequential:
    """Data augmentation: random flips and rotations (active only when training=True)."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),  # +/- 20% of a full circle (i.e. +/- 72 degrees)
        ],
        name="data_augmentation",
    )


def load_datasets(data_dir: str):
    """
    Load train/validation datasets from `data_dir` (80/20 split), then apply
    augmentation, MobileNetV2 preprocessing, caching and prefetching.

    Returns:
        train_ds, val_ds, class_names
    """
    common_args = dict(
        directory=data_dir,
        labels="inferred",
        label_mode="int",  # integer labels -> SparseCategoricalCrossentropy
        validation_split=VALIDATION_SPLIT,
        seed=SEED,  # identical seed guarantees non-overlapping subsets
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common_args)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common_args)

    # Must be read BEFORE any .map()/.cache()/.prefetch() calls
    class_names = train_ds.class_names
    print(f"\nDetected {len(class_names)} classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    augmentation = build_augmentation_layers()

    # Training: cache raw images -> shuffle -> augment (fresh each epoch) -> preprocess.
    # Caching BEFORE augmentation keeps the augmentations random every epoch.
    # NOTE: .cache() holds the decoded images in RAM. For very large datasets,
    # use .cache("/path/to/cache_file") to spill the cache to disk instead.
    train_ds = (
        train_ds.cache()
        .shuffle(buffer_size=1000, seed=SEED)
        .map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=autotune,
        )
        .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune)
        .prefetch(buffer_size=autotune)
    )

    # Validation: no augmentation, only preprocessing.
    val_ds = (
        val_ds.cache()
        .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune)
        .prefetch(buffer_size=autotune)
    )

    return train_ds, val_ds, class_names


# ---------------------------------------------------------------------------
# 2. Model architecture
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> tf.keras.Model:
    """MobileNetV2 (frozen, ImageNet weights) + custom classification head."""
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze all pre-trained weights

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_image")
    # training=False keeps BatchNorm layers in inference mode (important for frozen bases)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.2, name="dropout_0_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="plant_disease_mobilenetv2")
    return model


# ---------------------------------------------------------------------------
# 3. Training & evaluation
# ---------------------------------------------------------------------------
def compile_and_train(model, train_ds, val_ds):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Sanity check: ModelCheckpoint only writes a file when val_loss improves.
    # If val_loss is NaN/inf (or missing), no file is ever written.
    val_losses = history.history.get("val_loss", [])
    if not val_losses or any(not np.isfinite(v) for v in val_losses):
        print(
            "WARNING: val_loss contains NaN/inf values: "
            f"{val_losses}. Check your data (corrupted images?) and learning rate."
        )

    # Safety net: EarlyStopping(restore_best_weights=True) has already put the
    # best weights back into `model`, so saving it here is equivalent to the checkpoint.
    if not os.path.isfile(MODEL_PATH):
        print(
            f"WARNING: ModelCheckpoint did not create '{os.path.abspath(MODEL_PATH)}'. "
            "Saving the current model instead."
        )
        model.save(MODEL_PATH)

    return history


def plot_training_history(history, save_path: str = HISTORY_PLOT_PATH) -> None:
    """Plot training/validation accuracy and loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)  # early stopping may end before EPOCHS

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, marker="o", label="Training Accuracy")
    plt.plot(epochs_range, val_acc, marker="o", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, marker="o", label="Training Loss")
    plt.plot(epochs_range, val_loss, marker="o", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to '{save_path}'")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------
def _vegetation_ratio(img: Image.Image) -> float:
    """
    Fraction of pixels whose hue falls in the green/yellow/brown range typical
    of healthy or diseased leaves (PIL only, no model). Used as a cheap
    pre-filter, not a classifier -- see the note above CONFIDENCE_THRESHOLD.
    """
    h, s, v = img.convert("HSV").split()
    h_deg = np.array(h, dtype=np.float32) * (360.0 / 255.0)  # PIL hue is 0-255
    s_frac = np.array(s, dtype=np.float32) / 255.0
    v_frac = np.array(v, dtype=np.float32) / 255.0

    # ~20-160 degrees covers yellow -> green -> olive/brown; low saturation/value
    # pixels (gray, near-black/white backgrounds, paper, skin highlights) excluded.
    vegetation_mask = (h_deg >= 20) & (h_deg <= 160) & (s_frac >= 0.15) & (v_frac >= 0.10)
    return float(np.mean(vegetation_mask))


def predict_plant_disease(image_path: str, model: tf.keras.Model, class_names: list):
    """
    Predict the disease class of a single leaf image.

    Steps: PIL load -> RGB -> (optional vegetation-color sanity check) ->
    resize to 224x224 -> MobileNetV2 preprocessing (scale pixels to [-1, 1])
    -> model.predict -> print class + confidence.

    If the image doesn't look like vegetation, or the model's own confidence
    is below CONFIDENCE_THRESHOLD, the function reports INVALID_LABEL instead
    of forcing a guess.

    Returns:
        (predicted_class_name_or_INVALID_LABEL, confidence_percentage)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    # Load with PIL; convert() guarantees 3 channels (handles RGBA / grayscale / palette)
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        if USE_COLOR_HEURISTIC:
            ratio = _vegetation_ratio(img)
            if ratio < MIN_VEGETATION_RATIO:
                print(f"Image      : {image_path}")
                print(f"Prediction : {INVALID_LABEL}")
                print(f"Reason     : only {ratio * 100:.1f}% vegetation-like pixels "
                      f"(need >= {MIN_VEGETATION_RATIO * 100:.0f}%)")
                return INVALID_LABEL, 0.0

        img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32)  # (224, 224, 3), range 0-255

    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing -> [-1, 1]

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100.0

    print(f"Image      : {image_path}")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"Prediction : {INVALID_LABEL}")
        print(f"Reason     : low model confidence ({confidence:.2f}% < "
              f"{CONFIDENCE_THRESHOLD:.0f}%), closest guess was '{predicted_class}'")
        return INVALID_LABEL, confidence

    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")

    return predicted_class, confidence


def find_sample_image(data_dir: str):
    """Return the path of a random image from the dataset (used for the demo prediction)."""
    candidates = []
    for root, _, files in os.walk(data_dir):
        candidates.extend(
            os.path.join(root, f) for f in files if f.lower().endswith(VALID_EXTENSIONS)
        )
    return random.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available    : {tf.config.list_physical_devices('GPU')}")

    # 1. Data (downloads from Kaggle on first run, then reuses the cached copy)
    data_dir = resolve_data_dir()
    inspect_dataset_with_pil(data_dir)
    train_ds, val_ds, class_names = load_datasets(data_dir)

    # Persist class names so the model can be used later without the dataset folder
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to '{CLASS_NAMES_PATH}'")

    # 2. Model
    model = build_model(num_classes=len(class_names))

    # 3. Train + evaluate
    history = compile_and_train(model, train_ds, val_ds)
    plot_training_history(history)

    # Load the best checkpoint (lowest val_loss) for final evaluation and inference
    best_model = tf.keras.models.load_model(MODEL_PATH)
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
    print(f"\nBest model -> val_loss: {val_loss:.4f} | val_accuracy: {val_acc:.4f}")

    # 4. Demo inference on a random dataset image
    sample_path = find_sample_image(data_dir)
    if sample_path:
        print("\nDemo prediction:")
        predict_plant_disease(sample_path, best_model, class_names)


if __name__ == "__main__":
    main()