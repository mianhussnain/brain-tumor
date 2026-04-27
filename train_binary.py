"""
Binary Tumor Detection Training Script
Trains an EfficientNetB4 model to classify brain MRI images as:
    0 → No Tumor
    1 → Tumor

Dataset: data/yes-no_dataset/yes  (155 images)
         data/yes-no_dataset/no   (98 images)

Usage:
    python train_binary.py
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import config

# ─── Reproducibility ─────────────────────────────────────────────────────────
tf.random.set_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_binary_data():
    """Load yes/no images and return (X, y) arrays."""
    print("\n[1/5] Loading binary dataset...")

    X, y = [], []

    class_dirs = {
        "no":  0,   # No Tumor
        "yes": 1,   # Tumor
    }

    for folder, label in class_dirs.items():
        folder_path = config.BINARY_DATA_DIR / folder
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        paths = []
        for ext in exts:
            paths += glob.glob(str(folder_path / ext))
            paths += glob.glob(str(folder_path / ext.upper()))

        print(f"  '{folder}' ({label}): {len(paths)} images")

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
                X.append(np.array(img, dtype=np.float32))
                y.append(label)
            except Exception as e:
                print(f"    Warning: skipping {p} — {e}")

    X = np.array(X)
    y = np.array(y, dtype=np.int32)
    print(f"  Total: {len(X)} images  |  Tumor: {y.sum()}  No Tumor: {(y==0).sum()}")
    return X, y


# ─── Augmentation layers (Keras 3, applied inside the model graph) ────────────
def build_augmentation_pipeline():
    """Strong augmentation suitable for a small medical dataset."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.25),            # ±90°
        keras.layers.RandomZoom((-0.25, 0.15)),        # zoom out more than in
        keras.layers.RandomTranslation(0.15, 0.15),
        keras.layers.RandomBrightness(0.25),
        keras.layers.RandomContrast(0.25),
    ], name="augmentation")


# ─── Model definition ─────────────────────────────────────────────────────────
def build_binary_model(freeze_base=True):
    """
    EfficientNetB4 backbone + custom head for binary classification.

    Strategy for small datasets:
    • Heavy dropout (0.5) to prevent overfitting
    • L2 regularisation on Dense layer
    • Batch normalisation after dense for stable training
    • Global average pooling (no dense before it) to reduce parameter count
    """
    # EfficientNetB4 expects 0–255 inputs (preprocessing built-in)
    base = keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, config.CHANNELS),
        pooling=None,
    )
    base.trainable = not freeze_base

    inputs  = keras.Input(shape=(config.IMG_WIDTH, config.IMG_HEIGHT, config.CHANNELS))
    x       = build_augmentation_pipeline()(inputs)          # augment only during training
    x       = base(x, training=False)                        # freeze BN stats when base frozen
    x       = keras.layers.GlobalAveragePooling2D()(x)
    x       = keras.layers.Dropout(0.5)(x)
    x       = keras.layers.Dense(
                  256,
                  activation="relu",
                  kernel_regularizer=keras.regularizers.L2(1e-4),
              )(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="binary_tumor_detector")
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# ─── Callbacks ────────────────────────────────────────────────────────────────
def get_callbacks(model_path, patience_es=12, patience_lr=5):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=patience_lr,
            min_lr=1e-8,
            verbose=1,
        ),
    ]


# ─── Evaluation helpers ────────────────────────────────────────────────────────
def plot_history(history_dict, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_dict["accuracy"],     label="Train Acc")
    axes[0].plot(history_dict["val_accuracy"], label="Val Acc")
    axes[0].set_title(f"Accuracy {tag}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_dict["loss"],     label="Train Loss")
    axes[1].plot(history_dict["val_loss"], label="Val Loss")
    axes[1].set_title(f"Loss {tag}")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.MODELS_DIR / f"binary_training_history{tag}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  History plot saved → {out}")
    plt.close()


def plot_cm(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.BINARY_CLASSES,
                yticklabels=config.BINARY_CLASSES)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    out = config.MODELS_DIR / "binary_confusion_matrix.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Confusion matrix saved → {out}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("BINARY TUMOR DETECTOR — TRAINING")
    print("=" * 60)

    # 1. Load data
    X, y = load_binary_data()

    # Split: 70 % train | 15 % val | 15 % test  (stratified)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=config.RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.18,   # 0.18 of 0.85 ≈ 15 % of total
        random_state=config.RANDOM_STATE, stratify=y_trainval
    )

    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # Class weights to handle imbalance
    n_total = len(y_train)
    n_pos   = y_train.sum()
    n_neg   = n_total - n_pos
    class_weight = {
        0: n_total / (2.0 * n_neg),
        1: n_total / (2.0 * n_pos),
    }
    print(f"  Class weights: {class_weight}")

    # TF datasets (efficient pipeline)
    def make_dataset(X_, y_, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X_, y_.astype(np.float32)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X_), seed=config.RANDOM_STATE)
        ds = ds.batch(config.BINARY_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val)
    test_ds  = make_dataset(X_test,  y_test)

    # ── Phase 1: Train head only (base frozen) ────────────────────────────────
    print("\n[2/5] Building model (base frozen)...")
    model, base = build_binary_model(freeze_base=True)
    compile_model(model, config.BINARY_LEARNING_RATE)
    model.summary(line_length=80)

    print("\n[3/5] Phase 1 — training head only...")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.BINARY_EPOCHS,
        class_weight=class_weight,
        callbacks=get_callbacks(config.BINARY_BEST_MODEL_PATH, patience_es=12, patience_lr=4),
        verbose=1,
    )
    plot_history(h1.history, tag="_phase1")

    # ── Phase 2: Fine-tune top 50 layers of base ──────────────────────────────
    print("\n[4/5] Phase 2 — fine-tuning top 50 layers...")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    # Lower LR for fine-tuning
    compile_model(model, config.BINARY_LEARNING_RATE / 10)

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weight,
        callbacks=get_callbacks(config.BINARY_BEST_MODEL_PATH, patience_es=10, patience_lr=3),
        verbose=1,
    )
    plot_history(h2.history, tag="_phase2")

    # ── Save final model ──────────────────────────────────────────────────────
    print("\n[5/5] Saving final model...")
    model.save(str(config.BINARY_MODEL_PATH))
    print(f"  Final model → {config.BINARY_MODEL_PATH}")
    print(f"  Best model  → {config.BINARY_BEST_MODEL_PATH}")

    # Save combined history
    history_combined = {}
    for k, v in h1.history.items():
        history_combined[k] = v + h2.history.get(k, [])
    with open(str(config.MODELS_DIR / "binary_training_history.pkl"), "wb") as f:
        pickle.dump(history_combined, f)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION ON HOLD-OUT TEST SET")
    print("=" * 60)

    y_prob = model.predict(test_ds, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=config.BINARY_CLASSES, digits=4))

    plot_cm(y_test, y_pred)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
