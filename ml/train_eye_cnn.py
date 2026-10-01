"""Train the eye-state CNN on the MRL Eye Dataset and export it to TFLite — Role 1.

Run this on a laptop or Google Colab (GPU), NOT on the Pi. Then copy
models/eye_state.tflite to the Pi's fido/models/ folder.

Usage:
    python ml/train_eye_cnn.py --data /path/to/mrlEyes_2018_01 --epochs 8
    python ml/train_eye_cnn.py --data /path/to/kaggle_folder --limit 20000   # quicker run

Accepts either dataset layout:
  * Original MRL: subject folders of PNGs named
      s0001_00001_0_0_0_0_0_01.png  -> fields: subject_image_gender_glasses_EYESTATE_reflections_lighting_sensor
      EYESTATE 0 = closed, 1 = open
  * Kaggle re-uploads: class folders whose names contain "close"/"closed"/"sleepy" vs "open"/"awake"

Output: P(eye closed) in [0, 1] — matches fido/sensing/camera.py.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

IMG = 64  # must equal SensingConfig.eye_input_size


def label_for(path: Path) -> int | None:
    """1 = closed, 0 = open, None = skip."""
    parts = path.stem.split("_")
    if len(parts) >= 8 and parts[4] in ("0", "1"):
        return 1 if parts[4] == "0" else 0
    for parent in (p.name.lower() for p in path.parents):
        if any(k in parent for k in ("close", "sleepy", "drows")):
            return 1
        if any(k in parent for k in ("open", "awake", "alert")):
            return 0
    return None


def load_dataset(root: Path, limit: int | None, seed: int = 42):
    import cv2
    files = [p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    labelled = [(p, label_for(p)) for p in files]
    labelled = [(p, y) for p, y in labelled if y is not None]
    if not labelled:
        raise SystemExit(f"No labelled images found under {root}")
    random.Random(seed).shuffle(labelled)
    if limit:
        labelled = labelled[:limit]
    X = np.zeros((len(labelled), IMG, IMG, 1), np.float32)
    y = np.zeros((len(labelled),), np.float32)
    for i, (p, lab) in enumerate(labelled):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        X[i, ..., 0] = cv2.resize(img, (IMG, IMG)) / 255.0
        y[i] = lab
    print(f"Loaded {len(X)} images — closed: {int(y.sum())}, open: {int(len(y) - y.sum())}")
    return X, y


def build_models():
    """Returns (train_model, core_model). Augmentation lives only in train_model so the
    exported TFLite file contains just the CNN."""
    from tensorflow import keras
    from tensorflow.keras import layers
    core = keras.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.Conv2D(16, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="eye_state")
    train = keras.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2, value_range=(0, 1)),
        core,
    ])
    train.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return train, core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap number of images for a quick run")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    args = ap.parse_args()

    X, y = load_dataset(args.data, args.limit)
    n = len(X)
    i_val, i_test = int(n * 0.7), int(n * 0.85)
    Xtr, ytr, Xva, yva, Xte, yte = X[:i_val], y[:i_val], X[i_val:i_test], y[i_val:i_test], X[i_test:], y[i_test:]

    from tensorflow import keras
    import tensorflow as tf
    model, core = build_models()
    core.summary()
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=64,
              callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    results = model.evaluate(Xte, yte, return_dict=True)
    print("Test:", {k: round(v, 4) for k, v in results.items()})

    args.out.mkdir(parents=True, exist_ok=True)
    core.save(args.out / "eye_state.keras")
    conv = tf.lite.TFLiteConverter.from_keras_model(core)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    (args.out / "eye_state.tflite").write_bytes(conv.convert())
    (args.out / "eye_state_metrics.txt").write_text(str(results))
    print(f"Saved {args.out / 'eye_state.tflite'} — copy it to the Pi's fido/models/ folder")


if __name__ == "__main__":
    main()
