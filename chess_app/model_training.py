# model_training.py
"""
скрипт тренування моделі для шахового оцінювача.
Підтримує:
 - regression (tanh -> MSE) або classification (3-way softmax -> CategoricalCrossentropy)
 - роботу з mmap .npz (np.load(..., mmap_mode='r'))
 - tf.data pipeline для великих масивів
 - mixed precision (опціонально)
 - автоматичні callbacks, logging, class weights
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import gc

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
)

# Спроба імпорту create_basic_model з різних місць проекту
try:
    from ai.tf_evaluator import create_basic_model
except Exception:
    try:
        from chess_app.ai.tf_evaluator import create_basic_model
    except Exception as e:
        raise ImportError("Не вдалось знайти create_basic_model; перевір структуру проекту.") from e

# Константи (спроба імпорту)
try:
    from constants import MODELS_DIR, TENSOR_SHAPE, DEFAULT_MODEL
except Exception:
    try:
        from chess_app.constants import MODELS_DIR, TENSOR_SHAPE, DEFAULT_MODEL
    except Exception:
        MODELS_DIR = "models"
        TENSOR_SHAPE = (8, 8, 12)
        DEFAULT_MODEL = os.path.join(MODELS_DIR, "chess_evaluator.keras")

# Логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("model_training")

# -------------------- Модельні конструкції --------------------

def create_advanced_model(input_shape):
    """Розширена архітектура."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    out = tf.keras.layers.Dense(1, activation="tanh")(x)
    return tf.keras.Model(inputs=inputs, outputs=out)

def build_model(input_shape, learning_rate=1e-3, model_type="basic", target_type="regression"):
    """
    Повертає скомпільовану модель залежно від типу задачі.
    target_type: 'regression' -> tanh output + mse
                 'classification' -> 3-class softmax + cce
    """
    if model_type == "basic":
        model = create_basic_model(input_shape)
    elif model_type == "advanced":
        model = create_advanced_model(input_shape)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = Adam(learning_rate=learning_rate)

    if target_type == "regression":
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
    elif target_type == "classification":
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        raise ValueError("target_type must be 'regression' or 'classification'")

    logger.info("Model compiled")
    model.summary(print_fn=lambda s: logger.info(s))
    return model

# -------------------- Датасет / pipeline --------------------

def load_dataset_npz(path):
    """
    Підтримує npz файли. Використаємо mmap_mode='r' щоб не вантажити все у RAM.
    Повертаємо numpy-обʼєкти (можливо memmap) у форматі (X_train, y_train, X_test, y_test)
    Очікується що в npz ключі названі X_train, y_train, X_test, y_test або X,y.
    """
    logger.info(f"Loading dataset from {path} (with mmap_mode='r')")
    data = np.load(path, mmap_mode="r")
    # try common keys
    X_train = data.get("X_train", None)
    y_train = data.get("y_train", None)
    X_test = data.get("X_test", None)
    y_test = data.get("y_test", None)
    if X_train is None or y_train is None:
        # fallback keys
        X_train = data.get("X", X_train)
        y_train = data.get("y", y_train)
    # It's fine if X_test/y_test are None (we split later)
    logger.info(f"Loaded shapes: X_train={getattr(X_train,'shape',None)}, y_train={getattr(y_train,'shape',None)}, X_test={getattr(X_test,'shape',None)}, y_test={getattr(y_test,'shape',None)}")
    return X_train, y_train, X_test, y_test

def create_tf_dataset_from_numpy(X, y, batch_size=64, shuffle=True, buffer_size=10000, repeat=False):
    """
    Безпечне створення tf.data.Dataset з великих numpy (mmap) масивів.
    X, y --- numpy arrays (можуть бути mmap)
    Dataset.range(len) -> map через tf.numpy_function щоб уникнути повного копіювання в пам'ять.
    """
    if X is None or y is None:
        raise ValueError("X and y must be provided")

    length = int(X.shape[0])
    AUTOTUNE = tf.data.AUTOTUNE

    # determine y output shape for TF
    if hasattr(y, "ndim"):
        if y.ndim == 1:
            y_shape = ()            # scalar
        else:
            y_shape = (int(y.shape[1]),)
    else:
        y_shape = ()

    def _fetch_pair(i):
        idx = int(i)
        x = X[idx]  # numpy slice 
        yy = y[idx]
        return x.astype(np.float32), yy.astype(np.float32)

    def _py_map(i):
        x, yy = tf.numpy_function(func=_fetch_pair, inp=[i], Tout=[tf.float32, tf.float32])
        x.set_shape(TENSOR_SHAPE)
        # set y shape
        if y_shape == ():
            yy.set_shape(())
        else:
            yy.set_shape(y_shape)
        return x, yy

    ds = tf.data.Dataset.range(length)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(buffer_size, length))
    ds = ds.map(_py_map, num_parallel_calls=AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def compute_class_weights_from_onehot(y_onehot):
    """
    y_onehot: numpy array shape (N,3)
    Повертає dict для keras fit class_weight (мапа class_index->weight)
    """
    labels = np.argmax(y_onehot, axis=1)
    classes, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]
    class_weight = {}
    for c, cnt in zip(classes, counts):
        # inverse frequency
        class_weight[int(c)] = float(total) / float(cnt)
    logger.info(f"Computed class weights: {class_weight}")
    return class_weight

# -------------------- Callbacks & utils --------------------

def create_callbacks(model_name, outdir, patience=12, monitor="val_loss"):
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(outdir, f"{model_name}_{timestamp}.keras")
    logs_dir = os.path.join(outdir, "logs", f"{model_name}_{timestamp}")

    checkpoint = ModelCheckpoint(ckpt_path, monitor=monitor, save_best_only=True, verbose=1)
    early = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=1)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=1)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=max(2, patience//3), verbose=1, min_lr=1e-6)

    return [checkpoint, early, tb, reduce_lr], ckpt_path, logs_dir

def plot_history(history, outdir, model_name):
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(outdir, f"{model_name}_training_{timestamp}.png")

    plt.figure(figsize=(14, 8))
    # loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend(); plt.title("Loss")

    # metric 1
    plt.subplot(2, 2, 2)
    metric_names = [k for k in history.history.keys() if k.startswith("val_") and k != "val_loss"]
    if metric_names:
        m = metric_names[0].replace("val_", "")
        plt.plot(history.history.get(m, []), label=f"train_{m}")
        plt.plot(history.history.get(f"val_{m}", []), label=f"val_{m}")
        plt.legend(); plt.title(m)

    # optionally lr
    if "lr" in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history["lr"])
        plt.title("lr")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    logger.info(f"Saved training plot: {fname}")
    return fname

# -------------------- Main training routine --------------------

def train_entrypoint(args):
    # GPU config and mixed precision
    if args.use_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                if args.memory_limit:
                    for gpu in gpus:
                        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory_limit)])
                logger.info(f"Using GPUs: {len(gpus)} found")
            except RuntimeError as e:
                logger.warning(f"GPU config error: {e}")
        else:
            logger.info("No GPUs found; using CPU")
    else:
        # hide GPUs
        try:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("Forcing CPU usage (GPUs hidden)")
        except Exception:
            pass

    if args.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled")
        except Exception as e:
            logger.warning(f"Cannot enable mixed precision: {e}")

    # load dataset (mmap)
    X_train_all, y_train_all, X_test, y_test = load_dataset_npz(args.dataset)

    # --- TARGET conversions and sanity checks ---
    # Define order of one-hot columns in dataset. IMPORTANT: set according to data_preparation!
    # Example orders: ["black","draw","white"] or ["white","draw","black"]
    onehot_order = ["black", "draw", "white"]

    def onehot_to_scalar(arr, order=onehot_order):
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Expected one-hot array with shape (N,3)")
        idx = np.argmax(arr, axis=1)
        scalar = np.empty(len(idx), dtype=np.float32)
        mapping = {}
        for i, name in enumerate(order):
            if name == "white":
                mapping[i] = 1.0
            elif name == "draw":
                mapping[i] = 0.0
            elif name == "black":
                mapping[i] = -1.0
            else:
                raise ValueError(f"Unknown order token: {name}")
        for k, v in mapping.items():
            scalar[idx == k] = v
        return scalar

    def scalar_to_onehot(arr, order=onehot_order):
        arr = np.asarray(arr).reshape(-1)
        onehot = np.zeros((len(arr), 3), dtype=np.float32)
        idx_of = {}
        for i, name in enumerate(order):
            if name == "black":
                idx_of[-1.0] = i
            elif name == "draw":
                idx_of[0.0] = i
            elif name == "white":
                idx_of[1.0] = i
        for i, v in enumerate(arr):
            if abs(v - 1.0) < 1e-6:
                label = 1.0
            elif abs(v - 0.0) < 1e-6:
                label = 0.0
            elif abs(v + 1.0) < 1e-6:
                label = -1.0
            else:
                raise ValueError(f"Unexpected scalar label value: {v}")
            onehot[i, idx_of[label]] = 1.0
        return onehot

    target_type = args.target_type.lower()
    logger.info(f"Target type requested: {target_type}")

    # Normalize y arrays into y_train_all / y_test (do not yet split)
    if y_train_all is None:
        raise RuntimeError("y_train_all is None in dataset")

    # Convert based on requested target_type
    if target_type == "regression":
        # if one-hot -> scalar
        if getattr(y_train_all, "ndim", None) == 2 and y_train_all.shape[1] == 3:
            logger.info("Converting y_train_all one-hot -> scalar regression targets (-1/0/1)")
            y_train_all = onehot_to_scalar(y_train_all, order=onehot_order)
        # y_test
        if y_test is not None and getattr(y_test, "ndim", None) == 2 and y_test.shape[1] == 3:
            logger.info("Converting y_test one-hot -> scalar regression targets (-1/0/1)")
            y_test = onehot_to_scalar(y_test, order=onehot_order)
        # Ensure shapes: (N,1)
        y_train_all = np.asarray(y_train_all, dtype=np.float32).reshape(-1, 1)
        if y_test is not None:
            y_test = np.asarray(y_test, dtype=np.float32).reshape(-1, 1)
        logger.info(f"After conversion (regression): y_train_all shape={y_train_all.shape}, y_test shape={None if y_test is None else y_test.shape}")

    elif target_type == "classification":
        # if scalars -> one-hot
        if getattr(y_train_all, "ndim", None) == 1:
            logger.info("Converting y_train_all scalar -> one-hot classification targets")
            y_train_all = scalar_to_onehot(y_train_all, order=onehot_order)
        if y_test is not None and getattr(y_test, "ndim", None) == 1:
            logger.info("Converting y_test scalar -> one-hot classification targets")
            y_test = scalar_to_onehot(y_test, order=onehot_order)
        y_train_all = np.asarray(y_train_all, dtype=np.float32)
        if y_test is not None:
            y_test = np.asarray(y_test, dtype=np.float32)
        logger.info(f"After conversion (classification): y_train_all shape={y_train_all.shape}, y_test shape={None if y_test is None else y_test.shape}")

    else:
        raise ValueError("target_type must be 'regression' or 'classification'")

    # --- If no explicit test set, split train_all -> train/val/test ---
    if X_test is None or y_test is None:
        logger.info("No explicit test set found - splitting train -> train/val/test")
        total = int(X_train_all.shape[0])
        test_sz = int(total * args.test_split)
        val_sz = int(total * args.val_split)
        if test_sz + val_sz >= total:
            raise RuntimeError("Not enough data for requested splits")
        X_test = X_train_all[-test_sz:]
        y_test = y_train_all[-test_sz:]
        X_train_all = X_train_all[:total - test_sz]
        y_train_all = y_train_all[:total - test_sz]

    # create final train/val splits
    total_train = int(X_train_all.shape[0])
    val_sz = int(total_train * args.val_split)
    if val_sz <= 0:
        X_train = X_train_all
        y_train = y_train_all
        X_val = X_test[:0]
        y_val = y_test[:0]
    else:
        X_train = X_train_all[:-val_sz]
        y_train = y_train_all[:-val_sz]
        X_val = X_train_all[-val_sz:]
        y_val = y_train_all[-val_sz:]

    logger.info(f"Final sizes: train={int(X_train.shape[0])}, val={int(X_val.shape[0])}, test={int(X_test.shape[0])}")

    # Sanity asserts: shapes compatible with model expectations
    if target_type == "regression":
        assert y_train.ndim == 2 and y_train.shape[1] == 1, f"Regression expects y shape (N,1); got {y_train.shape}"
        if y_val.size:
            assert y_val.ndim == 2 and y_val.shape[1] == 1
        assert y_test.ndim == 2 and y_test.shape[1] == 1
    else:
        assert y_train.ndim == 2 and y_train.shape[1] == 3, f"Classification expects y shape (N,3); got {y_train.shape}"
        if y_val.size:
            assert y_val.ndim == 2 and y_val.shape[1] == 3
        assert y_test.ndim == 2 and y_test.shape[1] == 3

    # create dataset pipelines (attempt to use memmap directly)
    train_ds = create_tf_dataset_from_numpy(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset_from_numpy(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_ds = create_tf_dataset_from_numpy(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    # build model
    model = build_model(input_shape=TENSOR_SHAPE, learning_rate=args.learning_rate,
                        model_type=args.model_type, target_type=target_type)

    # optional class weights for classification
    class_weight = None
    if target_type == "classification":
        try:
            class_weight = compute_class_weights_from_onehot(y_train)
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")
            class_weight = None

    # callbacks
    cb_list, ckpt_path, logs_dir = create_callbacks(args.model_name, args.output_dir, patience=args.patience, monitor="val_loss")
    logger.info(f"Callbacks prepared, checkpoint path: {ckpt_path}")

    # fit
    logger.info("Starting model.fit()")
    history = model.fit(
        train_ds,
        validation_data=val_ds if int(X_val.shape[0]) > 0 else None,
        epochs=args.epochs,
        callbacks=cb_list,
        class_weight=class_weight,
        verbose=1
    )

    # evaluate
    logger.info("Evaluating on test set")
    eval_res = model.evaluate(test_ds, verbose=1)
    logger.info(f"Test evaluation: {eval_res}")

    # save model + metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_fname = f"{args.model_name}_{timestamp}.keras"
    model_path = os.path.join(args.output_dir, model_fname)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # save history & metadata
    hist_path = os.path.join(args.output_dir, f"history_{args.model_name}_{timestamp}.json")
    with open(hist_path, "w") as fh:
        json.dump(history.history, fh, indent=2)
    meta = {
        "model_name": args.model_name,
        "timestamp": timestamp,
        "model_path": model_path,
        "history": hist_path,
        "evaluation": eval_res,
        "args": vars(args)
    }
    meta_path = os.path.join(args.output_dir, f"metadata_{args.model_name}_{timestamp}.json")
    with open(meta_path, "w") as fm:
        json.dump(meta, fm, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    # plot history
    plot_history(history, args.output_dir, args.model_name)

    # cleanup
    del model
    gc.collect()

    return model_path, meta_path

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train chess evaluator model")
    p.add_argument("--dataset", required=True, help="Path to dataset .npz (generated by data_preparation)")
    p.add_argument("--model_type", choices=["basic", "advanced"], default="basic")
    p.add_argument("--target_type", choices=["regression", "classification"], default="regression",
                   help="regression: scalar (-1..1). classification: 3-way one-hot")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--output_dir", type=str, default=MODELS_DIR)
    p.add_argument("--model_name", type=str, default="chess_evaluator")
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--memory_limit", type=int, default=None)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=12)
    return p.parse_args()

def main():
    args = parse_args()
    logger.info("Starting training with args: %s", args)
    model_path, meta_path = train_entrypoint(args)
    logger.info(f"Training finished. Model: {model_path}, Metadata: {meta_path}")

if __name__ == "__main__":
    main()
