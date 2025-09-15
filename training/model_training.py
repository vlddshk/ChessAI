# model_training.py
"""
Покращений скрипт тренування моделі для шахового оцінювача.
Підтримує:
 - regression (tanh -> MSE) або classification (3-way softmax -> CategoricalCrossentropy)
 - роботу з mmap .npz (np.load(..., mmap_mode='r'))
 - tf.data pipeline
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
    """Приклад розширеної архітектури (використовується за запитом)."""
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
    Очікується що в npz ключі названі X_train, y_train, X_test, y_test
    """
    logger.info(f"Loading dataset from {path} (with mmap_mode='r')")
    with np.load(path, mmap_mode="r") as data:
        # гнучкість: допускаємо різні ключі
        def get(k):
            if k in data:
                return data[k]
            # fallback common names
            for alt in ["X_train", "x_train", "X", "X_train"]:
                if alt in data:
                    return data[alt]
            return None

        X_train = data.get("X_train", data.get("X", None))
        y_train = data.get("y_train", data.get("y", None))
        X_test = data.get("X_test", data.get("X_val", data.get("X_test", None)))
        y_test = data.get("y_test", data.get("y_val", data.get("y_test", None)))

        if X_train is None or y_train is None:
            raise RuntimeError("Не знайдено X_train/y_train у npz.")

    logger.info(f"Loaded shapes: X_train={getattr(X_train,'shape',None)}, y_train={getattr(y_train,'shape',None)}, X_test={getattr(X_test,'shape',None)}, y_test={getattr(y_test,'shape',None)}")
    return X_train, y_train, X_test, y_test

#def create_tf_dataset_from_numpy(X, y, batch_size=128, shuffle=True, buffer_size=10000):
#    """
#    Створює tf.data.Dataset прямо з numpy (працює і з memmap).
#    Якщо пам'ять критична — можна переключитись на generator-based approach.
#    """
#    n = int(X.shape[0])
#    ds = tf.data.Dataset.from_tensor_slices((X, y))
#    if shuffle:
#        ds = ds.shuffle(buffer_size=min(buffer_size, n))
#    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#    return ds

def create_tf_dataset_from_numpy(X, y, batch_size=64, shuffle=True, buffer_size=10000, repeat=False):
    """
    Безпечне створення tf.data.Dataset з великих numpy (mmap) масивів.
    X, y --- numpy arrays (можуть бути mmap)
    Підхід: створюємо Dataset.range(len) і map -> tf.numpy_function, щоб НЕ перетворювати весь масив в EagerTensor.
    """

    length = len(X)
    AUTOTUNE = tf.data.AUTOTUNE

    def _fetch_pair(i):
        # i -> numpy int64
        idx = int(i)
        x = X[idx]  # numpy slice (не копіює весь масив)
        yy = y[idx]
        # забезпечуємо потрібний тип
        return x.astype(np.float32), yy.astype(np.float32)

    def _py_map(i):
        x, yy = tf.numpy_function(func=_fetch_pair, inp=[i], Tout=[tf.float32, tf.float32])
        # (Важливо) задаємо форми, щоб TF знав shapes (необхідно для деяких моделей)
        x.set_shape((TENSOR_SHAPE))  # наприклад (8,8,12)
        # Якщо y скаляр -> set_shape(()) або (1,) якщо колонки
        # Припускаємо, що y має shape (n,) або (n,1) або (n,3)
        if hasattr(y, "ndim") and y.ndim == 1:
            yy.set_shape(())
        else:
            yy.set_shape((y.shape[1],))
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

    # load dataset (supports mmap)
    X_train_all, y_train_all, X_test, y_test = load_dataset_npz(args.dataset)

    # If labels are one-hot but user chose regression, reduce to scalar via mapping
    target_type = args.target_type.lower()
    if target_type == "regression":
        # If y_train_all is one-hot (N,3), convert to scalar in {-1,0,1}
        if y_train_all.ndim == 2 and y_train_all.shape[1] == 3:
            logger.info("Converting one-hot labels -> scalar regression targets (-1/0/1)")

            idx_train = np.argmax(y_train, axis=1)
            idx_val = np.argmax(y_val, axis=1) if y_val is not None else None
            idx_test = np.argmax(y_test, axis=1) if y_test is not None else None

            mapping = {0: -1.0, 1: 0.0, 2: 1.0}  # ПРИКЛАД: підлаштуй під реальну порядок колонок
            y_train = np.vectorize(mapping.get)(idx_train).astype(np.float32)
            if idx_val is not None:
                y_val = np.vectorize(mapping.get)(idx_val).astype(np.float32)
            if idx_test is not None:
                y_test = np.vectorize(mapping.get)(idx_test).astype(np.float32)

            idx = np.argmax(y_train_all, axis=1)
            scalar = np.zeros(len(idx), dtype=np.float32)
            scalar[idx == 0] = 1.0
            scalar[idx == 1] = 0.0
            scalar[idx == 2] = -1.0
            y_train_all = scalar
            if y_test is not None and y_test.ndim == 2 and y_test.shape[1] == 3:
                idx = np.argmax(y_test, axis=1)
                scalar = np.zeros(len(idx), dtype=np.float32)
                scalar[idx == 0] = 1.0
                scalar[idx == 1] = 0.0
                scalar[idx == 2] = -1.0
                y_test = scalar




    elif target_type == "classification":
        # If y are scalars -1/0/1 convert to one-hot with order [white,draw,black]
        if y_train_all.ndim == 1:
            logger.info("Converting scalar targets -> one-hot classification targets")
            y_new = np.zeros((len(y_train_all), 3), dtype=np.float32)
            y_new[y_train_all == 1.0, 0] = 1.0
            y_new[y_train_all == 0.0, 1] = 1.0
            y_new[y_train_all == -1.0, 2] = 1.0
            y_train_all = y_new
            if y_test is not None and y_test.ndim == 1:
                y_new = np.zeros((len(y_test), 3), dtype=np.float32)
                y_new[y_test == 1.0, 0] = 1.0
                y_new[y_test == 0.0, 1] = 1.0
                y_new[y_test == -1.0, 2] = 1.0
                y_test = y_new

    # Якщо тест-сет порожній — поділимо train на train/val/test
    if X_test is None or y_test is None:
        logger.info("No explicit test set found - splitting train -> train/val/test")
        total = int(X_train_all.shape[0])
        test_sz = int(total * args.test_split)
        val_sz = int(total * args.val_split)
        # last part as test, before that val
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
        X_val = X_test[:0]  # empty
        y_val = y_test[:0]
    else:
        X_train = X_train_all[:-val_sz]
        y_train = y_train_all[:-val_sz]
        X_val = X_train_all[-val_sz:]
        y_val = y_train_all[-val_sz:]

    logger.info(f"Final sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

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
        except Exception:
            class_weight = None

    # callbacks
    cb_list, ckpt_path, logs_dir = create_callbacks(args.model_name, args.output_dir, patience=args.patience, monitor="val_loss")
    logger.info(f"Callbacks prepared, checkpoint path: {ckpt_path}")

    # fit
    logger.info("Starting model.fit()")
    history = model.fit(
        train_ds,
        validation_data=val_ds if len(X_val) > 0 else None,
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
