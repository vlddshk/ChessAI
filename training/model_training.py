import os 
import sys 
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime 
import matplotlib.pyplot as plt 
import argparse
import json 
import gc 
from chess_app.ai.tf_evaluator import create_basic_model 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chess_app.constants import DEFAULT_MODEL, MODELS_DIR, TENSOR_SHAPE

def load_dataset(dataset_path):
    """
    Завантажує датасет з .npz файлу
    
    :param dataset_path: Шлях до .npz файлу
    :return: Кортеж (X_train, y_train, X_test, y_test)
    """
    print(f"Завантаження датасету: {dataset_path}")
    with np.load(dataset_path) as data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

    print(f"Тренувальний набір: {X_train.shape[0]} позицій")
    print(f"Тестовий набір: {X_test.shape[0]} позицій")
    print(f"Розмірність тензора: {X_train.shape[1:]}")

    return X_train, y_train, X_test, y_test 

def create_model(input_shape, learning_rate=0.001, model_type='basic'):
    """
    Створює модель нейромережі
    
    :param input_shape: Розмірність вхідних даних
    :param learning_rate: Швидкість навчання
    :param model_type: Тип моделі ('basic', 'advanced')
    :return: Модель Keras
    """
    if model_type == 'basic':
        print("Створення базової моделі...")
        model = create_basic_model(input_shape)
    elif model_type == 'advanced':
        print("Створення розширеної моделі...")
        model = create_advanced_model(input_shape)
    else:
        raise ValueError(f"Невідомий тип моделі: {model_type}")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=MeanSquaredError(),
        metrics=['mae', 'mse']
    )

    model.summary()
    return model 

def create_advanced_model(input_shape):
    """
    Створює розширену модель з більшою кількістю шарів та ускладненою архітектурою
    
    :param input_shape: Розмірність вхідних даних
    :return: Модель Keras
    """
    inputs = tf.keras.Input(shape=input_shape)

    #1
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    #2
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    #3
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(
                model, X_train, y_train, X_val, y_val, epochs=100, 
                batch_size=256, callbacks=None
                ):
    """
    Тренує модель нейромережі
    
    :param model: Модель Keras
    :param X_train: Тренувальні дані
    :param y_train: Тренувальні мітки
    :param X_val: Валідаційні дані
    :param y_val: Валідаційні мітки
    :param epochs: Кількість епох
    :param batch_size: Розмір пакету
    :param callbacks: Список зворотних викликів
    :return: Історія тренування
    """
    print("\nПочаток тренування моделі...")
    print(f"Епохи: {epochs}")
    print(f"Розмір пакету: {batch_size}")
    print(f"Тренувальні дані: {X_train.shape[0]} зразків")
    print(f"Валідаційні дані: {X_val.shape[0]} зразків")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history

def create_callbacks(model_name, checkpoint_dir, logs_dir, patience=10, monitor='val_loss'):
    """
    Створює зворотні виклики для тренування
    
    :param model_name: Назва моделі
    :param checkpoint_dir: Директорія для збереження чекпоінтів
    :param logs_dir: Директорія для логів TensorBoard
    :param patience: Терпеливість для ранньої зупинки
    :param monitor: Метрика для моніторингу
    :return: Список зворотних викликів
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{timestamp}.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    tensorboard = TensorBoard(
        log_dir=os.path.join(logs_dir, f"{model_name}_{timestamp}"),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    return [checkpoint, early_stopping, tensorboard, reduce_lr]

def evaluate_model(model, X_test, y_test):
    """
    Оцінює модель на тестовому наборі
    
    :param model: Тренована модель
    :param X_test: Тестові дані
    :param y_test: Тестові мітки
    :return: Результати оцінки
    """
    print("\nОцінка моделі на тестовому наборі...")
    results = model.evaluate(X_test, y_test, verbose=1)

    metrics = {
        'loss': results[0],
        'mae': results[1],
        'mse': results[2]
    }

    print("Результати оцінки:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics 

def save_model(model, output_dir, model_name, history, test_metrics):
    """
    Зберігає модель та метадані
    
    :param model: Тренована модель
    :param output_dir: Вихідна директорія
    :param model_name: Назва моделі
    :param history: Історія тренування
    :param test_metrics: Результати оцінки
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.keras"
    model_path = os.path.join(output_dir, model_filename)

    print(f"\nЗбереження моделі: {model_path}")
    model.save(model_path)

    history_filename = f"history_{model_name}_{timestamp}.json"
    history_path = os.path.join(output_dir, history_filename)

    with open(history_path, 'w') as f:
        json.dump(history.history, f)

    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'training_date': datetime.now().isoformat(),
        'test_metrics': test_metrics,
        'model_path': model_path,
        'history_path': history_path
        }
    
    metadata_filename = f"metadata_{model_name}_{timestamp}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Метадані збережено: {metadata_path}")

    return model_path 

def plot_training_history(history, output_dir, model_name):
    """
    Створює та зберігає графіки історії тренування
    
    :param history: Історія тренування
    :param output_dir: Вихідна директорія
    :param model_name: Назва моделі
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(15, 10))
    
    # Графік втрат
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Тренувальні втрати')
    plt.plot(history.history['val_loss'], label='Валідаційні втрати')
    plt.title('Втрати')
    plt.ylabel('Втрати')
    plt.xlabel('Епоха')
    plt.legend()
    
    # Графік MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Тренувальний MAE')
    plt.plot(history.history['val_mae'], label='Валідаційний MAE')
    plt.title('Середня абсолютна похибка (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('Епоха')
    plt.legend()
    
    # Графік MSE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['mse'], label='Тренувальний MSE')
    plt.plot(history.history['val_mse'], label='Валідаційний MSE')
    plt.title('Середня квадратична похибка (MSE)')
    plt.ylabel('MSE')
    plt.xlabel('Епоха')
    plt.legend()
    
    # Графік швидкості навчання
    if 'lr' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['lr'], label='Швидкість навчання')
        plt.title('Швидкість навчання')
        plt.ylabel('LR')
        plt.xlabel('Епоха')
        plt.legend()
    
    plt.tight_layout()
    
    # Збереження графіків
    plot_path = os.path.join(output_dir, f"training_plots_{model_name}_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Графіки тренування збережено: {plot_path}")
    return plot_path


def main():
    """Основна функція для тренування моделі"""
    # Парсинг аргументів командного рядка
    parser = argparse.ArgumentParser(description='Тренування шахової нейромережі')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Шлях до .npz файлу з датасетом')
    parser.add_argument('--model_type', type=str, default='basic',
                        choices=['basic', 'advanced'],
                        help='Тип моделі: basic або advanced (default: basic)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Кількість епох тренування (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Розмір пакету (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Початкова швидкість навчання (default: 0.001)')
    parser.add_argument('--output_dir', type=str, default=MODELS_DIR,
                        help=f'Директорія для збереження моделі (default: {MODELS_DIR})')
    parser.add_argument('--model_name', type=str, default='chess_evaluator',
                        help='Базове ім\'я моделі (default: chess_evaluator)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Використовувати GPU для тренування (якщо доступно)')
    parser.add_argument('--memory_limit', type=int, default=1024,
                        help='Обмеження пам\'яті GPU у МБ (default: 1024)')
    
    args = parser.parse_args()
    
    # Налаштування GPU
    if args.use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Обмеження пам'яті GPU
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory_limit)]
                    )
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"{len(gpus)} фізичних GPU, {len(logical_gpus)} логічних GPU")
                print(f"Обмеження пам'яті GPU: {args.memory_limit} МБ")
            except RuntimeError as e:
                print(e)
        else:
            print("GPU не знайдено, тренування буде на CPU")
    else:
        print("Тренування на CPU")
        # Вимикаємо використання GPU
        tf.config.set_visible_devices([], 'GPU')
    
    # Завантаження датасету
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    
    # Розділення тренувального набору на тренувальну та валідаційну частини
    val_split = 0.1
    split_idx = int(len(X_train) * (1 - val_split))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    print(f"\nФінальні розміри наборів:")
    print(f"Тренувальний: {X_train.shape[0]} зразків")
    print(f"Валідаційний: {X_val.shape[0]} зразків")
    print(f"Тестовий: {X_test.shape[0]} зразків")
    
    # Створення моделі
    model = create_model(
        input_shape=TENSOR_SHAPE,
        learning_rate=args.learning_rate,
        model_type=args.model_type
    )
    
    # Підготовка директорій
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", args.model_name)
    logs_dir = os.path.join(args.output_dir, "logs")
    
    # Створення зворотних викликів
    callbacks = create_callbacks(
        model_name=args.model_name,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        patience=15
    )
    
    # Тренування моделі
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Оцінка моделі
    test_metrics = evaluate_model(model, X_test, y_test)
    
    # Збереження моделі та метаданих
    model_path = save_model(
        model,
        args.output_dir,
        args.model_name,
        history,
        test_metrics
    )
    
    # Створення та збереження графіків
    plot_training_history(history, args.output_dir, args.model_name)
    
    # Звільнення пам'яті
    del model
    del X_train, y_train, X_test, y_test
    gc.collect()
    
    print("\nТренування успішно завершено!")
    print(f"Модель збережено: {model_path}")

if __name__ == "__main__":
    main()