import tensorflow as tf
import numpy as np 
import os 
import time 
from utils.fen_converter import fen_to_tensor 
from constants import TENSOR_SHAPE, DEFAULT_MODEL 
import logging 

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TFEvaluator")

class TFEvaluator:
    def __init__(self, model_path=DEFAULT_MODEL, use_gpu=True, cache_size=1000):
        """
        Ініціалізація оцінювача позицій на основі нейромережі TensorFlow.
        
        :param model_path: Шлях до файлу моделі (.h5 або .keras)
        :param use_gpu: Чи використовувати GPU для обчислень
        :param cache_size: Розмір кешу для зберігання оцінок позицій
        """
        self.model = None
        self.model_path = model_path 
        self.use_gpu = use_gpu
        self.cache = {}
        self.cache_size = cache_size 
        self.load_times = []
        self.eval_times = []

        self.gpu_available = tf.config.list_physical_devices('GPU') and use_gpu 

        self.load_model()

        self.initialize_session()

    def load_model(self):
        start_time = time.time()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            load_time = time.time() - start_time 
            self.load_times.append(load_time)

            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Load time: {load_time:.2f} seconds")
            logger.info(f"Model architecture: {self.model.summary()}")

            input_shape = self.model.input_shape[1:]
            if input_shape != TENSOR_SHAPE:
                logger.warning(f"Model expects input shape {input_shape}, "
                               f"but our tensor shape is {TENSOR_SHAPE}. "
                               "This may cause issues.")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def initialize_session(self):
        """Ініціалізація сесії TensorFlow для оптимальної продуктивності"""
        try:
            if self.gpu_available:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled for GPU acceleration")

                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled")
            else:
                logger.info("Using CPU for evaluation")
            
            test_tensor = np.zeros((1, *TENSOR_SHAPE), dtype=np.float32)
            _ = self.model.predict(test_tensor, verbose=0)

        except Exception as e:
            logger.error(f"Error initializing TensorFlow session: {e}")

    def evaluate(self, fen):
        """
        Оцінює шахову позицію, задану у форматі FEN.
        Повертає оцінку від -1 (перевага чорних) до 1 (перевага білих).
        
        :param fen: Рядок FEN, що представляє шахову позицію
        :return: float оцінка позиції
        """
        # Перевірка кешу
        if fen in self.cache:
            return self.cache[fen]
        
        start_time = time.time()
        
        try:
            # Конвертуємо FEN у тензор
            tensor = fen_to_tensor(fen)
            input_tensor = np.expand_dims(tensor, axis=0)  # Додаємо batch розмірність
            
            # Виконуємо передбачення
            prediction = self.model.predict(input_tensor, verbose=0)
            evaluation = float(prediction[0][0])  # Перетворюємо в простий float
            
            eval_time = time.time() - start_time
            self.eval_times.append(eval_time)
            
            # Додаємо в кеш
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))  # Видаляємо найстаріший елемент
            self.cache[fen] = evaluation
            
            if self.eval_times and len(self.eval_times) % 100 == 0:
                avg_time = sum(self.eval_times[-100:]) / 100
                logger.debug(f"Evaluation time (last 100): avg={avg_time:.4f}s")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating position: {fen}")
            logger.error(f"Error message: {str(e)}")
            logger.warning("NN failed, agent may fallback to random move")
            return 0.0  # Повертаємо нейтральну оцінку у разі помилки
    
    def batch_evaluate(self, fen_list):
        """
        Оцінює список позицій за один прохід для підвищення ефективності.
        
        :param fen_list: Список FEN рядків
        :return: Список оцінок
        """
        if not fen_list:
            return []
        
        start_time = time.time()
        
        try:
            # Готуємо вхідні дані
            tensors = []
            for fen in fen_list:
                # Використовуємо кешовані значення або створюємо новий тензор
                if fen in self.cache:
                    continue
                tensors.append(fen_to_tensor(fen))
            
            if tensors:
                input_tensor = np.array(tensors)
                
                # Виконуємо передбачення
                predictions = self.model.predict(input_tensor, verbose=0)
                
                # Оновлюємо кеш
                for fen, pred in zip(fen_list, predictions):
                    evaluation = float(pred[0])
                    if len(self.cache) >= self.cache_size:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[fen] = evaluation
            
            # Повертаємо оцінки для всіх запитів
            return [self.cache.get(fen, 0.0) for fen in fen_list]
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}")
            return [0.0] * len(fen_list)
    
    def clear_cache(self):
        """Очищує кеш оцінок"""
        self.cache.clear()
        logger.info("Evaluation cache cleared")
    
    def get_performance_stats(self):
        """Повертає статистику продуктивності"""
        stats = {
            "total_evaluations": len(self.eval_times),
            "average_eval_time": sum(self.eval_times) / len(self.eval_times) if self.eval_times else 0,
            "min_eval_time": min(self.eval_times) if self.eval_times else 0,
            "max_eval_time": max(self.eval_times) if self.eval_times else 0,
            "cache_size": len(self.cache),
            "cache_hits": len(self.eval_times) - len(self.cache) if self.eval_times else 0,
            "gpu_used": self.gpu_available
        }
        return stats
    
    def warmup(self, num_positions=50):
        """Прогрів моделі для стабілізації швидкості"""
        logger.info("Warming up model...")
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Початкова позиція
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",  # Сицилійська
            "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1",  # Нестандартна позиція
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Складніша
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"  # Позиція з пішаками
        ]
        
        # Створюємо більше позицій шляхом циклічного повторення
        warmup_fens = [test_fens[i % len(test_fens)] for i in range(num_positions)]
        
        # Оцінюємо позиції
        _ = self.batch_evaluate(warmup_fens)
        
        stats = self.get_performance_stats()
        logger.info(f"Warmup complete. Average eval time: {stats['average_eval_time']:.4f}s")

# Функція для створення базової моделі (використовується при тренуванні)
def create_basic_model(input_shape=TENSOR_SHAPE):
    """
    Створює базову модель нейромережі для оцінки шахових позицій.
    
    :param input_shape: Розмірність вхідних даних (за замовчуванням 8x8x12)
    :return: Модель Keras
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        # Згорткові шари
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        # Global Average Pooling замість Flatten для зменшення кількості параметрів
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Повнозв'язні шари
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Вихідний шар
        tf.keras.layers.Dense(1, activation='tanh')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

