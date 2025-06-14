import tensorflow as tf
import numpy as np
import logging
import os
from .tf_evaluator import TFEvaluator

class ModelLoader:
    """
    Завантажує та керує моделями TensorFlow для оцінки шахових позицій
    Підтримує кешування моделей для ефективного багаторазового використання
    """
    
    _cache = {}
    _current_model = None
    
    @classmethod
    def load_model(cls, model_path, model_type="evaluator"):
        """
        Завантажує модель з вказаного шляху або повертає кешовану версію
        
        Args:
            model_path (str): Шлях до файлу моделі (.h5 або .keras)
            model_type (str): Тип моделі ('evaluator', 'move_predictor', etc.)
            
        Returns:
            TFEvaluator: Інстанс оцінювача позицій
        """
        # Перевірка кешу
        cache_key = f"{model_path}_{model_type}"
        if cache_key in cls._cache:
            logging.info(f"Using cached model: {cache_key}")
            return cls._cache[cache_key]
        
        # Перевірка існування файлу
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file {model_path} does not exist")
        
        try:
            # Завантаження моделі TensorFlow
            logging.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Створення відповідного оцінювача
            if model_type == "evaluator":
                evaluator = TFEvaluator(model)
            # Додати інші типи моделей при необхідності
            else:
                evaluator = TFEvaluator(model)
            
            # Кешування моделі
            cls._cache[cache_key] = evaluator
            cls._current_model = evaluator
            
            logging.info(f"Model loaded successfully: {model_path}")
            return evaluator
            
        except Exception as e:
            logging.exception(f"Error loading model {model_path}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}") from e
    
    @classmethod
    def get_current_model(cls):
        """
        Повертає поточну активну модель
        
        Returns:
            TFEvaluator: Поточний активний оцінювач
        """
        if cls._current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return cls._current_model
    
    @classmethod
    def preload_models(cls, model_configs):
        """
        Попереднє завантаження моделей з конфігурації
        
        Args:
            model_configs (list): Список конфігурацій моделей у форматі:
                [{'path': 'models/model1.h5', 'type': 'evaluator'}, ...]
        """
        for config in model_configs:
            try:
                cls.load_model(config['path'], config.get('type', 'evaluator'))
            except Exception as e:
                logging.error(f"Preload failed for {config['path']}: {str(e)}")
    
    @classmethod
    def clear_cache(cls):
        """Очищає кеш моделей для звільнення пам'яті"""
        cls._cache = {}
        cls._current_model = None
        logging.info("Model cache cleared")
        tf.keras.backend.clear_session()  # Очищення сесії TensorFlow