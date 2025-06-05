from abc import ABC, abstractmethod
import chess

class AIInterface(ABC):
    """
    Абстрактний базовий клас для всіх реалізацій шахового штучного інтелекту.
    Визначає інтерфейс, який повинен бути реалізований конкретними AI.
    """

    @abstractmethod
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Генерує найкращий хід для поточної позиції на дошці.
        
        :param board: Об'єкт шахової дошки з поточною позицією
        :return: Об'єкт ходу chess.Move
        """
        pass

    @abstractmethod
    def set_difficulty(self, level: int):
        """
        Встановлює рівень складності AI.
        
        :param level: Ціле число, що визначає рівень складності (1-легкий, 2-середній, 3-складний)
        """
        pass

    @abstractmethod
    def set_time_limit(self, seconds: float):
        """
        Встановлює обмеження часу на обдумування ходу.
        
        :param seconds: Час у секундах, який AI може витратити на пошук ходу
        """
        pass

    @abstractmethod
    def use_neural_network(self, use_nn: bool):
        """
        Вмикає або вимикає використання нейромережі для оцінки позицій.
        
        :param use_nn: True - використовувати нейромережу, False - не використовувати
        """
        pass

    @abstractmethod
    def stop_calculation(self):
        """
        Припиняє поточні обчислення (якщо AI працює в окремому потоці).
        """
        pass

    @abstractmethod
    def set_position(self, board: chess.Board):
        """
        Встановлює поточну позицію для аналізу.
        Цей метод може бути використаний для підготовки AI перед генерацією ходу.
        
        :param board: Об'єкт шахової дошки з поточною позицією
        """
        pass

    @abstractmethod
    def get_evaluation(self, board: chess.Board) -> float:
        """
        Повертає оцінку поточної позиції з точки зору AI.
        Додатні значення - перевага білих, від'ємні - перевага чорних.
        
        :param board: Об'єкт шахової дошки з поточною позицією
        :return: Числова оцінка позиції
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Повертає назву AI.
        
        :return: Назва реалізації AI
        """
        pass

    @abstractmethod
    def get_version(self) -> str:
        """
        Повертає версію AI.
        
        :return: Версія реалізації AI
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Перевіряє, чи AI готовий до генерації ходу.
        
        :return: True, якщо AI готовий, False - якщо ні
        """
        pass

    @abstractmethod
    def set_verbose(self, verbose: bool):
        """
        Вмикає або вимикає додатковий вивід інформації під час роботи AI.
        
        :param verbose: True - виводити додаткову інформацію, False - мінімальний вивід
        """
        pass