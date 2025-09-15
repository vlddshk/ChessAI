from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from typing import List

class GameMode(Enum):
    """
    Перелік доступних режимів гри
    """
    PvP = auto()      
    PvAI = auto()     

@dataclass
class AIDifficulty:
    """
    Клас, що визначає параметри складності AI
    """
    name: str              # Назва рівня складності
    depth: int             # Глибина пошуку для алгоритму minimax
    model_path: Optional[str] = None  # Шлях до моделі нейромережі

class GameModeManager:
    """
    Менеджер для керування режимами гри та рівнями складності AI
    """
    def __init__(self):
        # Доступні режими гри
        self.available_modes = {
            GameMode.PvP: "Гравець vs Гравець",
            GameMode.PvAI: "Гравець vs ШІ"
        }
        
        # Доступні рівні складності для AI
        self.available_difficulties = [
            AIDifficulty(
                name="Легкий",
                depth=2,
            ), 
            AIDifficulty(
                name="Середній",
                depth=3,
                model_path="models/chess_evaluator1.keras"
            ),
            AIDifficulty(
                name="Складний",
                depth=4,
                model_path="models/chess_evaluator2.keras"
            )
        ]
        
        # Поточні налаштування
        self.current_mode = GameMode.PvP
        self.current_difficulty = self.available_difficulties[0]
    
    def set_mode(self, mode: GameMode):
        """Встановлює поточний режим гри"""
        if mode in self.available_modes:
            self.current_mode = mode
    
    def set_difficulty(self, difficulty_name: str):
        """Встановлює рівень складності AI за назвою"""
        for diff in self.available_difficulties:
            if diff.name == difficulty_name:
                self.current_difficulty = diff
                return
        # Якщо не знайдено - встановити перший доступний рівень
        self.current_difficulty = self.available_difficulties[0]
    
    def get_mode_name(self) -> str:
        """Повертає назву поточного режиму гри"""
        return self.available_modes.get(self.current_mode, "Невідомий режим")
    
    def get_difficulty_name(self) -> str:
        """Повертає назву поточного рівня складності"""
        return self.current_difficulty.name
    
    def is_ai_turn(self, is_white_turn: bool) -> bool:
        """
        Визначає, чи зараз ходить AI
        - У режимі PvP: AI ніколи не ходить
        - У режимі PvAI: AI ходить, коли черга чорних (або білих, залежно від налаштувань)
        """
        if self.current_mode != GameMode.PvAI:
            return False
        
        # За замовчуванням AI грає чорними
        # Можна додати вибір сторони в майбутньому
        return not is_white_turn
    
    def get_ai_config(self) -> AIDifficulty:
        """Повертає поточну конфігурацію AI"""
        return self.current_difficulty
    

    def get_available_difficulties(self) -> List[str]:
        """Повертає список доступних рівнів складності"""
        return [diff.name for diff in self.available_difficulties]