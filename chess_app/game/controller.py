import chess
import time
from PyQt5.QtCore import QObject, pyqtSignal
from chess_app.game.state import ChessState
from chess_app.game.mode import GameMode, GameModeManager
from chess_app.ai.interface import AIInterface
from chess_app.ai.minimax_ai import MinimaxAI
from chess_app.constants import (
    UI_TEXTS, GAME_STATE_ONGOING, 
    GAME_STATE_CHECKMATE, GAME_STATE_STALEMATE, 
    GAME_STATE_DRAW
)

class GameController(QObject):
    # Сигнали
    move_executed = pyqtSignal(str)              # Хід був виконаний (UCI)
    game_state_changed = pyqtSignal(str)         # Змінився стан гри
    ai_move_requested = pyqtSignal(str)          # Запит на генерацію ходу AI
    ai_move_generated = pyqtSignal(str)          # Хід AI згенеровано
    game_over = pyqtSignal(str)                  # Гра завершена (результат)
    promotion_required = pyqtSignal(str, chess.PieceType)  # Потрібно перетворення пішака
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = ChessState()
        self.mode_manager = GameModeManager()
        self.ai = None
        self.ai_thread = None
        self.ai_thinking = False
        self.ai_move_queue = None
        self.init_ai()
        
        # Підключення сигналів
        self.ai_move_requested.connect(self.generate_ai_move)
    
    def init_ai(self):
        """Ініціалізація штучного інтелекту"""
        # Створюємо базовий AI
        self.ai = MinimaxAI(depth=2)
        
        # Завантажуємо модель нейромережі (якщо доступно)
        # Це буде реалізовано в tf_evaluator.py пізніше
    
    def set_mode(self, mode):
        """Встановлення режиму гри"""
        self.mode_manager.set_mode(mode)
    
    def set_difficulty(self, difficulty_name):
        """Встановлення рівня складності"""
        self.mode_manager.set_difficulty(difficulty_name)
        
        # Оновлюємо параметри AI
        config = self.mode_manager.get_ai_config()
        if self.ai:
            self.ai.depth = config.depth
            self.ai.use_nn = config.use_nn
            if config.use_nn and config.model_path:
                # Завантаження моделі нейромережі
                # Буде реалізовано пізніше
                pass
    
    def make_move(self, move_uci):
        """
        Виконання ходу на дошці
        Повертає True, якщо хід був успішно виконаний
        """
        if self.state.make_move(move_uci):
            # Відправляємо сигнал про виконаний хід
            self.move_executed.emit(move_uci)
            
            # Оновлюємо стан гри
            self.update_game_state()
            
            # Перевіряємо чи гра продовжується
            if not self.state.is_game_over():
                # Якщо це режим PvAI і зараз ходить AI
                if (self.mode_manager.current_mode == GameMode.PvAI and 
                    self.mode_manager.is_ai_turn(self.state.get_turn())):
                    # Запускаємо генерацію ходу AI
                    self.request_ai_move()
            
            return True
        return False
    
    def request_ai_move(self):
        """Запит на генерацію ходу AI"""
        if self.ai_thinking:
            return
            
        self.ai_thinking = True
        fen = self.state.get_fen()
        self.ai_move_requested.emit(fen)
    
    def generate_ai_move(self, fen):
        """Генерація ходу AI (виконується в окремому потоці)"""
        if not self.ai:
            return
            
        try:
            # Генеруємо хід за допомогою AI
            move_uci = self.ai.get_move(fen)
            
            # Перевіряємо чи хід дійсний
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                # Якщо хід недійсний, генеруємо випадковий хід
                move = list(board.legal_moves)[0]
                move_uci = move.uci()
            
            # Відправляємо результат
            self.ai_move_generated.emit(move_uci)
        except Exception as e:
            print(f"Помилка генерації ходу AI: {e}")
            # У разі помилки робимо перший доступний хід
            board = chess.Board(fen)
            if board.legal_moves:
                move = list(board.legal_moves)[0]
                self.ai_move_generated.emit(move.uci())
        finally:
            self.ai_thinking = False
    
    def handle_ai_move(self, move_uci):
        """Обробка згенерованого ходу AI"""
        if not self.ai_thinking:
            return
            
        # Виконуємо хід AI
        self.make_move(move_uci)
        self.ai_thinking = False
    
    def update_game_state(self):
        """Оновлення та сповіщення про зміни стану гри"""
        state = self.state.get_game_state()
        result = self.state.get_result()
        
        if state == GAME_STATE_CHECKMATE:
            winner = "чорні" if self.state.get_turn() == chess.WHITE else "білі"
            message = f"{UI_TEXTS['game_over']} - {UI_TEXTS['checkmate_white' if winner == 'чорні' else 'checkmate_black']}"
            self.game_state_changed.emit(message)
            self.game_over.emit(result)
        elif state == GAME_STATE_STALEMATE:
            self.game_state_changed.emit(UI_TEXTS["stalemate"])
            self.game_over.emit(result)
        elif state == GAME_STATE_DRAW:
            self.game_state_changed.emit(UI_TEXTS["draw"])
            self.game_over.emit(result)
        elif state == "check":
            self.game_state_changed.emit(UI_TEXTS["check"])
        else:
            turn_text = UI_TEXTS["turn_white"] if self.state.get_turn() == chess.WHITE else UI_TEXTS["turn_black"]
            self.game_state_changed.emit(turn_text)
    
    def reset(self, fen=None):
        """Скидання гри до початкового стану або заданої FEN позиції"""
        self.state.reset(fen)
        self.ai_thinking = False
        self.ai_move_queue = None
        self.update_game_state()
    
    def get_current_fen(self):
        """Повертає поточну позицію у форматі FEN"""
        return self.state.get_fen()
    
    def is_game_over(self):
        """Перевіряє чи гра завершена"""
        return self.state.is_game_over()
    
    def handle_promotion(self, move_uci):
        """
        Обробка перетворення пішака
        :param move_uci: Базовий хід без вказання фігури (наприклад, "a7a8")
        """
        # Створюємо об'єкт ходу
        move = chess.Move.from_uci(move_uci)
        
        # Визначаємо колір гравця
        color = self.state.get_turn()
        
        # Відправляємо сигнал для відображення діалогу
        self.promotion_required.emit(move_uci, color)
    
    def complete_promotion(self, move_uci, piece_symbol):
        """
        Завершення перетворення пішака з обраною фігурою
        :param move_uci: Повний хід з фігурою (наприклад, "a7a8q")
        """
        self.make_move(move_uci)
    
    def get_legal_moves(self, square=None):
        """Повертає список легальних ходів для заданої клітинки"""
        return self.state.get_legal_moves(square)
    
    def get_board_state(self):
        """Повертає об'єкт поточного стану дошки"""
        return self.state.board
    
    def get_turn(self):
        """Повертає чия зараз черга ходити"""
        return self.state.get_turn()
    
    def get_last_move(self):
        """Повертає останній виконаний хід"""
        return self.state.get_last_move()
    
    def get_game_result(self):
        """Повертає результат гри, якщо гра завершена"""
        return self.state.get_result()