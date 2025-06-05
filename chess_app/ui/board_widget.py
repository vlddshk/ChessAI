from PyQt5.QtWidgets import QWidget, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
from .pieces import ChessSquare, PiecePromotionDialog
from game.state import ChessState
from constants import (
    SQUARE_SIZE, BOARD_SIZE, 
    SELECTED_COLOR, POSSIBLE_MOVE_COLOR,
    CHECK_COLOR, LAST_MOVE_COLOR,
    UI_TEXTS, GAME_STATE_CHECKMATE,
    GAME_STATE_STALEMATE, GAME_STATE_DRAW
)
import chess

class BoardWidget(QWidget):
    # Сигнали
    move_made = pyqtSignal(str)         # Сигнал про виконання ходу (UCI)
    game_state_changed = pyqtSignal(str)  # Сигнал про зміну стану гри
    promotion_required = pyqtSignal(str, chess.Piece)  # Сигнал про необхідність перетворення пішака
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(BOARD_SIZE, BOARD_SIZE)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Ініціалізація стану
        self.game_state = ChessState()
        self.selected_square = None
        self.possible_moves = []
        self.last_move_squares = []
        self.check_square = None
        self.promotion_move = None
        
        # Створення клітинок дошки
        self.squares = {}
        self.init_board()
        
        # Підключення сигналів
        self.game_state.game_state_changed = self.update_board_state
        
    def init_board(self):
        """Ініціалізація шахової дошки з 64 клітинками"""
        for square in chess.SQUARES:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            square_widget = ChessSquare(square)
            square_widget.mousePressEvent = lambda event, sq=square: self.square_clicked(sq)
            self.squares[square] = square_widget
        
        # Розміщення клітинок у сітці
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        for square in chess.SQUARES:
            rank = 7 - chess.square_rank(square)  # Перевертаємо ранг для коректного відображення
            file = chess.square_file(square)
            layout.addWidget(self.squares[square], rank, file)
        
        self.setLayout(layout)
        self.update_board()
    
    def reset_board(self, fen=None):
        """Скидання дошки до початкового стану або заданої FEN позиції"""
        self.game_state.reset(fen)
        self.selected_square = None
        self.possible_moves = []
        self.last_move_squares = []
        self.check_square = None
        self.promotion_move = None
        self.update_board()
        self.game_state_changed.emit(UI_TEXTS["turn_white"])
    
    def update_board(self):
        """Оновлення відображення всіх клітинок дошки"""
        # Оновлення фігур
        for square in chess.SQUARES:
            piece = self.game_state.get_piece_at(square)
            symbol = piece.symbol() if piece else None
            self.squares[square].set_piece(symbol)
        
        # Скидання всіх виділень
        for square in self.squares.values():
            square.set_selected(False)
            square.set_highlighted(False)
            square.set_last_move(False)
            square.set_in_check(False)
            square.set_possible_move(False)
        
        # Виділення обраної клітинки
        if self.selected_square is not None:
            self.squares[self.selected_square].set_selected(True)
            
            # Підсвічування можливих ходів
            for move_uci in self.possible_moves:
                move = chess.Move.from_uci(move_uci)
                if move.from_square == self.selected_square:
                    self.squares[move.to_square].set_possible_move(True)
        
        # Підсвічування останнього ходу
        last_move = self.game_state.get_last_move()
        if last_move:
            move = chess.Move.from_uci(last_move)
            self.squares[move.from_square].set_last_move(True)
            self.squares[move.to_square].set_last_move(True)
        
        # Підсвічування шаха
        if self.check_square is not None:
            self.squares[self.check_square].set_in_check(True)
    
    def update_board_state(self):
        """Оновлення стану дошки на основі гри"""
        # Перевірка шаха
        if self.game_state.get_game_state() == "check":
            turn = self.game_state.get_turn()
            self.check_square = self.game_state.get_king_square(turn)
        else:
            self.check_square = None
        
        # Оновлення текстового стану
        state = self.game_state.get_game_state()
        if state == GAME_STATE_CHECKMATE:
            winner = "чорні" if self.game_state.get_turn() == chess.WHITE else "білі"
            self.game_state_changed.emit(f"{UI_TEXTS['game_over']} - {UI_TEXTS['checkmate_white' if winner == 'чорні' else 'checkmate_black']}")
        elif state == GAME_STATE_STALEMATE:
            self.game_state_changed.emit(UI_TEXTS["stalemate"])
        elif state == GAME_STATE_DRAW:
            self.game_state_changed.emit(UI_TEXTS["draw"])
        elif state == "check":
            self.game_state_changed.emit(UI_TEXTS["check"])
        else:
            turn_text = UI_TEXTS["turn_white"] if self.game_state.get_turn() == chess.WHITE else UI_TEXTS["turn_black"]
            self.game_state_changed.emit(turn_text)
        
        self.update_board()
    
    def square_clicked(self, square):
        """Обробка кліку на клітинці"""
        if self.game_state.is_game_over():
            return
        
        piece = self.game_state.get_piece_at(square)
        
        # Якщо немає обраної клітинки
        if self.selected_square is None:
            # Клік на фігурі гравця
            if piece and piece.color == self.game_state.get_turn():
                self.selected_square = square
                self.possible_moves = self.game_state.get_legal_moves(square)
                self.update_board()
        
        # Якщо вже є обрана клітинка
        else:
            move_uci = None
            
            # Перевірка чи це хід на обрану клітинку (скасування вибору)
            if square == self.selected_square:
                self.selected_square = None
                self.possible_moves = []
                self.update_board()
                return
            
            # Перевірка чи це можливий хід
            for move in self.possible_moves:
                m = chess.Move.from_uci(move)
                if m.to_square == square:
                    move_uci = move
                    break
            
            # Якщо знайдено дійсний хід
            if move_uci:
                move_obj = chess.Move.from_uci(move_uci)
                
                # Перевірка чи це перетворення пішака
                if move_obj.promotion is not None:
                    self.promotion_move = move_uci
                    self.show_promotion_dialog(move_obj)
                    return
                
                # Виконання звичайного ходу
                self.make_move(move_uci)
            else:
                # Клік на іншу фігуру гравця - зміна вибору
                if piece and piece.color == self.game_state.get_turn():
                    self.selected_square = square
                    self.possible_moves = self.game_state.get_legal_moves(square)
                    self.update_board()
                else:
                    # Скасування вибору
                    self.selected_square = None
                    self.possible_moves = []
                    self.update_board()
    
    def make_move(self, move_uci):
        """Виконання ходу та оновлення стану"""
        if self.game_state.make_move(move_uci):
            self.move_made.emit(move_uci)
            self.selected_square = None
            self.possible_moves = []
            self.update_board_state()
            return True
        return False
    
    def show_promotion_dialog(self, move):
        """Відображення діалогу вибору фігури для перетворення"""
        # Визначення кольору гравця
        color = self.game_state.get_turn()
        
        # Створення діалогу
        dialog = PiecePromotionDialog(color, self)
        
        # Позиціонування діалогу біля місця перетворення
        square_widget = self.squares[move.to_square]
        pos = square_widget.mapToGlobal(QPoint(0, 0))
        dialog.move(pos)
        
        # Відображення діалогу та очікування вибору
        dialog.exec_()
        
        # Отримання обраної фігури
        selected_piece = dialog.get_selected_piece()
        
        if selected_piece:
            # Формування повного UCI ходу з вказанням фігури
            full_move = move.uci() + selected_piece.lower()
            self.make_move(full_move)
        
        self.promotion_move = None
    
    def set_position_from_fen(self, fen):
        """Встановлення позиції з FEN нотації"""
        self.reset_board(fen)
    
    def get_current_fen(self):
        """Повертає поточну позицію у форматі FEN"""
        return self.game_state.get_fen()
    
    def paintEvent(self, event):
        """Додаткове малювання поверх дошки (рамка, координати)"""
        super().paintEvent(event)
        painter = QPainter(self)
        self.draw_coordinates(painter)
        self.draw_border(painter)
    
    def draw_coordinates(self, painter):
        """Малювання координат дошки"""
        painter.setPen(Qt.black)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        
        # Малювання букв (файлів)
        files = "abcdefgh"
        for i, char in enumerate(files):
            painter.drawText(
                i * SQUARE_SIZE + SQUARE_SIZE // 2 - 5, 
                BOARD_SIZE - 5, 
                char
            )
            painter.drawText(
                i * SQUARE_SIZE + SQUARE_SIZE // 2 - 5, 
                15, 
                char
            )
        
        # Малювання цифр (рангів)
        ranks = "12345678"
        for i, char in enumerate(ranks):
            painter.drawText(
                5, 
                (7 - i) * SQUARE_SIZE + SQUARE_SIZE // 2 + 5, 
                char
            )
            painter.drawText(
                BOARD_SIZE - 15, 
                (7 - i) * SQUARE_SIZE + SQUARE_SIZE // 2 + 5, 
                char
            )
    
    def draw_border(self, painter):
        """Малювання рамки навколо дошки"""
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        painter.drawRect(0, 0, BOARD_SIZE, BOARD_SIZE)
    
    def handle_ai_move(self, move_uci):
        """Обробка ходу, зробленого AI"""
        self.make_move(move_uci)
    
    def highlight_move(self, move_uci):
        """Підсвічування конкретного ходу на дошці"""
        move = chess.Move.from_uci(move_uci)
        self.squares[move.from_square].set_highlighted(True)
        self.squares[move.to_square].set_highlighted(True)
        self.update()