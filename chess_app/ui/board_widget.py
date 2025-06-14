from PyQt5.QtWidgets import QWidget, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont
from .pieces import ChessSquare, PiecePromotionDialog
from game.state import ChessState
from constants import (
    SQUARE_SIZE, BOARD_SIZE, 
    SELECTED_COLOR, POSSIBLE_MOVE_COLOR,
    CHECK_COLOR, LAST_MOVE_COLOR,
    UI_TEXTS, GAME_STATE_CHECKMATE,
    GAME_STATE_STALEMATE, GAME_STATE_DRAW,
    LIGHT_SQUARE, DARK_SQUARE
)
import chess

class BoardWidget(QWidget):
    # Сигнали
    move_made = pyqtSignal(str)               # Сигнал про виконання ходу (UCI)
    game_state_changed = pyqtSignal(str)      # Сигнал про зміну стану гри
    promotion_required = pyqtSignal(str, chess.Piece)  # Сигнал про необхідність перетворення пішака
    square_selected = pyqtSignal(str)         # Сигнал про вибір клітинки
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(BOARD_SIZE, BOARD_SIZE)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Ініціалізація стану
        self.game_controller = None  # Буде встановлено ззовні
        self.selected_square = None
        self.possible_moves = []
        self.last_move = None
        self.check_square = None
        self.promotion_move = None
        
        # Створення клітинок дошки
        self.squares = {}
        self.init_board()
    
    def init_board(self):
        """Ініціалізація шахової дошки з 64 клітинками"""
        # Розміщення клітинок у сітці
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Перевертаємо ранг для коректного відображення
                square_widget = ChessSquare(square)
                square_widget.mousePressEvent = lambda event, sq=square: self.square_clicked(sq)
                self.squares[square] = square_widget
                layout.addWidget(square_widget, rank, file)
        
        self.setLayout(layout)
    
    def reset_board(self):
        """Скидання дошки до початкового стану"""
        if self.game_controller:
            self.game_controller.reset()
        self.selected_square = None
        self.possible_moves = []
        self.last_move = None
        self.check_square = None
        self.promotion_move = None
        self.update_board()
        self.game_state_changed.emit(UI_TEXTS["turn_white"])
    
    def update_board(self):
        """Оновлення відображення всіх клітинок дошки"""
        if not self.game_controller:
            return
            
        board = self.game_controller.get_board_state()
        
        # Оновлення фігур
        for square in chess.SQUARES:
            piece = board.piece_at(square)
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
            for move in self.possible_moves:
                move_obj = chess.Move.from_uci(move)
                if move_obj.from_square == self.selected_square:
                    self.squares[move_obj.to_square].set_possible_move(True)
        
        # Підсвічування останнього ходу
        if self.last_move:
            move_obj = chess.Move.from_uci(self.last_move)
            self.squares[move_obj.from_square].set_last_move(True)
            self.squares[move_obj.to_square].set_last_move(True)
        
        # Підсвічування шаха
        if self.check_square is not None:
            self.squares[self.check_square].set_in_check(True)
    
    def update_board_state(self, board):
        """Оновлення стану дошки на основі гри"""
        if not self.game_controller:
            return
            
        # Оновлення стану шаха
        if board.is_check():
            turn = board.turn
            king_square = board.king(turn)
            if king_square is not None:
                self.check_square = king_square
        else:
            self.check_square = None
        
        # Оновлення останнього ходу
        if board.move_stack:
            self.last_move = board.move_stack[-1].uci()
        else:
            self.last_move = None
        
        self.update_board()
    
    def square_clicked(self, square):
        """Обробка кліку на клітинці"""
        if not self.game_controller or self.game_controller.is_game_over():
            return
        
        # Повідомляємо про вибір клітинки
        square_name = chess.square_name(square)
        self.square_selected.emit(square_name)
        
        piece = self.game_controller.get_board_state().piece_at(square)
        current_turn = self.game_controller.get_turn()
        
        # Якщо немає обраної клітинки
        if self.selected_square is None:
            # Клік на фігурі гравця
            if piece and piece.color == current_turn:
                self.selected_square = square
                self.possible_moves = self.game_controller.get_legal_moves(square)
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
                if self.is_promotion_move(move_obj):
                    self.promotion_move = move_uci
                    self.show_promotion_dialog(move_obj)
                    return
                
                # Виконання звичайного ходу
                self.make_move(move_uci)
            else:
                # Клік на іншу фігуру гравця - зміна вибору
                if piece and piece.color == current_turn:
                    self.selected_square = square
                    self.possible_moves = self.game_controller.get_legal_moves(square)
                    self.update_board()
                else:
                    # Скасування вибору
                    self.selected_square = None
                    self.possible_moves = []
                    self.update_board()
    
    def is_promotion_move(self, move):
        """Перевіряє чи хід є перетворенням пішака"""
        board = self.game_controller.get_board_state()
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(move.to_square)
            if to_rank in [0, 7]:  # Пішак досяг кінця дошки
                return True
        return False
    
    def make_move(self, move_uci):
        """Виконання ходу та оновлення стану"""
        if self.game_controller.make_move(move_uci):
            self.move_made.emit(move_uci)
            self.selected_square = None
            self.possible_moves = []
            return True
        return False
    
    def show_promotion_dialog(self, move):
        """Відображення діалогу вибору фігури для перетворення"""
        # Визначення кольору гравця
        color = self.game_controller.get_turn()
        
        # Створення діалогу
        dialog = PiecePromotionDialog(color, self)
        
        # Позиціонування діалогу біля місця перетворення
        square_widget = self.squares[move.to_square]
        pos = square_widget.mapToGlobal(QPoint(0, 0))
        dialog.move(pos)
        
        # Відображення діалогу та очікування вибору
        if dialog.exec_() == PiecePromotionDialog.Accepted:
            # Отримання обраної фігури
            selected_piece = dialog.get_selected_piece()
            
            if selected_piece:
                # Формування повного UCI ходу з вказанням фігури
                full_move = move.uci() + selected_piece.lower()
                self.make_move(full_move)
        
        self.promotion_move = None
    
    def set_position_from_fen(self, fen):
        """Встановлення позиції з FEN нотації"""
        if self.game_controller:
            self.game_controller.reset(fen)
            self.update_board_state(self.game_controller.get_board_state())
    
    def get_current_fen(self):
        """Повертає поточну позицію у форматі FEN"""
        if self.game_controller:
            return self.game_controller.get_current_fen()
        return chess.STARTING_FEN
    
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
            # Нижній ряд
            painter.drawText(
                i * SQUARE_SIZE + SQUARE_SIZE // 2 - 5, 
                BOARD_SIZE - 5, 
                char
            )
            # Верхній ряд
            painter.drawText(
                i * SQUARE_SIZE + SQUARE_SIZE // 2 - 5, 
                15, 
                char
            )
        
        # Малювання цифр (рангів)
        ranks = "12345678"
        for i, char in enumerate(ranks):
            # Лівий бік
            painter.drawText(
                5, 
                (7 - i) * SQUARE_SIZE + SQUARE_SIZE // 2 + 5, 
                char
            )
            # Правий бік
            painter.drawText(
                BOARD_SIZE - 15, 
                (7 - i) * SQUARE_SIZE + SQUARE_SIZE // 2 + 5, 
                char
            )
    
    def draw_border(self, painter):
        """Малювання рамки навколо дошки"""
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(0, 0, BOARD_SIZE, BOARD_SIZE)
    
    def highlight_move(self, move_uci):
        """Підсвічування конкретного ходу на дошці"""
        try:
            move = chess.Move.from_uci(move_uci)
            self.squares[move.from_square].set_highlighted(True)
            self.squares[move.to_square].set_highlighted(True)
            self.update()
        except Exception as e:
            print(f"Error highlighting move: {e}")