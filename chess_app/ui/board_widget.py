
from PyQt5.QtWidgets import QWidget, QGridLayout, QDialog
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPen
from .pieces import ChessSquare, PiecePromotionDialog
from constants import UI_TEXTS, calculate_dimensions
import chess

class BoardWidget(QWidget):
    # Сигнали
    move_made = pyqtSignal(str)               
    game_state_changed = pyqtSignal(str)      
    promotion_required = pyqtSignal(str, object)  
    square_selected = pyqtSignal(str)       

    def __init__(self, parent=None):
        super().__init__(parent)

        # Початковий (fallback) розмір клітинки — буде перерахований у resizeEvent
        self.square_size = 64 #defaul 64

        # Ініціалізація стану
        self.game_controller = None  
        self.selected_square = None
        self.possible_moves = []
        self.last_move = None
        self.check_square = None
        self.promotion_move = None

        # Створення клітинок дошки
        self.squares = {}
        self.init_board()

        # Мінімальний розмір, щоб віджет не ставав надто малим
        self.setMinimumSize(8 * self.square_size, 8 * self.square_size)
        self.setFocusPolicy(Qt.StrongFocus)

    def init_board(self):
        """Ініціалізація шахової дошки з 64 клітинками"""
        # Розміщення клітинок у сітці
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(70, 0, 0, 80)#0

        # Якщо вже існували виджети — очистимо
        for w in list(self.squares.values()):
            w.setParent(None)
        self.squares.clear()

        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Перевертаємо ранг для коректного відображення
                square_widget = ChessSquare(square)
                # встановлюємо попередній мін/фіксований розмір — буде оновлено в resizeEvent
                square_widget.setMinimumSize(20, 20)
                square_widget.mousePressEvent = lambda event, sq=square: self.square_clicked(sq)
                self.squares[square] = square_widget
                layout.addWidget(square_widget, rank, file)

        self.setLayout(layout)

    def resizeEvent(self, event):
        """Перерахунок розміру клітинок при зміні розміру віджета"""
        # використовуємо фактичний розмір самого віджета (квадратна дошка займає мін(m.width,m.height))
        board_pixel = min(self.width(), self.height())
        if board_pixel <= 0:
            return super().resizeEvent(event)

        new_square = board_pixel // 8
        if new_square <= 0:
            return super().resizeEvent(event)

        if new_square != self.square_size:
            self.square_size = new_square
            # Оновлюємо розміри всіх клітинок
            for sq_widget in self.squares.values():
                sq_widget.setFixedSize(self.square_size, self.square_size)
            self.update()  

        super().resizeEvent(event)

    def reset_board(self):
        """Оновлення відображення після скидання стану контролера.
       Тепер контролер повинен робити reset() сам (наприклад з MainWindow).
        """
        self.selected_square = None
        self.possible_moves = []
        self.last_move = None
        self.check_square = None
        self.promotion_move = None

        if self.game_controller:
        # Попросимо контролер повернути поточний board state і відмалькуємо його
            try:
                self.update_board_state(self.game_controller.get_board_state())
            except Exception:
                self.update_board()  
        else:
            self.update_board()
        self.game_state_changed.emit(UI_TEXTS.get("turn_white", "White to move"))

    def update_board(self):
        """Оновлення відображення всіх клітинок дошки"""
        if not self.game_controller:
            return

        board = self.game_controller.get_board_state()

        # Оновлення фігур
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            symbol = piece.symbol() if piece else None
            if square in self.squares:
                self.squares[square].set_piece(symbol)

        # Скидання всіх виділень
        for square_widget in self.squares.values():
            square_widget.set_selected(False)
            square_widget.set_highlighted(False)
            square_widget.set_last_move(False)
            square_widget.set_in_check(False)
            square_widget.set_possible_move(False)

        # Виділення обраної клітинки
        if self.selected_square is not None and self.selected_square in self.squares:
            self.squares[self.selected_square].set_selected(True)

            # Підсвічування можливих ходів
            for move in self.possible_moves:
                try:
                    move_obj = chess.Move.from_uci(move)
                    if move_obj.from_square == self.selected_square:
                        self.squares[move_obj.to_square].set_possible_move(True)
                except Exception:
                    continue

        # Підсвічування останнього ходу
        if self.last_move:
            try:
                move_obj = chess.Move.from_uci(self.last_move)
                if move_obj.from_square in self.squares:
                    self.squares[move_obj.from_square].set_last_move(True)
                if move_obj.to_square in self.squares:
                    self.squares[move_obj.to_square].set_last_move(True)
            except Exception:
                pass

        # Підсвічування шаха
        if self.check_square is not None and self.check_square in self.squares:
            self.squares[self.check_square].set_in_check(True)

        # Перемалювати віджет
        self.update()

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
        """Обробка кліку на клітинці (square — int 0..63)"""
        if not self.game_controller or self.game_controller.is_game_over():
            return

        # Повідомляємо про вибір клітинки (назва в форматі 'e2')
        square_name = chess.square_name(square)
        self.square_selected.emit(square_name)

        board = self.game_controller.get_board_state()
        piece = board.piece_at(square)
        current_turn = self.game_controller.get_turn()

        # Якщо немає обраної клітинки
        if self.selected_square is None:
            # Клік на фігурі гравця
            if piece and piece.color == current_turn:
                self.selected_square = square
                self.possible_moves = self.game_controller.get_legal_moves(square)
                self.update_board()
            return

        # Якщо вже є обрана клітинка
        move_uci = None

        # Перевірка чи це хід на обрану клітинку (скасування вибору)
        if square == self.selected_square:
            self.selected_square = None
            self.possible_moves = []
            self.update_board()
            return

        # Перевірка чи це можливий хід
        for move in self.possible_moves:
            try:
                m = chess.Move.from_uci(move)
                if m.to_square == square:
                    move_uci = move
                    break
            except Exception:
                continue

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
        if not self.game_controller:
            return False
        board = self.game_controller.get_board_state()
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(move.to_square)
            if to_rank in [0, 7]:  # Пішак досяг кінця дошки
                return True
        return False

    def make_move(self, move_uci):
        """Не виконуємо хід самостійно — тільки повідомляємо про намір."""
        # Емітуємо сигнал для зовнішнього контролю (MainWindow -> GameController)
        self.move_made.emit(move_uci)

        # Скидаємо вибір і підсвічування у UI (UI оновиться після відповіді контролера)
        self.selected_square = None
        self.possible_moves = []
        return True

    def show_promotion_dialog(self, move):
        """Відображення діалогу вибору фігури для перетворення (move — chess.Move)"""
        if not self.game_controller:
            return

        # Визначення кольору гравця (True для white, False для black)
        color = self.game_controller.get_turn()

        # Створення діалогу
        dialog = PiecePromotionDialog(color, self)

        # Позиціонування діалогу біля місця перетворення (центруємо відносно клітинки)
        square_widget = self.squares.get(move.to_square)
        if square_widget:
            pos = square_widget.mapToGlobal(QPoint(0, 0))
            try:
                offset = QPoint((self.square_size - dialog.width()) // 2, 
                                (self.square_size - dialog.height()) // 2)
                dialog.move(pos + offset)
            except Exception:
                dialog.move(pos)

        if dialog.exec_() == QDialog.Accepted:
            selected_piece = dialog.get_selected_piece()
            if selected_piece:
                promo_char = selected_piece.lower()[0] # без складних mapping
                base_uci = move.uci()[:4]
                if self.game_controller.complete_promotion(base_uci, promo_char):
                    self.update_board_state(self.game_controller.get_board_state())
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
        # Розмір шрифту залежить від розміру клітинки
        font.setPointSize(max(8, self.square_size // 5))
        painter.setFont(font)

        # Малювання букв (файлів)
        files = "abcdefgh"
        board_pixel_size = self.square_size * 8
        for i, char in enumerate(files):
            painter.drawText(
                i * self.square_size + self.square_size // 2 - 5,
                board_pixel_size - 3,
                char
            )
            painter.drawText(
                i * self.square_size + self.square_size // 2 - 5,
                font.pointSize() + 2,
                char
            )

        # Малювання цифр (рангів)
        ranks = "12345678"
        for i, char in enumerate(ranks):
            painter.drawText(
                3,
                (7 - i) * self.square_size + self.square_size // 2 + 5,
                char
            )
            painter.drawText(
                board_pixel_size - 12,
                (7 - i) * self.square_size + self.square_size // 2 + 5,
                char
            )

    def draw_border(self, painter):
        """Малювання рамки навколо дошки"""
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        board_pixel_size = self.square_size * 8
        painter.drawRect(0, 0, board_pixel_size, board_pixel_size)

    def highlight_move(self, move_uci):
        """Підсвічування конкретного ходу на дошці"""
        try:
            move = chess.Move.from_uci(move_uci)
            if move.from_square in self.squares:
                self.squares[move.from_square].set_highlighted(True)
            if move.to_square in self.squares:
                self.squares[move.to_square].set_highlighted(True)
            self.update()
        except Exception as e:
            print(f"Error highlighting move: {e}")
