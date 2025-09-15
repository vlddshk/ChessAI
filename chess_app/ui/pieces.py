# chess_app/ui/pieces.py

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QPainter, QIcon, QColor, QGuiApplication
from PyQt5.QtWidgets import QLabel, QWidget, QDialog, QHBoxLayout, QVBoxLayout, QPushButton
from constants import (
    PIECES_DIR,
    PIECE_IMAGES,
    DARK_SQUARE,
    LIGHT_SQUARE,
    SELECTED_COLOR,
    POSSIBLE_MOVE_COLOR,
    CHECK_COLOR,
    LAST_MOVE_COLOR
)
import chess
import os

class PieceRenderer:
    """
    Рендерер для зображень фігур.
    Завантажує зображення один раз і кешує масштабовані варіанти.
    (Увага: QPixmap створюється тільки коли викликається цей клас після створення QApplication)
    """
    def __init__(self):
        self.pieces = {}
        self.scaled_pieces = {}
        self.load_pieces()

    def load_pieces(self):
        """Завантажує оригінальні QPixmap для всіх фігур."""
        for symbol, filename in PIECE_IMAGES.items():
            path = os.path.join(PIECES_DIR, filename)
            if os.path.exists(path):
                self.pieces[symbol] = QPixmap(path)

    def get_piece_pixmap(self, symbol, size: int):
        if not symbol:
            return None
        key = (symbol, int(size))
        if key in self.scaled_pieces:
            return self.scaled_pieces[key]
        pix = self.pieces.get(symbol)
        if pix is None:
            return None
        scaled = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pieces[key] = scaled
        return scaled

_renderer = None

def get_renderer():
    """
    Повертає глобальний PieceRenderer. Якщо він ще не створений — створює його,
    але тільки якщо вже існує QGuiApplication.instance(). Інакше повертає None
    (малювання фігур буде пропущено, доки QApplication не буде створено).
    """
    global _renderer
    if _renderer is None:
        if QGuiApplication.instance() is None:
            return None
        _renderer = PieceRenderer()
    return _renderer

# -------------------- Віджети --------------------

class ChessSquare(QLabel):
    def __init__(self, square: int, parent=None):
        super().__init__(parent)
        self.square = square
        rank, file = chess.square_rank(square), chess.square_file(square)
        self.is_dark = (rank + file) % 2 == 1
        self.base_color = QColor(DARK_SQUARE) if self.is_dark else QColor(LIGHT_SQUARE)

        self.selected = False
        self.highlighted = False
        self.last_move = False
        self.in_check = False
        self.possible_move = False

        self.piece_symbol = None
        self.setMinimumSize(20, 20)

    def set_piece(self, piece_symbol):
        self.piece_symbol = piece_symbol
        self.update()

    def set_selected(self, selected: bool):
        self.selected = bool(selected)
        self.update()

    def set_highlighted(self, highlighted: bool):
        self.highlighted = bool(highlighted)
        self.update()

    def set_last_move(self, last_move: bool):
        self.last_move = bool(last_move)
        self.update()

    def set_in_check(self, in_check: bool):
        self.in_check = bool(in_check)
        self.update()

    def set_possible_move(self, possible_move: bool):
        self.possible_move = bool(possible_move)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = max(1, self.width())
        h = max(1, self.height())
        size = min(w, h)

        bg_color = QColor(self.base_color)
        if self.selected:
            bg_color = QColor(SELECTED_COLOR)
        elif self.last_move:
            bg_color = QColor(LAST_MOVE_COLOR)
        elif self.in_check:
            bg_color = QColor(CHECK_COLOR)
        elif self.possible_move:
            bg_color = QColor(POSSIBLE_MOVE_COLOR)

        painter.fillRect(0, 0, w, h, bg_color)

        # Малюємо фігуру лише якщо рендерер доступний (тобто QApplication створено)
        renderer = get_renderer()
        if renderer and self.piece_symbol:
            piece_target = int(size * 0.85)
            pix = renderer.get_piece_pixmap(self.piece_symbol, piece_target)
            if pix:
                x = (w - pix.width()) // 2
                y = (h - pix.height()) // 2
                painter.drawPixmap(x, y, pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

class PieceWidget(QWidget):
    def __init__(self, symbol, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.setMinimumSize(24, 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        size = min(self.width(), self.height())
        renderer = get_renderer()
        if renderer:
            pix = renderer.get_piece_pixmap(self.symbol, int(size * 0.9))
            if pix:
                x = (self.width() - pix.width()) // 2
                y = (self.height() - pix.height()) // 2
                painter.drawPixmap(x, y, pix)

class PiecePromotionDialog(QDialog):
    def __init__(self, color, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setWindowTitle("Оберіть фігуру")
        self.color = color
        self.selected_piece = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        info_label = QLabel("Оберіть фігуру для перетворення:")
        main_layout.addWidget(info_label)

        pieces_layout = QHBoxLayout()
        pieces_layout.setSpacing(6)

        is_white = (self.color is True) or (self.color == chess.WHITE) or (self.color == 1)
        pieces = ['Q', 'R', 'B', 'N'] if is_white else ['q', 'r', 'b', 'n']

        for sym in pieces:
            btn = QPushButton()
            icon_size = 48
            renderer = get_renderer()
            if renderer:
                pix = renderer.get_piece_pixmap(sym, icon_size)
                if pix:
                    btn.setIcon(QIcon(pix))
                    btn.setIconSize(QSize(pix.width(), pix.height()))
            btn.setFixedSize(icon_size + 12, icon_size + 12)
            btn.clicked.connect(lambda checked, s=sym: self.on_piece_selected(s))
            pieces_layout.addWidget(btn)

        main_layout.addLayout(pieces_layout)
        self.setLayout(main_layout)
        self.adjustSize()

    def on_piece_selected(self, piece):
        self.selected_piece = piece
        self.accept()

    def get_selected_piece(self):
        return self.selected_piece

def get_piece_icon(symbol, size=32):
    renderer = get_renderer()
    if renderer:
        pix = renderer.get_piece_pixmap(symbol, size)
        if pix:
            return QIcon(pix)
    return QIcon()
