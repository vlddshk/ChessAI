from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QIcon, QColor
from PyQt5.QtWidgets import QLabel, QWidget
from constants import (
    PIECES_DIR,
    PIECE_IMAGES,
    SQUARE_SIZE,
    PIECE_SIZE,
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
    def __init__(self):
        """
        Ініціалізує рендерер фігур, завантажуючи всі зображення фігур
        """
        self.pieces = {}
        self.load_pieces()
        
        # Кеш для масштабованих зображень
        self.scaled_pieces = {}
    
    def load_pieces(self):
        """Завантажує всі зображення фігур з файлів"""
        for symbol, filename in PIECE_IMAGES.items():
            path = os.path.join(PIECES_DIR, filename)
            if os.path.exists(path):
                self.pieces[symbol] = QPixmap(path)
    
    def get_piece_pixmap(self, symbol, size=PIECE_SIZE):
        """
        Повертає масштабоване зображення фігури
        Використовує кеш для зменшення навантаження
        """
        cache_key = f"{symbol}_{size}"
        if cache_key in self.scaled_pieces:
            return self.scaled_pieces[cache_key]
        
        if symbol in self.pieces:
            pixmap = self.pieces[symbol].scaled(
                size, size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.scaled_pieces[cache_key] = pixmap
            return pixmap
        return None

class ChessSquare(QLabel):
    def __init__(self, square, size=SQUARE_SIZE, parent=None):
        """
        Віджет окремої клітинки шахової дошки
        :param square: Шахова клітинка (chess.Square)
        :param size: Розмір клітинки
        :param parent: Батьківський віджет
        """
        super().__init__(parent)
        self.square = square
        self.size = size
        self.setFixedSize(size, size)
        
        # Визначення кольору клітинки
        rank, file = chess.square_rank(square), chess.square_file(square)
        self.is_dark = (rank + file) % 2 == 1
        self.base_color = DARK_SQUARE if self.is_dark else LIGHT_SQUARE
        
        # Стани клітинки
        self.selected = False
        self.highlighted = False
        self.last_move = False
        self.in_check = False
        self.possible_move = False
        
        self.piece = None
        self.piece_symbol = None
    
    def set_piece(self, piece_symbol):
        """Встановлює фігуру на клітинку"""
        self.piece_symbol = piece_symbol
        self.piece = piece_symbol if piece_symbol else None
        self.update()
    
    def set_selected(self, selected):
        """Встановлює стан виділення клітинки"""
        self.selected = selected
        self.update()
    
    def set_highlighted(self, highlighted):
        """Встановлює стан підсвічування клітинки"""
        self.highlighted = highlighted
        self.update()
    
    def set_last_move(self, last_move):
        """Встановлює стан "останній хід" для клітинки"""
        self.last_move = last_move
        self.update()
    
    def set_in_check(self, in_check):
        """Встановлює стан "шах" для клітинки"""
        self.in_check = in_check
        self.update()
    
    def set_possible_move(self, possible_move):
        """Встановлює стан "можливий хід" для клітинки"""
        self.possible_move = possible_move
        self.update()
    
    def paintEvent(self, event):
        """Відображення клітинки та фігури"""
        painter = QPainter(self)
        
        # Малюємо фон клітинки
        color = QColor(self.base_color)
        
        # Додаткові ефекти в залежності від стану
        if self.selected:
            color = QColor(SELECTED_COLOR)
        elif self.last_move:
            color = QColor(LAST_MOVE_COLOR)
        elif self.in_check:
            color = QColor(CHECK_COLOR)
        elif self.possible_move:
            color = QColor(POSSIBLE_MOVE_COLOR)
        
        painter.fillRect(0, 0, self.size, self.size, color)
        
        # Малюємо фігуру, якщо вона є
        if self.piece_symbol:
            renderer = PieceRenderer()
            pixmap = renderer.get_piece_pixmap(self.piece_symbol)
            if pixmap:
                # Центруємо фігуру на клітинці
                x = (self.size - pixmap.width()) // 2
                y = (self.size - pixmap.height()) // 2
                painter.drawPixmap(x, y, pixmap)

class PieceWidget(QWidget):
    def __init__(self, symbol, size=PIECE_SIZE, parent=None):
        """
        Віджет для відображення окремої фігури
        Використовується в інтерфейсі вибору фігури при перетворенні пішака
        """
        super().__init__(parent)
        self.symbol = symbol
        self.size = size
        self.setFixedSize(size, size)
        self.renderer = PieceRenderer()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = self.renderer.get_piece_pixmap(self.symbol, self.size)
        if pixmap:
            painter.drawPixmap(0, 0, pixmap)

class PiecePromotionDialog(QWidget):
    def __init__(self, color, parent=None):
        """
        Діалогове вікно для вибору фігури при перетворенні пішака
        :param color: Колір фігур (chess.WHITE або chess.BLACK)
        """
        super().__init__(parent, Qt.Popup)
        self.setWindowTitle("Оберіть фігуру")
        self.color = color
        self.selected_piece = None
        self.init_ui()
    
    def init_ui(self):
        """Ініціалізація інтерфейсу діалогу"""
        from PyQt5.QtWidgets import QHBoxLayout
        
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Визначення фігур для вибору
        pieces = ['q', 'r', 'b', 'n'] if self.color == chess.BLACK else ['Q', 'R', 'B', 'N']
        
        for piece in pieces:
            piece_widget = PieceWidget(piece)
            piece_widget.mousePressEvent = lambda event, p=piece: self.on_piece_selected(p)
            layout.addWidget(piece_widget)
        
        self.setLayout(layout)
        self.adjustSize()
    
    def on_piece_selected(self, piece):
        """Обробка вибору фігури"""
        self.selected_piece = piece
        self.close()
    
    def get_selected_piece(self):
        """Повертає обрану фігуру"""
        return self.selected_piece

def get_piece_icon(symbol, size=32):
    """
    Повертає іконку фігури для використання в інтерфейсі
    :param symbol: Символ фігури (наприклад, 'K', 'q')
    :param size: Розмір іконки
    :return: QIcon
    """
    renderer = PieceRenderer()
    pixmap = renderer.get_piece_pixmap(symbol, size)
    if pixmap:
        return QIcon(pixmap)
    return QIcon()