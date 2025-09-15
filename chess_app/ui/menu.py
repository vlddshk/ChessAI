from PyQt5.QtWidgets import (
    QMenuBar, QAction, QActionGroup, 
    QWidget, QHBoxLayout, QLabel, QComboBox
)
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal
import chess
from constants import UI_TEXTS
from game.mode import GameMode
from utils.position import get_piece_icon

class ChessMenuBar(QMenuBar):
    # Сигнали
    new_game_requested = pyqtSignal()
    restart_game_requested = pyqtSignal()
    save_game_requested = pyqtSignal()
    load_game_requested = pyqtSignal()
    exit_requested = pyqtSignal()
    game_mode_changed = pyqtSignal(GameMode)
    difficulty_changed = pyqtSignal(str)
    about_requested = pyqtSignal()
    rules_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ChessMenuBar")
        self.init_menus()
        self.setup_connections()
    
    def init_menus(self):
        """Ініціалізація всіх пунктів меню"""
        # Меню "Гра"
        self.game_menu = self.addMenu(UI_TEXTS["game_menu"])
        
        self.new_game_action = self.create_action(
            UI_TEXTS["new_game"], 
            "new_game",
            QKeySequence.New,
            UI_TEXTS["new_game_tooltip"]
        )
        self.restart_action = self.create_action(
            UI_TEXTS["restart_game"], 
            "restart",
            QKeySequence("Ctrl+R"),
            UI_TEXTS["restart_game_tooltip"]
        )
        self.save_action = self.create_action(
            UI_TEXTS["save_game"], 
            "save",
            QKeySequence.Save,
            UI_TEXTS["save_game_tooltip"]
        )
        self.load_action = self.create_action(
            UI_TEXTS["load_game"], 
            "load",
            QKeySequence.Open,
            UI_TEXTS["load_game_tooltip"]
        )
        self.exit_action = self.create_action(
            UI_TEXTS["exit"], 
            "exit",
            QKeySequence.Quit,
            UI_TEXTS["exit_tooltip"]
        )
        
        self.game_menu.addAction(self.new_game_action)
        self.game_menu.addAction(self.restart_action)
        self.game_menu.addSeparator()
        self.game_menu.addAction(self.save_action)
        self.game_menu.addAction(self.load_action)
        self.game_menu.addSeparator()
        self.game_menu.addAction(self.exit_action)
        
        # Меню "Налаштування"
        self.settings_menu = self.addMenu(UI_TEXTS["settings_menu"])
        
        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.setExclusive(True)
        
        self.pvp_action = self.create_action(
            UI_TEXTS["pvp_mode"], 
            "pvp",
            checkable=True,
            checked=True,
            data=GameMode.PvP
        )
        self.pvai_action = self.create_action(
            UI_TEXTS["pvai_mode"], 
            "pvai",
            checkable=True,
            data=GameMode.PvAI
        )
        
        self.mode_action_group.addAction(self.pvp_action)
        self.mode_action_group.addAction(self.pvai_action)
        
        self.settings_menu.addAction(self.pvp_action)
        self.settings_menu.addAction(self.pvai_action)
        self.settings_menu.addSeparator()
        
        self.difficulty_menu = self.settings_menu.addMenu(UI_TEXTS["difficulty"])
        
        self.difficulty_action_group = QActionGroup(self)
        self.difficulty_action_group.setExclusive(True)
        
        self.easy_action = self.create_action(
            UI_TEXTS["easy"], 
            "easy",
            checkable=True,
            checked=True,
            data="easy"
        )
        self.medium_action = self.create_action(
            UI_TEXTS["medium"], 
            "medium",
            checkable=True,
            data="medium"
        )
        self.hard_action = self.create_action(
            UI_TEXTS["hard"], 
            "hard",
            checkable=True,
            data="hard"
        )
        
        self.difficulty_action_group.addAction(self.easy_action)
        self.difficulty_action_group.addAction(self.medium_action)
        self.difficulty_action_group.addAction(self.hard_action)
        
        self.difficulty_menu.addAction(self.easy_action)
        self.difficulty_menu.addAction(self.medium_action)
        self.difficulty_menu.addAction(self.hard_action)
        
        # Меню "Допомога"
        self.help_menu = self.addMenu(UI_TEXTS["help_menu"])
        
        self.about_action = self.create_action(
            UI_TEXTS["about"], 
            "about"
        )
        self.rules_action = self.create_action(
            UI_TEXTS["rules"], 
            "rules"
        )
        
        self.help_menu.addAction(self.about_action)
        self.help_menu.addAction(self.rules_action)
    
    def create_action(self, text, object_name, shortcut=None, tooltip=None, 
                     icon=None, checkable=False, checked=False, data=None):
        """Допоміжна функція для створення дій меню"""
        action = QAction(text, self)
        action.setObjectName(object_name)
        
        if shortcut:
            action.setShortcut(shortcut)
        
        if tooltip:
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
        
        if icon:
            action.setIcon(QIcon(icon))
        
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
        
        if data:
            action.setData(data)
        
        return action
    
    def setup_connections(self):
        """Налаштування з'єднань сигналів та слотів"""
        # Меню "Гра"
        self.new_game_action.triggered.connect(self.new_game_requested.emit)
        self.restart_action.triggered.connect(self.restart_game_requested.emit)
        self.save_action.triggered.connect(self.save_game_requested.emit)
        self.load_action.triggered.connect(self.load_game_requested.emit)
        self.exit_action.triggered.connect(self.exit_requested.emit)
        
        # Меню "Налаштування"
        self.pvp_action.triggered.connect(lambda: self.game_mode_changed.emit(GameMode.PvP))
        self.pvai_action.triggered.connect(lambda: self.game_mode_changed.emit(GameMode.PvAI))
        
        self.easy_action.triggered.connect(lambda: self.difficulty_changed.emit("easy"))
        self.medium_action.triggered.connect(lambda: self.difficulty_changed.emit("medium"))
        self.hard_action.triggered.connect(lambda: self.difficulty_changed.emit("hard"))
        
        # Меню "Допомога"
        self.about_action.triggered.connect(self.about_requested.emit)
        self.rules_action.triggered.connect(self.rules_requested.emit)
    
    def set_current_mode(self, mode):
        """Встановлює поточний режим гри"""
        if mode == GameMode.PvP:
            self.pvp_action.setChecked(True)
        else:
            self.pvai_action.setChecked(True)
    
    def set_current_difficulty(self, difficulty):
        """Встановлює поточний рівень складності"""
        for action in self.difficulty_action_group.actions():
            if action.data() == difficulty:
                action.setChecked(True)
                break
    
    def update_difficulty_menu_visibility(self, visible):
        """Показує або приховує меню складності"""
        self.difficulty_menu.menuAction().setVisible(visible)
    
    def apply_styles(self, style_sheet):
        """Застосовує стилі до меню"""
        self.setStyleSheet(style_sheet)

class GameModeSelector(QWidget):
    """Віджет для швидкого вибору режиму гри та складності"""
    mode_changed = pyqtSignal(GameMode)
    difficulty_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        
        # Мітка для вибору режиму
        mode_label = QLabel(UI_TEXTS["game_mode"] + ":")
        layout.addWidget(mode_label)
        
        # Випадаючий список режимів
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(UI_TEXTS["pvp_mode"], GameMode.PvP)
        self.mode_combo.addItem(UI_TEXTS["pvai_mode"], GameMode.PvAI)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo, 1)
        
        # Мітка для складності
        self.difficulty_label = QLabel(UI_TEXTS["difficulty"] + ":")
        layout.addWidget(self.difficulty_label)
        
        # Випадаючий список складності
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItem(UI_TEXTS["easy"], "easy")
        self.difficulty_combo.addItem(UI_TEXTS["medium"], "medium")
        self.difficulty_combo.addItem(UI_TEXTS["hard"], "hard")
        self.difficulty_combo.currentIndexChanged.connect(self.on_difficulty_changed)
        layout.addWidget(self.difficulty_combo, 1)
        
        self.setLayout(layout)
        self.update_difficulty_visibility()
    
    def on_mode_changed(self):
        """Обробник зміни режиму гри"""
        mode = self.mode_combo.currentData()
        self.mode_changed.emit(mode)
        self.update_difficulty_visibility()
    
    def on_difficulty_changed(self):
        """Обробник зміни складності"""
        difficulty = self.difficulty_combo.currentData()
        self.difficulty_changed.emit(difficulty)
    
    def update_difficulty_visibility(self):
        """Оновлює видимість елементів складності"""
        mode = self.mode_combo.currentData()
        is_pvai = (mode == GameMode.PvAI)
        
        self.difficulty_label.setVisible(is_pvai)
        self.difficulty_combo.setVisible(is_pvai)
    
    def set_current_mode(self, mode):
        """Встановлює поточний режим гри"""
        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
    
    def set_current_difficulty(self, difficulty):
        """Встановлює поточний рівень складності"""
        index = self.difficulty_combo.findData(difficulty)
        if index >= 0:
            self.difficulty_combo.setCurrentIndex(index)

class PiecePromotionMenu(QWidget):
    """Вікно вибору фігури для перетворення пішака"""
    piece_selected = pyqtSignal(str)
    
    def __init__(self, color, parent=None):
        super().__init__(parent, Qt.Popup)
        self.color = color
        self.selected_piece = None
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Визначення доступних фігур для перетворення
        pieces = ['q', 'r', 'b', 'n'] if self.color == chess.BLACK else ['Q', 'R', 'B', 'N']
        
        for piece in pieces:
            piece_widget = QLabel()
            pixmap = get_piece_icon(piece, 60).pixmap(60, 60)
            piece_widget.setPixmap(pixmap)
            piece_widget.setAlignment(Qt.AlignCenter)
            piece_widget.mousePressEvent = lambda event, p=piece: self.on_piece_selected(p)
            piece_widget.setStyleSheet("border: 1px solid gray; border-radius: 5px;")
            piece_widget.setCursor(Qt.PointingHandCursor)
            layout.addWidget(piece_widget)
        
        self.setLayout(layout)
        self.setFixedSize(250, 70)
    
    def on_piece_selected(self, piece):
        """Обробник вибору фігури"""
        self.selected_piece = piece
        self.piece_selected.emit(piece)
        self.close()