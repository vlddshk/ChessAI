from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QGroupBox,
    QComboBox, QFrame
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, pyqtSignal
from constants import (
    UI_TEXTS, BACKGROUND_COLOR, 
    PANEL_BACKGROUND, TEXT_COLOR, 
    BORDER_COLOR, BOARD_SIZE
)
from game.mode import GameMode, GameModeManager
import chess

class GameInfo(QWidget):
    new_game_requested = pyqtSignal()
    restart_game_requested = pyqtSignal()
    game_mode_changed = pyqtSignal(str)
    difficulty_changed = pyqtSignal(str)
    save_game_requested = pyqtSignal()
    load_game_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
        self.init_ui()
        self.setStyleSheet(self.get_stylesheet())

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)

        status_group = self.create_status_group()
        main_layout.addWidget(status_group)

        control_group = self.create_control_group()
        main_layout.addWidget(control_group)

        settings_group = self.create_settings_group()
        main_layout.addWidget(settings_group)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        extra_group = self.create_extra_group()
        main_layout.addWidget(extra_group)

        main_layout.addStretch()

        self.setLayout(main_layout)

    def create_status_group(self):
        group = QGroupBox(UI_TEXTS["game_state"])
        layout = QVBoxLayout()

        self.status_label = QLabel(UI_TEXTS["turn_white"])
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(30)

        self.turn_indicator = QLabel()
        self.turn_indicator.setFixedSize(20, 20)
        self.update_turn_indicator(chess.WHITE)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.turn_indicator)
        status_layout.addWidget(self.status_label, 1)

        layout.addLayout(status_layout)
        group.setLayout(layout)
        return group 
    
    def create_control_group(self):
        group = QGroupBox(UI_TEXTS["game_control"])
        layout = QVBoxLayout()

        self.new_game_btn = QPushButton(UI_TEXTS["new_game"])
        self.new_game_btn.setToolTip("Почати нову гру")
        self.new_game_btn.clicked.connect(self.new_game_requested.emit)

        self.restart_btn = QPushButton(UI_TEXTS["restart_game"])
        self.restart_btn.setToolTip("Перезапустити гру")
        self.restart_btn.clicked.connect(self.restart_game_requested.emit)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.new_game_btn)
        btn_layout.addWidget(self.restart_btn)

        layout.addLayout(btn_layout)
        group.setLayout(layout)
        return group 
    
    def create_settings_group(self):
        group = QGroupBox(UI_TEXTS["game_settings"])
        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_label = QLabel(UI_TEXTS["game_mode"] + ":")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(UI_TEXTS["pvp_mode"], GameMode.PvP)
        self.mode_combo.addItem(UI_TEXTS["pvai_mode"], GameMode.PvAI)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo, 1)

        difficulty_layout = QHBoxLayout()
        difficulty_label = QLabel(UI_TEXTS["difficulty"] + ":")
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItem(UI_TEXTS["easy"], "easy")
        self.difficulty_combo.addItem(UI_TEXTS["medium"], "medium")
        self.difficulty_combo.addItem(UI_TEXTS["hard"], "hard")
        self.difficulty_combo.currentIndexChanged.connect(self.on_difficulty_changed)

        difficulty_layout.addWidget(difficulty_label)
        difficulty_layout.addWidget(self.difficulty_combo, 1)

        layout.addLayout(mode_layout)
        layout.addLayout(difficulty_layout)
        group.setLayout(layout)
        return group
    
    def create_extra_group(self):
        group = QGroupBox(UI_TEXTS["extra_functions"])
        layout = QVBoxLayout()

                # Кнопки додаткових функцій
        self.save_btn = QPushButton(UI_TEXTS["save_game"])
        self.save_btn.setToolTip("Зберегти поточну гру")
        self.save_btn.clicked.connect(self.save_game_requested.emit)
        
        self.load_btn = QPushButton(UI_TEXTS["load_game"])
        self.load_btn.setToolTip("Завантажити збережену гру")
        self.load_btn.clicked.connect(self.load_game_requested.emit)
        
        # Розміщення кнопок
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        
        layout.addLayout(btn_layout)
        group.setLayout(layout)
        return group
    
    def update_game_state(self, state_text):
        """Оновлення статусу гри"""
        self.status_label.setText(state_text)
        
        # Оновлення індикатора черги
        if UI_TEXTS["turn_white"] in state_text:
            self.update_turn_indicator(chess.WHITE)
        elif UI_TEXTS["turn_black"] in state_text:
            self.update_turn_indicator(chess.BLACK)
        else:
            self.turn_indicator.setStyleSheet("background-color: transparent;")
    
    def update_turn_indicator(self, turn):
        """Оновлення індикатора черги"""
        color = "white" if turn == chess.WHITE else "black"
        self.turn_indicator.setStyleSheet(
            f"background-color: {color};"
            "border: 1px solid black;"
            "border-radius: 10px;"
        )
    
    def on_mode_changed(self, index):
        """Обробка зміни режиму гри"""
        mode = self.mode_combo.currentData()
        self.game_mode_changed.emit(mode.name)
        
        # Показувати/ховати вибір складності в залежності від режиму
        self.difficulty_combo.setVisible(mode == GameMode.PvAI)
    
    def on_difficulty_changed(self, index):
        """Обробка зміни рівня складності"""
        difficulty = self.difficulty_combo.currentData()
        self.difficulty_changed.emit(difficulty)
    
    def set_current_mode(self, mode):
        """Встановлення поточного режиму гри"""
        for i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(i) == mode:
                self.mode_combo.setCurrentIndex(i)
                break
    
    def set_current_difficulty(self, difficulty):
        """Встановлення поточного рівня складності"""
        for i in range(self.difficulty_combo.count()):
            if self.difficulty_combo.itemData(i) == difficulty:
                self.difficulty_combo.setCurrentIndex(i)
                break
    
    def get_stylesheet(self):
        """Повертає CSS для стилізації панелі"""
        return f"""
            QWidget {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
            QGroupBox {{
                background-color: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: transparent;
            }}
            QLabel {{
                font-size: 12px;
            }}
            QPushButton {{
                background-color: #4A4A4A;
                color: white;
                border: 1px solid #2A2A2A;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: #5A5A5A;
            }}
            QPushButton:pressed {{
                background-color: #3A3A3A;
            }}
            QComboBox {{
                background-color: white;
                color: black;
                border: 1px solid #A0A0A0;
                border-radius: 3px;
                padding: 2px;
            }}
        """
    
    def resizeEvent(self, event):
        """Обробка зміни розміру"""
        # Забезпечуємо, щоб панель залишалася правильної ширини
        self.setFixedWidth(250)
        super().resizeEvent(event)

# Приклад використання
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    info_panel = GameInfo()
    info_panel.show()
    sys.exit(app.exec_())