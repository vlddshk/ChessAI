from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QMenuBar, QAction, QFileDialog, QMessageBox, QActionGroup
)
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtCore import Qt, QTimer
from .board_widget import BoardWidget
from .game_info import GameInfo
from game.controller import GameController
from game.mode import GameMode, GameModeManager
from ai.minimax_ai import MinimaxAI
from ai.tf_evaluator import TFEvaluator
from constants import (
    UI_TEXTS, BOARD_SIZE, BACKGROUND_COLOR,
    MODELS_DIR, DEFAULT_MODEL
)
import os
import sys
import chess

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_TEXTS["window_title"])
        self.setMinimumSize(BOARD_SIZE + 300, BOARD_SIZE + 100)
        
        # Ініціалізація ключових компонентів
        self.game_controller = GameController()
        self.mode_manager = GameModeManager()
        self.ai = None
        
        # Налаштування головного інтерфейсу
        self.init_ui()
        self.init_signals()
        self.init_ai()
        
        # Застосування стилів
        self.apply_styles()
        
        # Оновлення стану гри
        self.update_game_state(UI_TEXTS["turn_white"])
    
    def init_ui(self):
        """Ініціалізація користувацького інтерфейсу"""
        # Центральний віджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Головний лейаут
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Створення компонентів
        self.board_widget = BoardWidget()
        self.game_info = GameInfo()
        
        # Додавання компонентів до лейауту
        main_layout.addWidget(self.board_widget, 1)
        main_layout.addWidget(self.game_info)
        
        central_widget.setLayout(main_layout)
        
        # Створення меню
        self.create_menu()
        
        # Встановлення зв'язків між компонентами
        self.board_widget.game_controller = self.game_controller
        self.game_controller.board_widget = self.board_widget
    
    def create_menu(self):
        """Створення головного меню додатку"""
        menu_bar = self.menuBar()
        
        # Меню "Гра"
        game_menu = menu_bar.addMenu(UI_TEXTS["game_menu"])
        
        # Дії меню "Гра"
        new_game_action = QAction(UI_TEXTS["new_game"], self)
        new_game_action.triggered.connect(self.on_new_game)
        new_game_action.setShortcut(QKeySequence.New)
        
        restart_action = QAction(UI_TEXTS["restart_game"], self)
        restart_action.triggered.connect(self.on_restart_game)
        restart_action.setShortcut("Ctrl+R")
        
        save_action = QAction(UI_TEXTS["save_game"], self)
        save_action.triggered.connect(self.on_save_game)
        save_action.setShortcut(QKeySequence.Save)
        
        load_action = QAction(UI_TEXTS["load_game"], self)
        load_action.triggered.connect(self.on_load_game)
        load_action.setShortcut(QKeySequence.Open)
        
        exit_action = QAction(UI_TEXTS["exit"], self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut(QKeySequence.Quit)
        
        game_menu.addAction(new_game_action)
        game_menu.addAction(restart_action)
        game_menu.addSeparator()
        game_menu.addAction(save_action)
        game_menu.addAction(load_action)
        game_menu.addSeparator()
        game_menu.addAction(exit_action)
        
        # Меню "Налаштування"
        settings_menu = menu_bar.addMenu(UI_TEXTS["settings_menu"])
        
        # Дії меню "Налаштування"
        self.pvp_action = QAction(UI_TEXTS["pvp_mode"], self)
        self.pvp_action.setCheckable(True)
        self.pvp_action.setChecked(True)
        self.pvp_action.triggered.connect(lambda: self.set_game_mode(GameMode.PvP))
        
        self.pvai_action = QAction(UI_TEXTS["pvai_mode"], self)
        self.pvai_action.setCheckable(True)
        self.pvai_action.triggered.connect(lambda: self.set_game_mode(GameMode.PvAI))
        
        mode_action_group = QActionGroup(self)
        mode_action_group.addAction(self.pvp_action)
        mode_action_group.addAction(self.pvai_action)
        
        settings_menu.addAction(self.pvp_action)
        settings_menu.addAction(self.pvai_action)
        settings_menu.addSeparator()
        
        # Підменю "Складність"
        difficulty_menu = settings_menu.addMenu(UI_TEXTS["difficulty"])
        
        self.easy_action = QAction(UI_TEXTS["easy"], self)
        self.easy_action.setCheckable(True)
        self.easy_action.setChecked(True)
        self.easy_action.triggered.connect(lambda: self.set_difficulty("Легкий"))
        
        self.medium_action = QAction(UI_TEXTS["medium"], self)
        self.medium_action.setCheckable(True)
        self.medium_action.triggered.connect(lambda: self.set_difficulty("Середній"))
        
        self.hard_action = QAction(UI_TEXTS["hard"], self)
        self.hard_action.setCheckable(True)
        self.hard_action.triggered.connect(lambda: self.set_difficulty("Складний"))
        
        difficulty_action_group = QActionGroup(self)
        difficulty_action_group.addAction(self.easy_action)
        difficulty_action_group.addAction(self.medium_action)
        difficulty_action_group.addAction(self.hard_action)
        
        difficulty_menu.addAction(self.easy_action)
        difficulty_menu.addAction(self.medium_action)
        difficulty_menu.addAction(self.hard_action)
        
        # Меню "Допомога"
        help_menu = menu_bar.addMenu(UI_TEXTS["help_menu"])
        
        about_action = QAction(UI_TEXTS["about"], self)
        about_action.triggered.connect(self.show_about)
        
        rules_action = QAction(UI_TEXTS["rules"], self)
        rules_action.triggered.connect(self.show_rules)
        
        help_menu.addAction(about_action)
        help_menu.addAction(rules_action)
    
    def init_signals(self):
        """Ініціалізація сигналів та слотів"""
        # Сигнали від дошки
        self.board_widget.square_selected.connect(self.handle_square_selected)
        self.board_widget.move_made.connect(self.handle_player_move)
        self.board_widget.promotion_required.connect(self.handle_promotion)
        
        # Сигнали від панелі інформації
        self.game_info.new_game_requested.connect(self.on_new_game)
        self.game_info.restart_game_requested.connect(self.on_restart_game)
        self.game_info.game_mode_changed.connect(self.on_game_mode_changed)
        self.game_info.difficulty_changed.connect(self.on_difficulty_changed)
        self.game_info.save_game_requested.connect(self.on_save_game)
        self.game_info.load_game_requested.connect(self.on_load_game)
        
        # Сигнали від контролера гри
        self.game_controller.ai_move_generated.connect(self.handle_ai_move)
        self.game_controller.game_state_changed.connect(self.update_game_state)
        self.game_controller.game_over.connect(self.handle_game_over)
    
    def init_ai(self):
        """Ініціалізація штучного інтелекту"""
        # Створення AI з базовими налаштуваннями
        self.ai = MinimaxAI(depth=2)
        
        # Спробуємо завантажити модель нейромережі
        self.nn_evaluator = None
        if os.path.exists(DEFAULT_MODEL):
            try:
                self.nn_evaluator = TFEvaluator(DEFAULT_MODEL)
                print("Модель нейромережі успішно завантажена")
            except Exception as e:
                print(f"Помилка завантаження моделі: {e}")
    
    def apply_styles(self):
        """Застосування стилів до головного вікна"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BACKGROUND_COLOR};
            }}
            QWidget {{
                font-family: "Arial";
                font-size: 12px;
            }}
        """)
    
    def handle_square_selected(self, square_name):
        """Обробка вибору клітинки на дошці"""
        # Цей метод обробляє вибір клітинки перед тим, як зробити хід
        pass
    
    def handle_player_move(self, move_uci):
        """Обробка ходу гравця"""
        # Передаємо хід до контролера гри
        if self.game_controller.make_move(move_uci):
            # Оновлюємо дошку
            self.board_widget.update_board_state(self.game_controller.get_board_state())
            
            # Якщо гра продовжується і режим PvAI, запускаємо хід AI
            if (self.mode_manager.current_mode == GameMode.PvAI and 
                not self.game_controller.is_game_over() and
                self.game_controller.get_turn() == chess.BLACK):
                self.make_ai_move()
    
    def make_ai_move(self):
        """Запуск генерації ходу AI"""
        # Використовуємо QTimer для додавання невеликої затримки
        QTimer.singleShot(300, self.game_controller.request_ai_move)
    
    def handle_ai_move(self, move_uci):
        """Обробка ходу, згенерованого AI"""
        # Виконуємо хід AI
        if self.game_controller.make_move(move_uci):
            # Оновлюємо дошку
            self.board_widget.update_board_state(self.game_controller.get_board_state())
            
            # Підсвічуємо хід AI
            self.board_widget.highlight_move(move_uci)
            
            # Оновлюємо стан гри в UI
            self.update_game_state(self.game_controller.get_game_state_text())
    
    def handle_game_over(self, result):
        """Обробка завершення гри"""
        # Показуємо повідомлення про результат гри
        message = {
            "1-0": "Білі перемогли!",
            "0-1": "Чорні перемогли!",
            "1/2-1/2": "Нічия!"
        }.get(result, "Гра завершена")
        
        QMessageBox.information(self, "Гра завершена", message)
    
    def handle_promotion(self, move_uci, piece):
        """Обробка вибору фігури при перетворенні пішака"""
        # Тут ми вже маємо повний хід з вибраною фігурою
        if self.game_controller.make_move(move_uci):
            self.board_widget.update_board_state(self.game_controller.get_board_state())
    
    def update_game_state(self, state_text):
        """Оновлення стану гри в інтерфейсі"""
        self.game_info.update_game_state(state_text)
        
        # Оновлення заголовка вікна для відображення стану
        self.setWindowTitle(f"{UI_TEXTS['window_title']} - {state_text}")
    
    def set_game_mode(self, mode):
        """Встановлення режиму гри"""
        self.mode_manager.set_mode(mode)
        self.game_controller.set_mode(mode)
        
        # Оновлення інтерфейсу
        self.game_info.set_current_mode(mode)
        
        # Оновлення меню
        if mode == GameMode.PvP:
            self.pvp_action.setChecked(True)
        else:
            self.pvai_action.setChecked(True)
    
    def set_difficulty(self, difficulty_name):
        """Встановлення рівня складності"""
        self.mode_manager.set_difficulty(difficulty_name)
        self.game_controller.set_difficulty(difficulty_name)
        
        # Оновлення інтерфейсу
        self.game_info.set_current_difficulty(difficulty_name)
        
        # Оновлення AI з новими параметрами
        self.update_ai_config()
    
    def update_ai_config(self):
        """Оновлення конфігурації AI на основі поточних налаштувань"""
        config = self.mode_manager.get_ai_config()
        
        # Оновлюємо параметри AI
        if self.ai:
            self.ai.depth = config.depth
            
            # Якщо використовуємо нейромережу і вона завантажена
            if config.use_nn and self.nn_evaluator:
                self.ai.use_nn = True
                self.ai.evaluator = self.nn_evaluator
            else:
                self.ai.use_nn = False
                self.ai.evaluator = None
    
    def on_game_mode_changed(self, mode_name):
        """Обробка зміни режиму гри з панелі інформації"""
        mode = GameMode.PvP if mode_name == "PvP" else GameMode.PvAI
        self.set_game_mode(mode)
    
    def on_difficulty_changed(self, difficulty_name):
        """Обробка зміни складності з панелі інформації"""
        self.set_difficulty(difficulty_name)
    
    def on_new_game(self):
        """Обробка запиту на нову гру"""
        self.reset_game()
    
    def on_restart_game(self):
        """Обробка запиту на перезапуск гри"""
        self.game_controller.reset()
        self.board_widget.reset_board()
        self.update_game_state(UI_TEXTS["turn_white"])
    
    def on_save_game(self):
        """Обробка запиту на збереження гри"""
        # У майбутньому реалізуємо збереження
        QMessageBox.information(
            self, 
            UI_TEXTS["save_game"], 
            "Функція збереження буде реалізована в наступних версіях"
        )
    
    def on_load_game(self):
        """Обробка запиту на завантаження гри"""
        # У майбутньому реалізуємо завантаження
        QMessageBox.information(
            self, 
            UI_TEXTS["load_game"], 
            "Функція завантаження буде реалізована в наступних версіях"
        )
    
    def reset_game(self):
        """Повне скидання гри до початкового стану"""
        self.game_controller.reset()
        self.board_widget.reset_board()
        self.update_game_state(UI_TEXTS["turn_white"])
    
    def show_about(self):
        """Відображення діалогу 'Про програму'"""
        about_text = f"""
        <b>{UI_TEXTS['window_title']}</b><br><br>
        Версія: 1.0.0<br>
        Розробник: DSHK MEGA CORPORATION<br><br>
        Шаховий додаток з підтримкою гри проти штучного інтелекту.<br>
        Реалізовано за допомогою Python, PyQt5 та TensorFlow.<br><br>
        © 2025 Всі права захищені ВлАдіКоМ та СоФієЮ.
        """
        QMessageBox.about(self, UI_TEXTS["about"], about_text)
    
    def show_rules(self):
        """Відображення діалогу з правилами гри"""
        rules_text = """
        <h3>Основні правила шахів:</h3>
        <ul>
            <li>Гра ведеться на дошці 8x8 клітин</li>
            <li>Кожен гравець починає з 16 фігур: король, ферзь, дві тури, два слони, два коні та вісім пішаків</li>
            <li>Мета гри - поставити мат королю суперника</li>
            <li>Фігури ходять за спеціальними правилами:
                <ul>
                    <li>Пішак: на одну клітинку вперед (на початку - на дві), б'є по діагоналі</li>
                    <li>Кінь: ходить буквою "Г"</li>
                    <li>Слон: по діагоналі</li>
                    <li>Тура: по горизонталі або вертикалі</li>
                    <li>Ферзь: по будь-якій прямій</li>
                    <li>Король: на одну клітинку у будь-якому напрямку</li>
                </ul>
            </li>
            <li>Особливі ходи: рокіровка, взяття на проході, перетворення пішака</li>
        </ul>
        """
        QMessageBox.information(self, UI_TEXTS["rules"], rules_text)
    
    def closeEvent(self, event):
        """Обробка закриття додатку"""
        reply = QMessageBox.question(
            self, 
            "Підтвердження виходу",
            "Ви впевнені, що хочете вийти?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

# Точка входу в додаток
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())