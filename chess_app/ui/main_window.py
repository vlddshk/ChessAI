from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QAction, QMessageBox, QActionGroup
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import QTimer
from .board_widget import BoardWidget
from .game_info import GameInfo
from game.controller import GameController
from game.mode import GameMode, GameModeManager
from ai.minimax_ai import MinimaxAI
from ai.tf_evaluator import TFEvaluator
from constants import (
    UI_TEXTS, BACKGROUND_COLOR, calculate_dimensions
)
import sys
import chess


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_TEXTS["window_title"])

        # Початкові розміри вікна (запасні значення)
        init_width = 2400 #default 1200
        init_height = 1600 #default 800
        dims = calculate_dimensions(init_width, init_height)
        board_size = dims.get("BOARD_SIZE", 2400) #defaul 640
        info_width = dims.get("INFO_PANEL_WIDTH", 300) #default 300

        # Мінімальний розмір вікна залежить від обчислених розмірів
        self.setMinimumSize(board_size + info_width + 60, board_size + 100) #default 60 100

        # Ініціалізація ключових компонентів
        self.game_controller = GameController()
        self.mode_manager = GameModeManager()
        self.ai = None
        self.nn_evaluator = None


        # Налаштування головного інтерфейсу
        self.init_ui(board_size, info_width)
        self.init_signals()
        self.load_model_for_difficulty()
        self.init_ai()

        # Застосування стилів
        self.apply_styles()

        # Оновлення стану гри
        self.update_game_state(UI_TEXTS["turn_white"])

    def init_ui(self, board_size, info_width):
        """Ініціалізація користувацького інтерфейсу"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(3)#default 10
        main_layout.setContentsMargins(5, 5, 5, 5)#default 10

        self.board_widget = BoardWidget()
        # Забезпечуємо, щоб дошка початково мала квадратний розмір
        self.board_widget.setFixedSize(board_size, board_size)


        self.game_info = GameInfo()
        self.game_info.setFixedWidth(info_width)

        main_layout.addWidget(self.board_widget, 0)
        main_layout.addWidget(self.game_info, 0)

        central_widget.setLayout(main_layout)

        # Створення меню
        self.create_menu()

        # Встановлення зв'язків між компонентами
        self.board_widget.game_controller = self.game_controller
        self.game_controller.board_widget = self.board_widget
        try:
            self.board_widget.update_board_state(self.game_controller.get_board_state())
        except Exception:
            pass

    def create_menu(self):
        menu_bar = self.menuBar()

        game_menu = menu_bar.addMenu(UI_TEXTS["game_menu"])

        new_game_action = QAction(UI_TEXTS["new_game"], self)
        new_game_action.triggered.connect(self.on_new_game)
        new_game_action.setShortcut(QKeySequence.New)

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
        game_menu.addSeparator()
        game_menu.addAction(save_action)
        game_menu.addAction(load_action)
        game_menu.addSeparator()
        game_menu.addAction(exit_action)

        # Меню "Налаштування"
        settings_menu = menu_bar.addMenu(UI_TEXTS["settings_menu"])

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

        difficulty_menu = settings_menu.addMenu(UI_TEXTS["difficulty"])

        self.easy_action = QAction(UI_TEXTS["easy"], self)
        self.easy_action.setCheckable(True)
        self.easy_action.setChecked(True)
        self.easy_action.triggered.connect(lambda: self.set_difficulty("EASY"))

        self.medium_action = QAction(UI_TEXTS["medium"], self)
        self.medium_action.setCheckable(True)
        self.medium_action.triggered.connect(lambda: self.set_difficulty("MEDIUM"))

        self.hard_action = QAction(UI_TEXTS["hard"], self)
        self.hard_action.setCheckable(True)
        self.hard_action.triggered.connect(lambda: self.set_difficulty("HARD"))

        difficulty_action_group = QActionGroup(self)
        difficulty_action_group.addAction(self.easy_action)
        difficulty_action_group.addAction(self.medium_action)
        difficulty_action_group.addAction(self.hard_action)

        difficulty_menu.addAction(self.easy_action)
        difficulty_menu.addAction(self.medium_action)
        difficulty_menu.addAction(self.hard_action)

        help_menu = menu_bar.addMenu(UI_TEXTS["help_menu"])

        about_action = QAction(UI_TEXTS["about"], self)
        about_action.triggered.connect(self.show_about)

        rules_action = QAction(UI_TEXTS["rules"], self)
        rules_action.triggered.connect(self.show_rules)

        help_menu.addAction(about_action)
        help_menu.addAction(rules_action)

    def init_signals(self):
        # Сигнали від дошки
        self.board_widget.square_selected.connect(self.handle_square_selected)
        self.board_widget.move_made.connect(self.handle_player_move)

        # Сигнали від панелі інформації
        self.game_info.new_game_requested.connect(self.on_new_game)
        self.game_info.game_mode_changed.connect(self.on_game_mode_changed)
        self.game_info.difficulty_changed.connect(self.on_difficulty_changed)
        self.game_info.save_game_requested.connect(self.on_save_game)
        self.game_info.load_game_requested.connect(self.on_load_game)

        # Сигнали від контролера гри
        self.game_controller.ai_move_generated.connect(self.handle_ai_move)
        self.game_controller.game_state_changed.connect(self.update_game_state)
        self.game_controller.game_over.connect(self.handle_game_over)

    def init_ai(self):
        # Створення AI з базовими налаштуваннями
        self.ai = MinimaxAI(depth=2)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BACKGROUND_COLOR};
            }}
            QWidget {{
                font-family: "Arial";
                font-size: 30px;
            }}
        """)#default 12px

    def handle_square_selected(self):
        return

    def handle_player_move(self, move_uci):
        if self.game_controller.make_move(move_uci):
            self.board_widget.update_board_state(self.game_controller.get_board_state())

            if (self.mode_manager.current_mode == GameMode.PvAI and 
                not self.game_controller.is_game_over() and
                self.game_controller.get_turn() == chess.BLACK):
                self.make_ai_move()

    def make_ai_move(self):
        QTimer.singleShot(300, self.game_controller.request_ai_move)

    def handle_ai_move(self, move_uci):
        if self.game_controller.make_move(move_uci):
            self.board_widget.update_board_state(self.game_controller.get_board_state())
            self.board_widget.highlight_move(move_uci)
            self.update_game_state(self.game_controller.get_game_state_text())

    def handle_game_over(self, result):
        message = {
            "1-0": "Білі перемогли!",
            "0-1": "Чорні перемогли!",
            "1/2-1/2": "Нічия!"
        }.get(result, "Гра завершена")

        QMessageBox.information(self, "Гра завершена", message)

    def handle_promotion(self):
        #не використовується
        self.board_widget.update_board_state(self.game_controller.get_board_state())

    def update_game_state(self, state_text):
        self.game_info.update_game_state(state_text)
        self.setWindowTitle(f"{UI_TEXTS['window_title']} - {state_text}")

    def set_game_mode(self, mode):
        self.mode_manager.set_mode(mode)
        self.game_controller.set_mode(mode)
        self.game_info.set_current_mode(mode)

        if mode == GameMode.PvP:
            self.pvp_action.setChecked(True)
        else:
            self.pvai_action.setChecked(True)

    def set_difficulty(self, difficulty_key):
        # Перетворюємо на формат, який розуміє mode_manager/game_controller
        self.mode_manager.set_difficulty(difficulty_key)
        self.game_controller.set_difficulty(difficulty_key)
        self.game_info.set_current_difficulty(difficulty_key)
        self.update_ai_config()

    def load_model_for_difficulty(self): #NEW
        """Завантажує модель під вибраний рівень складності (якщо є)"""
        config = self.mode_manager.get_ai_config()
        model_path = getattr(config, 'model_path', None)

        if model_path:
            if not self.nn_evaluator or getattr(self.nn_evaluator, 'model_path', None) != model_path:
                try:
                    self.nn_evaluator = TFEvaluator(model_path)
                    self.nn_evaluator.model_path = model_path  
                    print(f"[MODEL] Завантажено модель: {model_path}")
                except Exception as e:
                    print(f"[MODEL ERROR] Не вдалося завантажити модель {model_path}: {e}")
                    self.nn_evaluator = None
        else:
            self.nn_evaluator = None

    def update_ai_config(self):
        config = self.mode_manager.get_ai_config()

    # AI глибина залежить тільки від рівня
        depth = config.depth

    # NN вмикаємо тільки якщо є модель для цього рівня і вона завантажена
        use_nn = config.model_path is not None and self.nn_evaluator is not None

        if self.ai:
            self.ai.depth = depth
            self.ai.use_nn = use_nn
            self.ai.evaluator = self.nn_evaluator if use_nn else None

        print(f"[AI CONFIG] depth={depth}, use_nn={use_nn}")

    def on_game_mode_changed(self, mode_name):
        mode = GameMode.PvP if mode_name == "PvP" else GameMode.PvAI
        self.set_game_mode(mode)

    def on_difficulty_changed(self, difficulty_name):
        mapping = {
            "EASY": "Легкий",
            "MEDIUM": "Середній",
            "HARD": "Складний",
            UI_TEXTS['easy']: "Легкий",
            UI_TEXTS['medium']: "Середній",
           UI_TEXTS['hard']: "Складний"
        }
        resolved_name = mapping.get(difficulty_name, difficulty_name)

        # Передаємо у mode_manager вже локалізовану назву
        self.mode_manager.set_difficulty(resolved_name)
        self.game_controller.set_difficulty(resolved_name)
        self.game_info.set_current_difficulty(resolved_name)

        self.load_model_for_difficulty()
        self.update_ai_config()


    def on_new_game(self):
        self.reset_game()

    def on_save_game(self):
        QMessageBox.information(self, UI_TEXTS["save_game"], "Функція збереження буде реалізована в наступних версіях")

    def on_load_game(self):
        QMessageBox.information(self, UI_TEXTS["load_game"], "Функція завантаження буде реалізована в наступних версіях")

    def reset_game(self):
        self.game_controller.reset()
        #self.board_widget.reset_board()
        self.update_game_state(UI_TEXTS["turn_white"])

    def show_about(self):
        about_text = f"""
        <b>{UI_TEXTS['window_title']}</b><br><br>
        Версія: 1.0.0<br>
        Розробник: DSHK MEGA CORPORATION<br><br>
        Шаховий додаток з підтримкою гри проти штучного інтелекту.<br>
        Реалізовано за допомогою Python, PyQt5 та TensorFlow.<br><br>
        © 2025
        """
        QMessageBox.about(self, UI_TEXTS["about"], about_text)

    def show_rules(self):
        rules_text = """
        <h3>Основні правила шахів:</h3>
        <ul>
            <li>Гра ведеться на дошці 8x8 клітин</li>
            <li>Кожен гравець починає з 16 фігур: король, ферзь, дві тури, два слони, два коні та вісім пішаків</li>
            <li>Мета гри - поставити мат королю суперника</li>
            <li>Фігури ходять за спеціальними правилами:</li>
        </ul>
        """
        QMessageBox.information(self, UI_TEXTS["rules"], rules_text)

    def resizeEvent(self, event):
        # При зміні розміру вікна перераховуємо розміри дошки та панелі
        dims = calculate_dimensions(self.width(), self.height())
        board_size = dims.get('BOARD_SIZE')
        info_w = dims.get('INFO_PANEL_WIDTH')

        # Застосовуємо нові розміри
        if board_size:
            self.board_widget.setFixedSize(board_size, board_size)
        if info_w:
            self.game_info.setFixedWidth(info_w)

        super().resizeEvent(event)

    def closeEvent(self, event):
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
