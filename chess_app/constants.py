# chess_app/constants.py

# =================================================================================
# ФУНКЦІЯ МАСШТАБУВАННЯ
# =================================================================================
# Тепер замість фіксованих значень ми будемо обчислювати їх у реальному часі
# залежно від розміру вікна, яке надає PyQt5.

def calculate_dimensions(window_width: int, window_height: int):
    """
    Розраховує розміри шахової дошки, клітинок та інших елементів
    залежно від поточного розміру вікна.

    :param window_width: ширина вікна
    :param window_height: висота вікна
    :return: словник з усіма розмірами
    """
    # Ширина дошки = 80% від меншого боку (щоб залишити місце для панелі)
    board_size = int(min(window_width * 0.7, window_height * 0.7))#0.8
    square_size = board_size // 7#8
    piece_size = int(square_size * 0.9)
    highlight_size = int(square_size * 0.3)

    # Панель інформації = 20% від ширини вікна
    info_panel_width = int(window_width * 0.4)#default 0.2

    return {
        "BOARD_SIZE": board_size,
        "SQUARE_SIZE": square_size,
        "PIECE_SIZE": piece_size,
        "HIGHLIGHT_SIZE": highlight_size,
        "INFO_PANEL_WIDTH": info_panel_width
    }

# =================================================================================
# КОЛЬОРИ
# =================================================================================
DARK_SQUARE = "#B58863"
LIGHT_SQUARE = "#F0D9B5"

SELECTED_COLOR = "#BACA2B"
POSSIBLE_MOVE_COLOR = "#64646440"
CHECK_COLOR = "#FF0000"
LAST_MOVE_COLOR = "#CCC06850"

TEXT_COLOR = "#000000"
BACKGROUND_COLOR = "#F5F5F5"
PANEL_BACKGROUND = "#E8E8E8"
BORDER_COLOR = "#A0A0A0"

# =================================================================================
# ШЛЯХИ ДО РЕСУРСІВ
# =================================================================================
RESOURCES_DIR = "resources"
PIECES_DIR = f"{RESOURCES_DIR}/pieces/"
STYLES_DIR = f"{RESOURCES_DIR}/styles/"

PIECE_IMAGES = {
    'P': "wp.png",
    'N': "wn.png",
    'B': "wb.png",
    'R': "wr.png",
    'Q': "wq.png",
    'K': "wk.png",
    'p': "bp.png",
    'n': "bn.png",
    'b': "bb.png",
    'r': "br.png",
    'q': "bq.png",
    'k': "bk.png",
}

MAIN_STYLE = f"{STYLES_DIR}main_style.css"

MODELS_DIR = "models"
DEFAULT_MODEL = f"{MODELS_DIR}/chess_evaluator.keras"

# =================================================================================
# НАЛАШТУВАННЯ ГРИ
# =================================================================================
PVP_MODE = "PvP"
PVAI_MODE = "PvAI"

AI_DIFFICULTY = {
    "EASY": {"depth": 2, "use_nn": False},
    "MEDIUM": {"depth": 3, "use_nn": True},
    "HARD": {"depth": 4, "use_nn": True}
}

GAME_STATE_ONGOING = "ongoing"
GAME_STATE_CHECK = "check"
GAME_STATE_CHECKMATE = "checkmate"
GAME_STATE_STALEMATE = "stalemate"
GAME_STATE_DRAW = "draw"

# =================================================================================
# ТЕКСТИ ІНТЕРФЕЙСУ
# =================================================================================
UI_TEXTS = {
    "window_title": "Шахи Python",
    "new_game": "Нова гра",
    "restart_game": "Перезапустити гру",
    "game_mode": "Режим \n гри:",
    "difficulty": "Складність:",
    "turn_white": "Хід білих",
    "turn_black": "Хід чорних",
    "game_over": "Гра завершена!",
    "check": "ШАХ!",
    "checkmate_white": "Мат! Виграли чорні",
    "checkmate_black": "Мат! Виграли білі",
    "stalemate": "Пат - нічия",
    "draw": "Нічия",
    "pvp_mode": "Гравець vs Гравець",
    "pvai_mode": "Гравець vs ШІ",
    "easy": "Легкий",
    "medium": "Середній",
    "hard": "Складний",
    "game_state": "Стан гри",
    "game_control": "Керування грою",
    "game_settings": "Налаштування",
    "extra_functions": "Додаткові функції",
    "save_game": "Зберегти гру",
    "load_game": "Завантажити гру",
    "exit": "Вихід",
    "settings_menu": "Налаштування",
    "help_menu": "Допомога",
    "about": "Про програму",
    "rules": "Правила гри",
    "game_menu": "Меню гри"
}

# =================================================================================
# НАЛАШТУВАННЯ AI
# =================================================================================
TENSOR_SHAPE = (8, 8, 12)

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

# =================================================================================
# ШАХОВІ КОНСТАНТИ
# =================================================================================
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_VALUES = {
    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
}
