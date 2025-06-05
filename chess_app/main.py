import sys 
import os 
import logging 
from PyQt5.QtWidgets import QApplication 
from ui.main_window import MainWindow 
from constants import UI_TEXTS 

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "chess_app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.getLogger("PyQt5").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

def main():
    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Запуск шахів")

    try:
        app = QApplication(sys.argv)
        app.setApplicationName(UI_TEXTS["window_title"])
        app.setAplicationVersion("1.0.0")

        logger.info("Створення головного вікна")
        window = MainWindow()

        if not check_resources():
            logger.error("Не вдалося знайти необхідні ресурси")
            show_resource_error()
            return 1 
        
        window.show()
        logger.info("Головне вікно відображено")

        logger.info("Запуск головного циклу додатку")
        return_code = app.exec_()

        logger.info(f"Додаток завершено з кодом: {return_code}")
        return return_code
    
    except Exception as e:
        logger.exсeption("Критична помилка під час роботи додатку")
        show_critical_error(str(e))
        return 1
    
def check_resources():
    """Перевіряє наявність критично важливих ресурсів"""
    # Перевірка зображень фігур
    from chess_app.constants import PIECES_DIR, PIECE_IMAGES
    
    for piece_type, filename in PIECE_IMAGES.items():
        path = os.path.join(PIECES_DIR, filename)
        if not os.path.exists(path):
            logging.error(f"Файл ресурсу не знайдено: {path}")
            return False
    
    # Перевірка базової моделі AI (не є критичною)
    from chess_app.constants import DEFAULT_MODEL
    if not os.path.exists(DEFAULT_MODEL):
        logging.warning(f"Базова модель AI не знайдена: {DEFAULT_MODEL}")
    
    return True

def show_resource_error():
    """Відображає повідомлення про помилку ресурсів"""
    app = QApplication(sys.argv)
    error_msg = """
    <b>Помилка завантаження ресурсів</b><br><br>
    Додатку не вдалося знайти необхідні файли ресурсів (зображення фігур).<br>
    Будь ласка, переконайтеся, що ви встановили додаток правильно.<br><br>
    Додаток буде закрито.
    """
    msg_box = QMessageBox.critical(
        None,
        "Помилка ресурсів",
        error_msg,
        QMessageBox.Ok
    )
    sys.exit(1)

def show_critical_error(message):
    """Відображає повідомлення про критичну помилку"""
    app = QApplication(sys.argv)
    error_msg = f"""
    <b>Критична помилка</b><br><br>
    Виникла непередбачена помилка:<br>
    <i>{message}</i><br><br>
    Додаток буде закрито.<br>
    Більш детальну інформацію можна знайти у файлі журналу.
    """
    msg_box = QMessageBox.critical(
        None,
        "Критична помилка",
        error_msg,
        QMessageBox.Ok
    )
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())

