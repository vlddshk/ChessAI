import chess

def square_to_coords(square):
    """
    Перетворює шаховий квадрат (0-63) у координати (file, rank) на дошці.
    :param square: chess.Square (0-63)
    :return: tuple (file, rank) де file (0-7 = a-h), rank (0-7 = 1-8)
    """
    return chess.square_file(square), chess.square_rank(square)

def coords_to_square(file, rank):
    """
    Перетворює координати (file, rank) у шаховий квадрат.
    :param file: int (0-7 = a-h)
    :param rank: int (0-7 = 1-8)
    :return: chess.Square (0-63)
    """
    return chess.square(file, rank)

def square_to_pixel(square, square_size):
    """
    Перетворює шаховий квадрат у піксельні координати на дошці.
    :param square: chess.Square (0-63)
    :param square_size: розмір однієї клітинки у пікселях
    :return: tuple (x, y) координати верхнього лівого кута клітинки
    """
    file, rank = square_to_coords(square)
    # Перевертаємо ранг, оскільки у шахматах ранг 0 - це 1-а лінія (знизу),
    # але в нашому відображенні ранг 0 буде зверху
    return file * square_size, (7 - rank) * square_size

def pixel_to_square(x, y, square_size):
    """
    Перетворює піксельні координати на шаховий квадрат.
    :param x: координата X у пікселях
    :param y: координата Y у пікселях
    :param square_size: розмір однієї клітинки у пікселях
    :return: chess.Square (0-63) або None, якщо координати поза дошкою
    """
    file = int(x // square_size)
    rank = 7 - int(y // square_size)  # Інвертуємо ранг
    
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return coords_to_square(file, rank)
    return None

def square_name(square):
    """
    Повертає назву клітинки у шаховій нотації (наприклад, "e4").
    :param square: chess.Square (0-63)
    :return: str (назва клітинки)
    """
    return chess.square_name(square)

def square_from_name(name):
    """
    Перетворює назву клітинки у шаховий квадрат.
    :param name: str (наприклад, "e4")
    :return: chess.Square (0-63)
    """
    return chess.parse_square(name)

def get_center_of_square(square, square_size):
    """
    Повертає координати центру клітинки у пікселях.
    :param square: chess.Square (0-63)
    :param square_size: розмір однієї клітинки у пікселях
    :return: tuple (x, y) координати центру клітинки
    """
    x, y = square_to_pixel(square, square_size)
    return x + square_size // 2, y + square_size // 2

def is_dark_square(square):
    """
    Визначає, чи є клітинка темною.
    :param square: chess.Square (0-63)
    :return: True, якщо клітинка темна, False - якщо світла
    """
    file, rank = square_to_coords(square)
    return (file + rank) % 2 == 1

def get_square_color(square):
    """
    Повертає колір клітинки у форматі шістнадцяткового коду.
    :param square: chess.Square (0-63)
    :return: str (колір у форматі "#RRGGBB")
    """
    from chess_app.constants import DARK_SQUARE, LIGHT_SQUARE
    return DARK_SQUARE if is_dark_square(square) else LIGHT_SQUARE

def get_adjacent_squares(square):
    """
    Повертає сусідні клітинки у всіх напрямках.
    :param square: chess.Square (0-63)
    :return: list of chess.Square
    """
    adjacent = []
    for direction in [
        (1, 0), (-1, 0), (0, 1), (0, -1),   # Горизонталь/вертикаль
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Діагоналі
    ]:
        file, rank = square_to_coords(square)
        new_file, new_rank = file + direction[0], rank + direction[1]
        if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
            adjacent.append(coords_to_square(new_file, new_rank))
    return adjacent

def distance_between_squares(square1, square2):
    """
    Обчислює відстань між двома клітинками за шашковою метрикою.
    :param square1: chess.Square (0-63)
    :param square2: chess.Square (0-63)
    :return: int (відстань)
    """
    file1, rank1 = square_to_coords(square1)
    file2, rank2 = square_to_coords(square2)
    return max(abs(file1 - file2), abs(rank1 - rank2))

def get_line_between_squares(square1, square2):
    """
    Повертає клітинки на лінії між двома квадратами (включно з кінцями).
    :param square1: chess.Square (0-63)
    :param square2: chess.Square (0-63)
    :return: list of chess.Square
    """
    file1, rank1 = square_to_coords(square1)
    file2, rank2 = square_to_coords(square2)
    
    squares = []
    
    # Горизонтальна лінія
    if rank1 == rank2:
        start_file = min(file1, file2)
        end_file = max(file1, file2)
        for f in range(start_file, end_file + 1):
            squares.append(coords_to_square(f, rank1))
        return squares
    
    # Вертикальна лінія
    if file1 == file2:
        start_rank = min(rank1, rank2)
        end_rank = max(rank1, rank2)
        for r in range(start_rank, end_rank + 1):
            squares.append(coords_to_square(file1, r))
        return squares
    
    # Діагональна лінія
    if abs(file1 - file2) == abs(rank1 - rank2):
        file_step = 1 if file2 > file1 else -1
        rank_step = 1 if rank2 > rank1 else -1
        steps = abs(file1 - file2)
        
        for i in range(steps + 1):
            f = file1 + i * file_step
            r = rank1 + i * rank_step
            squares.append(coords_to_square(f, r))
        return squares
    
    # Якщо клітинки не на одній лінії
    return []

def get_square_from_mouse_event(event, board_widget):
    """
    Обчислює шаховий квадрат на основі події миші.
    :param event: QMouseEvent
    :param board_widget: BoardWidget, на якому відбулася подія
    :return: chess.Square або None
    """
    pos = event.pos()
    return pixel_to_square(pos.x(), pos.y(), board_widget.square_size)

def get_piece_symbol_at_square(board, square):
    """
    Повертає символ фігури на заданій клітинці.
    :param board: chess.Board
    :param square: chess.Square
    :return: str (символ фігури) або None
    """
    piece = board.piece_at(square)
    return piece.symbol() if piece else None

def get_king_square(board, color):
    """
    Повертає позицію короля заданого кольору.
    :param board: chess.Board
    :param color: chess.WHITE або chess.BLACK
    :return: chess.Square
    """
    return board.king(color)

def is_square_attacked(board, square, by_color):
    """
    Перевіряє, чи атакується клітинка фігурами заданого кольору.
    :param board: chess.Board
    :param square: chess.Square
    :param by_color: chess.WHITE або chess.BLACK
    :return: bool
    """
    return board.is_attacked_by(by_color, square)

def get_attacking_squares(board, square, attacker_color):
    """
    Повертає список клітинок, звідки фігури заданого кольору атакують цю клітинку.
    :param board: chess.Board
    :param square: chess.Square
    :param attacker_color: chess.WHITE або chess.BLACK
    :return: list of chess.Square
    """
    attackers = []
    for move in board.generate_legal_captures():
        if move.to_square == square and board.piece_at(move.from_square).color == attacker_color:
            attackers.append(move.from_square)
    return attackers

def get_castling_squares(color, kingside=True):
    """
    Повертає клітинки, пов'язані з рокіровкою.
    :param color: chess.WHITE або chess.BLACK
    :param kingside: True для короткої рокіровки, False для довгої
    :return: tuple (king_from, king_to, rook_from, rook_to)
    """
    if color == chess.WHITE:
        if kingside:
            return chess.E1, chess.G1, chess.H1, chess.F1
        else:
            return chess.E1, chess.C1, chess.A1, chess.D1
    else:
        if kingside:
            return chess.E8, chess.G8, chess.H8, chess.F8
        else:
            return chess.E8, chess.C8, chess.A8, chess.D8

def get_en_passant_capture_square(board):
    """
    Повертає клітинку, де може відбутися взяття на проході, або None.
    :param board: chess.Board
    :return: chess.Square або None
    """
    if board.ep_square is not None:
        return board.ep_square
    return None

def get_promotion_squares(color):
    """
    Повертає клітинки, на яких пішак може перетворитись на фігуру.
    :param color: chess.WHITE або chess.BLACK
    :return: list of chess.Square (8 клітинок)
    """
    if color == chess.WHITE:
        return [chess.square(file, 7) for file in range(8)]  # 8-а лінія
    else:
        return [chess.square(file, 0) for file in range(8)]  # 1-а лінія

def square_to_ui_position(square, square_size):
    """
    Повертає позицію для відображення фігури у віджеті.
    :param square: chess.Square
    :param square_size: розмір клітинки
    :return: QPoint
    """
    from PyQt5.QtCore import QPoint
    x, y = square_to_pixel(square, square_size)
    return QPoint(x, y)

def get_square_at_direction(square, direction, steps=1):
    """
    Повертає клітинку у заданому напрямку.
    :param square: початкова клітинка
    :param direction: tuple (delta_file, delta_rank)
    :param steps: кількість кроків
    :return: chess.Square або None, якщо поза дошкою
    """
    file, rank = square_to_coords(square)
    new_file = file + direction[0] * steps
    new_rank = rank + direction[1] * steps
    
    if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
        return coords_to_square(new_file, new_rank)
    return None

# Тестування функцій
if __name__ == "__main__":
    # Приклади використання
    e4 = square_from_name("e4")
    print(f"e4 square: {e4}")
    print(f"e4 coords: {square_to_coords(e4)}")
    print(f"e4 name: {square_name(e4)}")
    
    print("\nПіксельні координати для e4 (розмір клітинки=60):")
    print(f"Top-left: {square_to_pixel(e4, 60)}")
    print(f"Center: {get_center_of_square(e4, 60)}")
    
    print("\nПеревірка темних клітинок:")
    print(f"a1 (долі): {'dark' if is_dark_square(chess.A1) else 'light'}")
    print(f"h8 (угорі): {'dark' if is_dark_square(chess.H8) else 'light'}")
    
    print("\nСусідні клітинки для e4:")
    for sq in get_adjacent_squares(e4):
        print(square_name(sq))
    
    print("\nВідстань між a1 та h8:")
    print(distance_between_squares(chess.A1, chess.H8))
    
    print("\nЛінія між a1 та h8:")
    for sq in get_line_between_squares(chess.A1, chess.H8):
        print(square_name(sq))