import chess
import numpy as np
from typing import Dict
from constants import PIECE_VALUES, TENSOR_SHAPE, PIECE_TO_INDEX
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Перетворює шахову дошку у тензорне представлення (8x8x12)
    :param board: Шахова дошка
    :return: Тензор numpy розміром 8x8x12
    """
    tensor = np.zeros(TENSOR_SHAPE, dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Визначаємо індекс каналу для цієї фігури
            piece_symbol = piece.symbol()
            channel_idx = PIECE_TO_INDEX[piece_symbol]
            
            # Обчислюємо координати у тензорному представленні
            rank, file = chess.square_rank(square), chess.square_file(square)
            # Перевертаємо ранг, оскільки у шахматах ранг 0 - це перший ряд (знизу),
            # а в нашому тензорі ми хочемо мати перший ряд зверху
            tensor[7 - rank, file, channel_idx] = 1.0
    
    return tensor

def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Перетворює FEN-рядок у тензорне представлення
    :param fen: FEN-рядок
    :return: Тензор numpy розміром 8x8x12
    """
    board = chess.Board(fen)
    return board_to_tensor(board)

def get_piece_activity(board: chess.Board, color: chess.Color) -> float:
    """
    Обчислює активність фігур для заданого кольору
    :param board: Шахова дошка
    :param color: Колір (chess.WHITE або chess.BLACK)
    :return: Значення активності
    """
    activity = 0.0
    
    # Кількість легальних ходів для кольору
    legal_moves = list(board.legal_moves)
    color_moves = [move for move in legal_moves if board.piece_at(move.from_square).color == color]
    activity += len(color_moves) * 0.05
    
    # Центральний контроль
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    for square in center_squares:
        if board.is_attacked_by(color, square):
            activity += 0.1
    
    # Атака на ворожі фігури
    enemy_color = not color
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == enemy_color:
            if board.is_attacked_by(color, square):
                # Додаємо бонус за атаку на фігуру, пропорційно її вартості
                piece_value = abs(PIECE_VALUES.get(piece.symbol(), 0))
                activity += piece_value * 0.05
    
    return activity

def evaluate_position(board: chess.Board) -> float:
    """
    Комплексна оцінка позиції
    :param board: Шахова дошка
    :return: Оцінка позиції (додатня - перевага білих)
    """
    # Матеріальний баланс
    material = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            material += PIECE_VALUES.get(piece.symbol(), 0)
    
    # Активність фігур
    white_activity = get_piece_activity(board, chess.WHITE)
    black_activity = get_piece_activity(board, chess.BLACK)
    activity = white_activity - black_activity
    
    # Безпека короля
    white_king_safety = king_safety_score(board, chess.WHITE)
    black_king_safety = king_safety_score(board, chess.BLACK)
    king_safety = white_king_safety - black_king_safety
    
    # Структура пішаків
    white_pawn_structure = pawn_structure_score(board, chess.WHITE)
    black_pawn_structure = pawn_structure_score(board, chess.BLACK)
    pawn_structure = white_pawn_structure - black_pawn_structure
    
    # Комбінована оцінка
    evaluation = material + activity + king_safety * 0.5 + pawn_structure * 0.3
    
    return evaluation

def king_safety_score(board: chess.Board, color: chess.Color) -> float:
    """
    Оцінка безпеки короля
    :param board: Шахова дошка
    :param color: Колір (chess.WHITE або chess.BLACK)
    :return: Оцінка безпеки
    """
    king_square = board.king(color)
    if king_square is None:
        return 0.0
    
    safety = 0.0
    enemy_color = not color
    
    # Штраф за відкритість короля
    if board.has_castling_rights(color):
        # Король у безпеці, якщо він не рокірувався, але має права на рокіровку
        safety += 0.2
    elif not board.has_castling_rights(color) and board.castling_rights:
        # Рокіровка вже виконана - бонус
        safety += 0.3
    
    # Перевірка на шахи
    if board.is_check():
        safety -= 0.5
    
    # Кількість ворожих атак на короля
    attack_count = 0
    for attacker in board.attackers(enemy_color, king_square):
        piece = board.piece_at(attacker)
        if piece:
            # Вага атаки залежить від типу фігури
            attack_weight = {
                chess.PAWN: 0.1,
                chess.KNIGHT: 0.2,
                chess.BISHOP: 0.2,
                chess.ROOK: 0.3,
                chess.QUEEN: 0.4
            }.get(piece.piece_type, 0.1)
            safety -= attack_weight
            attack_count += 1
    
    # Додатковий штраф за кілька атак
    if attack_count > 1:
        safety -= 0.1 * attack_count
    
    # Захистні фігури навколо короля
    defender_count = 0
    for square in chess.SQUARES:
        if chess.square_distance(square, king_square) <= 1:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                defender_count += 1
    
    # Бонус за захист
    if defender_count >= 3:
        safety += 0.2
    elif defender_count <= 1:
        safety -= 0.2
    
    return safety

def pawn_structure_score(board: chess.Board, color: chess.Color) -> float:
    """
    Оцінка структури пішаків
    :param board: Шахова дошка
    :param color: Колір (chess.WHITE або chess.BLACK)
    :return: Оцінка структури пішаків
    """
    score = 0.0
    pawns = list(board.pieces(chess.PAWN, color))
    
    if not pawns:
        return 0.0
    
    # Перевірка на подвоєних пішаків
    files = {}
    for pawn in pawns:
        file = chess.square_file(pawn)
        files.setdefault(file, 0)
        files[file] += 1
    
    doubled = sum(count - 1 for count in files.values() if count > 1)
    score -= doubled * 0.2
    
    # Перевірка на ізольованих пішаків
    isolated = 0
    for file in files:
        if not any(f in (file-1, file+1) for f in files if f != file):
            isolated += files[file]
    score -= isolated * 0.15
    
    # Перевірка на прохідних пішаків
    passed = 0
    for pawn in pawns:
        file = chess.square_file(pawn)
        rank = chess.square_rank(pawn)
        
        # Визначаємо напрямок руху
        direction = 1 if color == chess.WHITE else -1
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        # Перевіряємо чи немає ворожих пішаків попереду
        has_enemy_pawns = False
        for enemy_pawn in enemy_pawns:
            enemy_file = chess.square_file(enemy_pawn)
            enemy_rank = chess.square_rank(enemy_pawn)
            
            # Перевіряємо чи ворожий пішак може атакувати або блокувати нашого
            if abs(enemy_file - file) <= 1:
                if color == chess.WHITE:
                    if enemy_rank > rank:
                        has_enemy_pawns = True
                        break
                else:
                    if enemy_rank < rank:
                        has_enemy_pawns = True
                        break
        
        if not has_enemy_pawns:
            passed += 1
            # Додатковий бонус за просунутість
            if color == chess.WHITE:
                advance_bonus = rank / 7.0
            else:
                advance_bonus = (7 - rank) / 7.0
            score += 0.3 + advance_bonus * 0.2
    
    # Бонус за пов'язані пішаки
    connected = 0
    for pawn in pawns:
        pawn_file = chess.square_file(pawn)
        pawn_rank = chess.square_rank(pawn)
        
        # Шукаємо пішака на сусідніх файлах
        for adj_file in (pawn_file-1, pawn_file+1):
            if 0 <= adj_file <= 7:
                # Перевіряємо чи є пішак на тому ж або попередньому рядку
                for adj_rank in (pawn_rank, pawn_rank - direction):
                    if 0 <= adj_rank <= 7:
                        adj_square = chess.square(adj_file, adj_rank)
                        if board.piece_type_at(adj_square) == chess.PAWN and board.color_at(adj_square) == color:
                            connected += 1
                            break
    
    score += connected * 0.1
    
    return score

def get_position_features(board: chess.Board) -> Dict[str, float]:
    """
    Повертає словник з ключовими характеристиками позиції
    :param board: Шахова дошка
    :return: Словник характеристик
    """
    features = {
        "material_balance": 0.0,
        "piece_activity_white": 0.0,
        "piece_activity_black": 0.0,
        "king_safety_white": 0.0,
        "king_safety_black": 0.0,
        "pawn_structure_white": 0.0,
        "pawn_structure_black": 0.0,
        "center_control_white": 0.0,
        "center_control_black": 0.0,
        "development_white": 0.0,
        "development_black": 0.0,
    }
    
    # Матеріальний баланс
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            features["material_balance"] += PIECE_VALUES.get(piece.symbol(), 0)
    
    # Активність фігур
    features["piece_activity_white"] = get_piece_activity(board, chess.WHITE)
    features["piece_activity_black"] = get_piece_activity(board, chess.BLACK)
    
    # Безпека короля
    features["king_safety_white"] = king_safety_score(board, chess.WHITE)
    features["king_safety_black"] = king_safety_score(board, chess.BLACK)
    
    # Структура пішаків
    features["pawn_structure_white"] = pawn_structure_score(board, chess.WHITE)
    features["pawn_structure_black"] = pawn_structure_score(board, chess.BLACK)
    
    # Контроль центру
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    for square in center_squares:
        if board.is_attacked_by(chess.WHITE, square):
            features["center_control_white"] += 0.1
        if board.is_attacked_by(chess.BLACK, square):
            features["center_control_black"] += 0.1
    
    # Розвиток фігур (кількість фігур на початковій позиції)
    back_rank = [0, 1, 2, 3, 4, 5, 6, 7] if board.turn == chess.WHITE else [56, 57, 58, 59, 60, 61, 62, 63]
    for square in back_rank:
        piece = board.piece_at(square)
        if piece and piece.color == chess.WHITE:
            if piece.piece_type != chess.KING:  # Король може залишатися
                features["development_white"] -= 0.1
        if piece and piece.color == chess.BLACK:
            if piece.piece_type != chess.KING:
                features["development_black"] -= 0.1
    
    return features

def is_endgame(board: chess.Board) -> bool:
    """
    Визначає, чи позиція є ендшпілем
    :param board: Шахова дошка
    :return: True, якщо ендшпіль, False - інакше
    """
    # Критерії ендшпілю:
    # 1. Залишилося мало фігур
    if len(board.piece_map()) <= 10:
        return True
    
    # 2. Ферзі обміняні
    if board.pieces(chess.QUEEN, chess.WHITE) and board.pieces(chess.QUEEN, chess.BLACK):
        return False
    
    # 3. У одного з гравців немає ферзя
    if not board.pieces(chess.QUEEN, chess.WHITE) or not board.pieces(chess.QUEEN, chess.BLACK):
        return True
    
    # 4. У гравця залишився тільки король і пішаки
    white_pieces = board.pieces_mask(chess.WHITE, chess.ALL_PIECES)
    black_pieces = board.pieces_mask(chess.BLACK, chess.ALL_PIECES)
    
    # Видалити короля з масок
    white_pieces &= ~board.pieces_mask(chess.WHITE, chess.KING)
    black_pieces &= ~board.pieces_mask(chess.BLACK, chess.KING)
    
    # Якщо у гравця тільки король і пішаки
    if chess.popcount(white_pieces) == 0 and chess.popcount(black_pieces) == 0:
        return False  # Не ендшпіль, якщо у обох тільки королі
    
    if (chess.popcount(white_pieces) == 0 and 
        board.pieces_mask(chess.WHITE, chess.PAWN) == white_pieces):
        return True
    
    if (chess.popcount(black_pieces) == 0 and 
        board.pieces_mask(chess.BLACK, chess.PAWN) == black_pieces):
        return True
    
    return False

def get_position_complexity(board: chess.Board) -> float:
    """
    Оцінює складність позиції (кількість можливих варіантів)
    :param board: Шахова дошка
    :return: Значення складності (0-1)
    """
    # Кількість легальних ходів
    legal_moves = list(board.legal_moves)
    total_moves = len(legal_moves)
    
    # Середня кількість відповідей
    avg_response = 0
    for move in legal_moves[:5]:  # Обмежуємо для продуктивності
        board.push(move)
        avg_response += len(list(board.legal_moves))
        board.pop()
    
    if len(legal_moves) > 0:
        avg_response /= min(5, len(legal_moves))
    
    # Складність = (кількість ходів * середня кількість відповідей) / 100
    complexity = (total_moves * avg_response) / 100.0
    return min(complexity, 1.0)  # Обмежуємо до 1.0

def compare_positions(fen1: str, fen2: str) -> float:
    """
    Порівнює дві позиції та повертає коефіцієнт подібності (0-1)
    :param fen1: FEN першої позиції
    :param fen2: FEN другої позиції
    :return: Коефіцієнт подібності (1 - ідентичні позиції)
    """
    board1 = chess.Board(fen1)
    board2 = chess.Board(fen2)
    
    total_squares = 64
    same_squares = 0
    
    for square in chess.SQUARES:
        piece1 = board1.piece_at(square)
        piece2 = board2.piece_at(square)
        
        if piece1 == piece2:
            same_squares += 1
        elif piece1 is None and piece2 is None:
            same_squares += 1
    
    return same_squares / total_squares

def get_piece_icon(symbol, size=32):
    """
    Повертає іконку фігури для використання в інтерфейсі
    :param symbol: Символ фігури (наприклад, 'K', 'q')
    :param size: Розмір іконки
    :return: QIcon
    """
    from ui.pieces import PieceRenderer  # Локальний імпорт, щоб уникнути циклічних залежностей
    renderer = PieceRenderer()
    pixmap = renderer.get_piece_pixmap(symbol, size)
    if pixmap:
        return QIcon(pixmap)
    return QIcon()