import chess 
import numpy as np
#from chess_app.constants import PIECE_TO_INDEX, TENSOR_SHAPE 
from constants import PIECE_TO_INDEX, TENSOR_SHAPE 

def fen_to_tensor(fen):
    """
    Перетворює FEN-нотацію шахової позиції у тензор 8x8x12.
    
    :param fen: Рядок у форматі FEN
    :return: Тензор numpy розміром 8x8x12
    """
    tensor = np.zeros(TENSOR_SHAPE, dtype=np.float32)

    try:
        board = chess.Board(fen)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Отримуємо індекс каналу для цієї фігури
                channel = PIECE_TO_INDEX[piece.symbol()]

                # Конвертуємо шаховий квадрат у координати тензора
                rank, file = square_to_tensor_coords(square)
                
                tensor[rank, file, channel] = 1.0

        # Додаємо додаткові канали (опціонально)
        # add_additional_channels(tensor, board)
    
    except Exception as e:
        print(f"Помилка при конвертації FEN: {fen}")
        print(f"Повідомлення про помилку: {str(e)}")

        return np.zeros(TENSOR_SHAPE, dtype=np.float32)
    
    return tensor 

def square_to_tensor_coords(square):
    """
    Конвертує шаховий квадрат у координати тензора (rank, file).
    
    У шаховій дошці:
    - Відлік починається знизу (a1 - лівий нижній кут)
    - У тензорі: рядок 0 - верхній, рядок 7 - нижній
    
    :param square: Шаховий квадрат (0-63)
    :return: Кортеж (rank, file) для тензора
    """
    # Отримуємо шахові координати
    chess_rank = chess.square_rank(square)  # 0 (rank 1) до 7 (rank 8)
    chess_file = chess.square_file(square)  # 0 (a) до 7 (h)
    
    # Інвертуємо ранг: у тензорі 0 - верхній ряд, 7 - нижній

    tensor_rank = 7- chess_rank 
    tensor_file = chess_file 

    return tensor_rank, tensor_file 

def add_additional_channels(tensor, board):
    """
    Додає додаткові канали з інформацією про стан гри.
    Ця функція може бути розширена для додавання додаткової інформації.
    
    :param tensor: Тензор, до якого додаються канали
    :param board: Шахова дошка (об'єкт chess.Board)
    """
    # Канал 12: Чия черга ходити (0 - чорні, 1 - білі)
    turn_channel = 12
    tensor[:, :, turn_channel] = 1.0 if board.turn == chess.WHITE else 0.0

    # Канал 13: Права на рокіровку
    castling_channel = 13
    castling_rights = 0

    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights += 0.25
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights += 0.25
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights += 0.25
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights += 0.25

    tensor[:, :, castling_channel] = castling_rights 

    # Канал 14: Поле для взяття на проході (якщо є)
    en_passant_channel = 14
    if board.ep_square:
        rank, file = square_to_tensor_coords(board.ep_square)
        tensor[rank, file, en_passant_channel] = 1.0

    # Канал 15: Лічильник половини ходів (нормалізований)
    halfmove_channel = 15
    halfmove_value = min(board.halfmove_clock / 50.0, 1.0)
    tensor[:, :, halfmove_channel] = halfmove_value 

def tensor_to_fen(tensor):
    """
    Конвертує тензор назад у FEN-нотацію (експериментально).
    
    :param tensor: Тензор розміром 8x8xN (мінімум 12 каналів)
    :return: Рядок FEN
    """
    board = chess.Board()
    board.clear()

    for rank in range(8):
        for file in range(8):
            square = tensor_coords_to_square(rank, file)

            piece_value = -1
            piece_symbol = None 

            for symbol, channel in PIECE_TO_INDEX.items():
                if tensor[rank, file, channel] > piece_value:
                    piece_value = tensor[rank, file, channel]
                    piece_symbol = symbol 

            if piece_symbol and piece_value > 0.5:
                piece = chess.Piece.from_symbol(piece_symbol)
                board.set_piece_at(square, piece)

    if tensor.shape[2] > 12:
        white_turn = tensor[0, 0, 12] > 0.5
        board.turn = chess.WHITE if white_turn else chess.BLACK

        # Права на рокіровку (спрощено)
        castling_rights = ""
        if tensor[0, 0, 13] > 0.1:  # Kingside white
            castling_rights += "K"
        if tensor[0, 0, 13] > 0.2:  # Queenside white
            castling_rights += "Q"
        if tensor[0, 0, 13] > 0.3:  # Kingside black
            castling_rights += "k"
        if tensor[0, 0, 13] > 0.4:  # Queenside black
            castling_rights += "q"
        board.set_castling_fen(castling_rights if castling_rights else "-")
    
    # Отримуємо FEN без інформації про ходи
    fen = board.fen().split(' ')[0]
    return fen + " w - - 0 1"

def tensor_coords_to_square(tensor_rank, tensor_file):
    """
    Конвертує координати тензора у шаховий квадрат.
    
    :param tensor_rank: Рядок у тензорі (0 - верхній)
    :param tensor_file: Колонка у тензорі (0 - ліва)
    :return: Шаховий квадрат (0-63)
    """
    # Інвертуємо ранг: тензорний 0 -> шаховий ранг 7 (8-й ранг)
    chess_rank = 7 - tensor_rank
    chess_file = tensor_file
    return chess_rank * 8 + chess_file

def test_conversion(fen):
    """
    Тестує конвертацію FEN -> тензор -> FEN
    
    :param fen: Вхідний FEN для тестування
    """
    print(f"Оригінальний FEN: {fen}")
    
    # Конвертуємо FEN у тензор
    tensor = fen_to_tensor(fen)
    
    # Конвертуємо тензор назад у FEN
    reconstructed_fen = tensor_to_fen(tensor)
    
    print(f"Відновлений FEN: {reconstructed_fen}")
    print("Співпадають:", fen.split()[0] == reconstructed_fen.split()[0])
    print("-" * 50)

if __name__ == "__main__":
    # Тестуємо на різних позиціях
    test_conversion("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")  # Початкова позиція
    test_conversion("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")  # Сицилійський захист
    test_conversion("8/8/8/4k3/8/8/8/R3K3 w Q - 0 1")  # Нестандартна позиція
    test_conversion("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")  # Складніша позиція
    test_conversion("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")  # Позиція з пішаками