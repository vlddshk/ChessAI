import chess
from constants import (
    STARTING_FEN,
    GAME_STATE_ONGOING,
    GAME_STATE_CHECK,
    GAME_STATE_CHECKMATE,
    GAME_STATE_STALEMATE,
    GAME_STATE_DRAW
)

class ChessState:
    def __init__(self, fen=STARTING_FEN):
        """
        Ініціалізація стану гри з заданою FEN позицією
        (за замовчуванням - початкова позиція)
        """
        self.board = chess.Board(fen)
        self.last_move = None
        self.game_state = GAME_STATE_ONGOING
        self.result = None
        self.update_game_state()

    def make_move(self, move_uci):
        """
        Виконання ходу у форматі UCI (наприклад, "e2e4")
        Повертає True, якщо хід успішно виконано, False - якщо хід недійсний
        """
        try:
            move = chess.Move.from_uci(move_uci)
            
            # Перевірка чи хід є легальним
            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                self.update_game_state()
                return True
            return False
        except (ValueError, AssertionError):
            return False

    def update_game_state(self):
        """Оновлює стан гри на основі поточної позиції"""
        if self.board.is_checkmate():
            self.game_state = GAME_STATE_CHECKMATE
            self.result = "0-1" if self.board.turn == chess.WHITE else "1-0"
        elif self.board.is_stalemate():
            self.game_state = GAME_STATE_STALEMATE
            self.result = "1/2-1/2"
        elif self.board.is_insufficient_material():
            self.game_state = GAME_STATE_DRAW
            self.result = "1/2-1/2"
        elif self.board.is_seventyfive_moves():
            self.game_state = GAME_STATE_DRAW
            self.result = "1/2-1/2"
        elif self.board.is_fivefold_repetition():
            self.game_state = GAME_STATE_DRAW
            self.result = "1/2-1/2"
        elif self.board.is_check():
            self.game_state = GAME_STATE_CHECK
        else:
            self.game_state = GAME_STATE_ONGOING

    def is_game_over(self):
        """Перевіряє чи гра завершена"""
        return self.board.is_game_over()

    def get_fen(self):
        """Повертає поточну позицію у форматі FEN"""
        return self.board.fen()

    def get_turn(self):
        """Повертає чия черга ходити (chess.WHITE або chess.BLACK)"""
        return self.board.turn

    def get_last_move(self):
        """Повертає останній виконаний хід у форматі UCI"""
        return self.last_move.uci() if self.last_move else None

    def get_game_state(self):
        """Повертає поточний стан гри (константа GAME_STATE_*)"""
        return self.game_state

    def get_result(self):
        """Повертає результат гри (якщо гра завершена)"""
        return self.result

    def get_legal_moves(self, square=None):
        """
        Повертає список легальних ходів:
        - Якщо вказано клітинку (chess.Square), повертає ходи для фігури на цій клітинці
        - Без аргументів - повертає всі легальні ходи
        """
        if square is not None:
            return [move.uci() for move in self.board.legal_moves if move.from_square == square]
        return [move.uci() for move in self.board.legal_moves]

    def get_piece_at(self, square):
        """Повертає фігуру на заданій клітинці (chess.Square)"""
        return self.board.piece_at(square)

    def get_king_square(self, color):
        """Повертає клітинку короля заданого кольору"""
        return self.board.king(color)


    def reset(self, fen=None): 
        """Скидає гру до початкового стану або заданої FEN позиції"""
        self.board = chess.Board(fen)
        self.last_move = None
        self.game_state = GAME_STATE_ONGOING
        self.result = None
        self.update_game_state()
        if fen is None:
            fen = STARTING_FEN
        self.board = chess.Board(fen)
        self.last_move = None

    def get_board_for_ai(self):
        """Повертає копію об'єкта дошки для використання AI"""
        return self.board.copy()

    def __str__(self):
        """Текстове представлення дошки"""
        return str(self.board)