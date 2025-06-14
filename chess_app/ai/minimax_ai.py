import chess
import numpy as np
import time
from .interface import AIInterface
from utils.fen_converter import fen_to_tensor
from constants import PIECE_VALUES, TENSOR_SHAPE

class MinimaxAI(AIInterface):
    def __init__(self, depth=3, use_nn=False, evaluator=None):
        self.depth = depth
        self.use_nn = use_nn
        self.evaluator = evaluator
        self.time_limit = 10.0
        self.verbose = False
        self.nodes_searched = 0
        self.transposition_table = {}
        self.name = "Minimax AI"
        self.version = "1.1"
        self.ready = True
        self.start_time = 0
        
    def get_move(self, board: chess.Board) -> chess.Move:
        if not self.ready:
            raise RuntimeError("AI is not ready")
            
        self.nodes_searched = 0
        self.start_time = time.time()
        self.transposition_table.clear()
        
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        legal_moves = self.order_moves(board)
        
        for move in legal_moves:
            if time.time() - self.start_time > self.time_limit:
                break
                
            board.push(move)
            move_value = self.minimax(board, self.depth - 1, alpha, beta, False)
            board.pop()
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
                
            alpha = max(alpha, best_value)
            
            if self.verbose:
                print(f"Move: {move.uci()}  Value: {move_value:.2f}  Nodes: {self.nodes_searched}")
        
        if best_move is None and legal_moves:
            best_move = legal_moves[0]
            
        elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"Best move: {best_move.uci()}  Value: {best_value:.2f}")
            print(f"Nodes searched: {self.nodes_searched}  Time: {elapsed:.2f}s")
            
        return best_move

    def minimax(self, board: chess.Board, depth, alpha, beta, maximizing_player):
        self.nodes_searched += 1
        
        # Перевірка на закінчення часу
        if time.time() - self.start_time > self.time_limit:
            return 0
        
        # Перевірка кінця гри
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1000
            elif result == "0-1":
                return -1000
            else:
                return 0
        
        if depth == 0:
            return self.evaluate_board(board)
        
        board_fen = board.fen()
        if board_fen in self.transposition_table:
            entry = self.transposition_table[board_fen]
            if entry["depth"] >= depth:
                return entry["value"]
        
        if maximizing_player:
            max_eval = -float('inf')
            for move in self.order_moves(board):
                # Перевірка часу перед кожним рекурсивним викликом
                if time.time() - self.start_time > self.time_limit:
                    break
                    
                board.push(move)
                eval_val = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            self.transposition_table[board_fen] = {"depth": depth, "value": max_eval}
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.order_moves(board):
                if time.time() - self.start_time > self.time_limit:
                    break
                    
                board.push(move)
                eval_val = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            self.transposition_table[board_fen] = {"depth": depth, "value": min_eval}
            return min_eval

    def order_moves(self, board: chess.Board):
        legal_moves = list(board.legal_moves)
        move_scores = []
        
        for move in legal_moves:
            score = 0
            if board.gives_check(move):
                score += 100
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_symbol = captured_piece.symbol()
                    score += abs(PIECE_VALUES.get(piece_symbol, 0))
            move_scores.append(score)
        
        sorted_moves = [move for _, move in sorted(zip(move_scores, legal_moves), 
                        key=lambda x: x[0], reverse=True)]
        return sorted_moves

    def evaluate_board(self, board):
        if self.use_nn and self.evaluator is not None:
            return self.evaluate_with_nn(board)
        else:
            return self.material_evaluation(board)

    def evaluate_with_nn(self, board: chess.Board):
        fen = board.fen()
        evaluation = self.evaluator.evaluate(fen)
        if board.turn == chess.BLACK:
            evaluation = -evaluation
        return evaluation

    def material_evaluation(self, board: chess.Board):
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                value = PIECE_VALUES.get(piece_symbol, 0)
                score += value
        
        score += self.mobility_score(board)
        score += self.king_safety_score(board)
        score += self.pawn_structure_score(board)
        return score

    def mobility_score(self, board: chess.Board):
        """Оцінка мобільності для поточного гравця"""
        mobility = len(list(board.legal_moves))
        normalized_mobility = mobility / 40
        return normalized_mobility * 0.5

    def king_safety_score(self, board: chess.Board):
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        white_score = self.king_position_score(white_king_square, chess.WHITE, board)
        black_score = self.king_position_score(black_king_square, chess.BLACK, board)
        return white_score - black_score

    def king_position_score(self, king_square, color, board):
        if king_square is None:
            return 0
            
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        piece_count = sum(1 for _ in board.piece_map())
        game_phase = piece_count / 32
        
        # Спрощена оцінка позиції короля
        if game_phase > 0.7:  # Дебют/мітельшпіль
            # Штрафуємо за знаходження на краю дошки
            edge_penalty = 0
            if file in (0, 7) or rank in (0, 7):
                edge_penalty = -0.2
            center_score = 3.5 - max(abs(3.5 - file), abs(3.5 - rank))
            return (center_score * 0.1) + edge_penalty
        else:  # Ендшпіль
            # Заохочуємо активність короля
            return 0.1 * (min(file, 7-file) + min(rank, 7-rank))

    def pawn_structure_score(self, board: chess.Board):
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
        
        white_score = self.evaluate_pawn_structure(white_pawns, chess.WHITE, board)
        black_score = self.evaluate_pawn_structure(black_pawns, chess.BLACK, board)
        return white_score - black_score

    def evaluate_pawn_structure(self, pawns, color, board):
        if not pawns:
            return 0
            
        files = [chess.square_file(sq) for sq in pawns]
        doubled = len(files) - len(set(files))
        isolated = 0
        
        # Перевірка ізольованих пішаків
        unique_files = set(files)
        for file in unique_files:
            adjacent_files = {file-1, file+1}
            if not any(f in unique_files for f in adjacent_files):
                isolated += 1
        
        # Перевірка прохідних пішаків
        passed = 0
        for pawn_sq in pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            enemy_pawns = board.pieces(chess.PAWN, not color)
            
            # Визначаємо діапазон ворожих вертикалей
            enemy_files = {chess.square_file(sq) for sq in enemy_pawns}
            if not any(f in range(file-1, file+2) for f in enemy_files):
                passed += 1
        
        return passed * 0.5 - doubled * 0.3 - isolated * 0.4

    # Решта методів залишаються без змін...
    def set_difficulty(self, level: int):
        if level == 1:
            self.depth = 2
        elif level == 2:
            self.depth = 3
        else:
            self.depth = 4

    def set_time_limit(self, seconds: float):
        self.time_limit = seconds

    def use_neural_network(self, use_nn: bool):
        self.use_nn = use_nn

    def stop_calculation(self):
        pass

    def set_position(self, board: chess.Board):
        pass

    def get_evaluation(self, board: chess.Board) -> float:
        return self.evaluate_board(board)

    def get_name(self) -> str:
        return self.name

    def get_version(self) -> str:
        return self.version

    def is_ready(self) -> bool:
        return self.ready

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

def create_minimax_ai(depth=3, use_nn=False, model_path=None):
    evaluator = None
    if use_nn and model_path:
        try:
            from .tf_evaluator import TFEvaluator
            evaluator = TFEvaluator(model_path)
        except ImportError:
            print("TensorFlow not available, using material evaluation")
            use_nn = False
        except Exception as e:
            print(f"Error loading neural network: {e}")
            use_nn = False
    
    return MinimaxAI(depth=depth, use_nn=use_nn, evaluator=evaluator)