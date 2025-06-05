import chess
import numpy as np
import time
from .interface import AIInterface
from ..utils.fen_converter import fen_to_tensor
from ..constants import PIECE_VALUES, TENSOR_SHAPE

class MinimaxAI(AIInterface):
    def __init__(self, depth=3, use_nn=False, evaluator=None):
        """
        Ініціалізація шахового AI на основі алгоритму мінімакс з альфа-бета відсіканням
        
        :param depth: Глибина пошуку (кількість півходів)
        :param use_nn: Чи використовувати нейромережу для оцінки позицій
        :param evaluator: Об'єкт для оцінки позицій нейромережею
        """
        self.depth = depth
        self.use_nn = use_nn
        self.evaluator = evaluator
        self.time_limit = 10.0  # Час на хід за замовчуванням (секунди)
        self.verbose = False
        self.nodes_searched = 0
        self.transposition_table = {}
        self.name = "Minimax AI"
        self.version = "1.1"
        self.ready = True
        
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Генерує найкращий хід для поточної позиції
        :param board: Поточна шахова позиція
        :return: Обраний хід
        """
        if not self.ready:
            raise RuntimeError("AI is not ready")
            
        self.nodes_searched = 0
        start_time = time.time()
        self.start_time = start_time
        
        # Очищення таблиці транспозицій для нової позиції
        self.transposition_table.clear()
        
        # Пошук найкращого ходу
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        # Сортування ходів для покращення ефективності альфа-бета відсікання
        legal_moves = self.order_moves(board)
        
        for move in legal_moves:
            if time.time() - start_time > self.time_limit:
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
            
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"Best move: {best_move.uci()}  Value: {best_value:.2f}")
            print(f"Nodes searched: {self.nodes_searched}  Time: {elapsed:.2f}s")
            
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """
        Рекурсивна реалізація алгоритму мінімакс з альфа-бета відсіканням
        """
        self.nodes_searched += 1
        
        # Перевірка на закінчення часу
        if time.time() - self.start_time > self.time_limit:
            return 0
        
        # Перевірка кінця гри
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1000  # Перемога білих
            elif result == "0-1":
                return -1000  # Перемога чорних
            else:
                return 0  # Нічия
        
        # Перевірка глибини
        if depth == 0:
            return self.evaluate_board(board)
        
        # Перевірка таблиці транспозицій
        board_fen = board.fen()
        if board_fen in self.transposition_table:
            entry = self.transposition_table[board_fen]
            if entry["depth"] >= depth:
                return entry["value"]
        
        # Основний алгоритм
        if maximizing_player:
            max_eval = -float('inf')
            for move in self.order_moves(board):
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.transposition_table[board_fen] = {"depth": depth, "value": max_eval}
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.order_moves(board):
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.transposition_table[board_fen] = {"depth": depth, "value": min_eval}
            return min_eval

    def order_moves(self, board):
        """
        Сортує ходи для покращення ефективності альфа-бета відсікання.
        Пріоритет: шахи, взяття, інші ходи.
        """
        legal_moves = list(board.legal_moves)
        move_scores = []
        
        for move in legal_moves:
            score = 0
            
            # Шах - високий пріоритет
            if board.gives_check(move):
                score += 100
            
            # Взяття - пріоритет за значенням фігури
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_symbol = captured_piece.symbol()
                    # Використовуємо абсолютне значення фігури
                    score += abs(PIECE_VALUES.get(piece_symbol, 0))
            
            move_scores.append(score)
        
        # Сортуємо ходи за спаданням оцінки
        sorted_moves = [move for _, move in sorted(zip(move_scores, legal_moves), key=lambda x: x[0], reverse=True)]
        return sorted_moves

    def evaluate_board(self, board):
        """
        Оцінює позицію на дошці.
        Використовує нейромережу, якщо активовано, або матеріальну оцінку.
        """
        if self.use_nn and self.evaluator is not None:
            return self.evaluate_with_nn(board)
        else:
            return self.material_evaluation(board)

    def evaluate_with_nn(self, board):
        """
        Оцінка позиції за допомогою нейромережі
        """
        fen = board.fen()
        evaluation = self.evaluator.evaluate(fen)
        
        # Якщо ходять чорні, інвертуємо оцінку
        if board.turn == chess.BLACK:
            evaluation = -evaluation
            
        return evaluation

    def material_evaluation(self, board):
        """
        Проста матеріальна оцінка позиції.
        Повертає оцінку з точки зору білих.
        """
        score = 0
        
        # Матеріальний баланс
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                value = PIECE_VALUES.get(piece_symbol, 0)
                score += value
        
        # Додаткові фактори
        score += self.mobility_score(board)
        score += self.king_safety_score(board)
        score += self.pawn_structure_score(board)
        
        return score

    def mobility_score(self, board):
        """
        Оцінка мобільності (кількість доступних ходів)
        """
        mobility = len(list(board.legal_moves))
        
        # Нормалізуємо мобільність
        normalized_mobility = mobility / 40  # 40 - приблизна максимальна мобільність
        
        # Білі мають більшу мобільність - додатній бал, чорні - від'ємний
        if board.turn == chess.WHITE:
            return normalized_mobility * 0.5
        else:
            return -normalized_mobility * 0.5

    def king_safety_score(self, board):
        """
        Оцінка безпеки короля
        """
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # Оцінка за розташування короля (центр безпечніший у дебюті, кут - у ендшпілі)
        white_score = self.king_position_score(white_king_square, chess.WHITE, board)
        black_score = self.king_position_score(black_king_square, chess.BLACK, board)
        
        return white_score - black_score

    def king_position_score(self, king_square, color, board):
        """
        Оцінка позиції короля
        """
        if king_square is None:
            return 0
            
        # Ранжуємо клітинки: кут кращий для ендшпілю, центр - для дебюту
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        # Визначаємо фазу гри за кількістю фігур
        piece_count = sum(1 for _ in board.piece_map())
        game_phase = piece_count / 32  # 0-1 (0 - кінець гри, 1 - початок)
        
        # Ідеальна позиція залежно від фази гри
        if game_phase > 0.7:  # Дебют/мітельшпіль
            # Центр кращий
            center_score = 3 - abs(3.5 - file) - abs(3.5 - rank)
        else:  # Ендшпіль
            # Кут кращий
            corner_score = min(file, 7-file) + min(rank, 7-rank)
            center_score = 14 - corner_score  # Інвертуємо
        
        return center_score * 0.1

    def pawn_structure_score(self, board):
        """
        Оцінка структури пішаків
        """
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        white_score = self.evaluate_pawn_structure(white_pawns, chess.WHITE)
        black_score = self.evaluate_pawn_structure(black_pawns, chess.BLACK)
        
        return white_score - black_score

    def evaluate_pawn_structure(self, pawns, color):
        """
        Оцінка структури пішаків для одного кольору
        """
        if not pawns:
            return 0
            
        # Перевірка на подвоєних пішаків
        files = [chess.square_file(sq) for sq in pawns]
        doubled = len(files) - len(set(files))
        
        # Перевірка на ізольованих пішаків
        isolated = 0
        for file in set(files):
            if not any(f in (file-1, file+1) for f in files if f != file):
                isolated += 1
        
        # Підрахунок прохідних пішаків
        passed = 0
        for pawn_sq in pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            # Визначаємо напрямок руху
            direction = 1 if color == chess.WHITE else -1
            enemy_pawns = board.pieces(chess.PAWN, not color)
            
            # Перевіряємо чи немає ворожих пішаків попереду
            if not any(
                chess.square_file(sq) in (file-1, file, file+1) and 
                chess.square_rank(sq) > rank if color == chess.WHITE else chess.square_rank(sq) < rank
                for sq in enemy_pawns
            ):
                passed += 1
        
        # Розрахунок загального балу
        score = passed * 0.5 - doubled * 0.3 - isolated * 0.4
        return score

    def set_difficulty(self, level: int):
        """Встановлення рівня складності (1-легкий, 2-середній, 3-складний)"""
        if level == 1:
            self.depth = 2
        elif level == 2:
            self.depth = 3
        else:  # level 3
            self.depth = 4

    def set_time_limit(self, seconds: float):
        """Встановлення обмеження часу на хід"""
        self.time_limit = seconds

    def use_neural_network(self, use_nn: bool):
        """Вмикає/вимикає використання нейромережі"""
        self.use_nn = use_nn

    def stop_calculation(self):
        """Припинення обчислень (не підтримується в поточній реалізації)"""
        # В цій реалізації ми просто ігноруємо цей виклик
        # Для підтримки потрібна багатопотокова реалізація
        pass

    def set_position(self, board: chess.Board):
        """Встановлення поточної позиції (не потрібно в поточній реалізації)"""
        # В нашому випадку позиція передається безпосередньо в get_move
        pass

    def get_evaluation(self, board: chess.Board) -> float:
        """Повертає оцінку поточної позиції"""
        return self.evaluate_board(board)

    def get_name(self) -> str:
        """Повертає назву AI"""
        return self.name

    def get_version(self) -> str:
        """Повертає версію AI"""
        return self.version

    def is_ready(self) -> bool:
        """Чи готовий AI до генерації ходу"""
        return self.ready

    def set_verbose(self, verbose: bool):
        """Встановлює режим додаткового виводу інформації"""
        self.verbose = verbose

# Допоміжна функція для створення AI з налаштуваннями за замовчуванням
def create_minimax_ai(depth=3, use_nn=False, model_path=None):
    """Фабрична функція для створення MinimaxAI"""
    evaluator = None
    if use_nn and model_path:
        try:
            # Сюди буде додано імпорт та ініціалізацію TFEvaluator
            from .tf_evaluator import TFEvaluator
            evaluator = TFEvaluator(model_path)
        except ImportError:
            print("TensorFlow not available, using material evaluation")
            use_nn = False
        except Exception as e:
            print(f"Error loading neural network: {e}")
            use_nn = False
    
    return MinimaxAI(depth=depth, use_nn=use_nn, evaluator=evaluator)