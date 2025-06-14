import os 
import sys 
import chess
import chess.pgn 
import numpy as np 
import pandas as pd
import multiprocessing
from tqdm import tqdm
from datetime import datetime 
from chess_app.utils.fen_converter import fen_to_tensor
from chess_app.constants import TENSOR_SHAPE, STARTING_FEN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#sys.path.append(os.path.dirname(os.path.dirname(os.path(__file__))))

def process_pgn_file(pgn_path, min_elo=1500, max_positions_per_game=50, max_games=None):
    """
    Обробляє PGN файл, витягуючи позиції та результати партій.
    
    :param pgn_path: Шлях до PGN файлу
    :param min_elo: Мінімальний рейтинг гравців для включення партії
    :param max_positions_per_game: Максимальна кількість позицій для вибірки з однієї партії
    :param max_games: Максимальна кількість партій для обробки (None - без обмежень)
    :return: Список кортежів (FEN, результат)
    """
    positions = []
    game_count = 0
    total_positions = 0

    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
        pbar = tqdm(desc=f"Обробка {os.path.basename(pgn_path)}", unit=" партій")

        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                if max_games is not None and game_count >= max_games:
                    break 

                white_elo = game.headers.get("WhiteElo", "0")
                black_elo = game.headers.get("BlackElo", "0")

                try:
                    white_elo = int(white_elo)
                    black_elo = int(black_elo)
                except ValueError:
                    continue

                if white_elo < min_elo or black_elo < min_elo:
                    continue

                result_str = game.headers.get("Result", "")
                if result_str == "1-0":
                    result = 1.0
                elif result_str == "0-1":
                    result = -1.0
                elif result_str == "1/2-1/2":
                    result = 0.0
                else:
                    continue

                board = game.board()
                game_positions = []

# Пропускаємо початкові позиції (перші 5 ходів)
                move_count = 0
                for move in game.mainline_moves():
                    move_count += 1
                    if move_count <= 10:  # Пропускаємо дебют
                        board.push(move)
                        continue

                    game_positions.append((board.fen(), result))
                    board.push(move)

                if len(game_positions) > max_positions_per_game:
                    indices = np.random.choice(
                        len(game_positions),
                        max_positions_per_game,
                        replace=False
                    )
                    selected_positions = [game_positions[i] for i in indices]
                    positions.extend(selected_positions)
                    total_positions += max_positions_per_game
                else:
                    positions.extend(game_positions)
                    total_positions += len(game_positions)

                game_count += 1
                pbar.update(1)
                pbar.set_postfix({"Позицій": total_positions})

            except Exception as e:
                print(f"\nПомилка при обробці гри: {e}")
                continue 

        pbar.close()
    
    return positions 

def worker_task(args):
    """
    Завдання для робочого процесу в пулі
    """
    pgn_path, min_elo, max_positions, max_games = args 
    return process_pgn_file(pgn_path, min_elo, max_positions, max_games)

def process_pgn_file_directory(pgn_dir, output_dir, min_elo=1500, 
                               max_positions_per_game=30, max_games_per_file=5000
                               ):
    """
    Обробляє всі PGN файли в директорії
    
    :param pgn_dir: Шлях до директорії з PGN файлами
    :param output_dir: Шлях для збереження результатів
    :param min_elo: Мінімальний рейтинг гравців
    :param max_positions_per_game: Максим. позицій на партію
    :param max_games_per_file: Максим. партій на файл
    """
    os.makedirs(output_dir, exist_ok=True)

    pgn_files = [f for f in os.listdir(pgn_dir) if f.lower().endswith('.pgn')]
    print(f"Знайдено {len(pgn_files)} PGN файлів для обробки")

    tasks = [
        (os.path.join(pgn_dir, f), min_elo, max_positions_per_game, max_games_per_file)
        for f in pgn_files
    ]

    all_positions = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(
            pool.imap(worker_task, tasks),
            total=len(tasks),
            desc="Обробка PGN файлів"
        ))

        for result in results:
            all_positions.extend(result)

    print(f"Загалом зібрано {len(all_positions)} позицій")

    save_dataset(all_positions, output_dir)

def save_dataset(positions, output_dir):
    """
    Зберігає набір даних у файли .npz
    
    :param positions: Список кортежів (FEN, результат)
    :param output_dir: Директорія для збереження
    """
    np.random.shuffle(positions)
    split_idx = int(len(positions) * 0.9)
    train_positions = positions[:split_idx]
    test_positions = positions[split_idx:]

    print("Створення тренувального набору...")
    X_train, y_train = create_tensors(train_positions)

    print("Створення тестового набору...")
    X_test, y_test = create_tensors(test_positions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"chess_dataset_{timestamp}.npz")

    print(f"Збереження датасету: {output_path}")
    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print(f"Тренувальний набір: {X_train.shape[0]} позицій")
    print(f"Тестовий набір: {X_test.shape[0]} позицій")
    print(f"Розмірність тензора: {X_train.shape[1:]}")

    save_metadata(positions, output_dir, timestamp)

def create_tensors(positions):
    """
    Створює тензори та мітки зі списку позицій
    
    :param positions: Список кортежів (FEN, результат)
    :return: Тензори та мітки (X, y)
    """
    X = np.zeros((len(positions), *TENSOR_SHAPE), dtype=np.float32)
    y = np.zeros(len(positions), dtype=np.float32)

    for i, (fen, result) in enumerate(tqdm(positions, desc="Створення те=нзорів")):
        try:
            X[i] = fen_to_tensor(fen)
            y[i] = result
        except Exception as e:
            print(f"Помилка при обробці FEN: {fen[:50]}... - {e}")
            X[i] = fen_to_tensor(STARTING_FEN)
            y[i] = 0.0

    return X, y

def save_metadata(positions, output_dir, timestamp):
    """
    Зберігає метадані про набір даних
    
    :param positions: Список позицій
    :param output_dir: Вихідна директорія
    :param timestamp: Мітка часу
    """
    results = [result for _, result in positions]
    white_wins = sum(1 for r in results if r == 1.0)
    black_wins = sum(1 for r in results if r == -1.0)
    draws = sum(1 for r in results if r == 0.0)

    metadata = {
        "total_positions": len(positions),
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "white_win_percent": white_wins / len(positions) * 100,
        "black_win_percent": black_wins / len(positions) * 100,
        "draw_percent": draws / len(positions) * 100,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tensor_shape": TENSOR_SHAPE
    }

    csv_path = os.path.join(output_dir, f"metadata_{timestamp}.csv")
    df = pd.DataFrame([metadata])
    df.to_csv(csv_path, index=False)
    print(f"Метадані збережено: {csv_path}")

    sample_fens = [fen for fen, _ in positions[:1000]]
    sample_path = os.path.join(output_dir, f"sample_fens_{timestamp}.txt")
    with open(sample_path, 'w') as f:
        f.write("\n".join(sample_fens))
    print(f"Зразки FEN збережено: {sample_path}")

def main():
    """Основна функція для підготовки даних"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Підготовка шахових даних для тренування нейромережі')
    parser.add_argument('--pgn_dir', type=str, default='dataset/pgn',
                        help='Директорія з PGN файлами (за замовчуванням: dataset/pgn)')
    parser.add_argument('--output_dir', type=str, default='dataset/processed',
                        help='Директорія для збереження результатів (за замовчуванням: dataset/processed)')
    parser.add_argument('--min_elo', type=int, default=1500,
                        help='Мінімальний рейтинг гравців (за замовчуванням: 1500)')
    parser.add_argument('--max_positions', type=int, default=30,
                        help='Максимальна кількість позицій на партію (за замовчуванням: 30)')
    parser.add_argument('--max_games', type=int, default=None,
                        help='Максимальна кількість партій на файл (за замовчуванням: без обмежень)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ПІДГОТОВКА ДАНИХ ДЛЯ ТРЕНУВАННЯ ШАХОВОЇ НЕЙРОМЕРЕЖІ")
    print("=" * 70)
    print(f"Вхідна директорія: {args.pgn_dir}")
    print(f"Вихідна директорія: {args.output_dir}")
    print(f"Мінімальний рейтинг: {args.min_elo}")
    print(f"Максимальні позиції на партію: {args.max_positions}")
    print(f"Максимальна кількість партій: {args.max_games or 'без обмежень'}")
    print("-" * 70)
    
    # Запуск обробки ??????????????
    process_pgn_file_directory(
        pgn_dir=args.pgn_dir,
        output_dir=args.output_dir,
        min_elo=args.min_elo,
        max_positions_per_game=args.max_positions,
        max_games_per_file=args.max_games
    )
    
    print("\nПідготовка даних успішно завершена!")

if __name__ == "__main__":
    main()