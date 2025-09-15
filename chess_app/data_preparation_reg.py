"""
data_preparation.py
Підготовка даних для регресійної моделі оцінки шахових позицій.

Вхід: директорія з PGN файлами
Вихід: стиснений .npz з X_train, y_train, X_val, y_val, X_test, y_test
Ціль: regression (-1..1) -> -1 чорні виграли, 0 нічия, 1 білі виграли

Підтримка:
 - багатопроцесорність (обробка по файлах)
 - рівномірний вибір позицій в кожній грі
 - меммапи для роботи з великими масивами
 - статистика та графіки
"""

import os
import sys
import argparse
import traceback
import tempfile
import shutil
import json
import math
import multiprocessing as mp
from functools import partial
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import chess
import chess.pgn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Підключаємо локальні утиліти (підкоригуй шляхи, якщо треба)
# Очікується, що структура проєкту має chess_app.utils.fen_converter та chess_app.constants
try:
    from chess_app.utils.fen_converter import fen_to_tensor
    from chess_app.constants import TENSOR_SHAPE, STARTING_FEN
except Exception:
    # якщо виконуєш файл з каталогу верхнього рівня, пробуємо інший шлях
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from chess_app.utils.fen_converter import fen_to_tensor
    from chess_app.constants import TENSOR_SHAPE, STARTING_FEN

# -----------------------
# Допоміжні функції
# -----------------------
def result_to_regression(result_str):
    """Перетворює рядок результату PGN в reg target: '1-0'->1.0, '0-1'->-1.0, '1/2-1/2'->0.0"""
    if result_str == "1-0":
        return 1.0
    if result_str == "0-1":
        return -1.0
    if result_str == "1/2-1/2":
        return 0.0
    return None

def sample_positions_evenly(game_positions, max_per_game):
    """
    Вибирає не більше max_per_game позицій з game_positions (список (fen,result))
    рівномірно по часу партії.
    """
    n = len(game_positions)
    if n == 0:
        return []
    if n <= max_per_game:
        return game_positions
    # рівномірно обрати індекси
    indices = np.linspace(0, n - 1, num=max_per_game, dtype=int)
    return [game_positions[i] for i in indices]

def safe_fen_to_tensor(fen):
    """Конвертує FEN у тензор, при помилці повертає тензор для STARTING_FEN"""
    try:
        return fen_to_tensor(fen)
    except Exception:
        try:
            return fen_to_tensor(STARTING_FEN)
        except Exception:
            # запас: повернути нульовий тензор
            return np.zeros(TENSOR_SHAPE, dtype=np.float32)

# -----------------------
# Обробка одиничного PGN файлу (виконується в воркері)
# -----------------------
def process_single_pgn(pgn_path, out_dir, min_elo=1500, max_positions_per_game=30, max_games=None, skip_debut_moves=10):
    """
    Обробляє один PGN файл і зберігає тимчасовий .npz з позиціями:
      arrays: fens (object array of strings), y (float32)
    Повертає словник з метаданими: {'file': pgn_path, 'out': out_npz, 'count': n, 'games': games_count}
    """
    basename = os.path.basename(pgn_path)
    tmp_name = os.path.join(out_dir, f"{basename}.npz")
    fens_list = []
    y_list = []
    game_count = 0
    positions_total = 0
    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                game_count += 1
                if max_games is not None and game_count > max_games:
                    break

                # рейтинги
                white_elo = game.headers.get("WhiteElo", "0")
                black_elo = game.headers.get("BlackElo", "0")
                try:
                    white_elo = int(white_elo)
                    black_elo = int(black_elo)
                except Exception:
                    # пропускаємо гри без коректного рейтингу
                    continue
                if white_elo < min_elo or black_elo < min_elo:
                    continue

                result_str = game.headers.get("Result", "")
                reg_result = result_to_regression(result_str)
                if reg_result is None:
                    continue

                board = game.board()
                game_positions = []

                # програємо кілька дебютних ходів (skip_debut_moves ходів)
                move_count = 0
                for mv in game.mainline_moves():
                    move_count += 1
                    board.push(mv)
                    if move_count <= skip_debut_moves:
                        continue
                    # зберігаємо позицію перед ходом, або після? Ми зберігаємо позицію до того, як зроблено поточний хід,
                    # тут вже board - після push(mv), але ми зберігаємо board.fen() => позицію після ходу — це теж ок
                    game_positions.append(board.fen())

                # вибираємо рівномірно
                sampled = sample_positions_evenly(game_positions, max_positions_per_game)
                for fen in sampled:
                    fens_list.append(fen)
                    y_list.append(float(reg_result))
                positions_total += len(sampled)

        # Зберігаємо тимчасовий результат
        if len(fens_list) == 0:
            # створимо пустий npz щоб клієнт не ламався
            np.savez_compressed(tmp_name, fens=np.array([], dtype=object), y=np.array([], dtype=np.float32))
        else:
            np.savez_compressed(tmp_name, fens=np.array(fens_list, dtype=object), y=np.array(y_list, dtype=np.float32))

        return {
            'pgn_file': pgn_path,
            'tmp_npz': tmp_name,
            'count': len(fens_list),
            'games': game_count,
            'positions': positions_total
        }

    except Exception as ex:
        # на помилку повертаємо індикатор
        tb = traceback.format_exc()
        return {
            'pgn_file': pgn_path,
            'tmp_npz': None,
            'count': 0,
            'games': game_count,
            'positions': positions_total,
            'error': str(ex),
            'traceback': tb
        }

# -----------------------
# Злиття тимчасових файлів у кінцевий memmap + .npz
# -----------------------
def merge_temp_npzs(temp_files, output_path, tensor_shape, test_split=0.1, val_split=0.1, show_progress=True):
    """
    Зчитує всі тимчасові .npz з fens/y, конвертує fens->тензори і зберігає
    фінальний .npz з train/val/test частинами. Для великих датасетів використовує memmap.
    Повертає шлях до фінального .npz та статистику.
    """
    total = 0
    counts = []
    for t in temp_files:
        try:
            with np.load(t, allow_pickle=True) as d:
                c = len(d['y'])
                counts.append(c)
                total += c
        except Exception:
            counts.append(0)

    if total == 0:
        raise RuntimeError("Немає зібраних позицій для злиття.")

    # готуємо тимчасові memmap-файли
    tmp_dir = os.path.dirname(output_path)
    memmap_X_path = os.path.join(tmp_dir, "X_all.memmap")
    memmap_y_path = os.path.join(tmp_dir, "y_all.memmap")

    X_shape = (total, *tensor_shape)
    y_shape = (total,)

    X_mm = np.memmap(memmap_X_path, dtype=np.float32, mode='w+', shape=X_shape)
    y_mm = np.memmap(memmap_y_path, dtype=np.float32, mode='w+', shape=y_shape)

    idx = 0
    pbar = tqdm(total=total, desc="Converting FEN->tensor", disable=not show_progress)
    for t in temp_files:
        if not t:
            continue
        try:
            with np.load(t, allow_pickle=True) as d:
                fens = d['fens']
                ys = d['y']
                n = len(ys)
                for i in range(n):
                    fen = fens[i]
                    y_mm[idx] = float(ys[i])
                    X_mm[idx] = safe_fen_to_tensor(fen).astype(np.float32)
                    idx += 1
                    pbar.update(1)
        except Exception as ex:
            # у разі помилки — пропускаємо ці записи
            print(f"[WARN] Error reading {t}: {ex}")
            continue
    pbar.close()

    # Переконуємося, що все записано
    assert idx == total, f"Expected {total} entries but wrote {idx}"

    # Тепер робимо змішування та розподіл на train/val/test
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(total)

    X_shuf = X_mm[perm]
    y_shuf = y_mm[perm]

    # Спліт
    test_n = int(total * test_split)
    val_n = int(total * val_split)
    train_n = total - test_n - val_n

    X_train = X_shuf[:train_n]
    y_train = y_shuf[:train_n]
    X_val = X_shuf[train_n:train_n + val_n]
    y_val = y_shuf[train_n:train_n + val_n]
    X_test = X_shuf[train_n + val_n:]
    y_test = y_shuf[train_n + val_n:]

    # Записуємо фінальний npz (стиснутий)
    np.savez_compressed(output_path,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)

    # Видаляємо memmap (файли .memmap залишимо видалити за бажанням)
    del X_mm, y_mm
    try:
        os.remove(memmap_X_path)
        os.remove(memmap_y_path)
    except Exception:
        pass

    stats = {
        'total': total,
        'train': train_n,
        'val': val_n,
        'test': test_n,
        'counts_per_temp': counts
    }
    return output_path, stats

# -----------------------
# Вивід статистики та графіків
# -----------------------
def generate_reports(temp_info_list, final_stats, output_dir):
    """
    Створює кілька графіків та CSV метаданих:
      - розподіл результатів (y)
      - кількість позицій з кожного тимчасового файла
      - довжини (кількість позицій) по файлах
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(output_dir, f"metadata_{timestamp}.json")

    # Збираємо загальні числа
    total_positions = final_stats['total']
    train_n = final_stats['train']
    val_n = final_stats['val']
    test_n = final_stats['test']

    # Результати по тимчасовим npz
    counts = final_stats.get('counts_per_temp', [])
    counts_arr = np.array(counts)

    # Гістограма counts_per_file
    plt.figure(figsize=(8, 4))
    plt.hist(counts_arr[counts_arr > 0], bins=50)
    plt.title("Розподіл позицій по PGN-файлах")
    plt.xlabel("Кількість позицій (в файлі)")
    plt.ylabel("Файлів")
    plt.tight_layout()
    p1 = os.path.join(output_dir, f"positions_per_file_{timestamp}.png")
    plt.savefig(p1)
    plt.close()

    # Статистика наборів
    stats_summary = {
        'total_positions': int(total_positions),
        'train': int(train_n),
        'val': int(val_n),
        'test': int(test_n),
        'files_processed': len(counts_arr),
        'files_with_positions': int((counts_arr > 0).sum()),
        'min_per_file': int(counts_arr.min()) if len(counts_arr)>0 else 0,
        'max_per_file': int(counts_arr.max()) if len(counts_arr)>0 else 0,
        'median_per_file': float(np.median(counts_arr)) if len(counts_arr)>0 else 0.0
    }

    # Розподіл меток (y) — приблизно (підвантажимо невелику вибірку з npz якщо є)
    # Для швидкого огляду підрахуємо сумарно з temp_info_list (вони містять 'tmp_npz' і count)
    y_counts = Counter()
    for info in temp_info_list:
        npz = info.get('tmp_npz')
        if not npz or not os.path.exists(npz):
            continue
        try:
            with np.load(npz, allow_pickle=True) as d:
                ys = d['y']
                # підрахунок значень
                # ys можуть бути [-1,0,1]
                vals, cts = np.unique(ys, return_counts=True)
                for v, c in zip(vals, cts):
                    y_counts[float(v)] += int(c)
        except Exception:
            continue

    # Графік розподілу y
    labels = sorted(y_counts.keys())
    counts = [y_counts[lbl] for lbl in labels]
    plt.figure(figsize=(6,4))
    plt.bar([str(l) for l in labels], counts)
    plt.title("Розподіл результатів позицій (y)")
    plt.xlabel("Цільові значення")
    plt.ylabel("Кількість позицій")
    p2 = os.path.join(output_dir, f"y_distribution_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()

    # Збережемо метадані
    metadata_full = {
        'generated_at': timestamp,
        'summary': stats_summary,
        'y_counts': dict(y_counts)
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_full, f, indent=2, ensure_ascii=False)

    return {
        'positions_per_file_plot': p1,
        'y_distribution_plot': p2,
        'metadata_json': meta_path
    }

# -----------------------
# Основна логіка CLI
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare chess dataset for regression model")
    parser.add_argument('--pgn_dir', default="dataset/pgn", help='Директорія з PGN файлами')
    parser.add_argument('--output_dir', type=str, default='dataset/processed', help='Куди зберігати датасет')
    parser.add_argument('--min_elo', type=int, default=1500, help='Мінімальний рейтинг гравців')
    parser.add_argument('--max_positions_per_game', type=int, default=30, help='Максимум позицій на партію')
    parser.add_argument('--max_games_per_file', type=int, default=None, help='Максимум партій з файлу (None — всі)')
    parser.add_argument('--skip_debut_moves', type=int, default=10, help='Скільки перших ходів пропускати')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count()-1), help='Кількість процесів')
    parser.add_argument('--test_split', type=float, default=0.1, help='Доля тестового набору')
    parser.add_argument('--val_split', type=float, default=0.1, help='Доля валідаційного набору')
    parser.add_argument('--plot', action='store_true', help='Генерувати графіки та метадані')
    parser.add_argument('--no_compress', action='store_true', help='Не стискати фінальний npz (для дебагу)')
    parser.add_argument('--seed', type=int, default=42, help='Seed для shuffle')
    return parser.parse_args()

def main():
    args = parse_args()

    pgn_dir = args.pgn_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Знаходимо PGN файли
    pgn_files = [os.path.join(pgn_dir, f) for f in sorted(os.listdir(pgn_dir)) if f.lower().endswith('.pgn')]
    if not pgn_files:
        print("Не знайдено PGN файлів у вказаній директорії.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_root = os.path.join(output_dir, f"tmp_{timestamp}")
    os.makedirs(tmp_root, exist_ok=True)

    print(f"[INFO] Знайдено {len(pgn_files)} файлів. Робота з {args.workers} воркерами.")
    # готуємо ворк
    worker = partial(process_single_pgn,
                     out_dir=tmp_root,
                     min_elo=args.min_elo,
                     max_positions_per_game=args.max_positions_per_game,
                     max_games=args.max_games_per_file,
                     skip_debut_moves=args.skip_debut_moves)

    temp_results = []
    with mp.Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(worker, pgn_files), total=len(pgn_files), desc="Processing PGN files"):
            temp_results.append(res)

    # збираємо шляхи тимчасових npz
    temp_npzs = [r['tmp_npz'] for r in temp_results if r.get('tmp_npz')]
    total_positions = sum([r.get('count', 0) for r in temp_results])

    print(f"[INFO] Зібрано позицій: {total_positions} (у {len(temp_npzs)} тимчасових файлах)")

    if total_positions == 0:
        print("[WARN] Немає позицій для подальшої обробки. Перевірте фільтри.")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return

    final_name = os.path.join(output_dir, f"chess_dataset_{timestamp}.npz")
    print("[INFO] Конвертація FEN -> тензор та злиття у фінальний файл (може зайняти час)...")
    out_path, stats = merge_temp_npzs(temp_npzs, final_name, TENSOR_SHAPE, test_split=args.test_split, val_split=args.val_split)

    print(f"[INFO] Готово. Файл збережено: {out_path}")
    print(f"Статистика: {stats}")

    report_files = {}
    if args.plot:
        print("[INFO] Генерація графіків та метаданих...")
        report_files = generate_reports(temp_results, stats, output_dir)
        print("[INFO] Звіти створено:", report_files)

    # Очистка тимчасових файлів
    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass

    print("[DONE] Підготовка даних завершена.")

if __name__ == "__main__":
    main()
