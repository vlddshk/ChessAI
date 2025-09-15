# data_preparation.py
"""
Оптимізована під великі обсяги data_preparation:
- двопрохідна стратегія (index files -> memmap -> finalize)
- багатопроцесорна обробка
- батчинг і streaming (щоб не тримати все в RAM)
- one-hot мітки (white, draw, black)
- LRU кеш для fen_to_tensor у воркері
- збереження метаданих та графіків
"""

import os
import sys
import argparse
import logging
import gzip
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Імпорт проектних утиліт (підлаштовуємо шлях, якщо скрипт запускають з іншої директорії)
try:
    from chess_app.utils.fen_converter import fen_to_tensor
    from chess_app.constants import TENSOR_SHAPE, STARTING_FEN
except Exception:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    from chess_app.utils.fen_converter import fen_to_tensor
    from chess_app.constants import TENSOR_SHAPE, STARTING_FEN

# Логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("data_prep")

# ---------- Утиліти ----------------------------------------------------------------

def result_to_index(result_str):
    if result_str == "1-0":
        return 0, 1.0  # white
    if result_str == "1/2-1/2":
        return 1, 0.0  # draw
    if result_str == "0-1":
        return 2, -1.0  # black
    return None, None

def one_hot_from_index(idx):
    v = np.zeros(3, dtype=np.float32)
    v[idx] = 1.0
    return v

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# ---------- PASS 1: обробка PGN у worker-ах (запис index-файлу) ---------------------

def process_single_pgn(pgn_path, out_index_path, min_elo=1500, max_positions_per_game=30, skip_debut_ply=10, max_games=None):
    """
    Читає PGN, вибирає рівномірні позиції з кожної гри і пише їх у gzipped TSV (one-per-line).
    Повертає кількість збережених позицій.
    Формат строк: FEN \t label_index \t result_float \t ply \t white_elo \t black_elo
    """
    import chess.pgn
    count = 0
    try:
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f_in, gzip.open(out_index_path, "wt", encoding="utf-8") as gz_out:
            writer = csv.writer(gz_out, delimiter="\t", lineterminator="\n")
            game_no = 0
            while True:
                game = chess.pgn.read_game(f_in)
                if game is None:
                    break
                if max_games is not None and game_no >= max_games:
                    break
                game_no += 1

                white_elo = safe_int(game.headers.get("WhiteElo", "0"))
                black_elo = safe_int(game.headers.get("BlackElo", "0"))
                if white_elo < min_elo or black_elo < min_elo:
                    continue

                result_hdr = game.headers.get("Result", "")
                idx, res_float = result_to_index(result_hdr)
                if idx is None:
                    continue

                board = game.board()
                positions = []
                ply = 0
                for mv in game.mainline_moves():
                    ply += 1
                    board.push(mv)
                    # пропускаємо дебют кількістю ply
                    if ply <= skip_debut_ply:
                        continue
                    positions.append((board.fen(), ply))

                if not positions:
                    continue

                # рівномірний вибір індексів
                n_positions = len(positions)
                n_select = min(max_positions_per_game, n_positions)
                if n_select >= n_positions:
                    sel_idxs = list(range(n_positions))
                else:
                    sel_idxs = list(np.round(np.linspace(0, n_positions - 1, n_select)).astype(int))

                for i in sel_idxs:
                    fen, fen_ply = positions[i]
                    writer.writerow([fen, str(idx), str(res_float), str(fen_ply), str(white_elo), str(black_elo)])
                    count += 1
    except Exception as e:
        logger.error(f"Error processing PGN {pgn_path}: {e}")
    logger.info(f"[PASS1] {os.path.basename(pgn_path)} -> {count} positions saved to {out_index_path}")
    return out_index_path, count

# ---------- PASS 2: конвертація FEN->tensor батчами паралельно ---------------------

def _worker_convert_fens(fen_list):
    """
    Виконується у процесах: перетворює список FEN->тензори і повертає ndarray X (N,x,y,ch) і ndarray y_onehot (N,3)
    Тут додаємо LRU-каст для fen_to_tensor через локальний wrapper.
    """
    # Локальний LRU-кеш для прискорення повторних FEN (якщо є дублікати)
    @lru_cache(maxsize=4096)
    def _cached_fen_to_tensor(fen):
        # fen_to_tensor повинен повертати np.array shape == TENSOR_SHAPE
        try:
            t = fen_to_tensor(fen)
            # на випадок, якщо shapes не співпадають — fallback
            if t.shape != TENSOR_SHAPE:
                return fen_to_tensor(STARTING_FEN)
            return t.astype(np.float32)
        except Exception:
            return fen_to_tensor(STARTING_FEN).astype(np.float32)

    import numpy as _np
    N = len(fen_list)
    if N == 0:
        return _np.zeros((0, *TENSOR_SHAPE), dtype=_np.float32), _np.zeros((0, 3), dtype=_np.float32)

    X = _np.zeros((N, *TENSOR_SHAPE), dtype=_np.float32)
    y = _np.zeros((N, 3), dtype=_np.float32)
    for i, (fen, label_idx) in enumerate(fen_list):
        t = _cached_fen_to_tensor(fen)
        X[i] = t
        y[i] = one_hot_from_index(int(label_idx))
    return X, y

# ---------- Помічники для читання index-файлів батчами --------------------------------

def iter_index_files(index_files, batch_size=4096):
    """
    Ітератор, який читає всі gzipped index-файли і повертає пачки (fen,label_index,rest_meta)
    Кожен елемент пачки: (fen, label_index, result_float, ply, white_elo, black_elo)
    """
    for idx_path in index_files:
        with gzip.open(idx_path, "rt", encoding="utf-8") as gz:
            reader = csv.reader(gz, delimiter="\t")
            batch = []
            for row in reader:
                if not row:
                    continue
                fen = row[0]
                label_idx = int(row[1])
                # rest we keep as strings (we'll convert later to metadata)
                rest = (row[2], row[3], row[4], row[5]) if len(row) >= 6 else ("0", "0", "0", "0")
                batch.append((fen, label_idx, rest))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

# ---------- Основна логіка pipeline ----------------------------------------------------

def prepare_dataset(pgn_dir, output_dir, tmp_dir=None,
                    min_elo=1500, max_positions_per_game=30, skip_debut_ply=10,
                    max_games=None, batch_size=4096, workers=None, test_split=0.1):
    """
    1) PASS1: паралельно створюємо index.gz файли (по одному на PGN)
    2) Підрахунок позицій -> створюємо memmap для X і y
    3) PASS2: читаємо index файли батчами, паралельно конвертуємо FEN->tensor, записуємо у memmap
    4) Фіналізація: зберігаємо npz + metadata + графіки
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = tmp_dir or os.path.join(output_dir, "tmp_index")
    os.makedirs(tmp_dir, exist_ok=True)

    pgn_files = sorted([os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.lower().endswith(".pgn")])
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {pgn_dir}")

    workers = workers or max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {workers} worker processes for PASS1")

    # PASS1: each PGN -> index.gz
    index_files = []
    counts = []
    tasks = []
    for p in pgn_files:
        base = Path(p).stem
        out_idx = os.path.join(tmp_dir, f"{base}.idx.tsv.gz")
        tasks.append((p, out_idx, min_elo, max_positions_per_game, skip_debut_ply, max_games))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_single_pgn, *t) for t in tasks]
        for fut in tqdm(futures, desc="PASS1 (parsing PGNs)", unit="file"):
            try:
                out_idx, cnt = fut.result()
                index_files.append(out_idx)
                counts.append(cnt)
            except Exception as e:
                logger.error(f"PASS1 task failed: {e}")

    total_positions = sum(counts)
    logger.info(f"PASS1 complete: total positions collected = {total_positions}")

    if total_positions == 0:
        raise RuntimeError("No positions collected after PASS1. Check filters / PGN files.")

    # Створення memmap-ів
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memmap_X_path = os.path.join(tmp_dir, f"X_memmap_{timestamp}.mmap")
    memmap_y_path = os.path.join(tmp_dir, f"y_memmap_{timestamp}.mmap")

    dtype = np.float32
    X_shape = (total_positions, *TENSOR_SHAPE)
    y_shape = (total_positions, 3)

    logger.info(f"Allocating memmap X shape={X_shape}, y shape={y_shape}")
    X_mm = np.memmap(memmap_X_path, dtype=dtype, mode="w+", shape=X_shape)
    y_mm = np.memmap(memmap_y_path, dtype=dtype, mode="w+", shape=y_shape)

    # PASS2: читаємо index файли батчами, паралельно конвертуємо fen->tensor
    logger.info("PASS2: Converting FEN->tensor in batches and writing to memmap")
    write_ptr = 0
    # збираємо метадані поки записуємо (щоб не тримати усі FEN у пам'яті, зберігаємо мета як список словників)
    metadata = []

    # workers_for_convert можна налаштувати менше ніж загальна кількість CPU
    workers_for_convert = max(1, min(workers, multiprocessing.cpu_count() - 1))

    with ProcessPoolExecutor(max_workers=workers_for_convert) as ex:
        # створюємо генератор батчів
        batch_iter = iter_index_files(index_files, batch_size=batch_size)
        # Завантажуємо послідовно — щоб зберегти порядок і знати offset
        for batch in tqdm(batch_iter, desc="PASS2 (batches)", unit="batch"):
            # batch: list of (fen,label_idx,rest)
            fen_list_for_workers = [(item[0], item[1]) for item in batch]  # keeping only fen,label for worker
            # запускаємо конвертацію у воркерах (один future на batch)
            # Оскільки worker повертає великі ndarrays, варто обмежити batch_size, наприклад 4096
            future = ex.submit(_worker_convert_fens, fen_list_for_workers)
            X_batch, y_batch = future.result()  # синхронно чекаємо і потім одразу пишемо у memmap

            n = X_batch.shape[0]
            if n == 0:
                continue

            # пишемо у memmap
            X_mm[write_ptr:write_ptr + n] = X_batch
            y_mm[write_ptr:write_ptr + n] = y_batch

            # додаємо метадані
            for i, (_, _, rest) in enumerate(batch):
                result_float, ply, welo, belo = rest
                metadata.append({
                    "index": write_ptr + i,
                    "ply": int(ply) if ply is not None else -1,
                    "result_float": float(result_float),
                    "white_elo": int(welo),
                    "black_elo": int(belo)
                })

            write_ptr += n

    assert write_ptr == total_positions, f"Written {write_ptr} positions but expected {total_positions}"

    # Синхронізуємо memmap на диск
    logger.info("Flushing memmaps to disk")
    X_mm.flush()
    y_mm.flush()

    # Фінал: збережемо в npz (компресованому) по шматках, не читаючи все в пам'ять
    out_npz = os.path.join(output_dir, f"chess_dataset_{timestamp}.npz")
    logger.info(f"Saving final dataset to {out_npz} (may take some time)...")

    # Для збереження у npz нам потрібно передати повні масиви.
    # Щоб не завантажувати X у RAM, ми будемо зберігати покроково у тимчасовий файл .npy та викликати savez_compressed потім.
    tmp_X_npy = os.path.join(tmp_dir, f"X_{timestamp}.npy")
    tmp_y_npy = os.path.join(tmp_dir, f"y_{timestamp}.npy")
    # np.lib.format.write_array дозволяє записати memmap прямо у .npy
    np.lib.format.open_memmap(tmp_X_npy, mode="w+", dtype=X_mm.dtype, shape=X_mm.shape)[:] = X_mm[:]
    np.lib.format.open_memmap(tmp_y_npy, mode="w+", dtype=y_mm.dtype, shape=y_mm.shape)[:] = y_mm[:]

    # Тепер можна закомпресувати у npz
    # Для великих масивів це може тривалий час зайняти; альтернативно можна віддати користувачу .npy файли.
    np.savez_compressed(out_npz, X_train=np.lib.format.open_memmap(tmp_X_npy, mode="r"), y_train=np.lib.format.open_memmap(tmp_y_npy, mode="r"))
    logger.info(f"Saved compressed dataset: {out_npz}")

    # Збережемо метадані CSV і графіки
    meta_df = pd.DataFrame(metadata)
    meta_csv = os.path.join(output_dir, f"metadata_{timestamp}.csv")
    meta_df.to_csv(meta_csv, index=False)
    logger.info(f"Saved metadata: {meta_csv}")

    # Графіки
    try:
        counts = meta_df['result_float'].map(lambda v: 1 if v > 0 else (0 if v == 0 else -1)).value_counts().to_dict()
        labels = ['white', 'draw', 'black']
        vals = [counts.get(1, 0), counts.get(0, 0), counts.get(-1, 0)]
        fig, ax = plt.subplots()
        ax.pie(vals, labels=labels, autopct="%1.1f%%")
        pie_path = os.path.join(output_dir, f"results_pie_{timestamp}.png")
        fig.savefig(pie_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved pie chart: {pie_path}")

        # Ply histogram
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.hist(meta_df['ply'].dropna(), bins=50)
        ax2.set_title("Ply distribution")
        ply_path = os.path.join(output_dir, f"ply_hist_{timestamp}.png")
        fig2.savefig(ply_path, bbox_inches='tight')
        plt.close(fig2)
        logger.info(f"Saved ply histogram: {ply_path}")
    except Exception as e:
        logger.warning(f"Could not create charts: {e}")

    logger.info("Dataset preparation complete.")
    return out_npz, meta_csv

# ---------- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Optimized data preparation for large chess datasets")
    p.add_argument("--pgn_dir", default="dataset/pgn", help="Directory with PGN files")
    p.add_argument("--output_dir", default="dataset/processed", help="Output directory")
    p.add_argument("--tmp_dir", default=None, help="Temporary directory (default: output_dir/tmp_index)")
    p.add_argument("--min_elo", type=int, default=1500)
    p.add_argument("--max_positions", type=int, default=30)
    p.add_argument("--skip_debut_ply", type=int, default=10)
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--test_split", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()
    logger.info("Starting optimized data preparation")
    out_npz, meta_csv = prepare_dataset(
        pgn_dir=args.pgn_dir,
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        min_elo=args.min_elo,
        max_positions_per_game=args.max_positions,
        skip_debut_ply=args.skip_debut_ply,
        max_games=args.max_games,
        batch_size=args.batch_size,
        workers=args.workers,
        test_split=args.test_split
    )
    logger.info(f"Done. Dataset saved to: {out_npz}, metadata: {meta_csv}")

if __name__ == "__main__":
    main()
