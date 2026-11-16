"""Repack the Kaggle histopathologic-cancer-detection TIFFs into uint8 memmaps.

Reading 277k individual .tif files once per epoch is the main bottleneck in this
project. This script does that read exactly once and writes, per split:

    <dst>/train_images.npy     (N, 96, 96, 3) uint8   raw pixels, lossless
    <dst>/train_index.parquet  id -> row mapping (+ label for train)
    <dst>/stats.json           exact per-channel mean/std over all train pixels

Row order is the sorted id order, so index i means the same image for everyone.

Afterwards a Dataset reads `images[i]` from the memmap instead of opening a file,
which turns a ~10 minute pass into seconds.

Usage:
    python scripts/repack.py                          # data/ -> data/memmap/
    python scripts/repack.py --workers 4 --verify 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

H = W = 96
C = 3


def read_image(path: Path) -> np.ndarray:
    """Decode one patch to a (96, 96, 3) uint8 array."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    if a.shape != (H, W, C) or a.dtype != np.uint8:
        raise ValueError(f"{path.name}: expected (96,96,3) uint8, got {a.shape} {a.dtype}")
    return a


def _worker(job):
    """Decode a contiguous batch straight into its slice of the memmap.

    Workers write disjoint row ranges, so no locking is needed. Returns the
    per-channel sums needed for the global mean/std, accumulated in float64.
    """
    src_dir, ids, mmap_path, start = job
    src_dir, mmap_path = Path(src_dir), Path(mmap_path)
    batch = np.empty((len(ids), H, W, C), np.uint8)
    for i, img_id in enumerate(ids):
        batch[i] = read_image(src_dir / f"{img_id}.tif")

    mm = np.load(mmap_path, mmap_mode="r+")
    mm[start:start + len(ids)] = batch
    mm.flush()
    del mm

    flat = batch.reshape(-1, C).astype(np.float64)
    return flat.sum(axis=0), (flat ** 2).sum(axis=0), flat.shape[0]


def repack(src_dir: Path, dst_dir: Path, split: str, batch: int, workers: int):
    ids = sorted(p.stem for p in src_dir.glob("*.tif"))
    if not ids:
        raise SystemExit(f"no .tif files found in {src_dir}")
    n = len(ids)

    images_path = dst_dir / f"{split}_images.npy"
    # Allocate the full file up front; workers then fill disjoint slices of it.
    mm = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8, shape=(n, H, W, C))
    del mm
    print(f"[{split}] {n} images -> {images_path.name} "
          f"({n * H * W * C / 1e9:.2f} GB), batch={batch}, workers={workers}", flush=True)

    jobs = [(str(src_dir), ids[s:s + batch], str(images_path), s) for s in range(0, n, batch)]
    tot_sum = np.zeros(C, np.float64)
    tot_sqsum = np.zeros(C, np.float64)
    tot_px = 0
    done = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, sq, px in ex.map(_worker, jobs):
            tot_sum += s
            tot_sqsum += sq
            tot_px += px
            done += 1
            if done % 25 == 0 or done == len(jobs):
                imgs = min(done * batch, n)
                el = time.perf_counter() - t0
                rate = imgs / el
                print(f"[{split}] {imgs}/{n}  {rate:7.0f} img/s  "
                      f"elapsed {el / 60:5.1f}m  eta {(n - imgs) / rate / 60:5.1f}m", flush=True)

    index = pd.DataFrame({"id": ids, "row": np.arange(n, dtype=np.int32)})
    print(f"[{split}] wrote {n} rows in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return index, tot_sum, tot_sqsum, tot_px


def verify(src_dir: Path, dst_dir: Path, split: str, index: pd.DataFrame, k: int, seed: int = 0):
    """Re-decode k random patches and compare them byte-for-byte with the memmap.
    """
    if k <= 0:
        return True
    images = np.load(dst_dir / f"{split}_images.npy", mmap_mode="r")
    rows = random.Random(seed).sample(range(len(index)), min(k, len(index)))
    ids = index["id"].to_numpy()
    bad = []
    for r in rows:
        if not np.array_equal(images[r], read_image(src_dir / f"{ids[r]}.tif")):
            bad.append(ids[r])
    if bad:
        print(f"[{split}] VERIFY FAILED on {len(bad)}/{len(rows)}: {bad[:5]}", flush=True)
        return False
    print(f"[{split}] verify OK: {len(rows)} random patches match the originals exactly", flush=True)
    return True


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=repo / "data", help="folder holding train/ test/ and the csvs")
    ap.add_argument("--dst", type=Path, default=repo / "data" / "memmap", help="output folder")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--verify", type=int, default=500, help="random patches to re-check per split")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.src / "train_labels.csv").set_index("id")["label"]
    stats = {"source": str(args.src), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "splits": {}}
    ok = True
    t_all = time.perf_counter()

    for split in args.splits:
        index, s, sq, px = repack(args.src / split, args.dst, split, args.batch, args.workers)

        if split == "train":
            index["label"] = index["id"].map(labels).astype("int8")
            missing = int(index["label"].isna().sum()) if index["label"].isna().any() else 0
            if missing:
                raise SystemExit(f"{missing} ids have no label - aborting")
            mean = s / px
            std = np.sqrt(sq / px - mean ** 2)
            stats["mean_0_1"] = (mean / 255).round(6).tolist()
            stats["std_0_1"] = (std / 255).round(6).tolist()
            stats["mean_0_255"] = mean.round(3).tolist()
            stats["std_0_255"] = std.round(3).tolist()
            stats["pixels"] = int(px)
            print(f"[train] exact per-channel mean (0-1): {np.round(mean / 255, 4).tolist()}", flush=True)
            print(f"[train] exact per-channel std  (0-1): {np.round(std / 255, 4).tolist()}", flush=True)
            print(f"[train] positives: {100 * index['label'].mean():.2f}%", flush=True)

        index.to_parquet(args.dst / f"{split}_index.parquet", index=False)
        stats["splits"][split] = {"n": int(len(index)), "shape": [H, W, C], "dtype": "uint8"}
        ok &= verify(args.src / split, args.dst, split, index, args.verify)

    (args.dst / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n{'ALL OK' if ok else 'FINISHED WITH VERIFY FAILURES'} "
          f"in {(time.perf_counter() - t_all) / 60:.1f} min -> {args.dst}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
