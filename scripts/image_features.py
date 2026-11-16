"""Per-image colour features and perceptual hashes for both splits.

Two things come out of one pass over the memmaps:

* per-channel mean and std  -> the stain signature of a tile. Patches from one slide
  share a stain signature, so clustering these is the closest thing we have to a
  slide id (Kaggle does not publish one).
* a 56-bit dHash           -> near-duplicate detection. Tiles sampled from
  overlapping positions on the same slide look nearly identical; if they straddle a
  fold boundary the validation score is inflated.

Writes data/cache/features_{train,test}.parquet.

Usage:
    python scripts/image_features.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_PX = 96 * 96


def features(images, chunk: int = 2048) -> pd.DataFrame:
    parts, t0 = [], time.perf_counter()
    for s in range(0, len(images), chunk):
        a = np.ascontiguousarray(images[s:s + chunk])          # uint8 (B,96,96,3)
        b = len(a)

        # per-channel mean / std from integer sums - no float copy of the big array
        s1 = a.sum(axis=(1, 2), dtype=np.int64)                 # (B,3)
        s2 = np.square(a, dtype=np.uint16).sum(axis=(1, 2), dtype=np.int64)
        mean = s1 / N_PX
        std = np.sqrt(np.maximum(s2 / N_PX - mean ** 2, 0))

        # overall (all-channel) mean/std, derived from the same sums
        g_mean = s1.sum(axis=1) / (N_PX * 3)
        g_std = np.sqrt(np.maximum(s2.sum(axis=1) / (N_PX * 3) - g_mean ** 2, 0))

        # 56-bit dHash on an 8x8 block-mean of the grayscale image
        gray = a.mean(axis=3, dtype=np.float32)                 # (B,96,96)
        small = gray.reshape(b, 8, 12, 8, 12).mean(axis=(2, 4))  # (B,8,8)
        bits = (small[:, :, :-1] > small[:, :, 1:]).reshape(b, 56)
        weights = (np.uint64(1) << np.arange(56, dtype=np.uint64))
        dhash = (bits.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)

        parts.append(pd.DataFrame({
            "r_mean": mean[:, 0], "g_mean": mean[:, 1], "b_mean": mean[:, 2],
            "r_std": std[:, 0],   "g_std": std[:, 1],   "b_std": std[:, 2],
            "mean": g_mean, "std": g_std, "dhash": dhash,
        }))
        if (s // chunk) % 20 == 0 and s:
            rate = (s + b) / (time.perf_counter() - t0)
            print(f"  {s + b}/{len(images)}  ({rate:.0f} img/s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    print(f"  {len(df)} images in {time.perf_counter() - t0:.0f}s", flush=True)
    return df


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap", type=Path, default=repo / "data" / "memmap")
    ap.add_argument("--out", type=Path, default=repo / "data" / "cache")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        print(f"{split}:", flush=True)
        index = pd.read_parquet(args.memmap / f"{split}_index.parquet")
        df = features(np.load(args.memmap / f"{split}_images.npy", mmap_mode="r"))
        df.insert(0, "id", index["id"].to_numpy())
        if "label" in index:
            df["label"] = index["label"].to_numpy()
        df.to_parquet(args.out / f"features_{split}.parquet", index=False)
        print(f"  -> features_{split}.parquet", flush=True)


if __name__ == "__main__":
    main()
