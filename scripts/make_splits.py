"""Build the cross-validation folds, once, for everybody.

Writes splits/folds.csv (id, label, fold). Every experiment in this project reads
that file, so "fold 0" means the same 44k images on every machine.

Two things it does beyond a plain StratifiedKFold:

1. Hashes every image and groups byte-identical duplicates. Duplicates split across
   folds would leak: the model would be validated on an image it trained on.
2. Uses StratifiedGroupKFold so each duplicate group lands in exactly one fold,
   while keeping the positive rate even across folds.

Usage:
    python scripts/make_splits.py                 # 5 folds, seed 42
    python scripts/make_splits.py --folds 10 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def image_hashes(images, chunk: int = 4096) -> np.ndarray:
    """8-byte blake2b per image. Identical pixels -> identical hash."""
    out, t0 = [], time.perf_counter()
    for s in range(0, len(images), chunk):
        block = np.ascontiguousarray(images[s:s + chunk])
        for row in block:
            out.append(hashlib.blake2b(row.tobytes(), digest_size=8).digest())
        if (s // chunk) % 10 == 0 and s:
            done = s + len(block)
            rate = done / (time.perf_counter() - t0)
            print(f"  hashed {done}/{len(images)}  ({rate:.0f} img/s)", flush=True)
    print(f"  hashed {len(out)} images in {time.perf_counter() - t0:.0f}s", flush=True)
    return np.array(out)


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap", type=Path, default=repo / "data" / "memmap")
    ap.add_argument("--out", type=Path, default=repo / "splits" / "folds.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", type=Path, default=repo / "data" / "cache" / "features_train.parquet",
                    help="output of scripts/image_features.py; enables near-duplicate grouping")
    ap.add_argument("--flat-std", type=float, default=10.0,
                    help="tiles below this std all hash alike, so they are never grouped together")
    args = ap.parse_args()

    index = pd.read_parquet(args.memmap / "train_index.parquet")      # id, row, label
    images = np.load(args.memmap / "train_images.npy", mmap_mode="r")
    assert len(index) == len(images), "index and memmap disagree"

    print(f"hashing {len(images)} images to find duplicates...", flush=True)
    h = image_hashes(images)
    uniq, group = np.unique(h, return_inverse=True)
    n_dup_groups = int((np.bincount(group) > 1).sum())
    n_dup_images = int(len(group) - len(uniq))
    print(f"unique images: {len(uniq)} | duplicate groups: {n_dup_groups} "
          f"| redundant copies: {n_dup_images}")

    if n_dup_groups:
        sizes = np.bincount(group)
        big = np.argsort(sizes)[::-1][:5]
        print("largest duplicate groups:", [int(sizes[i]) for i in big])
        # do duplicates ever disagree on their label?
        lab = index["label"].to_numpy()
        conflict = sum(1 for g in np.flatnonzero(sizes > 1) if len(set(lab[group == g])) > 1)
        print(f"duplicate groups with conflicting labels: {conflict}")

    # Near-duplicates: tiles that are not byte-identical but look the same (overlapping
    # samples from one slide). Merge their groups so a tile and its twin cannot end up on
    # opposite sides of a fold boundary.
    if args.features.exists():
        f = pd.read_parquet(args.features)
        assert (f["id"].to_numpy() == index["id"].to_numpy()).all(), "features are out of order"
        dh = f["dhash"].to_numpy()
        usable = (f["std"] >= args.flat_std).to_numpy()   # flat tiles all hash alike
        counts = pd.Series(dh[usable]).value_counts()
        multi = set(counts[counts > 1].index)
        rep_of, merged = {}, 0
        for i in np.flatnonzero(usable):
            if dh[i] in multi:
                g = rep_of.setdefault(dh[i], group[i])
                if group[i] != g:
                    group[i] = g
                    merged += 1
        print(f"near-duplicates: {len(multi)} dHash groups covering "
              f"{int(counts[counts > 1].sum())} tiles -> {merged} regrouped")
    else:
        print(f"no {args.features.name}; skipping near-duplicate grouping "
              f"(run scripts/image_features.py first)")

    index["fold"] = -1
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (_, val) in enumerate(cv.split(index, index["label"], groups=group)):
        index.loc[index.index[val], "fold"] = k
    assert (index["fold"] >= 0).all(), "some rows were not assigned a fold"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    index[["id", "label", "fold"]].to_csv(args.out, index=False)

    summary = index.groupby("fold").agg(n=("label", "size"), positives=("label", "sum"),
                                        pos_rate=("label", "mean")).round(4)
    print(f"\nwrote {args.out}  ({len(index)} rows, {args.folds} folds, seed {args.seed})")
    print(summary.to_string())
    print(f"\noverall positive rate: {index['label'].mean():.4f}")


if __name__ == "__main__":
    main()
