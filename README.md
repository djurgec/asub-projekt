# ASUB — Histopathologic Cancer Detection

Kaggle competition: <https://www.kaggle.com/competitions/histopathologic-cancer-detection>

96×96 lymph-node patches; predict the probability that the **centre 32×32 region** contains
tumour tissue. Metric: ROC AUC. 220,025 train / 57,458 test images.


---

## 1. Environment

```
conda create -n asub python=3.11 -y
conda activate asub
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas pyarrow pillow opencv-python-headless scikit-learn "albumentations<2" timm h5py tqdm matplotlib jupyterlab
```


## 2. Data

The dataset is not in this repo (8 GB). Download it from Kaggle as a zip file.

Unzip it so that `data/` looks like this:

```
data/
  train/                  220,025 .tif files
  test/                    57,458 .tif files
  train_labels.csv
  sample_submission.csv
```

## 3. Build the derived files

Run from the repo root, in this order:

```
python scripts/repack.py
```

```
python scripts/image_features.py
```

```
python scripts/make_splits.py
```

What they do:

| script | time | produces |
| --- | --- | --- |
| `repack.py` | ~2 min | `data/memmap/` — all images as one uint8 array per split, plus exact channel stats. Reading 277k small files every epoch is the main bottleneck; this removes it. |
| `image_features.py` | ~2 min | `data/cache/features_*.parquet` — per-tile colour stats and perceptual hashes |
| `make_splits.py` | ~1 min | `splits/folds.csv` — the shared 5-fold split (**already committed; see note below**) |

`repack.py` re-decodes 500 random patches at the end and compares them byte-for-byte with the
originals. It must print `ALL OK`. Once it does, `data/train/` and `data/test/` can be deleted
(~7.7 GB).

## 4. Check it worked

```
python -c "import numpy as np; a=np.load('data/memmap/train_images.npy', mmap_mode='r'); print(a.shape, a.dtype)"
```

Expected: `(220025, 96, 96, 3) uint8`

Notebooks (`jupyter lab`, or open them in VS Code):

- `notebooks/01_suspect_images.ipynb` — blank/degenerate tiles and whether to drop any (decision: no)
- `notebooks/02_stain_variation.ipynb` — stain variation across slides, near-duplicate check

Both read `data/memmap/` and `data/cache/`, so run step 3 first.

---

## Conventions

- Add a new package to the `pip install` line above in the same commit as the code that imports it.

## Dataset facts worth knowing

- Positive rate: **40.50%** (89,117 of 220,025) — balanced enough that no resampling is needed.
- Exact per-channel normalization constants (all 2.03B training pixels, in `data/memmap/stats.json`):
  `mean = [0.7025, 0.5463, 0.6965]`, `std = [0.2389, 0.2821, 0.2163]`
- The label depends on the centre 32×32 only, but use the **full 96×96** as model input — the
  surrounding tissue is informative context. Never augment in a way that moves the centre out of
  frame (flips and 90° rotations are safe; large random crops are not).
- Colour statistics alone reach **0.915 AUC**.
