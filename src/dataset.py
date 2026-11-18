
"""
PyTorch Dataset for the histopathology project.

Expected files:

data/
    memmap/
        train_images.npy
        test_images.npy
        train_index.parquet
        test_index.parquet
    train_labels.csv

splits/
    folds.csv

The parquet index created by repack.py contains:
    id, row, label (train)
    id, row (test)

The `row` column is the authoritative mapping between image IDs
and positions in the corresponding NumPy array.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]


class HistopathologyDataset(Dataset):
    """
    Dataset for histopathology image classification.

    Parameters
    ----------
    split : str
        "train", "val", "validation", or "test".

    fold : int, optional
        Validation fold (0-4).
        Training uses the other four folds.
        Validation uses only the selected fold.

    root : str or Path
        Project root directory.

    transform : callable, optional
        Albumentations transformations.

    return_id : bool
        If True, also return the image ID.
    """

    def __init__(
        self,
        split: str = "train",
        fold: Optional[int] = None,
        root: PathLike = ".",
        transform=None,
        mean=(0.7025, 0.5463, 0.6965),
        std=(0.2389, 0.2821, 0.2163),
        return_id: bool = False,
    ):

        self.root = Path(root)
        self.split = split.lower()
        self.transform = transform
        self.return_id = return_id

        if self.split not in {"train", "val", "validation", "test"}:
            raise ValueError(
                "split must be 'train', 'val', 'validation', or 'test'."
            )

        if self.split in {"train", "val", "validation"}:
            if fold is None or int(fold) not in range(5):
                raise ValueError(
                    "For train/validation, provide fold=0, 1, 2, 3, or 4."
                )
            self.fold = int(fold)
        else:
            self.fold = None

        is_test = self.split == "test"
        split_name = "test" if is_test else "train"

        images_path = (
            self.root / "data" / "memmap" /
            f"{split_name}_images.npy"
        )

        index_path = (
            self.root / "data" / "memmap" /
            f"{split_name}_index.parquet"
        )

        if not images_path.exists():
            raise FileNotFoundError(images_path)

        if not index_path.exists():
            raise FileNotFoundError(index_path)

        # Memory mapping: images are not fully loaded into RAM.
        self.images = np.load(images_path, mmap_mode="r")

        if self.images.ndim != 4 or self.images.shape[1:] != (96, 96, 3):
            raise ValueError(
                f"Expected (N, 96, 96, 3), got {self.images.shape}"
            )

        if self.images.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 images, got {self.images.dtype}"
            )

        # Load the image ID -> row mapping.
        index_df = pd.read_parquet(index_path)

        self._require_columns(
            index_df, ["id", "row"], str(index_path)
        )

        index_df = index_df.copy()
        index_df["id"] = index_df["id"].astype(str)
        index_df["row"] = pd.to_numeric(
            index_df["row"], errors="raise"
        ).astype(np.int64)

        if index_df["id"].duplicated().any():
            raise ValueError("Duplicate image IDs in parquet index.")

        if index_df["row"].duplicated().any():
            raise ValueError("Duplicate image rows in parquet index.")

        if len(index_df) != len(self.images):
            raise ValueError(
                "Number of parquet entries does not match image array."
            )

        if not np.array_equal(
            np.sort(index_df["row"].to_numpy()),
            np.arange(len(self.images))
        ):
            raise ValueError(
                "Parquet row values must cover every image row exactly once."
            )

        # Ensure that metadata order corresponds to the NumPy array.
        index_df = index_df.sort_values("row").reset_index(drop=True)

        self.ids = index_df["id"].tolist()

        self.mean = torch.tensor(
            mean, dtype=torch.float32
        ).view(3, 1, 1)

        self.std = torch.tensor(
            std, dtype=torch.float32
        ).view(3, 1, 1)

        if torch.any(self.std <= 0):
            raise ValueError("Normalization std must be positive.")

        # Test dataset does not have labels.
        if is_test:
            self.indices = np.arange(
                len(self.images), dtype=np.int64
            )
            self.labels = None
            return

        self._require_columns(
            index_df, ["label"], str(index_path)
        )

        labels_path = self.root / "data" / "train_labels.csv"
        folds_path = self.root / "splits" / "folds.csv"

        if not labels_path.exists():
            raise FileNotFoundError(labels_path)

        if not folds_path.exists():
            raise FileNotFoundError(folds_path)

        labels_df = pd.read_csv(labels_path)
        folds_df = pd.read_csv(folds_path)

        self._require_columns(
            labels_df, ["id", "label"], str(labels_path)
        )

        self._require_columns(
            folds_df, ["id", "label", "fold"], str(folds_path)
        )

        labels_df = labels_df[["id", "label"]].copy()
        folds_df = folds_df[["id", "label", "fold"]].copy()

        labels_df["id"] = labels_df["id"].astype(str)
        folds_df["id"] = folds_df["id"].astype(str)

        labels_df["label"] = pd.to_numeric(
            labels_df["label"], errors="raise"
        ).astype(int)

        folds_df["label"] = pd.to_numeric(
            folds_df["label"], errors="raise"
        ).astype(int)

        folds_df["fold"] = pd.to_numeric(
            folds_df["fold"], errors="raise"
        ).astype(int)

        if labels_df["id"].duplicated().any():
            raise ValueError("Duplicate IDs in train_labels.csv.")

        if folds_df["id"].duplicated().any():
            raise ValueError("Duplicate IDs in folds.csv.")

        # Check labels from repack.parquet against train_labels.csv.
        parquet_labels = index_df[["id", "label"]].copy()

        parquet_labels["label"] = pd.to_numeric(
            parquet_labels["label"], errors="raise"
        ).astype(int)

        label_check = parquet_labels.merge(
            labels_df,
            on="id",
            how="left",
            suffixes=("_parquet", "_csv"),
            validate="one_to_one",
        )

        if label_check["label_csv"].isna().any():
            raise ValueError(
                "Some image IDs are missing from train_labels.csv."
            )

        if not (
            label_check["label_parquet"] ==
            label_check["label_csv"]
        ).all():
            raise ValueError(
                "Labels in parquet and train_labels.csv do not match."
            )

        # Match fold assignments using image IDs.
        metadata = index_df[["id", "row", "label"]].merge(
            folds_df[["id", "label", "fold"]],
            on="id",
            how="left",
            suffixes=("_parquet", "_folds"),
            validate="one_to_one",
        )

        if metadata["fold"].isna().any():
            raise ValueError(
                "Some training images have no fold assignment."
            )

        if not (
            metadata["label_parquet"] ==
            metadata["label_folds"]
        ).all():
            raise ValueError(
                "Labels in folds.csv and parquet do not match."
            )

        if not metadata["fold"].between(0, 4).all():
            raise ValueError(
                "Fold values must be between 0 and 4."
            )

        # Select training or validation samples.
        if self.split == "train":
            selected = metadata[
                metadata["fold"] != self.fold
            ]
        else:
            selected = metadata[
                metadata["fold"] == self.fold
            ]

        selected = selected.sort_values("row")

        self.indices = selected["row"].to_numpy(
            dtype=np.int64
        )

        self.labels = selected["label_parquet"].to_numpy(
            dtype=np.int64
        )

        self.selected_ids = selected["id"].tolist()

        if len(self.indices) == 0:
            raise ValueError(
                f"No images found for {self.split}, fold={self.fold}."
            )

    @staticmethod
    def _require_columns(df, required, description):
        missing = [
            column for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{description} is missing columns {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        row = int(self.indices[index])

        # Copy image from read-only memmap.
        image = np.array(
            self.images[row],
            copy=True
        )

        # Apply Albumentations transformations.
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # Convert image to PyTorch tensor.
        if isinstance(image, torch.Tensor):

            tensor = image

            if tensor.ndim != 3:
                raise ValueError("Expected a 3D image tensor.")

            if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)

            tensor = tensor.float()

            if tensor.numel() and tensor.max().item() > 1.0:
                tensor = tensor / 255.0

        else:

            image = np.asarray(image)

            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Expected HWC RGB image, got {image.shape}"
                )

            tensor = torch.from_numpy(
                np.ascontiguousarray(image)
            )

            tensor = tensor.permute(2, 0, 1).float()
            tensor = tensor / 255.0

        # Normalize using statistics from README.
        tensor = (tensor - self.mean) / self.std

        # Test data has no labels.
        if self.split == "test":

            if self.return_id:
                return tensor, self.ids[row]

            return tensor

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        if self.return_id:
            return tensor, label, self.selected_ids[index]

        return tensor, label
