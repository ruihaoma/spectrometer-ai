import csv
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    Dataset = object


class NpySpectrumDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        split="train",
        x_file="x.npy",
        y_file="y.npy",
        wavelength_file="wavelength_nm.npy",
        split_file="split.json",
        expected_channels=4,
        expected_length=2501,
        normalize=True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.expected_channels = int(expected_channels)
        self.expected_length = int(expected_length)
        self.normalize = bool(normalize)

        self.x_path = self.dataset_dir / x_file
        self.y_path = self.dataset_dir / y_file
        self.wavelength_path = self.dataset_dir / wavelength_file
        self.split_path = self.dataset_dir / split_file

        self.x = np.load(self.x_path).astype(np.float32)
        self.y = np.load(self.y_path).astype(np.float32)
        self.wavelength_nm = np.load(self.wavelength_path).astype(np.float32)

        self._validate_shapes()
        self.indices = self._load_split_indices()

    def _validate_shapes(self):
        errors = []
        if self.x.ndim != 3:
            errors.append(f"x.ndim must be 3, got x.shape={self.x.shape}")
        if self.x.ndim == 3 and self.x.shape[1] != self.expected_channels:
            errors.append(f"x.shape[1] must be {self.expected_channels}, got x.shape={self.x.shape}")
        if self.x.ndim == 3 and self.x.shape[2] != self.expected_length:
            errors.append(f"x.shape[2] must be {self.expected_length}, got x.shape={self.x.shape}")
        if self.y.ndim != 2:
            errors.append(f"y.ndim must be 2, got y.shape={self.y.shape}")
        if self.y.ndim == 2 and self.y.shape[1] != self.expected_length:
            errors.append(f"y.shape[1] must be {self.expected_length}, got y.shape={self.y.shape}")
        if self.wavelength_nm.shape[0] != self.expected_length:
            errors.append(
                f"wavelength_nm.shape[0] must be {self.expected_length}, got wavelength_nm.shape={self.wavelength_nm.shape}"
            )
        if self.x.shape[0] != self.y.shape[0]:
            errors.append(f"x and y sample counts must match, got x.shape={self.x.shape}, y.shape={self.y.shape}")
        if errors:
            raise ValueError("; ".join(errors))

    def _load_manifest_id_map(self):
        manifest_path = self.dataset_dir / "manifest.csv"
        if not manifest_path.exists():
            return {}
        mapping = {}
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_number, row in enumerate(reader):
                sample_id = row.get("sample_id")
                if sample_id:
                    mapping[str(sample_id)] = row_number
        return mapping

    def _sample_id_to_index(self, sample_id, id_map):
        if sample_id in id_map:
            return int(id_map[sample_id])
        suffix = str(sample_id).rsplit("_", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
        raise ValueError(f"Cannot resolve sample id to index: {sample_id}")

    def _resolve_split_values(self, values):
        values = list(values)
        if not values:
            return []
        if all(isinstance(item, (int, np.integer)) for item in values):
            return [int(item) for item in values]
        id_map = self._load_manifest_id_map()
        return [self._sample_id_to_index(str(item), id_map) for item in values]

    def _load_split_indices(self):
        if self.split == "all":
            return list(range(self.x.shape[0]))
        if not self.split_path.exists():
            raise FileNotFoundError(f"split.json not found: {self.split_path}")

        with self.split_path.open("r", encoding="utf-8") as f:
            split_data = json.load(f)

        if isinstance(split_data.get("indices"), dict) and self.split in split_data["indices"]:
            return self._resolve_split_values(split_data["indices"][self.split])
        if isinstance(split_data.get("sample_ids"), dict) and self.split in split_data["sample_ids"]:
            return self._resolve_split_values(split_data["sample_ids"][self.split])
        if self.split in split_data:
            return self._resolve_split_values(split_data[self.split])

        raise ValueError(f"unknown split: {self.split}")

    def __len__(self):
        return len(self.indices)

    def _normalize_x(self, x_item):
        denom = np.maximum(np.max(x_item, axis=1, keepdims=True), 1e-8).astype(np.float32)
        return x_item / denom

    def _normalize_y(self, y_item):
        denom = np.float32(max(float(np.max(y_item)), 1e-8))
        return y_item / denom

    def __getitem__(self, item):
        real_index = int(self.indices[int(item)])
        x_item = self.x[real_index].astype(np.float32, copy=True)
        y_item = self.y[real_index].astype(np.float32, copy=True)

        if self.normalize:
            x_item = self._normalize_x(x_item)
            y_item = self._normalize_y(y_item)

        if torch is None:
            raise ImportError("PyTorch is required for tensor Dataset output. Install torch before training.")

        return {
            "x": torch.from_numpy(x_item).float(),
            "y": torch.from_numpy(y_item).float(),
            "index": real_index,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python npy_spectrum_dataset.py dataset_dir split")
        raise SystemExit(1)
    dataset_dir = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    ds = NpySpectrumDataset(dataset_dir, split=split)
    print("npy_spectrum_dataset.py ready")
    print("split:", split)
    print("length:", len(ds))
    if len(ds) > 0:
        sample = ds[0]
        print("first_real_index:", sample["index"])
        print("x0_shape:", tuple(sample["x"].shape))
        print("y0_shape:", tuple(sample["y"].shape))
