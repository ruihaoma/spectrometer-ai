import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: pyyaml") from exc


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset" / "relative_calib_mixed_v1.yaml"
CHANNELS = ["R", "G", "B", "Gray"]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def safe_prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output exists and is not a directory: {path}")
    if is_nonempty_dir(path):
        if not overwrite:
            raise FileExistsError(f"Refusing to write into non-empty output dir: {rel_path(path)}; pass --overwrite to replace it.")
        resolved = path.resolve()
        protected = {
            PROJECT_ROOT.resolve(),
            (PROJECT_ROOT / "data").resolve(),
            (PROJECT_ROOT / "data" / "processed").resolve(),
            (PROJECT_ROOT / "data" / "generated").resolve(),
            Path(resolved.anchor).resolve(),
        }
        if resolved in protected or len(resolved.parts) <= 3:
            raise ValueError(f"Refusing to overwrite protected directory: {resolved}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "sample_index",
        "sample_type",
        "source_dataset",
        "source_index",
        "source",
        "sample_role",
        "generator_type",
        "reference_sources",
        "x_source",
        "y_source",
        "comment",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_split(n: int, cfg: dict) -> dict:
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    indices = np.arange(n, dtype=np.int64)
    if bool(cfg.get("shuffle", True)):
        rng.shuffle(indices)
    train_count = int(round(n * float(cfg.get("train_ratio", 0.8))))
    val_count = int(round(n * float(cfg.get("val_ratio", 0.1))))
    train_count = min(max(train_count, 1), n)
    val_count = min(max(val_count, 1 if n >= 3 else 0), max(0, n - train_count))
    test_count = max(0, n - train_count - val_count)
    if n >= 3 and test_count == 0:
        if val_count > 1:
            val_count -= 1
        else:
            train_count -= 1
        test_count = 1
    train = indices[:train_count].tolist()
    val = indices[train_count : train_count + val_count].tolist()
    test = indices[train_count + val_count :].tolist()
    return {
        "indices": {"train": [int(i) for i in train], "val": [int(i) for i in val], "test": [int(i) for i in test]},
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": int(n)},
        "ratios": {
            "train": float(cfg.get("train_ratio", 0.8)),
            "val": float(cfg.get("val_ratio", 0.1)),
            "test": float(cfg.get("test_ratio", 0.1)),
        },
        "seed": int(cfg.get("seed", 42)),
        "shuffle": bool(cfg.get("shuffle", True)),
    }


def canonical_wavelength(config: dict) -> np.ndarray:
    wl_cfg = config["wavelength"]
    wavelength_nm = np.round(
        np.arange(float(wl_cfg["start_nm"]), float(wl_cfg["end_nm"]) + 0.0001, float(wl_cfg["step_nm"])),
        1,
    ).astype(np.float64)
    if wavelength_nm.shape != (int(wl_cfg["point_count"]),):
        raise RuntimeError(f"Unexpected canonical wavelength shape: {wavelength_nm.shape}")
    return wavelength_nm


def validate_wavelength(real_wl: np.ndarray, synthetic_wl: np.ndarray, target_wl: np.ndarray) -> None:
    if real_wl.shape != synthetic_wl.shape or real_wl.shape != target_wl.shape:
        raise RuntimeError(
            f"wavelength shape mismatch: real={real_wl.shape}, synthetic={synthetic_wl.shape}, target={target_wl.shape}"
        )
    if not np.allclose(real_wl.astype(np.float64), target_wl, atol=1e-4, rtol=1e-7):
        raise RuntimeError("real wavelength_nm.npy does not match the configured 400-650 nm axis")
    if not np.allclose(synthetic_wl.astype(np.float64), target_wl, atol=1e-4, rtol=1e-7):
        raise RuntimeError("synthetic wavelength_nm.npy does not match the configured 400-650 nm axis")


def select_real_rows(real_manifest: list[dict], forbidden: set[str], required_role: str) -> list[dict]:
    selected = []
    for row in real_manifest:
        source = row.get("source", "")
        if source in forbidden:
            continue
        if row.get("sample_role") != required_role or row.get("status") != "included_in_x_y":
            continue
        selected.append(row)
    selected.sort(key=lambda row: int(row["array_index"]))
    return selected


def build_mixed(config: dict, real_dir: Path, synthetic_dir: Path, output_dir: Path) -> dict:
    real_x = np.load(real_dir / "x.npy").astype(np.float32)
    real_y = np.load(real_dir / "y.npy").astype(np.float32)
    real_wl = np.load(real_dir / "wavelength_nm.npy").astype(np.float32)
    synthetic_x = np.load(synthetic_dir / "x.npy").astype(np.float32)
    synthetic_y = np.load(synthetic_dir / "y.npy").astype(np.float32)
    synthetic_wl = np.load(synthetic_dir / "wavelength_nm.npy").astype(np.float32)
    target_wl = canonical_wavelength(config)
    validate_wavelength(real_wl, synthetic_wl, target_wl)

    if real_x.ndim != 3 or real_x.shape[1:] != (4, 2501):
        raise RuntimeError(f"Unexpected real x shape: {real_x.shape}")
    if synthetic_x.ndim != 3 or synthetic_x.shape[1:] != (4, 2501):
        raise RuntimeError(f"Unexpected synthetic x shape: {synthetic_x.shape}")
    if real_y.shape != (real_x.shape[0], 2501) or synthetic_y.shape != (synthetic_x.shape[0], 2501):
        raise RuntimeError(f"Unexpected y shapes: real={real_y.shape}, synthetic={synthetic_y.shape}")

    real_manifest = read_csv_rows(real_dir / "manifest.csv")
    synthetic_manifest = read_csv_rows(synthetic_dir / "manifest.csv")
    real_rows = select_real_rows(
        real_manifest,
        forbidden=set(config["mixing"].get("forbidden_real_sources", [])),
        required_role=str(config["mixing"].get("real_sample_role_required", "paired_dataset_sample")),
    )
    if len(real_rows) != real_x.shape[0]:
        raise RuntimeError(f"Selected real manifest rows {len(real_rows)} do not match real x count {real_x.shape[0]}")
    if len(synthetic_manifest) != synthetic_x.shape[0]:
        raise RuntimeError("Synthetic manifest rows do not match synthetic x count")

    x = np.concatenate([real_x, synthetic_x], axis=0).astype(np.float32)
    y = np.concatenate([real_y, synthetic_y], axis=0).astype(np.float32)
    wavelength_nm = target_wl.astype(np.float64)

    rows = []
    sample_index = 0
    for row in real_rows:
        rows.append(
            {
                "sample_index": sample_index,
                "sample_type": "real",
                "source_dataset": rel_path(real_dir),
                "source_index": row.get("array_index", ""),
                "source": row.get("source", ""),
                "sample_role": row.get("sample_role", ""),
                "generator_type": "",
                "reference_sources": "",
                "x_source": row.get("profile_csv", ""),
                "y_source": row.get("label_reference_path", ""),
                "comment": row.get("comment", ""),
            }
        )
        sample_index += 1
    for row in synthetic_manifest:
        rows.append(
            {
                "sample_index": sample_index,
                "sample_type": "synthetic",
                "source_dataset": rel_path(synthetic_dir),
                "source_index": row.get("sample_index", ""),
                "source": "synthetic",
                "sample_role": "synthetic",
                "generator_type": row.get("generator_type", ""),
                "reference_sources": row.get("reference_sources", ""),
                "x_source": rel_path(synthetic_dir / "x.npy"),
                "y_source": row.get("y_source", ""),
                "comment": row.get("comment", ""),
            }
        )
        sample_index += 1

    split = make_split(x.shape[0], config["split"])
    np.save(output_dir / "x.npy", x)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "wavelength_nm.npy", wavelength_nm)
    write_manifest(output_dir / "manifest.csv", rows)
    write_json(output_dir / "split.json", split)
    metadata = {
        "dataset_name": "relative_calib_mixed_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_by": rel_path(SCRIPT_PATH),
        "real_dataset_dir": rel_path(real_dir),
        "synthetic_dataset_dir": rel_path(synthetic_dir),
        "output_dir": rel_path(output_dir),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "wavelength_shape": list(wavelength_nm.shape),
        "channel_order": CHANNELS,
        "real_sample_count": int(real_x.shape[0]),
        "synthetic_sample_count": int(synthetic_x.shape[0]),
        "total_sample_count": int(x.shape[0]),
        "sample_type_field": "sample_type",
        "forbidden_real_sources": config["mixing"].get("forbidden_real_sources", []),
        "calibration_status": config.get("warning", {}).get("calibration_status", "diagnostic relative calibration only"),
        "risk_notes": [
            "Real samples use current diagnostic relative calibration, not final high-confidence calibration.",
            "Current hg/na/hene/dark calibration-reference captures are not mixed into x.npy/y.npy.",
            "Large-scale generation/training should be run on the server, not this workstation.",
        ],
        "split": split,
        "hashes": {},
    }
    hashes = {
        name: sha256_file(output_dir / name)
        for name in ["x.npy", "y.npy", "wavelength_nm.npy", "manifest.csv", "split.json"]
    }
    metadata["hashes"] = hashes
    write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build relative calibration mixed dataset v1 from real LED6 and synthetic dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--real-dataset-dir", default=None)
    parser.add_argument("--synthetic-dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = read_yaml(config_path)
    real_dir = project_path(args.real_dataset_dir if args.real_dataset_dir else config["paths"]["real_dataset_dir"])
    synthetic_dir = project_path(args.synthetic_dataset_dir if args.synthetic_dataset_dir else config["paths"]["synthetic_dataset_dir"])
    output_dir = project_path(args.output_dir if args.output_dir else config["paths"]["output_dir"])
    safe_prepare_output_dir(output_dir, overwrite=args.overwrite)
    metadata = build_mixed(config, real_dir, synthetic_dir, output_dir)
    print("relative_calib_mixed_v1 generated")
    print("output_dir:", rel_path(output_dir))
    print("x_shape:", tuple(metadata["x_shape"]))
    print("y_shape:", tuple(metadata["y_shape"]))
    print("wavelength_shape:", tuple(metadata["wavelength_shape"]))
    print("real_sample_count:", metadata["real_sample_count"])
    print("synthetic_sample_count:", metadata["synthetic_sample_count"])
    print("split_counts:", metadata["split"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
