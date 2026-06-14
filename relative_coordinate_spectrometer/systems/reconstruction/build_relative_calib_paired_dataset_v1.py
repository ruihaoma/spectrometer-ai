import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

PROFILE_ROOT = PROJECT_ROOT / "data" / "processed" / "relative_calibration_profiles_v1"
PROFILE_MANIFEST = PROFILE_ROOT / "manifest.csv"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "raw" / "reference_spectrometer"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated" / "relative_calib_paired_dataset_v1"
CALIBRATION_CONFIG = PROJECT_ROOT / "configs" / "calibration" / "relative_calibration_linear_diagnostic_v1.json"

CHANNELS = ["R", "G", "B", "Gray"]
CALIBRATION_REFERENCE_SOURCES = {"hg", "na", "hene"}
PAIRED_DATASET_SOURCES = {"blue_led", "green_led", "red_led", "white_led", "purple_led", "yellow_led"}
EXCLUDED_SOURCES = {"dark"}


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {rel_path(path)}; pass --overwrite to replace it.")
        resolved = path.resolve()
        protected = {
            PROJECT_ROOT.resolve(),
            (PROJECT_ROOT / "data").resolve(),
            (PROJECT_ROOT / "data" / "generated").resolve(),
            Path(resolved.anchor).resolve(),
        }
        if resolved in protected or len(resolved.parts) <= 3:
            raise ValueError(f"Refusing to overwrite protected directory: {resolved}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def parse_reference_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%"):
            continue
        numbers = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", stripped)
        if len(numbers) < 2:
            continue
        try:
            rows.append((float(numbers[0]), float(numbers[1])))
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No numeric rows found in reference spectrum: {path}")
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def reference_file_for_source(source: str) -> Path:
    source_dir = REFERENCE_ROOT / source
    candidates = []
    if source_dir.exists():
        for ext in ("*.txt", "*.csv", "*.dat", "*.tsv"):
            candidates.extend(sorted(source_dir.glob(ext)))
    if not candidates:
        for ext in ("*.txt", "*.csv", "*.dat", "*.tsv"):
            candidates.extend(sorted(REFERENCE_ROOT.rglob(f"*{source}*{ext[1:]}")))
    if not candidates:
        raise FileNotFoundError(f"Missing reference spectrometer file for paired source: {source}")
    return candidates[0]


def normalize_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(max_value) or max_value <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values / max_value).astype(np.float32)


def load_profile(path: Path, wavelength_nm: np.ndarray) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    profile_wavelength = np.asarray(data["wavelength_nm"], dtype=np.float64)
    if profile_wavelength.shape != wavelength_nm.shape or not np.allclose(profile_wavelength, wavelength_nm, atol=1e-6):
        raise RuntimeError(f"Profile wavelength axis mismatch: {rel_path(path)}")
    channels = []
    for channel in CHANNELS:
        values = normalize_max(np.asarray(data[channel], dtype=np.float64))
        channels.append(values)
    arr = np.stack(channels, axis=0).astype(np.float32)
    if arr.shape != (4, 2501):
        raise RuntimeError(f"Unexpected profile shape {arr.shape}: {rel_path(path)}")
    return arr


def load_label(source: str, wavelength_nm: np.ndarray) -> tuple[np.ndarray, Path]:
    ref_path = reference_file_for_source(source)
    ref_wavelength, ref_intensity = parse_reference_spectrum(ref_path)
    valid = np.isfinite(ref_wavelength) & np.isfinite(ref_intensity)
    ref_wavelength = ref_wavelength[valid]
    ref_intensity = ref_intensity[valid]
    if ref_wavelength.size < 2:
        raise RuntimeError(f"Too few reference spectrum points: {rel_path(ref_path)}")
    order = np.argsort(ref_wavelength)
    ref_wavelength = ref_wavelength[order]
    ref_intensity = ref_intensity[order]
    unique_wavelength, unique_idx = np.unique(ref_wavelength, return_index=True)
    ref_wavelength = unique_wavelength
    ref_intensity = ref_intensity[unique_idx]
    label = np.interp(wavelength_nm, ref_wavelength, ref_intensity)
    return normalize_max(label), ref_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_split(n: int) -> dict:
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n < 3:
        return {"train": list(range(n)), "val": [], "test": []}
    train_end = max(1, int(round(n * 0.67)))
    val_end = min(n - 1, train_end + max(1, int(round(n * 0.17))))
    train = list(range(0, train_end))
    val = list(range(train_end, val_end))
    test = list(range(val_end, n))
    if not test:
        test = [train.pop()]
    return {"train": train, "val": val, "test": test}


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "sample_role",
        "array_index",
        "source",
        "image_path",
        "profile_csv",
        "label_reference_path",
        "roi_crop",
        "diagnostic_png",
        "profile_png",
        "calibration_model",
        "calibration_formula",
        "status",
        "comment",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(output_dir: Path) -> dict:
    if not PROFILE_MANIFEST.exists():
        raise FileNotFoundError(f"Missing profile manifest: {rel_path(PROFILE_MANIFEST)}")

    profile_rows = read_csv_rows(PROFILE_MANIFEST)
    calibration_config = json.loads(CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    wavelength_nm = np.round(np.arange(400.0, 650.0 + 0.0001, 0.1, dtype=np.float64), 1).astype(np.float32)
    if wavelength_nm.shape != (2501,):
        raise RuntimeError(f"Unexpected wavelength shape: {wavelength_nm.shape}")

    x_items = []
    y_items = []
    manifest_rows = []
    dark_reference_count = 0
    unconfigured_rows = []
    missing_reference_labels = []
    included_sources = []
    excluded_sources = []
    paired_count = 0
    calibration_reference_count = 0

    for row in profile_rows:
        source = str(row["source"]).strip()
        profile_csv = PROJECT_ROOT / row["profile_csv"]

        if source in CALIBRATION_REFERENCE_SOURCES:
            calibration_reference_count += 1
            excluded_sources.append(source)
            manifest_rows.append(
                {
                    "sample_role": "calibration_reference",
                    "array_index": "",
                    "source": source,
                    "image_path": row["image_path"],
                    "profile_csv": row["profile_csv"],
                    "label_reference_path": rel_path(reference_file_for_source(source)),
                    "roi_crop": row.get("roi_crop", ""),
                    "diagnostic_png": row.get("diagnostic_png", ""),
                    "profile_png": row.get("profile_png", ""),
                    "calibration_model": row.get("calibration_model", "relative_spectral_coordinate_linear_diagnostic_v1"),
                    "calibration_formula": row.get("calibration_formula", calibration_config["formula"]),
                    "status": "excluded_from_x_y",
                    "comment": "calibration reference only: used for s-to-wavelength calibration/validation; not written into x.npy/y.npy. If Hg/Na/He-Ne are needed as line-spectrum training samples, capture independent non-calibration samples.",
                }
            )
            continue

        if source in EXCLUDED_SOURCES:
            dark_reference_count += 1
            excluded_sources.append(source)
            manifest_rows.append(
                {
                    "sample_role": "dark_reference",
                    "array_index": "",
                    "source": source,
                    "image_path": row["image_path"],
                    "profile_csv": row["profile_csv"],
                    "label_reference_path": "",
                    "roi_crop": row.get("roi_crop", ""),
                    "diagnostic_png": row.get("diagnostic_png", ""),
                    "profile_png": row.get("profile_png", ""),
                    "calibration_model": row.get("calibration_model", "relative_spectral_coordinate_linear_diagnostic_v1"),
                    "calibration_formula": row.get("calibration_formula", calibration_config["formula"]),
                    "status": "excluded_from_x_y",
                    "comment": "dark reference only: not written into x.npy/y.npy and not used as a paired label sample.",
                }
            )
            continue

        if source not in PAIRED_DATASET_SOURCES:
            unconfigured_rows.append(
                {
                    "source": source,
                    "image_path": row["image_path"],
                    "reason": "source is not configured as calibration_reference or paired_dataset_sample",
                }
            )
            excluded_sources.append(source)
            continue

        x = load_profile(profile_csv, wavelength_nm.astype(np.float64))
        try:
            y, ref_path = load_label(source, wavelength_nm.astype(np.float64))
        except FileNotFoundError as exc:
            missing_reference_labels.append({"source": source, "image_path": row["image_path"], "error": str(exc)})
            excluded_sources.append(source)
            manifest_rows.append(
                {
                    "sample_role": "paired_dataset_sample",
                    "array_index": "",
                    "source": source,
                    "image_path": row["image_path"],
                    "profile_csv": row["profile_csv"],
                    "label_reference_path": "",
                    "roi_crop": row.get("roi_crop", ""),
                    "diagnostic_png": row.get("diagnostic_png", ""),
                    "profile_png": row.get("profile_png", ""),
                    "calibration_model": row.get("calibration_model", "relative_spectral_coordinate_linear_diagnostic_v1"),
                    "calibration_formula": row.get("calibration_formula", calibration_config["formula"]),
                    "status": "missing_reference_label_excluded_from_x_y",
                    "comment": f"missing standard reference label, excluded from x.npy/y.npy: {exc}",
                }
            )
            continue
        array_index = len(x_items)
        x_items.append(x)
        y_items.append(y)
        paired_count += 1
        included_sources.append(source)
        manifest_rows.append(
            {
                "sample_role": "paired_dataset_sample",
                "array_index": str(array_index),
                "source": source,
                "image_path": row["image_path"],
                "profile_csv": row["profile_csv"],
                "label_reference_path": rel_path(ref_path),
                "roi_crop": row.get("roi_crop", ""),
                "diagnostic_png": row.get("diagnostic_png", ""),
                "profile_png": row.get("profile_png", ""),
                "calibration_model": row.get("calibration_model", "relative_spectral_coordinate_linear_diagnostic_v1"),
                "calibration_formula": row.get("calibration_formula", calibration_config["formula"]),
                "status": "included_in_x_y",
                "comment": row.get("comment", ""),
            }
        )

    if not x_items:
        raise RuntimeError("No paired_dataset_sample rows were available for x.npy/y.npy.")

    x_arr = np.stack(x_items, axis=0).astype(np.float32)
    y_arr = np.stack(y_items, axis=0).astype(np.float32)
    if x_arr.shape[1:] != (4, 2501):
        raise RuntimeError(f"Unexpected x shape: {x_arr.shape}")
    if y_arr.shape[1:] != (2501,):
        raise RuntimeError(f"Unexpected y shape: {y_arr.shape}")

    split = make_split(x_arr.shape[0])
    np.save(output_dir / "x.npy", x_arr)
    np.save(output_dir / "y.npy", y_arr)
    np.save(output_dir / "wavelength_nm.npy", wavelength_nm)
    write_manifest(output_dir / "manifest.csv", manifest_rows)
    write_json(output_dir / "split.json", split)

    hashes = {
        name: sha256_file(output_dir / name)
        for name in ["x.npy", "y.npy", "wavelength_nm.npy", "manifest.csv", "split.json"]
    }
    metadata = {
        "dataset_name": "relative_calib_paired_dataset_v1",
        "created_by": rel_path(SCRIPT_PATH),
        "source_profile_manifest": rel_path(PROFILE_MANIFEST),
        "x_source": "data/processed/relative_calibration_profiles_v1/*/profile.csv",
        "y_source": "data/raw/reference_spectrometer/",
        "output_dir": rel_path(output_dir),
        "calibration_model": "relative_spectral_coordinate_linear_diagnostic_v1",
        "calibration_formula": calibration_config["formula"],
        "calibration_status": "current diagnostic calibration model, not final formal calibration",
        "risk_notes": [
            "Hg partial line error remains relatively large.",
            "Current reference photos include dim, tilted, and edge/background warnings.",
            "Calibration-reference captures are not written into x.npy/y.npy.",
            "If Hg/Na/He-Ne are needed as line-spectrum training samples, capture independent non-calibration samples.",
        ],
        "roles": {
            "calibration_reference": sorted(CALIBRATION_REFERENCE_SOURCES),
            "paired_dataset_sample": sorted(PAIRED_DATASET_SOURCES),
            "dark_reference": sorted(EXCLUDED_SOURCES),
        },
        "counts": {
            "manifest_rows": len(manifest_rows),
            "paired_dataset_samples_in_x_y": paired_count,
            "calibration_reference_rows_not_in_x_y": calibration_reference_count,
            "dark_reference_rows_not_in_x_y": dark_reference_count,
            "unconfigured_rows_not_in_manifest": len(unconfigured_rows),
            "missing_reference_labels": len(missing_reference_labels),
        },
        "included_sources": included_sources,
        "excluded_sources": sorted(set(excluded_sources)),
        "missing_reference_labels": missing_reference_labels,
        "unconfigured_rows": unconfigured_rows,
        "array_shapes": {
            "x.npy": list(x_arr.shape),
            "y.npy": list(y_arr.shape),
            "wavelength_nm.npy": list(wavelength_nm.shape),
        },
        "channel_order": CHANNELS,
        "wavelength_axis": {"start_nm": 400.0, "end_nm": 650.0, "step_nm": 0.1, "point_count": 2501},
        "normalization": {
            "x": "per sample and per channel x/max(x), inherited from profile.csv and rechecked on load",
            "y": "standard reference spectrum interpolated to wavelength axis and divided by max(y)",
        },
        "split": split,
        "hashes": hashes,
        "processing_guards": [
            "Did not read full images.",
            "Did not re-extract ROI.",
            "Did not rerun relative calibration.",
            "Only read existing profile.csv files for x and standard spectrometer files for y.",
        ],
    }
    write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the six real relative-coordinate LED pairs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = project_path(args.output_dir)
    safe_prepare_output_dir(output_dir, overwrite=args.overwrite)
    metadata = build_dataset(output_dir)

    print("relative_calib_paired_dataset_v1 built")
    print("x_shape:", tuple(metadata["array_shapes"]["x.npy"]))
    print("y_shape:", tuple(metadata["array_shapes"]["y.npy"]))
    print("wavelength_shape:", tuple(metadata["array_shapes"]["wavelength_nm.npy"]))
    print("included_sources:", sorted(set(metadata["included_sources"])))
    print("excluded_sources:", sorted(set(metadata["excluded_sources"])))
    print("missing_reference_labels:", metadata["missing_reference_labels"])
    print("output_dir:", rel_path(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
