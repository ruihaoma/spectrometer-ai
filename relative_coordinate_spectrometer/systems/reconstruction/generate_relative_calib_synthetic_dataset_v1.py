import argparse
import csv
import hashlib
import json
import math
import re
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset" / "relative_calib_synthetic_v1.yaml"
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


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(to_jsonable(data), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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


def parse_reference_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%"):
            continue
        nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", stripped)
        if len(nums) < 2:
            continue
        try:
            rows.append((float(nums[0]), float(nums[1])))
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No numeric rows found: {rel_path(path)}")
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def normalize_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(max_value) or max_value <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values / max_value).astype(np.float32)


def load_reference_library(reference_root: Path, sources: list[str], wavelength_nm: np.ndarray) -> list[dict]:
    library = []
    for source in sources:
        source_dir = reference_root / source
        candidates = []
        if source_dir.exists():
            for ext in ("*.txt", "*.csv", "*.dat", "*.tsv"):
                candidates.extend(sorted(source_dir.glob(ext)))
        if not candidates:
            for ext in ("*.txt", "*.csv", "*.dat", "*.tsv"):
                candidates.extend(sorted(reference_root.rglob(f"*{source}*{ext[1:]}")))
        for path in candidates:
            ref_wl, ref_y = parse_reference_file(path)
            valid = np.isfinite(ref_wl) & np.isfinite(ref_y)
            ref_wl = ref_wl[valid]
            ref_y = ref_y[valid]
            order = np.argsort(ref_wl)
            ref_wl = ref_wl[order]
            ref_y = ref_y[order]
            unique_wl, unique_idx = np.unique(ref_wl, return_index=True)
            y = np.interp(wavelength_nm, unique_wl, ref_y[unique_idx])
            library.append({"source": source, "path": path, "y": normalize_max(y)})
    if not library:
        raise RuntimeError(f"No reference spectra found under {rel_path(reference_root)}")
    return library


def rng_uniform(rng: np.random.Generator, pair: list[float]) -> float:
    return float(rng.uniform(float(pair[0]), float(pair[1])))


def rng_int_range(rng: np.random.Generator, pair: list[int]) -> int:
    lo, hi = int(pair[0]), int(pair[1])
    return int(rng.integers(lo, hi + 1))


def gaussian_kernel(sigma_points: float) -> np.ndarray:
    sigma_points = float(max(sigma_points, 0.0))
    if sigma_points <= 1e-6:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(4.0 * sigma_points)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_points) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_nm(values: np.ndarray, sigma_nm: float, step_nm: float) -> np.ndarray:
    kernel = gaussian_kernel(float(sigma_nm) / float(step_nm))
    if kernel.size == 1:
        return values.astype(np.float64)
    pad = kernel.size // 2
    return np.convolve(np.pad(values.astype(np.float64), pad, mode="edge"), kernel, mode="valid")


def shifted(values: np.ndarray, wavelength_nm: np.ndarray, shift_nm: float) -> np.ndarray:
    return np.interp(wavelength_nm - float(shift_nm), wavelength_nm, values, left=values[0], right=values[-1])


def synth_gaussian(wavelength_nm: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    return float(amp) * np.exp(-0.5 * ((wavelength_nm - float(center)) / max(float(sigma), 1e-6)) ** 2)


def make_programmatic_y(rng: np.random.Generator, cfg: dict, wavelength_nm: np.ndarray, generator_type: str) -> tuple[np.ndarray, str]:
    target_cfg = cfg["target_y"]
    y = np.zeros_like(wavelength_nm, dtype=np.float64)
    if generator_type == "gaussian_single_peak":
        peak_count = 1
        sigma_range = target_cfg["sigma_nm_range"]
    elif generator_type == "blended_broad_narrow":
        broad = synth_gaussian(
            wavelength_nm,
            rng_uniform(rng, target_cfg["center_nm_range"]),
            rng_uniform(rng, target_cfg["broad_sigma_nm_range"]),
            rng.uniform(0.35, 1.0),
        )
        narrow = synth_gaussian(
            wavelength_nm,
            rng_uniform(rng, target_cfg["center_nm_range"]),
            rng_uniform(rng, target_cfg["narrow_sigma_nm_range"]),
            rng.uniform(0.45, 1.2),
        )
        return normalize_max(broad + narrow), "programmatic broad+narrow blend"
    else:
        peak_count = rng_int_range(rng, target_cfg["peak_count_range"])
        sigma_range = target_cfg["sigma_nm_range"]
    centers = []
    for _ in range(peak_count):
        center = rng_uniform(rng, target_cfg["center_nm_range"])
        centers.append(center)
        sigma = rng_uniform(rng, sigma_range)
        amp = float(rng.uniform(0.25, 1.2))
        y += synth_gaussian(wavelength_nm, center, sigma, amp)
    return normalize_max(y), "programmatic peaks at " + ";".join(f"{v:.1f}" for v in centers)


def choose_weighted_type(rng: np.random.Generator, weights: dict) -> str:
    names = list(weights.keys())
    probs = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def make_y(rng: np.random.Generator, cfg: dict, wavelength_nm: np.ndarray, library: list[dict]) -> tuple[np.ndarray, dict]:
    gen_type = choose_weighted_type(rng, cfg["generation"]["generator_weights"])
    if gen_type == "reference":
        ref = library[int(rng.integers(0, len(library)))]
        y = ref["y"].astype(np.float64).copy()
        return normalize_max(y), {"generator_type": gen_type, "reference_sources": ref["source"], "y_source": rel_path(ref["path"])}
    if gen_type == "mixed_reference":
        count = min(len(library), rng_int_range(rng, cfg["target_y"]["mix_reference_count_range"]))
        idx = rng.choice(len(library), size=count, replace=False)
        weights = rng.uniform(0.2, 1.0, size=count)
        weights = weights / weights.sum()
        y = np.zeros_like(wavelength_nm, dtype=np.float64)
        refs = []
        for weight, ref_idx in zip(weights, idx):
            ref = library[int(ref_idx)]
            refs.append(ref["source"])
            y += float(weight) * ref["y"]
        if rng.random() < float(cfg["target_y"].get("splice_probability", 0.0)) and count >= 2:
            cut = int(rng.integers(300, wavelength_nm.size - 300))
            y[:cut] = library[int(idx[0])]["y"][:cut]
            y[cut:] = library[int(idx[1])]["y"][cut:]
        return normalize_max(y), {"generator_type": gen_type, "reference_sources": ";".join(refs), "y_source": "linear_mix_or_splice"}
    y, detail = make_programmatic_y(rng, cfg, wavelength_nm, gen_type)
    return y, {"generator_type": gen_type, "reference_sources": "", "y_source": detail}


def channel_response(wavelength_nm: np.ndarray, channel: str) -> np.ndarray:
    if channel == "B":
        return 0.18 + 0.95 * np.exp(-0.5 * ((wavelength_nm - 455.0) / 52.0) ** 2)
    if channel == "G":
        return 0.18 + 0.95 * np.exp(-0.5 * ((wavelength_nm - 535.0) / 48.0) ** 2)
    if channel == "R":
        return 0.18 + 0.95 * np.exp(-0.5 * ((wavelength_nm - 615.0) / 58.0) ** 2)
    return np.ones_like(wavelength_nm)


def simulate_x(rng: np.random.Generator, cfg: dict, wavelength_nm: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    sim = cfg["diy_x_simulation"]
    step_nm = float(cfg["wavelength"]["step_nm"])
    global_shift = rng_uniform(rng, sim["global_shift_nm_range"])
    base = shifted(y, wavelength_nm, global_shift)
    base = smooth_nm(base, rng_uniform(rng, sim["broadening_sigma_nm_range"]), step_nm)
    base *= rng_uniform(rng, sim["intensity_scale_range"])
    t = np.linspace(-1.0, 1.0, wavelength_nm.size)
    base += rng_uniform(rng, sim["background_offset_range"])
    base += rng_uniform(rng, sim["baseline_drift_range"]) * (t + 1.0) / 2.0
    base += rng_uniform(rng, sim["quadratic_drift_range"]) * (t**2)
    if rng.random() < float(sim.get("dropout_band_probability", 0.0)):
        center = float(rng.uniform(wavelength_nm[120], wavelength_nm[-120]))
        width = rng_uniform(rng, sim["dropout_band_width_nm_range"])
        mask = np.exp(-0.5 * ((wavelength_nm - center) / max(width / 2.35, 1e-6)) ** 2)
        base *= 1.0 - rng.uniform(0.15, 0.55) * mask

    channels = []
    channel_meta = {}
    for channel in ["R", "G", "B"]:
        channel_shift = rng_uniform(rng, sim["channel_shift_nm_range"])
        values = shifted(base, wavelength_nm, channel_shift)
        values = smooth_nm(values, rng_uniform(rng, sim["channel_extra_broadening_sigma_nm_range"]), step_nm)
        response = channel_response(wavelength_nm, channel)
        values = values * response * rng_uniform(rng, sim["channel_gain_range"])
        cross_talk = rng_uniform(rng, sim["cross_talk_range"])
        values = (1.0 - cross_talk) * values + cross_talk * base
        values = smooth_nm(values, rng_uniform(rng, sim["smoothing_sigma_nm_range"]), step_nm)
        noise_std = rng_uniform(rng, sim["gaussian_noise_std_range"])
        values += rng.normal(0.0, noise_std, size=values.shape)
        values = normalize_max(values)
        channels.append(values)
        channel_meta[channel] = {"shift_nm": channel_shift, "noise_std": noise_std}
    r, g, b = channels
    gray = normalize_max(0.299 * r + 0.587 * g + 0.114 * b + rng.normal(0.0, rng_uniform(rng, sim["gaussian_noise_std_range"]) * 0.5, size=r.shape))
    x = np.stack([r, g, b, gray], axis=0).astype(np.float32)
    return x, {"global_shift_nm": global_shift, "channel_meta": channel_meta}


def finite_check(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "sample_index",
        "sample_id",
        "source_type",
        "sample_type",
        "generator_type",
        "reference_sources",
        "y_source",
        "channel_order",
        "normalization",
        "comment",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(config: dict, output_dir: Path, sample_count: int, seed: int) -> dict:
    if int(sample_count) <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    rng = np.random.default_rng(int(seed))
    wl_cfg = config["wavelength"]
    wavelength_nm = np.round(
        np.arange(float(wl_cfg["start_nm"]), float(wl_cfg["end_nm"]) + 0.0001, float(wl_cfg["step_nm"])),
        1,
    ).astype(np.float64)
    if wavelength_nm.shape != (int(wl_cfg["point_count"]),):
        raise RuntimeError(f"Unexpected wavelength shape: {wavelength_nm.shape}")
    library = load_reference_library(project_path(config["paths"]["reference_root"]), config["generation"]["reference_sources"], wavelength_nm.astype(np.float64))

    x_items = []
    y_items = []
    manifest = []
    for idx in range(int(sample_count)):
        y, y_meta = make_y(rng, config, wavelength_nm.astype(np.float64), library)
        x, x_meta = simulate_x(rng, config, wavelength_nm.astype(np.float64), y)
        y = normalize_max(y)
        finite_check("x", x)
        finite_check("y", y)
        x_items.append(x.astype(np.float32))
        y_items.append(y.astype(np.float32))
        manifest.append(
            {
                "sample_index": idx,
                "sample_id": f"synthetic_{idx:06d}",
                "source_type": "synthetic",
                "sample_type": "synthetic",
                "generator_type": y_meta["generator_type"],
                "reference_sources": y_meta["reference_sources"],
                "y_source": y_meta["y_source"],
                "channel_order": ",".join(CHANNELS),
                "normalization": "x per-sample per-channel max; y per-sample max",
                "comment": f"simulated DIY x; global_shift_nm={x_meta['global_shift_nm']:.3f}",
            }
        )

    x_arr = np.stack(x_items, axis=0).astype(np.float32)
    y_arr = np.stack(y_items, axis=0).astype(np.float32)
    np.save(output_dir / "x.npy", x_arr)
    np.save(output_dir / "y.npy", y_arr)
    np.save(output_dir / "wavelength_nm.npy", wavelength_nm)
    write_manifest(output_dir / "manifest.csv", manifest)
    write_yaml(output_dir / "synthetic_config_used.yaml", config)
    metadata = {
        "dataset_name": "relative_calib_synthetic_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_by": rel_path(SCRIPT_PATH),
        "status": "synthetic_dataset",
        "sample_count": int(sample_count),
        "seed": int(seed),
        "x_shape": list(x_arr.shape),
        "y_shape": list(y_arr.shape),
        "wavelength_shape": list(wavelength_nm.shape),
        "channel_order": CHANNELS,
        "wavelength_axis": wl_cfg,
        "reference_root": config["paths"]["reference_root"],
        "reference_library_count": len(library),
        "calibration_status": config.get("warning", {}).get("calibration_status", "diagnostic relative calibration only"),
        "normalization": config["normalization"],
        "hashes": {},
    }
    hashes = {
        name: sha256_file(output_dir / name)
        for name in ["x.npy", "y.npy", "wavelength_nm.npy", "manifest.csv", "synthetic_config_used.yaml"]
    }
    metadata["hashes"] = hashes
    write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate relative-calibration synthetic spectrum dataset v1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-large", action="store_true", help="Required for local sample_count above YAML local_max_sample_count.")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = read_yaml(config_path)
    sample_count = int(args.sample_count if args.sample_count is not None else config["generation"]["sample_count"])
    seed = int(args.seed if args.seed is not None else config["generation"]["seed"])
    output_dir = project_path(args.output_dir if args.output_dir is not None else config["paths"]["output_dir"])
    local_limit = int(config["generation"].get("local_max_sample_count", 20))
    allow_large = bool(args.allow_large or config["generation"].get("allow_large_generation", False))
    if sample_count > local_limit and not allow_large:
        raise SystemExit(f"Refusing to generate {sample_count} samples locally. Limit is {local_limit}. Pass --allow-large on the server.")

    safe_prepare_output_dir(output_dir, overwrite=args.overwrite)
    metadata = build_dataset(config, output_dir, sample_count, seed)
    print("relative_calib_synthetic_v1 generated")
    print("output_dir:", rel_path(output_dir))
    print("x_shape:", tuple(metadata["x_shape"]))
    print("y_shape:", tuple(metadata["y_shape"]))
    print("wavelength_shape:", tuple(metadata["wavelength_shape"]))
    print("sample_count:", metadata["sample_count"])
    print("seed:", metadata["seed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
