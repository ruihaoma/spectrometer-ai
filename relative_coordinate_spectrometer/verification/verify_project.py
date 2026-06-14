import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Support direct execution from the project directory.
from shared.spectrum_unet_transformer_1d import SpectrumUNetTransformer1D  # noqa: E402
from systems.calibration import extract_profiles_with_relative_calibration as profiles  # noqa: E402
from systems.reconstruction import build_relative_calib_mixed_dataset_v1 as mixed_builder  # noqa: E402
from systems.reconstruction import build_relative_calib_paired_dataset_v1 as paired_builder  # noqa: E402
from systems.reconstruction import generate_relative_calib_synthetic_dataset_v1 as synthetic_builder  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = "dda1be29ae42f424d4ef000138c7e9de83d10cfbd4d968cef7304ea8f8b44ae0"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [REPOSITORY_ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def verify_tracked_files() -> dict:
    text_suffixes = {
        ".bat",
        ".css",
        ".csv",
        ".html",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".ps1",
        ".py",
        ".txt",
        ".yaml",
        ".yml",
    }
    text_names = {".gitattributes", ".gitignore"}
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    counts = Counter()

    files = tracked_files()
    if not files:
        fail("Git reports no tracked files.")

    for path in files:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        if not path.is_file():
            fail(f"Tracked file is missing: {relative}")
        if path.stat().st_size == 0:
            fail(f"Tracked file is empty: {relative}")

        suffix = path.suffix.lower()
        counts[suffix or path.name] += 1
        text = None
        if suffix in text_suffixes or path.name in text_names:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                fail(f"Tracked text file is not UTF-8: {relative}: {exc}")

        try:
            if suffix == ".json":
                json.loads(text)
            elif suffix in {".yaml", ".yml"}:
                yaml.safe_load(text)
            elif suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    list(csv.reader(handle))
            elif suffix in image_suffixes:
                encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
                if cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) is None:
                    fail(f"Tracked image cannot be decoded: {relative}")
        except (csv.Error, json.JSONDecodeError, yaml.YAMLError) as exc:
            fail(f"Tracked structured file is invalid: {relative}: {exc}")

    return {
        "count": len(files),
        "nonempty": True,
        "utf8_text": True,
        "structured_files_valid": True,
        "images_decodable": True,
        "extensions": dict(sorted(counts.items())),
    }


def verify_no_dataset_arrays() -> None:
    forbidden_names = {"x.npy", "y.npy", "split.json"}
    found = [path.relative_to(PROJECT_ROOT).as_posix() for path in PROJECT_ROOT.rglob("*") if path.name in forbidden_names]
    if found:
        fail(f"Generated dataset files must not be committed: {found}")


def verify_project_data() -> dict:
    raw_images = sorted((PROJECT_ROOT / "data" / "raw" / "calibration").rglob("*_full.png"))
    reference_files = sorted((PROJECT_ROOT / "data" / "raw" / "reference_spectrometer").rglob("*.txt"))
    profile_files = sorted((PROJECT_ROOT / "data" / "processed" / "relative_calibration_profiles_v1").rglob("profile.csv"))
    if len(raw_images) != 12:
        fail(f"Expected 12 raw calibration images, found {len(raw_images)}")
    if len(reference_files) != 9:
        fail(f"Expected 9 reference spectra, found {len(reference_files)}")
    if len(profile_files) != 12:
        fail(f"Expected 12 saved relative-coordinate profiles, found {len(profile_files)}")

    manifest_path = PROJECT_ROOT / "data" / "processed" / "relative_calibration_profiles_v1" / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("image_path", "profile_csv", "roi_crop", "diagnostic_png", "profile_png"):
            value = str(row.get(field, "")).strip()
            if value and not (PROJECT_ROOT / value).exists():
                fail(f"Profile manifest references a missing file: {field}={value}")
    return {
        "raw_calibration_images": len(raw_images),
        "reference_spectra": len(reference_files),
        "saved_profiles": len(profile_files),
        "manifest_rows": len(rows),
    }


def verify_relative_profile() -> float:
    config_path = PROJECT_ROOT / "configs" / "calibration" / "relative_calibration_linear_diagnostic_v1.json"
    config = profiles.read_json(config_path)
    if config.get("calibration_type") != "relative_spectral_coordinate_linear_diagnostic":
        fail("Calibration config is not the relative-coordinate linear diagnostic.")
    geometry = profiles.resolve_calibration_geometry(config)
    if geometry["y_short"] != 286.0 or geometry["y_long"] != 537.0:
        fail(f"Unexpected relative anchors: {geometry['y_short']}, {geometry['y_long']}")

    image_path = PROJECT_ROOT / "data" / "raw" / "calibration" / "white_led" / "white_led_001_full.png"
    saved_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "relative_calibration_profiles_v1"
        / "white_led"
        / "white_led_001_full"
        / "profile.csv"
    )
    wavelength_nm = np.round(np.arange(400.0, 650.0 + 0.0001, 0.1, dtype=np.float64), 1)
    rgb = profiles.read_image_rgb(image_path)
    crop = profiles.crop_roi(rgb, geometry["roi"])
    regenerated, _metadata = profiles.extract_calibrated_profiles(crop, config, geometry, wavelength_nm)
    saved = np.genfromtxt(saved_path, delimiter=",", names=True, dtype=np.float64)
    max_error = max(float(np.max(np.abs(regenerated[channel] - saved[channel]))) for channel in profiles.CHANNELS)
    if max_error > 1e-6:
        fail(f"Regenerated profile differs from committed profile: max_abs_error={max_error}")
    return max_error


def verify_final_model() -> dict:
    checkpoint_path = PROJECT_ROOT / "results" / "final_model" / "best_model.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        fail(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model_config = config["model"]
    training_config = config["training"]
    expected_training = {
        "epochs": 350,
        "batch_size": 64,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "seed": 42,
    }
    for key, expected in expected_training.items():
        actual = training_config.get(key)
        if actual != expected:
            fail(f"Checkpoint training config mismatch for {key}: {actual} != {expected}")

    model = SpectrumUNetTransformer1D(
        in_channels=int(model_config["in_channels"]),
        out_length=int(model_config["out_length"]),
        base_channels=int(model_config["base_channels"]),
        trans_heads=int(model_config["trans_heads"]),
        trans_layers=int(model_config["trans_layers"]),
        dropout=float(model_config["dropout"]),
        output_activation=str(model_config["output_activation"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 4, 2501), dtype=torch.float32))
    if tuple(output.shape) != (1, 2501):
        fail(f"Unexpected model output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        fail("Model smoke-test output contains NaN or Inf.")

    metrics = json.loads((PROJECT_ROOT / "results" / "final_model" / "metrics.json").read_text(encoding="utf-8"))
    if metrics["best_epoch"] != 74 or not math.isclose(
        float(metrics["best_val_loss"]), 0.045239608498145144, rel_tol=0.0, abs_tol=1e-15
    ):
        fail("Stored metrics do not match the final checkpoint run.")
    return {
        "sha256": checkpoint_hash,
        "epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "output_shape": list(output.shape),
    }


def verify_configs() -> dict:
    synthetic = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "dataset" / "relative_calib_synthetic_v1.yaml").read_text(encoding="utf-8")
    )
    mixed = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "dataset" / "relative_calib_mixed_v1.yaml").read_text(encoding="utf-8")
    )
    train = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "train" / "relative_calib_mixed_v1_80k_train.yaml").read_text(encoding="utf-8")
    )
    if synthetic["generation"]["sample_count"] != 80000 or synthetic["generation"]["seed"] != 42:
        fail("Synthetic generation config is not the final 80,000-sample seed-42 config.")
    if train["data"]["dataset_dir"] != mixed["paths"]["output_dir"]:
        fail("Training config does not point to the generated relative-coordinate mixed dataset.")
    return {
        "synthetic_samples": synthetic["generation"]["sample_count"],
        "real_samples": 6,
        "mixed_samples": 80006,
        "split_seed": mixed["split"]["seed"],
    }


def verify_artifact_hashes(dataset_dir: Path) -> int:
    metadata_path = dataset_dir / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hashes = metadata.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        fail(f"Dataset metadata has no artifact hashes: {metadata_path}")
    for name, expected_digest in hashes.items():
        artifact_path = dataset_dir / name
        if not artifact_path.exists():
            fail(f"Dataset metadata references a missing artifact: {artifact_path}")
        if sha256_file(artifact_path) != expected_digest:
            fail(f"Dataset artifact hash mismatch: {artifact_path}")
    return len(hashes)


def verify_dataset_smoke_pipeline() -> dict:
    synthetic_config = synthetic_builder.read_yaml(
        PROJECT_ROOT / "configs" / "dataset" / "relative_calib_synthetic_v1.yaml"
    )
    mixed_config = mixed_builder.read_yaml(PROJECT_ROOT / "configs" / "dataset" / "relative_calib_mixed_v1.yaml")

    with tempfile.TemporaryDirectory(prefix="relative-coordinate-smoke-") as temp:
        temp_root = Path(temp)
        real_dir = temp_root / "real"
        synthetic_dir = temp_root / "synthetic"
        mixed_dir = temp_root / "mixed"
        for path in (real_dir, synthetic_dir, mixed_dir):
            path.mkdir(parents=True)

        real_metadata = paired_builder.build_dataset(real_dir)
        synthetic_metadata = synthetic_builder.build_dataset(synthetic_config, synthetic_dir, sample_count=20, seed=42)
        mixed_metadata = mixed_builder.build_mixed(mixed_config, real_dir, synthetic_dir, mixed_dir)

        if real_metadata["array_shapes"]["x.npy"] != [6, 4, 2501]:
            fail(f"Unexpected real smoke shape: {real_metadata['array_shapes']['x.npy']}")
        if synthetic_metadata["x_shape"] != [20, 4, 2501]:
            fail(f"Unexpected synthetic smoke shape: {synthetic_metadata['x_shape']}")
        if mixed_metadata["x_shape"] != [26, 4, 2501]:
            fail(f"Unexpected mixed smoke shape: {mixed_metadata['x_shape']}")
        if mixed_metadata["split"]["counts"] != {"train": 21, "val": 3, "test": 2, "total": 26}:
            fail(f"Unexpected smoke split: {mixed_metadata['split']['counts']}")

        return {
            "real_shape": real_metadata["array_shapes"]["x.npy"],
            "synthetic_shape": synthetic_metadata["x_shape"],
            "mixed_shape": mixed_metadata["x_shape"],
            "split_counts": mixed_metadata["split"]["counts"],
            "artifact_hashes_verified": {
                "real": verify_artifact_hashes(real_dir),
                "synthetic": verify_artifact_hashes(synthetic_dir),
                "mixed": verify_artifact_hashes(mixed_dir),
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the relative-coordinate spectrometer project.")
    parser.add_argument("--full-smoke", action="store_true", help="Also build 6 real, 20 synthetic, and 26 mixed samples.")
    args = parser.parse_args()

    verify_no_dataset_arrays()
    summary = {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "tracked_files": verify_tracked_files(),
        "data": verify_project_data(),
        "config": verify_configs(),
        "profile_max_abs_error": verify_relative_profile(),
        "model": verify_final_model(),
    }
    if args.full_smoke:
        summary["dataset_smoke_pipeline"] = verify_dataset_smoke_pipeline()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
