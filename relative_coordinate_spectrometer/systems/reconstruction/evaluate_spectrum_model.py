import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    print("Missing dependency: pyyaml. Please install it with: py -m pip install pyyaml", file=sys.stderr)
    raise SystemExit(1)

try:
    import numpy as np
except ModuleNotFoundError:
    print("Missing dependency: numpy. Please install it with: py -m pip install numpy", file=sys.stderr)
    raise SystemExit(1)

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    print("Missing dependency: torch. Please install PyTorch before evaluation.", file=sys.stderr)
    raise SystemExit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Missing dependency: matplotlib. Please install it before evaluation.", file=sys.stderr)
    raise SystemExit(1)


SCRIPT_NAME = "evaluate_spectrum_model.py"
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Support direct execution from the project directory.
from shared.data_loader.npy_spectrum_dataset import NpySpectrumDataset  # noqa: E402
from shared.spectrum_unet_transformer_1d import SpectrumUNetTransformer1D  # noqa: E402


def read_text_auto(path):
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = yaml.safe_load(read_text_auto(path))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data, path


def project_path(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path):
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def choose_device(device_name):
    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_dir_from_config(config):
    output_cfg = config["output"]
    return project_path(output_cfg.get("runs_dir", "results/reproduction_runs")) / output_cfg.get(
        "run_name", "spectrum_unet_transformer_1d_v1"
    )


def make_dataset(config, split):
    data_cfg = config["data"]
    return NpySpectrumDataset(
        dataset_dir=project_path(data_cfg["dataset_dir"]),
        split=split,
        x_file=data_cfg.get("x_file", "x.npy"),
        y_file=data_cfg.get("y_file", "y.npy"),
        wavelength_file=data_cfg.get("wavelength_file", "wavelength_nm.npy"),
        split_file=data_cfg.get("split_file", "split.json"),
        expected_channels=len(data_cfg["input_channels"]),
        expected_length=int(data_cfg["wavelength"]["point_count"]),
        normalize=True,
    )


def make_model(config):
    model_cfg = config["model"]
    return SpectrumUNetTransformer1D(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_length=int(model_cfg.get("out_length", 2501)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        trans_heads=int(model_cfg.get("trans_heads", 4)),
        trans_layers=int(model_cfg.get("trans_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_activation=str(model_cfg.get("output_activation", "none")),
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = project_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {rel_path(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint_path


def load_manifest_rows(dataset_dir):
    manifest_path = Path(dataset_dir) / "manifest.csv"
    if not manifest_path.exists():
        return {}
    rows = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader):
            rows[row_number] = row
    return rows


def load_sample_metadata_rows(dataset):
    rows = load_manifest_rows(dataset.dataset_dir)
    split_path = getattr(dataset, "split_path", None)
    if split_path is None or not Path(split_path).exists():
        return rows
    try:
        split_data = json.loads(Path(split_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return rows

    sample_ids = None
    if isinstance(split_data.get("sample_ids"), dict):
        sample_ids = split_data["sample_ids"].get(dataset.split)
    if not isinstance(sample_ids, list):
        return rows

    for sample_index, sample_id in zip(getattr(dataset, "indices", []), sample_ids):
        row = rows.setdefault(int(sample_index), {})
        if not row.get("sample_id"):
            row["sample_id"] = str(sample_id)
    return rows


def sample_metadata(manifest_rows, sample_index):
    row = manifest_rows.get(int(sample_index), {})
    sample_id = str(row.get("sample_id", "") or "")
    source_name = str(
        row.get("source_name", "")
        or row.get("source", "")
        or row.get("source_type", "")
        or row.get("source_dataset", "")
        or ""
    )
    return sample_id, source_name


def sample_label(sample_index, sample_id):
    if sample_id:
        return sample_id
    return str(int(sample_index))


def finite_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def finite_median(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def format_csv_float(value):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.9g}"


def write_rows_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pearson_corr_per_sample(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * target_centered, axis=1)
    denominator = np.sqrt(np.sum(pred_centered**2, axis=1) * np.sum(target_centered**2, axis=1))
    corr = np.full(pred.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    corr[valid] = numerator[valid] / denominator[valid]
    return corr


def pearson_corr(pred, target):
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * target_centered, axis=1)
    denominator = np.sqrt(np.sum(pred_centered**2, axis=1) * np.sum(target_centered**2, axis=1))
    corr = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(corr))


def interpolate_crossing_nm(wavelength_nm, spectrum, left_idx, right_idx, level):
    y0 = float(spectrum[left_idx])
    y1 = float(spectrum[right_idx])
    x0 = float(wavelength_nm[left_idx])
    x1 = float(wavelength_nm[right_idx])
    if not all(math.isfinite(value) for value in (x0, x1, y0, y1, level)):
        return float("nan")
    if abs(y1 - y0) <= 1e-12:
        return float("nan")
    return x0 + (float(level) - y0) * (x1 - x0) / (y1 - y0)


def main_peak_fwhm_nm(spectrum, wavelength_nm):
    spectrum = np.asarray(spectrum, dtype=np.float64)
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    if spectrum.ndim != 1 or wavelength_nm.ndim != 1 or spectrum.shape[0] != wavelength_nm.shape[0]:
        return float("nan"), float("nan")
    if spectrum.size < 3 or not np.all(np.isfinite(spectrum)) or not np.all(np.isfinite(wavelength_nm)):
        return float("nan"), float("nan")

    peak_idx = int(np.argmax(spectrum))
    peak_value = float(spectrum[peak_idx])
    peak_nm = float(wavelength_nm[peak_idx])
    if peak_idx == 0 or peak_idx == spectrum.size - 1:
        return peak_nm, float("nan")
    if not math.isfinite(peak_value) or peak_value <= 1e-8:
        return peak_nm, float("nan")

    half_max = peak_value * 0.5
    left_candidates = np.where(spectrum[:peak_idx] < half_max)[0]
    right_candidates = np.where(spectrum[peak_idx + 1 :] < half_max)[0]
    if left_candidates.size == 0 or right_candidates.size == 0:
        return peak_nm, float("nan")

    left_below_idx = int(left_candidates[-1])
    right_below_idx = int(peak_idx + 1 + right_candidates[0])
    left_nm = interpolate_crossing_nm(wavelength_nm, spectrum, left_below_idx, left_below_idx + 1, half_max)
    right_nm = interpolate_crossing_nm(wavelength_nm, spectrum, right_below_idx - 1, right_below_idx, half_max)
    if not math.isfinite(left_nm) or not math.isfinite(right_nm) or right_nm <= left_nm:
        return peak_nm, float("nan")
    return peak_nm, float(right_nm - left_nm)


def sam_angle(pred, target):
    numerator = np.sum(pred * target, axis=1)
    denominator = np.sqrt(np.sum(pred**2, axis=1) * np.sum(target**2, axis=1))
    cos_value = numerator / np.maximum(denominator, 1e-12)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    angles = np.arccos(cos_value)
    return float(np.mean(angles))


def compute_metrics(pred, target, wavelength_nm):
    diff = pred - target
    mse = float(np.mean(diff**2))
    pred_peak_idx = np.argmax(pred, axis=1)
    target_peak_idx = np.argmax(target, axis=1)
    peak_position_error_nm = np.abs(wavelength_nm[pred_peak_idx] - wavelength_nm[target_peak_idx])
    pred_peak_values = pred[np.arange(pred.shape[0]), pred_peak_idx]
    target_peak_values = target[np.arange(target.shape[0]), target_peak_idx]
    return {
        "MAE": float(np.mean(np.abs(diff))),
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "Pearson_correlation": pearson_corr(pred, target),
        "SAM": sam_angle(pred, target),
        "SAM_degrees": float(np.degrees(sam_angle(pred, target))),
        "peak_position_error_nm": float(np.mean(peak_position_error_nm)),
        "peak_intensity_error": float(np.mean(np.abs(pred_peak_values - target_peak_values))),
    }


def save_prediction_plot(path, wavelength_nm, target, pred, title):
    plt.figure(figsize=(9, 4.8))
    plt.plot(wavelength_nm, target, label="target spectrum", linewidth=1.8)
    plt.plot(wavelength_nm, pred, label="predicted spectrum", linewidth=1.5)
    plt.xlabel("wavelength_nm")
    plt.ylabel("normalized intensity")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_pearson_heatmap(path, pearson_values, labels):
    values = np.asarray(pearson_values, dtype=np.float64).reshape(-1, 1)
    sample_count = values.shape[0]
    fig_height = min(max(4.0, sample_count * 0.22), 18.0)
    fig, ax = plt.subplots(figsize=(4.8, fig_height))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("Pearson Correlation Heatmap")
    ax.set_xlabel("Pearson")
    ax.set_ylabel("Sample")
    ax.set_xticks([0])
    ax.set_xticklabels(["Pearson"])

    if sample_count <= 50:
        ax.set_yticks(np.arange(sample_count))
        ax.set_yticklabels(labels, fontsize=8)
        for row_idx, value in enumerate(pearson_values):
            text_value = "nan" if not math.isfinite(float(value)) else f"{float(value):.3f}"
            ax.text(0, row_idx, text_value, ha="center", va="center", color="white", fontsize=7)
    else:
        tick_count = min(10, sample_count)
        tick_positions = np.linspace(0, sample_count - 1, tick_count, dtype=int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([labels[idx] for idx in tick_positions], fontsize=7)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_fwhm_bar(path, rows, labels):
    plot_count = min(50, len(rows))
    plot_rows = rows[:plot_count]
    target_values = np.asarray([row["target_fwhm_value"] for row in plot_rows], dtype=np.float64)
    pred_values = np.asarray([row["pred_fwhm_value"] for row in plot_rows], dtype=np.float64)
    x = np.arange(plot_count)
    width = 0.38

    fig_width = min(max(8.0, plot_count * 0.35), 18.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(x - width / 2, target_values, width, label="target_fwhm_nm")
    ax.bar(x + width / 2, pred_values, width, label="pred_fwhm_nm")
    title = "FWHM Comparison Between Target and Prediction"
    if len(rows) > 50:
        title += " (first 50 samples)"
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("FWHM (nm)")
    ax.set_xticks(x)
    rotation = 90 if plot_count > 15 else 45
    ax.set_xticklabels(labels[:plot_count], rotation=rotation, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_per_sample_evaluation_outputs(output_dir, dataset, indices, pred, target, wavelength_nm):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_sample_metadata_rows(dataset)
    pearson_values = pearson_corr_per_sample(pred, target)
    pearson_rows = []
    fwhm_rows = []
    labels = []
    fwhm_errors = []

    for local_idx, sample_index in enumerate(indices):
        sample_id, source_name = sample_metadata(manifest_rows, sample_index)
        label = sample_label(sample_index, sample_id)
        labels.append(label[:40])

        pearson_value = float(pearson_values[local_idx])
        pearson_rows.append(
            {
                "sample_index": int(sample_index),
                "sample_id": sample_id,
                "source_name": source_name,
                "pearson_r": format_csv_float(pearson_value),
            }
        )

        target_peak_nm, target_fwhm_nm = main_peak_fwhm_nm(target[local_idx], wavelength_nm)
        pred_peak_nm, pred_fwhm_nm = main_peak_fwhm_nm(pred[local_idx], wavelength_nm)
        if math.isfinite(target_fwhm_nm) and math.isfinite(pred_fwhm_nm):
            fwhm_abs_error_nm = abs(pred_fwhm_nm - target_fwhm_nm)
        else:
            fwhm_abs_error_nm = float("nan")
        fwhm_errors.append(fwhm_abs_error_nm)
        fwhm_rows.append(
            {
                "sample_index": int(sample_index),
                "sample_id": sample_id,
                "source_name": source_name,
                "target_peak_nm": format_csv_float(target_peak_nm),
                "pred_peak_nm": format_csv_float(pred_peak_nm),
                "target_fwhm_nm": format_csv_float(target_fwhm_nm),
                "pred_fwhm_nm": format_csv_float(pred_fwhm_nm),
                "fwhm_abs_error_nm": format_csv_float(fwhm_abs_error_nm),
                "target_fwhm_value": target_fwhm_nm,
                "pred_fwhm_value": pred_fwhm_nm,
            }
        )

    pearson_csv_path = output_dir / "evaluation_pearson.csv"
    fwhm_csv_path = output_dir / "evaluation_fwhm.csv"
    pearson_heatmap_path = output_dir / "pearson_heatmap.png"
    fwhm_bar_path = output_dir / "fwhm_bar.png"

    write_rows_csv(
        pearson_csv_path,
        pearson_rows,
        ["sample_index", "sample_id", "source_name", "pearson_r"],
    )
    write_rows_csv(
        fwhm_csv_path,
        [
            {key: value for key, value in row.items() if key not in {"target_fwhm_value", "pred_fwhm_value"}}
            for row in fwhm_rows
        ],
        [
            "sample_index",
            "sample_id",
            "source_name",
            "target_peak_nm",
            "pred_peak_nm",
            "target_fwhm_nm",
            "pred_fwhm_nm",
            "fwhm_abs_error_nm",
        ],
    )
    save_pearson_heatmap(pearson_heatmap_path, pearson_values, labels)
    save_fwhm_bar(fwhm_bar_path, fwhm_rows, labels)

    return {
        "evaluation_dir": rel_path(output_dir),
        "pearson_csv": rel_path(pearson_csv_path),
        "pearson_heatmap": rel_path(pearson_heatmap_path),
        "fwhm_csv": rel_path(fwhm_csv_path),
        "fwhm_bar": rel_path(fwhm_bar_path),
        "pearson_mean": finite_mean(pearson_values),
        "pearson_median": finite_median(pearson_values),
        "fwhm_abs_error_mean_nm": finite_mean(fwhm_errors),
        "fwhm_abs_error_median_nm": finite_median(fwhm_errors),
        "fwhm_bar_sample_count": int(min(50, len(fwhm_rows))),
        "fwhm_csv_sample_count": int(len(fwhm_rows)),
    }


@torch.no_grad()
def evaluate_split(model, dataset, split_name, config, device, example_dir, evaluation_dir):
    loader = DataLoader(dataset, batch_size=int(config["training"].get("batch_size", 4)), shuffle=False, num_workers=0)
    model.eval()
    raw_preds = []
    targets = []
    indices = []
    for batch in loader:
        x = batch["x"].to(device)
        pred = model(x).detach().cpu().numpy()
        raw_preds.append(pred)
        targets.append(batch["y"].numpy())
        indices.extend([int(item) for item in batch["index"]])

    if not raw_preds:
        return {"skipped": True, "reason": f"{split_name} split is empty"}

    raw_pred = np.concatenate(raw_preds, axis=0)
    target = np.concatenate(targets, axis=0)
    clipped_pred = np.clip(raw_pred, 0.0, 1.0)
    wavelength_nm = dataset.wavelength_nm

    for local_idx in range(min(8, raw_pred.shape[0])):
        save_prediction_plot(
            example_dir / f"{split_name}_sample_{indices[local_idx]:06d}.png",
            wavelength_nm,
            target[local_idx],
            clipped_pred[local_idx],
            f"{split_name} sample index {indices[local_idx]}",
        )

    per_sample_outputs = save_per_sample_evaluation_outputs(
        evaluation_dir,
        dataset,
        indices,
        raw_pred,
        target,
        wavelength_nm,
    )

    return {
        "sample_count": int(raw_pred.shape[0]),
        "raw_pred": compute_metrics(raw_pred, target, wavelength_nm),
        "clipped_pred": compute_metrics(clipped_pred, target, wavelength_nm),
        "per_sample_outputs": per_sample_outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpectrumUNetTransformer1D.")
    parser.add_argument("--config", default="configs/train/relative_calib_mixed_v1_80k_train.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--use-checkpoint-config",
        action="store_true",
        help="Use the config saved inside the checkpoint instead of the YAML config.",
    )
    args = parser.parse_args()

    config, config_path = load_yaml(args.config)
    config_source = rel_path(config_path)
    if args.use_checkpoint_config:
        initial_run_dir = run_dir_from_config(config)
        checkpoint_for_config = project_path(args.checkpoint or (initial_run_dir / "best_model.pt"))
        if not checkpoint_for_config.exists():
            raise FileNotFoundError(f"Checkpoint not found: {rel_path(checkpoint_for_config)}")
        checkpoint_for_config_data = torch.load(checkpoint_for_config, map_location="cpu")
        if not isinstance(checkpoint_for_config_data, dict) or not isinstance(
            checkpoint_for_config_data.get("config"), dict
        ):
            raise ValueError(f"Checkpoint has no embedded config: {rel_path(checkpoint_for_config)}")
        config = checkpoint_for_config_data["config"]
        config_source = f"{rel_path(checkpoint_for_config)}::config"

    device = choose_device(config["training"].get("device", "auto"))
    run_dir = run_dir_from_config(config)
    example_dir = run_dir / "prediction_examples"
    evaluation_dir = run_dir / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or (run_dir / "best_model.pt")
    model = make_model(config).to(device)
    checkpoint_path = load_checkpoint(model, checkpoint_path, device)

    evaluation = {
        "script_name": SCRIPT_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": rel_path(config_path),
        "config_source": config_source,
        "checkpoint": rel_path(checkpoint_path),
        "device": str(device),
        "evaluation_dir": rel_path(evaluation_dir),
        "splits": {},
    }
    for split_name in args.splits:
        dataset = make_dataset(config, split_name)
        split_evaluation_dir = evaluation_dir if len(args.splits) == 1 else evaluation_dir / split_name
        evaluation["splits"][split_name] = evaluate_split(
            model,
            dataset,
            split_name,
            config,
            device,
            example_dir,
            split_evaluation_dir,
        )

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    else:
        metrics = {}
    metrics["evaluation"] = evaluation
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("evaluation completed")
    print("metrics:", rel_path(metrics_path))
    print("prediction_examples:", rel_path(example_dir))
    print("evaluation_dir:", rel_path(evaluation_dir))
    for split_name, split_metrics in evaluation["splits"].items():
        print(f"{split_name}:", json.dumps(split_metrics.get("raw_pred", split_metrics), ensure_ascii=False))
        per_sample_outputs = split_metrics.get("per_sample_outputs", {})
        for key in (
            "pearson_mean",
            "pearson_median",
            "fwhm_abs_error_mean_nm",
            "fwhm_abs_error_median_nm",
        ):
            print(f"{split_name} {key}: {per_sample_outputs.get(key)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
