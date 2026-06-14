import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "calibration" / "relative_calibration_linear_diagnostic_v1.json"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "calibration"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "profile_diagnostics"

CHANNELS = ["R", "G", "B", "Gray"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REFERENCE_SOURCES = {"hg", "na", "hene"}
LED_SOURCES = {"blue_led", "green_led", "red_led", "white_led", "purple_led", "yellow_led"}


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


def safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("_")
    return value or "item"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_image_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise OSError(f"Empty image file: {path}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def infer_source(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    filename = path.name.lower()
    sources = [
        "white_led",
        "purple_led",
        "yellow_led",
        "green_led",
        "blue_led",
        "red_led",
        "hene",
        "dark",
        "hg",
        "na",
    ]
    for source in sources:
        if source in parts or source in filename:
            return source
    return path.parent.name.lower()


def scan_full_images(input_root: Path) -> list[Path]:
    images = []
    if not input_root.exists():
        return images
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lowered = path.name.lower()
        if "_full" not in lowered:
            continue
        if any(token in lowered for token in ("_roi", "diagnostic", "profile")):
            continue
        images.append(path)
    return images


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_calibration_geometry(config: dict) -> dict:
    roi_cfg = config.get("roi_source") or {}
    s_cfg = config.get("relative_s_definition") or {}
    source_dir = project_path(config.get("source", "results/calibration"))

    image_peak_rows = read_csv_rows(source_dir / "image_peak_candidates.csv")
    if image_peak_rows:
        row = image_peak_rows[0]
        roi = {
            "x": int(float(row["roi_x"])),
            "y": int(float(row["roi_y"])),
            "w": int(float(row["roi_w"])),
            "h": int(float(row["roi_h"])),
            "source": rel_path(source_dir / "image_peak_candidates.csv"),
        }
    else:
        roi = {
            "x": int(roi_cfg.get("x", 0)),
            "y": int(roi_cfg.get("y", 0)),
            "w": int(roi_cfg.get("w", 1)),
            "h": int(roi_cfg.get("h", 1)),
            "source": "config.roi_source",
        }

    selected_rows = read_csv_rows(source_dir / "selected_anchors.csv")
    y_short = s_cfg.get("y_short_anchor")
    y_long = s_cfg.get("y_long_anchor")
    if selected_rows:
        parsed = []
        for row in selected_rows:
            try:
                parsed.append((float(row["relative_s"]), float(row["selected_y_local"]), row))
            except (KeyError, TypeError, ValueError):
                continue
        if parsed:
            parsed.sort(key=lambda item: item[0])
            y_short = parsed[0][1]
            y_long = parsed[-1][1]

    if y_short is None or y_long is None:
        raise RuntimeError("Config must provide relative_s_definition y_short_anchor and y_long_anchor.")
    y_short = float(y_short)
    y_long = float(y_long)
    if abs(y_long - y_short) < 1e-9:
        raise RuntimeError("Invalid relative s definition: y_long_anchor equals y_short_anchor.")

    return {
        "roi": roi,
        "y_short": y_short,
        "y_long": y_long,
        "source_dir": source_dir,
    }


def crop_roi(rgb: np.ndarray, roi: dict) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0 = max(0, int(roi["x"]))
    y0 = max(0, int(roi["y"]))
    x1 = min(w, x0 + max(1, int(roi["w"])))
    y1 = min(h, y0 + max(1, int(roi["h"])))
    return rgb[y0:y1, x0:x1].copy()


def channel_profiles(rgb_crop: np.ndarray) -> dict[str, np.ndarray]:
    rgb_f = rgb_crop.astype(np.float64)
    r = rgb_f[..., 0].mean(axis=1)
    g = rgb_f[..., 1].mean(axis=1)
    b = rgb_f[..., 2].mean(axis=1)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return {"R": r, "G": g, "B": b, "Gray": gray}


def normalize_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(max_value) or max_value <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / max_value, 0.0, 1.0)


def interp_with_linear_extrapolation(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = np.asarray(x, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    if x.size == 0:
        return np.zeros_like(target, dtype=np.float64)
    if x.size == 1:
        return np.full_like(target, y[0], dtype=np.float64)

    result = np.interp(target, x, y)
    left = target < x[0]
    right = target > x[-1]
    if np.any(left):
        slope = (y[1] - y[0]) / max(x[1] - x[0], 1e-12)
        result[left] = y[0] + slope * (target[left] - x[0])
    if np.any(right):
        slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-12)
        result[right] = y[-1] + slope * (target[right] - x[-1])
    return np.clip(result, 0.0, None)


def extract_calibrated_profiles(
    rgb_crop: np.ndarray,
    config: dict,
    geometry: dict,
    target_wavelength: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    raw_profiles = channel_profiles(rgb_crop)
    y_local = np.arange(rgb_crop.shape[0], dtype=np.float64)
    s = (y_local - geometry["y_short"]) / (geometry["y_long"] - geometry["y_short"])
    source_wavelength = float(config["coefficient_a"]) * s + float(config["coefficient_b"])
    source_min = float(np.min(source_wavelength))
    source_max = float(np.max(source_wavelength))
    extrapolated_points = int(np.count_nonzero((target_wavelength < source_min) | (target_wavelength > source_max)))

    calibrated = {}
    for channel in CHANNELS:
        interpolated = interp_with_linear_extrapolation(source_wavelength, raw_profiles[channel], target_wavelength)
        calibrated[channel] = normalize_max(interpolated)

    meta = {
        "source_wavelength_min_nm": source_min,
        "source_wavelength_max_nm": source_max,
        "target_extrapolated_points": extrapolated_points,
        "target_extrapolated_fraction": extrapolated_points / float(target_wavelength.size),
        "row_count": int(rgb_crop.shape[0]),
    }
    return calibrated, meta


def write_profile_csv(path: Path, wavelength: np.ndarray, profiles: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["wavelength_nm", "R", "G", "B", "Gray"])
        for idx, wl in enumerate(wavelength):
            writer.writerow(
                [
                    f"{float(wl):.1f}",
                    f"{profiles['R'][idx]:.9g}",
                    f"{profiles['G'][idx]:.9g}",
                    f"{profiles['B'][idx]:.9g}",
                    f"{profiles['Gray'][idx]:.9g}",
                ]
            )


def profile_score(rgb_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_f = rgb_crop.astype(np.float64)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    max_ch = np.max(rgb_f, axis=2)
    min_ch = np.min(rgb_f, axis=2)
    saturation = np.zeros_like(max_ch)
    nonzero = max_ch > 1e-9
    saturation[nonzero] = (max_ch[nonzero] - min_ch[nonzero]) / max_ch[nonzero]
    score = 0.55 * gray + 0.30 * max_ch + 0.15 * saturation * max_ch
    return score, gray


def bright_mask(score: np.ndarray) -> np.ndarray:
    if score.size == 0:
        return np.zeros_like(score, dtype=bool)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) + 1e-9
    threshold = max(float(np.percentile(score, 97.0)), median + 4.0 * mad)
    return score >= threshold


def estimate_tilt(mask: np.ndarray, score: np.ndarray) -> tuple[float | None, bool]:
    if mask.sum() < 40:
        return None, False
    ys, xs = np.where(mask)
    weights = score[ys, xs].astype(np.float64)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)
    x_mean = float(np.average(xs, weights=weights))
    y_mean = float(np.average(ys, weights=weights))
    centered = np.column_stack([xs - x_mean, ys - y_mean]).astype(np.float64)
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    angle_from_vertical = min(abs(angle - 90.0), abs(angle + 90.0))
    return angle_from_vertical, angle_from_vertical > 8.0


def local_prominence(values: np.ndarray, idx: int, radius: int) -> float:
    lo = max(0, idx - radius)
    hi = min(values.size, idx + radius + 1)
    window = values[lo:hi]
    if window.size <= 1:
        return 0.0
    baseline = float(np.percentile(window, 15))
    return float(values[idx] - baseline)


def find_peaks(wavelength: np.ndarray, values: np.ndarray, max_peaks: int = 8, min_distance_nm: float = 2.0) -> list[dict]:
    wavelength = np.asarray(wavelength, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.size < 5:
        return []
    kernel = np.ones(7, dtype=np.float64) / 7.0
    y = np.convolve(np.pad(values, 3, mode="edge"), kernel, mode="valid")
    baseline = float(np.percentile(y, 10))
    p95 = float(np.percentile(y, 95))
    threshold = baseline + 0.10 * max(p95 - baseline, 1e-9)
    candidates = []
    radius = max(5, int(0.025 * y.size))
    for idx in range(1, y.size - 1):
        if y[idx] < threshold:
            continue
        if y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1]:
            prominence = local_prominence(y, idx, radius)
            if prominence <= 1e-4:
                continue
            candidates.append(
                {
                    "idx": idx,
                    "wavelength_nm": float(wavelength[idx]),
                    "height": float(y[idx]),
                    "prominence": float(prominence),
                }
            )
    candidates.sort(key=lambda row: (row["prominence"], row["height"]), reverse=True)
    selected = []
    for candidate in candidates:
        if all(abs(candidate["wavelength_nm"] - row["wavelength_nm"]) >= min_distance_nm for row in selected):
            selected.append(candidate)
        if len(selected) >= max_peaks:
            break
    return sorted(selected, key=lambda row: row["wavelength_nm"])


def dominant_channel(profiles: dict[str, np.ndarray]) -> str:
    scores = {channel: float(np.trapezoid(profiles[channel])) for channel in CHANNELS}
    return max(scores, key=scores.get)


def led_shape_comment(source: str, profiles: dict[str, np.ndarray], wavelength: np.ndarray) -> str:
    channel = dominant_channel(profiles)
    gray_peaks = find_peaks(wavelength, profiles["Gray"], max_peaks=3, min_distance_nm=8.0)
    main_peak = gray_peaks[np.argmax([p["height"] for p in gray_peaks])]["wavelength_nm"] if gray_peaks else float("nan")
    ok = True
    reason = []
    if source == "blue_led":
        ok = channel in {"B", "Gray"} or (np.isfinite(main_peak) and main_peak < 510)
        reason.append(f"dominant={channel}, gray_peak={main_peak:.1f} nm")
    elif source == "green_led":
        ok = channel in {"G", "Gray"} or (500 <= main_peak <= 575)
        reason.append(f"dominant={channel}, gray_peak={main_peak:.1f} nm")
    elif source == "red_led":
        ok = channel in {"R", "Gray"} or main_peak >= 570
        reason.append(f"dominant={channel}, gray_peak={main_peak:.1f} nm")
    elif source == "yellow_led":
        ok = channel in {"R", "G", "Gray"} and 520 <= main_peak <= 620
        reason.append(f"dominant={channel}, gray_peak={main_peak:.1f} nm")
    elif source == "purple_led":
        b_max = float(np.max(profiles["B"]))
        r_max = float(np.max(profiles["R"]))
        ok = b_max > 0.35 or r_max > 0.35
        reason.append(f"dominant={channel}, Bmax={b_max:.2f}, Rmax={r_max:.2f}, gray_peak={main_peak:.1f} nm")
    elif source == "white_led":
        active = [ch for ch in CHANNELS[:3] if float(np.max(profiles[ch])) > 0.25]
        ok = len(active) >= 2
        reason.append(f"active_channels={','.join(active) if active else 'none'}, gray_peak={main_peak:.1f} nm")
    else:
        return ""
    return ("reasonable: " if ok else "needs review: ") + "; ".join(reason)


def edge_anomaly_comment(profiles: dict[str, np.ndarray], extrapolated_fraction: float) -> str:
    parts = []
    if extrapolated_fraction > 0:
        parts.append(f"{extrapolated_fraction:.3%} of target wavelength grid required extrapolation")
    for side, slc in [("400nm edge", slice(0, 20)), ("650nm edge", slice(-20, None))]:
        edge_max = max(float(np.max(profiles[channel][slc])) for channel in CHANNELS)
        if edge_max > 0.65:
            parts.append(f"{side} has high normalized intensity ({edge_max:.2f}); inspect for edge/background dominance")
    return "; ".join(parts) if parts else "no obvious 400-650 edge extrapolation anomaly"


def quality_flags(rgb: np.ndarray, rgb_crop: np.ndarray, roi: dict) -> dict:
    score, gray = profile_score(rgb_crop)
    mask = bright_mask(score)
    edge_count = 0
    if mask.size:
        edge_count = int(mask[:3, :].sum() + mask[-3:, :].sum() + mask[:, :3].sum() + mask[:, -3:].sum())
    bright_touches_edge = edge_count > max(20, int(0.002 * mask.size))
    image_h, image_w = rgb.shape[:2]
    roi_touches_image_edge = (
        int(roi["x"]) <= 2
        or int(roi["y"]) <= 2
        or int(roi["x"]) + int(roi["w"]) >= image_w - 2
        or int(roi["y"]) + int(roi["h"]) >= image_h - 2
    )
    saturated_crop_ratio = float((rgb_crop >= 250).any(axis=2).mean()) if rgb_crop.size else 0.0
    saturated_full_ratio = float((rgb >= 250).any(axis=2).mean()) if rgb.size else 0.0
    gray_p99 = float(np.percentile(gray, 99)) if gray.size else 0.0
    tilt_angle, tilt_warning = estimate_tilt(mask, score)
    return {
        "roi_touches_image_edge": roi_touches_image_edge,
        "bright_signal_touches_roi_edge": bright_touches_edge,
        "overexposed": saturated_crop_ratio > 0.01 or saturated_full_ratio > 0.005,
        "underexposed": gray_p99 < 25.0,
        "tilt_angle_deg": tilt_angle,
        "tilt_warning": tilt_warning,
        "saturated_crop_ratio": saturated_crop_ratio,
        "saturated_full_ratio": saturated_full_ratio,
        "gray_p99": gray_p99,
    }


def draw_diagnostic(
    out_path: Path,
    rgb: np.ndarray,
    roi: dict,
    image_path: Path,
    source: str,
    geometry: dict,
    config: dict,
    flags: dict,
) -> None:
    diag = rgb.copy()
    x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    cv2.rectangle(diag, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 4)
    for anchor_y, color in [(geometry["y_short"], (0, 255, 255)), (geometry["y_long"], (255, 0, 255))]:
        yy = int(round(y + anchor_y))
        if 0 <= yy < diag.shape[0]:
            cv2.line(diag, (x, yy), (min(diag.shape[1] - 1, x + w - 1), yy), color, 2)
    lines = [
        f"{source} | {image_path.name}",
        f"ROI x={x} y={y} w={w} h={h} from current relative diagnostic",
        f"s=(y_local-{geometry['y_short']:.1f})/({geometry['y_long']:.1f}-{geometry['y_short']:.1f})",
        f"wavelength_nm={config['coefficient_a']:.9f}*s+{config['coefficient_b']:.7f}",
        f"over={flags['overexposed']} dark={flags['underexposed']} bright_edge={flags['bright_signal_touches_roi_edge']} tilt={flags['tilt_warning']}",
    ]
    y_text = 34
    for line in lines:
        cv2.putText(diag, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(diag, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        y_text += 34
    write_image_rgb(out_path, diag)


def plot_profiles(path: Path, wavelength: np.ndarray, profiles: dict[str, np.ndarray], source: str, image_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4", "Gray": "#444444"}
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    for channel in CHANNELS:
        ax.plot(wavelength, profiles[channel], color=colors[channel], linewidth=1.15, label=channel)
    ax.set_xlim(400, 650)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"{source} relative-calibrated RGB/Gray profile: {image_path.name}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_standard_peaks(source_dir: Path) -> dict[str, list[dict]]:
    rows = read_csv_rows(source_dir / "standard_peaks.csv")
    grouped = defaultdict(list)
    for row in rows:
        try:
            grouped[row["source"]].append(
                {
                    "wavelength_nm": float(row["standard_wavelength_nm"]),
                    "prominence": float(row.get("standard_prominence", 0.0)),
                    "height": float(row.get("standard_peak_height", 0.0)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    for source in grouped:
        grouped[source].sort(key=lambda row: (row["prominence"], row["height"]), reverse=True)
    return grouped


def reference_alignment_comment(source: str, profiles: dict[str, np.ndarray], wavelength: np.ndarray, standard_peaks: dict[str, list[dict]]) -> str:
    standards = standard_peaks.get(source, [])[:5]
    if not standards:
        return "no standard peaks available for comparison"
    image_peaks = []
    for channel in CHANNELS:
        for peak in find_peaks(wavelength, profiles[channel], max_peaks=8, min_distance_nm=2.0):
            image_peaks.append({**peak, "channel": channel})
    if not image_peaks:
        return "no clear calibrated image peaks detected"

    parts = []
    good = 0
    for standard in standards:
        nearest = min(image_peaks, key=lambda peak: abs(peak["wavelength_nm"] - standard["wavelength_nm"]))
        delta = nearest["wavelength_nm"] - standard["wavelength_nm"]
        if abs(delta) <= 8.0:
            good += 1
        parts.append(
            f"{standard['wavelength_nm']:.1f}nm->{nearest['wavelength_nm']:.1f}nm "
            f"({nearest['channel']}, delta={delta:+.1f})"
        )
    prefix = "roughly aligned" if good >= max(1, min(2, len(standards))) else "needs review"
    return f"{prefix}: " + "; ".join(parts)


def process_image(
    image_path: Path,
    config: dict,
    geometry: dict,
    output_dir: Path,
    target_wavelength: np.ndarray,
    standard_peaks: dict[str, list[dict]],
) -> dict:
    source = infer_source(image_path)
    rgb = read_image_rgb(image_path)
    rgb_crop = crop_roi(rgb, geometry["roi"])
    profiles, extract_meta = extract_calibrated_profiles(rgb_crop, config, geometry, target_wavelength)
    flags = quality_flags(rgb, rgb_crop, geometry["roi"])

    item_name = safe_name(f"{source}_{image_path.stem}")
    profile_csv = output_dir / f"{item_name}_profile.csv"
    profile_png = output_dir / f"{item_name}_profile.png"
    crop_png = output_dir / f"{item_name}_roi_crop.png"
    diagnostic_png = output_dir / f"{item_name}_diagnostic.png"

    write_profile_csv(profile_csv, target_wavelength, profiles)
    plot_profiles(profile_png, target_wavelength, profiles, source, image_path)
    write_image_rgb(crop_png, rgb_crop)
    draw_diagnostic(diagnostic_png, rgb, geometry["roi"], image_path, source, geometry, config, flags)

    if source in REFERENCE_SOURCES:
        shape_comment = reference_alignment_comment(source, profiles, target_wavelength, standard_peaks)
    elif source in LED_SOURCES:
        shape_comment = led_shape_comment(source, profiles, target_wavelength)
    elif source == "dark":
        max_signal = max(float(np.max(profiles[channel])) for channel in CHANNELS)
        shape_comment = f"dark source: normalized max={max_signal:.2f}; raw darkness check gray_p99={flags['gray_p99']:.2f}"
    else:
        shape_comment = "source not in reference/LED heuristic list"

    edge_comment = edge_anomaly_comment(profiles, extract_meta["target_extrapolated_fraction"])
    top_peaks = find_peaks(target_wavelength, profiles["Gray"], max_peaks=5, min_distance_nm=6.0)
    top_peak_text = "; ".join(f"{peak['wavelength_nm']:.1f}" for peak in top_peaks) if top_peaks else ""

    return {
        "source": source,
        "image_path": rel_path(image_path),
        "profile_csv": rel_path(profile_csv),
        "profile_png": rel_path(profile_png),
        "roi_crop_png": rel_path(crop_png),
        "diagnostic_png": rel_path(diagnostic_png),
        "roi_x": geometry["roi"]["x"],
        "roi_y": geometry["roi"]["y"],
        "roi_w": geometry["roi"]["w"],
        "roi_h": geometry["roi"]["h"],
        "source_wavelength_min_nm": f"{extract_meta['source_wavelength_min_nm']:.3f}",
        "source_wavelength_max_nm": f"{extract_meta['source_wavelength_max_nm']:.3f}",
        "target_extrapolated_fraction": f"{extract_meta['target_extrapolated_fraction']:.6f}",
        "overexposed": str(bool(flags["overexposed"])).lower(),
        "underexposed": str(bool(flags["underexposed"])).lower(),
        "roi_touches_image_edge": str(bool(flags["roi_touches_image_edge"])).lower(),
        "bright_signal_touches_roi_edge": str(bool(flags["bright_signal_touches_roi_edge"])).lower(),
        "tilt_warning": str(bool(flags["tilt_warning"])).lower(),
        "tilt_angle_deg": "" if flags["tilt_angle_deg"] is None else f"{flags['tilt_angle_deg']:.3f}",
        "gray_top_peaks_nm": top_peak_text,
        "shape_comment": shape_comment,
        "edge_comment": edge_comment,
        "status": "success",
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report_recommendation(rows: list[dict]) -> tuple[bool, list[str]]:
    problems = []
    formal_lock_blockers = []
    for row in rows:
        if row["status"] != "success":
            problems.append(f"{row['source']} failed")
            formal_lock_blockers.append(f"{row['source']} failed")
        if row["target_extrapolated_fraction"] != "0.000000":
            problems.append(f"{row['source']} required wavelength-grid extrapolation")
            formal_lock_blockers.append(f"{row['source']} required wavelength-grid extrapolation")
        if row["overexposed"] == "true" and row["source"] in REFERENCE_SOURCES:
            problems.append(f"{row['source']} reference image appears overexposed")
            formal_lock_blockers.append(f"{row['source']} reference image appears overexposed")
        if row["underexposed"] == "true" and row["source"] in REFERENCE_SOURCES:
            problems.append(f"{row['source']} reference image appears underexposed")
            formal_lock_blockers.append(f"{row['source']} reference image appears underexposed")
        if row["tilt_warning"] == "true":
            problems.append(f"{row['source']} has tilt warning")
            formal_lock_blockers.append(f"{row['source']} has tilt warning")
        if str(row.get("shape_comment", "")).startswith("needs review"):
            problems.append(f"{row['source']} shape needs review")
            formal_lock_blockers.append(f"{row['source']} shape needs review")
        if "edge/background dominance" in str(row.get("edge_comment", "")) and row["source"] in REFERENCE_SOURCES:
            formal_lock_blockers.append(f"{row['source']} reference profile has edge/background warning")
    return len(formal_lock_blockers) == 0, problems


def write_report(
    path: Path,
    rows: list[dict],
    config_path: Path,
    input_root: Path,
    output_dir: Path,
    config: dict,
    geometry: dict,
) -> None:
    counts = Counter(row["source"] for row in rows)
    recommended, problems = report_recommendation(rows)
    lines = [
        "# Relative Calibration Profile Check",
        "",
        "This is a diagnostic extraction using the current monotonic linear relative spectral-coordinate calibration.",
        "It does not train, predict, generate `x.npy`/`y.npy`, use the old fixed ROI, or use the old pixel-to-wavelength formula.",
        "",
        f"- input_root: `{rel_path(input_root)}`",
        f"- output_dir: `{rel_path(output_dir)}`",
        f"- config: `{rel_path(config_path)}`",
        f"- formula: `{config['formula']}`",
        f"- ROI source: `{geometry['roi']['source']}`",
        f"- ROI used: x={geometry['roi']['x']}, y={geometry['roi']['y']}, w={geometry['roi']['w']}, h={geometry['roi']['h']}",
        f"- relative s: `(y_local - {geometry['y_short']:.1f}) / ({geometry['y_long']:.1f} - {geometry['y_short']:.1f})`",
        "- target wavelength grid: 400.0-650.0 nm, 0.1 nm step, 2501 points",
        "",
        "## Sources Processed",
        "",
    ]
    for source, count in sorted(counts.items()):
        lines.append(f"- {source}: {count}")

    lines.extend(["", "## Per-Image Results", ""])
    lines.append(
        "| source | image | profile | ROI crop | diagnostic | gray top peaks nm | quality flags | shape check | edge/extrapolation |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        quality = []
        for key in ["overexposed", "underexposed", "bright_signal_touches_roi_edge", "tilt_warning"]:
            if row[key] == "true":
                quality.append(key)
        if row["roi_touches_image_edge"] == "true":
            quality.append("roi_touches_image_edge")
        quality_text = ", ".join(quality) if quality else "none"
        lines.append(
            f"| {row['source']} | `{row['image_path']}` | `{row['profile_png']}` | "
            f"`{row['roi_crop_png']}` | `{row['diagnostic_png']}` | {row['gray_top_peaks_nm']} | "
            f"{quality_text} | {row['shape_comment']} | {row['edge_comment']} |"
        )

    lines.extend(["", "## Reference Peak Check", ""])
    for row in rows:
        if row["source"] in REFERENCE_SOURCES:
            lines.append(f"- {row['source']} `{row['image_path']}`: {row['shape_comment']}")

    lines.extend(["", "## LED Shape Check", ""])
    for row in rows:
        if row["source"] in LED_SOURCES:
            lines.append(f"- {row['source']} `{row['image_path']}`: {row['shape_comment']}")

    lines.extend(["", "## 400-650 nm Edge And Extrapolation Check", ""])
    for row in rows:
        lines.append(
            f"- {row['source']} `{row['image_path']}`: source wavelength span "
            f"{row['source_wavelength_min_nm']} to {row['source_wavelength_max_nm']} nm; {row['edge_comment']}"
        )

    lines.extend(["", "## Risk Summary", ""])
    if problems:
        for item in problems:
            lines.append(f"- {item}")
    else:
        lines.append("- No severe automatic extraction problems were detected.")

    lines.extend(["", "## Recommendation", ""])
    if recommended:
        lines.append(
            "- The diagnostic profiles were generated successfully and no automatic blocker was detected. "
            "This can proceed to a formal relative-coordinate version-lock review, but it is still not final calibration."
        )
    else:
        lines.append(
            "- Do not lock this as a formal calibration yet. The profiles are useful for diagnostic review, "
            "but the listed exposure, tilt, shape, or edge/background warnings should be resolved or manually accepted first."
        )
    lines.append("")
    lines.append("No training, prediction, neural-network dataset generation, old dataset overwrite, old ROI, or old pixel-to-wavelength formula was used.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- summary CSV: `{rel_path(output_dir / 'profile_check_summary.csv')}`")
    for row in rows:
        lines.append(f"- {row['source']} `{Path(row['image_path']).name}`: `{row['profile_csv']}`, `{row['profile_png']}`, `{row['roi_crop_png']}`, `{row['diagnostic_png']}`")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RGB/Gray profiles using diagnostic relative spectral calibration.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Relative calibration diagnostic config JSON.")
    parser.add_argument("--input_root", default=str(DEFAULT_INPUT_ROOT), help="Directory containing new *_full.png captures.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for profile diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    input_root = project_path(args.input_root)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(config_path)
    if config.get("calibration_type") != "relative_spectral_coordinate_linear_diagnostic":
        raise RuntimeError(f"Unexpected calibration_type in {config_path}")
    if not bool(config.get("monotonic_on_0_1")):
        raise RuntimeError("Refusing to extract profiles with a non-monotonic relative calibration.")

    geometry = resolve_calibration_geometry(config)
    images = scan_full_images(input_root)
    if not images:
        raise RuntimeError(f"No *_full image files found under {input_root}")

    target_wavelength = np.round(np.arange(400.0, 650.0 + 0.0001, 0.1, dtype=np.float64), 1)
    standard_peaks = load_standard_peaks(geometry["source_dir"])

    rows = []
    for image_path in images:
        try:
            rows.append(process_image(image_path, config, geometry, output_dir, target_wavelength, standard_peaks))
        except Exception as exc:
            source = infer_source(image_path)
            rows.append(
                {
                    "source": source,
                    "image_path": rel_path(image_path),
                    "profile_csv": "",
                    "profile_png": "",
                    "roi_crop_png": "",
                    "diagnostic_png": "",
                    "roi_x": geometry["roi"]["x"],
                    "roi_y": geometry["roi"]["y"],
                    "roi_w": geometry["roi"]["w"],
                    "roi_h": geometry["roi"]["h"],
                    "source_wavelength_min_nm": "",
                    "source_wavelength_max_nm": "",
                    "target_extrapolated_fraction": "",
                    "overexposed": "",
                    "underexposed": "",
                    "roi_touches_image_edge": "",
                    "bright_signal_touches_roi_edge": "",
                    "tilt_warning": "",
                    "tilt_angle_deg": "",
                    "gray_top_peaks_nm": "",
                    "shape_comment": f"failed: {exc}",
                    "edge_comment": "",
                    "status": "failed",
                }
            )

    write_summary_csv(output_dir / "profile_check_summary.csv", rows)
    write_report(output_dir / "profile_check_report.md", rows, config_path, input_root, output_dir, config, geometry)

    counts = Counter(row["source"] for row in rows)
    recommended, problems = report_recommendation(rows)
    print("Relative calibration profile check complete")
    print("  images_processed:", len(rows))
    print("  sources:", ", ".join(f"{source}={count}" for source, count in sorted(counts.items())))
    print("  output_dir:", rel_path(output_dir))
    print("  report:", rel_path(output_dir / "profile_check_report.md"))
    print("  recommended_for_formal_lock:", recommended)
    if problems:
        print("  warnings:")
        for item in problems[:12]:
            print("    -", item)
        if len(problems) > 12:
            print(f"    - ... {len(problems) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
