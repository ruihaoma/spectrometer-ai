import argparse
import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "calibration"
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "data" / "raw" / "reference_spectrometer"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "calibration"

REFERENCE_SOURCES = ["hg", "na", "hene"]
CHANNELS = ["R", "G", "B", "Gray"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REFERENCE_EXTENSIONS = {".txt", ".tex", ".csv", ".tsv", ".dat"}
TARGET_WAVELENGTH_MIN_NM = 400.0
TARGET_WAVELENGTH_MAX_NM = 650.0

# These are only loose labels for plots/templates. They are never used as final anchors.
POSSIBLE_LINE_HINTS_NM = {
    "hg": [435.8, 546.1, 577.0],
    "na": [589.3],
    "hene": [632.8],
}


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("_")
    return value or "item"


def possible_hint(source: str, wavelength_nm: float | None = None) -> str:
    hints = POSSIBLE_LINE_HINTS_NM.get(source, [])
    if not hints:
        return ""
    if wavelength_nm is None:
        return "/".join(f"{value:.1f}" for value in hints)
    nearest = min(hints, key=lambda value: abs(value - wavelength_nm))
    delta = abs(nearest - wavelength_nm)
    if delta <= 3.0:
        return f"near {nearest:.1f} nm possible_hint_only"
    return ""


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
    for source in ["white_led", "purple_led", "yellow_led", "green_led", "blue_led", "red_led", "hene", "dark", "hg", "na"]:
        if source in parts or source in filename:
            return source
    return path.parent.name.lower()


def scan_full_images(image_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    if not image_root.exists():
        return grouped
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.name.lower()
        if "_full" not in name or "_roi" in name or "_diagnostic" in name:
            continue
        grouped[infer_source(path)].append(path)
    return grouped


def reference_files_for_source(reference_root: Path, source: str) -> list[Path]:
    if not reference_root.exists():
        return []
    files = []
    for path in sorted(reference_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in REFERENCE_EXTENSIONS:
            continue
        lower_name = path.name.lower()
        lower_parts = [part.lower() for part in path.parts]
        if lower_name.startswith(source) or source in lower_name or source in lower_parts:
            files.append(path)
    return files


def parse_numeric_reference_file(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%"):
            continue
        numbers = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", stripped)
        if len(numbers) < 2:
            continue
        try:
            rows.append([float(item) for item in numbers])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 2), dtype=np.float64)

    width = max(len(row) for row in rows)
    table = np.full((len(rows), width), np.nan, dtype=np.float64)
    for idx, row in enumerate(rows):
        table[idx, : len(row)] = row
    return table


def choose_wavelength_intensity_columns(table: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if table.size == 0 or table.shape[1] < 2:
        return None
    best = None
    for wavelength_col in range(table.shape[1]):
        wavelength = table[:, wavelength_col]
        finite_w = np.isfinite(wavelength)
        if finite_w.sum() < 5:
            continue
        in_range = (wavelength >= TARGET_WAVELENGTH_MIN_NM) & (wavelength <= TARGET_WAVELENGTH_MAX_NM)
        range_count = int(np.count_nonzero(in_range & finite_w))
        span = float(np.nanmax(wavelength) - np.nanmin(wavelength))
        if range_count < 5 or span <= 1:
            continue
        for intensity_col in range(table.shape[1]):
            if intensity_col == wavelength_col:
                continue
            intensity = table[:, intensity_col]
            finite = finite_w & np.isfinite(intensity)
            valid = finite & in_range
            if np.count_nonzero(valid) < 5:
                continue
            contrast = float(np.nanpercentile(intensity[valid], 99) - np.nanpercentile(intensity[valid], 10))
            score = range_count * max(contrast, 1e-9)
            if best is None or score > best[0]:
                best = (score, wavelength[valid], intensity[valid])
    if best is None:
        return None
    wavelength = best[1].astype(np.float64)
    intensity = best[2].astype(np.float64)
    order = np.argsort(wavelength)
    return wavelength[order], intensity[order]


def smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = values.astype(np.float64)
    if values.size < 3:
        return values
    window = max(3, int(window) | 1)
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def local_prominence(values: np.ndarray, idx: int, radius: int) -> float:
    lo = max(0, idx - radius)
    hi = min(values.size, idx + radius + 1)
    window = values[lo:hi]
    if window.size <= 1:
        return 0.0
    baseline = float(np.percentile(window, 15))
    return float(values[idx] - baseline)


def find_profile_peaks(axis: np.ndarray, intensity: np.ndarray, max_peaks: int = 12, min_distance_axis: float | None = None) -> list[dict]:
    if axis.size < 5:
        return []
    y = smooth(intensity, window=7)
    baseline = float(np.percentile(y, 10))
    p95 = float(np.percentile(y, 95))
    mad = float(np.median(np.abs(y - np.median(y)))) + 1e-9
    threshold = max(baseline + 3.0 * mad, baseline + 0.12 * max(p95 - baseline, 1e-9))
    radius = max(3, int(0.02 * y.size))
    candidates = []
    for idx in range(1, y.size - 1):
        if y[idx] < threshold:
            continue
        if y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1]:
            prominence = local_prominence(y, idx, radius=radius)
            if prominence <= max(1e-9, 1.5 * mad):
                continue
            candidates.append(
                {
                    "idx": idx,
                    "axis_value": float(axis[idx]),
                    "height": float(y[idx]),
                    "prominence": float(prominence),
                }
            )
    candidates.sort(key=lambda item: (item["prominence"], item["height"]), reverse=True)
    selected = []
    for candidate in candidates:
        if min_distance_axis is None:
            min_distance_ok = all(abs(candidate["idx"] - existing["idx"]) >= 8 for existing in selected)
        else:
            min_distance_ok = all(abs(candidate["axis_value"] - existing["axis_value"]) >= min_distance_axis for existing in selected)
        if min_distance_ok:
            selected.append(candidate)
        if len(selected) >= max_peaks:
            break
    return sorted(selected, key=lambda item: item["axis_value"])


def load_standard_peaks(reference_root: Path) -> tuple[list[dict], dict[str, list[Path]], list[str]]:
    rows = []
    files_by_source = {}
    warnings = []
    for source in REFERENCE_SOURCES:
        files = reference_files_for_source(reference_root, source)
        files_by_source[source] = files
        if not files:
            warnings.append(f"No standard spectrum file found for {source} under {rel_path(reference_root)}")
            continue
        for file_path in files:
            table = parse_numeric_reference_file(file_path)
            chosen = choose_wavelength_intensity_columns(table)
            if chosen is None:
                warnings.append(f"Could not parse wavelength/intensity columns from {rel_path(file_path)}")
                continue
            wavelength, intensity = chosen
            peaks = find_profile_peaks(wavelength, intensity, max_peaks=12, min_distance_axis=0.3)
            for rank, peak in enumerate(peaks, start=1):
                rows.append(
                    {
                        "source": source,
                        "standard_file": rel_path(file_path),
                        "standard_peak_rank": rank,
                        "standard_wavelength_nm": f"{peak['axis_value']:.6f}",
                        "standard_peak_height": f"{peak['height']:.9g}",
                        "standard_prominence": f"{peak['prominence']:.9g}",
                        "possible_hint": possible_hint(source, peak["axis_value"]),
                    }
                )
    return rows, files_by_source, warnings


def score_image(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    max_ch = np.max(rgb_f, axis=2)
    min_ch = np.min(rgb_f, axis=2)
    saturation = np.zeros_like(max_ch, dtype=np.float32)
    nonzero = max_ch > 1e-6
    saturation[nonzero] = (max_ch[nonzero] - min_ch[nonzero]) / max_ch[nonzero]
    return 0.65 * gray + 0.25 * max_ch + 0.10 * saturation * max_ch


def detect_white_roi(white_path: Path) -> tuple[int, int, int, int]:
    rgb = read_image_rgb(white_path)
    score = score_image(rgb)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) + 1e-6
    threshold = max(float(np.percentile(score, 92)), median + 3.5 * mad)
    mask = (score >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        h, w = rgb.shape[:2]
        return w // 4, h // 4, w // 2, h // 2
    best_label = max(range(1, count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    image_h, image_w = rgb.shape[:2]
    margin_x = max(30, int(0.18 * w))
    margin_y = max(30, int(0.18 * h))
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(image_w, x + w + margin_x)
    y1 = min(image_h, y + h + margin_y)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def crop(rgb: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    image_h, image_w = rgb.shape[:2]
    return rgb[max(0, y) : min(image_h, y + h), max(0, x) : min(image_w, x + w)].copy()


def profile_channels(rgb_crop: np.ndarray) -> dict[str, np.ndarray]:
    rgb_f = rgb_crop.astype(np.float32)
    r = rgb_f[..., 0].mean(axis=1)
    g = rgb_f[..., 1].mean(axis=1)
    b = rgb_f[..., 2].mean(axis=1)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return {"R": r, "G": g, "B": b, "Gray": gray}


def load_image_peak_candidates(image_root: Path, output_dir: Path) -> tuple[list[dict], tuple[int, int, int, int] | None, Path | None]:
    grouped = scan_full_images(image_root)
    white_images = grouped.get("white_led", [])
    white_roi = None
    white_path = white_images[0] if white_images else None
    if white_path is not None:
        white_roi = detect_white_roi(white_path)

    rows = []
    for source in REFERENCE_SOURCES:
        for image_path in grouped.get(source, []):
            rgb = read_image_rgb(image_path)
            if white_roi is None:
                h, w = rgb.shape[:2]
                roi = (w // 4, h // 4, w // 2, h // 2)
            else:
                roi = white_roi
            image_crop = crop(rgb, roi)
            profiles = profile_channels(image_crop)
            y_axis = np.arange(len(profiles["Gray"]), dtype=np.float64)
            for channel in CHANNELS:
                peaks = find_profile_peaks(y_axis, profiles[channel], max_peaks=8, min_distance_axis=8.0)
                for rank, peak in enumerate(peaks, start=1):
                    rows.append(
                        {
                            "source": source,
                            "image_path": rel_path(image_path),
                            "channel": channel,
                            "image_peak_rank": rank,
                            "peak_y_local": f"{peak['axis_value']:.3f}",
                            "peak_height": f"{peak['height']:.9g}",
                            "prominence": f"{peak['prominence']:.9g}",
                            "roi_x": roi[0],
                            "roi_y": roi[1],
                            "roi_w": roi[2],
                            "roi_h": roi[3],
                            "possible_hint": possible_hint(source, None),
                        }
                    )
            plot_image_profiles(output_dir, source, image_path, profiles, rows_for_image=[row for row in rows if row["image_path"] == rel_path(image_path)])
    return rows, white_roi, white_path


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    baseline = float(np.percentile(values, 10))
    high = float(np.percentile(values, 99))
    return np.clip((values - baseline) / max(high - baseline, 1e-9), 0, None)


def plot_image_profiles(output_dir: Path, source: str, image_path: Path, profiles: dict[str, np.ndarray], rows_for_image: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4", "Gray": "#444444"}
    y_axis = np.arange(len(profiles["Gray"]))
    for channel in CHANNELS:
        ax.plot(y_axis, normalize(profiles[channel]), label=channel, color=colors[channel], linewidth=1.1)
    for row in rows_for_image:
        y = float(row["peak_y_local"])
        ax.axvline(y, color=colors.get(row["channel"], "#888888"), alpha=0.22, linewidth=1)
    ax.set_title(f"{source} image peak candidates: {image_path.name}")
    ax.set_xlabel("y_local in white-reference ROI")
    ax.set_ylabel("normalized intensity")
    ax.set_ylim(0, 1.25)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    out_path = output_dir / f"image_peak_diagnostic_{safe_name(source)}_{safe_name(image_path.stem)}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_standard_peaks(output_dir: Path, reference_root: Path) -> None:
    fig_count = 0
    for source in REFERENCE_SOURCES:
        for file_path in reference_files_for_source(reference_root, source):
            table = parse_numeric_reference_file(file_path)
            chosen = choose_wavelength_intensity_columns(table)
            if chosen is None:
                continue
            wavelength, intensity = chosen
            peaks = find_profile_peaks(wavelength, intensity, max_peaks=12, min_distance_axis=0.3)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(wavelength, normalize(intensity), color="#222222", linewidth=1.1)
            for peak in peaks:
                ax.axvline(peak["axis_value"], color="#d62728", alpha=0.35)
                ax.text(peak["axis_value"], 1.03, f"{peak['axis_value']:.1f}", rotation=90, fontsize=8, ha="center", va="bottom")
            ax.set_xlim(TARGET_WAVELENGTH_MIN_NM, TARGET_WAVELENGTH_MAX_NM)
            ax.set_ylim(0, 1.25)
            ax.set_xlabel("wavelength (nm)")
            ax.set_ylabel("normalized intensity")
            ax.set_title(f"{source} standard spectrum peaks: {file_path.name}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / f"standard_peak_diagnostic_{safe_name(source)}_{safe_name(file_path.stem)}.png", dpi=180)
            plt.close(fig)
            fig_count += 1
    if fig_count == 0:
        return


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def channel_compatibility(wavelength_nm: float, channel: str) -> float:
    channel = str(channel)
    if wavelength_nm < 480:
        return {"B": 1.0, "Gray": 0.65, "G": 0.55, "R": 0.25}.get(channel, 0.3)
    if wavelength_nm < 560:
        return {"G": 1.0, "Gray": 0.75, "B": 0.45, "R": 0.35}.get(channel, 0.3)
    if wavelength_nm < 610:
        return {"R": 0.90, "G": 0.85, "Gray": 0.75, "B": 0.25}.get(channel, 0.3)
    return {"R": 1.0, "Gray": 0.65, "G": 0.35, "B": 0.25}.get(channel, 0.3)


def image_candidate_edge_penalty(row: dict) -> float:
    y = as_float(row, "peak_y_local")
    h = as_float(row, "roi_h", 0.0)
    if h <= 0:
        return 0.0
    edge_dist = min(y, h - y)
    if edge_dist < 20:
        return 0.45
    if edge_dist < 45:
        return 0.18
    return 0.0


def prepare_standard_candidates(standard_rows: list[dict]) -> tuple[list[dict], dict[str, float], list[str]]:
    by_source = defaultdict(list)
    warnings = []
    for row in standard_rows:
        wavelength = as_float(row, "standard_wavelength_nm", np.nan)
        if not np.isfinite(wavelength) or not (TARGET_WAVELENGTH_MIN_NM <= wavelength <= TARGET_WAVELENGTH_MAX_NM):
            continue
        item = dict(row)
        item["_wavelength"] = wavelength
        item["_std_prominence"] = as_float(row, "standard_prominence")
        item["_std_height"] = as_float(row, "standard_peak_height")
        by_source[row["source"]].append(item)

    max_prom = {source: max([item["_std_prominence"] for item in rows] + [1e-9]) for source, rows in by_source.items()}
    strong_by_source = {}
    for source, rows in by_source.items():
        threshold = 0.05 * max_prom[source]
        if source == "hene":
            threshold = 0.0
        strong = [item for item in rows if item["_std_prominence"] >= threshold]
        if not strong and rows:
            strong = [max(rows, key=lambda item: item["_std_prominence"])]
            warnings.append(f"{source}: no standard peak passed strength threshold; using strongest peak only")
        strong.sort(key=lambda item: item["_wavelength"])

        clustered = []
        for item in strong:
            if clustered and abs(item["_wavelength"] - clustered[-1][-1]["_wavelength"]) <= 2.5:
                clustered[-1].append(item)
            else:
                clustered.append([item])
        reduced = [max(cluster, key=lambda item: (item["_std_prominence"], item["_std_height"])) for cluster in clustered]
        strong_by_source[source] = reduced

    final = []
    for rows in strong_by_source.values():
        final.extend(rows)
    final.sort(key=lambda item: item["_wavelength"])
    return final, max_prom, warnings


def prepare_image_candidates(image_rows: list[dict]) -> tuple[list[dict], dict[str, float]]:
    max_prom = defaultdict(lambda: 1e-9)
    for row in image_rows:
        max_prom[row["source"]] = max(max_prom[row["source"]], as_float(row, "prominence"))

    candidates = []
    for row in image_rows:
        y = as_float(row, "peak_y_local", np.nan)
        if not np.isfinite(y):
            continue
        h = as_float(row, "roi_h", 0.0)
        edge_penalty = image_candidate_edge_penalty(row)
        if edge_penalty >= 0.45:
            continue
        item = dict(row)
        item["_y"] = y
        item["_img_prominence"] = as_float(row, "prominence")
        item["_img_height"] = as_float(row, "peak_height")
        item["_img_norm"] = item["_img_prominence"] / max(max_prom[row["source"]], 1e-9)
        item["_edge_penalty"] = edge_penalty
        item["_roi_h"] = h
        candidates.append(item)
    return candidates, dict(max_prom)


def make_candidate_matches(standard_rows: list[dict], image_rows: list[dict], top_n_per_standard: int = 3) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
    strong_standards, std_max_prom, _warnings = prepare_standard_candidates(standard_rows)
    image_candidates, img_max_prom = prepare_image_candidates(image_rows)
    images_by_source = defaultdict(list)
    for candidate in image_candidates:
        images_by_source[candidate["source"]].append(candidate)

    all_review_rows = []
    internal_matches_by_standard = defaultdict(list)
    strong_keys = set()
    for std in strong_standards:
        key = f"{std['source']}|{std['standard_file']}|{std['standard_peak_rank']}|{std['standard_wavelength_nm']}"
        strong_keys.add(key)

    for std in standard_rows:
        source = std["source"]
        wavelength = as_float(std, "standard_wavelength_nm")
        std_prom = as_float(std, "standard_prominence")
        std_norm = std_prom / max(std_max_prom.get(source, max(std_prom, 1e-9)), 1e-9)
        key = f"{source}|{std['standard_file']}|{std['standard_peak_rank']}|{std['standard_wavelength_nm']}"
        scored = []
        for image in images_by_source.get(source, []):
            channel_score = channel_compatibility(wavelength, image["channel"])
            image_norm = image["_img_prominence"] / max(img_max_prom.get(source, 1e-9), 1e-9)
            rank_penalty = min(0.18, 0.018 * max(0, as_int(image, "image_peak_rank") - 1))
            hint_bonus = 0.08 if std.get("possible_hint") else 0.0
            match_score = (
                0.35 * std_norm
                + 0.33 * image_norm
                + 0.22 * channel_score
                + hint_bonus
                - image["_edge_penalty"]
                - rank_penalty
            )
            scored.append((match_score, image, channel_score, image_norm))
        scored.sort(key=lambda item: item[0], reverse=True)

        for rank, (match_score, image, channel_score, image_norm) in enumerate(scored[:top_n_per_standard], start=1):
            row = {
                "source": source,
                "standard_file": std["standard_file"],
                "standard_peak_rank": std["standard_peak_rank"],
                "standard_peak_wavelength_nm": f"{wavelength:.6f}",
                "standard_peak_prominence": f"{std_prom:.9g}",
                "standard_possible_hint": std.get("possible_hint", ""),
                "candidate_rank_for_standard": rank,
                "candidate_image_path": image["image_path"],
                "candidate_channel": image["channel"],
                "candidate_y_local": f"{image['_y']:.3f}",
                "candidate_peak_height": image["peak_height"],
                "candidate_peak_prominence": image["prominence"],
                "candidate_image_rank": image["image_peak_rank"],
                "match_score": f"{match_score:.6f}",
                "channel_compatibility": f"{channel_score:.3f}",
                "selection_status": "",
                "selection_note": "",
            }
            all_review_rows.append(row)

        if key in strong_keys:
            for match_score, image, channel_score, image_norm in scored[:6]:
                internal = {
                    "standard": std,
                    "image": image,
                    "standard_key": key,
                    "wavelength": wavelength,
                    "y": image["_y"],
                    "match_score": float(match_score),
                    "channel_compatibility": float(channel_score),
                    "image_norm": float(image_norm),
                    "standard_norm": float(std_norm),
                }
                internal_matches_by_standard[key].append(internal)
    return all_review_rows, internal_matches_by_standard, strong_standards


def monotonic_check(coefficients: np.ndarray) -> tuple[bool, float]:
    s_grid = np.linspace(0.0, 1.0, 1000)
    values = np.polyval(coefficients, s_grid)
    diffs = np.diff(values)
    min_delta = float(np.min(diffs)) if diffs.size else 0.0
    return bool(np.all(diffs >= -1e-9)), min_delta


def fit_combo(matches: list[dict], force_degree: int | None = None) -> dict | None:
    sorted_matches = sorted(matches, key=lambda item: item["wavelength"])
    wavelengths = np.array([item["wavelength"] for item in sorted_matches], dtype=np.float64)
    y_values = np.array([item["y"] for item in sorted_matches], dtype=np.float64)
    if np.any(np.diff(wavelengths) <= 0):
        return None
    if np.any(np.diff(y_values) <= 12.0):
        return None
    if float(y_values[-1] - y_values[0]) < 80.0:
        return None
    if len({round(float(y), 1) for y in y_values}) != len(y_values):
        return None

    s_values = (y_values - y_values[0]) / (y_values[-1] - y_values[0])
    degree = int(force_degree) if force_degree is not None else (2 if len(sorted_matches) >= 4 else 1)
    if degree == 2 and len(sorted_matches) < 3:
        return None
    if degree == 1 and len(sorted_matches) < 2:
        return None
    coeff = np.polyfit(s_values, wavelengths, deg=degree)
    fitted = np.polyval(coeff, s_values)
    residuals = fitted - wavelengths
    monotonic, min_monotonic_delta = monotonic_check(coeff)
    max_abs = float(np.max(np.abs(residuals)))
    mean_abs = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    coverage = float(wavelengths[-1] - wavelengths[0])
    max_gap = float(np.max(np.diff(wavelengths))) if wavelengths.size > 1 else 0.0
    source_count = len({item["standard"]["source"] for item in sorted_matches})
    has_hene = any(item["standard"]["source"] == "hene" for item in sorted_matches)
    mean_match = float(np.mean([item["match_score"] for item in sorted_matches]))
    penalty = 0.0
    if coverage < 150:
        penalty += 30.0
    if max_gap > 130:
        penalty += 18.0
    elif max_gap > 115:
        penalty += 7.0
    if source_count < 2:
        penalty += 20.0
    if not has_hene:
        penalty += 12.0
    if len(sorted_matches) < 4:
        penalty += 100.0
    if len(sorted_matches) < 5:
        penalty += 3.0
    score = rmse + 0.25 * max_abs + 0.01 * max_gap - 0.30 * mean_match + penalty
    return {
        "matches": sorted_matches,
        "s": s_values,
        "coefficients": coeff,
        "model_type": "quadratic" if degree == 2 else "linear",
        "fitted": fitted,
        "residuals": residuals,
        "monotonic_on_0_1": monotonic,
        "min_monotonic_delta_nm": min_monotonic_delta,
        "max_abs_residual_nm": max_abs,
        "mean_abs_residual_nm": mean_abs,
        "rmse_nm": rmse,
        "coverage_nm": coverage,
        "max_anchor_gap_nm": max_gap,
        "source_count": source_count,
        "has_hene": has_hene,
        "score": float(score),
    }


def auto_select_anchors(
    internal_matches_by_standard: dict[str, list[dict]],
    strong_standards: list[dict],
    force_degree: int | None = None,
) -> tuple[dict | None, list[str]]:
    warnings = []
    standards_with_matches = []
    for std in strong_standards:
        key = f"{std['source']}|{std['standard_file']}|{std['standard_peak_rank']}|{std['standard_wavelength_nm']}"
        matches = internal_matches_by_standard.get(key, [])
        if matches:
            standards_with_matches.append((key, std, matches[:5]))
    if len(standards_with_matches) < 4:
        return None, [f"Only {len(standards_with_matches)} strong standard peaks have reliable image candidates; need at least 4"]

    best = None
    max_subset = min(6, len(standards_with_matches))
    for subset_size in range(max_subset, 3, -1):
        for subset in itertools.combinations(standards_with_matches, subset_size):
            match_lists = [item[2] for item in subset]
            combo_counter = 0
            for combo in itertools.product(*match_lists):
                combo_counter += 1
                if combo_counter > 25000:
                    break
                candidate = fit_combo(list(combo), force_degree=force_degree)
                if candidate is None:
                    continue
                if best is None or candidate["score"] < best["score"]:
                    best = candidate
    if best is None:
        warnings.append("No monotonic 4-6 anchor combination could be fit from current standard/image candidates")
    return best, warnings


def write_anchor_match_review(path: Path, review_rows: list[dict], selected_fit: dict | None) -> None:
    selected_keys = set()
    if selected_fit:
        for match in selected_fit["matches"]:
            selected_keys.add(
                (
                    match["standard"]["source"],
                    str(match["standard"]["standard_peak_rank"]),
                    f"{match['wavelength']:.6f}",
                    match["image"]["image_path"],
                    match["image"]["channel"],
                    f"{match['y']:.3f}",
                )
            )
    for row in review_rows:
        key = (
            row["source"],
            str(row["standard_peak_rank"]),
            row["standard_peak_wavelength_nm"],
            row["candidate_image_path"],
            row["candidate_channel"],
            row["candidate_y_local"],
        )
        if key in selected_keys:
            row["selection_status"] = "selected"
            row["selection_note"] = "chosen by automatic monotonic fit search"
    fieldnames = [
        "source",
        "standard_file",
        "standard_peak_rank",
        "standard_peak_wavelength_nm",
        "standard_peak_prominence",
        "standard_possible_hint",
        "candidate_rank_for_standard",
        "candidate_image_path",
        "candidate_channel",
        "candidate_y_local",
        "candidate_peak_height",
        "candidate_peak_prominence",
        "candidate_image_rank",
        "match_score",
        "channel_compatibility",
        "selection_status",
        "selection_note",
    ]
    write_csv(path, review_rows, fieldnames)


def selected_anchor_rows(selected_fit: dict | None) -> list[dict]:
    if not selected_fit:
        return []
    rows = []
    for idx, match in enumerate(selected_fit["matches"]):
        std = match["standard"]
        image = match["image"]
        rows.append(
            {
                "source": std["source"],
                "standard_peak_wavelength_nm": f"{match['wavelength']:.6f}",
                "standard_peak_rank": std["standard_peak_rank"],
                "selected_y_local": f"{match['y']:.3f}",
                "selected_channel": image["channel"],
                "image_peak_height": image["peak_height"],
                "image_peak_prominence": image["prominence"],
                "relative_s": f"{selected_fit['s'][idx]:.9f}",
                "fit_used": "yes",
                "selection_reason": (
                    "high standard/image prominence, channel-compatible candidate, "
                    "unique y position, monotonic wavelength coverage"
                ),
                "image_path": image["image_path"],
                "standard_file": std["standard_file"],
            }
        )
    return rows


def residual_rows(selected_fit: dict | None) -> list[dict]:
    if not selected_fit:
        return []
    rows = []
    for idx, match in enumerate(selected_fit["matches"]):
        rows.append(
            {
                "source": match["standard"]["source"],
                "standard_peak_wavelength_nm": f"{match['wavelength']:.6f}",
                "selected_y_local": f"{match['y']:.3f}",
                "relative_s": f"{selected_fit['s'][idx]:.9f}",
                "fitted_wavelength_nm": f"{selected_fit['fitted'][idx]:.6f}",
                "residual_nm": f"{selected_fit['residuals'][idx]:.6f}",
            }
        )
    return rows


def calibration_recommendation(selected_fit: dict | None) -> tuple[bool, list[str], str]:
    if selected_fit is None:
        return False, ["no valid automatic anchor fit was found"], "failed"
    warnings = []
    if len(selected_fit["matches"]) < 4:
        warnings.append("fewer than 4 anchors selected")
    if selected_fit["coverage_nm"] < 150:
        warnings.append(f"anchor wavelength coverage is narrow: {selected_fit['coverage_nm']:.2f} nm")
    if selected_fit.get("max_anchor_gap_nm", 0.0) > 130:
        warnings.append(f"anchor distribution has a large wavelength gap: {selected_fit['max_anchor_gap_nm']:.2f} nm")
    if not selected_fit["has_hene"]:
        warnings.append("He-Ne/long red-end anchor is missing; red end is unreliable")
    if not selected_fit.get("monotonic_on_0_1", False):
        warnings.append("model is not monotonic nondecreasing on s=0..1")
    if selected_fit["max_abs_residual_nm"] > 5:
        warnings.append(f"low confidence: max residual {selected_fit['max_abs_residual_nm']:.3f} nm exceeds 5 nm")
    recommended = True
    if selected_fit["max_abs_residual_nm"] > 10:
        warnings.append("recommended_for_next_step=false because max residual exceeds 10 nm")
        recommended = False
    if (
        selected_fit["coverage_nm"] < 150
        or selected_fit.get("max_anchor_gap_nm", 0.0) > 130
        or not selected_fit["has_hene"]
        or not selected_fit.get("monotonic_on_0_1", False)
        or len(selected_fit["matches"]) < 4
    ):
        recommended = False
    confidence = "low" if warnings else "diagnostic_pass"
    return recommended, warnings, confidence


def formula_for_fit(fit: dict) -> str:
    coeff = [float(value) for value in fit["coefficients"]]
    if fit["model_type"] == "quadratic":
        return f"wavelength_nm = {coeff[0]:.12g}*s^2 + {coeff[1]:.12g}*s + {coeff[2]:.12g}"
    return f"wavelength_nm = {coeff[0]:.12g}*s + {coeff[1]:.12g}"


def fit_summary(fit: dict | None) -> dict:
    if fit is None:
        return {}
    return {
        "model_type": fit["model_type"],
        "formula": formula_for_fit(fit),
        "coefficients": [float(value) for value in fit["coefficients"]],
        "selected_anchor_count": len(fit["matches"]),
        "max_abs_residual_nm": fit["max_abs_residual_nm"],
        "mean_abs_residual_nm": fit["mean_abs_residual_nm"],
        "rmse_nm": fit["rmse_nm"],
        "wavelength_coverage_nm": fit["coverage_nm"],
        "max_anchor_gap_nm": fit["max_anchor_gap_nm"],
        "monotonic_on_0_1": fit["monotonic_on_0_1"],
        "min_monotonic_delta_nm": fit["min_monotonic_delta_nm"],
    }


def write_single_fit_json(path: Path, fit: dict | None, recommended: bool, warnings: list[str]) -> None:
    if fit is None:
        payload = {
            "calibration_type": "relative_spectral_coordinate_diagnostic",
            "model_type": None,
            "formula": None,
            "coefficients": [],
            "selected_anchor_count": 0,
            "selected_anchors": [],
            "residual_summary": {},
            "monotonic_on_0_1": False,
            "warning": "; ".join(warnings),
            "recommended_for_next_step": False,
        }
    else:
        payload = {
            "calibration_type": "relative_spectral_coordinate_diagnostic",
            "model_type": fit["model_type"],
            "formula": formula_for_fit(fit),
            "coefficients": [float(value) for value in fit["coefficients"]],
            "selected_anchor_count": len(fit["matches"]),
            "selected_anchors": selected_anchor_rows(fit),
            "residual_summary": {
                "max_abs_residual_nm": fit["max_abs_residual_nm"],
                "mean_abs_residual_nm": fit["mean_abs_residual_nm"],
                "rmse_nm": fit["rmse_nm"],
                "wavelength_coverage_nm": fit["coverage_nm"],
                "max_anchor_gap_nm": fit["max_anchor_gap_nm"],
            },
            "monotonic_on_0_1": fit["monotonic_on_0_1"],
            "min_monotonic_delta_nm": fit["min_monotonic_delta_nm"],
            "warning": "; ".join(warnings) if warnings else "diagnostic calibration only; not production calibration",
            "recommended_for_next_step": bool(recommended),
        }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_fit_outputs(
    output_dir: Path,
    quadratic_fit: dict | None,
    quadratic_selection_warnings: list[str],
    linear_fit: dict | None,
    linear_selection_warnings: list[str],
) -> tuple[dict | None, bool, list[str], str, dict]:
    linear_recommended, linear_qc_warnings, linear_confidence = calibration_recommendation(linear_fit)
    quadratic_recommended_raw, quadratic_qc_warnings, _quadratic_confidence = calibration_recommendation(quadratic_fit)
    # Quadratic is retained only for diagnostics; it is never the final recommended model.
    quadratic_recommended = False
    if quadratic_recommended_raw:
        quadratic_qc_warnings.append("quadratic retained only as diagnostic; final recommendation priority is linear")

    final_fit = linear_fit if linear_recommended else None
    recommended = bool(linear_recommended)
    final_model = "linear" if linear_recommended else "none"
    selected_rows = selected_anchor_rows(final_fit)
    write_csv(
        output_dir / "selected_anchors.csv",
        selected_rows,
        [
            "source",
            "standard_peak_wavelength_nm",
            "standard_peak_rank",
            "selected_y_local",
            "selected_channel",
            "image_peak_height",
            "image_peak_prominence",
            "relative_s",
            "fit_used",
            "selection_reason",
            "image_path",
            "standard_file",
        ],
    )
    write_csv(
        output_dir / "relative_calibration_residuals.csv",
        residual_rows(final_fit),
        ["source", "standard_peak_wavelength_nm", "selected_y_local", "relative_s", "fitted_wavelength_nm", "residual_nm"],
    )

    linear_warnings = linear_selection_warnings + linear_qc_warnings
    quadratic_warnings = quadratic_selection_warnings + quadratic_qc_warnings
    write_single_fit_json(output_dir / "relative_calibration_fit_linear.json", linear_fit, linear_recommended, linear_warnings)

    payload = {
        "calibration_type": "relative_spectral_coordinate_diagnostic",
        "final_recommended_model": final_model,
        "recommended_for_next_step": recommended,
        "linear": fit_summary(linear_fit),
        "linear_warnings": linear_warnings,
        "quadratic_diagnostic": fit_summary(quadratic_fit),
        "quadratic_warnings": quadratic_warnings,
        "selected_anchor_count": len(final_fit["matches"]) if final_fit is not None else 0,
        "selected_anchors": selected_rows,
        "warning": "; ".join(linear_warnings if not recommended else ["diagnostic calibration only; not production calibration"]),
    }
    (output_dir / "relative_calibration_fit_diagnostic.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    if final_fit is not None:
        plot_relative_curve(output_dir / "relative_calibration_curve.png", final_fit, formula_for_fit(final_fit))
    plot_curve_compare(output_dir / "relative_calibration_curve_compare.png", linear_fit, quadratic_fit)
    all_warnings = linear_warnings + [f"quadratic: {warning}" for warning in quadratic_warnings]
    return final_fit, recommended, all_warnings, final_model, {
        "linear_recommended": linear_recommended,
        "linear_warnings": linear_warnings,
        "quadratic_recommended": quadratic_recommended,
        "quadratic_warnings": quadratic_warnings,
        "linear_confidence": linear_confidence,
    }


def plot_relative_curve(path: Path, selected_fit: dict, formula: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    s = selected_fit["s"]
    wavelengths = np.array([match["wavelength"] for match in selected_fit["matches"]], dtype=np.float64)
    s_grid = np.linspace(-0.05, 1.05, 400)
    fitted_grid = np.polyval(selected_fit["coefficients"], s_grid)
    ax.plot(s_grid, fitted_grid, color="#1f77b4", label=selected_fit["model_type"])
    ax.scatter(s, wavelengths, color="#d62728", zorder=3, label="selected anchors")
    for idx, match in enumerate(selected_fit["matches"]):
        label = f"{match['standard']['source']} {match['wavelength']:.1f}"
        ax.text(float(s[idx]), float(wavelengths[idx]), label, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("relative spectral coordinate s")
    ax.set_ylabel("wavelength (nm)")
    ax.set_ylim(TARGET_WAVELENGTH_MIN_NM, TARGET_WAVELENGTH_MAX_NM)
    ax.set_title("Diagnostic relative calibration curve\n" + formula)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_curve_compare(path: Path, linear_fit: dict | None, quadratic_fit: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    s_grid = np.linspace(0.0, 1.0, 1000)
    if quadratic_fit is not None:
        ax.plot(
            s_grid,
            np.polyval(quadratic_fit["coefficients"], s_grid),
            color="#d62728",
            label=f"quadratic diagnostic, monotonic={quadratic_fit['monotonic_on_0_1']}",
            linewidth=1.4,
        )
        q_s = quadratic_fit["s"]
        q_w = np.array([match["wavelength"] for match in quadratic_fit["matches"]], dtype=np.float64)
        ax.scatter(q_s, q_w, color="#d62728", s=36, alpha=0.6)
    if linear_fit is not None:
        ax.plot(
            s_grid,
            np.polyval(linear_fit["coefficients"], s_grid),
            color="#1f77b4",
            label=f"linear candidate, monotonic={linear_fit['monotonic_on_0_1']}",
            linewidth=1.8,
        )
        l_s = linear_fit["s"]
        l_w = np.array([match["wavelength"] for match in linear_fit["matches"]], dtype=np.float64)
        ax.scatter(l_s, l_w, color="#1f77b4", s=55, marker="*", zorder=3)
    ax.set_xlabel("relative spectral coordinate s")
    ax.set_ylabel("wavelength (nm)")
    ax.set_ylim(TARGET_WAVELENGTH_MIN_NM, TARGET_WAVELENGTH_MAX_NM)
    ax.set_title("Relative calibration curve comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_anchor_match_review(path: Path, review_rows: list[dict], selected_fit: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"hg": "#1f77b4", "na": "#ff7f0e", "hene": "#d62728"}
    for row in review_rows:
        x = as_float(row, "candidate_y_local", np.nan)
        y = as_float(row, "standard_peak_wavelength_nm", np.nan)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        selected = row.get("selection_status") == "selected"
        ax.scatter(
            x,
            y,
            s=95 if selected else 24,
            color=colors.get(row["source"], "#777777"),
            edgecolor="black" if selected else "none",
            alpha=0.95 if selected else 0.35,
            marker="*" if selected else "o",
        )
    ax.set_xlabel("candidate image y_local")
    ax.set_ylabel("standard peak wavelength (nm)")
    ax.set_ylim(TARGET_WAVELENGTH_MIN_NM, TARGET_WAVELENGTH_MAX_NM)
    ax.set_title("Standard-to-image peak match review\nselected anchors are star markers")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    reference_root: Path,
    image_root: Path,
    standard_rows: list[dict],
    image_rows: list[dict],
    files_by_source: dict[str, list[Path]],
    warnings: list[str],
    white_roi: tuple[int, int, int, int] | None,
    white_path: Path | None,
    selected_fit: dict | None = None,
    fit_warnings: list[str] | None = None,
    recommended_for_next_step: bool = False,
    fit_compare: dict | None = None,
) -> None:
    fit_warnings = fit_warnings or []
    lines = [
        "# Relative Spectral Coordinate Calibration Diagnostic",
        "",
        "This is a diagnostic calibration, not a final high-confidence calibration.",
        "No old ROI and no old pixel-to-wavelength formula are used.",
        "Known physical wavelengths may appear only as possible hints; selected anchor wavelengths come from `standard_peaks.csv`.",
        f"Target wavelength interval: {TARGET_WAVELENGTH_MIN_NM:.0f}-{TARGET_WAVELENGTH_MAX_NM:.0f} nm.",
        "",
        f"- reference_root: `{rel_path(reference_root)}`",
        f"- image_root: `{rel_path(image_root)}`",
        "",
        "## Reference Spectrum Files",
        "",
    ]
    for source in REFERENCE_SOURCES:
        files = files_by_source.get(source, [])
        lines.append(f"- {source}: {len(files)}")
        for file_path in files:
            lines.append(f"  - `{rel_path(file_path)}`")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(["", "## Standard Peaks", ""])
    lines.append(f"- detected standard peaks: {len(standard_rows)}")
    for row in standard_rows:
        hint = f" ({row['possible_hint']})" if row["possible_hint"] else ""
        lines.append(f"- {row['source']}: {row['standard_wavelength_nm']} nm{hint} from `{row['standard_file']}`")

    lines.extend(["", "## Image Peak Candidates", ""])
    image_counts = defaultdict(int)
    for row in image_rows:
        image_counts[row["source"]] += 1
    for source in REFERENCE_SOURCES:
        lines.append(f"- {source}: {image_counts[source]} candidates")
    if white_path and white_roi:
        lines.extend(["", "## White Reference ROI For Image Profiles", ""])
        lines.append(f"- white image: `{rel_path(white_path)}`")
        lines.append(f"- ROI from current white full image: x={white_roi[0]}, y={white_roi[1]}, w={white_roi[2]}, h={white_roi[3]}")
        lines.append("- this is a temporary profile extraction range, not a final calibration ROI")

    lines.extend(["", "## Automatic Anchor Selection", ""])
    if selected_fit is None:
        lines.append("- automatic fit failed")
        for warning in fit_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append(f"- selected anchors: {len(selected_fit['matches'])}")
        lines.append(f"- model_type: {selected_fit['model_type']}")
        coeff = [float(value) for value in selected_fit["coefficients"]]
        if selected_fit["model_type"] == "quadratic":
            formula = f"wavelength_nm = {coeff[0]:.12g}*s^2 + {coeff[1]:.12g}*s + {coeff[2]:.12g}"
        else:
            formula = f"wavelength_nm = {coeff[0]:.12g}*s + {coeff[1]:.12g}"
        lines.append(f"- formula: `{formula}`")
        lines.append(f"- max residual: {selected_fit['max_abs_residual_nm']:.6f} nm")
        lines.append(f"- mean abs residual: {selected_fit['mean_abs_residual_nm']:.6f} nm")
        lines.append(f"- RMSE: {selected_fit['rmse_nm']:.6f} nm")
        lines.append(f"- wavelength coverage: {selected_fit['coverage_nm']:.6f} nm")
        lines.append(f"- max anchor gap: {selected_fit['max_anchor_gap_nm']:.6f} nm")
        lines.append(f"- recommended_for_next_step: {str(recommended_for_next_step).lower()}")
        if fit_warnings:
            lines.append("- quality warnings:")
            for warning in fit_warnings:
                lines.append(f"  - {warning}")
        lines.extend(["", "### Selected Anchors", ""])
        lines.append("| source | wavelength_nm | y_local | s | channel | residual_nm | reason |")
        lines.append("|---|---:|---:|---:|---|---:|---|")
        for idx, match in enumerate(selected_fit["matches"]):
            residual = float(selected_fit["residuals"][idx])
            lines.append(
                f"| {match['standard']['source']} | {match['wavelength']:.6f} | {match['y']:.3f} | "
                f"{selected_fit['s'][idx]:.6f} | {match['image']['channel']} | {residual:.6f} | "
                "strong standard peak + clear image candidate + monotonic coverage |"
            )
        lines.extend(["", "### Exclusion Notes", ""])
        lines.append("- weak standard-spectrum peaks were not used as final anchors")
        lines.append("- image peaks near ROI edges or with low prominence were penalized or excluded")
        lines.append("- duplicate or nearly identical image y positions were not allowed in the same fit")
        lines.append("- non-monotonic wavelength-to-y combinations were rejected")

    if fit_compare:
        lines.extend(["", "## Linear vs Quadratic Model Check", ""])
        linear = fit_compare.get("linear", {})
        quadratic = fit_compare.get("quadratic_diagnostic", {})
        lines.append("| model | max residual nm | mean abs residual nm | RMSE nm | monotonic | recommended |")
        lines.append("|---|---:|---:|---:|---|---|")
        if quadratic:
            lines.append(
                f"| quadratic diagnostic | {quadratic.get('max_abs_residual_nm', float('nan')):.6f} | "
                f"{quadratic.get('mean_abs_residual_nm', float('nan')):.6f} | "
                f"{quadratic.get('rmse_nm', float('nan')):.6f} | "
                f"{str(quadratic.get('monotonic_on_0_1', False)).lower()} | false |"
            )
        if linear:
            lines.append(
                f"| linear | {linear.get('max_abs_residual_nm', float('nan')):.6f} | "
                f"{linear.get('mean_abs_residual_nm', float('nan')):.6f} | "
                f"{linear.get('rmse_nm', float('nan')):.6f} | "
                f"{str(linear.get('monotonic_on_0_1', False)).lower()} | "
                f"{str(fit_compare.get('linear_recommended', False)).lower()} |"
            )
        lines.append(f"- final_recommended_model: `{fit_compare.get('final_recommended_model', 'none')}`")
        if quadratic and not quadratic.get("monotonic_on_0_1", False):
            lines.append("- quadratic is not recommended because it is not monotonic on s=0..1")
        if linear and linear.get("max_abs_residual_nm", float("inf")) > 10:
            lines.append("- linear is not recommended because max residual exceeds 10 nm")

    lines.extend(["", "## Output Files", ""])
    output_dir = path.parent
    for name in [
        "standard_peaks.csv",
        "image_peak_candidates.csv",
        "anchor_match_review.csv",
        "selected_anchors.csv",
        "relative_calibration_fit_diagnostic.json",
        "relative_calibration_fit_linear.json",
        "relative_calibration_residuals.csv",
        "relative_calibration_curve.png",
        "relative_calibration_curve_compare.png",
        "anchor_match_review.png",
    ]:
        candidate = output_dir / name
        if candidate.exists():
            lines.append(f"- `{rel_path(candidate)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-pass standard-spectrum and DIY-image peak candidate matching.")
    parser.add_argument("--reference_root", default=str(DEFAULT_REFERENCE_ROOT), help="Directory containing hg.*, na.*, hene.* standard spectra.")
    parser.add_argument("--image_root", default=str(DEFAULT_IMAGE_ROOT), help="Directory containing current *_full.png captures.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_root = Path(args.reference_root)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    if not reference_root.is_absolute():
        reference_root = PROJECT_ROOT / reference_root
    if not image_root.is_absolute():
        image_root = PROJECT_ROOT / image_root
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_rows, files_by_source, warnings = load_standard_peaks(reference_root)
    image_rows, white_roi, white_path = load_image_peak_candidates(image_root, output_dir)
    plot_standard_peaks(output_dir, reference_root)

    write_csv(
        output_dir / "standard_peaks.csv",
        standard_rows,
        ["source", "standard_file", "standard_peak_rank", "standard_wavelength_nm", "standard_peak_height", "standard_prominence", "possible_hint"],
    )
    write_csv(
        output_dir / "image_peak_candidates.csv",
        image_rows,
        [
            "source",
            "image_path",
            "channel",
            "image_peak_rank",
            "peak_y_local",
            "peak_height",
            "prominence",
            "roi_x",
            "roi_y",
            "roi_w",
            "roi_h",
            "possible_hint",
        ],
    )
    review_rows, internal_matches_by_standard, strong_standards = make_candidate_matches(standard_rows, image_rows, top_n_per_standard=3)
    quadratic_fit, quadratic_selection_warnings = auto_select_anchors(
        internal_matches_by_standard,
        strong_standards,
        force_degree=2,
    )
    linear_fit, linear_selection_warnings = auto_select_anchors(
        internal_matches_by_standard,
        strong_standards,
        force_degree=1,
    )
    final_fit, recommended, fit_warnings, final_model, fit_compare_extra = write_fit_outputs(
        output_dir,
        quadratic_fit,
        quadratic_selection_warnings,
        linear_fit,
        linear_selection_warnings,
    )
    review_fit = final_fit if final_fit is not None else quadratic_fit
    write_anchor_match_review(output_dir / "anchor_match_review.csv", review_rows, review_fit)
    plot_anchor_match_review(output_dir / "anchor_match_review.png", review_rows, review_fit)

    fit_compare = {
        "linear": fit_summary(linear_fit),
        "quadratic_diagnostic": fit_summary(quadratic_fit),
        "linear_recommended": fit_compare_extra.get("linear_recommended", False),
        "quadratic_recommended": fit_compare_extra.get("quadratic_recommended", False),
        "final_recommended_model": final_model,
    }

    write_report(
        output_dir / "relative_calibration_report.md",
        reference_root,
        image_root,
        standard_rows,
        image_rows,
        files_by_source,
        warnings,
        white_roi,
        white_path,
        selected_fit=final_fit,
        fit_warnings=fit_warnings,
        recommended_for_next_step=recommended,
        fit_compare=fit_compare,
    )

    print("Relative calibration diagnostic outputs:")
    print(f"  reference_root: {rel_path(reference_root)}")
    print(f"  standard peaks: {len(standard_rows)}")
    print(f"  image peak candidates: {len(image_rows)}")
    print(f"  selected anchors: {len(final_fit['matches']) if final_fit else 0}")
    print(f"  standard_peaks.csv: {rel_path(output_dir / 'standard_peaks.csv')}")
    print(f"  image_peak_candidates.csv: {rel_path(output_dir / 'image_peak_candidates.csv')}")
    print(f"  anchor_match_review.csv: {rel_path(output_dir / 'anchor_match_review.csv')}")
    print(f"  selected_anchors.csv: {rel_path(output_dir / 'selected_anchors.csv')}")
    print(f"  fit_json: {rel_path(output_dir / 'relative_calibration_fit_diagnostic.json')}")
    if final_fit:
        formula = formula_for_fit(final_fit)
        print(f"  formula: {formula}")
        print(f"  max_abs_residual_nm: {final_fit['max_abs_residual_nm']:.6f}")
        print(f"  mean_abs_residual_nm: {final_fit['mean_abs_residual_nm']:.6f}")
        print(f"  rmse_nm: {final_fit['rmse_nm']:.6f}")
        print(f"  monotonic_on_0_1: {final_fit['monotonic_on_0_1']}")
        print(f"  final_recommended_model: {final_model}")
        print(f"  recommended_for_next_step: {recommended}")
    else:
        print("  fit failed")
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    if fit_warnings:
        print("  fit warnings:")
        for warning in fit_warnings:
            print(f"    - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
