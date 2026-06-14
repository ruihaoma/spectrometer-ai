import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "new_capture_roi_inspection"

EXPECTED_SOURCES = [
    "purple_led",
    "yellow_led",
    "blue_led",
    "green_led",
    "red_led",
    "white_led",
    "dark",
    "hene",
    "hg",
    "na",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:120] or "item"


def read_image_rgb(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def paired_metadata_path(image_path: Path) -> Path | None:
    name = image_path.name
    candidates = []
    if "_full" in name:
        candidates.append(image_path.with_name(name.replace("_full", "_meta")).with_suffix(".json"))
    candidates.append(image_path.with_suffix(".json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_metadata(image_path: Path) -> dict:
    meta_path = paired_metadata_path(image_path)
    if meta_path is None:
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def infer_source(image_path: Path, metadata: dict) -> str:
    meta_source = str(metadata.get("source", "")).strip().lower()
    if meta_source:
        return meta_source

    parts = [part.lower() for part in image_path.parts]
    filename = image_path.name.lower()
    for source in sorted(EXPECTED_SOURCES, key=len, reverse=True):
        if source in parts or source in filename:
            return source
    stem = image_path.stem.lower()
    if "_" in stem:
        return stem.split("_")[0]
    return "unknown"


def is_full_capture_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    lowered = path.name.lower()
    if any(tag in lowered for tag in ("_roi", "roi_crop", "diagnostic", "profile")):
        return False
    return "_full" in lowered


def scan_images(input_root: Path) -> list[Path]:
    full_images = [path for path in input_root.rglob("*") if path.is_file() and is_full_capture_image(path)]
    if full_images:
        return sorted(full_images)

    fallback = []
    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lowered = path.name.lower()
        if any(tag in lowered for tag in ("_roi", "roi_crop", "diagnostic", "profile")):
            continue
        fallback.append(path)
    return sorted(fallback)


def robust_score(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    max_ch = np.max(rgb_f, axis=2)
    min_ch = np.min(rgb_f, axis=2)
    saturation = np.zeros_like(max_ch, dtype=np.float32)
    nonzero = max_ch > 1e-6
    saturation[nonzero] = (max_ch[nonzero] - min_ch[nonzero]) / max_ch[nonzero]
    score = 0.55 * gray + 0.30 * max_ch + 0.15 * saturation * max_ch
    return np.clip(score, 0, 255), gray, saturation


def threshold_score(score: np.ndarray) -> float:
    score8 = np.clip(score, 0, 255).astype(np.uint8)
    _otsu, _ = cv2.threshold(score8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) + 1e-6
    p95 = float(np.percentile(score, 95))
    p99 = float(np.percentile(score, 99))
    return max(float(_otsu), median + 3.5 * mad, 0.50 * p95, 0.35 * p99)


def connected_bbox(mask: np.ndarray, score: np.ndarray, saturation: np.ndarray) -> tuple[tuple[int, int, int, int] | None, np.ndarray]:
    h, w = mask.shape
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open)

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if labels_count <= 1:
        return None, clean

    min_area = max(80, int(0.00003 * h * w))
    component_scores = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == label)
        mean_score = float(score[ys, xs].mean()) if ys.size else 0.0
        mean_sat = float(saturation[ys, xs].mean()) if ys.size else 0.0
        component_scores.append((area * mean_score * (1.0 + mean_sat), label))

    if not component_scores:
        return None, clean

    component_scores.sort(reverse=True)
    main_label = component_scores[0][1]
    main_x = int(stats[main_label, cv2.CC_STAT_LEFT])
    main_w = int(stats[main_label, cv2.CC_STAT_WIDTH])
    main_x2 = main_x + main_w

    selected = labels == main_label
    for _score_value, label in component_scores[1:]:
        x = int(stats[label, cv2.CC_STAT_LEFT])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        x2 = x + comp_w
        horizontal_overlap = max(0, min(main_x2, x2) - max(main_x, x))
        overlap_ratio = horizontal_overlap / max(1, min(main_w, comp_w))
        close_in_x = abs((x + x2) / 2 - (main_x + main_x2) / 2) < max(main_w, comp_w, 80)
        if overlap_ratio > 0.15 or close_in_x:
            selected |= labels == label

    ys, xs = np.where(selected)
    if xs.size == 0:
        return None, clean
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1), selected.astype(np.uint8)


def expand_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    image_h, image_w = image_shape[:2]
    margin_x = max(12, int(0.15 * w))
    margin_y = max(12, int(0.15 * h))
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(image_w, x + w + margin_x)
    y1 = min(image_h, y + h + margin_y)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def fallback_roi(image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    roi_w = max(1, int(w * 0.25))
    roi_h = max(1, int(h * 0.30))
    return (max(0, (w - roi_w) // 2), max(0, (h - roi_h) // 2), roi_w, roi_h)


def estimate_tilt(mask_roi: np.ndarray, score_roi: np.ndarray) -> tuple[float | None, bool]:
    if mask_roi.sum() < 40:
        return None, False
    ys, xs = np.where(mask_roi > 0)
    weights = score_roi[ys, xs].astype(np.float64)
    if weights.sum() <= 0:
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


def top_blue_bottom_red(rgb_roi: np.ndarray) -> tuple[bool, str]:
    if rgb_roi.shape[0] < 6 or rgb_roi.shape[1] < 2:
        return False, "ROI too small for color-order check"
    rgb_f = rgb_roi.astype(np.float32)
    h = rgb_f.shape[0]
    top = rgb_f[: max(1, h // 3)]
    bottom = rgb_f[-max(1, h // 3) :]
    top_b = float(top[..., 2].mean())
    top_r = float(top[..., 0].mean())
    bottom_b = float(bottom[..., 2].mean())
    bottom_r = float(bottom[..., 0].mean())
    contrast = max(float(rgb_f.max() - rgb_f.min()), 1.0)
    blue_delta = (top_b - bottom_b) / contrast
    red_delta = (bottom_r - top_r) / contrast
    likely = blue_delta > 0.03 and red_delta > 0.03
    detail = f"blue_delta={blue_delta:.3f}, red_delta={red_delta:.3f}"
    return likely, detail


def fbool(value: bool) -> str:
    return "yes" if bool(value) else "no"


def make_profile_csv(path: Path, rgb_roi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_f = rgb_roi.astype(np.float32)
    r = rgb_f[..., 0].mean(axis=1)
    g = rgb_f[..., 1].mean(axis=1)
    b = rgb_f[..., 2].mean(axis=1)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["y_local", "R", "G", "B", "Gray"])
        for idx in range(rgb_roi.shape[0]):
            writer.writerow([idx, f"{r[idx]:.6f}", f"{g[idx]:.6f}", f"{b[idx]:.6f}", f"{gray[idx]:.6f}"])


def draw_diagnostic(rgb: np.ndarray, roi: tuple[int, int, int, int], row: dict, out_path: Path) -> None:
    diag = rgb.copy()
    x, y, w, h = roi
    cv2.rectangle(diag, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 4)
    lines = [
        f"{row['source']} | {Path(row['image_path']).name}",
        f"ROI x={x} y={y} w={w} h={h}",
        f"inside={row['spectrum_inside_roi']} clip={row['clipped_or_touching_edge']} over={row['overexposed']} dark={row['underexposed']}",
        f"blue_top_red_bottom={row['top_blue_bottom_red_likely']} tilt={row['tilt_warning']}",
    ]
    y_text = 38
    for line in lines:
        cv2.putText(diag, line, (30, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(diag, line, (30, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        y_text += 38
    write_image_rgb(out_path, diag)


def analyze_image(image_path: Path, output_dir: Path) -> dict:
    metadata = load_metadata(image_path)
    source = infer_source(image_path, metadata)
    rgb = read_image_rgb(image_path)
    if rgb is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    score, gray, saturation = robust_score(rgb)
    threshold = threshold_score(score)
    mask = score >= threshold
    bbox, selected_mask = connected_bbox(mask, score, saturation)

    detection_ok = bbox is not None and np.count_nonzero(selected_mask) > 80
    if detection_ok:
        roi = expand_bbox(bbox, rgb.shape)
        comment_parts = ["auto ROI from bright/color stripe"]
    else:
        roi = fallback_roi(rgb.shape)
        selected_mask = np.zeros(gray.shape, dtype=np.uint8)
        comment_parts = ["no clear spectrum stripe detected; fallback crop only"]

    x, y, w, h = roi
    rgb_roi = rgb[y : y + h, x : x + w]
    score_roi = score[y : y + h, x : x + w]
    gray_roi = gray[y : y + h, x : x + w]
    mask_roi = selected_mask[y : y + h, x : x + w] if selected_mask is not None else np.zeros((h, w), dtype=np.uint8)

    image_h, image_w = rgb.shape[:2]
    roi_touches_image_edge = x <= 2 or y <= 2 or x + w >= image_w - 2 or y + h >= image_h - 2
    mask_touches_edge = False
    if detection_ok:
        edge_pixels = (
            selected_mask[:3, :].sum()
            + selected_mask[-3:, :].sum()
            + selected_mask[:, :3].sum()
            + selected_mask[:, -3:].sum()
        )
        mask_touches_edge = edge_pixels > 0

    clipped_or_touching_edge = bool(roi_touches_image_edge or mask_touches_edge)
    saturated_full_ratio = float((rgb >= 250).any(axis=2).mean())
    saturated_roi_ratio = float((rgb_roi >= 250).any(axis=2).mean()) if rgb_roi.size else 0.0
    overexposed = saturated_roi_ratio > 0.01 or saturated_full_ratio > 0.005

    source_is_dark = source == "dark"
    roi_p99_gray = float(np.percentile(gray_roi, 99)) if gray_roi.size else 0.0
    roi_p99_score = float(np.percentile(score_roi, 99)) if score_roi.size else 0.0
    underexposed = (roi_p99_gray < 25.0 or roi_p99_score < 30.0) and not source_is_dark
    if source_is_dark:
        comment_parts.append("dark source; low signal is expected")

    selected_pixels = int(np.count_nonzero(selected_mask))
    inside_pixels = int(np.count_nonzero(mask_roi))
    inside_ratio = inside_pixels / max(1, selected_pixels)
    spectrum_inside_roi = bool(detection_ok and inside_ratio > 0.85 and not clipped_or_touching_edge)

    tilt_angle, tilt_warning = estimate_tilt(mask_roi, score_roi)
    top_bottom_likely, color_detail = top_blue_bottom_red(rgb_roi)

    if clipped_or_touching_edge:
        comment_parts.append("candidate touches image edge or detected stripe reaches image edge")
    if overexposed:
        comment_parts.append(f"possible saturation: roi={saturated_roi_ratio:.4f}, full={saturated_full_ratio:.4f}")
    if underexposed:
        comment_parts.append(f"weak signal: roi_p99_gray={roi_p99_gray:.2f}, roi_p99_score={roi_p99_score:.2f}")
    if tilt_warning:
        comment_parts.append(f"tilt angle from vertical about {tilt_angle:.2f} deg")
    if not top_bottom_likely:
        comment_parts.append(f"top-blue/bottom-red not strongly supported ({color_detail})")

    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    item_id = safe_name(f"{source}_{image_path.stem}_{digest}")
    diagnostic_path = output_dir / f"{item_id}_diagnostic.png"
    crop_path = output_dir / f"{item_id}_roi_crop.png"
    profile_path = output_dir / f"{item_id}_profile.csv"

    row = {
        "source": source,
        "image_path": rel_path(image_path),
        "candidate_roi_x": x,
        "candidate_roi_y": y,
        "candidate_roi_w": w,
        "candidate_roi_h": h,
        "spectrum_inside_roi": fbool(spectrum_inside_roi),
        "clipped_or_touching_edge": fbool(clipped_or_touching_edge),
        "overexposed": fbool(overexposed),
        "underexposed": fbool(underexposed),
        "top_blue_bottom_red_likely": fbool(top_bottom_likely),
        "tilt_warning": fbool(tilt_warning),
        "comment": "; ".join(comment_parts),
        "diagnostic_png": rel_path(diagnostic_path),
        "roi_crop_png": rel_path(crop_path),
        "profile_csv": rel_path(profile_path),
    }

    draw_diagnostic(rgb, roi, row, diagnostic_path)
    write_image_rgb(crop_path, rgb_roi)
    make_profile_csv(profile_path, rgb_roi)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "image_path",
        "candidate_roi_x",
        "candidate_roi_y",
        "candidate_roi_w",
        "candidate_roi_h",
        "spectrum_inside_roi",
        "clipped_or_touching_edge",
        "overexposed",
        "underexposed",
        "top_blue_bottom_red_likely",
        "tilt_warning",
        "comment",
        "diagnostic_png",
        "roi_crop_png",
        "profile_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], input_root: Path, output_dir: Path) -> None:
    counts = Counter(row["source"] for row in rows)
    problems = defaultdict(list)
    ready = []
    for row in rows:
        image_name = Path(row["image_path"]).name
        bad_flags = []
        for key in ("overexposed", "underexposed", "clipped_or_touching_edge", "tilt_warning"):
            if row[key] == "yes":
                bad_flags.append(key)
                problems[key].append(image_name)
        if row["spectrum_inside_roi"] == "yes" and not bad_flags:
            ready.append(row)

    lines = [
        "# New Capture ROI Inspection",
        "",
        "This diagnostic only reads new captured photos and proposes candidate ROIs for manual review.",
        "It does not train, predict, fit wavelength calibration, use old ROI, or use old pixel-to-wavelength formula.",
        "",
        f"- input_root: `{rel_path(input_root)}`",
        f"- output_dir: `{rel_path(output_dir)}`",
        f"- images_processed: {len(rows)}",
        "",
        "## Sources Found",
        "",
    ]
    for source, count in sorted(counts.items()):
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Per-Image Candidate ROI", ""])
    lines.append(
        "| source | image | ROI x,y,w,h | inside | clipped | overexposed | underexposed | blue-top/red-bottom | tilt | comment |"
    )
    lines.append("|---|---|---:|---|---|---|---|---|---|---|")
    for row in rows:
        roi_text = f"{row['candidate_roi_x']},{row['candidate_roi_y']},{row['candidate_roi_w']},{row['candidate_roi_h']}"
        lines.append(
            f"| {row['source']} | `{row['image_path']}` | {roi_text} | "
            f"{row['spectrum_inside_roi']} | {row['clipped_or_touching_edge']} | {row['overexposed']} | "
            f"{row['underexposed']} | {row['top_blue_bottom_red_likely']} | {row['tilt_warning']} | "
            f"{row['comment']} |"
        )

    lines.extend(["", "## Problem Summary", ""])
    if not problems:
        lines.append("- No overexposure, underexposure, clipping, or tilt warnings were detected by the heuristic.")
    else:
        for key in ("overexposed", "underexposed", "clipped_or_touching_edge", "tilt_warning"):
            values = problems.get(key, [])
            lines.append(f"- {key}: {', '.join(values) if values else 'none'}")

    lines.extend(["", "## Likely Ready For Next Relative-Coordinate Calibration Step", ""])
    if ready:
        for row in ready:
            lines.append(f"- {row['source']}: `{row['image_path']}`")
    else:
        lines.append("- None passed all automatic checks. Review diagnostic PNGs manually before proceeding.")

    lines.extend(["", "## Output Files", ""])
    lines.append(f"- roi_candidates.csv: `{rel_path(output_dir / 'roi_candidates.csv')}`")
    for row in rows:
        lines.append(f"- {Path(row['image_path']).name}: `{row['diagnostic_png']}`, `{row['roi_crop_png']}`, `{row['profile_csv']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect new DIY spectrum photos and propose candidate ROIs.")
    parser.add_argument("--input_root", default=str(DEFAULT_INPUT_ROOT), help="Root directory to recursively scan.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = PROJECT_ROOT / input_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = scan_images(input_root)
    if not images:
        raise RuntimeError(f"No full capture images found under {input_root}")

    rows = []
    for image_path in images:
        try:
            rows.append(analyze_image(image_path, output_dir))
        except Exception as exc:
            metadata = load_metadata(image_path)
            source = infer_source(image_path, metadata)
            rows.append(
                {
                    "source": source,
                    "image_path": rel_path(image_path),
                    "candidate_roi_x": "",
                    "candidate_roi_y": "",
                    "candidate_roi_w": "",
                    "candidate_roi_h": "",
                    "spectrum_inside_roi": "no",
                    "clipped_or_touching_edge": "no",
                    "overexposed": "no",
                    "underexposed": "no",
                    "top_blue_bottom_red_likely": "no",
                    "tilt_warning": "no",
                    "comment": f"analysis failed: {exc}",
                    "diagnostic_png": "",
                    "roi_crop_png": "",
                    "profile_csv": "",
                }
            )

    csv_path = output_dir / "roi_candidates.csv"
    report_path = output_dir / "roi_inspection_report.md"
    write_csv(csv_path, rows)
    write_report(report_path, rows, input_root, output_dir)

    counts = Counter(row["source"] for row in rows)
    print("Sources found:")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")
    print(f"Processed images: {len(rows)}")
    print(f"CSV: {rel_path(csv_path)}")
    print(f"Report: {rel_path(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
