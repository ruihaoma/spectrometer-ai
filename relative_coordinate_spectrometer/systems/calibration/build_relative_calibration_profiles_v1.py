import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Support direct execution from the project directory.
from systems.calibration import extract_profiles_with_relative_calibration as rpc  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "calibration" / "relative_calibration_linear_diagnostic_v1.json"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "calibration"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "relative_calibration_profiles_v1"


def sample_dir_for(image_path: Path, source: str, output_dir: Path) -> Path:
    return output_dir / source / rpc.safe_name(image_path.stem)


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def build_comment(source: str, shape_comment: str, edge_comment: str, flags: dict) -> str:
    parts = [
        "current diagnostic calibration model; not final calibration",
        "risk: Hg partial line error remains relatively large",
        "risk: current reference captures include dim/tilted/edge-background warnings",
    ]
    flag_notes = []
    if flags["overexposed"]:
        flag_notes.append("overexposed")
    if flags["underexposed"]:
        flag_notes.append("dim/underexposed")
    if flags["bright_signal_touches_roi_edge"]:
        flag_notes.append("bright signal touches ROI edge")
    if flags["roi_touches_image_edge"]:
        flag_notes.append("ROI touches image edge")
    if flags["tilt_warning"]:
        flag_notes.append("tilt warning")
    if flag_notes:
        parts.append("sample risk: " + ", ".join(flag_notes))
    parts.append(shape_comment)
    parts.append(edge_comment)
    return "; ".join(part for part in parts if part)


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source",
        "image_path",
        "profile_csv",
        "roi_crop",
        "diagnostic_png",
        "profile_png",
        "calibration_model",
        "calibration_formula",
        "status",
        "comment",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
        "wavelength_start_nm",
        "wavelength_end_nm",
        "wavelength_step_nm",
        "point_count",
        "normalization",
        "overexposed",
        "underexposed",
        "tilt_warning",
        "bright_signal_touches_roi_edge",
        "roi_touches_image_edge",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, rows: list[dict], config: dict, geometry: dict, input_root: Path, output_dir: Path) -> None:
    counts = Counter(row["source"] for row in rows)
    failures = [row for row in rows if row["status"] != "success"]
    risks = []
    for row in rows:
        flags = []
        for key in ["overexposed", "underexposed", "tilt_warning", "bright_signal_touches_roi_edge", "roi_touches_image_edge"]:
            if row.get(key) == "true":
                flags.append(key)
        if flags:
            risks.append((row, flags))

    ready_for_paired = not failures
    formal_lock_ready = ready_for_paired and not risks

    lines = [
        "# Relative Calibration Profiles V1",
        "",
        "This folder contains per-image 4-channel spectral profiles extracted from the current new full-frame captures.",
        "It does not contain `x.npy`, `y.npy`, labels, model predictions, or training data arrays.",
        "",
        "## Calibration Model",
        "",
        "- calibration_model: `relative_spectral_coordinate_linear_diagnostic_v1`",
        f"- formula: `{config['formula']}`",
        "- status: current diagnostic calibration model, not final formal calibration",
        "- wavelength_axis: 400-650 nm, step 0.1 nm, point_count 2501",
        "- normalization: each sample and each channel divided by that channel maximum",
        f"- ROI source: `{geometry['roi']['source']}`",
        f"- ROI used: x={geometry['roi']['x']}, y={geometry['roi']['y']}, w={geometry['roi']['w']}, h={geometry['roi']['h']}",
        f"- relative s: `(y_local - {geometry['y_short']:.1f}) / ({geometry['y_long']:.1f} - {geometry['y_short']:.1f})`",
        "",
        "Risk note retained for all downstream use:",
        "- Hg partial line error remains relatively large.",
        "- Current reference photos include dim, tilted, and edge/background warnings.",
        "- This model can be used to continue the current diagnostic workflow and generate candidate input profiles, but it is not a high-confidence final calibration.",
        "",
        "## Inputs And Outputs",
        "",
        f"- input_root: `{rpc.rel_path(input_root)}`",
        f"- output_dir: `{rpc.rel_path(output_dir)}`",
        f"- manifest: `{rpc.rel_path(output_dir / 'manifest.csv')}`",
        "",
        "## Sources Processed",
        "",
    ]
    for source, count in sorted(counts.items()):
        lines.append(f"- {source}: {count}")

    lines.extend(["", "## Per-Sample Status", ""])
    lines.append("| source | image | status | profile | risks/comment |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['source']} | `{row['image_path']}` | {row['status']} | "
            f"`{row['profile_csv']}` | {row['comment']} |"
        )

    lines.extend(["", "## Risk Summary", ""])
    if risks:
        for row, flags in risks:
            lines.append(f"- {row['source']} `{row['image_path']}`: {', '.join(flags)}")
    else:
        lines.append("- No automatic exposure, tilt, edge, or ROI-touch warnings were detected.")

    lines.extend(["", "## Paired Dataset Readiness", ""])
    if ready_for_paired:
        lines.append(
            "- The profile files were generated successfully and can be used as candidate inputs for the next paired-dataset construction step."
        )
    else:
        lines.append("- Some profile files failed, so do not build a paired dataset until failures are resolved.")
    if not formal_lock_ready:
        lines.append(
            "- This does not mean the calibration should be formally locked. The diagnostic risks above must remain visible in later reports."
        )
    lines.append("")
    lines.append("No training, prediction, neural-network modification, old ROI, old pixel-to-wavelength formula, or `x.npy`/`y.npy` generation was performed.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def process_one(image_path: Path, config: dict, geometry: dict, target_wavelength: np.ndarray, standard_peaks: dict, output_dir: Path) -> dict:
    source = rpc.infer_source(image_path)
    sample_dir = sample_dir_for(image_path, source, output_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    profile_csv = sample_dir / "profile.csv"
    roi_crop = sample_dir / "roi_crop.png"
    diagnostic_png = sample_dir / "diagnostic.png"
    profile_png = sample_dir / "profile.png"

    rgb = rpc.read_image_rgb(image_path)
    rgb_crop = rpc.crop_roi(rgb, geometry["roi"])
    profiles, extract_meta = rpc.extract_calibrated_profiles(rgb_crop, config, geometry, target_wavelength)
    flags = rpc.quality_flags(rgb, rgb_crop, geometry["roi"])

    rpc.write_profile_csv(profile_csv, target_wavelength, profiles)
    rpc.write_image_rgb(roi_crop, rgb_crop)
    rpc.draw_diagnostic(diagnostic_png, rgb, geometry["roi"], image_path, source, geometry, config, flags)
    rpc.plot_profiles(profile_png, target_wavelength, profiles, source, image_path)

    if source in rpc.REFERENCE_SOURCES:
        shape_comment = rpc.reference_alignment_comment(source, profiles, target_wavelength, standard_peaks)
    elif source in rpc.LED_SOURCES:
        shape_comment = rpc.led_shape_comment(source, profiles, target_wavelength)
    elif source == "dark":
        shape_comment = f"dark source: raw gray_p99={flags['gray_p99']:.2f}"
    else:
        shape_comment = "source not in reference/LED heuristic list"
    edge_comment = rpc.edge_anomaly_comment(profiles, extract_meta["target_extrapolated_fraction"])

    return {
        "source": source,
        "image_path": rpc.rel_path(image_path),
        "profile_csv": rpc.rel_path(profile_csv),
        "roi_crop": rpc.rel_path(roi_crop),
        "diagnostic_png": rpc.rel_path(diagnostic_png),
        "profile_png": rpc.rel_path(profile_png),
        "calibration_model": "relative_spectral_coordinate_linear_diagnostic_v1",
        "calibration_formula": config["formula"],
        "status": "success",
        "comment": build_comment(source, shape_comment, edge_comment, flags),
        "roi_x": geometry["roi"]["x"],
        "roi_y": geometry["roi"]["y"],
        "roi_w": geometry["roi"]["w"],
        "roi_h": geometry["roi"]["h"],
        "wavelength_start_nm": "400.0",
        "wavelength_end_nm": "650.0",
        "wavelength_step_nm": "0.1",
        "point_count": str(target_wavelength.size),
        "normalization": "per-sample per-channel x/max(x)",
        "overexposed": bool_text(flags["overexposed"]),
        "underexposed": bool_text(flags["underexposed"]),
        "tilt_warning": bool_text(flags["tilt_warning"]),
        "bright_signal_touches_roi_edge": bool_text(flags["bright_signal_touches_roi_edge"]),
        "roi_touches_image_edge": bool_text(flags["roi_touches_image_edge"]),
    }


def main() -> int:
    config = rpc.read_json(DEFAULT_CONFIG)
    geometry = rpc.resolve_calibration_geometry(config)
    images = rpc.scan_full_images(DEFAULT_INPUT_ROOT)
    if not images:
        raise RuntimeError(f"No *_full.png images found under {DEFAULT_INPUT_ROOT}")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_wavelength = np.round(np.arange(400.0, 650.0 + 0.0001, 0.1, dtype=np.float64), 1)
    if target_wavelength.shape != (2501,):
        raise RuntimeError(f"Unexpected wavelength point count: {target_wavelength.shape}")

    standard_peaks = rpc.load_standard_peaks(geometry["source_dir"])
    rows = []
    for image_path in images:
        try:
            rows.append(process_one(image_path, config, geometry, target_wavelength, standard_peaks, DEFAULT_OUTPUT_DIR))
        except Exception as exc:
            source = rpc.infer_source(image_path)
            sample_dir = sample_dir_for(image_path, source, DEFAULT_OUTPUT_DIR)
            rows.append(
                {
                    "source": source,
                    "image_path": rpc.rel_path(image_path),
                    "profile_csv": rpc.rel_path(sample_dir / "profile.csv"),
                    "roi_crop": rpc.rel_path(sample_dir / "roi_crop.png"),
                    "diagnostic_png": rpc.rel_path(sample_dir / "diagnostic.png"),
                    "profile_png": rpc.rel_path(sample_dir / "profile.png"),
                    "calibration_model": "relative_spectral_coordinate_linear_diagnostic_v1",
                    "calibration_formula": config["formula"],
                    "status": "failed",
                    "comment": f"failed: {exc}; diagnostic calibration only, not final formal calibration",
                    "roi_x": geometry["roi"]["x"],
                    "roi_y": geometry["roi"]["y"],
                    "roi_w": geometry["roi"]["w"],
                    "roi_h": geometry["roi"]["h"],
                    "wavelength_start_nm": "400.0",
                    "wavelength_end_nm": "650.0",
                    "wavelength_step_nm": "0.1",
                    "point_count": str(target_wavelength.size),
                    "normalization": "per-sample per-channel x/max(x)",
                    "overexposed": "",
                    "underexposed": "",
                    "tilt_warning": "",
                    "bright_signal_touches_roi_edge": "",
                    "roi_touches_image_edge": "",
                }
            )

    write_manifest(DEFAULT_OUTPUT_DIR / "manifest.csv", rows)
    write_readme(DEFAULT_OUTPUT_DIR / "README.md", rows, config, geometry, DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_DIR)
    counts = Counter(row["source"] for row in rows)
    failures = [row for row in rows if row["status"] != "success"]
    print("Relative calibration profiles v1 generated")
    print("  output_dir:", rpc.rel_path(DEFAULT_OUTPUT_DIR))
    print("  images_processed:", len(rows))
    print("  sources:", ", ".join(f"{source}={count}" for source, count in sorted(counts.items())))
    print("  manifest:", rpc.rel_path(DEFAULT_OUTPUT_DIR / "manifest.csv"))
    print("  readme:", rpc.rel_path(DEFAULT_OUTPUT_DIR / "README.md"))
    print("  generated_x_npy_y_npy:", "false")
    if failures:
        print("  failures:")
        for row in failures:
            print("    -", row["image_path"], row["comment"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
