import json

import numpy as np

from app import config


def make_target_wavelength_axis() -> np.ndarray:
    axis = config.TARGET_START_NM + np.arange(config.TARGET_POINT_COUNT, dtype=np.float64) * config.TARGET_STEP_NM
    axis[-1] = config.TARGET_END_NM
    return np.round(axis, 6).astype(np.float32)


def load_calibration() -> dict[str, object]:
    path = config.CALIBRATION_PATH
    if not path.exists():
        raise FileNotFoundError(f"Relative-coordinate calibration config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("calibration_type") != "relative_spectral_coordinate_linear_diagnostic":
        raise ValueError(f"Unsupported calibration type: {data.get('calibration_type')}")
    return {
        "label": "relative_calibration_linear_diagnostic_v1",
        "path": path,
        "data": data,
        "version": str(data["calibration_type"]),
    }


def calibration_formula(calibration: dict[str, object]) -> str:
    data = calibration["data"]
    if isinstance(data, dict):
        return str(data.get("formula") or "wavelength_nm = 223.039214714 * s + 411.1831404")
    return "wavelength_nm = 223.039214714 * s + 411.1831404"


def source_wavelength_for_roi(
    roi_height: int,
    calibration: dict[str, object],
    roi: dict[str, int] | None = None,
) -> np.ndarray:
    data = calibration["data"]
    pixels = np.arange(roi_height, dtype=np.float64)
    if not isinstance(data, dict) or "relative_s_definition" not in data:
        raise ValueError("Relative-coordinate anchors are missing from the calibration config.")
    s_cfg = data["relative_s_definition"]
    roi_cfg = data.get("roi_source") or {}
    y_short = float(s_cfg["y_short_anchor"])
    y_long = float(s_cfg["y_long_anchor"])
    if abs(y_long - y_short) < 1e-9:
        raise ValueError("Relative-coordinate anchors must be different.")
    roi_y = float((roi or {}).get("y", 0))
    calibration_roi_y = float(roi_cfg.get("y", 0))
    y_local = pixels + roi_y - calibration_roi_y
    s = (y_local - y_short) / (y_long - y_short)
    return (float(data["coefficient_a"]) * s + float(data["coefficient_b"])).astype(np.float64)
