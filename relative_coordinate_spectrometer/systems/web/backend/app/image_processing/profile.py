import numpy as np

from app import config
from app.image_processing.calibration import make_target_wavelength_axis, source_wavelength_for_roi


def _interp_profile(source_wavelength_nm: np.ndarray, values: np.ndarray, target_wavelength_nm: np.ndarray) -> np.ndarray:
    order = np.argsort(source_wavelength_nm)
    source = source_wavelength_nm[order]
    profile = values.astype(np.float64)[order]
    return np.interp(
        target_wavelength_nm.astype(np.float64),
        source,
        profile,
        left=float(profile[0]),
        right=float(profile[-1]),
    ).astype(np.float32)


def _normalize_channels(x_raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    x = x_raw.copy().astype(np.float32)
    for channel_index, channel_name in enumerate(config.CHANNELS):
        max_value = float(np.max(x[channel_index]))
        if max_value <= 0:
            warnings.append(f"{channel_name} 通道最大值为 0，已避免除零并保持原值。")
            continue
        x[channel_index] = x[channel_index] / max_value
    return x, warnings


def build_profiles(
    roi_rgb: np.ndarray,
    calibration: dict[str, object],
    roi: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, list[float]], list[float], list[str]]:
    crop = roi_rgb.astype(np.float32)
    profile_r = crop[:, :, 0].mean(axis=1)
    profile_g = crop[:, :, 1].mean(axis=1)
    profile_b = crop[:, :, 2].mean(axis=1)
    profile_gray = 0.299 * profile_r + 0.587 * profile_g + 0.114 * profile_b

    source_wavelength_nm = source_wavelength_for_roi(crop.shape[0], calibration, roi)
    target_wavelength_nm = make_target_wavelength_axis()
    raw_channels = [profile_r, profile_g, profile_b, profile_gray]
    interpolated = np.stack(
        [_interp_profile(source_wavelength_nm, channel, target_wavelength_nm) for channel in raw_channels],
        axis=0,
    )
    normalized, warnings = _normalize_channels(interpolated)
    profile_payload = {
        channel_name: normalized[index].round(6).astype(float).tolist()
        for index, channel_name in enumerate(config.CHANNELS)
    }
    x_input = normalized.reshape(1, 4, config.TARGET_POINT_COUNT).astype(np.float32)
    return x_input, profile_payload, target_wavelength_nm.round(6).astype(float).tolist(), warnings
