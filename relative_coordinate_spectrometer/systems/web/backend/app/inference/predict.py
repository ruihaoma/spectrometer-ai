import math

import numpy as np

from app import config
from app.inference.model_loader import registry, torch


def run_model_prediction(x_input: np.ndarray) -> tuple[np.ndarray, bool, list[str]]:
    if not registry.ensure_loaded() or registry.model is None or torch is None:
        raise FileNotFoundError(registry.error or "模型未加载。")

    warnings: list[str] = []
    x_tensor = torch.from_numpy(x_input).to(device=registry.device, dtype=torch.float32)
    with torch.inference_mode():
        output = registry.model(x_tensor)

    spectrum_pred = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if spectrum_pred.shape != (config.TARGET_POINT_COUNT,):
        raise ValueError(f"模型输出 shape 应为 [2501]，实际为 {tuple(spectrum_pred.shape)}")

    display_clamped = False
    if not np.isfinite(spectrum_pred).all():
        warnings.append("模型输出包含 NaN/Inf，已替换为 0。")
        spectrum_pred = np.nan_to_num(spectrum_pred, nan=0.0, posinf=0.0, neginf=0.0)

    if float(np.min(spectrum_pred)) < 0.0 or float(np.max(spectrum_pred)) > 1.0:
        display_clamped = True
        spectrum_pred = np.clip(spectrum_pred, 0.0, 1.0)

    return spectrum_pred.astype(np.float32), display_clamped, warnings


def summarize_prediction(wavelength_nm: list[float], spectrum_pred: np.ndarray) -> tuple[float, float]:
    peak_index = int(np.argmax(spectrum_pred))
    peak_wavelength_nm = float(wavelength_nm[peak_index])
    peak_intensity = float(spectrum_pred[peak_index])
    if not math.isfinite(peak_wavelength_nm) or not math.isfinite(peak_intensity):
        raise ValueError("预测摘要包含无效数值。")
    return peak_wavelength_nm, peak_intensity
