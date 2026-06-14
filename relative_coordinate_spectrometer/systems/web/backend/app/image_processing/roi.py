import numpy as np


MIN_SIGNAL_PIXELS = 80


def _crop(image_rgb: np.ndarray, roi: dict[str, int]) -> np.ndarray:
    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["w"])
    h = int(roi["h"])
    return image_rgb[y : y + h, x : x + w, :]


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < 3:
        return values.astype(np.float32)
    window = max(3, int(window) | 1)
    if window >= values.size:
        window = max(3, int(values.size // 2) | 1)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _robust_threshold(values: np.ndarray, percentile: float, mad_scale: float) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-6
    return max(float(np.percentile(values, percentile)), median + mad_scale * mad)


def _score_image(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = image_rgb.astype(np.float32)
    max_ch = np.max(rgb, axis=2)
    min_ch = np.min(rgb, axis=2)
    chroma = max_ch - min_ch
    saturation = np.zeros_like(max_ch, dtype=np.float32)
    np.divide(chroma, max_ch, out=saturation, where=max_ch > 1e-6)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    score = 0.50 * gray + 0.25 * max_ch + 0.25 * chroma * (0.5 + 0.5 * saturation)
    return np.clip(score, 0, 255), gray, saturation, chroma


def _spectrum_mask(
    score: np.ndarray,
    gray: np.ndarray,
    saturation: np.ndarray,
    chroma: np.ndarray,
) -> np.ndarray:
    threshold = max(_robust_threshold(score, 98.0, 3.5), 8.0)
    chroma_threshold = max(_robust_threshold(chroma, 97.0, 3.0), 6.0)
    mask = (score >= threshold) & (chroma >= chroma_threshold) & (saturation >= 0.10)

    if int(mask.sum()) < MIN_SIGNAL_PIXELS:
        threshold = max(_robust_threshold(score, 96.5, 2.8), 5.0)
        chroma_threshold = max(_robust_threshold(chroma, 95.0, 2.5), 4.0)
        mask = (score >= threshold) & (chroma >= chroma_threshold) & (saturation >= 0.07)

    if int(mask.sum()) < MIN_SIGNAL_PIXELS:
        gray_p99 = float(np.percentile(gray, 99.0))
        raise ValueError(
            "自动 ROI 失败：未检测到足够的彩色谱线信号。"
            f" 请重新拍摄或调整摆放，使红/绿/蓝谱带进入画面；当前 gray_p99={gray_p99:.2f}。"
        )
    return mask


def _segments(active: np.ndarray) -> list[tuple[int, int]]:
    indices = np.where(active)[0]
    if indices.size == 0:
        return []
    starts = [int(indices[0])]
    ends: list[int] = []
    breaks = np.where(np.diff(indices) > 1)[0]
    for idx in breaks:
        ends.append(int(indices[idx]) + 1)
        starts.append(int(indices[idx + 1]))
    ends.append(int(indices[-1]) + 1)
    return list(zip(starts, ends))


def _merge_close_segments(segments: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _centered_expand(start: int, end: int, min_size: int, limit: int) -> tuple[int, int]:
    size = end - start
    if size >= min_size:
        return max(0, start), min(limit, end)
    center = (start + end) // 2
    new_start = max(0, center - min_size // 2)
    new_end = min(limit, new_start + min_size)
    new_start = max(0, new_end - min_size)
    return new_start, new_end


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> dict[str, int]:
    x, y, w, h = bbox
    image_h, image_w = image_shape[:2]
    pad_x = max(12, int(0.16 * w))
    pad_y = max(12, int(0.16 * h))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_w, x + w + pad_x)
    y1 = min(image_h, y + h + pad_y)

    min_w = min(image_w, max(80, int(image_w * 0.05)))
    min_h = min(image_h, max(120, int(image_h * 0.10)))
    x0, x1 = _centered_expand(x0, x1, min_w, image_w)
    y0, y1 = _centered_expand(y0, y1, min_h, image_h)

    return {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}


def _detect_spectrum_roi(image_rgb: np.ndarray) -> tuple[dict[str, int], list[str]]:
    score, gray, saturation, chroma = _score_image(image_rgb)
    mask = _spectrum_mask(score, gray, saturation, chroma)
    weighted = np.where(mask, score, 0.0).astype(np.float32)
    image_h, image_w = mask.shape

    col_score = _smooth_1d(weighted.sum(axis=0), max(5, int(image_w * 0.006)))
    positive_col = col_score[col_score > 0]
    if positive_col.size == 0:
        raise ValueError("自动 ROI 失败：彩色谱线列投影为空。")

    col_threshold = max(float(col_score.max()) * 0.08, float(np.percentile(positive_col, 55)))
    x_segments = _merge_close_segments(_segments(col_score >= col_threshold), max(8, int(image_w * 0.012)))
    candidates = []
    for x0, x1 in x_segments:
        local_mask = mask[:, x0:x1]
        local_weighted = weighted[:, x0:x1]
        pixel_count = int(local_mask.sum())
        if pixel_count < MIN_SIGNAL_PIXELS:
            continue

        row_score = _smooth_1d(local_weighted.sum(axis=1), max(5, int(image_h * 0.006)))
        positive_row = row_score[row_score > 0]
        if positive_row.size == 0:
            continue
        row_threshold = max(float(row_score.max()) * 0.06, float(np.percentile(positive_row, 45)))
        ys = np.where(row_score >= row_threshold)[0]
        if ys.size == 0:
            ys = np.where(local_mask.any(axis=1))[0]
        if ys.size == 0:
            continue

        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        h = y1 - y0
        w = x1 - x0
        if h < max(24, int(image_h * 0.02)) or w < max(12, int(image_w * 0.006)):
            continue

        sat_values = saturation[:, x0:x1][local_mask]
        mean_saturation = float(sat_values.mean()) if sat_values.size else 0.0
        total_score = float(local_weighted.sum())
        candidate_score = total_score * (1.0 + min(h / 180.0, 2.0)) * (1.0 + mean_saturation)
        candidates.append((candidate_score, (x0, y0, w, h), pixel_count, mean_saturation))

    if not candidates:
        raise ValueError("自动 ROI 失败：未找到稳定的彩色谱线候选区域。")

    candidates.sort(key=lambda item: item[0], reverse=True)
    roi = _expand_bbox(candidates[0][1], image_rgb.shape)
    selected_pixels = candidates[0][2]
    selected_saturation = candidates[0][3]
    warnings: list[str] = []
    if selected_pixels < 160:
        warnings.append("自动 ROI 彩色信号像素偏少，请检查 ROI 裁剪图是否覆盖谱带。")
    if selected_saturation < 0.12:
        warnings.append("自动 ROI 色彩饱和度偏低，请检查是否拍到真实彩色谱带。")
    return roi, warnings


def extract_roi(image_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, int], str, list[str]]:
    roi, warnings = _detect_spectrum_roi(image_rgb)
    roi_rgb = _crop(image_rgb, roi)
    if roi_rgb.size == 0:
        raise ValueError("自动 ROI 失败：裁剪结果为空。")
    return roi_rgb, roi, "相对彩色谱线自动 ROI", warnings
