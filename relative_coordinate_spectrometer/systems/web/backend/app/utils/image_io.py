from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import SUPPORTED_IMAGE_SUFFIXES


def validate_image_filename(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"不支持的图片格式：{suffix or '未知'}。支持格式：{allowed}")
    return suffix.replace(".", "")


def read_upload_image(filename: str, content: bytes) -> tuple[np.ndarray, dict[str, object]]:
    image_format = validate_image_filename(filename)
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("无法读取图片，请确认文件是有效图像。") from exc

    width, height = image.size
    image_rgb = np.asarray(image, dtype=np.uint8)
    info = {
        "filename": filename,
        "format": image_format,
        "width": int(width),
        "height": int(height),
    }
    return image_rgb, info
