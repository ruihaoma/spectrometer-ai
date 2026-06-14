import base64
from io import BytesIO

import numpy as np
from PIL import Image


def image_rgb_to_base64_png(image_rgb: np.ndarray) -> str:
    image_uint8 = np.clip(image_rgb, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image_uint8, mode="RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
