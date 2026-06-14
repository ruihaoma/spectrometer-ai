import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
WEB_DIR = BACKEND_DIR.parent
SYSTEMS_DIR = WEB_DIR.parent
PROJECT_ROOT = SYSTEMS_DIR.parent

API_PREFIX = "/api"

MODEL_NAME = "spectrum_unet_transformer_1d"
MODEL_CLASS_NAME = "SpectrumUNetTransformer1D"
INPUT_SHAPE = [1, 4, 2501]
OUTPUT_SHAPE = [1, 2501]

TARGET_START_NM = 400.0
TARGET_END_NM = 650.0
TARGET_STEP_NM = 0.1
TARGET_POINT_COUNT = 2501
CHANNELS = ["R", "G", "B", "Gray"]

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = int(os.getenv("SPECTRUM_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

_allowed_origins = os.getenv("SPECTRUM_ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [origin.strip() for origin in _allowed_origins.split(",") if origin.strip()]
    if _allowed_origins
    else ["http://127.0.0.1:5173", "http://localhost:5173"]
)

MODEL_PATH = PROJECT_ROOT / "results" / "final_model" / "best_model.pt"
CALIBRATION_PATH = PROJECT_ROOT / "configs" / "calibration" / "relative_calibration_linear_diagnostic_v1.json"

RISK_NOTICE = "当前定标为候选/诊断定标，预测结果需要结合 ROI、profile 和实验条件判断。"


def rel_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
