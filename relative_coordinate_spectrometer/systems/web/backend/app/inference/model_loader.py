import os
import sys
from pathlib import Path
from typing import Any

from app import config

try:
    import torch
except ModuleNotFoundError:
    torch = None


if str(config.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PROJECT_ROOT))


class ModelRegistry:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.device = "unavailable"
        self.model_path = ""
        self.model_source = ""
        self.error = ""
        self.checkpoint: dict[str, Any] | None = None
        self._loaded = False

    def candidate_status(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        env_path = os.getenv("SPECTRUM_MODEL_PATH", "").strip()
        if env_path:
            path = Path(env_path)
            if not path.is_absolute():
                path = config.PROJECT_ROOT / path
            items.append({"label": "环境变量 SPECTRUM_MODEL_PATH", "path": config.rel_path(path), "exists": path.exists()})
        items.append(
            {
                "label": "final relative-coordinate model",
                "path": config.rel_path(config.MODEL_PATH),
                "exists": config.MODEL_PATH.exists(),
            }
        )
        return items

    def _resolve_model_path(self) -> tuple[Path | None, str]:
        env_path = os.getenv("SPECTRUM_MODEL_PATH", "").strip()
        if env_path:
            path = Path(env_path)
            if not path.is_absolute():
                path = config.PROJECT_ROOT / path
            if path.exists():
                return path, "环境变量 SPECTRUM_MODEL_PATH"

        if config.MODEL_PATH.exists():
            return config.MODEL_PATH, "final relative-coordinate model"
        return None, ""

    def ensure_loaded(self) -> bool:
        if self._loaded:
            return self.model is not None
        self._loaded = True

        if torch is None:
            self.error = "缺少 PyTorch 依赖，无法加载模型。"
            return False

        path, source = self._resolve_model_path()
        if path is None:
            self.error = "未找到最后一次 8万训练模型 checkpoint。"
            return False

        try:
            from shared.spectrum_unet_transformer_1d import SpectrumUNetTransformer1D
        except Exception as exc:
            self.error = f"模型结构导入失败：{exc}"
            return False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=self.device)

            model_cfg = checkpoint.get("config", {}).get("model", {}) if isinstance(checkpoint, dict) else {}
            model = SpectrumUNetTransformer1D(
                in_channels=int(model_cfg.get("in_channels", 4)),
                out_length=int(model_cfg.get("out_length", config.TARGET_POINT_COUNT)),
                base_channels=int(model_cfg.get("base_channels", 32)),
                trans_heads=int(model_cfg.get("trans_heads", 4)),
                trans_layers=int(model_cfg.get("trans_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                output_activation=model_cfg.get("output_activation", "none"),
            ).to(self.device)

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=True)
            model.eval()
        except Exception as exc:
            self.error = f"模型加载失败：{exc}"
            return False

        self.model = model
        self.checkpoint = checkpoint if isinstance(checkpoint, dict) else None
        self.model_path = config.rel_path(path)
        self.model_source = source
        self.error = ""
        return True


registry = ModelRegistry()
