from app.config import RISK_NOTICE


def build_logs(filename: str, roi_mode: str, model_path: str, calibration_version: str) -> list[str]:
    return [
        f"已接收图片：{filename}",
        f"ROI 模式：{roi_mode}",
        "已生成 R/G/B/Gray 四通道 profile",
        f"定标版本：{calibration_version}",
        f"模型路径：{model_path}",
        "推理完成，已计算主峰波长和峰值强度",
        f"风险提示：{RISK_NOTICE}",
    ]
