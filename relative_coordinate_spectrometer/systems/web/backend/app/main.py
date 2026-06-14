from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.image_processing.calibration import calibration_formula, load_calibration
from app.inference.model_loader import registry
from app.inference.predict import run_model_prediction, summarize_prediction
from app.inference.preprocess import preprocess_image
from app.schemas import (
    ExperimentInfo,
    HealthResponse,
    ImageInfo,
    ModelInfo,
    PredictResponse,
    PredictionPayload,
    ProcessingInfo,
)
from app.utils.encoding import image_rgb_to_base64_png
from app.utils.image_io import read_upload_image
from app.utils.report import build_logs

app = FastAPI(title="智能光谱重建预测系统 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _health_payload() -> HealthResponse:
    calibration = load_calibration()
    model_loaded = registry.ensure_loaded()
    warnings = []
    if not model_loaded:
        warnings.append(registry.error or "模型未加载。")
    if calibration["version"] != "relative_spectral_coordinate_linear_diagnostic":
        warnings.append("未使用最后一版相对光谱坐标定标 relative_calibration_linear_diagnostic_v1。")

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device=registry.device,
        model_path=registry.model_path,
        model_source=registry.model_source,
        calibration_version=str(calibration["version"]),
        calibration_path=config.rel_path(calibration["path"]) if calibration["path"] else "",
        candidate_model_paths=registry.candidate_status(),
        warnings=warnings,
    )


@app.get(f"{config.API_PREFIX}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _health_payload()


@app.post(f"{config.API_PREFIX}/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    source_type: str = Form("未知"),
    exposure_status: str = Form("自动"),
    note: str = Form(""),
) -> PredictResponse:
    try:
        content = await file.read()
        if not content:
            raise ValueError("Uploaded image is empty.")
        if len(content) > config.MAX_UPLOAD_BYTES:
            limit_mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
            raise ValueError(f"Uploaded image exceeds the {limit_mb:g} MB limit.")
        image_rgb, original_info = read_upload_image(file.filename or "uploaded_image", content)
        preprocessed = preprocess_image(image_rgb)
        spectrum_pred, display_clamped, prediction_warnings = run_model_prediction(preprocessed["x_input"])
        peak_wavelength_nm, peak_intensity = summarize_prediction(preprocessed["wavelength_nm"], spectrum_pred)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"预测失败：{exc}") from exc

    calibration = preprocessed["calibration"]
    model_path = registry.model_path
    calibration_path = config.rel_path(calibration["path"]) if calibration["path"] else ""
    warnings = preprocessed["warnings"] + prediction_warnings + [config.RISK_NOTICE]
    logs = build_logs(
        str(original_info["filename"]),
        str(preprocessed["roi_mode"]),
        model_path,
        str(calibration["version"]),
    )

    return PredictResponse(
        status="success",
        message="预测成功",
        original_image_info=ImageInfo(**original_info),
        roi_image_base64=image_rgb_to_base64_png(preprocessed["roi_rgb"]),
        wavelength_nm=preprocessed["wavelength_nm"],
        profile=preprocessed["profile"],
        prediction=PredictionPayload(
            spectrum_pred=spectrum_pred.round(6).astype(float).tolist(),
            peak_wavelength_nm=round(peak_wavelength_nm, 6),
            peak_intensity=round(peak_intensity, 6),
        ),
        model_info=ModelInfo(
            model_name=config.MODEL_NAME,
            model_path=model_path,
            model_source=registry.model_source,
            input_shape=config.INPUT_SHAPE,
            output_shape=config.OUTPUT_SHAPE,
            device=registry.device,
        ),
        processing_info=ProcessingInfo(
            roi_mode=str(preprocessed["roi_mode"]),
            roi=preprocessed["roi"],
            calibration_version=str(calibration["version"]),
            calibration_path=calibration_path,
            calibration_formula=calibration_formula(calibration),
            wavelength_range="400-650 nm",
            wavelength_step=config.TARGET_STEP_NM,
            point_count=config.TARGET_POINT_COUNT,
            normalization="每样本每通道 x/max(x)",
            display_clamped=display_clamped,
        ),
        experiment_info=ExperimentInfo(source_type=source_type, exposure_status=exposure_status, note=note),
        logs=logs,
        warnings=warnings,
    )
