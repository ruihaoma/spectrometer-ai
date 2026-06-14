from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str
    model_source: str
    calibration_version: str
    calibration_path: str
    candidate_model_paths: list[dict[str, Any]]
    warnings: list[str]


class ImageInfo(BaseModel):
    filename: str
    format: str
    width: int
    height: int


class ModelInfo(BaseModel):
    model_name: str
    model_path: str
    model_source: str
    input_shape: list[int]
    output_shape: list[int]
    device: str


class ProcessingInfo(BaseModel):
    roi_mode: str
    roi: dict[str, int]
    calibration_version: str
    calibration_path: str
    calibration_formula: str
    wavelength_range: str
    wavelength_step: float
    point_count: int
    normalization: str
    display_clamped: bool


class ExperimentInfo(BaseModel):
    source_type: str
    exposure_status: str
    note: str


class PredictionPayload(BaseModel):
    spectrum_pred: list[float]
    peak_wavelength_nm: float
    peak_intensity: float


class PredictResponse(BaseModel):
    status: str
    message: str
    original_image_info: ImageInfo
    roi_image_base64: str
    wavelength_nm: list[float]
    profile: dict[str, list[float]]
    prediction: PredictionPayload
    model_info: ModelInfo
    processing_info: ProcessingInfo
    experiment_info: ExperimentInfo
    logs: list[str]
    warnings: list[str]
