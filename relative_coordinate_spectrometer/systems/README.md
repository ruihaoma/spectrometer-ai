# Functional Systems

The executable code is grouped here by workflow responsibility.

| Directory | Responsibility | Main entry points |
| --- | --- | --- |
| `capture/` | Camera discovery, preview, image capture, ROI crops, and capture metadata | `capture_spectrum_images.py` |
| `calibration/` | Relative-coordinate calibration, anchor diagnostics, ROI inspection, and profile extraction | `relative_spectral_coordinate_calibration_diagnostic.py`, `build_relative_calibration_profiles_v1.py` |
| `reconstruction/` | Paired/synthetic dataset generation, mixed dataset assembly, model training, and evaluation | `build_relative_calib_paired_dataset_v1.py`, `train_spectrum_unet_transformer_1d.py`, `evaluate_spectrum_model.py` |
| `web/` | FastAPI inference backend, React frontend, and local launch/stop scripts | `start_system.bat`, `start_system.py` |

Shared model, loss, and dataset-loader code lives in `../shared/`. Configuration, measurements, and generated evidence remain outside the executable systems under `../configs/`, `../data/`, and `../results/`.

The capture system only collects images and metadata. Calibration, dataset generation, training, evaluation, and prediction remain separate downstream operations.
