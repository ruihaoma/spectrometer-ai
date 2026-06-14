# Relative-Coordinate Spectrometer

This project uses **relative spectral-coordinate calibration**. Previous fixed-ROI, direct pixel-to-wavelength, and candidate-calibration experiments are not used by the current pipeline.

## Repository Contents

```text
relative_coordinate_spectrometer/
|-- systems/          Four functional systems
|   |-- capture/          Camera capture code
|   |-- calibration/      Relative-coordinate calibration and profile extraction
|   |-- reconstruction/   Dataset generation, training, and evaluation
|   `-- web/              FastAPI + React inference application
|-- shared/           Shared model, loss, and data-loader code
|-- configs/          Final calibration, dataset, and training settings
|-- data/             Raw measurements and committed intermediate profiles
|-- results/          Calibration evidence and final best model
`-- verification/     Automated reproduction verification
```

The repository includes:

- 12 raw calibration/capture images.
- 9 reference spectrometer text files.
- 12 saved relative-coordinate four-channel profiles.
- Relative calibration anchors, residuals, reports, and diagnostic plots.
- The final `best_model.pt` checkpoint and its metrics.
- All code required to regenerate the omitted datasets and retrain the model.

The repository intentionally does **not** include generated `x.npy`, `y.npy`, or split files.

## Calibration Definition

The current wavelength mapping is:

```text
s = (y_local - 286.0) / (537.0 - 286.0)
wavelength_nm = 223.039214714 * s + 411.1831404
```

The selected anchors are Hg at 408.0 nm and HeNe at 631.6 nm. The linear diagnostic fit has a maximum absolute residual of 4.1923 nm. This calibration is used by the profile pipeline and web application.

## 1. Clone And Install

Git LFS is required because the final PyTorch checkpoint is stored with LFS.

```powershell
git lfs install
git clone https://github.com/ruihaoma/spectrometer-ai.git
cd spectrometer-ai\relative_coordinate_spectrometer
git lfs pull

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recommended resources for the full reproduction:

- At least 12 GB free disk space.
- At least 16 GB system RAM.
- An NVIDIA GPU is strongly recommended for the 350-epoch training run.
- Node.js LTS is required only for the web frontend.

## 2. Verify The Project

Run this before generating any dataset:

```powershell
python verification\verify_project.py
```

Run the complete temporary 6-real + 20-synthetic + 26-mixed smoke pipeline:

```powershell
python verification\verify_project.py --full-smoke
```

The verifier checks:

- Every Git-tracked file exists and is nonempty.
- UTF-8 text, JSON, YAML, CSV, and committed images are readable.
- No generated training dataset arrays are committed.
- Raw-image, reference-spectrum, and saved-profile counts.
- Relative-coordinate anchors and formula.
- Regeneration of the committed white-LED profile from the raw image.
- Final checkpoint SHA256, architecture, training settings, metrics, and CPU inference.

The expected final checkpoint SHA256 is:

```text
dda1be29ae42f424d4ef000138c7e9de83d10cfbd4d968cef7304ea8f8b44ae0
```

## 3. Reproduce Relative Calibration

Re-run the calibration diagnostic from the committed raw images and reference spectra:

```powershell
python systems\calibration\relative_spectral_coordinate_calibration_diagnostic.py
```

Outputs are written to `results/calibration/`. The current linear configuration is stored at:

```text
configs/calibration/relative_calibration_linear_diagnostic_v1.json
```

Inspect new captures and ROI candidates when adding new source images:

```powershell
python systems\calibration\inspect_new_capture_and_roi_candidates.py
```

Regenerate the per-image relative-coordinate profiles:

```powershell
python systems\calibration\build_relative_calibration_profiles_v1.py
```

These profiles are intermediate measurement data, not neural-network datasets.

## 4. Regenerate The Omitted Datasets

All generated datasets are written under `data/generated/`, which is ignored by Git.

Build the six real LED pairs:

```powershell
python systems\reconstruction\build_relative_calib_paired_dataset_v1.py --overwrite
```

For a small smoke test, generate 20 synthetic samples:

```powershell
python systems\reconstruction\generate_relative_calib_synthetic_dataset_v1.py --sample-count 20 --overwrite
```

For the final run, generate exactly 80,000 synthetic samples with seed 42:

```powershell
python systems\reconstruction\generate_relative_calib_synthetic_dataset_v1.py --sample-count 80000 --seed 42 --allow-large --overwrite
```

Combine 6 real and 80,000 synthetic samples:

```powershell
python systems\reconstruction\build_relative_calib_mixed_dataset_v1.py --overwrite
```

Expected shapes:

```text
x.npy             (80006, 4, 2501), float32
y.npy             (80006, 2501), float32
wavelength_nm.npy (2501,)
train / val / test = 64005 / 8001 / 8000
```

## 5. Reproduce Training

The final run used:

- `SpectrumUNetTransformer1D`
- 2,211,073 trainable parameters
- 350 epochs
- batch size 64
- AdamW, learning rate 0.0002, weight decay 0.0001
- `weighted_l1 + 0.1 * grad_l1 + 0.05 * mse`
- `ReduceLROnPlateau`
- random seed 42

Run:

```powershell
python systems\reconstruction\train_spectrum_unet_transformer_1d.py --config configs\train\relative_calib_mixed_v1_80k_train.yaml
```

The reproduction run is written to:

```text
results/reproduction_runs/relative_calib_mixed_v1_80k_train/
```

The recorded training run reached:

```text
best_epoch: 74
best_val_loss: 0.045239608498145144
train_size: 64005
val_size: 8001
```

Exact floating-point values can vary across GPU models, CUDA versions, PyTorch versions, and nondeterministic GPU kernels. The code, seed, dataset generation, architecture, optimizer, scheduler, and loss are fixed in the project configuration.

The original training checkpoint did not record the exact GPU, CUDA, driver, or PyTorch versions. The project supports procedural reproduction and checkpoint verification, but it does not claim bit-for-bit retraining on different hardware. See `results/final_model/checkpoint_metadata.json`.

The original metrics retain their historical dataset and output paths. See `results/final_model/README.md` for the mapping to the current reproduction layout.

## 6. Evaluate

After regenerating the mixed dataset:

```powershell
python systems\reconstruction\evaluate_spectrum_model.py `
  --config configs\train\relative_calib_mixed_v1_80k_train.yaml `
  --checkpoint results\final_model\best_model.pt `
  --splits val test
```

## 7. Run The Web Application

On Windows:

```powershell
.\systems\web\start_system.bat
```

To use different ports:

```powershell
.\systems\web\start_system.bat --backend-port 8011 --frontend-port 5174
```

Stop a custom-port run with:

```powershell
.\systems\web\stop_system.ps1 -Ports 5174,8011
```

Then open:

```text
http://127.0.0.1:5173/
```

Stop it with:

```powershell
.\systems\web\stop_system.ps1
```

The backend loads only:

- `configs/calibration/relative_calibration_linear_diagnostic_v1.json`
- `results/final_model/best_model.pt`

It does not fall back to the previous candidate calibration or old model versions.

GitHub Actions runs the full smoke test, dependency audit, and frontend build on every push and pull request.
