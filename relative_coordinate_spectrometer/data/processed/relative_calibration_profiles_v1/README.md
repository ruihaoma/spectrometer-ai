# relative_calibration_profiles_v1

This directory contains per-image four-channel spectrum profiles extracted from the current relative spectral coordinate calibration workflow.

Current calibration model:

```text
wavelength_nm = 223.039214714 * s + 411.1831404
```

Important notes:

- This is diagnostic relative calibration, not final high-confidence calibration.
- Channel order is `R, G, B, Gray`.
- Wavelength axis is 400-650 nm with 0.1 nm step, 2501 points.
- `hg`, `na`, and `hene` are calibration references and are not automatically used as paired neural-network training samples.
- `blue_led`, `green_led`, `red_led`, `white_led`, `purple_led`, and `yellow_led` are paired dataset samples.
- `dark` is a dark reference and is not included in `x.npy/y.npy`.
- The canonical calibration configuration is `configs/calibration/relative_calibration_linear_diagnostic_v1.json`.
- Only reusable `profile.csv` files are committed here. Optional ROI crops and diagnostic plots can be regenerated and are intentionally not duplicated in this directory.

The paired dataset generated from these profiles is:

```text
data/generated/relative_calib_paired_dataset_v1/
```
