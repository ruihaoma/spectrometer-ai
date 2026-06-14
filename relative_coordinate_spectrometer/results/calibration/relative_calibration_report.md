# Relative Spectral Coordinate Calibration Diagnostic

This is a diagnostic calibration, not a final high-confidence calibration.
No old ROI and no old pixel-to-wavelength formula are used.
Known physical wavelengths may appear only as possible hints; selected anchor wavelengths come from `standard_peaks.csv`.
Target wavelength interval: 400-650 nm.

- reference_root: `data/raw/reference_spectrometer`
- image_root: `data/raw/calibration`

## Reference Spectrum Files

- hg: 1
  - `data/raw/reference_spectrometer/hg/hg.txt`
- na: 1
  - `data/raw/reference_spectrometer/na/na.txt`
- hene: 1
  - `data/raw/reference_spectrometer/hene/hene.txt`

## Standard Peaks

- detected standard peaks: 25
- hg: 404.800000 nm from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 408.000000 nm from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 410.700000 nm from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 411.000000 nm from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 434.300000 nm (near 435.8 nm possible_hint_only) from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 435.000000 nm (near 435.8 nm possible_hint_only) from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 435.900000 nm (near 435.8 nm possible_hint_only) from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 439.500000 nm from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 491.600000 nm from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 546.000000 nm (near 546.1 nm possible_hint_only) from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 576.800000 nm (near 577.0 nm possible_hint_only) from `data/raw/reference_spectrometer/hg/hg.txt`
- hg: 578.800000 nm (near 577.0 nm possible_hint_only) from `data/raw/reference_spectrometer/hg/hg.txt`
- na: 568.200000 nm from `data/raw/reference_spectrometer/na/na.txt`
- na: 568.700000 nm from `data/raw/reference_spectrometer/na/na.txt`
- na: 569.100000 nm from `data/raw/reference_spectrometer/na/na.txt`
- na: 584.000000 nm from `data/raw/reference_spectrometer/na/na.txt`
- na: 584.400000 nm from `data/raw/reference_spectrometer/na/na.txt`
- na: 584.700000 nm from `data/raw/reference_spectrometer/na/na.txt`
- na: 585.100000 nm from `data/raw/reference_spectrometer/na/na.txt`
- na: 587.400000 nm (near 589.3 nm possible_hint_only) from `data/raw/reference_spectrometer/na/na.txt`
- na: 589.500000 nm (near 589.3 nm possible_hint_only) from `data/raw/reference_spectrometer/na/na.txt`
- na: 591.100000 nm (near 589.3 nm possible_hint_only) from `data/raw/reference_spectrometer/na/na.txt`
- na: 591.500000 nm (near 589.3 nm possible_hint_only) from `data/raw/reference_spectrometer/na/na.txt`
- na: 591.900000 nm (near 589.3 nm possible_hint_only) from `data/raw/reference_spectrometer/na/na.txt`
- hene: 631.600000 nm (near 632.8 nm possible_hint_only) from `data/raw/reference_spectrometer/hene/hene.txt`

## Image Peak Candidates

- hg: 36 candidates
- na: 36 candidates
- hene: 22 candidates

## White Reference ROI For Image Profiles

- white image: `data/raw/calibration/white_led/white_led_001_full.png`
- ROI from current white full image: x=0, y=0, w=915, h=1080
- this is a temporary profile extraction range, not a final calibration ROI

## Automatic Anchor Selection

- selected anchors: 4
- model_type: linear
- formula: `wavelength_nm = 223.039214714*s + 411.1831404`
- max residual: 4.192300 nm
- mean abs residual: 2.902748 nm
- RMSE: 3.049053 nm
- wavelength coverage: 223.600000 nm
- max anchor gap: 110.100000 nm
- recommended_for_next_step: true
- quality warnings:
  - quadratic: model is not monotonic nondecreasing on s=0..1

### Selected Anchors

| source | wavelength_nm | y_local | s | channel | residual_nm | reason |
|---|---:|---:|---:|---|---:|---|
| hg | 408.000000 | 286.000 | 0.000000 | B | 3.183140 | strong standard peak + clear image candidate + monotonic coverage |
| hg | 435.900000 | 312.000 | 0.103586 | B | -1.613196 | strong standard peak + clear image candidate + monotonic coverage |
| hg | 546.000000 | 433.000 | 0.585657 | G | -4.192300 | strong standard peak + clear image candidate + monotonic coverage |
| hene | 631.600000 | 537.000 | 1.000000 | R | 2.622355 | strong standard peak + clear image candidate + monotonic coverage |

### Exclusion Notes

- weak standard-spectrum peaks were not used as final anchors
- image peaks near ROI edges or with low prominence were penalized or excluded
- duplicate or nearly identical image y positions were not allowed in the same fit
- non-monotonic wavelength-to-y combinations were rejected

## Linear vs Quadratic Model Check

| model | max residual nm | mean abs residual nm | RMSE nm | monotonic | recommended |
|---|---:|---:|---:|---|---|
| quadratic diagnostic | 0.443517 | 0.268817 | 0.307521 | false | false |
| linear | 4.192300 | 2.902748 | 3.049053 | true | true |
- final_recommended_model: `linear`
- quadratic is not recommended because it is not monotonic on s=0..1

## Output Files

- `results/calibration/standard_peaks.csv`
- `results/calibration/image_peak_candidates.csv`
- `results/calibration/anchor_match_review.csv`
- `results/calibration/selected_anchors.csv`
- `results/calibration/relative_calibration_fit_diagnostic.json`
- `results/calibration/relative_calibration_fit_linear.json`
- `results/calibration/relative_calibration_residuals.csv`
- `results/calibration/relative_calibration_curve.png`
- `results/calibration/relative_calibration_curve_compare.png`
- `results/calibration/anchor_match_review.png`
