from app.image_processing.calibration import load_calibration
from app.image_processing.profile import build_profiles
from app.image_processing.roi import extract_roi


def preprocess_image(image_rgb):
    calibration = load_calibration()
    roi_rgb, roi, roi_mode, roi_warnings = extract_roi(image_rgb)
    x_input, profile_payload, wavelength_nm, profile_warnings = build_profiles(roi_rgb, calibration, roi)
    return {
        "calibration": calibration,
        "roi_rgb": roi_rgb,
        "roi": roi,
        "roi_mode": roi_mode,
        "x_input": x_input,
        "profile": profile_payload,
        "wavelength_nm": wavelength_nm,
        "warnings": roi_warnings + profile_warnings,
    }
