import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    print("Missing dependency: opencv-python. Install it with: python -m pip install opencv-python", file=sys.stderr)
    raise SystemExit(1)


SCRIPT_NAME = "capture_spectrum_images.py"
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

DEFAULT_OUTPUT_DIR = "data/raw/calibration"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_FOURCC = "MJPG"
DEFAULT_PREVIEW_WIDTH = 960
DEFAULT_PREVIEW_HEIGHT = 540
DEFAULT_ROI = {"x": 730, "y": 314, "w": 262, "h": 270}
KNOWN_SOURCES = [
    "hg",
    "na",
    "hene",
    "blue_led",
    "green_led",
    "red_led",
    "purple_led",
    "yellow_led",
    "white_led",
    "dark",
    "unknown",
]
CAPTURE_SESSION_PREFIX = "capture"
NOTE = "new capture for relative spectral coordinate calibration"


def project_path(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_source_name(text):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip().lower()).strip("_")
    return value or "unknown"


def safe_file_stem(text):
    value = str(text).strip()
    if not value:
        return None
    value = Path(value).stem
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or None


def fourcc_to_text(value):
    value = int(value)
    chars = []
    for shift in (0, 8, 16, 24):
        char_code = (value >> shift) & 0xFF
        if 32 <= char_code <= 126:
            chars.append(chr(char_code))
    return "".join(chars) if chars else str(value)


def parse_roi(values):
    if values is None:
        return dict(DEFAULT_ROI)
    x, y, w, h = [int(item) for item in values]
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI width/height must be positive, got w={w}, h={h}")
    return {"x": x, "y": y, "w": w, "h": h}


def backend_value(use_dshow):
    return cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY


def open_camera(camera_index, use_dshow):
    return cv2.VideoCapture(int(camera_index), backend_value(use_dshow))


def apply_requested_camera_settings(cap, width, height, fps, fourcc):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    if fourcc:
        code = cv2.VideoWriter_fourcc(*str(fourcc)[:4])
        cap.set(cv2.CAP_PROP_FOURCC, code)


def camera_properties(cap):
    return {
        "actual_width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "actual_height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "actual_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "actual_fourcc": fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC)),
    }


def list_cameras(max_index, use_dshow, width, height, fps, fourcc):
    print("camera_index, opened, frame_width, frame_height, fps, fourcc")
    found = 0
    for idx in range(int(max_index) + 1):
        cap = open_camera(idx, use_dshow)
        if cap is None or not cap.isOpened():
            print(f"{idx}, no, , , , ")
            if cap is not None:
                cap.release()
            continue
        apply_requested_camera_settings(cap, width, height, fps, fourcc)
        ok, frame = cap.read()
        props = camera_properties(cap)
        if ok and frame is not None:
            found += 1
            print(
                f"{idx}, yes, {props['actual_width']}, {props['actual_height']}, "
                f"{props['actual_fps']:.3g}, {props['actual_fourcc']}"
            )
        else:
            print(f"{idx}, opened_but_no_frame, , , , ")
        cap.release()
    if found == 0:
        print("No camera returned a frame.")


def setup_preview_window(window_name, preview_width, preview_height, preview_left, preview_top, topmost):
    try:
        cv2.startWindowThread()
    except cv2.error:
        pass
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if preview_width > 0 and preview_height > 0:
        cv2.resizeWindow(window_name, int(preview_width), int(preview_height))
    cv2.moveWindow(window_name, int(preview_left), int(preview_top))
    topmost_prop = getattr(cv2, "WND_PROP_TOPMOST", None)
    if topmost and topmost_prop is not None:
        try:
            cv2.setWindowProperty(window_name, topmost_prop, 1)
        except cv2.error:
            pass


def validate_roi_for_frame(frame, roi):
    height, width = frame.shape[:2]
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return f"ROI outside frame: frame_width={width}, frame_height={height}, roi={roi}"
    return None


def roi_crop(frame, roi):
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    return frame[y : y + h, x : x + w].copy()


def imwrite_unicode(path, image):
    """Save an OpenCV image to paths containing Chinese or other non-ASCII text on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True


def prepare_capture_session(output_dir, source):
    source_dir = Path(output_dir) / source
    source_dir.mkdir(parents=True, exist_ok=True)

    session_id = f"{CAPTURE_SESSION_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = source_dir / session_id
    suffix = 1
    while session_dir.exists():
        session_dir = source_dir / f"{session_id}_{suffix:03d}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return source_dir, session_dir, session_dir.name


def unique_stem(capture_dir, base):
    full_path = capture_dir / f"{base}_full.png"
    roi_path = capture_dir / f"{base}_roi.png"
    meta_path = capture_dir / f"{base}_meta.json"
    if not full_path.exists() and not roi_path.exists() and not meta_path.exists():
        return base
    for idx in range(1, 1000):
        candidate = f"{base}_{idx:03d}"
        full_path = capture_dir / f"{candidate}_full.png"
        roi_path = capture_dir / f"{candidate}_roi.png"
        meta_path = capture_dir / f"{candidate}_meta.json"
        if not full_path.exists() and not roi_path.exists() and not meta_path.exists():
            return candidate
    raise FileExistsError(f"Too many captures with the same name under {rel_path(capture_dir)}")


def save_capture(
    frame,
    source,
    camera_index,
    requested,
    actual,
    roi,
    capture_dir,
    capture_session,
    save_trigger,
    save_name=None,
    manual_save_dir=False,
):
    roi_error = validate_roi_for_frame(frame, roi)
    if roi_error is not None:
        raise ValueError(roi_error)

    now = datetime.now()
    timestamp_for_name = now.strftime("%Y%m%d_%H%M%S")
    timestamp_iso = now.isoformat(timespec="seconds")
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    requested_stem = safe_file_stem(save_name)
    base = requested_stem if requested_stem is not None else f"{source}_{timestamp_for_name}"
    stem = unique_stem(capture_dir, base)

    full_path = capture_dir / f"{stem}_full.png"
    roi_path = capture_dir / f"{stem}_roi.png"
    meta_path = capture_dir / f"{stem}_meta.json"

    crop = roi_crop(frame, roi)
    if not imwrite_unicode(full_path, frame):
        raise OSError(f"Failed to save full image: {full_path}")
    if not imwrite_unicode(roi_path, crop):
        raise OSError(f"Failed to save ROI crop: {roi_path}")

    metadata = {
        "source": source,
        "timestamp": timestamp_iso,
        "camera_index": int(camera_index),
        "requested_width": int(requested["width"]),
        "requested_height": int(requested["height"]),
        "requested_fps": float(requested["fps"]),
        "requested_fourcc": str(requested["fourcc"]),
        "actual_width": int(actual["actual_width"]),
        "actual_height": int(actual["actual_height"]),
        "actual_fps": float(actual["actual_fps"]),
        "actual_fourcc": str(actual["actual_fourcc"]),
        "roi": dict(roi),
        "capture_session": capture_session,
        "capture_dir": rel_path(capture_dir),
        "manual_save_dir": bool(manual_save_dir),
        "requested_save_name": save_name,
        "actual_file_stem": stem,
        "save_trigger": save_trigger,
        "full_image_path": rel_path(full_path),
        "roi_crop_path": rel_path(roi_path),
        "metadata_path": rel_path(meta_path),
        "image_write_backend": "cv2.imencode + Path.write_bytes",
        "note": NOTE,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return full_path, roi_path, meta_path


def draw_text_block(frame, lines, origin=(20, 30)):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 1
    line_height = 25
    max_width = 0
    for line in lines:
        (text_w, _text_h), _baseline = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_w)
    block_height = line_height * len(lines) + 12
    cv2.rectangle(frame, (x - 8, y - 22), (x + max_width + 12, y - 22 + block_height), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + idx * line_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_preview(frame, roi, source, camera_index, actual, requested, show_roi):
    preview = frame.copy()
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    height, width = frame.shape[:2]
    if show_roi:
        color = (0, 255, 255) if validate_roi_for_frame(frame, roi) is None else (0, 0, 255)
        cv2.rectangle(preview, (x, y), (min(x + w, width - 1), min(y + h, height - 1)), color, 2)
    lines = [
        f"source={source}  camera={camera_index}",
        f"frame={actual['actual_width']}x{actual['actual_height']}  fps={actual['actual_fps']:.3g}  fourcc={actual['actual_fourcc']}",
        f"requested={requested['width']}x{requested['height']} @ {requested['fps']} fps {requested['fourcc']}",
        "keys: s save full+roi+meta | r save roi review full+roi+meta | q quit",
    ]
    if show_roi:
        lines.append(f"ROI preview x={x} y={y} w={w} h={h}")
    else:
        lines.append("ROI preview hidden; no calibration box is drawn.")
    draw_text_block(preview, lines)
    return preview


def capture_loop(args):
    if args.camera_index is None:
        raise RuntimeError("camera_index is required. Run --list_cameras first, then pass the correct --camera_index.")
    source = safe_source_name(args.source)
    output_dir = project_path(args.output_dir)
    manual_save_dir = project_path(args.save_dir) if args.save_dir else None
    manual_save_name = safe_file_stem(args.save_name) if args.save_name else None
    roi = parse_roi(args.roi)
    requested = {
        "width": int(args.width),
        "height": int(args.height),
        "fps": float(args.fps),
        "fourcc": str(args.fourcc),
    }

    cap = open_camera(args.camera_index, args.use_dshow)
    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. "
            "Try --list_cameras or a different --camera_index."
        )
    apply_requested_camera_settings(cap, args.width, args.height, args.fps, args.fourcc)
    capture_dir = None
    capture_session = None
    window_name = f"DIY spectrum capture - {source}"
    setup_preview_window(
        window_name,
        args.preview_width,
        args.preview_height,
        args.preview_left,
        args.preview_top,
        not args.no_topmost,
    )
    first_frame_logged = False

    print("Camera opened.")
    print("source:", source)
    print("camera_index:", args.camera_index)
    if manual_save_dir is not None:
        print("save_dir:", rel_path(manual_save_dir))
        print("save_name:", manual_save_name if manual_save_name else "auto: source_YYYYMMDD_HHMMSS")
    else:
        print("output_root:", rel_path(output_dir / source))
        print("session_dir: created only after the first save")
    print("Press s to save full image + ROI crop + metadata.")
    print("Press r to save ROI review full image + ROI crop + metadata.")
    print("Press q or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame from camera")
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            actual = camera_properties(cap)
            if not first_frame_logged:
                print(
                    "First frame received:",
                    f"{frame.shape[1]}x{frame.shape[0]}",
                    "preview window:",
                    f"{args.preview_width}x{args.preview_height}",
                )
                first_frame_logged = True
            preview = draw_preview(frame, roi, source, args.camera_index, actual, requested, args.show_roi)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if capture_dir is None:
                    if manual_save_dir is not None:
                        capture_dir = manual_save_dir
                        capture_session = "manual_save_dir"
                    else:
                        _source_dir, capture_dir, capture_session = prepare_capture_session(output_dir, source)
                paths = save_capture(
                    frame,
                    source,
                    args.camera_index,
                    requested,
                    actual,
                    roi,
                    capture_dir,
                    capture_session,
                    "s_full_frame",
                    save_name=manual_save_name,
                    manual_save_dir=manual_save_dir is not None,
                )
                print("saved:", ", ".join(rel_path(path) for path in paths))
            if key == ord("r"):
                if capture_dir is None:
                    if manual_save_dir is not None:
                        capture_dir = manual_save_dir
                        capture_session = "manual_save_dir"
                    else:
                        _source_dir, capture_dir, capture_session = prepare_capture_session(output_dir, source)
                paths = save_capture(
                    frame,
                    source,
                    args.camera_index,
                    requested,
                    actual,
                    roi,
                    capture_dir,
                    capture_session,
                    "r_roi_review",
                    save_name=manual_save_name,
                    manual_save_dir=manual_save_dir is not None,
                )
                print("saved:", ", ".join(rel_path(path) for path in paths))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Capture new DIY color spectrum images for relative spectral calibration.")
    parser.add_argument(
        "--source",
        default="unknown",
        help="Light source name, e.g. hg, na, hene, blue_led, green_led, red_led, purple_led, yellow_led, white_led, dark.",
    )
    parser.add_argument(
        "--camera_index",
        "--camera-index",
        type=int,
        default=None,
        help="Camera index to open. Required unless --list_cameras is used.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for captured images. Each source gets a separate capture session subfolder.",
    )
    parser.add_argument(
        "--save_dir",
        "--save-dir",
        default=None,
        help="Exact directory to save captures. If set, it bypasses the automatic source/session folder.",
    )
    parser.add_argument(
        "--save_name",
        "--save-name",
        default=None,
        help="Base file name to use when saving, without suffix. Files become NAME_full.png, NAME_roi.png, NAME_meta.json.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--fourcc", default=DEFAULT_FOURCC)
    parser.add_argument("--roi", nargs=4, metavar=("X", "Y", "W", "H"), help="Override coarse preview ROI.")
    parser.add_argument("--show_roi", action="store_true", help="Draw the coarse ROI box in the preview window.")
    parser.add_argument("--preview_width", type=int, default=DEFAULT_PREVIEW_WIDTH)
    parser.add_argument("--preview_height", type=int, default=DEFAULT_PREVIEW_HEIGHT)
    parser.add_argument("--preview_left", type=int, default=60)
    parser.add_argument("--preview_top", type=int, default=60)
    parser.add_argument("--no_topmost", action="store_true", help="Do not request the preview window to stay on top.")
    parser.add_argument("--list_cameras", "--list-cameras", action="store_true", help="List camera indices and exit.")
    parser.add_argument("--max_camera_index", type=int, default=8)
    parser.add_argument("--no_dshow", action="store_true", help="Do not force cv2.CAP_DSHOW on Windows.")
    args = parser.parse_args()
    args.use_dshow = not bool(args.no_dshow)
    return args


def main():
    args = parse_args()
    if args.list_cameras:
        list_cameras(args.max_camera_index, args.use_dshow, args.width, args.height, args.fps, args.fourcc)
        return 0
    capture_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
