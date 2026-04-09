from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn


SIZE = 300
CLASS_NAMES = ["Plain", "Pothole"]
ROI_TOP_RATIO = 0.28
MAX_CANDIDATES = 14
PATCH_THRESHOLD_OFFSET = 0.07
MIN_TRACK_HITS = 3
TRACK_ACTIVE_MAX_MISSING = 1
MAX_TRACK_MISSING = 8


class PotholeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class FramePrediction:
    class_id: int
    class_name: str
    confidence: float
    pothole_detected: bool


@dataclass
class TrackState:
    bbox: Optional[tuple[int, int, int, int]] = None
    missing_frames: int = 0


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    missing_frames: int = 0
    hits: int = 1
    age: int = 1


def find_pothole_bbox(frame_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Potholes tend to appear as darker irregular regions relative to local road texture.
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape
    frame_area = h * w
    min_area = frame_area * 0.002
    max_area = frame_area * 0.35

    best_bbox = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / max(bh, 1)
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.9:
            # Very circular objects are likely not road potholes.
            continue

        score = float(area)
        if score > best_score:
            best_score = score
            best_bbox = (x, y, bw, bh)

    return best_bbox


def find_candidate_boxes(
    frame_bgr: np.ndarray,
    roi_top_ratio: float,
    max_candidates: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    frame_area = h * w

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Black-hat highlights darker pothole-like regions against brighter road.
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel_bh)

    _, otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = frame_area * min_area_ratio
    max_area = frame_area * max_area_ratio
    boxes: list[tuple[int, int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 18 or bh < 18:
            continue

        # Ignore top area where sky/vehicles/buildings cause false positives.
        center_y = y + bh * 0.5
        if center_y < h * roi_top_ratio:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.30 or aspect > 4.0:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.92:
            continue

        pad_w = int(0.08 * bw)
        pad_h = int(0.08 * bh)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_candidates]


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.35,
) -> list[Detection]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [d for d in ordered if iou(d.bbox, best.bbox) < iou_threshold]

    return kept


def classify_patch_confidence(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    model: nn.Module,
    device: torch.device,
) -> float:
    x, y, w, h = bbox
    patch = frame_bgr[y : y + h, x : x + w]
    if patch.size == 0:
        return 0.0

    tensor = preprocess_frame(patch).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return float(probs[1])


def patch_precheck(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    brightness_limit: float,
    std_limit: float,
) -> bool:
    x, y, w, h = bbox
    patch = frame_bgr[y : y + h, x : x + w]
    if patch.size == 0:
        return False

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Potholes are usually darker than surrounding road and not texture-flat.
    if mean_val > brightness_limit:
        return False
    if std_val < std_limit:
        return False
    return True


def detect_potholes(
    frame_bgr: np.ndarray,
    model: nn.Module,
    device: torch.device,
    threshold: float,
    roi_top_ratio: float,
    max_candidates: int,
    patch_threshold_offset: float,
    min_area_ratio: float,
    max_area_ratio: float,
    precheck_brightness_limit: float,
    precheck_std_limit: float,
) -> list[Detection]:
    candidates = find_candidate_boxes(
        frame_bgr,
        roi_top_ratio=roi_top_ratio,
        max_candidates=max_candidates,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    detections: list[Detection] = []
    patch_threshold = min(0.99, threshold + patch_threshold_offset)
    for bbox in candidates:
        if not patch_precheck(
            frame_bgr,
            bbox,
            brightness_limit=precheck_brightness_limit,
            std_limit=precheck_std_limit,
        ):
            continue
        conf = classify_patch_confidence(frame_bgr, bbox, model, device)
        if conf >= patch_threshold:
            detections.append(Detection(bbox=bbox, confidence=conf))
    return non_max_suppression(detections, iou_threshold=0.35)


def smooth_rect(
    old_bbox: tuple[int, int, int, int],
    new_bbox: tuple[int, int, int, int],
    alpha: float = 0.35,
) -> tuple[int, int, int, int]:
    ox, oy, ow, oh = old_bbox
    nx, ny, nw, nh = new_bbox
    x = int(alpha * nx + (1 - alpha) * ox)
    y = int(alpha * ny + (1 - alpha) * oy)
    w = int(alpha * nw + (1 - alpha) * ow)
    h = int(alpha * nh + (1 - alpha) * oh)
    return (x, y, max(1, w), max(1, h))


def update_tracks(
    tracks: dict[int, Track],
    detections: list[Detection],
    next_track_id: int,
    iou_match_threshold: float = 0.30,
    max_missing: int = MAX_TRACK_MISSING,
) -> tuple[dict[int, Track], int]:
    track_ids = list(tracks.keys())
    unmatched_tracks = set(track_ids)
    unmatched_dets = set(range(len(detections)))

    # Greedy IoU assignment for lightweight multi-object tracking.
    pairs: list[tuple[float, int, int]] = []
    for tid in track_ids:
        for did, det in enumerate(detections):
            score = iou(tracks[tid].bbox, det.bbox)
            if score >= iou_match_threshold:
                pairs.append((score, tid, did))
    pairs.sort(reverse=True, key=lambda x: x[0])

    for _, tid, did in pairs:
        if tid not in unmatched_tracks or did not in unmatched_dets:
            continue
        det = detections[did]
        tr = tracks[tid]
        tr.bbox = smooth_rect(tr.bbox, det.bbox)
        tr.confidence = 0.7 * tr.confidence + 0.3 * det.confidence
        tr.missing_frames = 0
        tr.hits += 1
        tr.age += 1
        unmatched_tracks.remove(tid)
        unmatched_dets.remove(did)

    for tid in list(unmatched_tracks):
        tracks[tid].missing_frames += 1
        tracks[tid].age += 1
        if tracks[tid].missing_frames > max_missing:
            del tracks[tid]

    for did in unmatched_dets:
        det = detections[did]
        tracks[next_track_id] = Track(track_id=next_track_id, bbox=det.bbox, confidence=det.confidence)
        next_track_id += 1

    return tracks, next_track_id


def smooth_bbox(
    old_bbox: Optional[tuple[int, int, int, int]],
    new_bbox: Optional[tuple[int, int, int, int]],
    alpha: float = 0.55,
) -> Optional[tuple[int, int, int, int]]:
    if new_bbox is None:
        return old_bbox
    if old_bbox is None:
        return new_bbox

    ox, oy, ow, oh = old_bbox
    nx, ny, nw, nh = new_bbox

    x = int(alpha * nx + (1 - alpha) * ox)
    y = int(alpha * ny + (1 - alpha) * oy)
    w = int(alpha * nw + (1 - alpha) * ow)
    h = int(alpha * nh + (1 - alpha) * oh)
    return (x, y, max(w, 1), max(h, 1))


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PotholeCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (SIZE, SIZE)).astype("float32") / 255.0
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_frame(
    frame_bgr: np.ndarray,
    model: nn.Module,
    device: torch.device,
    pothole_threshold: float,
) -> FramePrediction:
    input_tensor = preprocess_frame(frame_bgr).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    class_name = CLASS_NAMES[class_id]
    pothole_detected = class_id == 1 and confidence >= pothole_threshold

    return FramePrediction(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        pothole_detected=pothole_detected,
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    pred: FramePrediction,
    frame_idx: int,
    fps: float | None,
    tracks: Optional[Iterable[Track]] = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    color = (0, 0, 255) if pred.pothole_detected else (0, 200, 0)
    status = "POTHOLE" if pred.pothole_detected else pred.class_name.upper()
    prob_text = f"Conf: {pred.confidence * 100:.2f}%"
    frame_text = f"Frame: {frame_idx}"
    fps_text = f"FPS: {fps:.2f}" if fps and fps > 0 else "FPS: N/A"

    cv2.putText(out, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(out, prob_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, frame_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(out, fps_text, (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if tracks:
        for tr in tracks:
            x, y, w, h = tr.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)
            label = f"Pothole #{tr.track_id} {tr.confidence * 100:.1f}%"
            cv2.putText(out, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    return out


def run_webcam(
    model: nn.Module,
    device: torch.device,
    camera_id: int,
    threshold: float,
    window_name: str,
    roi_top_ratio: float,
    max_candidates: int,
    patch_threshold_offset: float,
    min_track_hits: int,
    track_active_max_missing: int,
    min_area_ratio: float,
    max_area_ratio: float,
    precheck_brightness_limit: float,
    precheck_std_limit: float,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam camera_id={camera_id}")

    print("Webcam started. Press q to quit.")
    frame_idx = 0
    pothole_hits = 0
    tracks: dict[int, Track] = {}
    next_track_id = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame_idx += 1
        detections = detect_potholes(
            frame,
            model,
            device,
            threshold,
            roi_top_ratio=roi_top_ratio,
            max_candidates=max_candidates,
            patch_threshold_offset=patch_threshold_offset,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            precheck_brightness_limit=precheck_brightness_limit,
            precheck_std_limit=precheck_std_limit,
        )
        tracks, next_track_id = update_tracks(tracks, detections, next_track_id)

        active_tracks = [
            t
            for t in tracks.values()
            if t.missing_frames <= track_active_max_missing and t.hits >= min_track_hits and t.confidence >= threshold
        ]
        detected = len(active_tracks) > 0
        conf = max((t.confidence for t in active_tracks), default=0.0)
        pred = FramePrediction(
            class_id=1 if detected else 0,
            class_name="Pothole" if detected else "Plain",
            confidence=conf if detected else 1.0 - conf,
            pothole_detected=detected,
        )
        if detected:
            pothole_hits += 1

        overlay = draw_overlay(frame, pred, frame_idx, cap.get(cv2.CAP_PROP_FPS), tracks=active_tracks)
        cv2.imshow(window_name, overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Webcam session finished. Frames processed: {frame_idx}, pothole-positive frames: {pothole_hits}")


def list_cameras(max_id: int = 10) -> list[int]:
    found_ids: list[int] = []
    print(f"Scanning camera IDs 0..{max_id - 1}")
    for camera_id in range(max_id):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            found_ids.append(camera_id)
    if found_ids:
        print(f"Available camera IDs: {found_ids}")
    else:
        print("No working cameras found.")
    return found_ids


def choose_camera_interactively(max_id: int = 10) -> int:
    available = list_cameras(max_id=max_id)
    if not available:
        raise RuntimeError("No camera found. Connect a webcam/mobile camera and retry.")

    print("Enter camera ID from the list above.")
    while True:
        raw = input("Camera ID: ").strip()
        if raw == "":
            chosen = available[0]
            print(f"No input provided. Using default camera ID: {chosen}")
            return chosen
        try:
            chosen = int(raw)
        except ValueError:
            print("Invalid input. Please enter a numeric camera ID.")
            continue
        if chosen not in available:
            print(f"Camera ID {chosen} is not in detected list {available}. Try again.")
            continue
        return chosen


def run_video_file(
    model: nn.Module,
    device: torch.device,
    video_path: Path,
    output_path: Path | None,
    threshold: float,
    show_window: bool,
    window_name: str,
    roi_top_ratio: float,
    max_candidates: int,
    patch_threshold_offset: float,
    min_track_hits: int,
    track_active_max_missing: int,
    min_area_ratio: float,
    max_area_ratio: float,
    precheck_brightness_limit: float,
    precheck_std_limit: float,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 20.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {output_path}")

    print(f"Processing video: {video_path}")
    if total_frames_hint > 0:
        print(f"Estimated total frames: {total_frames_hint}")

    frame_idx = 0
    pothole_frames = 0
    pothole_events = 0
    in_event = False
    tracks: dict[int, Track] = {}
    next_track_id = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        detections = detect_potholes(
            frame,
            model,
            device,
            threshold,
            roi_top_ratio=roi_top_ratio,
            max_candidates=max_candidates,
            patch_threshold_offset=patch_threshold_offset,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            precheck_brightness_limit=precheck_brightness_limit,
            precheck_std_limit=precheck_std_limit,
        )
        tracks, next_track_id = update_tracks(tracks, detections, next_track_id)
        active_tracks = [
            t
            for t in tracks.values()
            if t.missing_frames <= track_active_max_missing and t.hits >= min_track_hits and t.confidence >= threshold
        ]
        detected = len(active_tracks) > 0
        conf = max((t.confidence for t in active_tracks), default=0.0)

        pred = FramePrediction(
            class_id=1 if detected else 0,
            class_name="Pothole" if detected else "Plain",
            confidence=conf if detected else 1.0 - conf,
            pothole_detected=detected,
        )

        if detected:
            pothole_frames += 1
            if not in_event:
                pothole_events += 1
                in_event = True
        else:
            in_event = False

        overlay = draw_overlay(frame, pred, frame_idx, fps, tracks=active_tracks)

        if writer is not None:
            writer.write(overlay)

        if show_window:
            cv2.imshow(window_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Video processing completed.")
    print(f"Frames processed: {frame_idx}")
    print(f"Pothole-positive frames: {pothole_frames}")
    print(f"Detected pothole events: {pothole_events}")
    if output_path is not None:
        print(f"Saved annotated output video to: {output_path}")


def parse_args() -> argparse.Namespace:
    default_model_path = Path(__file__).resolve().parent / "pytorch_pothole_gpu_model.pth"
    parser = argparse.ArgumentParser(
        description="Run pothole detection using the trained PyTorch model on webcam or video file."
    )
    parser.add_argument("--source", choices=["webcam", "video"], default="webcam")
    parser.add_argument("--video-path", type=Path, help="Path to input video (required when --source video)")
    parser.add_argument("--output-path", type=Path, help="Optional path to save annotated video")
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--camera-id", type=int, default=None, help="Camera ID (if omitted, interactive prompt is used)")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera IDs and exit")
    parser.add_argument("--max-camera-id", type=int, default=10, help="Upper bound for camera scan range")
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for pothole label")
    parser.add_argument(
        "--scene-mode",
        choices=["road", "screen"],
        default="road",
        help="road = stable real-road camera, screen = more sensitive for monitor/photo-collage tests",
    )
    parser.add_argument("--min-track-hits", type=int, default=MIN_TRACK_HITS)
    parser.add_argument("--track-active-max-missing", type=int, default=TRACK_ACTIVE_MAX_MISSING)
    parser.add_argument("--patch-threshold-offset", type=float, default=PATCH_THRESHOLD_OFFSET)
    parser.add_argument("--roi-top-ratio", type=float, default=ROI_TOP_RATIO)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--min-area-ratio", type=float, default=0.0015)
    parser.add_argument("--max-area-ratio", type=float, default=0.12)
    parser.add_argument("--precheck-brightness-limit", type=float, default=165.0)
    parser.add_argument("--precheck-std-limit", type=float, default=12.0)
    parser.add_argument("--show", action="store_true", help="Show output window while processing video file")
    parser.add_argument("--window-name", default="Pothole Detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    roi_top_ratio = args.roi_top_ratio
    patch_threshold_offset = args.patch_threshold_offset
    min_track_hits = args.min_track_hits
    track_active_max_missing = args.track_active_max_missing
    max_candidates = args.max_candidates
    min_area_ratio = args.min_area_ratio
    max_area_ratio = args.max_area_ratio
    precheck_brightness_limit = args.precheck_brightness_limit
    precheck_std_limit = args.precheck_std_limit

    if args.scene_mode == "screen":
        # Screen mode relaxes road assumptions for static images/grids shown on monitor.
        roi_top_ratio = 0.0
        patch_threshold_offset = min(patch_threshold_offset, 0.0)
        min_track_hits = min(min_track_hits, 1)
        track_active_max_missing = max(track_active_max_missing, 2)
        max_candidates = max(max_candidates, 24)
        min_area_ratio = min(min_area_ratio, 0.0006)
        max_area_ratio = max(max_area_ratio, 0.25)
        precheck_brightness_limit = max(precheck_brightness_limit, 220.0)
        precheck_std_limit = min(precheck_std_limit, 6.0)

    if args.list_cameras:
        list_cameras(max_id=max(args.max_camera_id, 1))
        return

    device = get_device(args.device)
    print(f"Running on device: {device}")

    model = load_model(args.model_path, device)

    if args.source == "webcam":
        camera_id = args.camera_id
        if camera_id is None:
            camera_id = choose_camera_interactively(max_id=max(args.max_camera_id, 1))
        run_webcam(
            model,
            device,
            camera_id,
            args.threshold,
            args.window_name,
            roi_top_ratio=roi_top_ratio,
            max_candidates=max_candidates,
            patch_threshold_offset=patch_threshold_offset,
            min_track_hits=min_track_hits,
            track_active_max_missing=track_active_max_missing,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            precheck_brightness_limit=precheck_brightness_limit,
            precheck_std_limit=precheck_std_limit,
        )
        return

    if args.video_path is None:
        raise ValueError("--video-path is required when --source video")
    run_video_file(
        model=model,
        device=device,
        video_path=args.video_path,
        output_path=args.output_path,
        threshold=args.threshold,
        show_window=args.show,
        window_name=args.window_name,
        roi_top_ratio=roi_top_ratio,
        max_candidates=max_candidates,
        patch_threshold_offset=patch_threshold_offset,
        min_track_hits=min_track_hits,
        track_active_max_missing=track_active_max_missing,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        precheck_brightness_limit=precheck_brightness_limit,
        precheck_std_limit=precheck_std_limit,
    )


if __name__ == "__main__":
    main()