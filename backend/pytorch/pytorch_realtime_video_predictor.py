from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


SIZE = 300
CLASS_NAMES = ["Plain", "Pothole"]


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


def draw_overlay(frame_bgr: np.ndarray, pred: FramePrediction, frame_idx: int, fps: float | None) -> np.ndarray:
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
    return out


def run_webcam(
    model: nn.Module,
    device: torch.device,
    camera_id: int,
    threshold: float,
    window_name: str,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam camera_id={camera_id}")

    print("Webcam started. Press q to quit.")
    frame_idx = 0
    pothole_hits = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame_idx += 1
        pred = predict_frame(frame, model, device, threshold)
        if pred.pothole_detected:
            pothole_hits += 1

        overlay = draw_overlay(frame, pred, frame_idx, cap.get(cv2.CAP_PROP_FPS))
        cv2.imshow(window_name, overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Webcam session finished. Frames processed: {frame_idx}, pothole-positive frames: {pothole_hits}")


def run_video_file(
    model: nn.Module,
    device: torch.device,
    video_path: Path,
    output_path: Path | None,
    threshold: float,
    show_window: bool,
    window_name: str,
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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        pred = predict_frame(frame, model, device, threshold)
        if pred.pothole_detected:
            pothole_frames += 1
            if not in_event:
                pothole_events += 1
                in_event = True
        else:
            in_event = False

        overlay = draw_overlay(frame, pred, frame_idx, fps)

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
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for pothole label")
    parser.add_argument("--show", action="store_true", help="Show output window while processing video file")
    parser.add_argument("--window-name", default="Pothole Detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Running on device: {device}")

    model = load_model(args.model_path, device)

    if args.source == "webcam":
        run_webcam(model, device, args.camera_id, args.threshold, args.window_name)
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
    )


if __name__ == "__main__":
    main()