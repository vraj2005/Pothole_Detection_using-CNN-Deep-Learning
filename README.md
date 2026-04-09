# Pothole Detection System (PyTorch + Realtime Tracking)

This project detects potholes from images, webcam, and video using a CNN-based PyTorch pipeline.

Current status:
- GPU training with PyTorch is the recommended path.
- Realtime inference supports camera selection, video-file processing, and multi-pothole tracking overlays.
- Legacy TensorFlow pipelines are preserved under [backend/tensorflow](backend/tensorflow).

## Repository Structure

```text
backend/
   my_dataset/
      train/
         Pothole/
         Plain/
      test/
         Pothole/
         Plain/
   tensorflow/
      legacy/
         main.py
         Predictor.py
         sample.h5
      realtime/
         main.py
         Predictor.py
         realtimePredictor.py
         full_model.h5
   pytorch/
      train_gpu_pytorch.py
      pytorch_realtime_video_predictor.py
      pytorch_pothole_gpu_model.pth

PROJECT_FULL_DOCUMENTATION.md
run.md
```

## Dataset Layout

Put images only in these folders:
- [backend/my_dataset/train/Pothole](backend/my_dataset/train/Pothole)
- [backend/my_dataset/train/Plain](backend/my_dataset/train/Plain)
- [backend/my_dataset/test/Pothole](backend/my_dataset/test/Pothole)
- [backend/my_dataset/test/Plain](backend/my_dataset/test/Plain)

Image names can be anything. Supported extensions: `.jpg`, `.jpeg`, `.png`.

Current counts:
- `train/Pothole`: 974
- `train/Plain`: 340
- `test/Pothole`: 618
- `test/Plain`: 118
- `Total`: 2050

## Quick Setup

Run from project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy opencv-python scikit-learn imutils
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Train Model (PyTorch)

```powershell
python backend/pytorch/train_gpu_pytorch.py
```

Longer training:

```powershell
$env:EPOCHS='100'
python backend/pytorch/train_gpu_pytorch.py
```

Model output:
- [backend/pytorch/pytorch_pothole_gpu_model.pth](backend/pytorch/pytorch_pothole_gpu_model.pth)

## Realtime Webcam Inference

List cameras:

```powershell
python backend/pytorch/pytorch_realtime_video_predictor.py --list-cameras --max-camera-id 10
```

Interactive camera selection:

```powershell
python backend/pytorch/pytorch_realtime_video_predictor.py --source webcam
```

Fixed camera id:

```powershell
python backend/pytorch/pytorch_realtime_video_predictor.py --source webcam --camera-id 1 --threshold 0.85
```

## Video File Inference

```powershell
python backend/pytorch/pytorch_realtime_video_predictor.py --source video --video-path "D:/path/input.mp4" --output-path "outputs/pothole_pred.mp4" --show --threshold 0.85
```

## Scene Modes

- `road` (default): stable for real road-camera feeds.
- `screen`: more sensitive for monitor/image-grid tests.

Example:

```powershell
python backend/pytorch/pytorch_realtime_video_predictor.py --source webcam --scene-mode screen --threshold 0.70
```

## Overlay Meaning

- `PLAIN` / `POTHOLE`: frame-level status.
- `Conf: XX%`: strongest active pothole confidence.
- `Frame: N`: processed frame number.
- `FPS: Y`: processing frame rate.
- `Pothole #k ZZ%`: tracked pothole id and confidence per box.

## Notes and Limitations

1. Realtime bounding boxes are heuristic + patch classification + tracking.
2. This is not a fully supervised object detector with bbox training labels.
3. Domain shift (road vs monitor photos) can affect results.
4. For highest localization accuracy, migrate to detector training (YOLO/Mask R-CNN).

## Detailed Documentation

For full technical explanation, commands, and architecture details, see:
- [PROJECT_FULL_DOCUMENTATION.md](PROJECT_FULL_DOCUMENTATION.md)
- [run.md](run.md)
