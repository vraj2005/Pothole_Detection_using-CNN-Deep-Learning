# Pothole Detection - Backend Run Guide

This guide matches the new backend structure and explains full start-to-end usage.

## 1) New project structure

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
```

## 2) Dataset rules

Use this exact layout inside `backend/my_dataset`.

Rules:
- pothole images go in `Pothole`
- normal road images go in `Plain`
- supported formats: `.jpg`, `.jpeg`, `.png`

## 3) Environment setup (from scratch)

Run from project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy opencv-python scikit-learn imutils
```

Install TensorFlow only if you want TensorFlow scripts:

```powershell
pip install tensorflow keras
```

Install CUDA PyTorch for GPU path:

```powershell
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 4) Training options

### A) TensorFlow legacy training (100x100)

```powershell
python backend/tensorflow/legacy/main.py
```

Optional epochs:

```powershell
$env:EPOCHS='100'
python backend/tensorflow/legacy/main.py
```

Outputs are saved in `backend/tensorflow/legacy`.

### B) TensorFlow realtime training (300x300)

```powershell
python backend/tensorflow/realtime/main.py
```

Outputs are saved in `backend/tensorflow/realtime`.

### C) PyTorch GPU training (recommended on your setup)

```powershell
python backend/pytorch/train_gpu_pytorch.py
```

Optional longer run:

```powershell
$env:EPOCHS='100'
python backend/pytorch/train_gpu_pytorch.py
```

Best model is saved as `backend/pytorch/pytorch_pothole_gpu_model.pth`.

## 5) Model checking (your 2 required ways)

### A) Live mobile camera / webcam detection

```powershell
python backend/pytorch/pytorch_realtime_video_predictor.py --source webcam --camera-id 0 --device auto --threshold 0.85
```

Notes:
- try `--camera-id 1` or `--camera-id 2` for DroidCam/IP camera
- press `q` to close

### B) Video file pothole detection

```powershell
python backend/pytorch/pytorch_realtime_video_predictor.py --source video --video-path "path\to\road_video.mp4" --output-path "outputs\road_video_pred.mp4" --device auto --threshold 0.85 --show
```

You get:
- live preview (`--show`)
- saved output video in `outputs/`
- console summary with processed frames and pothole events

## 6) TensorFlow prediction scripts

Legacy model test:

```powershell
python backend/tensorflow/legacy/Predictor.py
```

Realtime-branch test:

```powershell
python backend/tensorflow/realtime/Predictor.py
```

TensorFlow webcam script:

```powershell
python backend/tensorflow/realtime/realtimePredictor.py
```

## 7) Common errors

Error: `No TensorFlow-visible GPU found`
- TensorFlow GPU is not available in this environment
- use the PyTorch path for local Windows GPU

Error: `No images found` or `No readable images found`
- check folder names and class names under `backend/my_dataset`
- verify image formats and file integrity

## 8) Best quick flow for your friend

1. setup venv and packages
2. put dataset in `backend/my_dataset`
3. train with `python backend/pytorch/train_gpu_pytorch.py`
4. run camera test with `python backend/pytorch/pytorch_realtime_video_predictor.py --source webcam --camera-id 0`
5. run video test with `--source video --video-path ...`
