# Pothole Detection Project - Start to End Run Guide

This guide explains the full project flow in simple steps so a new person can train and run it from scratch.

## 1) What this project contains

### Root branch (original flow)
- `main.py`: trains TensorFlow/Keras model at 100x100 and saves `sample.h5`.
- `Predictor.py`: loads `sample.h5`, runs test-set prediction.
- `sample.h5`: root trained model (if already available).
- `My Dataset/`: train/test images in `Pothole` and `Plain` classes.

### Real-time Files branch (improved TensorFlow flow)
- `Real-time Files/main.py`: trains improved TensorFlow/Keras model at 300x300.
- `Real-time Files/Predictor.py`: batch test prediction using improved model.
- `Real-time Files/realtimePredictor.py`: webcam inference with key controls.
- `Real-time Files/full_model.h5`: improved pre-trained model.

### Practical GPU training path added for your machine
- `train_gpu_pytorch.py`: PyTorch CUDA trainer (works on this Windows + RTX setup).
- `pytorch_pothole_gpu_model.pth`: best PyTorch checkpoint produced by training.

## 2) Dataset structure (must be exact)

Keep this folder structure exactly:

```text
My Dataset/
  train/
    Pothole/
    Plain/
  test/
    Pothole/
    Plain/
```

Rules:
- Put pothole images only in `Pothole` folders.
- Put non-pothole road images in `Plain` folders.
- Use `.jpg`, `.jpeg`, or `.png`.

## 3) Environment setup from scratch (Windows)

Open terminal in project root and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install base packages:

```powershell
pip install numpy opencv-python scikit-learn imutils
```

## 4) Choose your training path

You have 2 valid paths.

### Path A: Original TensorFlow training (`main.py`)

Use this if your friend wants to train exactly the original architecture and produce `sample.h5`.

Install TensorFlow/Keras:

```powershell
pip install tensorflow keras
```

Run root training:

```powershell
python main.py
```

Optional epochs override:

```powershell
$env:EPOCHS='100'
python main.py
```

Expected outputs:
- `sample.h5`
- `truesample.json`
- `truesample.weights.h5`

Important note:
- In this codebase, `main.py` requires a TensorFlow-visible GPU and will raise an error if GPU is not visible.
- On native Windows with modern TensorFlow, CUDA GPU is often not available.

### Path B: GPU training that is confirmed working here (`train_gpu_pytorch.py`)

Use this if your friend wants guaranteed GPU training on this machine style (RTX + Windows).

Install CUDA PyTorch build:

```powershell
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Run training:

```powershell
python train_gpu_pytorch.py
```

Run longer training:

```powershell
$env:EPOCHS='100'
python train_gpu_pytorch.py
```

Expected output:
- `pytorch_pothole_gpu_model.pth` (best checkpoint)

## 5) Run predictions after training

### Root TensorFlow prediction (uses `sample.h5`)

```powershell
python Predictor.py
```

What it prints:
- Predicted label per image (`1 = pothole`, `0 = plain`)
- Final test accuracy

### Improved TensorFlow prediction (uses `Real-time Files/full_model.h5`)

```powershell
python "Real-time Files/Predictor.py"
```

## 6) Run real-time webcam pothole detection

```powershell
python "Real-time Files/realtimePredictor.py"
```

Controls:
- Press `e` to show/hide prediction text overlay.
- Press `q` to quit.

## 7) Full end-to-end sequence for a new friend

If someone receives this project and asks "what to run first", use this order:

1. Create venv and install dependencies (Section 3).
2. Verify dataset structure and class folders (Section 2).
3. Pick one training path:
   - Original TensorFlow (`python main.py`), or
   - Working GPU PyTorch (`python train_gpu_pytorch.py`).
4. Run batch prediction script to validate model behavior.
5. Run real-time webcam predictor (TensorFlow branch).

## 8) Which file is used for what (quick memory)

- Want to train original root model: run `main.py`.
- Want to test root model: run `Predictor.py`.
- Want improved TensorFlow training/prediction: run scripts inside `Real-time Files/`.
- Want reliable GPU training on this Windows setup: run `train_gpu_pytorch.py`.

## 9) Common errors and fixes

### Error: "No TensorFlow-visible GPU found"
Cause:
- TensorFlow cannot access CUDA GPU in this environment.

Fix options:
- Use PyTorch path (`train_gpu_pytorch.py`) for local Windows GPU.
- Or use WSL2/Linux TensorFlow GPU setup.

### Error: "No readable images found"
Cause:
- Empty folder, wrong extension, or wrong directory path.

Fix:
- Recheck `My Dataset/train/...` and `My Dataset/test/...` structure.
- Ensure images are valid `.jpg/.jpeg/.png` files.

### Very unstable accuracy
Cause:
- Very small test set gives noisy metrics.

Fix:
- Add more train/test images.
- Keep class balance close between `Pothole` and `Plain`.

## 10) Interview-style one-line explanation

This project is a binary image-classification deep learning system that preprocesses road images (grayscale + resize), trains a CNN to classify `Pothole` vs `Plain`, evaluates on test images, and can perform live webcam inference in the real-time branch.
