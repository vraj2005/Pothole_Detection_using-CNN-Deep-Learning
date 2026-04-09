import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


size = 100
root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir / "My Dataset" / "test"
model_path = root_dir / "sample.h5"

model = load_model(model_path, compile=False)


def load_images(folder: Path):
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in folder.glob(pattern):
            img = cv2.imread(str(img_path), 0)
            if img is None:
                continue
            images.append(cv2.resize(img, (size, size)))
    return np.asarray(images)


plain_images = load_images(dataset_dir / "Plain")
pothole_images = load_images(dataset_dir / "Pothole")

X_test = np.asarray([*pothole_images, *plain_images]).reshape(-1, size, size, 1)
y_test = np.asarray([
    *np.ones([pothole_images.shape[0]], dtype=int),
    *np.zeros([plain_images.shape[0]], dtype=int),
])
y_test = to_categorical(y_test, num_classes=2)

probs = model.predict(X_test, verbose=0)
preds = np.argmax(probs, axis=1)
y_true = np.argmax(y_test, axis=1)
for i, pred in enumerate(preds):
    print(f">>> Predicted {i} = {pred}")

accuracy = float(np.mean(preds == y_true))
print(f"Test Accuracy: {accuracy * 100:.2f}%")