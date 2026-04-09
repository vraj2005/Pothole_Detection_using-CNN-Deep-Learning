import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


SIZE = 300
EPOCHS = int(os.getenv("EPOCHS", "1000"))


def configure_gpu_or_fail():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "No TensorFlow-visible GPU found. Your hardware GPU may still exist, "
            "but this Python setup cannot use it. On native Windows, TensorFlow >= 2.11 "
            "does not provide CUDA GPU training. Use WSL2 TensorFlow GPU or Windows DirectML."
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU devices detected: {len(gpus)}")
    for idx, gpu in enumerate(gpus):
        print(f"  GPU[{idx}]: {gpu}")


def keras_model4(input_size: int):
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="valid", input_shape=(input_size, input_size, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dropout(0.1))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    return model


def load_images(folder: Path, size: int):
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in folder.glob(pattern):
            img = cv2.imread(str(img_path), 0)
            if img is None:
                continue
            images.append(cv2.resize(img, (size, size)))
    data = np.asarray(images)
    if data.size == 0:
        raise FileNotFoundError(f"No readable images found in: {folder}")
    return data


def main():
    configure_gpu_or_fail()

    base_dir = Path(__file__).resolve().parent
    dataset_dir = Path(__file__).resolve().parents[2] / "my_dataset"

    train_pothole = load_images(dataset_dir / "train" / "Pothole", SIZE)
    train_plain = load_images(dataset_dir / "train" / "Plain", SIZE)
    test_pothole = load_images(dataset_dir / "test" / "Pothole", SIZE)
    test_plain = load_images(dataset_dir / "test" / "Plain", SIZE)

    X_train = np.asarray([*train_pothole, *train_plain])
    X_test = np.asarray([*test_pothole, *test_plain])

    y_train = np.asarray([
        *np.ones([train_pothole.shape[0]], dtype=int),
        *np.zeros([train_plain.shape[0]], dtype=int),
    ])
    y_test = np.asarray([
        *np.ones([test_pothole.shape[0]], dtype=int),
        *np.zeros([test_plain.shape[0]], dtype=int),
    ])

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], SIZE, SIZE, 1).astype("float32") / 255.0
    X_test = X_test.reshape(X_test.shape[0], SIZE, SIZE, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    print("Train shape X:", X_train.shape)
    print("Train shape y:", y_train.shape)
    print("Test shape X:", X_test.shape)
    print("Test shape y:", y_test.shape)

    model = keras_model4(SIZE)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.1, verbose=1)

    metrics_train = model.evaluate(X_train, y_train, verbose=0)
    print(f"Training Accuracy: {metrics_train[1] * 100:.2f}%")

    metrics_test = model.evaluate(X_test, y_test, verbose=0)
    print(f"Testing Accuracy: {metrics_test[1] * 100:.2f}%")

    print("Saving model weights and configuration file")
    model.save(base_dir / "latest_full_model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    main()