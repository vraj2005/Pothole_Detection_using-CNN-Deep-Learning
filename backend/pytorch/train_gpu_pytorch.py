from pathlib import Path
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SIZE = 300
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
LEARNING_RATE = float(os.getenv("LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
SEED = int(os.getenv("SEED", "42"))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available for PyTorch in this environment. "
            "Install CUDA-enabled PyTorch wheel and retry."
        )
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    return device


def augment_image(img: np.ndarray) -> list[np.ndarray]:
    # Keep augmentations light to preserve pothole geometry.
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat_1 = cv2.getRotationMatrix2D(center, random.uniform(-8, 8), 1.0)
    rot_mat_2 = cv2.getRotationMatrix2D(center, random.uniform(-8, 8), 1.0)
    rot1 = cv2.warpAffine(img, rot_mat_1, (w, h), borderMode=cv2.BORDER_REFLECT)
    rot2 = cv2.warpAffine(img, rot_mat_2, (w, h), borderMode=cv2.BORDER_REFLECT)
    flip = cv2.flip(img, 1)
    return [flip, rot1, rot2]


def load_images(folder: Path, label: int, do_augment: bool = False):
    images, labels = [], []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in folder.glob(pattern):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (SIZE, SIZE)).astype("float32") / 255.0
            images.append(img)
            labels.append(label)
            if do_augment:
                for aug_img in augment_image(img):
                    images.append(aug_img.astype("float32"))
                    labels.append(label)
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")
    return images, labels


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

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def make_loaders(root: Path):
    train_pothole_x, train_pothole_y = load_images(root / "train" / "Pothole", 1, do_augment=True)
    train_plain_x, train_plain_y = load_images(root / "train" / "Plain", 0, do_augment=True)
    test_pothole_x, test_pothole_y = load_images(root / "test" / "Pothole", 1, do_augment=False)
    test_plain_x, test_plain_y = load_images(root / "test" / "Plain", 0, do_augment=False)

    X_train = np.array(train_pothole_x + train_plain_x, dtype="float32")
    y_train = np.array(train_pothole_y + train_plain_y, dtype="int64")
    X_test = np.array(test_pothole_x + test_plain_x, dtype="float32")
    y_test = np.array(test_pothole_y + test_plain_y, dtype="int64")

    rng = np.random.default_rng(42)
    train_perm = rng.permutation(len(X_train))
    test_perm = rng.permutation(len(X_test))

    X_train = X_train[train_perm][:, None, :, :]
    y_train = y_train[train_perm]
    X_test = X_test[test_perm][:, None, :, :]
    y_test = y_test[test_perm]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    class_counts = np.bincount(y_train, minlength=2)
    return train_loader, test_loader, class_counts


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def main():
    set_seed(SEED)
    device = require_gpu()

    root = Path(__file__).resolve().parents[1] / "my_dataset"
    train_loader, test_loader, class_counts = make_loaders(root)
    print(f"Class counts after augmentation [plain, pothole]: {class_counts.tolist()}")

    model = PotholeCNN().to(device)
    weights = torch.tensor(
        [1.0 / max(class_counts[0], 1), 1.0 / max(class_counts[1], 1)], dtype=torch.float32, device=device
    )
    weights = weights / weights.sum() * 2.0
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f} "
            f"- Train Acc: {train_acc*100:.2f}% - Test Acc: {test_acc*100:.2f}%"
        )
        scheduler.step(test_acc)
        if test_acc >= best_acc:
            best_acc = test_acc
            out_path = Path(__file__).resolve().parent / "pytorch_pothole_gpu_model.pth"
            torch.save(model.state_dict(), out_path)
            print(f"Saved best model (Test Acc: {best_acc*100:.2f}%) to: {out_path}")

    print(f"Best Test Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
