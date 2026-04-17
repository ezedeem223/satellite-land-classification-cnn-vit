from __future__ import annotations

from pathlib import Path

from .augmentation import pytorch_normalization
from .data import validate_dataset_dir
from .utils import ensure_dir


def _torch():
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms

    return torch, nn, datasets, transforms


def build_pytorch_cnn(num_classes: int = 2):
    """Build the PyTorch CNN architecture used in Module 2."""

    _, nn, _, _ = _torch()
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 256, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 512, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 1024, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(1024),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes),
    )


def create_pytorch_dataloaders(
    dataset_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 128,
    validation_split: float = 0.2,
    seed: int = 7331,
    num_workers: int = 0,
):
    """Create PyTorch train/validation dataloaders with notebook-aligned transforms."""

    torch, _, datasets, transforms = _torch()
    root = validate_dataset_dir(dataset_dir)
    means, stds = pytorch_normalization()
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=0.2),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )
    base_dataset = datasets.ImageFolder(root)
    generator = torch.Generator().manual_seed(seed)
    train_size = int((1.0 - validation_split) * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        list(range(len(base_dataset))), [train_size, val_size], generator=generator
    )
    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root, transform=train_transform),
        train_subset.indices,
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root, transform=val_transform),
        val_subset.indices,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def train_pytorch_cnn(
    dataset_dir: str | Path,
    output_model_path: str | Path,
    image_size: int = 64,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 3,
    validation_split: float = 0.2,
    seed: int = 7331,
    num_workers: int = 0,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Train the PyTorch CNN and save the best state dict."""

    torch, nn, _, _ = _torch()
    model_path = Path(output_model_path)
    ensure_dir(model_path.parent)
    train_loader, val_loader = create_pytorch_dataloaders(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
        num_workers=num_workers,
    )
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pytorch_cnn().to(runtime_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    for _epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(runtime_device), labels.to(runtime_device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1)
            train_correct += int((preds == labels).sum().item())
            train_total += int(labels.size(0))
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(runtime_device), labels.to(runtime_device)
                outputs = model(images)
                val_loss += float(criterion(outputs, labels).item())
                preds = torch.argmax(outputs, dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(labels.size(0))
        avg_val_loss = val_loss / max(len(val_loader), 1)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
        history["train_loss"].append(train_loss / max(len(train_loader), 1))
        history["val_loss"].append(avg_val_loss)
        history["train_accuracy"].append(train_correct / max(train_total, 1))
        history["val_accuracy"].append(val_correct / max(val_total, 1))
    return history


def load_pytorch_cnn(model_path: str | Path, num_classes: int = 2, device: str | None = None):
    """Load a PyTorch CNN checkpoint."""

    torch, _, _, _ = _torch()
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pytorch_cnn(num_classes=num_classes).to(runtime_device)
    state_dict = torch.load(model_path, map_location=runtime_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_with_pytorch_cnn(model_path: str | Path, image_path: str | Path) -> dict[str, float]:
    """Predict a single image with the PyTorch CNN."""

    torch, _, _, transforms = _torch()
    means, stds = pytorch_normalization()
    model = load_pytorch_cnn(model_path)
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )
    from PIL import Image

    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {"non_agri": float(probs[0]), "agri": float(probs[1])}
