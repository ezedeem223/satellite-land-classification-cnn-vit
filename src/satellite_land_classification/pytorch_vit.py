from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from .augmentation import pytorch_normalization
from .data import validate_dataset_dir
from .utils import AssetNotFoundError, ensure_dir


class ConvNet(nn.Module):
    """CNN backbone matching the preserved PyTorch baseline implementation."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
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
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
        return self.classifier(x)


class PatchEmbed(nn.Module):
    """Patch embedding bridge between the CNN feature map and the transformer."""

    def __init__(self, input_channel: int = 1024, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(input_channel, embed_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MHSA(nn.Module):
    """Multi-head self-attention block used in the preserved hybrid model."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        k = k.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        v = v.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """Transformer encoder block used in the preserved hybrid model."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Transformer head used in the PyTorch CNN-ViT hybrid model."""

    def __init__(
        self,
        in_ch: int = 1024,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_tokens: int = 50,
    ):
        super().__init__()
        self.patch = PatchEmbed(in_ch, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, max_tokens, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        batch, tokens, _ = x.shape
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat((cls, x), 1)
        x = x + self.pos[:, : tokens + 1]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


class CNNViTHybrid(nn.Module):
    """CNN-ViT hybrid model matching the preserved PyTorch implementation."""

    def __init__(self, num_classes: int = 2, embed_dim: int = 768, depth: int = 3, heads: int = 6):
        super().__init__()
        self.cnn = ConvNet(num_classes)
        self.vit = ViT(num_classes=num_classes, embed_dim=embed_dim, depth=depth, heads=heads)

    def forward(self, x):
        return self.vit(self.cnn.forward_features(x))


def create_pytorch_vit_dataloaders(
    dataset_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 7331,
    num_workers: int = 0,
):
    """Create the dataloaders used in the preserved PyTorch hybrid workflow."""

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


def train_epoch(model, loader, optimizer, criterion, device: str):
    """Train one epoch of the PyTorch hybrid model."""

    model.train()
    loss_sum = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * images.size(0)
        correct += int((outputs.argmax(1) == labels).sum().item())
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def evaluate_epoch(model, loader, criterion, device: str):
    """Evaluate one epoch of the PyTorch hybrid model."""

    with torch.no_grad():
        model.eval()
        loss_sum = 0.0
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += float(loss.item()) * images.size(0)
            correct += int((outputs.argmax(1) == labels).sum().item())
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def train_pytorch_cnn_vit(
    dataset_dir: str | Path,
    pretrained_cnn_path: str | Path,
    output_model_path: str | Path,
    image_size: int = 64,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    epochs: int = 5,
    validation_split: float = 0.2,
    seed: int = 7331,
    num_workers: int = 0,
    depth: int = 3,
    attention_heads: int = 6,
    embed_dim: int = 768,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Train the PyTorch CNN-ViT hybrid and save the best state dict."""

    cnn_path = Path(pretrained_cnn_path)
    if not cnn_path.exists():
        raise AssetNotFoundError(
            f"Pretrained PyTorch CNN backbone not found at {cnn_path}. "
            "Train the baseline PyTorch CNN first or provide a compatible pretrained state dict."
        )
    model_path = Path(output_model_path)
    ensure_dir(model_path.parent)
    train_loader, val_loader = create_pytorch_vit_dataloaders(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
        num_workers=num_workers,
    )
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNViTHybrid(
        num_classes=2,
        heads=attention_heads,
        depth=depth,
        embed_dim=embed_dim,
    ).to(runtime_device)
    model.cnn.load_state_dict(torch.load(cnn_path, map_location=runtime_device), strict=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    for _epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            runtime_device,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, runtime_device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
    return history


def load_pytorch_hybrid(
    model_path: str | Path,
    num_classes: int = 2,
    depth: int = 3,
    attention_heads: int = 6,
    embed_dim: int = 768,
    device: str | None = None,
):
    """Load a PyTorch hybrid checkpoint."""

    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNViTHybrid(
        num_classes=num_classes,
        heads=attention_heads,
        depth=depth,
        embed_dim=embed_dim,
    ).to(runtime_device)
    state_dict = torch.load(model_path, map_location=runtime_device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict_with_pytorch_hybrid(
    model_path: str | Path,
    image_path: str | Path,
    depth: int = 3,
    attention_heads: int = 6,
    embed_dim: int = 768,
) -> dict[str, float]:
    """Predict a single image with the PyTorch hybrid model."""

    means, stds = pytorch_normalization()
    model = load_pytorch_hybrid(
        model_path,
        depth=depth,
        attention_heads=attention_heads,
        embed_dim=embed_dim,
    )
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
