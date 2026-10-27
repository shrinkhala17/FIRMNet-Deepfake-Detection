
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from pathlib import Path

# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)

# Preprocessing
weights = MobileNet_V3_Small_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    weights.transforms(),
])

val_transform = weights.transforms()

# Dataset
data_dir = Path("data/expanded_split")

train_data = datasets.ImageFolder(
    root=str(data_dir / "train"),
    transform=train_transform,
)
val_data = datasets.ImageFolder(
    root=str(data_dir / "val"),
    transform=val_transform,
)

print("Classes:", train_data.class_to_idx)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

if train_data.class_to_idx != val_data.class_to_idx:
    raise RuntimeError("Training and validation class mappings differ.")

if len(train_data) == 0 or len(val_data) == 0:
    raise RuntimeError("Training or validation dataset is empty.")

# Data loaders
train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

# Model
model = mobilenet_v3_small(weights=weights)
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, 2)
model = model.to(device)

# Fine-tune the full feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = True

# Class mapping: FAKE = 0, REAL = 1
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.features.parameters(),
        "lr": 0.000005,
    },
    {
        "params": model.classifier.parameters(),
        "lr": 0.0001,
    },
])

# Training settings
epochs = 10
best_val_accuracy = -1.0

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        train_total += labels.size(0)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss / train_total:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss / val_total:.4f} | "
        f"Val Accuracy: {val_accuracy:.2%}"
    )

    # Save the best checkpoint based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            checkpoint_dir / "mobilenet_best.pth",
        )
        print("Saved new best model.")

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print("Best checkpoint: checkpoints/mobilenet_best.pth")
