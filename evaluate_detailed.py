
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device)

# Use exactly the preprocessing used during training
weights = MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()

# Load the held-out validation images
val_dir = Path("data/expanded_split/val")
val_data = datasets.ImageFolder(
    root=str(val_dir),
    transform=transform,
)
val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

print("Class mapping:", val_data.class_to_idx)
print("Validation images:", len(val_data))

# Load the trained model
model = mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)

checkpoint = torch.load(
    "checkpoints/mobilenet_deepfake.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

# Metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, pos_label=0, zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, pos_label=0, zero_division=0
)
f1 = f1_score(
    all_labels, all_predictions, pos_label=0, zero_division=0
)

print("\nDetailed Evaluation")
print(f"Accuracy:  {accuracy:.2%}")
print(f"FAKE precision: {precision:.2%}")
print(f"FAKE recall:    {recall:.2%}")
print(f"FAKE F1-score:  {f1:.2%}")

print("\nConfusion Matrix")
print("Rows = actual [FAKE, REAL]")
print("Columns = predicted [FAKE, REAL]")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))

print("\nClassification Report")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        zero_division=0,
    )
)
