
import torch
from torchvision import datasets
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# Use the same device and preprocessing as the app
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

weights = MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()

val_data = datasets.ImageFolder(
    "data/expanded_split/val",
    transform=transform,
)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# Build the same model architecture
model = mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)

checkpoint = torch.load(
    "checkpoints/mobilenet_best.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)
        predictions = model(images).argmax(dim=1).cpu()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

print("Class mapping:", val_data.class_to_idx)
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))
print("\nDetailed report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    zero_division=0,
))
