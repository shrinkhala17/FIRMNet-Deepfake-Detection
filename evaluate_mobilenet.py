
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pathlib import Path

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

weights = MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()

dataset = datasets.ImageFolder(
    "data/expanded_split/val",
    transform=transform
)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

model = mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)

model.load_state_dict(
    torch.load(
        "checkpoints/mobilenet_deepfake.pth",
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)
model.eval()

correct = 0
total = 0
class_correct = {name: 0 for name in dataset.classes}
class_total = {name: 0 for name in dataset.classes}
predicted_counts = {name: 0 for name in dataset.classes}

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for label, prediction in zip(labels.cpu(), predictions.cpu()):
            true_name = dataset.classes[label.item()]
            predicted_name = dataset.classes[prediction.item()]

            class_total[true_name] += 1
            class_correct[true_name] += int(label.item() == prediction.item())
            predicted_counts[predicted_name] += 1

print(f"\nOverall accuracy: {correct}/{total} = {correct / total:.1%}")

for name in dataset.classes:
    accuracy = class_correct[name] / class_total[name]
    print(f"{name} recall: {accuracy:.1%} ({class_correct[name]}/{class_total[name]})")

print("Predicted class counts:", predicted_counts)
