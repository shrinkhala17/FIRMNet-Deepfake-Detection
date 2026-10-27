
import torch
from torchvision import datasets
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

weights = MobileNet_V3_Small_Weights.DEFAULT
val_data = datasets.ImageFolder(
    "data/expanded_split/val",
    transform=weights.transforms(),
)
loader = DataLoader(val_data, batch_size=8, shuffle=False)

model = mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)
model.load_state_dict(torch.load(
    "checkpoints/mobilenet_best.pth",
    map_location="cpu",
    weights_only=True,
))
model.to(device).eval()

fake_scores = []
real_scores = []

with torch.inference_mode():
    for images, labels in loader:
        probs = F.softmax(model(images.to(device)), dim=1).cpu()
        for i, label in enumerate(labels.tolist()):
            fake_scores.append(probs[i, 0].item())
            real_scores.append(probs[i, 1].item())

for class_id, class_name in [(0, "FAKE"), (1, "REAL")]:
    indices = [
        i for i, (_, label) in enumerate(val_data.samples)
        if label == class_id
    ]
    print(f"\nActual {class_name} images:")
    print("Mean FAKE score:", round(sum(fake_scores[i] for i in indices) / len(indices), 3))
    print("Mean REAL score:", round(sum(real_scores[i] for i in indices) / len(indices), 3))
    print("Predicted FAKE:", sum(fake_scores[i] >= real_scores[i] for i in indices), "/", len(indices))
