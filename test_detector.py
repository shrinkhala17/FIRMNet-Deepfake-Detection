import torch
from PIL import Image
from torchvision import transforms
from models.backbones import FIRMNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = FIRMNet(num_classes=2)
model.load_state_dict(torch.load(
    "checkpoints/epoch15.pth",
    map_location="cpu",
    weights_only=True
))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

correct = 0
total = 0

for actual in ["REAL", "FAKE"]:
    import os
    files = sorted(os.listdir(f"data/images/{actual}"))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))][:5]

    for filename in files:
        image = Image.open(f"data/images/{actual}/{filename}").convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

        predicted = "FAKE" if probabilities.argmax().item() == 0 else "REAL"
        is_correct = predicted == actual
        correct += int(is_correct)
        total += 1

        print(
            f"Actual: {actual:4} | Predicted: {predicted:4} | "
            f"FAKE: {probabilities[0].item():.1%} | "
            f"REAL: {probabilities[1].item():.1%} | "
            f"{'CORRECT' if is_correct else 'WRONG'} | {filename}"
        )

print(f"\nAccuracy on these {total} images: {correct}/{total} ({correct / total:.1%})")
print("Note: This is a quick sanity check, not a reliable measure of generalization.")
