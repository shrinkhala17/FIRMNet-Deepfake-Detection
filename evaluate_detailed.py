# evaluate_detailed.py
#
# Evaluate the model that is actually used by the website (app.py) on a
# properly labelled, held-out test/validation dataset.
#
# Supports both FIRMNet and MobileNetV3-Small checkpoints.  The script
# auto-detects which checkpoint exists and loads the matching architecture.

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Checkpoint discovery (same priority as app.py)
# ---------------------------------------------------------------------------
FIRMNET_CHECKPOINTS = [
    "checkpoints/firmnet_best.pth",
    "checkpoints/epoch24.pth",
    "checkpoints/epoch15.pth",
]
MOBILENET_CHECKPOINTS = [
    "checkpoints/mobilenet_best.pth",
    "checkpoints/mobilenet_deepfake.pth",
]


def find_checkpoint(override_path=None):
    """Return (path, model_type) or raise."""
    if override_path and os.path.isfile(override_path):
        # Guess type from filename
        name = os.path.basename(override_path).lower()
        mtype = "mobilenet" if "mobilenet" in name else "firmnet"
        return override_path, mtype

    for path in FIRMNET_CHECKPOINTS:
        if os.path.isfile(path):
            return path, "firmnet"
    for path in MOBILENET_CHECKPOINTS:
        if os.path.isfile(path):
            return path, "mobilenet"

    raise FileNotFoundError(
        "No model checkpoint found in checkpoints/.\n"
        "Train a model first:\n"
        "  FIRMNet:    python main.py --config config/config.yaml --data_dir data/images\n"
        "  MobileNet:  python train_mobilenet.py"
    )


def load_model(checkpoint_path, model_type, device):
    """Load model and return (model, transform)."""
    if model_type == "firmnet":
        from models.backbones import FIRMNet

        model = FIRMNet(num_classes=2).to(device)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incompatible with FIRMNet: {exc}"
            )
        model.to(device).eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return model, transform

    else:  # mobilenet
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, 2
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incompatible with "
                f"{'FIRMNet' if model_type == 'firmnet' else 'MobileNetV3-Small'}: {exc}"
            )
        model.to(device).eval()
        return model, weights.transforms()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection model on a held-out dataset"
    )
    parser.add_argument(
        "--val_dir", type=str, default="data/expanded_split/val",
        help="Path to validation/test directory with FAKE/ and REAL/ subfolders",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override checkpoint path (auto-detected if omitted)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
    )
    args = parser.parse_args()

    # Validate dataset
    val_dir = Path(args.val_dir)
    if not val_dir.is_dir():
        print(f"ERROR: Validation directory not found: {val_dir}")
        print("Prepare a dataset with FAKE/ and REAL/ subfolders first.")
        sys.exit(1)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Checkpoint
    checkpoint_path, model_type = find_checkpoint(args.checkpoint)
    print(f"Model type: {model_type}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load model and matching transform
    model, transform = load_model(checkpoint_path, model_type, device)

    # Load dataset
    val_data = datasets.ImageFolder(
        root=str(val_dir),
        transform=transform,
    )
    if len(val_data) == 0:
        print("ERROR: Validation dataset is empty.")
        sys.exit(1)

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Class mapping: {val_data.class_to_idx}")
    print(f"Validation images: {len(val_data)}")

    # Check class balance (use .targets to avoid loading all images)
    from collections import Counter
    label_counts = Counter(val_data.targets)
    class_counts = {val_data.classes[k]: v for k, v in label_counts.items()}
    print(f"Class distribution: {class_counts}")
    if len(class_counts) < 2:
        print("WARNING: Only one class present -- metrics may be misleading.")

    # Run evaluation
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
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
    recall_val = recall_score(
        all_labels, all_predictions, pos_label=0, zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, pos_label=0, zero_division=0
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:        {accuracy:.2%}")
    print(f"FAKE precision:  {precision:.2%}")
    print(f"FAKE recall:     {recall_val:.2%}")
    print(f"FAKE F1-score:   {f1:.2%}")

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

    # Data leakage warning
    print("\nNote: Ensure this validation set was NOT used during training.")
    print(f"Results are from {len(all_labels)} images in {val_dir}.")


if __name__ == "__main__":
    main()
