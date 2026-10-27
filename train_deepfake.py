"""
train_deepfake.py — Train MobileNetV3-Small on a deepfake detection dataset.

Downloads the Deepfake-vs-Real-60K dataset from Hugging Face, splits it
into train/val/test, and trains a MobileNetV3-Small classifier.

Usage:
    python train_deepfake.py                     # Full training
    python train_deepfake.py --max_images 2000   # Quick test run
    python train_deepfake.py --epochs 20         # More epochs

Class mapping (matches train_mobilenet.py):
    0 = FAKE
    1 = REAL
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import Counter

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def download_and_prepare(data_dir, max_images=None):
    """Download the Deepfake-vs-Real dataset and organize into FAKE/REAL folders."""
    data_path = Path(data_dir)

    # Check if already prepared
    train_dir = data_path / "train"
    if (train_dir / "FAKE").exists() and (train_dir / "REAL").exists():
        fake_count = len(list((train_dir / "FAKE").glob("*")))
        real_count = len(list((train_dir / "REAL").glob("*")))
        if fake_count > 0 and real_count > 0:
            print(f"Dataset already prepared: {fake_count} FAKE, {real_count} REAL in train/")
            return True

    print("Downloading 140K Real-and-Fake-Faces dataset from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: 'huggingface_hub' not installed. Run: pip install huggingface_hub")
        return False

    repo_id = "TheKernel01/140k-Real-and-Fake-Faces"
    import io
    import pandas as pd

    # Map HF splits to our local split names
    hf_splits = {
        "train": [f"data/train-0000{i}-of-00006.parquet" for i in range(6)],
        "val": [f"data/validation-0000{i}-of-00002.parquet" for i in range(2)],
        "test": [f"data/test-0000{i}-of-00002.parquet" for i in range(2)],
    }

    for split_name in ["train", "val", "test"]:
        for cls in ["FAKE", "REAL"]:
            (data_path / split_name / cls).mkdir(parents=True, exist_ok=True)

    label_map = {0: "FAKE", 1: "REAL"}
    total_saved = 0
    per_split_max = None
    if max_images:
        per_split_max = {"train": int(max_images * 0.8),
                         "val": int(max_images * 0.1),
                         "test": int(max_images * 0.1)}

    try:
        for split_name, pq_files in hf_splits.items():
            split_count = 0
            split_limit = per_split_max[split_name] if per_split_max else None

            for pq_file in pq_files:
                if split_limit and split_count >= split_limit:
                    break

                print(f"  Downloading {pq_file}...")
                local_path = hf_hub_download(repo_id, pq_file, repo_type="dataset")
                df = pd.read_parquet(local_path)

                for idx, row in df.iterrows():
                    if split_limit and split_count >= split_limit:
                        break

                    # Extract image and label
                    try:
                        img_data = row["image"]
                        label = int(row["label"])
                    except (KeyError, TypeError):
                        continue

                    cls = label_map.get(label)
                    if cls is None:
                        continue

                    # Handle image data (dict with 'bytes' key or raw bytes)
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_bytes = img_data["bytes"]
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                    else:
                        continue

                    dest = data_path / split_name / cls / f"{cls.lower()}_{total_saved:06d}.jpg"
                    try:
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img.save(str(dest), "JPEG", quality=92)
                        split_count += 1
                        total_saved += 1
                    except Exception:
                        continue

                    if split_count % 500 == 0:
                        print(f"    {split_name}: {split_count} images saved")

            print(f"  {split_name}: {split_count} images total")

        _print_split_summary(data_path)
        return total_saved > 0

    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _print_split_summary(data_path):
    """Print dataset split summary."""
    for split_name in ["train", "val", "test"]:
        fake_count = len(list((data_path / split_name / "FAKE").glob("*")))
        real_count = len(list((data_path / split_name / "REAL").glob("*")))
        print(f"  {split_name}: {fake_count} FAKE + {real_count} REAL = {fake_count + real_count}")


def train(data_dir, epochs=10, batch_size=16, lr_backbone=5e-6, lr_head=1e-4):
    """Train MobileNetV3-Small on the prepared dataset."""
    data_path = Path(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Preprocessing
    weights = MobileNet_V3_Small_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        weights.transforms(),
    ])

    val_transform = weights.transforms()

    # Datasets
    train_data = datasets.ImageFolder(str(data_path / "train"), transform=train_transform)
    val_data = datasets.ImageFolder(str(data_path / "val"), transform=val_transform)

    print(f"\nClass mapping: {train_data.class_to_idx}")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")

    # Verify class mapping
    expected = {"FAKE": 0, "REAL": 1}
    if train_data.class_to_idx != expected:
        print(f"WARNING: Class mapping is {train_data.class_to_idx}, expected {expected}")
        print("This may affect prediction labels in inference.")

    # Check class balance
    train_labels = Counter(train_data.targets)
    print(f"Train class balance: FAKE={train_labels.get(0, 0)}, REAL={train_labels.get(1, 0)}")

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    # Fine-tune full model
    for p in model.features.parameters():
        p.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": lr_backbone},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ])

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_val_acc = -1.0
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training MobileNetV3-Small for {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss/train_total:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_dir / "mobilenet_best.pth")
            print(f"  -> Saved best model (val_acc={val_acc:.2%})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping (no improvement for {patience} epochs)")
                break

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: checkpoints/mobilenet_best.pth")
    print(f"{'='*60}")

    return best_val_acc


def evaluate(data_dir, batch_size=16):
    """Evaluate the best checkpoint on the test set."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
    )

    data_path = Path(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path("checkpoints/mobilenet_best.pth")
    if not checkpoint_path.exists():
        print("ERROR: No trained checkpoint found. Run training first.")
        return

    # Load model
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.to(device).eval()

    # Test dataset
    test_data = datasets.ImageFolder(str(data_path / "test"), transform=weights.transforms())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nEvaluating on test set: {len(test_data)} images")
    print(f"Class mapping: {test_data.class_to_idx}")

    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", pos_label=0)
    rec = recall_score(all_labels, all_preds, average="binary", pos_label=0)
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%} (for FAKE class)")
    print(f"Recall:    {rec:.2%} (for FAKE class)")
    print(f"F1-score:  {f1:.2%} (for FAKE class)")
    print(f"\nConfusion Matrix:")
    print(f"              Pred FAKE  Pred REAL")
    print(f"  True FAKE    {cm[0][0]:>6d}     {cm[0][1]:>6d}")
    print(f"  True REAL    {cm[1][0]:>6d}     {cm[1][1]:>6d}")
    print(f"\nFalse Positives (REAL predicted as FAKE): {cm[1][0]}")
    print(f"False Negatives (FAKE predicted as REAL): {cm[0][1]}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['FAKE', 'REAL'])}")

    return acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--data_dir", type=str, default="data/deepfake_dataset",
                        help="Directory for dataset")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Max images to download (None = all ~60K)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download if data already exists")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Only evaluate, don't train")
    args = parser.parse_args()

    if not args.evaluate_only:
        if not args.skip_download:
            ok = download_and_prepare(args.data_dir, max_images=args.max_images)
            if not ok:
                sys.exit(1)

        train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)

    evaluate(args.data_dir, batch_size=args.batch_size)
