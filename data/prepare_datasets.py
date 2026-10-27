
import os
import argparse
import random
from pathlib import Path

from torchvision import datasets

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)


def split_dataset_by_class(data_dir, val_ratio=0.2, random_state=42, logger=None):
    """
    Split REAL and FAKE images by video group to reduce data leakage.

    Expected REAL filenames:
        183_01.jpg, 183_02.jpg, ...

    Expected FAKE filenames:
        183_253_01.jpg, 253_183_01.jpg, ...

    Related original videos and their fake pairs are assigned
    to the same train or validation split.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    dataset = datasets.ImageFolder(root=data_dir)
    class_to_idx = dataset.class_to_idx

    if "FAKE" not in class_to_idx or "REAL" not in class_to_idx:
        raise ValueError(
            "The dataset folder must contain 'FAKE' and 'REAL' subfolders."
        )

    # Each group contains two related original videos.
    # Both directions of their fake pair stay in the same split.
    groups = [
        {"183", "253"},
        {"469", "481"},
        {"585", "599"},
        {"672", "720"},
        {"866", "878"},
    ]

    # Validate the REAL video IDs and FAKE pairs against the dataset.
    real_ids = set()
    fake_pairs = set()

    for path, label in dataset.samples:
        parts = Path(path).stem.split("_")

        if label == class_to_idx["REAL"]:
            if len(parts) < 2:
                raise ValueError(f"Unexpected REAL filename: {path}")
            real_ids.add(parts[0])

        elif label == class_to_idx["FAKE"]:
            if len(parts) < 3:
                raise ValueError(f"Unexpected FAKE filename: {path}")
            fake_pairs.add((parts[0], parts[1]))

    for group in groups:
        for video_id in group:
            if video_id not in real_ids:
                raise ValueError(f"Missing REAL video ID: {video_id}")

        for a in group:
            for b in group:
                if a != b and (a, b) not in fake_pairs:
                    raise ValueError(f"Missing FAKE pair: {a}_{b}")

    # Randomize whole groups, never individual images.
    shuffled_groups = groups.copy()
    random.Random(random_state).shuffle(shuffled_groups)

    n_val_groups = max(1, round(len(groups) * val_ratio))
    n_val_groups = min(n_val_groups, len(groups) - 1)

    val_groups = shuffled_groups[:n_val_groups]
    train_groups = shuffled_groups[n_val_groups:]

    def belongs_to_group(path, label, selected_groups):
        parts = Path(path).stem.split("_")

        if label == class_to_idx["REAL"]:
            video_id = parts[0]
            return any(video_id in group for group in selected_groups)

        # FAKE filename: first two components are the video pair IDs.
        video_a, video_b = parts[0], parts[1]

        return any(
            video_a in group and video_b in group
            for group in selected_groups
        )

    train_indices = []
    val_indices = []

    for index, (path, label) in enumerate(dataset.samples):
        if belongs_to_group(path, label, train_groups):
            train_indices.append(index)
        elif belongs_to_group(path, label, val_groups):
            val_indices.append(index)
        else:
            raise ValueError(f"Image was not assigned to a split: {path}")

    # Ensure every image was assigned exactly once.
    if len(train_indices) + len(val_indices) != len(dataset):
        raise RuntimeError("Some images were lost during splitting.")

    if set(train_indices) & set(val_indices):
        raise RuntimeError("Training and validation indices overlap.")

    if logger:
        logger.info("Dataset split by video group:")
        logger.info(f"Training groups: {train_groups}")
        logger.info(f"Validation groups: {val_groups}")
        logger.info(f"Total images: {len(dataset)}")
        logger.info(f"Training images: {len(train_indices)}")
        logger.info(f"Validation images: {len(val_indices)}")

    return train_indices, val_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split FAKE/REAL images into train and validation sets."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder containing FAKE/ and REAL/ subfolders."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Approximate validation ratio; whole video groups are held out."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for log files."
    )

    args = parser.parse_args()
    split_logger = get_logger(
        name="prepare_dataset",
        log_dir=args.log_dir
    )

    split_logger.info("Starting dataset split check...")

    train_indices, val_indices = split_dataset_by_class(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        logger=split_logger
    )

    split_logger.info("Dataset split check completed.")
