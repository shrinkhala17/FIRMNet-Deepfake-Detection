
from pathlib import Path
import random
import shutil

random.seed(42)

source_dir = Path("data/images")
output_dir = Path("data/split")

real_files = list((source_dir / "REAL").glob("*.jpg"))
fake_files = list((source_dir / "FAKE").glob("*.jpg"))

# Each real video is a node; fake pairs connect two nodes.
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    parent[find(a)] = find(b)

for path in real_files:
    video_id = path.stem.rsplit("_", 1)[0]
    parent.setdefault(video_id, video_id)

for path in fake_files:
    parts = path.stem.split("_")
    if len(parts) >= 3:
        union(parts[0], parts[1])

groups = {}
for video_id in list(parent):
    groups.setdefault(find(video_id), set()).add(video_id)

groups = list(groups.values())
random.shuffle(groups)

print("Related source groups:")
for group in groups:
    print(sorted(group))

# Keep connected source IDs together.
val_groups = set()
val_count = 0
target_val = 2

for i, group in enumerate(groups):
    if val_count < target_val and i < len(groups) - 1:
        val_groups.update(group)
        val_count += 1

val_ids = val_groups
train_ids = set(parent) - val_ids

print("Train source IDs:", sorted(train_ids))
print("Validation source IDs:", sorted(val_ids))

# Create separate folders for ImageFolder.
for split in ["train", "val"]:
    for label in ["REAL", "FAKE"]:
        (output_dir / split / label).mkdir(parents=True, exist_ok=True)

# Copy images without modifying the originals.
for path in real_files:
    video_id = path.stem.rsplit("_", 1)[0]
    split = "val" if video_id in val_ids else "train"
    shutil.copy2(path, output_dir / split / "REAL" / path.name)

for path in fake_files:
    parts = path.stem.split("_")
    first_id, second_id = parts[0], parts[1]

    if first_id in val_ids and second_id in val_ids:
        split = "val"
    elif first_id in train_ids and second_id in train_ids:
        split = "train"
    else:
        raise ValueError(f"Fake pair crosses split boundary: {path.name}")

    shutil.copy2(path, output_dir / split / "FAKE" / path.name)

print("\nSplit created at data/split/")
for split in ["train", "val"]:
    for label in ["REAL", "FAKE"]:
        count = len(list((output_dir / split / label).glob("*.jpg")))
        print(f"{split}/{label}: {count} images")
i