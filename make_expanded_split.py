
from pathlib import Path
from collections import defaultdict
import random
import shutil

random.seed(42)

source_dir = Path("data/expanded_images")
output_dir = Path("data/expanded_split")

# Refuse to mix new output with an older split.
if output_dir.exists() and any(output_dir.rglob("*.jpg")):
    raise RuntimeError(
        "data/expanded_split already contains JPG files. "
        "Rename or safely clear that folder before rerunning."
    )

real_files = list((source_dir / "REAL").glob("*.jpg"))
fake_files = list((source_dir / "FAKE").glob("*.jpg"))

# Group source videos that are connected by a fake pair.
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    parent[find(a)] = find(b)

for path in real_files:
    # Names look like 183_00.jpg; source ID is 183.
    video_id = path.stem.rsplit("_", 1)[0]
    parent.setdefault(video_id, video_id)

for path in fake_files:
    # Names look like 183_253_00.jpg.
    parts = path.stem.split("_")
    if len(parts) >= 3:
        union(parts[0], parts[1])

groups = defaultdict(set)
for video_id in list(parent):
    groups[find(video_id)].add(video_id)

groups = list(groups.values())
random.shuffle(groups)

# Hold out about 20% of source groups, never splitting connected IDs.
val_ids = set()
target_val = max(1, round(len(parent) * 0.2))
val_count = 0

for i, group in enumerate(groups):
    if val_count < target_val and i < len(groups) - 1:
        val_ids.update(group)
        val_count += len(group)

train_ids = set(parent) - val_ids

print("Train source IDs:", sorted(train_ids))
print("Validation source IDs:", sorted(val_ids))

for split in ("train", "val"):
    for label in ("REAL", "FAKE"):
        (output_dir / split / label).mkdir(parents=True, exist_ok=True)

# Copy real frames according to their source video ID.
for path in real_files:
    video_id = path.stem.rsplit("_", 1)[0]
    split = "val" if video_id in val_ids else "train"
    shutil.copy2(path, output_dir / split / "REAL" / path.name)

# Copy fake frames only when both source IDs belong to one split.
for path in fake_files:
    parts = path.stem.split("_")
    first_id, second_id = parts[0], parts[1]

    if first_id in val_ids and second_id in val_ids:
        split = "val"
    elif first_id in train_ids and second_id in train_ids:
        split = "train"
    else:
        raise RuntimeError(f"Pair crosses split boundary: {path.name}")

    shutil.copy2(path, output_dir / split / "FAKE" / path.name)

print("\nImage counts:")
for split in ("train", "val"):
    for label in ("REAL", "FAKE"):
        count = len(list((output_dir / split / label).glob("*.jpg")))
        print(f"{split}/{label}: {count}")
