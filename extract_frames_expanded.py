
import cv2
from pathlib import Path
import numpy as np

original_dir = Path("data/ffpp/original_sequences/youtube/c23/videos")
fake_dir = Path("data/ffpp/manipulated_sequences/Deepfakes/c23/videos")

output_dir = Path("data/expanded_images")
real_out = output_dir / "REAL"
fake_out = output_dir / "FAKE"

real_out.mkdir(parents=True, exist_ok=True)
fake_out.mkdir(parents=True, exist_ok=True)

frames_per_video = 10


def extract_frames(video_path, output_folder, prefix):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"No frames found: {video_path}")
        return 0

    positions = np.linspace(
        0, total_frames - 1, frames_per_video, dtype=int
    )

    saved = 0

    for index, position in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.resize(frame, (256, 256))
        filename = f"{prefix}_{index:02d}.jpg"
        destination = output_folder / filename

        if cv2.imwrite(str(destination), frame):
            saved += 1

    cap.release()
    print(f"{video_path.name}: saved {saved} frames")
    return saved


total = 0

for video in sorted(original_dir.glob("*.mp4")):
    total += extract_frames(video, real_out, video.stem)

for video in sorted(fake_dir.glob("*.mp4")):
    total += extract_frames(video, fake_out, video.stem)

print(f"\nFinished. Total frames saved: {total}")
print(f"REAL images: {len(list(real_out.glob('*.jpg')))}")
print(f"FAKE images: {len(list(fake_out.glob('*.jpg')))}")
print(f"Output: {output_dir}")
