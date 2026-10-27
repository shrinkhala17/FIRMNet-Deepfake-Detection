from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Dataset location
DATASET_DIR = Path("data/expanded_images")
CLASSES = ["FAKE", "REAL"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_fft_features(image):
    image = image.convert("L").resize((224, 224))
    pixels = np.asarray(image, dtype=np.float32)

    # Centre the image values to reduce the DC component
    pixels = pixels - pixels.mean()

    # Calculate the 2D Fourier transform
    spectrum = np.fft.fftshift(np.fft.fft2(pixels))
    power = np.abs(spectrum) ** 2

    height, width = power.shape
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2

    distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    normalized_distance = distance / min(height, width)

    # Define low, mid and high frequency bands
    low_mask = normalized_distance < 0.08
    mid_mask = (
        (normalized_distance >= 0.08)
        & (normalized_distance < 0.25)
    )
    high_mask = normalized_distance >= 0.25

    total_energy = power.sum()

    if total_energy <= 0:
        return {
            "low_frequency_ratio": 0.0,
            "mid_frequency_ratio": 0.0,
            "high_frequency_ratio": 0.0,
        }

    return {
        "low_frequency_ratio": power[low_mask].sum() / total_energy,
        "mid_frequency_ratio": power[mid_mask].sum() / total_energy,
        "high_frequency_ratio": power[high_mask].sum() / total_energy,
    }


def main():
    results = []

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"ERROR: Folder not found: {class_dir}")
            return

        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_name}: {len(image_paths)} images")

        for path in image_paths:
            try:
                with Image.open(path) as image:
                    features = extract_fft_features(image)

                results.append({
                    "class": class_name,
                    "image": path.name,
                    **features,
                })

            except Exception as error:
                print(f"Skipped {path}: {error}")

    if not results:
        print("No images were analyzed.")
        return

    df = pd.DataFrame(results)
    output_path = Path("fft_band_comparison.csv")
    df.to_csv(output_path, index=False)

    feature_columns = [
        "low_frequency_ratio",
        "mid_frequency_ratio",
        "high_frequency_ratio",
    ]

    print("\n========== FREQUENCY BAND SUMMARY ==========")

    summary = df.groupby("class")[feature_columns].agg(
        ["mean", "std", "median"]
    )

    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))

    print("\n========== FAKE - REAL MEAN DIFFERENCES ==========")

    means = df.groupby("class")[feature_columns].mean()

    if "FAKE" in means.index and "REAL" in means.index:
        for feature in feature_columns:
            difference = means.loc["FAKE", feature] - means.loc["REAL", feature]
            print(f"{feature}: {difference:.6f}")

    print(f"\nSaved results to: {output_path}")
    print("Note: Frequency differences alone do not establish detection ability.")


if __name__ == "__main__":
    main()