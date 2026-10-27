from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from frequency_analysis import analyze_frequency


# Dataset configuration
DATASET_DIR = Path("data/expanded_images")
CLASSES = ["FAKE", "REAL"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    results = []

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"ERROR: Dataset folder not found: {class_dir}")
            return

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\n{class_name}: {len(image_paths)} images")

        for image_path in image_paths:
            figure = None

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    figure, ratio = analyze_frequency(image)

                results.append({
                    "class": class_name,
                    "image": image_path.name,
                    "high_frequency_ratio": ratio,
                })

            except Exception as error:
                print(f"Skipped {image_path}: {error}")

            finally:
                if figure is not None:
                    plt.close(figure)

    if not results:
        print("\nNo images were successfully analyzed.")
        return

    # Save per-image results
    df = pd.DataFrame(results)
    output_path = Path("frequency_comparison.csv")
    df.to_csv(output_path, index=False)

    # Display image counts
    print("\n========== IMAGE COUNTS ==========")
    print(df.groupby("class").size().to_string())

    # Display summary statistics
    print("\n========== FREQUENCY RATIO SUMMARY ==========")

    summary = (
        df.groupby("class")["high_frequency_ratio"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            minimum="min",
            median="median",
            maximum="max",
        )
    )

    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))

    # Compare class means
    if "FAKE" in summary.index and "REAL" in summary.index:
        fake_mean = summary.loc["FAKE", "mean"]
        real_mean = summary.loc["REAL", "mean"]
        difference = fake_mean - real_mean

        print("\n========== CLASS COMPARISON ==========")
        print(f"FAKE mean ratio: {fake_mean:.6f}")
        print(f"REAL mean ratio: {real_mean:.6f}")
        print(f"Difference (FAKE - REAL): {difference:.6f}")

    print(f"\nSaved per-image results to: {output_path}")
    print("\nAnalysis complete.")
    print(
        "Note: Differences in frequency ratios alone do not "
        "establish deepfake detection ability."
    )


if __name__ == "__main__":
    main()