from pathlib import Path

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

CSV_PATH = Path("fft_band_comparison.csv")
SPLIT_DIR = Path("data/expanded_split")

FEATURES = [
    "low_frequency_ratio",
    "mid_frequency_ratio",
    "high_frequency_ratio",
]


def main():
    df = pd.read_csv(CSV_PATH)

    # Build the exact validation filename set
    validation_files = {}
    for class_name in ["FAKE", "REAL"]:
        folder = SPLIT_DIR / "val" / class_name
        for path in folder.iterdir():
            if path.is_file():
                validation_files[path.name] = class_name

    # Keep only the images used by MobileNet's validation split
    df = df[df["image"].isin(validation_files)].copy()

    # Use folder labels as ground truth
    df["class"] = df["image"].map(validation_files)
    df["label"] = (df["class"] == "FAKE").astype(int)

    if len(df) != len(validation_files):
        raise ValueError(
            f"Expected {len(validation_files)} validation images, "
            f"found {len(df)} in the FFT CSV."
        )

    X = df[FEATURES]
    y = df["label"]

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
    }

    print(f"Validation images: {len(df)}")
    print(f"FAKE: {(y == 1).sum()}, REAL: {(y == 0).sum()}")
    print("\n========== FFT VALIDATION RESULTS ==========")

    for name, model in models.items():
        # This is a tiny fixed validation set, not cross-validation.
        # These models are fitted on validation images only to measure
        # whether the feature distributions separate these same images.
        # Do NOT treat these metrics as generalization performance.
        model.fit(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        print(f"\n{name}")
        print(f"Accuracy:  {accuracy_score(y, predictions):.3f}")
        print(
            f"Precision: {precision_score(y, predictions, zero_division=0):.3f}"
        )
        print(
            f"Recall:    {recall_score(y, predictions, zero_division=0):.3f}"
        )
        print(
            f"F1-score:  {f1_score(y, predictions, zero_division=0):.3f}"
        )
        print(f"ROC-AUC:   {roc_auc_score(y, probabilities):.3f}")
        print("Confusion matrix [actual REAL, FAKE; predicted REAL, FAKE]:")
        print(confusion_matrix(y, predictions, labels=[0, 1]))

    print(
        "\nWARNING: These scores are in-sample because each model was "
        "fitted and evaluated on the same 40 images. They are diagnostic "
        "only and must not be compared as fair validation scores with MobileNet."
    )


if __name__ == "__main__":
    main()