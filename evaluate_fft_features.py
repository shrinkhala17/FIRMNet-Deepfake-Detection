from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CSV_PATH = Path("fft_band_comparison.csv")
FEATURES = [
    "low_frequency_ratio",
    "mid_frequency_ratio",
    "high_frequency_ratio",
]


def get_source_ids(row):
    """
    REAL: 183_00.jpg -> source ID 183
    FAKE: 183_253_00.jpg -> source IDs 183 and 253
    """
    stem = Path(row["image"]).stem
    numbers = stem.split("_")

    if row["class"] == "REAL":
        return {numbers[0]}

    if row["class"] == "FAKE":
        return {numbers[0], numbers[1]}

    raise ValueError(f"Unknown class: {row['class']}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH).dropna(subset=FEATURES).copy()
    df["label"] = (df["class"] == "FAKE").astype(int)

    # Build connected groups: any samples sharing a source ID
    # must stay in the same validation fold.
    source_sets = [get_source_ids(row) for _, row in df.iterrows()]
    parent = list(range(len(df)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    first_index_for_source = {}

    for index, ids in enumerate(source_sets):
        for source_id in ids:
            if source_id in first_index_for_source:
                union(index, first_index_for_source[source_id])
            else:
                first_index_for_source[source_id] = index

    groups = np.array([find(i) for i in range(len(df))])

    print(f"Images: {len(df)}")
    print(f"REAL images: {(df['label'] == 0).sum()}")
    print(f"FAKE images: {(df['label'] == 1).sum()}")
    print(f"Independent connected groups: {len(set(groups))}")

    # A fold must have enough distinct groups and both classes.
    group_count = len(set(groups))
    if group_count < 2:
        print(
            "\nCannot perform group-aware validation: "
            "all images are connected through shared source IDs."
        )
        print(
            "Use a larger dataset with additional independent "
            "source videos before evaluating these features."
        )
        return

    n_splits = min(5, group_count)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

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

    X = df[FEATURES].to_numpy()
    y = df["label"].to_numpy()

    for model_name, model in models.items():
        actual_all = []
        predicted_all = []
        probabilities_all = []
        valid_auc = True

        print(f"\n========== {model_name} ==========")

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            y_train = y[train_idx]
            y_test = y[test_idx]

            if len(np.unique(y_train)) < 2:
                print(f"Fold {fold}: skipped; training has one class.")
                continue

            model.fit(X[train_idx], y_train)
            predictions = model.predict(X[test_idx])

            actual_all.extend(y_test)
            predicted_all.extend(predictions)

            if len(np.unique(y_test)) == 2:
                probabilities_all.extend(
                    model.predict_proba(X[test_idx])[:, 1]
                )
            else:
                valid_auc = False

            print(
                f"Fold {fold}: test={len(test_idx)}, "
                f"accuracy={accuracy_score(y_test, predictions):.3f}"
            )

        if not actual_all:
            print("No valid folds were evaluated.")
            continue

        print("\nCombined out-of-fold metrics:")
        print(f"Accuracy:  {accuracy_score(actual_all, predicted_all):.3f}")
        print(
            f"Precision: {precision_score(actual_all, predicted_all, zero_division=0):.3f}"
        )
        print(
            f"Recall:    {recall_score(actual_all, predicted_all, zero_division=0):.3f}"
        )
        print(
            f"F1-score:  {f1_score(actual_all, predicted_all, zero_division=0):.3f}"
        )

        if valid_auc and len(probabilities_all) == len(actual_all):
            print(
                f"ROC-AUC:   {roc_auc_score(actual_all, probabilities_all):.3f}"
            )
        else:
            print("ROC-AUC: unavailable for these folds.")

    print(
        "\nImportant: results are exploratory. "
        "A small number of source videos limits reliability."
    )


if __name__ == "__main__":
    main()