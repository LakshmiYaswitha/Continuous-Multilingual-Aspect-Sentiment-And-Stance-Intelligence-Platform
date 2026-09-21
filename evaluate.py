"""Evaluate the saved CMASSIP sentiment reference model.

Usage:
python evaluate.py --file data/processed/kaggle_sentiment_train.csv \
    --text-col text --label-col label
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


MODEL_PATH = Path("models/kaggle_sentiment_reference.joblib")
OUTPUT_PATH = Path("models/sentiment_evaluation.json")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--label-col", required=True)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    df = pd.read_csv(
        args.file,
        sep=None,
        engine="python"
    )

    df = df.dropna(
        subset=[args.text_col, args.label_col]
    )

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    vectorizer = model["vectorizer"]
    classifier = model["classifier"]

    X = vectorizer.transform(
        df[args.text_col].astype(str)
    )

    y_true = (
        df[args.label_col]
        .astype(str)
        .str.lower()
    )

    y_pred = classifier.predict(X)

    y_pred = pd.Series(y_pred).astype(str).str.lower()

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "rows_evaluated": int(len(df)),
    }

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2
        ),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------

    print("\n====================================")
    print("CMASSIP SENTIMENT EVALUATION")
    print("====================================")

    print(
        f"Rows evaluated       : {len(df)}"
    )

    print(
        f"Accuracy             : {accuracy:.4f}"
    )

    print(
        f"Macro Precision      : {precision:.4f}"
    )

    print(
        f"Macro Recall         : {recall:.4f}"
    )

    print(
        f"Macro F1             : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1          : {weighted_f1:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print(
        f"\nSaved evaluation to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()