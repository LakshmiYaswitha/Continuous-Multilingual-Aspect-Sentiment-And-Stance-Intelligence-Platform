"""Train and evaluate the reproducible sentiment reference model.

Usage:

python train.py \
    --file data/train.csv \
    --text-col review \
    --label-col sentiment \
    --limit 30000
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.model_store import train_sentiment


MODEL_PATH = Path("models/sentiment.joblib")
METRICS_PATH = Path("models/sentiment_metrics.json")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        required=True
    )

    parser.add_argument(
        "--text-col",
        required=True
    )

    parser.add_argument(
        "--label-col",
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30000
    )

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
        subset=[
            args.text_col,
            args.label_col
        ]
    )

    df = (
        df.sample(
            frac=1,
            random_state=42
        )
        .head(args.limit)
    )

    X = df[args.text_col].astype(str)

    y = (
        df[args.label_col]
        .astype(str)
        .str.lower()
    )

    # ---------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows : {len(X_test)}"
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    model = train_sentiment(
        X_train,
        y_train
    )

    predictions = model["classifier"].predict(
        model["vectorizer"].transform(X_test)
    )

    predictions = (
        pd.Series(predictions)
        .astype(str)
        .str.lower()
    )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    labels = sorted(
        set(y_test) | set(predictions)
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    print("\n====================================")
    print("SENTIMENT TRAINING + EVALUATION")
    print("====================================")

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Macro Precision : {precision:.4f}"
    )

    print(
        f"Macro Recall    : {recall:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1     : {weighted_f1:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        f"Metrics saved to: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()