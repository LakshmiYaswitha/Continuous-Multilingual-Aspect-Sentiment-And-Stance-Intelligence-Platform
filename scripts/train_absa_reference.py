"""Prepare and train an ABSA sentiment classifier from the Kaggle
SemEval-2014 Task 4 dataset.

This is intentionally separate from the multilingual sentiment datasets because
those only contain document/sentence sentiment labels and do not contain aspect
annotations.

Download source:
charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis

Run:
    python scripts/train_absa_reference.py --limit 6000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw" / "semeval_absa"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SLUG = "charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis"


def main():

    # ============================================================
    # ARGUMENTS
    # ============================================================

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--limit",
        type=int,
        default=6000,
        help="Maximum number of aspect-level records to use.",
    )

    args = ap.parse_args()

    # ============================================================
    # DOWNLOAD DATASET
    # ============================================================

    if not any(RAW.rglob("*.xml")):

        print("Downloading SemEval-2014 ABSA dataset...")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                SLUG,
                "-p",
                str(RAW),
                "--unzip",
            ],
            check=True,
        )

    # ============================================================
    # READ XML ASPECT ANNOTATIONS
    # ============================================================

    records = []

    for xml in RAW.rglob("*.xml"):

        try:
            root = ET.parse(xml).getroot()

        except Exception:
            continue

        for sent in root.iter("sentence"):

            text = (sent.findtext("text") or "").strip()

            opinions = sent.find("aspectTerms")

            if not text or opinions is None:
                continue

            for op in opinions.findall("aspectTerm"):

                term = (op.attrib.get("term") or "").strip()

                polarity = (
                    op.attrib.get("polarity") or ""
                ).strip().lower()

                if (
                    term
                    and polarity
                    in {
                        "positive",
                        "negative",
                        "neutral",
                        "conflict",
                    }
                ):

                    records.append(
                        {
                            "text": text,
                            "aspect": term,
                            "polarity": polarity,
                        }
                    )

    # ============================================================
    # PREPARE DATA
    # ============================================================

    df = pd.DataFrame(records).drop_duplicates()

    if df.empty:
        raise SystemExit(
            "No SemEval XML aspect annotations found."
        )

    if len(df) > args.limit:

        df = df.sample(
            args.limit,
            random_state=42,
        )

    df = df.reset_index(drop=True)

    df.to_csv(
        PROC / "semeval_absa_pairs.csv",
        index=False,
    )

    print()
    print("ABSA DATASET")
    print("-----------------------------")
    print(f"Total rows: {len(df):,}")
    print()
    print("Polarity distribution:")
    print(df["polarity"].value_counts())
    print()

    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================

    Xtr, Xte, ytr, yte = train_test_split(
        df[["text", "aspect"]],
        df["polarity"],
        test_size=0.2,
        random_state=42,
        stratify=df["polarity"],
    )

    print(
        f"Training rows: {len(Xtr):,}"
    )

    print(
        f"Testing rows:  {len(Xte):,}"
    )

    # ============================================================
    # FEATURE EXTRACTION + CLASSIFIER
    # ============================================================

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    def fe(frame):

        return (
            frame["text"]
            + " [ASPECT] "
            + frame["aspect"]
        ).tolist()

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    # ============================================================
    # TRAIN
    # ============================================================

    print()
    print("Training ABSA reference classifier...")

    pipe.fit(
        fe(Xtr),
        ytr,
    )

    # ============================================================
    # PREDICTION
    # ============================================================

    pred = pipe.predict(
        fe(Xte)
    )

    # ============================================================
    # EVALUATION
    # ============================================================

    labels = sorted(
        set(yte) | set(pred)
    )

    accuracy = accuracy_score(
        yte,
        pred,
    )

    macro_precision = precision_score(
        yte,
        pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        yte,
        pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        yte,
        pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        yte,
        pred,
        average="weighted",
        zero_division=0,
    )

    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    report = classification_report(
        yte,
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    matrix = confusion_matrix(
        yte,
        pred,
        labels=labels,
    )

    # ============================================================
    # METRICS JSON
    # ============================================================

    metrics = {

        "task": "Aspect-Based Sentiment Analysis",

        "dataset": (
            "SemEval-2014 Task 4 "
            "Aspect-Based Sentiment Analysis"
        ),

        "dataset_source": SLUG,

        "rows": int(len(df)),

        "train_rows": int(len(Xtr)),

        "test_rows": int(len(Xte)),

        "test_size": 0.20,

        "random_state": 42,

        "labels": labels,

        "label_distribution": {
            str(k): int(v)
            for k, v in df["polarity"]
            .value_counts()
            .items()
        },

        "accuracy": float(
            accuracy
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "classification_report": report,

        "confusion_matrix": (
            matrix.tolist()
        ),

        "model": (
            "TF-IDF (unigram + bigram) "
            "+ Logistic Regression"
        ),
    }

    # ============================================================
    # SAVE MODEL
    # ============================================================

    import joblib

    model_path = (
        MODELS
        / "kaggle_absa_polarity_reference.joblib"
    )

    joblib.dump(
        pipe,
        model_path,
    )

    # ============================================================
    # SAVE METRICS
    # ============================================================

    metrics_path = (
        MODELS
        / "kaggle_absa_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print()
    print("ABSA EVALUATION RESULTS")
    print("=============================")

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : {macro_recall:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1       : {weighted_f1:.4f}"
    )

    print()
    print("Classification Report")
    print("-----------------------------")

    print(
        classification_report(
            yte,
            pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion Matrix")
    print("-----------------------------")

    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )
    )

    print()
    print(
        "Model saved:",
        model_path,
    )

    print(
        "Metrics saved:",
        metrics_path,
    )


if __name__ == "__main__":
    main()