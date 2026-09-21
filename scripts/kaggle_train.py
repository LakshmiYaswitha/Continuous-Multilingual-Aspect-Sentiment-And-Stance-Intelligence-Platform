"""
Download/reuse Kaggle datasets and train the CMASSIP O2 sentiment reference model.

Usage (Windows CMD):

    set KAGGLE_API_TOKEN=YOUR_TOKEN
    python scripts\\kaggle_train.py --limit-per-dataset 10000

The script reuses files already present in data/raw, so a rerun does not
redownload an already extracted dataset.

Overall sentiment datasets are NOT ABSA datasets.
ABSA reference training is handled separately by train_absa_reference.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
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


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

for folder in (RAW, PROC, MODELS):
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# KAGGLE DATASETS
# ============================================================

DATASETS = {
    "amazon": {
        "slug": "cynthiarempel/amazon-us-customer-reviews-dataset",
        "file": "amazon_reviews_us_Electronics_v1_00.tsv",
    },

    "hindi": {
        "slug": "warcoder/iit-patna-movie-reviews-hindi",
        "file": None,
    },

    "tamil": {
        "slug": "joebeachcapital/tamilsentimix",
        "file": None,
    },
}


# ============================================================
# FILE HELPERS
# ============================================================

def locate_files(folder: Path):
    """
    Find usable text/tabular files inside a dataset folder.
    """

    return [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {
            ".csv",
            ".tsv",
            ".txt",
        }
    ]


def extract_local_zips(folder: Path):
    """
    Extract downloaded Kaggle ZIP files if they are still present.
    """

    changed = False

    for zip_path in folder.rglob("*.zip"):

        try:

            with zipfile.ZipFile(zip_path) as archive:

                archive.extractall(folder)

            changed = True

            print(
                f"[INFO] Extracted {zip_path.name}"
            )

        except Exception as exc:

            print(
                f"[WARN] Could not extract "
                f"{zip_path.name}: {exc}"
            )

    return changed


def download_if_needed(
    slug: str,
    folder: Path,
    file_name: str | None = None,
):
    """
    Download a Kaggle dataset only if usable files are not already present.
    """

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    extract_local_zips(folder)

    existing_files = locate_files(folder)

    if existing_files:

        print(
            f"[INFO] Reusing existing dataset files "
            f"in {folder}"
        )

        return

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(folder),
        "--unzip",
    ]

    if file_name:

        command += [
            "-f",
            file_name,
        ]

    print(
        "\n$",
        " ".join(command)
    )

    subprocess.run(
        command,
        check=True
    )

    extract_local_zips(folder)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value):
    """
    Normalize different dataset sentiment labels into:

        positive
        negative
        neutral
    """

    text = str(value).strip().lower()

    if text in {
        "positive",
        "pos",
        "p",
        "1",
        "1.0",
        "4",
        "5",
        "+1",
    }:
        return "positive"

    if text in {
        "negative",
        "neg",
        "n",
        "-1",
        "-1.0",
        "0",
        "0.0",
        "2",
        "1 star",
        "2 stars",
    }:
        return "negative"

    if text in {
        "neutral",
        "neu",
        "0.5",
        "3",
        "3.0",
    }:
        return "neutral"

    if "positive" in text:
        return "positive"

    if "negative" in text:
        return "negative"

    if "neutral" in text:
        return "neutral"

    return None


# ============================================================
# AMAZON
# ============================================================

def sample_amazon(
    path: Path,
    limit: int,
):
    """
    Read the large Amazon Electronics TSV in chunks.

    Only review text and star rating are retained.
    """

    header = pd.read_csv(
        path,
        sep="\t",
        nrows=0,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    columns = list(
        header.columns
    )

    text_col = next(
        (
            column
            for column in columns
            if str(column).lower()
            in {
                "review_body",
                "reviewtext",
                "review",
            }
        ),
        None,
    )

    label_col = next(
        (
            column
            for column in columns
            if str(column).lower()
            in {
                "star_rating",
                "overall",
                "rating",
            }
        ),
        None,
    )

    if text_col is None or label_col is None:

        raise ValueError(
            f"Amazon columns not found: "
            f"{columns[:20]}"
        )

    chunks = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            text_col,
            label_col,
        ],
        chunksize=50000,
        encoding="utf-8",
        on_bad_lines="skip",
    ):

        chunk = chunk.dropna(
            subset=[
                text_col,
                label_col,
            ]
        )

        chunk["label"] = (
            chunk[label_col]
            .map(normalize_label)
        )

        chunk = chunk[
            chunk["label"].notna()
        ]

        chunk = chunk.rename(
            columns={
                text_col: "text"
            }
        )[
            [
                "text",
                "label",
            ]
        ]

        if not chunk.empty:

            sample_size = min(
                len(chunk),
                max(
                    1,
                    limit // 10
                ),
            )

            chunks.append(
                chunk.sample(
                    sample_size,
                    random_state=42,
                )
            )

        if sum(
            len(part)
            for part in chunks
        ) >= limit * 2:

            break

    if not chunks:

        raise ValueError(
            "No usable Amazon rows found."
        )

    output = (
        pd.concat(
            chunks,
            ignore_index=True,
        )
        .drop_duplicates(
            "text"
        )
    )

    if len(output) > limit:

        output = output.sample(
            limit,
            random_state=42,
        )

    output["language_source"] = "amazon"

    return output


# ============================================================
# HINDI
# ============================================================

def read_hindi(path: Path):
    """
    Read IIT Patna Hindi movie reviews.

    The commonly downloaded mirror stores
    label,text without a conventional header.
    """

    df = pd.read_csv(
        path,
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    if df.shape[1] < 2:

        raise ValueError(
            f"Hindi file has unexpected "
            f"shape {df.shape}"
        )

    candidates = []

    for label_column, text_column in [
        (0, 1),
        (1, 0),
    ]:

        labels = (
            df[label_column]
            .map(normalize_label)
        )

        score = labels.notna().mean()

        candidates.append(
            (
                score,
                label_column,
                text_column,
            )
        )

    _, label_column, text_column = max(
        candidates
    )

    output = pd.DataFrame(
        {
            "text": df[
                text_column
            ].astype(str),

            "label": df[
                label_column
            ].map(
                normalize_label
            ),
        }
    )

    output = output[
        output["label"].notna()
    ]

    output = output[
        output["text"]
        .str.strip()
        .ne("")
    ]

    output["language_source"] = "hindi"

    return output


# ============================================================
# GENERIC LABELED DATASET
# ============================================================

def generic_labeled_file(folder: Path):
    """
    Try to locate a generic text/label CSV, TSV or TXT file.
    """

    for path in locate_files(folder):

        try:

            separator = (
                "\t"
                if path.suffix.lower() == ".tsv"
                else ","
            )

            df = pd.read_csv(
                path,
                sep=separator,
                encoding="utf-8",
                on_bad_lines="skip",
            )

        except Exception:

            continue

        columns = {
            str(column).lower(): column
            for column in df.columns
        }

        text_column = next(
            (
                columns[column]
                for column in [
                    "text",
                    "comment",
                    "review",
                    "sentence",
                ]
                if column in columns
            ),
            None,
        )

        label_column = next(
            (
                columns[column]
                for column in [
                    "label",
                    "sentiment",
                    "polarity",
                    "category",
                ]
                if column in columns
            ),
            None,
        )

        if (
            text_column is not None
            and label_column is not None
        ):

            output = pd.DataFrame(
                {
                    "text": df[
                        text_column
                    ].astype(str),

                    "label": df[
                        label_column
                    ].map(
                        normalize_label
                    ),
                }
            )

            output = output[
                output["label"].notna()
            ]

            output = output[
                output["text"]
                .str.strip()
                .ne("")
            ]

            return path, output

    return None, None


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
):
    """
    Calculate the complete sentiment evaluation.
    """

    labels = sorted(
        set(y_true)
        | set(y_pred)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    classification = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
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

        "labels": labels,

        "confusion_matrix": (
            matrix.tolist()
        ),

        "classification_report": (
            classification
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--skip-amazon",
        action="store_true",
    )

    parser.add_argument(
        "--skip-hindi",
        action="store_true",
    )

    parser.add_argument(
        "--skip-tamil",
        action="store_true",
    )

    args = parser.parse_args()

    rows = []

    manifest = {}

    # ========================================================
    # PREPARE DATASETS
    # ========================================================

    for name, config in DATASETS.items():

        if getattr(
            args,
            f"skip_{name}",
        ):

            print(
                f"[INFO] Skipping {name}"
            )

            continue

        folder = RAW / name

        try:

            print(
                f"\n{'=' * 60}"
            )

            print(
                f"Preparing dataset: {name}"
            )

            print(
                f"{'=' * 60}"
            )

            download_if_needed(
                config["slug"],
                folder,
                config["file"],
            )

            extract_local_zips(
                folder
            )

            files = locate_files(
                folder
            )

            if not files:

                print(
                    f"[WARN] No tabular file "
                    f"found for {name}; skipping."
                )

                continue

            # ------------------------------------------------
            # AMAZON
            # ------------------------------------------------

            if name == "amazon":

                chosen = next(
                    (
                        path
                        for path in files
                        if path.name
                        == config["file"]
                    ),
                    max(
                        files,
                        key=lambda path:
                        path.stat().st_size,
                    ),
                )

                prepared = sample_amazon(
                    chosen,
                    args.limit_per_dataset,
                )

            # ------------------------------------------------
            # HINDI
            # ------------------------------------------------

            elif name == "hindi":

                chosen = next(
                    (
                        path
                        for path in files
                        if path.name
                        == "hi-train.csv"
                    ),
                    files[0],
                )

                prepared = read_hindi(
                    chosen
                )

                if len(prepared) > args.limit_per_dataset:

                    prepared = prepared.sample(
                        args.limit_per_dataset,
                        random_state=42,
                    )

            # ------------------------------------------------
            # TAMIL
            # ------------------------------------------------

            else:

                chosen, prepared = (
                    generic_labeled_file(
                        folder
                    )
                )

                if prepared is None:

                    print(
                        "[WARN] The downloaded "
                        "TamilSentiMix files do not "
                        "expose usable text/label "
                        "columns in this copy."
                    )

                    print(
                        "[INFO] Tamil is being skipped "
                        "rather than training on "
                        "unrelated numeric data."
                    )

                    continue

                if len(prepared) > args.limit_per_dataset:

                    prepared = prepared.sample(
                        args.limit_per_dataset,
                        random_state=42,
                    )

                prepared[
                    "language_source"
                ] = "tamil"

            # ------------------------------------------------
            # STORE DATASET
            # ------------------------------------------------

            rows.append(
                prepared
            )

            manifest[name] = {
                "file": str(chosen),
                "rows": int(
                    len(prepared)
                ),
                "labels": (
                    prepared[
                        "label"
                    ]
                    .value_counts()
                    .to_dict()
                ),
            }

            print(
                f"[{name}] prepared "
                f"{len(prepared):,} rows"
            )

            print(
                f"[{name}] source: {chosen}"
            )

            print(
                f"[{name}] labels:"
            )

            print(
                prepared[
                    "label"
                ].value_counts()
            )

        except Exception as exc:

            print(
                f"[WARN] Could not prepare "
                f"{name}: {exc}"
            )

    # ========================================================
    # CHECK DATA
    # ========================================================

    if not rows:

        raise SystemExit(
            "No datasets were successfully "
            "prepared. Check downloaded files "
            "and Kaggle credentials."
        )

    # ========================================================
    # COMBINE
    # ========================================================

    data = (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .drop_duplicates(
            "text"
        )
    )

    # ========================================================
    # REMOVE CLASSES WITH < 2 EXAMPLES
    # ========================================================

    counts = (
        data["label"]
        .value_counts()
    )

    keep = counts[
        counts >= 2
    ].index

    data = data[
        data["label"].isin(
            keep
        )
    ].reset_index(
        drop=True
    )

    print(
        f"\nCombined dataset: "
        f"{len(data):,} rows"
    )

    print(
        "\nCombined label distribution:"
    )

    print(
        data[
            "label"
        ].value_counts()
    )

    print(
        "\nLanguage/source distribution:"
    )

    print(
        data[
            "language_source"
        ].value_counts()
    )

    # ========================================================
    # SAVE PROCESSED DATA
    # ========================================================

    processed_file = (
        PROC
        / "kaggle_sentiment_train.csv"
    )

    data.to_csv(
        processed_file,
        index=False,
    )

    manifest["combined"] = {
        "rows": int(
            len(data)
        ),

        "labels": (
            data[
                "label"
            ]
            .value_counts()
            .to_dict()
        ),

        "language_sources": (
            data[
                "language_source"
            ]
            .value_counts()
            .to_dict()
        ),
    }

    manifest_file = (
        PROC
        / "kaggle_manifest.json"
    )

    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            data["text"],
            data["label"],
            test_size=0.20,
            random_state=42,
            stratify=data["label"],
        )
    )

    print(
        "\n======================================"
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "======================================"
    )

    print(
        f"Training rows : {len(X_train):,}"
    )

    print(
        f"Testing rows  : {len(X_test):,}"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    from src.model_store import train_sentiment

    print(
        "\nTraining sentiment reference model..."
    )

    bundle = train_sentiment(
        X_train,
        y_train,
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = (
        bundle["classifier"]
        .predict(
            bundle[
                "vectorizer"
            ].transform(
                X_test
            )
        )
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluation = evaluate_model(
        y_test,
        predictions,
    )

    # ========================================================
    # COMPLETE METRICS
    # ========================================================

    metrics = {
        **evaluation,

        "train_rows": int(
            len(X_train)
        ),

        "test_rows": int(
            len(X_test)
        ),

        "datasets": manifest,
    }

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODELS
        / "kaggle_sentiment_reference.joblib"
    )

    joblib.dump(
        bundle,
        model_path,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = (
        MODELS
        / "kaggle_sentiment_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CMASSIP SENTIMENT EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Training rows      : "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows       : "
        f"{len(X_test):,}"
    )

    print(
        f"Accuracy           : "
        f"{evaluation['accuracy']:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{evaluation['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{evaluation['macro_recall']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{evaluation['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{evaluation['weighted_f1']:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=evaluation["labels"],
            zero_division=0,
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "Confusion Matrix:"
    )

    print(
        pd.DataFrame(
            evaluation[
                "confusion_matrix"
            ],
            index=evaluation[
                "labels"
            ],
            columns=evaluation[
                "labels"
            ],
        )
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 60
    )

    print(
        f"Processed data : "
        f"{processed_file}"
    )

    print(
        f"Manifest       : "
        f"{manifest_file}"
    )

    print(
        f"Model          : "
        f"{model_path}"
    )

    print(
        f"Metrics        : "
        f"{metrics_path}"
    )

    print(
        "\nTRAINING + EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()