# CMASSIP

**Continuous Multilingual Aspect Sentiment and Stance Intelligence Platform**

CMASSIP is the actual end-user platform: users enter text or upload documents/datasets and receive multilingual aspect sentiment, stance, dimensional-affect and evidence outputs. The capstone objectives are implemented inside the system rather than exposed as process pages.

## User workflow

`Text/PDF/DOCX/CSV/XLSX -> extraction -> language detection -> ABSA -> target-aware stance -> valence/arousal -> evidence -> results`

## Kaggle training is included

The repository now contains an executable Kaggle download + training pipeline. The datasets are **not bundled in GitHub** because some are large and/or subject to dataset terms. The code downloads them locally, normalizes their labels, creates leakage-resistant train/test splits, trains the reference model, evaluates it, and saves the model for the website.

### Sentiment/reference training datasets

- **Amazon US Customer Reviews** — `cynthiarempel/amazon-us-customer-reviews-dataset` (the source contains review text and 1–5 star ratings).
- **IIT Patna Hindi Movie Reviews** — `warcoder/iit-patna-movie-reviews-hindi`.
- **TamilSentiMix NLP** — `joebeachcapital/tamilsentimix`, a Tamil-English code-switched sentiment corpus.

The Amazon source is very large, so the training script samples a configurable number of rows per dataset.

### ABSA-specific training/reference dataset

Overall sentiment datasets do **not** contain aspect-term annotations. For that reason, the repository separately supports the Kaggle **SemEval-2014 Task 4 ABSA** dataset (`charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis`) for aspect-level polarity supervision.

This distinction is deliberate: it prevents us from falsely claiming that a document-level sentiment dataset trained an ABSA model.

## 1. Environment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements-transformers.txt
```

## 2. Kaggle credentials

The current Kaggle CLI supports the API-token method. In Windows CMD, set the token for the current terminal session:

```cmd
set KAGGLE_API_TOKEN=YOUR_TOKEN_HERE
python -m kaggle --version
python -m kaggle competitions list
```

Do **not** put the token in the project files or commit it to GitHub. If the token is exposed, revoke it in Kaggle and create a replacement.

The training script reuses already-downloaded files under `data/raw`, so rerunning it does not intentionally download a dataset again when an extracted tabular file is already present.

## 3. Train the Kaggle sentiment reference model

```powershell
python scripts\kaggle_train.py --limit-per-dataset 10000
```

Outputs:

```text
data/processed/kaggle_sentiment_train.csv
data/processed/kaggle_manifest.json
models/kaggle_sentiment_reference.joblib
models/kaggle_sentiment_metrics.json
```

The saved model is automatically picked up by the Streamlit application.

## 4. Train the ABSA polarity reference component

```powershell
python scripts/train_absa_reference.py --limit 6000
```

Outputs:

```text
data/processed/semeval_absa_pairs.csv
models/kaggle_absa_polarity_reference.joblib
models/kaggle_absa_metrics.json
```

The current production ABSA UI still uses the multilingual end-to-end transformer for inference. This Kaggle-trained ABSA polarity model is retained as the reproducible O2 reference experiment and can be integrated into the final O3 comparison once the joint model is implemented.

## 5. Run the website

```powershell
streamlit run app.py
```

## Research boundary

Valence/arousal is currently a reference estimator. A trained dimensional-affect model requires a validated corpus with continuous valence/arousal labels. It will be added during the later training stage rather than being represented as trained when it is not.
