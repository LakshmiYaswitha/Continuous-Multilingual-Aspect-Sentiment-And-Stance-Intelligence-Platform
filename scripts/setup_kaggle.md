# Kaggle training setup

## 1. Install the Kaggle API

```powershell
pip install kaggle
```

## 2. Configure Kaggle credentials

Create an API token from your Kaggle account settings and place `kaggle.json` at:

```text
C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json
```

Do **not** commit this file to GitHub. The repository `.gitignore` excludes it and the `data/` directory.

## 3. Train the O2 multilingual/reference sentiment model

```powershell
python scripts/kaggle_train.py --limit-per-dataset 10000
```

This downloads and normalizes:

- Amazon US Customer Reviews — `cynthiarempel/amazon-us-customer-reviews-dataset`
- IIT Patna Hindi Movie Reviews — `warcoder/iit-patna-movie-reviews-hindi`
- TamilSentiMix NLP — `joebeachcapital/tamilsentimix`

The script creates:

```text
data/processed/kaggle_sentiment_train.csv
models/kaggle_sentiment_reference.joblib
models/kaggle_sentiment_metrics.json
```

The Amazon dataset is large, so the script samples `--limit-per-dataset` rows instead of copying the full corpus into GitHub.

## 4. Train the ABSA reference component

The Amazon/Hindi/Tamil sentiment datasets do not contain aspect-term annotations, so they cannot by themselves train aspect extraction. For the ABSA reference experiment use the Kaggle SemEval-2014 Task 4 dataset:

```powershell
python scripts/train_absa_reference.py --limit 6000
```

This creates:

```text
data/processed/semeval_absa_pairs.csv
models/kaggle_absa_polarity_reference.joblib
models/kaggle_absa_metrics.json
```

This distinction is intentional: overall sentiment training and aspect-level supervision are different tasks.

## 5. Connect the trained model to the website

The Streamlit app will prefer `models/kaggle_sentiment_reference.joblib` when it exists. If it is absent, it uses the development/reference model.

```powershell
streamlit run app.py
```
