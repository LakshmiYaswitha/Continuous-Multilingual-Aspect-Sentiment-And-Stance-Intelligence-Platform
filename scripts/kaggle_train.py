"""Download/reuse Kaggle datasets and train the CMASSIP O2 sentiment reference model.

Usage (Windows CMD):
  set KAGGLE_API_TOKEN=YOUR_TOKEN
  python scripts\\kaggle_train.py --limit-per-dataset 10000

The script reuses files already present in data/raw, so a rerun does not redownload
an already extracted dataset. Overall sentiment datasets are NOT ABSA datasets;
ABSA reference training is handled separately by train_absa_reference.py.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
for p in (RAW, PROC, MODELS): p.mkdir(parents=True, exist_ok=True)

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


def locate_files(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}]

def extract_local_zips(folder: Path):
    """Extract downloaded Kaggle zip files when the current CLI leaves them zipped."""
    import zipfile
    changed = False
    for zp in folder.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as z:
                z.extractall(folder)
            changed = True
        except Exception as exc:
            print(f"[WARN] Could not extract {zp.name}: {exc}")
    return changed

def download_if_needed(slug: str, folder: Path, file_name: str | None = None):
    folder.mkdir(parents=True, exist_ok=True)
    extract_local_zips(folder)
    if locate_files(folder):
        return
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug,
           "-p", str(folder), "--unzip"]
    if file_name:
        cmd += ["-f", file_name]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    extract_local_zips(folder)


def normalize_label(x):
    s = str(x).strip().lower()
    if s in {"positive", "pos", "p", "1", "1.0", "4", "5", "+1"}: return "positive"
    if s in {"negative", "neg", "n", "-1", "-1.0", "0", "0.0", "2", "1 star", "2 stars"}: return "negative"
    if s in {"neutral", "neu", "0.5", "3", "3.0"}: return "neutral"
    if "positive" in s: return "positive"
    if "negative" in s: return "negative"
    if "neutral" in s: return "neutral"
    return None


def sample_amazon(path: Path, limit: int):
    # The Electronics TSV is large. Read only useful columns in chunks and sample rows.
    header = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8", on_bad_lines="skip")
    cols = list(header.columns)
    text_col = next((c for c in cols if str(c).lower() in {"review_body", "reviewtext", "review"}), None)
    label_col = next((c for c in cols if str(c).lower() in {"star_rating", "overall", "rating"}), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Amazon columns not found: {cols[:20]}")
    chunks=[]
    for chunk in pd.read_csv(path, sep="\t", usecols=[text_col,label_col], chunksize=50000,
                             encoding="utf-8", on_bad_lines="skip"):
        chunk = chunk.dropna(subset=[text_col,label_col])
        chunk["label"] = chunk[label_col].map(normalize_label)
        chunk = chunk[chunk["label"].notna()]
        chunk = chunk.rename(columns={text_col:"text"})[["text","label"]]
        if not chunk.empty:
            chunks.append(chunk.sample(min(len(chunk), max(1, limit // 10)), random_state=42))
        if sum(len(x) for x in chunks) >= limit * 2:
            break
    if not chunks: raise ValueError("No usable Amazon rows found")
    out=pd.concat(chunks,ignore_index=True).drop_duplicates("text")
    if len(out)>limit: out=out.sample(limit,random_state=42)
    out["language_source"]="amazon"
    return out


def read_hindi(path: Path):
    # IIT Patna mirror commonly stores headerless rows: label,text. The first row
    # was previously mistaken for the CSV header by the generic parser.
    df = pd.read_csv(path, header=None, encoding="utf-8", on_bad_lines="skip")
    if df.shape[1] < 2:
        raise ValueError(f"Hindi file has unexpected shape {df.shape}")
    # Usually column 0 is the sentiment and column 1 is the review.
    candidates=[]
    for a,b in [(0,1),(1,0)]:
        labels=df[a].map(normalize_label)
        score=labels.notna().mean()
        candidates.append((score,a,b))
    _, lc, tc=max(candidates)
    out=pd.DataFrame({"text":df[tc].astype(str),"label":df[lc].map(normalize_label)})
    out=out[out.label.notna()]
    out=out[out.text.str.strip().ne("")]
    out["language_source"]="hindi"
    return out


def generic_labeled_file(folder: Path):
    for p in locate_files(folder):
        try:
            df=pd.read_csv(p, sep="\t" if p.suffix.lower()==".tsv" else ",", encoding="utf-8", on_bad_lines="skip")
        except Exception:
            continue
        cols={str(c).lower():c for c in df.columns}
        tc=next((cols[c] for c in ["text","comment","review","sentence"] if c in cols),None)
        lc=next((cols[c] for c in ["label","sentiment","polarity","category"] if c in cols),None)
        if tc is not None and lc is not None:
            out=pd.DataFrame({"text":df[tc].astype(str),"label":df[lc].map(normalize_label)})
            out=out[out.label.notna() & out.text.str.strip().ne("")]
            return p,out
    return None,None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit-per-dataset",type=int,default=10000)
    ap.add_argument("--skip-amazon",action="store_true")
    ap.add_argument("--skip-hindi",action="store_true")
    ap.add_argument("--skip-tamil",action="store_true")
    args=ap.parse_args()

    rows=[]; manifest={}
    for name,cfg in DATASETS.items():
        if getattr(args,f"skip_{name}"): continue
        folder=RAW/name
        try:
            download_if_needed(cfg["slug"],folder,cfg["file"])
            extract_local_zips(folder)
            files=locate_files(folder)
            if not files:
                print(f"[WARN] No tabular file found for {name}; skipping."); continue
            if name=="amazon":
                chosen=next((p for p in files if p.name==cfg["file"]),max(files,key=lambda p:p.stat().st_size))
                tmp=sample_amazon(chosen,args.limit_per_dataset)
            elif name=="hindi":
                # Prefer hi-train.csv; it is headerless label,text in this mirror.
                chosen=next((p for p in files if p.name=="hi-train.csv"),files[0])
                tmp=read_hindi(chosen)
                if len(tmp)>args.limit_per_dataset: tmp=tmp.sample(args.limit_per_dataset,random_state=42)
            else:
                chosen,tmp=generic_labeled_file(folder)
                if tmp is None:
                    print("[WARN] The downloaded TamilSentiMix files do not expose usable text/label columns in this copy; skipping Tamil rather than training on unrelated numeric data.")
                    continue
                if len(tmp)>args.limit_per_dataset: tmp=tmp.sample(args.limit_per_dataset,random_state=42)
                tmp["language_source"]="tamil"
            rows.append(tmp)
            manifest[name]={"file":str(chosen),"rows":len(tmp),"labels":tmp.label.value_counts().to_dict()}
            print(f"[{name}] prepared {len(tmp):,} rows from {chosen}")
        except Exception as e:
            print(f"[WARN] Could not prepare {name}: {e}")

    if not rows:
        raise SystemExit("No datasets were successfully prepared. Check downloaded files and Kaggle credentials.")
    data=pd.concat(rows,ignore_index=True).drop_duplicates("text")
    # Ensure every class has enough examples for a stratified split.
    counts=data.label.value_counts()
    keep=counts[counts>=2].index
    data=data[data.label.isin(keep)].reset_index(drop=True)
    data.to_csv(PROC/"kaggle_sentiment_train.csv",index=False)
    manifest["combined"]={"rows":len(data),"labels":data.label.value_counts().to_dict()}
    (PROC/"kaggle_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

    from src.model_store import train_sentiment
    Xtr,Xte,ytr,yte=train_test_split(data.text,data.label,test_size=.2,random_state=42,stratify=data.label)
    bundle=train_sentiment(Xtr,ytr)
    pred=bundle["classifier"].predict(bundle["vectorizer"].transform(Xte))
    metrics={"accuracy":float(accuracy_score(yte,pred)),"macro_f1":float(f1_score(yte,pred,average="macro")),"train_rows":len(Xtr),"test_rows":len(Xte),"datasets":manifest}
    import joblib
    joblib.dump(bundle,MODELS/"kaggle_sentiment_reference.joblib")
    (MODELS/"kaggle_sentiment_metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nTRAINING COMPLETE")
    print(json.dumps({k:v for k,v in metrics.items() if k!="datasets"},indent=2))
    print("Model:",MODELS/"kaggle_sentiment_reference.joblib")

if __name__=="__main__": main()
