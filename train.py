"""Train the reproducible sentiment reference model from a Kaggle-downloaded CSV/TSV.
Usage: python train.py --file data/train.csv --text-col review --label-col sentiment --limit 30000
"""
import argparse, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from src.model_store import train_sentiment

p=argparse.ArgumentParser(); p.add_argument('--file',required=True); p.add_argument('--text-col',required=True); p.add_argument('--label-col',required=True); p.add_argument('--limit',type=int,default=30000); a=p.parse_args()
df=pd.read_csv(a.file,sep=None,engine='python').dropna(subset=[a.text_col,a.label_col]).sample(frac=1,random_state=42).head(a.limit)
Xtr,Xte,ytr,yte=train_test_split(df[a.text_col].astype(str),df[a.label_col].astype(str),test_size=.2,random_state=42,stratify=df[a.label_col].astype(str))
m=train_sentiment(Xtr,ytr); pred=m['classifier'].predict(m['vectorizer'].transform(Xte)); print({'accuracy':accuracy_score(yte,pred),'macro_f1':f1_score(yte,pred,average='macro')})
