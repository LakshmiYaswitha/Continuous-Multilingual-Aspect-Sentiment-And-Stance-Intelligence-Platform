"""Evaluate the saved reference model on a labelled CSV/TSV."""
import argparse,pandas as pd,joblib
from sklearn.metrics import accuracy_score,f1_score
p=argparse.ArgumentParser(); p.add_argument('--file',required=True); p.add_argument('--text-col',required=True); p.add_argument('--label-col',required=True); a=p.parse_args()
df=pd.read_csv(a.file,sep=None,engine='python').dropna(subset=[a.text_col,a.label_col]); m=joblib.load('models/sentiment.joblib'); pred=m['classifier'].predict(m['vectorizer'].transform(df[a.text_col].astype(str))); print('accuracy',accuracy_score(df[a.label_col].astype(str),pred)); print('macro_f1',f1_score(df[a.label_col].astype(str),pred,average='macro'))
