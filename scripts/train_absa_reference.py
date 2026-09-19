"""Prepare and train an ABSA sentiment classifier from the Kaggle SemEval-2014 Task 4 dataset.

This is intentionally separate from the multilingual sentiment datasets because those
only contain document/sentence sentiment labels and do not contain aspect annotations.

Download source: charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis
Run:
  python scripts/train_absa_reference.py --limit 6000
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW=ROOT/'data'/'raw'/'semeval_absa'; MODELS=ROOT/'models'; RAW.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
SLUG='charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=6000); args=ap.parse_args()
    if not any(RAW.rglob('*.xml')):
        subprocess.run([sys.executable,'-m','kaggle','datasets','download','-d',SLUG,'-p',str(RAW),'--unzip'],check=True)
    records=[]
    for xml in RAW.rglob('*.xml'):
        try: root=ET.parse(xml).getroot()
        except Exception: continue
        for sent in root.iter('sentence'):
            text=(sent.findtext('text') or '').strip(); opinions=sent.find('aspectTerms')
            if not text or opinions is None: continue
            for op in opinions.findall('aspectTerm'):
                term=(op.attrib.get('term') or '').strip(); pol=(op.attrib.get('polarity') or '').strip().lower()
                if term and pol in {'positive','negative','neutral','conflict'}:
                    records.append({'text':text,'aspect':term,'polarity':pol})
    df=pd.DataFrame(records).drop_duplicates()
    if df.empty: raise SystemExit('No SemEval XML aspect annotations found.')
    if len(df)>args.limit: df=df.sample(args.limit,random_state=42)
    df.to_csv(ROOT/'data'/'processed'/'semeval_absa_pairs.csv',index=False)
    Xtr,Xte,ytr,yte=train_test_split(df[['text','aspect']],df.polarity,test_size=.2,random_state=42,stratify=df.polarity)
    # Lightweight, reproducible aspect-context classifier for the O2 ABSA reference.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    def fe(frame): return (frame.text + ' [ASPECT] ' + frame.aspect).tolist()
    pipe=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000)),('clf',LogisticRegression(max_iter=1000))])
    pipe.fit(fe(Xtr),ytr); pred=pipe.predict(fe(Xte))
    metrics={'accuracy':float(accuracy_score(yte,pred)),'macro_f1':float(f1_score(yte,pred,average='macro')),'rows':len(df)}
    import joblib; joblib.dump(pipe,MODELS/'kaggle_absa_polarity_reference.joblib')
    (MODELS/'kaggle_absa_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2)); print('Saved:',MODELS/'kaggle_absa_polarity_reference.joblib')
if __name__=='__main__': main()
