from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'models'/'kaggle_sentiment_reference.joblib'

def train_sentiment(X,y):
    v=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)
    Xv=v.fit_transform(X)
    clf=LogisticRegression(max_iter=1000,class_weight='balanced')
    clf.fit(Xv,y)
    return {'vectorizer':v,'classifier':clf}

def load_sentiment_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def predict_sentiment(model,text):
    if model is None: return None, None
    X=model['vectorizer'].transform([text])
    label=model['classifier'].predict(X)[0]
    proba=getattr(model['classifier'],'predict_proba',lambda x:None)(X)
    conf=float(proba.max()) if proba is not None else None
    return str(label),conf
