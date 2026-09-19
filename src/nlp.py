import re

POS={'good','great','excellent','amazing','awesome','love','loved','fast','easy','beautiful','helpful','perfect','best','happy','recommend','wonderful','fantastic','superb','like','durable','smooth','clear','comfortable'}
NEG={'bad','poor','terrible','awful','hate','hated','slow','difficult','ugly','worst','sad','broken','disappointing','disappointed','drains','expensive','problem','problems','horrible','annoying','useless','weak'}
NEGATORS={'not','never','no','dont',"don't",'didnt',"didn't",'cannot','cant',"can't",'wont',"won't"}
DOMAIN=['camera','battery','display','screen','price','design','performance','quality','service','delivery','product','phone','mobile','software','support','policy','government','movie','acting','story','music','sound','food','hotel','app','brand']

def clean(text):
    return re.sub(r'\s+',' ',str(text)).strip()

def language(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        if re.search(r'[\u0900-\u097F]',text): return 'hi'
        if re.search(r'[\u0B80-\u0BFF]',text): return 'ta'
        return 'en'

def sentiment_fallback(text):
    toks=re.findall(r"[A-Za-z']+",text.lower()); score=0
    for i,t in enumerate(toks):
        if t in POS: score += -1 if i and toks[i-1] in NEGATORS else 1
        if t in NEG: score += 1 if i and toks[i-1] in NEGATORS else -1
    if score>0: return 'Positive', min(.99,.55+.1*score)
    if score<0: return 'Negative', min(.99,.55+.1*abs(score))
    return 'Neutral', .50

def extract_aspects(text):
    low=text.lower(); found=[]
    for w in DOMAIN:
        if re.search(r'\b'+re.escape(w)+r'\b',low): found.append(w)
    return list(dict.fromkeys(found))[:12]
