from functools import lru_cache

@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline
    return pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def stance(text,target=None):
    if not target:
        return {'target':'Not specified','label':'Not evaluated','confidence':0.0,'method':'Provide a target to run target-aware stance analysis.'}
    try:
        clf=_classifier()
        labels=[f'supports {target}',f'opposes {target}',f'neutral toward {target}']
        r=clf(text, labels)
        raw=r['labels'][0]
        if raw==labels[0]: label='Support'
        elif raw==labels[1]: label='Against'
        else: label='Neutral'
        return {'target':target,'label':label,'confidence':float(r['scores'][0]),'method':'BART-MNLI target-aware zero-shot NLI'}
    except Exception as exc:
        return {'target':target,'label':'Unavailable','confidence':0.0,'method':f'Stance model unavailable: {type(exc).__name__}'}
