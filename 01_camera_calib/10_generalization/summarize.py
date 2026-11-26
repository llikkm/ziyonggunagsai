from collections import Counter
import numpy as np


def stats(values):
    return dict(zip(('median','p95','max'),np.quantile(values,[.5,.95,1]).tolist())) if len(values) else None


def summarize(rows,seed):
    result={}
    for mode in ['ideal','jitter','pooled']:
        rng=np.random.default_rng(seed+191)
        group=[row for row in rows if mode=='pooled' or row['mode']==mode]
        scores=np.zeros(10000)
        classes={}
        for category,weight in zip('ABC',[.2,.3,.5]):
            cases=[row for row in group if row['category']==category]
            errors=np.array([row['abs_pct'] if not row['failure'] else np.nan for row in cases])
            classes[category]=dict(n=len(cases),failures=int(np.isnan(errors).sum()),abs_pct=stats(errors[np.isfinite(errors)]))
            scores+=weight*rng.choice(errors,len(scores))/100
        result[mode]=dict(classes=classes,unrounded_P=stats(scores[np.isfinite(scores)]),invalid_draws=int((~np.isfinite(scores)).sum()),
                          draws=len(scores),dominant=dict(Counter(row['dominant'] for row in group)))
    return result
