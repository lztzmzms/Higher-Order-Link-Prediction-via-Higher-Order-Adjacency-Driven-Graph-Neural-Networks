"""AP and AP/prevalence Lift from actual held-out predictions."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

FIELDS = ('dataset', 'model', 'seed', 'AP', 'Lift')

def evaluate(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or labels.shape != predictions.shape or not np.isfinite(predictions).all():
        raise ValueError('Invalid labels/predictions')
    if not np.isin(labels, [0,1]).all() or not 0 < labels.sum() < len(labels):
        raise ValueError('Binary candidate set must contain both classes')
    ap = float(average_precision_score(labels, predictions))
    return dict(AP=ap, Lift=ap/float(labels.mean()))

def save_results(folder, row):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'seed_results.csv'
    rows = []
    if path.exists():
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames) != FIELDS:
                raise ValueError('Unexpected result schema')
            rows = [dict(dataset=r['dataset'], model=r['model'], seed=int(r['seed']),
                         AP=float(r['AP']), Lift=float(r['Lift'])) for r in reader]
    key = lambda r:(r['dataset'], r['model'], int(r['seed']))
    rows = [r for r in rows if key(r) != key(row)] + [row]
    rows.sort(key=key)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    summaries = []
    for dataset,model in sorted({(r['dataset'],r['model']) for r in rows}):
        group = [r for r in rows if r['dataset']==dataset and r['model']==model]
        seeds = [int(r['seed']) for r in group]
        from data.prepare import config
        expected_seeds = sorted(config(dataset)['models'][model]['seeds'])
        if seeds != expected_seeds:
            continue
        summaries.append(dict(dataset=dataset, model=model, seeds=seeds,
            AP_mean=float(np.mean([r['AP'] for r in group])),
            AP_std=float(np.std([r['AP'] for r in group], ddof=1)),
            Lift_mean=float(np.mean([r['Lift'] for r in group]))))
    target = folder/'summary.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(target)
    return next((r for r in summaries if r['dataset']==row['dataset'] and r['model']==row['model']), None)
