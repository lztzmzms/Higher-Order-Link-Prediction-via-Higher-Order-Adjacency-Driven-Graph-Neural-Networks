                                                                     
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if os.name == 'nt' and not str(ROOT).startswith('\\\\?\\'):
    ROOT = Path('\\\\?\\' + str(ROOT))
BACKEND = ROOT / 'backend'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def score_file(p):
    import numpy as np
    from sklearn.metrics import average_precision_score
    z = np.load(p, allow_pickle=False)
    q, y, score = z['triplets'], z['y_true'], z['final']
    assert len(q) == len(y) == len(score) and np.isfinite(score).all()
    assert len(set(map(tuple, q))) == len(q)
    ap = float(average_precision_score(y, score))
    return dict(candidates=len(y), positives=int(y.sum()), AP=ap, Lift=ap/float(y.mean()))


def summarize(j):
    import numpy as np
    rows = []
    for seed in j['seeds']:
        d = BACKEND / 'results/jobs' / j['id'] / str(seed)
        if not (d/'RESULT.json').exists():
            continue
        r = read(d/'RESULT.json'); p = d/'PREDICTIONS.npz'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['prediction_sha256']
        m = score_file(p)
        assert abs(m['AP']-r['AP']) < 1e-10
        rows.append(dict(dataset=j['effective']['dataset'], model=j['effective']['model'], seed=seed, **m))
    out = ROOT/'results'/j['effective']['dataset']/j['effective']['model']
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(completed=len(rows), expected=len(j['seeds']), status='COMPLETE' if len(rows)==len(j['seeds']) else 'PARTIAL', layers=j['effective']['layers'], mode='full_pipeline', per_seed=rows)
    if rows:
        for key in ('AP','Lift'):
            summary[key+'_mean']=float(np.mean([r[key] for r in rows]))
            summary[key+'_sample_sd']=float(np.std([r[key] for r in rows],ddof=1)) if len(rows)>1 else None
        with (out/'per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description='Run a K=2 full prediction pipeline using dataset and model only.')
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--model',required=True,help='dual / Dual-Mapping or mlshol / MLSHoL')
    parser.add_argument('--check',action='store_true',help='Validate files, candidates and the resolved configuration without training.')
    parser.add_argument('--verify',action='store_true',help='Recompute metrics from current completed training outputs.')
    parser.add_argument('--smoke',action='store_true',help='One-epoch execution check; not a scientific result.')
    parser.add_argument('--device',default='auto',help='auto / cpu / cuda:0')
    parser.add_argument('--seeds',help='Optional comma-separated subset of the frozen seeds.')
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--_seed',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    aliases={'dual':'Dual-Mapping','dual-mapping':'Dual-Mapping','mlshol':'MLSHoL'}
    model=aliases.get(args.model.lower())
    if model is None: parser.error('model must be dual or mlshol')
    aliases_ds={'ndc':'NDC-substances','ndc-substances':'NDC-substances'}
    dataset=aliases_ds.get(args.dataset.lower(),args.dataset)
    plan=read(BACKEND/'PLAN.json')
    jobs=[j for j in plan['jobs'] if j['effective']['dataset']==dataset and j['effective']['model']==model]
    if len(jobs)!=1:parser.error('Unknown dataset. Available: '+', '.join(sorted({j['effective']['dataset'] for j in plan['jobs']})))
    j=jobs[0]
    if args.seeds:
        seeds=[int(x) for x in args.seeds.split(',')]
        if not seeds or not set(seeds)<=set(j['seeds']):parser.error('seeds must be a subset of '+str(j['seeds']))
        j['seeds']=seeds
    if args._worker:
        import torch
        sys.path.insert(0,str(BACKEND))
        os.environ['DEVICE']=args.device if args.device!='auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        from scatter_compat_20261015 import ensure_scatter
        ensure_scatter()
        if args.smoke:
            if j['effective']['pipeline']=='residual':
                import residual_worker as w;w.worker(j,args._seed,True)
            elif j['effective']['pipeline']=='native':
                import native_worker as w;w.worker(j['id'],args._seed,True)
            else:raise RuntimeError('Frozen high-school pipeline has no one-epoch smoke mode; use --check or train.')
        else:
            import server_run as w;w.worker(j,args._seed)
        return
    sys.path.insert(0,str(BACKEND))
    if args.check:
        import server_run as w
        q,y,context=w.expected(j)
        for key in ('source_config','code','protocol'):
            assert (BACKEND/j[key]).exists(),j[key]
        cfg=BACKEND/j['source_config']
        assert hashlib.sha256(cfg.read_bytes()).hexdigest()==j['effective']['source_config_sha256']
        print(json.dumps(dict(status='PASS',dataset=dataset,model=model,configuration=j['effective'],seeds=j['seeds'],candidates=len(y),positives=int(y.sum()),inference_context=context),indent=2));return
    if args.verify:summarize(j);return
    (BACKEND/'results').mkdir(exist_ok=True)
                                                                                   
    planfile=BACKEND/'results/PLAN.json'
    if not planfile.exists():planfile.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for seed in j['seeds'][:1] if args.smoke else j['seeds']:
        d=BACKEND/'results/jobs'/j['id']/str(seed)
        if not args.smoke and (d/'RESULT.json').exists():
            score_file(d/'PREDICTIONS.npz');continue
        logs=ROOT/'results/logs';logs.mkdir(parents=True,exist_ok=True)
        log=logs/f'{dataset}_{model}_{seed}{"_smoke" if args.smoke else ""}.log'
        cmd=[sys.executable,str(ROOT/'run.py'),'--dataset',dataset,'--model',model,'--device',args.device,'--_worker','--_seed',str(seed)]
        if args.smoke:cmd+=['--smoke']
        print(f'Running {dataset} / {model} / seed {seed}; log: {log}',flush=True)
        env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
        with log.open('w',encoding='utf-8') as f:
            r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:
            print(log.read_text(encoding='utf-8',errors='replace')[-6000:]);raise SystemExit(r.returncode)
    if args.smoke:print('SMOKE PASSED; not a performance result')
    else:summarize(j)


if __name__=='__main__':main()
