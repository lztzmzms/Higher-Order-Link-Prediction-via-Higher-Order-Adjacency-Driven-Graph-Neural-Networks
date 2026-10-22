#!/usr/bin/env python3
"""Select EASY K2 candidates using causal cross-time prediction error, never test labels."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import time
import argparse
import traceback
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import core as s

HERE = Path(__file__).resolve().parent
ORIGINS = (.1, .2, .3, .4, .5, .6)
VALIDATION = (.4, .5, .6)
SAMPLERS = ('uniform', 'easy_learned_90', 'easy_learned_70', 'easy_learned_50', 'easy_learned_30', 'easy_margin_50')
CORE_REPORT = s.report


def candidate_batches(history, batch_size=50000):
    batch=[]
    for a in sorted(history.adj):
        for b in sorted(v for v in history.adj[a] if v>a):
            for c in sorted(v for v in history.adj[a]&history.adj[b] if v>b):
                if (a,b,c) not in history.closed:
                    batch.append((a,b,c))
                    if len(batch)>=batch_size:
                        yield np.asarray(batch,dtype=np.int64); batch=[]
    if batch:
        yield np.asarray(batch,dtype=np.int64)


def proposal(history, cap, seed, heartbeat=None):
    """Uniform reservoir of FULL legal population. No outcome inputs; exact population count."""
    rng=np.random.default_rng(seed)
    q=np.empty((0,3),dtype=np.int64); priority=np.empty(0); total=0
    for block in candidate_batches(history):
        total+=len(block)
        if heartbeat:
            heartbeat(total)
        q=np.concatenate([q,block]); priority=np.r_[priority,rng.random(len(block))]
        if len(q)>cap:
            ids=np.argpartition(priority,cap-1)[:cap]
            q,priority=q[ids],priority[ids]
    order=np.lexsort((q[:,2],q[:,1],q[:,0])) if len(q) else []
    return q[order],total


def population_outcomes(history, future, total):
    """Exact full-population positive count without storing millions of candidate triples."""
    positive=set()
    for _,_,vs in future:
        for q in s.combinations(vs,3):
            if q not in history.closed and all(b in history.adj.get(a,set()) for a,b in s.combinations(q,2)):
                positive.add(q)
    assert len(positive)<=total
    # Only the prevalence/counts are used by constraint(); this is not a candidate-aligned label vector.
    y=np.zeros(total,dtype=np.int8); y[:len(positive)]=1
    return y


def capsnap(snapshots, cut, cap, seed=41, profile='last3'):
    permitted = [v for v in snapshots if v['label_end'] <= cut]
    permitted = permitted[-(1 if profile == 'last1' else 3):]
    if not permitted:
        raise ValueError('No completed past training window')
    x, y = np.concatenate([v['x'] for v in permitted]), np.concatenate([v['y'] for v in permitted])
    ids = s.quota_sample(y, cap, seed)
    gate = {'max_training_label_end': max(v['label_end'] for v in permitted),
            'prediction_cut': cut, 'training_rows': len(ids), 'training_positives': int(y[ids].sum())}
    gate['training_sample_gate'] = s.constraint(y[ids],y,cap,
        dict(ratio_min=.5,ratio_max=2.,size_min=.8,size_max=1.25))
    gate['warmup_below_frozen_scale'] = not gate['training_sample_gate']['size_valid']
    assert gate['max_training_label_end'] <= cut
    return x[ids], y[ids], gate


class Teacher:
    def __init__(self, snapshots, cut, cap):
        x, y, self.gate = capsnap(snapshots, cut, cap)
        self.prior = float(y.mean())
        self.models = []
        if len(np.unique(y)) == 2:
            self.models = [ExtraTreesClassifier(n_estimators=64, min_samples_leaf=10,
                max_depth=10, max_features=.7, n_jobs=1, random_state=41),
                HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
                    min_samples_leaf=20, l2_regularization=8., random_state=41),
                make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))]
            for model in self.models:
                model.fit(x, y)
        self.gate['warmup_constant_teacher'] = not bool(self.models)

    def scores(self, x):
        if not self.models:
            return np.full((len(x), 3), self.prior)
        return np.column_stack([model.predict_proba(x)[:, 1] for model in self.models])


def difficulty_covariates(x, teacher_scores):
    p = np.clip(teacher_scores.mean(axis=1), 1e-6, 1-1e-6)
    extras = np.column_stack([teacher_scores, p, teacher_scores.std(axis=1),
        np.abs(p-.5), -p*np.log(p)-(1-p)*np.log(1-p)])
    return np.column_stack([x, extras]).astype(np.float32)


class Difficulty:
    def __init__(self, past_oof, cut):
        allowed = [v for v in past_oof if v['label_end'] <= cut]
        self.gate = {'max_difficulty_label_end': max((v['label_end'] for v in allowed), default=0),
                     'prediction_cut': cut, 'oof_windows': len(allowed)}
        assert self.gate['max_difficulty_label_end'] <= cut
        self.calibrator, self.regressor = None, None
        if allowed:
            x = np.concatenate([difficulty_covariates(v['x'], v['scores']) for v in allowed])
            p = np.concatenate([v['scores'].mean(axis=1) for v in allowed])
            y = np.concatenate([v['y'] for v in allowed])
            if len(np.unique(y)) == 2:
                self.calibrator = IsotonicRegression(y_min=1e-5, y_max=1-1e-5, out_of_bounds='clip').fit(p, y)
                # True loss of predictions made BEFORE their outcomes. No in-sample errors.
                loss = -y*np.log(np.clip(p, 1e-6, 1-1e-6)) -(1-y)*np.log(np.clip(1-p, 1e-6, 1-1e-6))
                weight = np.where(y == 1, .5/y.mean(), .5/(1-y.mean()))
                self.regressor = ExtraTreesRegressor(n_estimators=64, max_depth=8,
                    min_samples_leaf=40, max_features=.7, random_state=71, n_jobs=1).fit(x, loss,
                                                                                         sample_weight=weight)
        self.gate['learned_difficulty_available'] = self.regressor is not None

    def evaluate(self, x, scores):
        mean = scores.mean(axis=1)
        p = self.calibrator.predict(mean) if self.calibrator is not None else mean
        # Confidence baseline explicitly distinguished from learned historical error.
        margin_loss = 1-np.abs(2*mean-1) + scores.std(axis=1)
        learned_loss = self.regressor.predict(difficulty_covariates(x, scores)) if self.regressor is not None else margin_loss
        return p, margin_loss, learned_loss


def select_easy(q, propensity, margin_loss, learned_loss, n, method, seed, strata_score=None):
    """NO LABEL ARGUMENT. Preserve propensity strata, favor predicted low error."""
    if len(q) < n:
        raise ValueError('Population smaller than frozen-sized sample')
    rng = np.random.default_rng(seed)
    if method == 'uniform':
        ids = np.sort(rng.choice(len(q), n, replace=False))
        return ids, {'method': method, 'eligible_count': len(q), 'labels_used': False,
                     'full_population_probability_weighting_valid': True}
    loss = margin_loss if method.startswith('easy_margin') else learned_loss
    retain = int(method.rsplit('_', 1)[1])/100
    # Historical calibrated positive propensity, not the unknown final class.
    # Stratification guards against selecting only confident negatives.
    # Raw teacher probability keeps ranking resolution when isotonic calibration is flat.
    ranking = propensity if strata_score is None else strata_score
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0], np.round(ranking, 12)))
    bins = np.array_split(order, 10)
    exact = n*np.asarray([len(v) for v in bins])/len(q)
    counts = np.floor(exact).astype(int)
    # Stable tie rule for quota allocation across platforms.
    rest = np.lexsort((np.arange(len(bins)), -(exact-counts)))[:n-int(counts.sum())]
    counts[rest] += 1
    selected, records, eligible_count = [], [], 0
    for index, (members, count) in enumerate(zip(bins, counts)):
        easiest = members[np.lexsort((q[members, 2], q[members, 1], q[members, 0],
                                     np.round(loss[members], 12)))]
        m = max(int(count), int(np.ceil(retain*len(members))))
        pool = easiest[:m]
        ids = rng.choice(pool, int(count), replace=False)
        selected.extend(ids.tolist())
        eligible_count += len(pool)
        records.append({'stratum': index, 'population': len(members), 'eligible': len(pool),
            'sample': int(count), 'population_mean_predicted_loss': float(loss[members].mean()),
            'sample_mean_predicted_loss': float(loss[ids].mean())})
    ids = np.sort(selected)
    return ids, {'method': method, 'labels_used': False, 'propensity_strata': 10,
        'retain_fraction': retain, 'eligible_count': eligible_count, 'strata': records,
        'population_predicted_prevalence': float(propensity.mean()),
        'sample_predicted_prevalence': float(propensity[ids].mean()),
        'full_population_probability_weighting_valid': False,
        'reason': 'Excluded difficult candidates have zero inclusion probability; no full-population IPW claim'}


def build_material(args, cfg):
    ds, out = args.dataset, args.output/args.dataset
    events = s.load_events(args.raw_root, ds)
    cap = cfg['train_sizes'][ds]
    proposal_cap=cfg['proposal_caps'][ds]
    snapshots, oof, pools = [], [], {}
    for fraction in ORIGINS:
        key = int(round(fraction*100))
        path = out/'cache'/f'{key}.npz'
        s.progress(out, 'causal_history', f'prefix {fraction}; supervised gates checked separately')
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                v = {k:z[k] for k in z.files}
            snapshots.append({'x':v['x'], 'y':v['y'], 'names':v['names'].tolist(),
                'cut':int(v['cut']), 'label_end':int(v['label_end'])})
        else:
            cut, end = s.cutoff(events, fraction), s.cutoff(events, round(fraction+.2, 2))
            history = s.History(events[:cut])
            q,total = proposal(history,proposal_cap,20261100+key)
            if not len(q):
                continue
            y = s.labels(events[cut:end], q)
            real_y=population_outcomes(history,events[cut:end],total)
            ids = s.quota_sample(y, cap, key+51)
            x, names = history.features(q[ids])
            s.atomic(out/'cache'/f'{key}_TRAIN_AUDIT.json',
                s.constraint(y[ids],real_y,s.reference(ds)['train_candidates'],cfg))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, y=y[ids], q=q[ids], names=names, cut=cut, label_end=end)
            snapshots.append({'x':x, 'y':y[ids], 'names':names, 'cut':cut, 'label_end':end})
        if fraction >= .3:
            cut, end = s.cutoff(events, fraction), s.cutoff(events, round(fraction+.2, 2))
            history = s.History(events[:cut])
            q,total = proposal(history,proposal_cap,20261100+key)
            # Uniform history-only OOF cohort, forecast lock BEFORE attaching its outcomes.
            ids = np.sort(np.random.default_rng(key+2026).choice(len(q), min(max(6000,cap),len(q)), replace=False))
            x, names = history.features(q[ids])
            try:
                teacher = Teacher(snapshots, cut, cap)
            except ValueError:
                continue
            scores = teacher.scores(x)
            oof_path = out/'oof'/f'{key}_FORECAST.npz'
            oof_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(oof_path, q=q[ids], x=x, scores=scores, cut=cut, label_end=end)
            s.atomic(oof_path.with_suffix('.json'), {**teacher.gate, 'forecast_sha256':s.digest(oof_path),
                'outcomes_attached_after_forecast':True})
            oof.append({'x':x, 'scores':scores, 'y':s.labels(events[cut:end], q[ids]), 'label_end':end})
    for fraction in (*VALIDATION, .8):
        path = out/'pools'/f'{int(fraction*100)}.npz'
        cut = s.cutoff(events, fraction)
        difficulty = Difficulty(oof, cut)
        teacher = Teacher(snapshots, cut, cap)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                pools[fraction] = {k:z[k] for k in z.files}
            continue
        history = s.History(events[:cut])
        q,total = proposal(history,proposal_cap,20261100+int(fraction*100),
            lambda count:s.progress(out,'proposal_scan',f'prefix {fraction}; full candidates enumerated={count}'))
        propensity, margin, learned, scores,features = [], [], [], [],[]
        for start in range(0, len(q), 10000):
            s.progress(out, 'difficulty_forecast', f'prefix {fraction}; {start}/{len(q)} candidates')
            x, names = history.features(q[start:start+10000])
            features.append(x)
            sc = teacher.scores(x)
            p, ma, le = difficulty.evaluate(x, sc)
            propensity.append(p); margin.append(ma); learned.append(le); scores.append(sc)
        pool = {'q':q, 'propensity':np.concatenate(propensity), 'margin':np.concatenate(margin),
            'learned':np.concatenate(learned), 'scores':np.concatenate(scores), 'cut':np.asarray(cut),
            'full_population_count':np.asarray(total), 'proposal_seed':np.asarray(20261100+int(fraction*100))}
        pool['x']=np.concatenate(features); pool['names']=np.asarray(names)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **pool)
        s.atomic(path.with_suffix('.json'), {**teacher.gate, **difficulty.gate,
            'forecast_sha256':s.digest(path), 'candidate_sha256':hashlib_array(q),
            'feature_prefix_end':cut, 'final_labels_visible':False,
            'proposal_candidates':len(q),'full_population_candidates':total,
            'proposal_uniform_label_blind':True})
        pools[fraction] = pool
    return events, snapshots, oof, pools


def hashlib_array(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def head_predict(snapshots, target, recipe, cap, seed, teacher_score):
    if recipe['family'] == 'teacher':
        _, _, gate = capsnap(snapshots,target['cut'],cap)
        return teacher_score, gate
    return s.predict(snapshots, target, recipe, cap, seed, 61011)


def run_job(args, cfg):
    ds, out = args.dataset, args.output/args.dataset
    out.mkdir(parents=True, exist_ok=True)
    events, snapshots, oof, pools = build_material(args, cfg)
    cap, n = cfg['train_sizes'][ds], cfg['test_sizes'][ds]
    arms = [(method, seed) for method in cfg['samplers'] for seed in cfg['sample_seeds']]
    heads = {}
    validation_material={}
    for fraction in VALIDATION:
        pool=pools[fraction]; cut=int(pool['cut']); end=s.cutoff(events,round(fraction+.2,2))
        history=s.History(events[:cut])
        validation_material[fraction]=(s.labels(events[cut:end],pool['q']),
            population_outcomes(history,events[cut:end],int(pool['full_population_count'])))
        del history
    for method, seed in arms:
        arm = f'{method}_{seed}'
        targets = []
        for fraction in VALIDATION:
            pool = pools[fraction]
            ids, audit = select_easy(pool['q'], pool['propensity'], pool['margin'], pool['learned'], n,
                                     method, seed, pool['scores'].mean(axis=1))
            cut, end = int(pool['cut']), s.cutoff(events, round(fraction+.2, 2))
            full_y,real_y = validation_material[fraction]
            x,names=pool['x'][ids],pool['names'].tolist()
            # These outcomes are PAST at final selection (.8), never sampler inputs.
            gate = s.constraint(full_y[ids], real_y, cfg['test_reference_sizes'][ds], cfg)
            s.atomic(out/'validation_samples'/f'{arm}_{int(fraction*100)}.json', {**audit, 'constraint':gate})
            targets.append({'cut':cut, 'label_end':end, 'x':x, 'names':names, 'y':full_y[ids],
                            'teacher':pool['propensity'][ids], 'gate':gate})
        for index, recipe in enumerate(cfg['models']):
            path = out/'trials'/arm/f'{index:03d}.json'
            previous = s.read(path)
            if previous.get('state') in ('COMPLETE', 'SKIPPED'):
                row = previous
            else:
                row = None
                for attempt in (1, 2):
                    try:
                        folds = []
                        for target in targets:
                            if not target['gate']['ratio_valid'] or not target['gate']['size_valid']:
                                raise ValueError('Historical EASY sample violates ratio/scale; arm not promoted')
                            s.progress(out, 'validation_search', f'{arm} head={index} attempt={attempt}')
                            prediction, gate = head_predict(snapshots, target, recipe, cap, 41, target['teacher'])
                            metrics = s.metrics(target['y'], prediction)
                            if metrics['Lift'] is None:
                                raise ValueError('Historical EASY sample needs both classes')
                            folds.append({**metrics, 'gate':gate})
                        logs = np.log([v['Lift'] for v in folds])
                        row = {'state':'COMPLETE', 'config':recipe, 'folds':folds,
                               'quality':float(np.median(logs)+.25*logs.min())}
                        break
                    except Exception as exc:
                        row = {'state':'SKIPPED', 'error':repr(exc), 'config':recipe, 'attempts':attempt}
                s.atomic(path, row)
            if row['state'] == 'COMPLETE' and row['quality'] > heads.get(arm, {}).get('quality', -1e99):
                heads[arm] = {**row, 'method':method, 'sample_seed':seed}
    if not heads:
        raise ValueError('No historically admissible arm; final outcomes remain unopened')
    recommended = max(heads, key=lambda k:heads[k]['quality'])
    s.atomic(out/'PARAMETER_LOCK.json', {'selected_per_arm':heads, 'recommended_arm':recommended,
        'final_labels_used':False, 'plan_sha256':cfg['plan_sha256']})
    pool = pools[.8]
    history = s.History(events[:int(pool['cut'])])
    forecasts = {}
    for arm, chosen in heads.items():
        ids, audit = select_easy(pool['q'], pool['propensity'], pool['margin'], pool['learned'], n,
                                 chosen['method'], chosen['sample_seed'], pool['scores'].mean(axis=1))
        x,names=pool['x'][ids],pool['names'].tolist()
        target = {'x':x, 'names':names, 'cut':int(pool['cut'])}
        scores, gates = [], []
        for seed in cfg['model_seeds']:
            s.progress(out, 'locked_final_forecast', f'{arm} seed={seed}')
            sc, gate = head_predict(snapshots, target, chosen['config'], cap, seed, pool['propensity'][ids])
            scores.append(sc); gates.append(gate)
        path = out/'predictions'/f'{arm}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=pool['q'][ids], population_indices=ids, scores=scores,
            score=np.mean(scores, axis=0), propensity=pool['propensity'][ids],
            predicted_loss=pool['learned'][ids])
        forecasts[arm] = {'sha256':s.digest(path), 'sampling':audit, 'gates':gates}
    s.atomic(out/'PREDICTION_LOCK.json', {'parameter_sha256':s.digest(out/'PARAMETER_LOCK.json'),
        'all_forecasts_saved_before_final_labels':True, 'predictions':forecasts})
    # First and only opening of the final outcomes in the worker.
    y = s.labels(events[int(pool['cut']):], pool['q'])
    real_y=population_outcomes(history,events[int(pool['cut']):],int(pool['full_population_count']))
    rows = []
    for arm, forecast in forecasts.items():
        with np.load(out/'predictions'/f'{arm}.npz', allow_pickle=False) as z:
            yy = y[z['population_indices']]
            gate = s.constraint(yy, real_y, cfg['test_reference_sizes'][ds], cfg)
            target_x,names=pool['x'][z['population_indices']],pool['names'].tolist()
            export=out/'local_debug'/arm
            export.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(export/'test_locked.npz',q=z['q'],x=target_x,y=yy,names=names,
                score=z['score'],scores=z['scores'],prediction_cut=pool['cut'])
            s.atomic(export/'PROTOCOL.json',{'dataset':ds,'sampling':forecast['sampling'],'constraint':gate,
                'candidate_sha256':hashlib_array(z['q']),'prediction_cut':int(pool['cut']),
                'raw_context':'data files sorted by (timestamp,row_id), prefix before prediction_cut',
                'fit_caches':'../../cache','test_labels_attached_only_after_forecast_lock':True,
                'test_now_available_for_exploratory_local_debug':True})
            rows.append({'dataset':ds, 'arm':arm, 'recommended_before_final_labels':arm==recommended,
                'state':'VALID' if gate['ratio_valid'] and gate['size_valid'] else 'INVALID_RATIO_OR_SCALE',
                'audit':gate, 'metrics':s.metrics(yy,z['score']),
                'model_seed_metrics':[s.metrics(yy,sc) for sc in z['scores']],
                'population_weighted_metrics':{'Lift':None, 'AP':None},
                'config':heads[arm]['config'], 'prediction':forecast,
                'full_population_performance_claim':False,
                'test_previously_inspected_exploratory':True})
    s.atomic(out/'RESULTS.json', {'rows':rows, 'recommended_arm':recommended,
        'final_pool':{'candidates':len(real_y), 'positives':int(real_y.sum()),'proposal_candidates':len(y)},
        'invalid_samples_not_resampled_or_promoted':True})
    valid_easy=[r['arm'] for r in rows if r['state']=='VALID' and not r['arm'].startswith('uniform_')]
    s.atomic(out/'SUITABLE_SAMPLES.json',{'found':bool(valid_easy),'valid_easy_arms':valid_easy,
        'historical_recommended_arm':recommended,'highest_test_score_not_used_for_sampling':True,
        'no_guarantee_any_easy_sample_satisfies_future_ratio':True})
    s.atomic(out/'DONE.json', {'state':'COMPLETE', 'utc':s.utc()})
    report(args.output)


def report(out):
    CORE_REPORT(out)
    path = out/'REPORT.md'
    text = path.read_text(encoding='utf-8').replace('服务器时序采样与模型比较','历史难度驱动的容易单纯形采样').replace(
        '同时报告 AP、Lift、实际正例数及按抽样概率加权的总体估计',
        '报告容易子集 AP、Lift 和实际正例数，不估计完整总体性能')
    text += ('\n本包明确筛选预测容易的候选：十个历史预测正例倾向层内，优先保留低预测误差者。'
        '\nlearned 难度模型仅学习已完成的跨时间预测误差；无足够历史OOF时退回置信度基线，记录fallback。'
        '\n容易子集性能不能解释为原完整总体性能；被排除候选抽样概率为零，不报告完整总体IPW成绩。'
        '\n样本锁定后若真实正负比或规模不合格，标记INVALID，不重抽、不根据最终成绩换推荐。\n')
    path.write_text(text, encoding='utf-8')


def models(quick):
    result = [{'family':'teacher'}, dict(family='HG',leaf=20,profile='last3',balanced=False,view='compact',fusion=0.)]
    if not quick:
        for family in ('ET','HG','LR'):
            for profile in ('last1','last3'):
                for balanced in (False,True):
                    result.append(dict(family=family,leaf=20,profile=profile,balanced=balanced,view='compact',fusion=0.))
    return result


def canonical_ds(ds):
    return 'threads-math-sx' if ds=='thread-math-sx' else ds


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('run','worker','preflight','report','verify'))
    p.add_argument('--raw-root', type=Path, default=HERE/'data')
    p.add_argument('--output', type=Path, default=HERE/'results/easy')
    p.add_argument('--datasets', nargs='+', type=canonical_ds,choices=s.DATASETS, default=list(s.DATASETS))
    p.add_argument('--dataset', type=canonical_ds,choices=s.DATASETS)
    p.add_argument('--samplers', nargs='+', choices=SAMPLERS, default=list(SAMPLERS))
    p.add_argument('--sample-seeds', nargs='+', type=int, default=[2026,2027,2028])
    p.add_argument('--model-seeds', nargs='+', type=int, default=[41])
    p.add_argument('--ndc-size', type=int, default=1602)
    p.add_argument('--math-size', type=int, default=573)
    p.add_argument('--threads-size',type=int,default=30000)
    p.add_argument('--hours', type=float, default=6.)
    p.add_argument('--job-seconds', type=int, default=18000)
    p.add_argument('--stale-seconds', type=int, default=1800)
    p.add_argument('--quick', action='store_true',help='Compatibility flag: small performance test is already default')
    p.add_argument('--full-model-search',action='store_true',help='Optional; default only two test heads')
    p.add_argument('--refit', action='store_true', help='Also independently refit locked final models during verify')
    return p.parse_args()


def verify(args):
    checks = {}
    cfg = s.read(args.output/'EFFECTIVE_CONFIG.json')
    for ds in args.datasets:
        out = args.output/ds
        if not (out/'DONE.json').exists():
            checks[ds] = {'state':'NOT_COMPLETE'}; continue
        events = s.load_events(args.raw_root, ds)
        history=s.History(events[:s.cutoff(events,.8)])
        lock = s.read(out/'PREDICTION_LOCK.json')
        assert s.digest(out/'PARAMETER_LOCK.json') == lock['parameter_sha256']
        with np.load(out/'pools/80.npz', allow_pickle=False) as z:
            pool = {k:z[k] for k in z.files}
        assert s.digest(out/'pools/80.npz') == s.read(out/'pools/80.json')['forecast_sha256']
        q,total=proposal(history,cfg['proposal_caps'][ds],int(pool['proposal_seed']))
        assert np.array_equal(q,pool['q']) and total==int(pool['full_population_count'])
        y=s.labels(events[s.cutoff(events,.8):],q)
        real_y=population_outcomes(history,events[s.cutoff(events,.8):],total)
        selected = s.read(out/'PARAMETER_LOCK.json')['selected_per_arm']
        results = {r['arm']:r for r in s.read(out/'RESULTS.json')['rows']}
        snapshots, oof = [], []
        if args.refit:
            for path in sorted((out/'cache').glob('*.npz')):
                with np.load(path,allow_pickle=False) as z:
                    snapshots.append({'x':z['x'],'y':z['y'],'names':z['names'].tolist(),
                        'cut':int(z['cut']),'label_end':int(z['label_end'])})
            for path in sorted((out/'oof').glob('*_FORECAST.npz')):
                assert s.digest(path)==s.read(path.with_suffix('.json'))['forecast_sha256']
                with np.load(path,allow_pickle=False) as z:
                    oof.append({'x':z['x'],'scores':z['scores'],'label_end':int(z['label_end']),
                        'y':s.labels(events[int(z['cut']):int(z['label_end'])],z['q'])})
            teacher = Teacher(snapshots,int(pool['cut']),cfg['train_sizes'][ds])
            difficulty = Difficulty(oof,int(pool['cut']))
        for arm, meta in lock['predictions'].items():
            assert s.digest(out/'predictions'/f'{arm}.npz') == meta['sha256']
            chosen = selected[arm]
            ids, audit = select_easy(q,pool['propensity'],pool['margin'],pool['learned'],cfg['test_sizes'][ds],
                                     chosen['method'],chosen['sample_seed'],pool['scores'].mean(axis=1))
            with np.load(out/'predictions'/f'{arm}.npz',allow_pickle=False) as z:
                assert np.array_equal(ids,z['population_indices']) and np.array_equal(q[ids],z['q'])
                m = s.metrics(y[ids],z['score'])
                assert all(v is None and results[arm]['metrics'][k] is None or v is not None and
                           np.isclose(v,results[arm]['metrics'][k]) for k,v in m.items())
                assert s.constraint(y[ids],real_y,cfg['test_reference_sizes'][ds],cfg)==results[arm]['audit']
                if args.refit:
                    x,names=pool['x'][ids],pool['names'].tolist()
                    propensity,_,_=difficulty.evaluate(x,teacher.scores(x))
                    np.testing.assert_allclose(propensity,pool['propensity'][ids],rtol=1e-9,atol=1e-12)
                    target={'x':x,'names':names,'cut':int(pool['cut'])}
                    for i,seed in enumerate(cfg['model_seeds']):
                        score,gate=head_predict(snapshots,target,chosen['config'],cfg['train_sizes'][ds],seed,propensity)
                        np.testing.assert_allclose(score,z['scores'][i],rtol=1e-9,atol=1e-12)
                        assert gate['max_training_label_end']<=target['cut']
            assert audit == meta['sampling']
        for path in (out/'pools').glob('*.json'):
            gate = s.read(path)
            assert gate['max_training_label_end']<=gate['prediction_cut']
            assert gate['max_difficulty_label_end']<=gate['prediction_cut']
        checks[ds] = {'state':'PASSED','arms':len(lock['predictions']),
            'locked_pool_hash_sampling_replay_metrics_constraints_and_label_gates_verified':True,
            'model_refit_performed':args.refit}
    s.atomic(args.output/'VERIFICATION.json', {'utc':s.utc(),'datasets':checks})
    print(checks)


s.report = report
# Supervisor dispatches this worker, while core.py remains immutable protocol utilities.
s.__file__ = __file__
if __name__ == '__main__':
    args = arguments(); args.output=args.output.resolve(); args.raw_root=args.raw_root.resolve()
    if args.command=='worker':
        run_job(args,s.read(args.output/'EFFECTIVE_CONFIG.json'))
    elif args.command=='report':
        report(args.output)
    elif args.command=='verify':
        verify(args)
    else:
        cfg = {'datasets':args.datasets,'samplers':args.samplers,'sample_seeds':args.sample_seeds,
            'model_seeds':args.model_seeds,'test_sizes':{'NDC-substances':args.ndc_size,'tags-math-sx':args.math_size,
                'threads-math-sx':args.threads_size},
            'train_sizes':{ds:s.reference(ds)['train_candidates'] for ds in s.DATASETS},
            'ratio_min':.5,'ratio_max':2.,'size_min':.8,'size_max':1.25,'models':models(not args.full_model_search),
            'proposal_caps':{'NDC-substances':100000,'tags-math-sx':200000,'threads-math-sx':300000},
            'raw_root':str(args.raw_root),'code_sha256':s.digest(Path(__file__)),
            'core_sha256':s.digest(HERE/'core.py'),'difficulty':'past temporal OOF class-balanced log-loss regression',
            'final_outcomes_used_for_sampling':False,'difficulty_selection_explicit':True}
        s.supervise(args,cfg)
