#!/usr/bin/env python3
"""Standalone causal K2 sampling/model comparison with post-lock test auditing."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import csv
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
import traceback
import platform
from collections import Counter, defaultdict
from itertools import combinations, groupby
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SAMPLERS = ('uniform', 'activity_stratified', 'recent_mixture')
DATASETS = ('NDC-substances', 'tags-math-sx', 'threads-math-sx')


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else (default or {})


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def progress(out, stage, detail):
    atomic(out / 'STATUS.json', {'stage': stage, 'detail': detail, 'updated_utc': utc(), 'epoch': time.time()})
    print(f'{utc()} {stage}: {detail}', flush=True)


def load_events(raw, dataset):
    paths = {s: raw / dataset / f'{dataset}-{s}.txt' for s in ('nverts', 'simplices', 'times')}
    values = {s: np.loadtxt(p, dtype=np.int64, ndmin=1) for s, p in paths.items()}
    assert len(values['nverts']) == len(values['times'])
    assert int(values['nverts'].sum()) == len(values['simplices'])
    result = []
    offset = 0
    for i, (n, timestamp) in enumerate(zip(values['nverts'], values['times'])):
        vertices = tuple(sorted(set(map(int, values['simplices'][offset:offset + n]))))
        offset += int(n)
        # Keep even empty rows so global event-count fractions preserve the source timeline.
        result.append((int(timestamp), i, vertices))
    result.sort(key=lambda v: (v[0], v[1]))
    return result


def cutoff(events, fraction):
    index = min(len(events), int(len(events) * fraction))
    if index == len(events):
        return index
    while index and events[index - 1][0] == events[index][0]:
        index -= 1
    return index


class History:
    """Only the supplied prefix is visible to this object."""
    def __init__(self, prefix):
        self.adj = defaultdict(set)
        self.closed = set()
        self.node = Counter()
        self.edge = Counter()
        self.group = Counter()
        self.nc = Counter()
        self.ec = Counter()
        self.last = {}
        self.decay = {t: Counter() for t in (.01, .05, .2, 1.)}
        self.start = prefix[0][0]
        self.end = prefix[-1][0]
        self.span = max(1, self.end - self.start)
        for timestamp, iterator in groupby(prefix, key=lambda v: v[0]):
            block = list(iterator)
            new = {q for _, _, vs in block for q in combinations(vs, 3)} - self.closed
            # Simultaneous edge observations cannot precede a closure in the same block.
            eligible = [q for q in new if all(b in self.adj[a] for a, b in combinations(q, 2))]
            for q in eligible:
                self.nc.update(q)
                self.ec.update(combinations(q, 2))
            for _, _, vs in block:
                self.node.update(vs)
                edges = list(combinations(vs, 2))
                self.edge.update(edges)
                if len(vs) >= 3:
                    self.group.update(edges)
                for edge in edges:
                    x, y = edge
                    self.adj[x].add(y)
                    self.adj[y].add(x)
                    self.last[edge] = timestamp
                    for tau, count in self.decay.items():
                        count[edge] += math.exp(-(self.end - timestamp) / (tau * self.span))
            self.closed.update(new)

    def candidates(self):
        result = []
        for x in sorted(self.adj):
            for y in sorted(v for v in self.adj[x] if v > x):
                for z in sorted(v for v in self.adj[x] & self.adj[y] if v > y):
                    q = (x, y, z)
                    if q not in self.closed:
                        result.append(q)
        return np.asarray(result, dtype=np.int64).reshape(-1, 3)

    def proxies(self, q):
        activity, recent = [], []
        for tri in q:
            edges = list(combinations(map(int, tri), 2))
            activity.append(sum(math.log1p(self.edge[e]) for e in edges))
            recent.append(sum(self.decay[.05][e] for e in edges))
        return np.asarray(activity), np.asarray(recent)

    def features(self, queries):
        output = defaultdict(list)
        for tri in queries:
            tri = tuple(map(int, tri))
            edges = list(combinations(tri, 2))
            groups = {
                'edge_count': [self.edge[e] for e in edges],
                'edge_group': [self.group[e] for e in edges],
                'node_count': [self.node[v] for v in tri],
                'node_degree': [len(self.adj[v]) for v in tri],
                'node_closure': [self.nc[v] for v in tri],
                'edge_closure': [self.ec[e] for e in edges],
                'edge_age': [(self.end - self.last[e]) / self.span for e in edges],
                'edge_common': [len(self.adj[x] & self.adj[y]) for x, y in edges],
                'edge_jaccard': [len(self.adj[x] & self.adj[y]) / max(1, len(self.adj[x] | self.adj[y])) for x, y in edges],
                'edge_coactivity': [self.edge[x, y] / math.sqrt(max(1, self.node[x] * self.node[y])) for x, y in edges],
            }
            for tau, counts in self.decay.items():
                groups[f'decay_{tau}'] = [counts[e] for e in edges]
            for name, values in groups.items():
                v = np.asarray(values, dtype=float)
                for suffix, val in [('min', v.min()), ('mean', v.mean()), ('max', v.max()),
                        ('std', v.std()), ('logsum', np.log1p(v).sum()),
                        ('harmonic', 3 / np.sum(1 / (v + 1e-12)))]:
                    output[name + '_' + suffix].append(val)
            output['common_three'].append(len(set.intersection(*(self.adj[v] for v in tri))))
        names = sorted(output)
        x = np.column_stack([output[n] for n in names])
        return np.log1p(np.maximum(x, 0)).astype(np.float32), names


def labels(future, q):
    lookup = {tuple(map(int, tri)): i for i, tri in enumerate(q)}
    y = np.zeros(len(q), dtype=np.int8)
    for _, _, vs in future:
        for tri in combinations(vs, 3):
            index = lookup.get(tri)
            if index is not None:
                y[index] = 1
    return y


def constraint(y, population_y, reference_size, config):
    p, n = int(np.sum(y)), len(y) - int(np.sum(y))
    pp, nn = int(np.sum(population_y)), len(population_y) - int(np.sum(population_y))
    factor = (p / n) / (pp / nn) if n and pp and nn else None
    size_factor = len(y) / reference_size
    return {'sample_candidates': len(y), 'sample_positives': p, 'sample_negatives': n,
        'population_candidates': len(population_y), 'population_positives': pp,
        'population_positive_negative_ratio': pp / nn if nn else None,
        'sample_positive_negative_ratio': p / n if n else None, 'ratio_factor': factor,
        'size_factor': size_factor,
        'ratio_valid': factor is not None and config['ratio_min'] <= factor <= config['ratio_max'],
        'size_valid': config['size_min'] <= size_factor <= config['size_max']}


def quota_sample(y, n, seed):
    """Past training labels only: preserve the full historical window's prevalence."""
    rng = np.random.default_rng(seed)
    n = min(n, len(y))
    pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
    np_ = int(round(n * len(pos) / len(y)))
    selected = np.r_[rng.choice(pos, min(np_, len(pos)), replace=False),
        rng.choice(neg, n - min(np_, len(pos)), replace=False)]
    return np.sort(selected)


def sample_candidates(q, activity, recent, n, sampler, seed):
    """No labels accepted: cohort identity must depend on past observations only."""
    if len(q) < n:
        raise ValueError(f'candidate pool {len(q)} is smaller than requested sample {n}')
    rng = np.random.default_rng(seed)
    if sampler == 'uniform':
        ids = np.sort(rng.choice(len(q), n, replace=False))
        return ids, np.full(n, len(q) / n), {'strata': 1}
    proxy = activity if sampler == 'activity_stratified' else recent
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0], proxy))
    bins = list(np.array_split(order, 4))
    allocation = np.asarray([len(v) for v in bins], dtype=float)
    if sampler == 'recent_mixture':
        allocation *= np.asarray([.75, .9, 1.1, 1.25])
    exact = n * allocation / allocation.sum()
    counts = np.floor(exact).astype(int)
    for k in np.argsort(-(exact - counts))[:n - int(counts.sum())]:
        counts[k] += 1
    if any(c <= 0 or c > len(b) for c, b in zip(counts, bins)):
        raise ValueError('infeasible stratum quota; refusing silent size reduction')
    chosen, weights = [], []
    for b, c in zip(bins, counts):
        chosen.extend(rng.choice(b, int(c), replace=False))
        weights.extend([len(b) / c] * int(c))
    sorted_ids = np.argsort(chosen)
    return np.asarray(chosen)[sorted_ids], np.asarray(weights)[sorted_ids], {
        'strata': 4, 'population_counts': [len(v) for v in bins], 'sample_counts': counts.tolist()}


def metrics(y, score, weight=None):
    if len(np.unique(y)) != 2:
        return {'AP': None, 'Lift': None, 'prevalence': float(np.mean(y)), 'ROC_AUC': None}
    prevalence = float(np.average(y, weights=weight))
    ap = float(average_precision_score(y, score, sample_weight=weight))
    return {'AP': ap, 'Lift': ap / prevalence, 'prevalence': prevalence,
        'ROC_AUC': float(roc_auc_score(y, score, sample_weight=weight))}


def configs(quick=False):
    rules = [{'family': 'rule', 'feature': f, 'sign': sign, 'profile': 'last1'}
        for f, sign in [('edge_count_harmonic', 1), ('edge_group_logsum', 1),
            ('decay_0.05_harmonic', 1), ('decay_0.2_mean', 1), ('edge_coactivity_min', 1), ('edge_age_max', -1)]]
    if quick:
        return rules[:2] + [dict(family='ET', leaf=10, profile='last1', balanced=True, view='all', fusion=0.)]
    models = []
    for profile in ('last1', 'last3'):
        for family in ('ET', 'RF', 'HG', 'LR'):
            for leaf in (5, 20):
                for fusion in (0., .5):
                    models.append(dict(family=family, leaf=leaf, profile=profile, balanced=True,
                        view='all' if leaf == 5 else 'compact', fusion=fusion))
    return rules + models


def predict(train_snapshots, target, cfg, train_size, seed, cap_seed):
    assert 'y' not in target or target['label_end'] > target['cut']
    if cfg['family'] == 'rule':
        col = target['names'].index(cfg['feature'])
        return cfg['sign'] * target['x'][:, col], {'max_training_label_end': 0, 'training_rows': 0}
    allowed = [v for v in train_snapshots if v['label_end'] <= target['cut']]
    allowed = allowed[-(1 if cfg['profile'] == 'last1' else 3):]
    if not allowed:
        raise ValueError('no completed historical training windows')
    x = np.concatenate([v['x'] for v in allowed])
    y = np.concatenate([v['y'] for v in allowed])
    indices = quota_sample(y, train_size, cap_seed)
    xx, yy = x[indices], y[indices]
    gate = constraint(yy, y, train_size, {'ratio_min': .5, 'ratio_max': 2., 'size_min': .8, 'size_max': 1.25})
    if not gate['ratio_valid'] or not gate['size_valid'] or len(np.unique(yy)) < 2:
        raise ValueError('historical fit sample fails class-ratio/size constraints')
    names = target['names']
    columns = list(range(len(names))) if cfg['view'] == 'all' else [i for i, n in enumerate(names)
        if n.endswith(('_min', '_mean', '_harmonic', '_logsum'))]
    if cfg['family'] in ('ET', 'RF'):
        factory = ExtraTreesClassifier if cfg['family'] == 'ET' else RandomForestClassifier
        model = factory(n_estimators=120, min_samples_leaf=cfg['leaf'], max_features=.7,
            random_state=seed, n_jobs=1)
    elif cfg['family'] == 'HG':
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=cfg['leaf'] + 3,
            min_samples_leaf=20, l2_regularization=8., random_state=seed)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1500, random_state=seed))
    p = float(yy.mean())
    w = np.where(yy, .5 / p, .5 / (1 - p)) if cfg['balanced'] else np.ones(len(yy))
    if hasattr(model, 'steps'):
        model.fit(xx[:, columns], yy, logisticregression__sample_weight=w)
    else:
        model.fit(xx[:, columns], yy, sample_weight=w)
    score = model.predict_proba(target['x'][:, columns])[:, 1]
    if cfg['fusion']:
        prior = target['x'][:, names.index('edge_count_harmonic')]
        score = ((1 - cfg['fusion']) * rankdata(score) + cfg['fusion'] * rankdata(prior)) / len(score)
    assert np.isfinite(score).all()
    return score, {'max_training_label_end': max(v['label_end'] for v in allowed),
        'prediction_cut': target['cut'], 'training_rows': len(yy), 'training_sample_gate': gate}


def reference(ds):
    return read(HERE / 'FROZEN_REFERENCE.json')['datasets'][ds]


def run_job(args, cfg):
    ds = args.dataset
    ref = reference(ds)
    out = args.output / ds
    out.mkdir(parents=True, exist_ok=True)
    events = load_events(args.raw_root, ds)
    snapshot_dir = out / 'cache'
    train_size = cfg['train_sizes'][ds]
    test_size = cfg['test_sizes'][ds]
    train_snapshots, val_arms = [], defaultdict(list)
    arms = [(sampler, seed) for sampler in cfg['samplers'] for seed in cfg['sample_seeds']]
    for fraction in (.2, .3, .4, .5, .6):
        key = int(round(fraction * 100))
        cache = snapshot_dir / f'{key}.npz'
        progress(out, 'history_features', f'{ds} prefix={fraction}; no later events visible')
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                material = {k: z[k] for k in z.files}
        else:
            cut = cutoff(events, fraction)
            end = cutoff(events, round(fraction + .2, 2))
            state = History(events[:cut])
            q = state.candidates()
            if not len(q):
                raise ValueError('empty historical candidate pool')
            # The training label horizon is already in the past at every later fit.
            y = labels(events[cut:end], q)
            ids = quota_sample(y, train_size, 20261011 + key)
            train_gate = constraint(y[ids], y, ref['train_candidates'], cfg)
            x, names = state.features(q[ids])
            activity, recent = state.proxies(q)
            material = dict(cut=np.asarray(cut), label_end=np.asarray(end), x=x, y=y[ids],
                q=q[ids], names=np.asarray(names), train_valid=np.asarray(train_gate['ratio_valid'] and train_gate['size_valid']))
            atomic(snapshot_dir / f'{key}_TRAIN_AUDIT.json', train_gate)
            if fraction >= .4:
                for j, (sampler, seed) in enumerate(arms):
                    vi, weights, audit = sample_candidates(q, activity, recent, test_size, sampler, seed)
                    vx, vn = state.features(q[vi])
                    assert names == vn
                    gate = constraint(y[vi], y, ref['test_candidates'], cfg)
                    material.update({f'v{j}_x': vx, f'v{j}_y': y[vi], f'v{j}_w': weights,
                        f'v{j}_valid': np.asarray(gate['ratio_valid'] and gate['size_valid'])})
                    atomic(snapshot_dir / f'{key}_ARM{j}_AUDIT.json', {**gate, 'sampling': audit})
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **material)
        base = {'cut': int(material['cut']), 'label_end': int(material['label_end']),
            'x': material['x'], 'y': material['y'], 'names': material['names'].tolist()}
        if bool(material['train_valid']):
            train_snapshots.append(base)
        if fraction >= .4:
            for j, arm in enumerate(arms):
                val_arms[arm].append({**base, 'x': material[f'v{j}_x'], 'y': material[f'v{j}_y'],
                    'weights': material[f'v{j}_w'], 'valid': bool(material[f'v{j}_valid'])})
    recipes = cfg['models']
    locks = {}
    for arm in arms:
        sampler, sample_seed = arm
        arm_name = f'{sampler}_{sample_seed}'
        trials = []
        for index, recipe in enumerate(recipes):
            path = out / 'trials' / arm_name / f'{index:03d}.json'
            if path.exists() and read(path).get('state') in ('COMPLETE', 'SKIPPED'):
                row = read(path)
            else:
                row = None
                for attempt in (1, 2):
                    try:
                        folds = []
                        for target in val_arms[arm]:
                            if not target['valid']:
                                raise ValueError('a historical validation arm fails ratio/scale limits')
                            progress(out, 'validation_search', f'{arm_name} config={index} attempt={attempt}')
                            pred, gate = predict(train_snapshots, target, recipe, train_size,
                                cfg['model_seeds'][0], 61011)
                            assert gate['max_training_label_end'] <= target['cut']
                            folds.append({**metrics(target['y'], pred), 'gate': gate})
                        if any(f['Lift'] is None for f in folds):
                            raise ValueError('validation requires both label classes')
                        row = {'state': 'COMPLETE', 'config': recipe, 'folds': folds,
                            'quality': float(np.mean(np.log([f['Lift'] for f in folds])))}
                        break
                    except Exception as exc:
                        row = {'state': 'SKIPPED', 'config': recipe, 'error': repr(exc), 'attempts': attempt}
                atomic(path, row)
            if row['state'] == 'COMPLETE':
                trials.append(row)
        if trials:
            locks[arm_name] = {**max(trials, key=lambda v: v['quality']), 'sampler': sampler,
                'sample_seed': sample_seed, 'selection_uses_only_completed_past_windows': True}
    if not locks:
        raise ValueError('no arm has three admissible historical validation folds; see audit files')
    primary = max(locks, key=lambda arm: locks[arm]['quality'])
    atomic(out / 'PARAMETER_LOCK.json', {'utc': utc(), 'selected_per_arm': locks,
        'recommended_arm': primary, 'final_labels_used': False, 'plan_sha256': cfg['plan_sha256']})
    progress(out, 'final_prefix', 'sampling and forecasting locked before opening final labels')
    cut = cutoff(events, .8)
    state = History(events[:cut])
    full_q = state.candidates()
    activity, recent = state.proxies(full_q)
    # No test outcomes are evaluated until ALL predeclared arms' forecasts are on disk.
    predictions = {}
    for arm_name, locked in locks.items():
        sampler, seed = locked['sampler'], locked['sample_seed']
        ids, weights, audit = sample_candidates(full_q, activity, recent, test_size, sampler, seed)
        x, names = state.features(full_q[ids])
        target = {'cut': cut, 'x': x, 'names': names}
        candidate_path = out / 'predictions' / f'{arm_name}_CANDIDATES.npz'
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(candidate_path, q=full_q[ids], population_indices=ids, weights=weights)
        values, gates = [], []
        for model_seed in cfg['model_seeds']:
            progress(out, 'final_forecast', f'{arm_name} model_seed={model_seed}')
            pred, gate = predict(train_snapshots, target, locked['config'], train_size, model_seed, 61011)
            assert gate['max_training_label_end'] <= cut
            values.append(pred)
            gates.append(gate)
        path = out / 'predictions' / f'{arm_name}.npz'
        np.savez_compressed(path, q=full_q[ids], population_indices=ids, weights=weights,
            scores=np.asarray(values), score=np.mean(values, axis=0))
        predictions[arm_name] = {'path': str(path), 'sha256': digest(path), 'gates': gates, 'sampling': audit}
    atomic(out / 'PREDICTION_LOCK.json', {'utc': utc(), 'parameter_sha256': digest(out / 'PARAMETER_LOCK.json'),
        'all_forecasts_saved_before_final_labels': True, 'predictions': predictions})
    y = labels(events[cut:], full_q)
    rows = []
    for arm_name, meta in predictions.items():
        path = Path(meta['path'])
        assert digest(path) == meta['sha256']
        with np.load(path, allow_pickle=False) as z:
            ids, score, weights = z['population_indices'], z['score'], z['weights']
            gate = constraint(y[ids], y, ref['test_candidates'], cfg)
            valid = gate['ratio_valid'] and gate['size_valid']
            row = {'dataset': ds, 'arm': arm_name, 'recommended_before_final_labels': arm_name == primary,
                'state': 'VALID' if valid else 'INVALID_RATIO_OR_SCALE', 'audit': gate,
                'metrics': metrics(y[ids], score), 'population_weighted_metrics': metrics(y[ids], score, weights),
                'model_seed_metrics': [metrics(y[ids], p) for p in z['scores']],
                'config': locks[arm_name]['config'], 'validation_quality': locks[arm_name]['quality'],
                'reference': ref, 'prediction': meta, 'independent_holdout_claim': False}
            row['within_10pct_of_reference_lift'] = {name: bool(valid and row['metrics']['Lift'] is not None
                and row['metrics']['Lift'] >= .9 * value['Lift']) for name, value in ref['models'].items()}
            rows.append(row)
    atomic(out / 'RESULTS.json', {'rows': rows, 'recommended_arm': primary,
        'final_pool': {'candidates': len(full_q), 'positives': int(y.sum()), 'cut': cut},
        'invalid_samples_are_not_resampled_or_promoted': True,
        'reference_is_selected_cohort_not_paired_equivalence': True})
    atomic(out / 'DONE.json', {'state': 'COMPLETE', 'utc': utc()})
    report(args.output)


def report(out):
    rows = []
    for ds in DATASETS:
        rows.extend(read(out / ds / 'RESULTS.json').get('rows', []))
    lines = ['# 服务器时序采样与模型比较', '',
        '真实比例定义：同一全节点历史图、同一预测窗口中，全部合法未闭合 K2 候选的正例/负例。比例因子使用正负 odds，不是只比较正例百分比。',
        '样本规模要求为冻结候选规模的 0.8–1.25 倍；默认保持完全相同。所有候选身份与模型预测先保存，再附最终标签。失败的最终样本不重抽、不用于性能恢复结论。',
        '冻结基线来自已选择的旧候选，跨采样集只能作为参考；同时报告 AP、Lift、实际正例数及按抽样概率加权的总体估计。历史窗口选出的推荐方案单独标注，不按最终成绩重新选样本。', '',
        '| 数据集 | 采样方案 | 历史推荐 | 状态 | P/N | 比例倍数 | 规模倍数 | AP | Lift | 加权 Lift |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        gate, met, weighted = row['audit'], row['metrics'], row['population_weighted_metrics']
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['recommended_before_final_labels']} | {row['state']} | {gate['sample_positives']}/{gate['sample_negatives']} | {gate['ratio_factor']} | {gate['size_factor']:.3f} | {met['AP']} | {met['Lift']} | {weighted['Lift']} |")
    lines.extend(['', '## 运行状态与失败', ''])
    for ds in read(out / 'PLAN.json').get('datasets', DATASETS):
        done = read(out / ds / 'DONE.json')
        failure = read(out / ds / 'FAILURE.json')
        skipped = sum(read(p).get('state') == 'SKIPPED' for p in (out / ds / 'trials').glob('*/*.json'))
        status = done.get('state', failure.get('state', read(out / ds / 'STATUS.json').get('stage', '未启动')))
        lines.append(f'- {ds}: {status}; 历史配置跳过 {skipped}; 最后失败信息 {failure or "无"}。')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    flat = [{'dataset': row['dataset'], 'arm': row['arm'], 'state': row['state'],
        'recommended': row['recommended_before_final_labels'], **row['metrics'],
        **{k: row['audit'][k] for k in ('sample_candidates', 'sample_positives', 'ratio_factor', 'size_factor')}} for row in rows]
    with (out / 'SUMMARY.csv').open('w', encoding='utf-8', newline='') as f:
        if flat:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def terminate(process):
    if process.poll() is not None:
        return
    if os.name == 'posix':
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def preflight(args, cfg):
    ref = read(HERE / 'FROZEN_REFERENCE.json')
    for ds in cfg['datasets']:
        for suffix, expected in ref['datasets'][ds]['raw_sha256'].items():
            path = args.raw_root / ds / f'{ds}-{suffix}.txt'
            if digest(path) != expected:
                raise ValueError(f'raw source hash mismatch: {path}')
        rr = ref['datasets'][ds]
        for name, key in [('train', 'train_candidates'), ('test', 'test_candidates')]:
            size = cfg[name + '_sizes'][ds]
            if not cfg['size_min'] <= size / rr[key] <= cfg['size_max']:
                raise ValueError(f'{ds} {name} size violates frozen-scale bounds')
    return {'state': 'PASSED', 'utc': utc(), 'raw_sha256_verified': True,
        'sampling_never_accepts_labels': True, 'datasets': cfg['datasets']}


def supervise(args, cfg):
    args.output.mkdir(parents=True, exist_ok=True)
    # File lock remains held for the supervisor lifetime; no double launches.
    lock = (args.output / 'supervisor.lock').open('a+b')
    if os.name == 'posix':
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        import msvcrt
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    stored = read(args.output / 'PLAN.json')
    comparable = {k: v for k, v in cfg.items() if k != 'plan_sha256'}
    if stored and stored != comparable:
        raise ValueError('this output directory belongs to a different plan; choose a new --output')
    if not stored:
        atomic(args.output / 'PLAN.json', comparable)
    cfg['plan_sha256'] = digest(args.output / 'PLAN.json')
    atomic(args.output / 'PREFLIGHT.json', preflight(args, cfg))
    import scipy
    import sklearn
    atomic(args.output / 'ENVIRONMENT.json', {'python': sys.version, 'platform': platform.platform(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__})
    if args.command == 'preflight':
        print(json.dumps(read(args.output / 'PREFLIGHT.json'), indent=2)); return
    run = read(args.output / 'RUN.json')
    if not run:
        run = {'start_epoch': time.time(), 'deadline_epoch': time.time() + args.hours * 3600,
            'start_utc': utc(), 'hours': args.hours, 'plan_sha256': cfg['plan_sha256']}
        atomic(args.output / 'RUN.json', run)
    atomic(args.output / 'EFFECTIVE_CONFIG.json', cfg)
    current = None
    try:
        for ds in cfg['datasets']:
            if read(args.output / ds / 'DONE.json').get('state') == 'COMPLETE':
                continue
            for attempt in (1, 2):
                remaining = run['deadline_epoch'] - time.time()
                if remaining <= 0:
                    break
                job_out = args.output / ds
                job_out.mkdir(parents=True, exist_ok=True)
                logfile = job_out / 'experiment.log'
                with logfile.open('a', encoding='utf-8') as log:
                    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--dataset', ds,
                        '--raw-root', str(args.raw_root), '--output', str(args.output)]
                    current = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=os.name == 'posix')
                    launch = time.time()
                    while current.poll() is None:
                        status = read(job_out / 'STATUS.json')
                        stale = time.time() - max(launch, status.get('epoch', launch)) > args.stale_seconds
                        expired = time.time() >= run['deadline_epoch'] or time.time() - launch >= args.job_seconds
                        atomic(args.output / 'WATCHDOG.json', {'state': 'RUNNING', 'dataset': ds,
                            'pid': current.pid, 'attempt': attempt, 'checked_utc': utc(),
                            'deadline_epoch': run['deadline_epoch'], 'last_job_status': status})
                        if stale or expired:
                            terminate(current)
                            atomic(job_out / 'FAILURE.json', {'attempt': attempt, 'reason': 'stale' if stale else 'time_limit', 'utc': utc()})
                            break
                        time.sleep(min(20., max(.01, run['deadline_epoch'] - time.time())))
                    code = current.wait()
                if code == 0 and (job_out / 'DONE.json').exists():
                    break
                atomic(job_out / 'FAILURE.json', {'attempt': attempt, 'exit_code': code,
                    'state': 'SKIPPED' if attempt == 2 else 'RETRY', 'utc': utc()})
            if time.time() >= run['deadline_epoch']:
                break
    finally:
        if current is not None:
            terminate(current)
        finished = all((args.output / ds / 'DONE.json').exists() for ds in cfg['datasets'])
        atomic(args.output / 'WATCHDOG.json', {'state': 'COMPLETE' if finished else
            ('TIME_LIMIT' if time.time() >= run['deadline_epoch'] else 'FINISHED_WITH_FAILURES'), 'checked_utc': utc()})
        report(args.output)
        lock.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'worker', 'preflight', 'report'))
    parser.add_argument('--raw-root', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=HERE / 'results/default')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--samplers', nargs='+', choices=SAMPLERS, default=list(SAMPLERS))
    parser.add_argument('--sample-seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--model-seeds', nargs='+', type=int, default=[41, 42, 43])
    parser.add_argument('--ndc-size', type=int, default=1602)
    parser.add_argument('--math-size', type=int, default=573)
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--job-seconds', type=int, default=14400)
    parser.add_argument('--stale-seconds', type=int, default=1800)
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    args.output = args.output.resolve()
    args.raw_root = args.raw_root.resolve()
    if args.command == 'report':
        report(args.output)
    elif args.command == 'worker':
        run_job(args, read(args.output / 'EFFECTIVE_CONFIG.json'))
    else:
        cfg = {'datasets': args.datasets, 'samplers': args.samplers, 'sample_seeds': args.sample_seeds,
            'model_seeds': args.model_seeds, 'test_sizes': {'NDC-substances': args.ndc_size, 'tags-math-sx': args.math_size},
            'train_sizes': {ds: reference(ds)['train_candidates'] for ds in DATASETS},
            'ratio_min': .5, 'ratio_max': 2., 'size_min': .8, 'size_max': 1.25,
            'models': configs(args.quick), 'raw_root': str(args.raw_root),
            'code_sha256': digest(Path(__file__).resolve()),
            'event_order': 'stable(timestamp,row_id)', 'timestamp_ties': 'whole_blocks',
            'true_ratio_reference': 'full-node history-legal K2 candidates in the same prediction horizon',
            'final_test_outcomes_used_for_tuning_or_resampling': False}
        supervise(args, cfg)
