def eligible(records, target):
    allowed = [r for r in records if r['label_end'] is not None and r['label_end'] <= target['cut']]
    assert all((r['cut'] < target['cut'] for r in allowed))
    return allowed

def fit_score(w, records, target, cfg, seeds=(41,), force=False):
    key = w.b.s.sha({'cut': target['cut'], 'config': cfg, 'seeds': seeds})[:20]
    path = OUT / 'meta_cache' / f'{key}.npz'
    if path.exists() and (not force):
        with np.load(path, allow_pickle=False) as z:
            return (z['local'], n.read(path.with_suffix('.json')))
    past = eligible(records, target)
    assert past
    pools = [high_pool(r['base'], cfg['pool']) for r in past]
    x = np.concatenate([r['x'][p] for r, p in zip(past, pools)])
    y = np.concatenate([r['y'][p] for r, p in zip(past, pools)])
    q = np.concatenate([r['q'][p] for r, p in zip(past, pools)])
    origin = np.concatenate([np.full(len(p), i) for i, p in enumerate(pools)])
    _, inv, count = np.unique(np.column_stack([q, y]), axis=0, return_inverse=True, return_counts=True)
    weight = 1 / count[inv]
    weight /= weight.mean()
    gate = {'max_meta_label_end': max((int(r['label_end']) for r in past)), 'prediction_cut': target['cut'], 'origins': [r['fraction'] for r in past], 'training_rows': len(y), 'training_positives': int(y.sum()), 'distinct_positive_triangles': len(np.unique(q[y == 1], axis=0)), 'training_backbone_scores_are_out_of_time': True, 'full_cohort_preserved': True}
    if len(np.unique(y)) < 2:
        local = target['base'].copy()
        gate['fallback'] = 'insufficient_classes'
    else:
        selected = w.b.r.top_features(x, y, target['names'], cfg['features'])
        selected = list(dict.fromkeys([0] + list(selected)))
        xt = x[:, selected]
        xv = target['x'][:, selected]
        scaler = StandardScaler()
        xt = scaler.fit_transform(xt)
        xv = scaler.transform(xv)
        values = []
        for seed in seeds:
            if cfg['family'] == 'HG':
                model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=15, l2_regularization=cfg['strength'], learning_rate=0.03, early_stopping=False, random_state=seed)
                model.fit(xt, y, sample_weight=weight)
                values.append(model.predict_proba(xv)[:, 1])
            elif cfg['family'] == 'LR':
                model = LogisticRegression(C=cfg['strength'], max_iter=1000, random_state=seed)
                model.fit(xt, y, sample_weight=weight)
                values.append(model.decision_function(xv))
            else:
                rng = np.random.default_rng(seed)
                left = []
                right = []
                pw = []
                for oi in np.unique(origin):
                    pos = np.flatnonzero((origin == oi) & (y == 1))
                    neg = np.flatnonzero((origin == oi) & (y == 0))
                    if not len(pos) or not len(neg):
                        continue
                    for pi in pos:
                        ni = rng.choice(neg, min(32, len(neg)), replace=False)
                        left.extend([pi] * len(ni))
                        right.extend(ni.tolist())
                        pw.extend([weight[pi]] * len(ni))
                if not left:
                    values.append(target['base'])
                    continue
                keep = rng.choice(len(left), min(12000, len(left)), replace=False)
                diff = xt[np.asarray(left)[keep]] - xt[np.asarray(right)[keep]]
                pair_weight = np.asarray(pw)[keep]
                model = LogisticRegression(C=cfg['strength'], fit_intercept=False, max_iter=1000, random_state=seed)
                model.fit(np.concatenate([diff, -diff]), np.r_[np.ones(len(diff)), np.zeros(len(diff))], sample_weight=np.r_[pair_weight, pair_weight])
                values.append(model.decision_function(xv))
        local = np.mean(values, axis=0)
        gate['selected_features'] = [target['names'][i] for i in selected]
    assert np.all(np.isfinite(local))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, local=local, q=target['q'])
    n.atomic(path.with_suffix('.json'), gate)
    return (local, gate)
