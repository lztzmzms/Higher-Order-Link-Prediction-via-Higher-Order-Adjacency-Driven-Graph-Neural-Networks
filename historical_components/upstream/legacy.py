def train(target, cfg):
    bundle = a.DATA[a.CURRENT]
    original = bundle['dense']
    profile = cfg['profile']
    if profile == 'dense20':
        bundle['dense'] = [v for v in original if v['fraction'] >= 0.2]
    elif profile in ('last1', 'last2'):
        allowed = [v for v in original if int(v['label_end']) <= int(target['cut'])]
        bundle['dense'] = allowed[-int(profile[-1]):]
    try:
        x, y, used = a.train_data([], int(target['cut']), 'dense' if profile in ('dense20', 'last1', 'last2') else profile)
    finally:
        bundle['dense'] = original
    weights = np.ones(len(y)) if a.CURRENT_WEIGHT is None else a.CURRENT_WEIGHT.copy()
    if cfg['recent_weight']:
        if profile in ('event', 'background'):
            ends = None
        else:
            ends = np.concatenate([np.full(len(v['y']), int(v['label_end'])) for v in used])
        if ends is not None:
            weights *= np.exp(cfg['recent_weight'] * (ends / int(target['cut']) - 1.0))
    if cfg['balanced']:
        p = float(np.average(y, weights=weights))
        weights *= np.where(y == 1, 0.5 / p, 0.5 / (1 - p))
    assert max((int(v['label_end']) for v in used)) <= int(target['cut'])
    return (x, y, weights, {'max_training_label_end': max((int(v['label_end']) for v in used)), 'prediction_cut': int(target['cut']), 'train_positives': int(y.sum()), 'train_rows': len(y)})

def cols(names, view):
    if view in ('all', 'compact'):
        return a.columns(names, view)
    prefixes = ('event_decay_', 'event_age_', 'event_tenure_') if view == 'decay' else ('event_edge_count_', 'event_node_count_', 'event_edge_group_', 'event_node_group_', 'event_edge_closures_', 'event_node_closures_', 'event_common_', 'event_coactivity_')
    return [i for i, n in enumerate(names) if n.startswith(prefixes)]

def estimator(cfg, seed):
    if cfg['family'] in ('ET', 'RF'):
        cls = ExtraTreesClassifier if cfg['family'] == 'ET' else RandomForestClassifier
        return cls(n_estimators=100, min_samples_leaf=cfg['leaf'], max_depth=cfg['depth'] or None, max_features=cfg['fraction'], random_state=seed, n_jobs=1)
    if cfg['family'] == 'HG':
        return HistGradientBoostingClassifier(max_iter=cfg['iterations'], learning_rate=cfg['learning_rate'], max_leaf_nodes=cfg['leaf'] + 3, min_samples_leaf=max(5, cfg['leaf']), l2_regularization=cfg['l2'], random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], max_iter=1500, random_state=seed))

def predict(target, cfg, seeds=(41,)):
    x, y, weights, audit = train(target, cfg)
    indices = cols(target['names'].tolist(), cfg['view'])
    assert indices
    xt, xv = (x[:, indices], target['x'][:, indices])
    if cfg['rank']:
        transformed = []
        for matrix in (xt, xv):
            transformed.append(np.column_stack([np.searchsorted(np.sort(xt[:, j]), matrix[:, j], side='right') / len(xt) for j in range(xt.shape[1])]).astype(np.float32))
        xt, xv = transformed
    values = []
    for seed in seeds:
        model = estimator(cfg, seed)
        if hasattr(model, 'steps'):
            model.fit(xt, y, **{model.steps[-1][0] + '__sample_weight': weights})
        else:
            model.fit(xt, y, sample_weight=weights)
        values.append(model.predict_proba(xv)[:, 1])
    score = np.mean(values, axis=0)
    assert np.all(np.isfinite(score))
    return (score, audit)
