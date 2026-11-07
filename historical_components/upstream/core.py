def aggregate(values, prefix, result):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError('Expected candidate-by-member feature matrix')
    pos = np.maximum(a, 0.0)
    result[f'{prefix}_min'] = a.min(axis=1)
    result[f'{prefix}_mean'] = a.mean(axis=1)
    result[f'{prefix}_max'] = a.max(axis=1)
    result[f'{prefix}_sum'] = a.sum(axis=1)
    result[f'{prefix}_std'] = a.std(axis=1)
    result[f'{prefix}_logsum'] = np.log1p(pos).sum(axis=1)
    result[f'{prefix}_geomean'] = np.exp(np.log1p(pos).mean(axis=1)) - 1.0
    result[f'{prefix}_harmonic'] = a.shape[1] / np.maximum(1e-12, (1.0 / (pos + 1e-12)).sum(axis=1))
    result[f'{prefix}_nonzero'] = (pos > 0).sum(axis=1)

def finite_matrix(features, names):
    x = np.column_stack([features[name] for name in names]).astype(np.float64, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    return np.sign(x) * np.log1p(np.minimum(np.abs(x), 1000000000000.0))

def score_metrics(y, score):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {'AP': None, 'Lift': None, 'ROC_AUC': None}
    ap = float(average_precision_score(y, score))
    return {'AP': ap, 'Lift': ap / float(y.mean()), 'ROC_AUC': float(roc_auc_score(y, score))}

def top_features(x, y, names, count):
    if len(np.unique(y)) < 2:
        return list(range(x.shape[1]))
    values = []
    for j in range(x.shape[1]):
        plus = average_precision_score(y, x[:, j])
        minus = average_precision_score(y, -x[:, j])
        values.append((max(plus, minus), names[j], j))
    return [j for _, _, j in sorted(values, reverse=True)[:min(count, len(values))]]
