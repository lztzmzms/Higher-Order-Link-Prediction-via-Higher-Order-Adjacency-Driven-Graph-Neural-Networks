import numpy as np
from sklearn.metrics import average_precision_score

def evaluate(labels, predictions):
    labels, predictions = (np.asarray(labels), np.asarray(predictions))
    if labels.ndim != 1 or labels.shape != predictions.shape or (not np.isfinite(predictions).all()):
        raise ValueError('Invalid labels/predictions')
    if not np.isin(labels, [0, 1]).all() or not 0 < labels.sum() < len(labels):
        raise ValueError('Binary candidate set must contain both classes')
    ap = float(average_precision_score(labels, predictions))
    return dict(AP=ap, Lift=ap / float(labels.mean()))
