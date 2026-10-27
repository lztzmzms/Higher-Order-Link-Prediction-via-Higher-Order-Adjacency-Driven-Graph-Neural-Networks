import numpy as np
from types import SimpleNamespace
from pipelines.history_sampling_core import quota_sample, constraint
s=SimpleNamespace(quota_sample=quota_sample,constraint=constraint)

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
