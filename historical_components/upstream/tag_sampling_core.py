import numpy as np

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
    pass
    rng = np.random.default_rng(seed)
    n = min(n, len(y))
    pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
    np_ = int(round(n * len(pos) / len(y)))
    selected = np.r_[rng.choice(pos, min(np_, len(pos)), replace=False),
        rng.choice(neg, n - min(np_, len(pos)), replace=False)]
    return np.sort(selected)
