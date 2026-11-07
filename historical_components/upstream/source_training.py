def train_data(snapshots, start, profile):
    global CURRENT_WEIGHT
    bundle = DATA[CURRENT]
    weights = None
    if profile in ('event', 'background'):
        data = bundle[profile]
        indices = np.flatnonzero(data['label_end'] <= start)
        if len(indices) > 30000:
            pos = indices[data['y'][indices] == 1]
            neg = indices[data['y'][indices] == 0]
            sampled = np.random.default_rng(2026).choice(neg, 30000 - len(pos), replace=False)
            indices = np.sort(np.r_[pos, sampled])
        x, y = (data['x'][indices], data['y'][indices])
        end = int(data['label_end'][indices].max())
        used = [{'label_end': end}]
    else:
        allowed = [v for v in bundle['dense'] if int(v['label_end']) <= start]
        if profile == 'coarse':
            allowed = [v for v in allowed if v['fraction'] in s.STARTS]
        elif profile == 'latest3':
            allowed = allowed[-3:]
        if not allowed:
            raise ValueError('No completed historical label windows')
        x = np.concatenate([v['x'] for v in allowed])
        y = np.concatenate([v['y'] for v in allowed])
        used = allowed
        if profile == 'unique':
            q = np.concatenate([v['q'] for v in allowed])
            _, inverse, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
            weights = 1 / counts[inverse]
            weights = weights / weights.mean()
    if len(np.unique(y)) < 2:
        raise ValueError('Too few historical training classes')
    assert max((int(v['label_end']) for v in used)) <= start
    CURRENT_WEIGHT = weights
    return (x, y, used)

def columns(names, view):
    if view == 'all':
        return list(range(len(names)))
    prefixes = ('event_edge_count_', 'event_node_count_', 'event_edge_group_', 'event_node_group_', 'event_edge_closures_', 'event_node_closures_', 'event_coactivity_', 'event_jaccard_', 'event_common_', 'event_age_edge_', 'event_decay_edge_0.05_', 'event_decay_edge_group_0.05_')
    return [i for i, n in enumerate(names) if n.startswith(prefixes) and n.endswith(('_min', '_harmonic', '_mean'))]
