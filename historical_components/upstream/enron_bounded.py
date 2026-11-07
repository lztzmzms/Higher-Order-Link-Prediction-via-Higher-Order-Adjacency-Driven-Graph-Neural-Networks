def bounded_order(priority, max_move):
    pass
    size = len(priority)
    remaining = set(range(size))
    order = []
    for slot in range(size):
        available = [i for i in remaining if i - max_move <= slot]
        urgent = [i for i in available if i + max_move <= slot]
        chosen = min(urgent) if urgent else max(available, key=lambda i: (priority[i], -i))
        assert abs(chosen - slot) <= max_move
        order.append(chosen)
        remaining.remove(chosen)
    return np.asarray(order, dtype=int)

def rearrange(base, local, count, alpha, max_move):
    pool = high_pool(base, count)
    if alpha == 0:
        return (base.copy(), {'pool_size': len(pool), 'moved_candidates': 0, 'maximum_move': 0})
    priority = (1 - alpha) * rankdata(base[pool]) / len(pool) + alpha * rankdata(local[pool]) / len(pool)
    order = bounded_order(priority, max_move)
    score = base.copy()
    score[pool[order]] = base[pool]
    positions = np.empty(len(pool), dtype=int)
    positions[order] = np.arange(len(pool))
    movement = np.abs(positions - np.arange(len(pool)))
    full_order = np.argsort(-score, kind='stable')
    new_position = np.empty(len(base), dtype=int)
    new_position[full_order] = np.arange(len(base))
    old_order = np.argsort(-base, kind='stable')
    old_position = np.empty(len(base), dtype=int)
    old_position[old_order] = np.arange(len(base))
    effective = np.abs(new_position[pool] - old_position[pool])
    tail = np.setdiff1d(np.arange(len(base)), pool)
    fallback = bool(np.max(effective, initial=0) > max_move or not np.array_equal(new_position[tail], old_position[tail]))
    if fallback:
        score = base.copy()
        movement[:] = 0
        effective[:] = 0
    assert np.array_equal(score[tail], base[tail])
    assert np.array_equal(np.sort(score[pool]), np.sort(base[pool]))
    assert np.max(effective, initial=0) <= max_move
    return (score, {'pool_size': len(pool), 'moved_candidates': int(np.sum(effective > 0)), 'maximum_move': int(np.max(effective, initial=0)), 'tie_constraint_fallback': fallback, 'unchanged_tail': len(tail)})

def localized(record, cfg):
    result = {**record, 'x': record['x'].copy()}
    if cfg['feature_mode'] == 'rank':
        result['x'] = np.column_stack([rankdata(record['x'][:, j]) / len(record['x']) for j in range(record['x'].shape[1])])
    order = np.argsort(-record['base'], kind='stable')
    position = np.empty(len(order), dtype=int)
    position[order] = np.arange(len(order))
    result['x'][:, 0] = np.maximum(0.0, 1 - position / min(cfg['pool'], len(order)))
    return result
