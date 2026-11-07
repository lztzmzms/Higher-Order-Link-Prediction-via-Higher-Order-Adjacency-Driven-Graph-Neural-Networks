def rank(value):
    return rankdata(value, method='average') / len(value)

def signal_values(prefix, q):
    pass

    size = len(prefix)
    node_count = Counter()
    edge_count = Counter()
    edge_rows = defaultdict(list)
    hist = defaultdict(lambda: np.zeros(48, dtype=np.float64))
    for index, (timestamp, _, vertices) in enumerate(prefix):
        nodes = tuple(sorted(set(vertices)))
        node_count.update(nodes)
        for edge in combinations(nodes, 2):
            edge_count[edge] += 1
            edge_rows[edge].append(index)
            hist[edge][timestamp % 86400 // 1800] += 1
    rows = {edge: np.asarray(indices, dtype=np.int64) for edge, indices in edge_rows.items()}
    empty = np.empty(0, dtype=np.int64)
    normalized_hist = {edge: values / max(1.0, float(values.sum())) for edge, values in hist.items()}
    zero = np.zeros(48, dtype=np.float64)
    output = {name: np.empty(len(q), dtype=np.float64) for name in SIGNALS}
    radii = {value: max(1, int(round(size * value))) for value in (0.005, 0.01)}

    def close_pairs(left, right, radius):
        if not len(left) or not len(right):
            return 0
        return int(np.sum(np.searchsorted(right, left + radius, side='right') - np.searchsorted(right, left - radius, side='left')))
    for index, triangle in enumerate(q):
        edges = tuple(combinations(tuple(map(int, triangle)), 2))
        ratios = np.asarray([edge_count[e] / max(1.0, min(node_count[e[0]], node_count[e[1]])) for e in edges])
        output[SIGNALS[0]][index] = np.log1p(ratios).prod()
        profiles = [normalized_hist.get(edge, zero) for edge in edges]
        output[SIGNALS[1]][index] = min((float(np.minimum(profiles[a], profiles[b]).sum()) for a, b in ((0, 1), (0, 2), (1, 2))))
        for fraction, field in ((0.01, SIGNALS[2]), (0.005, SIGNALS[3])):
            overlaps = np.asarray([close_pairs(rows.get(edges[a], empty), rows.get(edges[b], empty), radii[fraction]) for a, b in ((0, 1), (0, 2), (1, 2))], dtype=np.float64)
            transformed = np.log1p(overlaps)
            output[field][index] = np.log1p(transformed).sum() if fraction == 0.01 else transformed.mean()
    return output

def direction_selection(values, y, label_end, prediction_cut):
    assert label_end <= prediction_cut
    directions = {}
    evidence = {}
    for name in SIGNALS:
        x = np.nan_to_num(values[name])
        positive = float(average_precision_score(y, x))
        negative = float(average_precision_score(y, -x))
        directions[name] = 1 if positive >= negative else -1
        evidence[name] = dict(sign=directions[name], positive_AP=positive, negative_AP=negative)
    return (directions, dict(direction_selection_label_end=label_end, prediction_cut=prediction_cut, signals=evidence))

def replay(recipe, values, directions):

    def signal(name):
        return rank(directions[name] * np.nan_to_num(values[name])).astype(np.float32)

    def combine(a, b, formula):
        if formula == 'mean75':
            return 0.75 * a + 0.25 * b
        if formula == 'minimum':
            return np.minimum(a, b)
        raise ValueError(formula)

    def recurse(item):
        if item['stage'] == 'pair':
            return combine(signal(item['first']), signal(item['second']), item['formula'])
        return combine(rank(recurse(item['pair'])).astype(np.float32), signal(item['third']), item['formula'])
    return rank(recurse(recipe))
