def sha(value):
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()

def cutoff(events, fraction):
    n = min(len(events), max(1, int(len(events) * fraction)))
    if n == len(events):
        return n
    timestamp = events[n][0]
    while n and events[n - 1][0] == timestamp:
        n -= 1
    if not n:
        raise ValueError('Insufficient timestamp blocks')
    assert events[n - 1][0] < events[n][0]
    return n

def uniform_open_candidates(context, cap=CAP):
    adjacency = defaultdict(set)
    closed = set()
    for _, _, vertices in context:
        for a, b in combinations(vertices, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
        closed.update(combinations(vertices, 3))
    rng = random.Random(62026)
    candidates = []
    count = 0
    for a in sorted(adjacency):
        for b in sorted((n for n in adjacency[a] if n > a)):
            for c in sorted((n for n in adjacency[a] & adjacency[b] if n > b)):
                tri = (a, b, c)
                if tri in closed:
                    continue
                count += 1
                if len(candidates) < cap:
                    candidates.append(tri)
                else:
                    replacement = rng.randrange(count)
                    if replacement < cap:
                        candidates[replacement] = tri
    candidates.sort()
    assert all((q not in closed and all((b in adjacency[a] for a, b in combinations(q, 2))) for q in candidates))
    return (candidates, count, adjacency)

def attach_labels(future, candidates):
    lookup = {q: i for i, q in enumerate(candidates)}
    y = np.zeros(len(candidates), dtype=np.int8)
    for _, _, vertices in future:
        for q in combinations(vertices, 3):
            i = lookup.get(q)
            if i is not None:
                y[i] = 1
    return y
