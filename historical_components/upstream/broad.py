def novel_features(context, candidates):
    pass
    last = context[-1][0]
    start = context[0][0]
    span = max(1, last - start)
    edges = sorted({edge for q in candidates for edge in itertools.combinations(q, 2)})
    ix = {edge: i for i, edge in enumerate(edges)}
    qix = np.asarray([[ix[edge] for edge in itertools.combinations(q, 2)] for q in candidates])
    decay = np.zeros((2, len(edges), len(TAUS)))
    hist = {bins: np.zeros((2, len(edges), bins)) for bins in (20, 64, 256)}
    node_events = Counter()
    edge_events = Counter()
    closure_node = Counter()
    closure_edge = Counter()
    novel_node = Counter()
    novel_edge = Counter()
    closure_decay_node = Counter()
    closure_decay_edge = Counter()
    known_edges = set()
    closed = set()
    for timestamp, block in itertools.groupby(context, key=lambda row: row[0]):
        new_edges = set()
        new_triples = set()
        for _, _, vs in block:
            node_events.update(vs)
            weight = np.exp(-max(0, last - timestamp) / np.maximum(1, np.asarray(TAUS) * span))
            for edge in itertools.combinations(vs, 2):
                edge_events[edge] += 1
                new_edges.add(edge)
                ei = ix.get(edge)
                if ei is not None:
                    decay[0, ei] += weight
                    if len(vs) >= 3:
                        decay[1, ei] += weight
                    for bins, values in hist.items():
                        bi = min(bins - 1, int(bins * (timestamp - start) / span))
                        values[0, ei, bi] += 1
                        if len(vs) >= 3:
                            values[1, ei, bi] += 1
            new_triples.update(itertools.combinations(vs, 3))
        for tri in new_triples - closed:
            es = list(itertools.combinations(tri, 2))
            novel_node.update(tri)
            novel_edge.update(es)
            if all((edge in known_edges for edge in es)):
                closure_node.update(tri)
                closure_edge.update(es)
                w = math.exp(-max(0, last - timestamp) / max(1, 0.1 * span))
                for v in tri:
                    closure_decay_node[v] += w
                for edge in es:
                    closure_decay_edge[edge] += w
        closed.update(new_triples)
        known_edges.update(new_edges)
    out = {}
    for mode, label in enumerate(('all', 'group')):
        for ti, tau in enumerate(TAUS):
            tag = str(tau).replace('.', 'p')
            s.r.aggregate(decay[mode, :, ti][qix], f'continue_decay_{label}_{tag}', out)
        for bins, values in hist.items():
            a = values[mode][qix]
            p = a / np.maximum(1e-12, a.sum(axis=2, keepdims=True))
            cos = []
            overlap = []
            for i, j in ((0, 1), (0, 2), (1, 2)):
                cos.append(np.sum(p[:, i] * p[:, j], axis=1) / np.maximum(1e-12, np.linalg.norm(p[:, i], axis=1) * np.linalg.norm(p[:, j], axis=1)))
                overlap.append(np.minimum(p[:, i], p[:, j]).sum(axis=1))
            s.r.aggregate(np.column_stack(cos), f'continue_edge_sync_cos_{label}_{bins}', out)
            s.r.aggregate(np.column_stack(overlap), f'continue_edge_sync_overlap_{label}_{bins}', out)
            out[f'continue_edge_sync_triple_{label}_{bins}'] = p.min(axis=1).sum(axis=1)
            smoothed = a + np.concatenate([np.zeros_like(a[:, :, :1]), a[:, :, :-1]], axis=2) + np.concatenate([a[:, :, 1:], np.zeros_like(a[:, :, :1])], axis=2)
            out[f'continue_edge_sync_smooth_{label}_{bins}'] = smoothed.min(axis=1).sum(axis=1)
    for kind, queries, counts, novel, eligible, decayed in (('node', candidates, node_events, novel_node, closure_node, closure_decay_node), ('edge', [[edge for edge in itertools.combinations(q, 2)] for q in candidates], edge_events, novel_edge, closure_edge, closure_decay_edge)):
        for label, values in (('novel', novel), ('eligible_closures', eligible), ('closure_decay', decayed), ('novel_per_event', {v: novel[v] / max(1, counts[v]) for v in counts}), ('closure_per_event', {v: eligible[v] / max(1, counts[v]) for v in counts}), ('eligible_fraction', {v: eligible[v] / max(1, novel[v]) for v in counts})):
            s.r.aggregate(np.asarray([[values.get(v, 0) for v in q] for q in queries]), f'continue_{kind}_{label}', out)
    return out
