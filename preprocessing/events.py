import json
from itertools import combinations
from pathlib import Path
import numpy as np

def load_events(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            times, counts, vertices = (z[k] for k in ('times', 'counts', 'vertices'))
            if (len(times) != len(counts) or np.any(counts < 0) or
                    np.any(counts != counts.astype(int)) or sum(counts) != len(vertices)):
                raise ValueError('Invalid event array offsets')
            offsets = np.r_[0, np.cumsum(counts)].astype(int)
            records = [dict(timestamp=float(t), nodes=vertices[offsets[i]:offsets[i+1]].tolist())
                       for i, t in enumerate(times)]
    else:
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids, rows = {}, []
    for i, record in enumerate(records):
        t, nodes = float(record['timestamp']), record['nodes']
        if not np.isfinite(t) or not isinstance(nodes, list):
            raise ValueError('Expected finite numeric timestamp and nodes list')
        if any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in nodes):
            raise ValueError('Node IDs must be strings or integers')
        keys = [(type(v).__name__, str(v)) for v in nodes]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate node in an event')
        for key in keys:
            if key not in ids:
                ids[key] = len(ids)
        rows.append((t, i, tuple(sorted(ids[k] for k in keys))))
    if not rows or any(a[0] > b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('Events must be nonempty and chronological')
    return rows, {v: dict(type=k[0], id=k[1]) for k, v in ids.items()}

def open_triangles(rows):
    adjacency, closed = {}, set()
    for _, _, nodes in rows:
        for v in nodes:
            adjacency.setdefault(v, set())
        for u, v in combinations(nodes, 2):
            adjacency[u].add(v)
            adjacency[v].add(u)
        closed.update(combinations(nodes, 3))
    candidates = []
    for u in sorted(adjacency):
        for v in sorted(w for w in adjacency[u] if w > u):
            for w in sorted(adjacency[u] & adjacency[v]):
                if w > v and (u, v, w) not in closed:
                    candidates.append((u, v, w))
    return np.asarray(candidates, dtype=np.int64).reshape(-1, 3)

def labels(rows, q):
    wanted = set(map(tuple, q))
    closed = set()
    for _, _, nodes in rows:
        closed.update(t for t in combinations(nodes, 3) if t in wanted)
    return np.asarray([tuple(t) in closed for t in q], dtype=np.int64)
