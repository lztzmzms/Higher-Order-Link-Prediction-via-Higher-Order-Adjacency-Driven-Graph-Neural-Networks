import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from models import dual
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school', 'tags-ask-ubuntu', 'NDC-substances', 'tags-math-sx', 'threads-math-sx')
MODELS = ('Dual-Mapping', 'MLSHoL')

def config(dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    return json.loads((ROOT / 'configs' / f'{dataset}.json').read_text(encoding='utf-8'))

def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

@lru_cache(maxsize=5)
def events(dataset):
    values = load_arrays(ROOT / 'data' / dataset / 'events.npz')
    times, counts, vertices = (values[k] for k in ('times', 'counts', 'vertices'))
    offsets = np.r_[0, np.cumsum(counts)]
    if len(times) != len(counts) or offsets[-1] != len(vertices) or np.any(times[1:] < times[:-1]):
        raise ValueError('Original event positions or order changed')
    return [(int(t), i, tuple(map(int, vertices[offsets[i]:offsets[i + 1]]))) for i, t in enumerate(times)]

def closure_labels(rows, q):
    wanted = set(map(tuple, q))
    closed = set()
    for _, _, vertices in rows:
        closed.update((tri for tri in combinations(sorted(set(vertices)), 3) if tri in wanted))
    return np.asarray([tuple(tri) in closed for tri in q], dtype=np.int64)

def validate_candidates(rows, q):
    adjacency, closed = ({}, set())
    for _, _, vertices in rows:
        for node in vertices:
            adjacency.setdefault(node, set())
        for a, b in combinations(sorted(set(vertices)), 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
        closed.update(combinations(sorted(set(vertices)), 3))
    if len(q) != len(set(map(tuple, q))):
        raise ValueError('Duplicate original candidate')
    for tri in map(tuple, q):
        if tri != tuple(sorted(tri)) or tri in closed or (not all((b in adjacency.get(a, ()) for a, b in combinations(tri, 2)))):
            raise ValueError(f'Invalid prefix-only open triangle: {tri}')

def evaluation_labels(dataset, q):
    cfg = config(dataset)
    return closure_labels(events(dataset)[cfg['target_cut']:cfg['label_end']], q)

def geometry(dataset, model, device):
    import torch
    from data.complexes import PackedComplexes
    folder = ROOT / 'data' / dataset
    blocks = {r: load_arrays(folder / f'dual_{r}.npz') for r in ('fit', 'target')}
    rw = {}
    for key in ('local', 'heuristic', 'rw'):
        scaler = StandardScaler().fit(blocks['fit'][key])
        for role, block in blocks.items():
            block[key] = torch.tensor(scaler.transform(block[key]), dtype=torch.float32, device=device)
            if key == 'rw':
                rw[role] = block[key]
    for block in blocks.values():
        block['edge_index'] = torch.tensor(block['edge_index'], dtype=torch.long, device=device)
        if 'edge_weight' in block:
            block['edge_weight'] = torch.tensor(block['edge_weight'], dtype=torch.float32, device=device)
    if model == 'Dual-Mapping':
        return (blocks, rw)
    return ({r: PackedComplexes(folder / f'complexes_{r}.npz') for r in ('fit', 'target')}, rw)

def rebuild_dual(dataset, model, prefix):
    pass
    cfg = config(dataset)
    source = load_arrays(ROOT / 'data' / dataset / f'{model}.npz')
    role = 'train' if prefix == 'fit' else 'target'
    cut = cfg['train_cut' if prefix == 'fit' else 'target_cut']
    q = list(map(tuple, source[role + '_q']))
    edges, triangles = (set(), set())
    for _, _, vertices in events(dataset)[:cut]:
        edges.update(combinations(sorted(set(vertices)), 2))
        triangles.update(combinations(sorted(set(vertices)), 3))
    graph = dual.build_dual_graph(q)
    features, raw = dual.build_features(q, dict(edges=edges, triangles=triangles), graph, return_raw=True)
    return dict(local=dual.select_feature_block(features, dual.FEATURE_NAMES, dual.local_feature_names()), heuristic=dual.build_heuristic_feature_block(raw), rw=dual.build_random_walk_structural_features(graph['edge_index'], len(q), rw_steps=cfg['random_walk']['steps']), edge_index=graph['edge_index'])

def rebuild_complex(dataset, prefix, index):
    pass
    import networkx as nx
    from data.complexes import PackedComplexes
    from data.local_complex import remove_target_link, construct_labeled_networkx_graph, to_pyg_data
    from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi
    cfg = config(dataset)
    source = load_arrays(ROOT / 'data' / dataset / 'MLSHoL.npz')
    role = 'train' if prefix == 'fit' else 'target'
    tri = tuple(map(int, source[role + '_q'][index]))
    cut = cfg['train_cut' if prefix == 'fit' else 'target_cut']
    packed = PackedComplexes(ROOT / 'data' / dataset / f'complexes_{prefix}.npz')
    ordered_nodes = list(map(int, packed.original_nodes(index)))
    nodes = set(ordered_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(ordered_nodes)
    observed = set()
    for _, _, vertices in events(dataset)[:cut]:
        local = sorted(set(vertices) & nodes)
        graph.add_edges_from(combinations(local, 2))
        if len(local) >= 3:
            observed.add(tuple(sorted(set(vertices))))
    local = remove_target_link(graph, tri)
    labeled = construct_labeled_networkx_graph(local, observed, tri, set(), disconnected_policy='unreachable_zero', observed_2_skeleton=cfg['mls_observed_2_skeleton'])
    pyg = to_pyg_data(labeled)
    return convert_graph_dataset_with_gudhi([pyg], expansion_dim=2, include_down_adj=False, init_method='sum', expand_cliques=cfg['mls_expand_cliques'])[0]
