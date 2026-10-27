import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import io
import zipfile

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
DATASETS = (
    'email-Enron', 'email-Eu', 'contact-primary-school', 'contact-high-school',
    'NDC-substances', 'tags-ask-ubuntu', 'tags-math-sx', 'threads-math-sx',
)
METHODS = ('Simplicial_CN', 'RA', 'Local Path', 'katz', 'AA', 'PA', 'Global AA', 'Global RA', 'Degree')
DEFAULT_METHODS = METHODS[:-1]


def read_events(dataset, raw_dir=None):
    folder = Path(raw_dir).resolve() if raw_dir else PACKAGE / 'backend/protocol' / dataset / 'dataset/raw_data' / dataset
    paths = [folder / f'{dataset}-{suffix}.txt' for suffix in ('nverts', 'times', 'simplices')]
    if all(p.is_file() for p in paths):
        counts, times, vertices = [np.loadtxt(p, dtype=np.int64, ndmin=1) for p in paths]
        inputs = paths
    elif raw_dir:
        raise FileNotFoundError('Missing input: ' + ', '.join(str(p) for p in paths if not p.is_file()))
    else:
        path = PACKAGE / 'backend/material' / dataset / 'EVENTS.npz'
        prepared = ROOT / 'data' / dataset / 'events.npz'
        if not prepared.is_file():
            prepared = PACKAGE / 'data' / dataset / 'events.npz'
        archive = PACKAGE / 'data' / dataset / 'inputs.zip'
        if path.is_file() or prepared.is_file():
            path = path if path.is_file() else prepared
            with np.load(path, allow_pickle=False) as z:
                counts, times, vertices = z['counts'], z['times'], z['vertices']
            inputs = [path]
        elif archive.is_file():
            with zipfile.ZipFile(archive) as z:
                prefix = f'protocol/dataset/raw_data/{dataset}/'
                names = [prefix + f'{dataset}-{suffix}.txt' for suffix in ('nverts','times','simplices')]
                counts, times, vertices = [np.loadtxt(io.BytesIO(z.read(name)), dtype=np.int64, ndmin=1) for name in names]
            inputs = [archive]
        else:
            raise FileNotFoundError(f'No input events for {dataset}; provide --raw-dir.')
    if len(counts) != len(times) or np.any(counts < 0) or int(counts.sum()) != len(vertices):
        raise ValueError('Event counts, timestamps and simplex vertices do not align.')
    if len(counts) < 2:
        raise ValueError('At least two events are required.')
    cut = int(0.8 * len(counts))
    signature = hashlib.sha256()
    for a in (counts, times, vertices):
        signature.update(np.ascontiguousarray(a, dtype='<i8').tobytes())
    info = dict(dataset=dataset, events=len(counts), vertices=len(vertices),
                observation_events=cut, future_events=len(counts) - cut,
                split_ratio=0.8, event_order='input',
                input_files=[str(p) for p in inputs], input_sha256=signature.hexdigest())
    return counts, times, vertices, info


def event_tables(counts, vertices):
    nodes, edges, triangles = set(), set(), set()
    offset = 0
    for count in counts:
        event = sorted(set(int(x) for x in vertices[offset:offset + int(count)]))
        offset += int(count)
        nodes.update((v,) for v in event)
        edges.update(itertools.combinations(event, 2))
        triangles.update(itertools.combinations(event, 3))
    return [sorted(nodes), sorted(edges), sorted(triangles)]


def open_triangles(tables):
    adjacent = {u[0]: set() for u in tables[0]}
    for u, v in tables[1]:
        adjacent[u].add(v)
        adjacent[v].add(u)
    closed = set(tables[2])
    candidates = []
    for u, v in tables[1]:
        for w in sorted(adjacent[u] & adjacent[v]):
            if w > v and (u, v, w) not in closed:
                candidates.append((u, v, w))
    return candidates


def prepare(counts, vertices, info, cache):
    cut = info['observation_events']
    offset = int(counts[:cut].sum())
    old = event_tables(counts[:cut], vertices[:offset])
    new = event_tables(counts[cut:], vertices[offset:])
    id_maps = [{q: i for i, q in enumerate(table)} for table in old]
    row, col = [], []
    for j, triangle in enumerate(old[2]):
        for edge in itertools.combinations(triangle, 2):
            row.append(id_maps[1][edge])
            col.append(j)
    incidence = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(old[1]), len(old[2])))
    adjacency = (incidence @ incidence.T).tocsr()
    degree = adjacency.diagonal().copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    reciprocal = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree != 0)
    log_reciprocal = np.zeros_like(degree)
    mask = degree > 1
    log_reciprocal[mask] = 1 / np.log(degree[mask])
    candidates = open_triangles(old)
    future = set(new[2])
    labels = np.asarray([q in future for q in candidates], dtype=np.int64)
    if not len(labels):
        raise ValueError('The observation window has no open triangles.')
    cache.mkdir(parents=True, exist_ok=True)
    for name, obj in [('simplex_tables_old', old), ('simplex_tables_new', new), ('id_maps', id_maps)]:
        with (cache / f'{name}.pkl').open('wb') as f:
            pickle.dump(obj, f, protocol=4)
    for name, matrix in [('lower_adj', adjacency), ('Degree', sp.diags(degree)),
                         ('Degree_inverse', sp.diags(reciprocal)), ('Degree_log_inverse', sp.diags(log_reciprocal))]:
        sp.save_npz(cache / f'{name}.npz', matrix.tocsr())
    np.savez_compressed(cache / 'CANDIDATES.npz', triplets=np.asarray(candidates, dtype=np.int64), y_true=labels)
    info = dict(info, candidates=len(labels), positives=int(labels.sum()))
    (cache / 'INPUT.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info


def choosing_sim(adj, degree, reciprocal, log_reciprocal, method):
    eye = sp.eye(adj.shape[0], format='csc')
    if method == 'Simplicial_CN':
        return (adj @ adj).tocsr()
    if method == 'Local Path':
        return (adj @ adj + 0.001 * (adj @ adj @ adj)).tocsr()
    if method == 'katz':
        return (inv((eye - 0.01 * adj).tocsc()) - eye).tocsr()
    if method == 'RA':
        return (adj @ reciprocal @ adj).tocsr()
    if method == 'Global RA':
        return (adj @ inv((eye - 0.01 * reciprocal @ adj).tocsc()) - adj).tocsr()
    if method == 'AA':
        return (adj @ log_reciprocal @ adj).tocsr()
    if method == 'Global AA':
        return (adj @ inv((eye - 0.01 * log_reciprocal @ adj).tocsc()) - adj).tocsr()
    if method == 'PA':
        values = degree.diagonal()
        return values
    if method == 'Degree':
        return degree.diagonal()
    raise ValueError(method)


def candidate_scores(candidates, id_maps, matrix, method):
    ids = np.asarray([[id_maps[1][edge] for edge in itertools.combinations(tuple(q), 2)]
                      for q in candidates], dtype=np.int64)
    if method == 'Degree':
        return matrix[ids].sum(axis=1)
    if method == 'PA':
        values = matrix[ids]
        return values[:, 0] * values[:, 1] + values[:, 0] * values[:, 2] + values[:, 1] * values[:, 2]
    score = np.zeros(len(candidates), dtype=np.float64)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        score += np.asarray(matrix[ids[:, i], ids[:, j]]).ravel()
    return score


def evaluate(cache, methods, out):
    with (cache / 'id_maps.pkl').open('rb') as f:
        id_maps = pickle.load(f)
    with np.load(cache / 'CANDIDATES.npz', allow_pickle=False) as z:
        candidates, labels = z['triplets'], z['y_true']
    if not 0 < float(labels.mean()) <= 1:
        raise ValueError('Lift requires a positive prevalence in (0, 1].')
    adj, degree, reciprocal, log_reciprocal = [sp.load_npz(cache / f'{name}.npz') for name in
                                             ('lower_adj', 'Degree', 'Degree_inverse', 'Degree_log_inverse')]
    results = []
    out.mkdir(parents=True, exist_ok=True)
    for method in methods:
        matrix = choosing_sim(adj, degree, reciprocal, log_reciprocal, method)
        scores = candidate_scores(candidates, id_maps, matrix, method)
        if not np.isfinite(scores).all():
            raise ValueError(f'Nonfinite scores for {method}.')
        ap = float(average_precision_score(labels, scores))
        result = dict(method=method, candidates=len(labels), positives=int(labels.sum()),
                      prevalence=float(labels.mean()), AP=ap, Lift=ap / float(labels.mean()))
        slug = method.replace(' ', '_')
        np.savez_compressed(out / f'{slug}_PREDICTIONS.npz', triplets=candidates, y_true=labels, score=scores)
        (out / f'{slug}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


def main(default_stage='run'):
    parser = argparse.ArgumentParser(description='Run the Hodge-Laplacian heuristics with AP and Lift.')
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--raw-dir', type=Path, help='Directory containing <dataset>-{nverts,times,simplices}.txt.')
    parser.add_argument('--method', choices=METHODS, action='append', help='Repeat to run several methods; default: all eight original methods.')
    parser.add_argument('--stage', choices=('run', 'prepare', 'evaluate'), default=default_stage)
    parser.add_argument('--check', action='store_true', help='Validate input events without constructing matrices.')
    parser.add_argument('--cache-dir', type=Path, help='Preprocessed input directory.')
    parser.add_argument('--output-dir', type=Path, help='Directory for new predictions and metrics.')
    args = parser.parse_args()
    counts, times, vertices, info = read_events(args.dataset, args.raw_dir)
    if args.check:
        print(json.dumps(dict(status='PASS', **info), indent=2))
        return
    cache = args.cache_dir.resolve() if args.cache_dir else PACKAGE / 'results/heuristics_inputs' / args.dataset / info['input_sha256'][:16]
    out = args.output_dir.resolve() if args.output_dir else PACKAGE / 'results/heuristics' / args.dataset / info['input_sha256'][:16]
    manifest = cache / 'INPUT.json'
    if manifest.is_file():
        stored = json.loads(manifest.read_text(encoding='utf-8'))
        if stored['input_sha256'] != info['input_sha256']:
            raise ValueError('Cache inputs do not match this event file set; choose another --cache-dir.')
    elif args.stage == 'evaluate':
        raise FileNotFoundError(f'Run --stage prepare first; no input manifest at {manifest}.')
    else:
        prepare(counts, vertices, info, cache)
    if args.stage == 'prepare':
        print(json.dumps(dict(status='PREPARED', cache_dir=str(cache))))
    else:
        evaluate(cache, args.method or DEFAULT_METHODS, out)


if __name__ == '__main__':
    main()
