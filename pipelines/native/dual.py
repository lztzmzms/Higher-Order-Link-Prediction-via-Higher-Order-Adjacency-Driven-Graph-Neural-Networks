import os
import re
import time
import json
import pickle
import random
import hashlib
import argparse
import math
import csv
import importlib.util
import gc
import faulthandler
import sys
import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from pipelines.native.k2_provenance import validate_feature_provenance

faulthandler.enable()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.cuda.amp import autocast, GradScaler

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_curve, auc,
    average_precision_score, fbeta_score, matthews_corrcoef
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
try:
    from torch_geometric.loader import NeighborLoader
except Exception:
    NeighborLoader = None
try:
    from torch_geometric.utils import add_self_loops, k_hop_subgraph
except Exception:
    from torch_geometric.utils import add_self_loops
    k_hop_subgraph = None
from torch_geometric.nn.conv import GATConv

try:
    from torch_geometric.nn.pool import global_mean_pool
except ImportError:
    from torch_geometric.nn import global_mean_pool

torch.backends.cudnn.benchmark = True

def set_global_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"WARNING: deterministic algorithms request failed: {e}")

def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

def resolve_device(device_arg):
    device_arg = str(device_arg or 'auto').strip().lower()
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg.startswith('cuda') and not torch.cuda.is_available():
        print(f"WARNING: requested device={device_arg}, but CUDA is not available; falling back to cpu.")
        return torch.device('cpu')
    return torch.device(device_arg)

def _path_or_none(path_text):
    if path_text is None:
        return None
    path = Path(path_text)
    if path.exists():
        return path
    if not path.is_absolute():
        root_path = Path.cwd() / path
        if root_path.exists():
            return root_path
    return None

def _infer_protocol_dataset_name(args):
    if getattr(args, 'dataset', None):
        return str(args.dataset)

    patterns = [
        r'^(?P<name>.+)_openTriangles_(?:train|val|test|all)\.txt$',
        r'^(?P<name>.+)_closeTriangles_(?:train|val|test|all)\.txt$',
        r'^(?P<name>.+)_new_closures_(?:train|val|test|all)\.txt$',
        r'^classified_network_(?P<name>.+)_(?:train|all|test)(?:_\d+)?\.gpickle$',
        r'^(?P<name>.+)-nverts\.txt$',
    ]
    for value in (getattr(args, 'target', None), getattr(args, 'priority_nodes', None),
                  getattr(args, 'train', None), getattr(args, 'test', None),
                  getattr(args, 'nverts', None)):
        if not value:
            continue
        basename = Path(str(value)).name
        for pattern in patterns:
            match = re.match(pattern, basename)
            if match:
                return match.group('name')
    raise ValueError(
        "input_mode=raw/protocol requires --dataset, or paths whose filenames include the dataset name "
        "(for example dataset/test/<dataset>_openTriangles_test.txt)."
    )

def _dataset_name_from_protocol_filename(path_text):
    if not path_text:
        return None
    basename = Path(str(path_text)).name
    for pattern in (
        r'^(?P<name>.+)_openTriangles_(?:train|val|test|all)\.txt$',
        r'^(?P<name>.+)_closeTriangles_(?:train|val|test|all)\.txt$',
        r'^(?P<name>.+)_new_closures_(?:train|val|test|all)\.txt$',
    ):
        match = re.match(pattern, basename)
        if match:
            return match.group('name')
    return None

def _candidate_protocol_roots(args):
    roots = []
    protocol_root = getattr(args, 'protocol_root', None)
    if protocol_root:
        roots.append(Path(protocol_root))

    for value in (getattr(args, 'target', None), getattr(args, 'priority_nodes', None)):
        path = _path_or_none(value)
        if path is None:
            continue
        parent = path.parent
        roots.append(parent.parent)
        if parent.parent.name in {'train', 'test', 'val'}:
            roots.append(parent.parent.parent)

    processed_dir = getattr(args, 'processed_dir', None)
    if processed_dir:
        roots.append(Path(processed_dir))

    roots.append(Path('dataset'))
    roots.append(Path('processed'))
    out = []
    seen = set()
    for root in roots:
        root = root if root.is_absolute() else Path.cwd() / root
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out

def _protocol_dataset_name_aliases(dataset):
    aliases = [str(dataset)]
    for src, dst in (('_', '-'), ('-', '_')):
        alias = str(dataset).replace(src, dst)
        if alias not in aliases:
            aliases.append(alias)
    return aliases

def _resolve_protocol_triangle_path(args, dataset, split, kind):
    short_filename = f"{kind}_{split}.txt"
    candidates = []
    for root in _candidate_protocol_roots(args):
        for dataset_name in _protocol_dataset_name_aliases(dataset):
            filename = f"{dataset_name}_{kind}_{split}.txt"
            candidates.extend([
                root / split / filename,
                root / split / dataset_name / filename,
                root / split / dataset_name / short_filename,
                root / dataset_name / split / filename,
                root / dataset_name / split / short_filename,
            ])
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Protocol triangle file not found for dataset={dataset}, split={split}, kind={kind}. "
        f"Tried:\n  {tried}"
    )

def _unique_extend(items, seen, out, sources, source):
    for item in items:
        tri = tuple(sorted(map(int, item)))
        if tri not in seen:
            seen.add(tri)
            out.append(tri)
        sources[tri].add(source)

def _build_legacy_protocol_gpickle_graph(
        close_path, open_path, new_closures_path, split_name,
        feature_mode='ids', closed_label=1, graph_triangle_scope='legacy_closed_open'):
    pass
    try:
        from pipelines.native.generate_dual_mapping_gpickle import (
            read_triangles, build_face_index, build_lower_adjacency_edges,
            make_features,
        )
        import networkx as nx
    except Exception as exc:
        raise RuntimeError("Failed to import protocol gpickle helpers.") from exc

    close_triangles = read_triangles(Path(close_path))
    open_triangles = read_triangles(Path(open_path))
    new_closures = read_triangles(Path(new_closures_path))
    new_set = set(new_closures)

    triplets = []
    seen = set()
    sources = defaultdict(set)
    open_set = set(open_triangles)
    if graph_triangle_scope == 'open_only':

        _unique_extend(open_triangles, seen, triplets, sources, 'open')
        for tri in new_closures:
            if tri in open_set and tri in sources:
                sources[tri].add('new_closure')
    else:
        _unique_extend(close_triangles, seen, triplets, sources, 'closed')

        _unique_extend(open_triangles, seen, triplets, sources, 'open')
        for tri in new_closures:
            if tri in sources:
                sources[tri].add('new_closure')

    face_to_nodes = build_face_index(triplets)
    derived_edges = build_lower_adjacency_edges(face_to_nodes)
    graph = nx.Graph()
    for node_id, tri in enumerate(triplets):
        tri_sources = sources[tri]
        if graph_triangle_scope == 'open_only':
            label = 1 if 'new_closure' in tri_sources else 0
        elif split_name == 'test' and 'new_closure' in tri_sources:
            label = 1
        else:
            label = int(closed_label) if 'closed' in tri_sources else 0
        graph.add_node(
            node_id,
            triplet=tri,
            label=int(label),
            features=make_features(tri, node_id, face_to_nodes, feature_mode),
            sources=tuple(sorted(tri_sources)),
        )
    graph.add_edges_from(derived_edges)

    labels = [data['label'] for _, data in graph.nodes(data=True)]
    stats = {
        'split': split_name,
        'close_lines': len(close_triangles),
        'close_unique': len(set(close_triangles)),
        'open_lines': len(open_triangles),
        'open_unique': len(set(open_triangles)),
        'new_closure_lines': len(new_closures),
        'new_closure_unique': len(set(new_closures)),
        'derived_nodes': graph.number_of_nodes(),
        'derived_edges': graph.number_of_edges(),
        'label_0': labels.count(0),
        'label_1': labels.count(1),
        'feature_dim': int(next(iter(graph.nodes(data=True)))[1]['features'].shape[0])
        if graph.number_of_nodes()
        else 0,
        'lower_faces': len(face_to_nodes),
        'max_triangles_per_face': max((len(v) for v in face_to_nodes.values()), default=0),
        'label_mode': 'legacy_node_classification',
        'graph_triangle_scope': graph_triangle_scope,
    }
    graph.graph.update({
        'dual_mapping': True,
        'split': split_name,
        'edge_rule': 'lower_adjacency_shared_1_face',
        'label_rule': (
            'train: closed=1/open=0; test: closed and future closures=1, '
            'remaining open=0'
        ),
        'feature_mode': feature_mode,
        'graph_triangle_scope': graph_triangle_scope,
        'stats': dict(stats),
    })
    return graph, stats

def _generate_protocol_gpickles_for_raw_mode(args, base_result_folder):
    pass
    try:
        from pipelines.native.generate_dual_mapping_gpickle import write_gpickle
    except Exception as exc:
        raise RuntimeError("Failed to import generate_dual_mapping_gpickle.py for raw protocol preprocessing.") from exc

    dataset = _infer_protocol_dataset_name(args)
    output_dir = Path(getattr(args, 'generated_gpickle_dir', None) or Path(base_result_folder) / 'generated_gpickle')
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_mode = getattr(args, 'generated_gpickle_feature_mode', 'ids')
    closed_label = int(getattr(args, 'generated_gpickle_closed_label', 1))
    graph_triangle_scope = getattr(args, 'protocol_graph_triangle_scope', 'legacy_closed_open')
    jobs = [
        ('train', output_dir / f"classified_network_{dataset}_train_1.gpickle"),
        ('test', output_dir / f"classified_network_{dataset}_all_1.gpickle"),
    ]

    stats_by_split = {}
    for split, output_path in jobs:
        generation_started = time.perf_counter()
        close_path = _resolve_protocol_triangle_path(args, dataset, split, 'closeTriangles')
        open_path = _resolve_protocol_triangle_path(args, dataset, split, 'openTriangles')
        new_path = _resolve_protocol_triangle_path(args, dataset, split, 'new_closures')
        graph, stats = _build_legacy_protocol_gpickle_graph(
            close_path=close_path,
            open_path=open_path,
            new_closures_path=new_path,
            split_name=split,
            feature_mode=feature_mode,
            closed_label=closed_label,
            graph_triangle_scope=graph_triangle_scope,
        )
        write_gpickle(graph, output_path, overwrite=True, backup_existing=False)
        stats.update({
            'close_path': str(close_path),
            'open_path': str(open_path),
            'new_closures_path': str(new_path),
            'output_path': str(output_path),
            'generation_seconds': float(time.perf_counter() - generation_started),
        })
        stats_by_split[split] = stats
        print(
            f"generated {split} gpickle: {output_path} | "
            f"nodes={stats['derived_nodes']} edges={stats['derived_edges']} "
            f"label0={stats['label_0']} label1={stats['label_1']} "
            f"feature_dim={stats['feature_dim']}"
        )

    stats_path = output_dir / 'generation_stats.json'
    with stats_path.open('w', encoding='utf-8') as handle:
        json.dump({'dataset': dataset, 'splits': stats_by_split}, handle, indent=2)
    args.train = str(jobs[0][1])
    args.test = str(jobs[1][1])
    preserve_cli_paths = parse_bool(getattr(args, 'protocol_use_cli_target_paths', False))
    target_dataset = _dataset_name_from_protocol_filename(getattr(args, 'target', None))
    priority_dataset = _dataset_name_from_protocol_filename(getattr(args, 'priority_nodes', None))
    if (not preserve_cli_paths
            or not getattr(args, 'target', None)
            or (target_dataset is not None and target_dataset != dataset)):
        args.target = str(_resolve_protocol_triangle_path(args, dataset, 'test', 'openTriangles'))
    if (not preserve_cli_paths
            or not getattr(args, 'priority_nodes', None)
            or (priority_dataset is not None and priority_dataset != dataset)):
        args.priority_nodes = str(_resolve_protocol_triangle_path(args, dataset, 'train', 'new_closures'))
    print(f"protocol target: {args.target}")
    print(f"protocol priority_nodes: {args.priority_nodes}")
    print(f"protocol gpickle generation stats: {stats_path}")

def parse_pos_ratio(s):
    if s is None:
        return None
    s = str(s).strip()
    if '/' in s:
        a, b = s.split('/', 1)
        try:
            val = float(a) / float(b)
        except Exception:
            raise argparse.ArgumentTypeError(f"无法解析 pos_ratio: {s}")
    else:
        try:
            val = float(s)
        except Exception:
            raise argparse.ArgumentTypeError(f"无法解析 pos_ratio: {s}")
    if val <= 0:
        return 0.0
    if val > 1:
        val = 1.0
    return val

def parse_seeds(s):
    if s is None:
        return list(range(48, 57))
    s = str(s).strip()
    seeds = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                a = int(a); b = int(b)
                if a <= b:
                    seeds.update(range(a, b + 1))
                else:
                    seeds.update(range(b, a + 1))
            except Exception:
                raise argparse.ArgumentTypeError(f"无法解析 seeds 段: {part}")
        else:
            try:
                seeds.add(int(part))
            except Exception:
                raise argparse.ArgumentTypeError(f"无法解析 seeds 项: {part}")
    return sorted(seeds)

def _file_hash(path, block_size=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _stable_hash_payload(payload):
    data = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.md5(data).hexdigest()

def load_g_as_arrays(gpickle_path, cache_dir=".cache_graph"):
    os.makedirs(cache_dir, exist_ok=True)
    sig = _file_hash(gpickle_path)
    cache_path = os.path.join(cache_dir, f"{os.path.basename(gpickle_path)}.{sig}.npz")

    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        arr_features = z["features"]
        arr_labels = z["labels"]
        triplets = z["triplets"]
        edges = z["edges"]
        num_nodes = int(z["num_nodes"])
        return arr_features, arr_labels, triplets, edges, num_nodes

    import pickle as pkl
    with open(gpickle_path, 'rb') as f:
        G = pkl.load(f)

    required = ("features", "label", "triplet")
    for n, data in G.nodes(data=True):
        for k in required:
            if k not in data:
                raise ValueError(f"节点 {n} 缺少必要属性: {k}")

    num_nodes = G.number_of_nodes()
    nodes_sorted = list(G.nodes())
    id_map = {n: i for i, n in enumerate(nodes_sorted)}

    sample_feat = G.nodes[nodes_sorted[0]]["features"]
    feat_dim = len(sample_feat) if hasattr(sample_feat, "__len__") else 1
    arr_features = np.empty((num_nodes, feat_dim), dtype=np.float32)
    arr_labels = np.empty((num_nodes,), dtype=np.int64)
    triplets = np.empty((num_nodes,), dtype=object)

    for i, n in enumerate(nodes_sorted):
        nd = G.nodes[n]
        arr_features[i] = np.asarray(nd["features"], dtype=np.float32)
        arr_labels[i] = int(nd["label"])
        t = tuple(sorted(map(int, nd["triplet"])))
        triplets[i] = t

    edges = np.array([(id_map[u], id_map[v]) for u, v in G.edges()], dtype=np.int64)

    np.savez_compressed(cache_path,
                        features=arr_features,
                        labels=arr_labels,
                        triplets=triplets,
                        edges=edges,
                        num_nodes=np.array(num_nodes))
    return arr_features, arr_labels, triplets, edges, num_nodes

class TripletNetworkFast:
    def __init__(self, triplets):
        self.triplet_to_node = {t: i for i, t in enumerate(triplets)}
        self.node_to_triplet = list(triplets)
        self.index = {}
        for i, t in enumerate(triplets):
            a, b, c = t
            for k in (a, b, c):
                if k not in self.index:
                    self.index[k] = set()
                self.index[k].add(i)

    def find_node(self, input_nums):
        if len(input_nums) != 3:
            raise ValueError("必须输入三个整数")
        query = tuple(sorted(map(int, input_nums)))
        node = self.triplet_to_node.get(query)
        if node is not None:
            return node
        sets = [self.index.get(k, set()) for k in query]
        if not all(sets):
            return None
        cand = set.intersection(*sets)
        for n in cand:
            if self.node_to_triplet[n] == query:
                return n
        return None

    def get_triplet_by_node(self, node_id):
        if 0 <= node_id < len(self.node_to_triplet):
            return self.node_to_triplet[node_id]
        return None

def load_target_triplets(file_path):
    target_triplets = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                numbers = re.findall(r'\d+', line)
                if len(numbers) >= 3:
                    triplet = tuple(sorted(map(int, numbers[:3])))
                    target_triplets.append(triplet)
    except FileNotFoundError:
        print(f"错误: 目标三元组文件不存在 {file_path}")
        raise
    return target_triplets

def _controlled_array_hash(array):
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()

def _load_raw_event_vertices(dataset, event_order='raw', data_root=None):
    base_root = Path(data_root).resolve() if data_root else Path(__file__).resolve().parent
    raw_dir = base_root / 'dataset' / 'raw_data' / str(dataset)
    nverts_path = raw_dir / f'{dataset}-nverts.txt'
    simplices_path = raw_dir / f'{dataset}-simplices.txt'
    times_path = raw_dir / f'{dataset}-times.txt'
    if not nverts_path.is_file() or not simplices_path.is_file() or not times_path.is_file():
        raise FileNotFoundError(
            f'Interaction-strength raw inputs not found for {dataset}: {raw_dir}'
        )
    counts = [int(line) for line in nverts_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    flat = [int(line) for line in simplices_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    times = [int(line) for line in times_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(counts) != len(times):
        raise ValueError(f'Malformed raw simplices for {dataset}: len(nverts) != len(times)')
    if sum(counts) != len(flat):
        raise ValueError(f'Malformed raw simplices for {dataset}: sum(nverts) != len(flat)')
    rows = []
    offset = 0
    for row_id, (count, timestamp) in enumerate(zip(counts, times)):
        rows.append((timestamp, row_id, tuple(sorted(set(flat[offset:offset + count])))))
        offset += count
    if event_order == 'timestamp':
        rows.sort(key=lambda row: (row[0], row[1]))
    elif event_order != 'raw':
        raise ValueError(f'Unsupported protocol event order: {event_order}')
    return [row[2] for row in rows]

def _context_pair_counts(events, fraction):
    end = int(float(fraction) * len(events))
    pair_counts = defaultdict(int)
    for simplex in events[:end]:
        for edge in combinations(simplex, 2):
            pair_counts[tuple(sorted(map(int, edge)))] += 1
    return pair_counts, end

def _triplet_log_strength_features(triplets, pair_counts):
    values = np.zeros((len(triplets), 3), dtype=np.float32)
    for row, triangle in enumerate(triplets):
        edge_values = sorted(
            math.log1p(pair_counts.get(tuple(sorted(map(int, edge))), 0))
            for edge in combinations(tuple(map(int, triangle)), 2)
        )
        values[row] = np.asarray(edge_values, dtype=np.float32)
    return values

def _standardize_strength_features(train_values, test_values):
    mean = np.asarray(train_values, dtype=np.float64).mean(axis=0)
    std = np.asarray(train_values, dtype=np.float64).std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    train_z = ((np.asarray(train_values) - mean) / std).astype(np.float32)
    test_z = ((np.asarray(test_values) - mean) / std).astype(np.float32)
    return train_z, test_z, mean, std

def _derived_edge_strength_weights(triplets, edges, pair_counts, scale=1.0):
    weights = np.ones((len(edges),), dtype=np.float32)
    for edge_id, (left, right) in enumerate(np.asarray(edges, dtype=np.int64)):
        shared = tuple(sorted(set(map(int, triplets[int(left)])) & set(map(int, triplets[int(right)]))))
        if len(shared) == 2:
            weights[edge_id] = 1.0 + math.log1p(pair_counts.get(shared, 0))
    scale = float(scale) if float(scale) > 1e-12 else 1.0
    return (weights / scale).astype(np.float32)

def build_controlled_strength_inputs(
    dataset, tr_trip, te_trip, tr_edges, te_edges,
    event_order='raw', data_root=None,
    train_context_fraction=0.6, test_context_fraction=0.8,
):
    events = _load_raw_event_vertices(
        dataset, event_order=event_order, data_root=data_root
    )
    train_counts, train_end = _context_pair_counts(events, train_context_fraction)
    test_counts, test_end = _context_pair_counts(events, test_context_fraction)
    train_raw = _triplet_log_strength_features(tr_trip, train_counts)
    test_raw = _triplet_log_strength_features(te_trip, test_counts)
    train_features, test_features, feature_mean, feature_std = _standardize_strength_features(
        train_raw, test_raw
    )
    train_edge_raw = _derived_edge_strength_weights(tr_trip, tr_edges, train_counts)
    edge_scale = float(train_edge_raw.mean()) if train_edge_raw.size else 1.0
    train_edge_weight = _derived_edge_strength_weights(
        tr_trip, tr_edges, train_counts, scale=edge_scale
    )
    test_edge_weight = _derived_edge_strength_weights(
        te_trip, te_edges, test_counts, scale=edge_scale
    )
    audit = {
        'definition': f'per-pair log1p(historical event count), {event_order} event order',
        'train_context_fraction': float(train_context_fraction),
        'test_context_fraction': float(test_context_fraction),
        'train_context_events': int(train_end),
        'test_context_events': int(test_end),
        'feature_mean_train_only': feature_mean.tolist(),
        'feature_std_train_only': feature_std.tolist(),
        'edge_scale_train_only': edge_scale,
        'train_triplet_hash': _controlled_array_hash(np.asarray([tuple(x) for x in tr_trip], dtype=np.int64)),
        'test_triplet_hash': _controlled_array_hash(np.asarray([tuple(x) for x in te_trip], dtype=np.int64)),
        'train_edge_hash': _controlled_array_hash(np.asarray(tr_edges, dtype=np.int64)),
        'test_edge_hash': _controlled_array_hash(np.asarray(te_edges, dtype=np.int64)),
        'train_strength_feature_hash': _controlled_array_hash(train_features),
        'test_strength_feature_hash': _controlled_array_hash(test_features),
        'train_edge_weight_hash': _controlled_array_hash(train_edge_weight),
        'test_edge_weight_hash': _controlled_array_hash(test_edge_weight),
    }
    return train_features, test_features, train_edge_weight, test_edge_weight, audit

def parse_triplet_key(text):
    numbers = re.findall(r'\d+', str(text))
    if len(numbers) < 3:
        raise ValueError(f"无法解析 triplet: {text!r}")
    tri = tuple(sorted(map(int, numbers[:3])))
    if len(set(tri)) != 3:
        raise ValueError(f"非法 triplet: {text!r}")
    return tri

def load_edge_maxsimplex_features(txt_path):
    feature_dict = {}
    feature_dim = None
    duplicate_count = 0
    with open(txt_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames is None or 'triplet' not in reader.fieldnames:
            raise ValueError(f"{txt_path} 必须是包含 triplet 表头的 TSV 文件")
        feature_names = [name for name in reader.fieldnames if name != 'triplet']
        feature_dim = len(feature_names)
        for row_no, row in enumerate(reader, start=2):
            tri = parse_triplet_key(row.get('triplet', ''))
            values = []
            for name in feature_names:
                text = row.get(name, '')
                values.append(float(text) if text not in ('', None) else 0.0)
            if tri in feature_dict:
                duplicate_count += 1
            feature_dict[tri] = np.asarray(values, dtype=np.float32)
            if feature_dim != len(values):
                raise ValueError(f"{txt_path}:{row_no} 特征维度不一致")
    if feature_dim is None:
        raise ValueError(f"{txt_path} 为空或没有表头")
    if duplicate_count:
        print(f"WARNING: {txt_path} 中有 {duplicate_count} 个重复 triplet，已使用最后一次出现的特征")
    print(f"Loaded edge maximal-simplex features: {txt_path}, rows={len(feature_dict)}, dim={feature_dim}")
    return feature_dict, int(feature_dim)

def align_edge_maxsimplex_features(triplets, feature_dict, feature_dim, split_name='data', return_stats=False):
    extra = np.zeros((len(triplets), feature_dim), dtype=np.float32)
    matched = 0
    missing = 0
    for i, tri in enumerate(triplets):
        key = tuple(sorted(map(int, tri)))
        values = feature_dict.get(key)
        if values is None:
            missing += 1
            continue
        if values.shape[0] != feature_dim:
            raise ValueError(f"{split_name} triplet {key} 的最大单纯形特征维度不一致")
        extra[i] = values
        matched += 1
    if missing:
        print(f"WARNING: {split_name} missing edge maximal-simplex features for {missing}/{len(triplets)} triplets; filled zeros")
    print(f"{split_name} edge maximal-simplex feature match: matched={matched}, missing={missing}, dim={feature_dim}")
    if return_stats:
        return extra, {'matched': int(matched), 'missing': int(missing), 'feature_dim': int(feature_dim)}
    return extra

def maxsimplex_matrix_stats(matrix, prefix):
    if matrix is None:
        return {
            f'{prefix}_max_feat_nonzero_rate': np.nan,
            f'{prefix}_max_feat_all_zero_rows': 0,
        }
    arr = np.asarray(matrix)
    row_nonzero = np.any(arr != 0, axis=1) if arr.size else np.zeros((arr.shape[0],), dtype=bool)
    return {
        f'{prefix}_max_feat_nonzero_rate': float(np.mean(arr != 0)) if arr.size else 0.0,
        f'{prefix}_max_feat_all_zero_rows': int(np.sum(~row_nonzero)),
    }

def load_aligned_edge_maxsimplex_matrices(tr_trip, te_trip, args):
    if not args.train_edge_maxsimplex_txt or not args.test_edge_maxsimplex_txt:
        raise ValueError(
            "edge maximal-simplex fusion requires --train_edge_maxsimplex_txt "
            "and --test_edge_maxsimplex_txt"
        )
    train_dict, train_dim = load_edge_maxsimplex_features(args.train_edge_maxsimplex_txt)
    test_dict, test_dim = load_edge_maxsimplex_features(args.test_edge_maxsimplex_txt)
    if train_dim != test_dim:
        raise ValueError(f"训练/测试最大单纯形特征维度不一致: {train_dim} != {test_dim}")

    tr_max_feat, train_align_stats = align_edge_maxsimplex_features(
        tr_trip, train_dict, train_dim, split_name='train', return_stats=True
    )
    te_max_feat, test_align_stats = align_edge_maxsimplex_features(
        te_trip, test_dict, test_dim, split_name='test', return_stats=True
    )
    stats = {
        'max_feat_dim': int(train_dim),
        'train_max_feat_matched': int(train_align_stats['matched']),
        'train_max_feat_missing': int(train_align_stats['missing']),
        'test_max_feat_matched': int(test_align_stats['matched']),
        'test_max_feat_missing': int(test_align_stats['missing']),
    }
    stats.update(maxsimplex_matrix_stats(tr_max_feat, 'train'))
    stats.update(maxsimplex_matrix_stats(te_max_feat, 'test'))
    print(
        "edge maximal-simplex matrices: "
        f"dim={train_dim}, "
        f"train_nonzero_rate={stats['train_max_feat_nonzero_rate']:.6f}, "
        f"test_nonzero_rate={stats['test_max_feat_nonzero_rate']:.6f}, "
        f"train_all_zero_rows={stats['train_max_feat_all_zero_rows']}, "
        f"test_all_zero_rows={stats['test_max_feat_all_zero_rows']}"
    )
    return tr_max_feat, te_max_feat, int(train_dim), stats

def apply_edge_maxsimplex_ablation(tr_max_feat, te_max_feat, args):
    mode = getattr(args, 'edge_maxsimplex_ablation', 'real')
    if mode == 'real':
        return tr_max_feat, te_max_feat
    if mode == 'zero':
        print("Applying edge maximal-simplex ablation: zero")
        return np.zeros_like(tr_max_feat), np.zeros_like(te_max_feat)
    if mode == 'shuffle':
        print(f"Applying edge maximal-simplex ablation: shuffle seed={args.maxsimplex_shuffle_seed}")
        rng = np.random.default_rng(int(args.maxsimplex_shuffle_seed))
        tr_perm = rng.permutation(tr_max_feat.shape[0])
        te_perm = rng.permutation(te_max_feat.shape[0])
        return tr_max_feat[tr_perm], te_max_feat[te_perm]
    raise ValueError(f"Unknown edge_maxsimplex_ablation: {mode}")

def validate_open_candidate_maxsimplex_values(te_trip, target_nodes, te_max_feat, feature_path, column):
    pass
    with Path(feature_path).open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
    if column not in header:
        raise ValueError(f"Maximal-simplex column {column!r} not found in {feature_path}")
    feature_index = header.index(column) - 1
    if feature_index < 0 or feature_index >= te_max_feat.shape[1]:
        raise ValueError(f"Invalid maximal-simplex column index for {column!r}")
    indices = np.asarray(list(target_nodes), dtype=np.int64)
    values = te_max_feat[indices, feature_index] if indices.size else np.empty(0, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {column} values on open candidates")
    nonzero = np.flatnonzero(values != 0.0)
    if nonzero.size:
        raise ValueError(
            f"Future/protocol mismatch in maximal-simplex cache {feature_path}: "
            f"{nonzero.size}/{values.size} open candidates have {column} != 0"
        )

def infer_dataset_name_from_target(target_path):
    if not target_path:
        return None
    stem = Path(target_path).stem
    for marker in ("_openTriangles", "_new_closures", "_closures"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return None

def infer_edge_maxsimplex_raw_dir(args):
    raw_dir_arg = getattr(args, 'edge_maxsimplex_raw_dir', None)
    if raw_dir_arg:
        raw_dir = Path(raw_dir_arg)
        if raw_dir.exists():
            return raw_dir

    dataset_name = infer_dataset_name_from_target(getattr(args, 'target', None))
    candidates = []
    if dataset_name:
        target_path = Path(args.target)
        if target_path.parent.name in ('test', 'train', 'val'):
            dataset_root = target_path.parent.parent
            candidates.append(dataset_root / 'raw_data' / dataset_name)
        candidates.append(Path.cwd() / 'dataset' / 'raw_data' / dataset_name)
        candidates.append(Path(__file__).resolve().parent / 'dataset' / 'raw_data' / dataset_name)
        for root_value in (
            getattr(args, 'protocol_data_root', None),
            getattr(args, 'processed_dir', None),
        ):
            if root_value:
                candidates.append(Path(root_value) / 'dataset' / 'raw_data' / dataset_name)

    for raw_dir in candidates:
        if raw_dir.exists():
            return raw_dir
    return None

def load_edge_maxsimplex_generator_module():
    module_path = Path(__file__).resolve().with_name('generate_edge_maxsimplex_features.py')
    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot auto-generate edge maximal-simplex features because {module_path} is missing"
        )
    spec = importlib.util.spec_from_file_location('generate_edge_maxsimplex_features_auto', str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def edge_feature_existing_candidates(requested_path, split_name):
    candidates = []
    if requested_path:
        p = Path(requested_path)
        candidates.append(p)
        candidates.append(p.with_suffix('.tsv'))
        candidates.append(p.with_suffix('.txt'))
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        candidates.extend([
            base / '.cache_features' / f'{split_name}_edge_maxsimplex_features.tsv',
            base / '.cache_features' / f'{split_name}_edge_maxsimplex_features.txt',
            base / 'dataset' / 'processed' / f'{split_name}_edge_maxsimplex_features.tsv',
            base / 'dataset' / 'processed' / f'{split_name}_edge_maxsimplex_features.txt',
        ])
    seen = set()
    out = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out

def resolve_existing_edge_maxsimplex_txt(requested_path, split_name):
    if requested_path:
        candidate = Path(requested_path)
        return str(candidate) if candidate.exists() else None
    return None

def default_edge_maxsimplex_cache_path(args, split_name):
    dataset = str(getattr(args, 'dataset', '') or infer_dataset_name_from_target(getattr(args, 'target', None)) or 'dataset')
    safe_dataset = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in dataset)
    return str(Path.cwd() / '.cache_features' / f'{safe_dataset}_{split_name}_edge_maxsimplex_features.tsv')

def generate_edge_maxsimplex_txt(raw_dir, gpickle_path, output_txt, context_ratio, event_order='raw'):
    generator = load_edge_maxsimplex_generator_module()
    raw_dir = Path(raw_dir)
    gpickle_path = Path(gpickle_path)
    output_txt = Path(output_txt)
    print(
        "Auto-generating edge maximal-simplex features: "
        f"raw_dir={raw_dir}, gpickle={gpickle_path}, output={output_txt}, "
        f"context_ratio={context_ratio}, event_order={event_order}"
    )
    records = generator.load_timestamped_simplices(raw_dir, event_order=event_order)
    context_records = generator.select_context(records, float(context_ratio), event_order)
    context_simplices = [record[2] for record in context_records]
    maximal_simplices = generator.extract_maximal_simplices(context_simplices)
    closure_count = generator.count_closure_simplices(maximal_simplices)
    edge_to_max_ids = generator.build_edge_index(maximal_simplices)
    triplets = generator.read_gpickle_triplets(gpickle_path)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    missing = 0
    with output_txt.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(generator.header())
        for triplet in triplets:
            features, has_match = generator.triplet_features(triplet, edge_to_max_ids, maximal_simplices)
            matched += int(has_match)
            missing += int(not has_match)
            writer.writerow(["|".join(str(v) for v in triplet)] + [f"{v:.10g}" for v in features])

    provenance = generator.build_feature_provenance(
        raw_dir,
        raw_dir.name,
        event_order,
        context_ratio,
        gpickle_path,
        feature_header=generator.header(),
    )
    metadata_path = generator.write_feature_provenance(output_txt, provenance)

    print(f"raw simplices: {len(records)}")
    print(f"context_ratio: {context_ratio}")
    print(f"historical simplices: {len(context_records)}")
    print(f"closure simplices: {closure_count}")
    print(f"maximal simplices: {len(maximal_simplices)}")
    print(f"edge index edges: {len(edge_to_max_ids)}")
    print(f"gpickle triplets: {len(triplets)}")
    print(f"matched triplets: {matched}")
    print(f"missing triplets: {missing}")
    print(f"wrote: {output_txt}")
    print(f"wrote provenance: {metadata_path}")
    return str(output_txt)

def ensure_edge_maxsimplex_feature_files(args):
    fusion = getattr(args, 'edge_maxsimplex_fusion', 'none')
    prior_kind = str(getattr(args, 'temporal_prior_kind', 'recent_activity'))
    needs_features = fusion != 'none' or prior_kind in ('maxsimplex_rank', 'activity_maxsimplex')
    if not needs_features:
        return

    train_requested = getattr(args, 'train_edge_maxsimplex_txt', None)
    test_requested = getattr(args, 'test_edge_maxsimplex_txt', None)
    if not train_requested:
        args.train_edge_maxsimplex_txt = default_edge_maxsimplex_cache_path(args, 'train')
    if not test_requested:
        args.test_edge_maxsimplex_txt = default_edge_maxsimplex_cache_path(args, 'test')

    train_existing = resolve_existing_edge_maxsimplex_txt(args.train_edge_maxsimplex_txt, 'train')
    test_existing = resolve_existing_edge_maxsimplex_txt(args.test_edge_maxsimplex_txt, 'test')
    if train_existing:
        args.train_edge_maxsimplex_txt = train_existing
    if test_existing:
        args.test_edge_maxsimplex_txt = test_existing
    if train_existing and test_existing:
        return

    raw_dir = infer_edge_maxsimplex_raw_dir(args)
    if raw_dir is None:
        raise FileNotFoundError(
            "edge maximal-simplex TSV files are missing and raw_dir could not be inferred. "
            "Pass --edge_maxsimplex_raw_dir or provide existing train/test TSV paths."
        )

    if not Path(args.train_edge_maxsimplex_txt).exists():
        args.train_edge_maxsimplex_txt = generate_edge_maxsimplex_txt(
            raw_dir,
            args.train,
            args.train_edge_maxsimplex_txt,
            getattr(args, 'train_edge_maxsimplex_context_ratio', 0.6),
            getattr(args, 'protocol_event_order', 'raw'),
        )
    if not Path(args.test_edge_maxsimplex_txt).exists():
        args.test_edge_maxsimplex_txt = generate_edge_maxsimplex_txt(
            raw_dir,
            args.test,
            args.test_edge_maxsimplex_txt,
            getattr(args, 'test_edge_maxsimplex_context_ratio', 0.8),
            getattr(args, 'protocol_event_order', 'raw'),
        )

def validate_edge_maxsimplex_args(args):
    fusion = getattr(args, 'edge_maxsimplex_fusion', 'none')
    prior_kind = str(getattr(args, 'temporal_prior_kind', 'recent_activity'))
    needs_features = fusion != 'none' or prior_kind in ('maxsimplex_rank', 'activity_maxsimplex')
    use_features = parse_bool(getattr(args, 'use_edge_maxsimplex_features', False))
    if not needs_features:
        if use_features:
            print("edge_maxsimplex_fusion=none: ignoring --use_edge_maxsimplex_features and TSV paths")
        return
    if fusion != 'none' and not use_features:
        raise ValueError("--edge_maxsimplex_fusion != none requires --use_edge_maxsimplex_features true")
    if not args.train_edge_maxsimplex_txt or not args.test_edge_maxsimplex_txt:
        raise ValueError("--edge_maxsimplex_fusion != none requires train/test edge maximal-simplex TSV paths")
    for path in (args.train_edge_maxsimplex_txt, args.test_edge_maxsimplex_txt):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    raw_dir = infer_edge_maxsimplex_raw_dir(args)
    if raw_dir is None:
        raise FileNotFoundError(
            "Cannot validate maximal-simplex provenance: raw_dir could not be inferred."
        )
    validate_feature_provenance(
        args.train_edge_maxsimplex_txt,
        raw_dir,
        str(args.dataset),
        str(getattr(args, 'protocol_event_order', 'raw')),
        float(getattr(args, 'train_edge_maxsimplex_context_ratio', 0.6)),
        args.train,
    )
    validate_feature_provenance(
        args.test_edge_maxsimplex_txt,
        raw_dir,
        str(args.dataset),
        str(getattr(args, 'protocol_event_order', 'raw')),
        float(getattr(args, 'test_edge_maxsimplex_context_ratio', 0.8)),
        args.test,
    )

def create_target_mask_from_nodes(target_nodes, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    for n in target_nodes:
        if 0 <= n < num_nodes:
            mask[n] = True
    return mask

def simple_random_walk(edge_index, start_nodes, walk_length):
    pass

    device = edge_index.device
    num_walks = start_nodes.size(0)
    walks = torch.full((num_walks, walk_length + 1), -1, dtype=torch.long, device=device)
    walks[:, 0] = start_nodes

    edge_index_cpu = edge_index.cpu()
    num_nodes = int(torch.max(edge_index_cpu)) + 1
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        neighbors[u].append(v)
    neighbors = [torch.tensor(nbs, dtype=torch.long) if len(nbs) > 0 else torch.tensor([], dtype=torch.long) for nbs in neighbors]

    for step in range(1, walk_length + 1):
        prev_nodes = walks[:, step - 1].tolist()
        next_nodes = []
        for node in prev_nodes:
            nbs = neighbors[node]
            if len(nbs) == 0:
                next_nodes.append(node)
            else:
                idx = random.randint(0, len(nbs) - 1)
                next_nodes.append(nbs[idx].item())
        walks[:, step] = torch.tensor(next_nodes, device=device)

    return walks

def generate_random_walks(edge_index, num_nodes, walk_length=10, walks_per_node=20, device='cpu'):
    walks = []
    for start_node in range(num_nodes):
        start_nodes = torch.full((walks_per_node,), start_node, dtype=torch.long, device=device)
        walks_batch = simple_random_walk(edge_index, start_nodes, walk_length)
        walks.append(walks_batch.cpu())
    walks = torch.cat(walks, dim=0)
    return walks

class SkipGramModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.out_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, center_nodes, context_nodes):
        center_emb = self.embeddings(center_nodes)
        context_emb = self.out_embeddings(context_nodes)
        scores = (center_emb * context_emb).sum(dim=1)
        log_probs = F.logsigmoid(scores)
        return log_probs

def prepare_training_pairs(walks, window_size=2):
    pairs = []
    for walk in walks:
        for i in range(len(walk)):
            center = walk[i].item()
            left = max(0, i - window_size)
            right = min(len(walk), i + window_size + 1)
            for j in range(left, right):
                if i != j:
                    context = walk[j].item()
                    pairs.append((center, context))
    return pairs

def train_skipgram(num_nodes, pairs, embedding_dim=64, batch_size=1024, epochs=5, device='cpu'):
    model = SkipGramModel(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(centers, contexts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for c_batch, ctx_batch in loader:
            c_batch = c_batch.to(device)
            ctx_batch = ctx_batch.to(device)

            optimizer.zero_grad()
            output = model(c_batch, ctx_batch)
            loss = -output.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * c_batch.size(0)
        print(f"SkipGram Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(pairs):.4f}")

    return model.embeddings.weight.data.cpu().numpy()

def build_neighbor_lists_from_edge_index(edge_index, num_nodes):
    edge_index_cpu = edge_index.detach().cpu()
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if 0 <= int(u) < num_nodes:
            neighbors[int(u)].append(int(v))
    for i in range(num_nodes):
        if not neighbors[i]:
            neighbors[i].append(i)
    return neighbors

def preprocess_rw_graph(edge_index, num_nodes, split_name='rw'):
    if num_nodes <= 0:
        diagnostics = {
            'num_nodes': int(num_nodes),
            'num_edges': 0,
            'avg_degree': 0.0,
            'isolated_node_count': 0,
            'isolated_node_rate': 0.0,
            'largest_connected_component_ratio': 0.0,
        }
        print(f"RW graph diagnostics [{split_name}]: {diagnostics}")
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        empty_adjacency = sparse.csr_matrix((0, 0), dtype=np.float32)
        return empty_edge_index, [], empty_adjacency, np.empty(0, dtype=np.float64), diagnostics

    edge_index_cpu = edge_index.detach().cpu().numpy()
    if edge_index_cpu.size == 0:
        src = np.empty(0, dtype=np.int64)
        dst = np.empty(0, dtype=np.int64)
    else:
        src = edge_index_cpu[0].astype(np.int64, copy=False)
        dst = edge_index_cpu[1].astype(np.int64, copy=False)
        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
        src = src[valid]
        dst = dst[valid]

    if src.size:
        undirected = np.concatenate(
            [
                np.stack([src, dst], axis=1),
                np.stack([dst, src], axis=1),
            ],
            axis=0,
        )
        undirected = np.unique(undirected, axis=0)
    else:
        undirected = np.empty((0, 2), dtype=np.int64)

    degree = np.zeros(num_nodes, dtype=np.float64)
    non_self = undirected[:, 0] != undirected[:, 1] if undirected.size else np.empty(0, dtype=bool)
    if np.any(non_self):
        degree += np.bincount(undirected[non_self, 0], minlength=num_nodes).astype(np.float64)
    isolated = np.flatnonzero(degree == 0)
    if isolated.size:
        self_loops = np.stack([isolated, isolated], axis=1).astype(np.int64, copy=False)
        undirected = np.unique(np.concatenate([undirected, self_loops], axis=0), axis=0)
        degree[isolated] = 1.0

    rows = undirected[:, 0] if undirected.size else np.empty(0, dtype=np.int64)
    cols = undirected[:, 1] if undirected.size else np.empty(0, dtype=np.int64)
    data = np.ones(rows.size, dtype=np.float32)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    if adjacency.nnz:
        adjacency.data[:] = 1.0
    n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
    largest_component = int(np.bincount(labels, minlength=max(n_components, 1)).max()) if num_nodes else 0
    diagnostics = {
        'num_nodes': int(num_nodes),
        'num_edges': int(adjacency.nnz),
        'avg_degree': float(adjacency.nnz / max(num_nodes, 1)),
        'isolated_node_count': int(isolated.size),
        'isolated_node_rate': float(isolated.size / max(num_nodes, 1)),
        'largest_connected_component_ratio': float(largest_component / max(num_nodes, 1)),
    }
    print(f"RW graph diagnostics [{split_name}]: {diagnostics}")
    processed_edge_index = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64, copy=False))
    neighbors = build_neighbor_lists_from_edge_index(processed_edge_index, num_nodes)
    return processed_edge_index, neighbors, adjacency, degree, diagnostics

def skipgram_pairs_per_walk(walk_positions, window_size):
    walk_positions = max(1, int(walk_positions))
    window_size = max(0, min(int(window_size), walk_positions - 1))
    return 2 * sum(walk_positions - distance for distance in range(1, window_size + 1))

def build_bounded_fast_pairs(neighbors, num_nodes, walk_length=10, walks_per_node=20,
                             window_size=2, max_pairs=20_000_000,
                             restart_prob=0.0, chunk_starts=2048):
    pass
    num_nodes = max(0, int(num_nodes))
    walk_length = max(0, int(walk_length))
    walks_per_node = max(0, int(walks_per_node))
    max_pairs = max(1, int(max_pairs))
    restart_prob = float(restart_prob)
    pairs_per_walk = skipgram_pairs_per_walk(walk_length + 1, window_size)
    requested_walks = num_nodes * walks_per_node
    requested_pairs = requested_walks * pairs_per_walk
    allocation_pairs = min(max_pairs, requested_pairs)
    centers = np.empty(allocation_pairs, dtype=np.int64)
    contexts = np.empty(allocation_pairs, dtype=np.int64)

    if pairs_per_walk <= 0 or requested_walks <= 0:
        return centers[:0], contexts[:0], {
            'requested_walks': int(requested_walks),
            'generated_walks': 0,
            'requested_pairs': int(requested_pairs),
            'generated_pairs': 0,
            'pair_cap': int(max_pairs),
            'capped': bool(requested_pairs > max_pairs),
        }

    max_walks = min(requested_walks, (allocation_pairs + pairs_per_walk - 1) // pairs_per_walk)
    if requested_pairs <= max_pairs:
        starts = [node for node in range(num_nodes) for _ in range(walks_per_node)]
    elif max_walks >= num_nodes:
        starts = list(range(num_nodes))
        remaining = max_walks - num_nodes
        starts.extend(random.randrange(num_nodes) for _ in range(remaining))
    else:
        starts = list(range(num_nodes))
        random.shuffle(starts)
        starts = starts[:max_walks]

    cursor = 0
    generated_walks = 0
    chunk_pair_offsets = [0]
    walk_batch_size = max(1, int(chunk_starts) * max(1, walks_per_node))
    for start in range(0, len(starts), walk_batch_size):
        starts_arr = np.asarray(starts[start:start + walk_batch_size], dtype=np.int64)
        walks = np.empty((len(starts_arr), walk_length + 1), dtype=np.int64)
        walks[:, 0] = starts_arr
        current = starts_arr.copy()
        for step in range(1, walk_length + 1):
            next_nodes = np.empty_like(current)
            for index, node in enumerate(current):
                if restart_prob > 0.0 and random.random() < restart_prob:
                    next_nodes[index] = starts_arr[index]
                else:
                    node_neighbors = neighbors[int(node)]
                    next_nodes[index] = node_neighbors[random.randint(0, len(node_neighbors) - 1)]
            walks[:, step] = next_nodes
            current = next_nodes
        generated_walks += len(walks)
        for walk in walks:
            for index, center in enumerate(walk):
                left = max(0, index - window_size)
                right = min(len(walk), index + window_size + 1)
                for context_index in range(left, right):
                    if index == context_index:
                        continue
                    if cursor >= allocation_pairs:
                        break
                    centers[cursor] = int(center)
                    contexts[cursor] = int(walk[context_index])
                    cursor += 1
                if cursor >= allocation_pairs:
                    break
            if cursor >= allocation_pairs:
                break
        if cursor >= allocation_pairs:
            chunk_pair_offsets.append(cursor)
            break
        chunk_pair_offsets.append(cursor)

    stats = {
        'requested_walks': int(requested_walks),
        'generated_walks': int(generated_walks),
        'requested_pairs': int(requested_pairs),
        'generated_pairs': int(cursor),
        'pair_cap': int(max_pairs),
        'capped': bool(requested_pairs > max_pairs),

        'chunk_pair_offsets': tuple(chunk_pair_offsets),
    }
    return centers[:cursor], contexts[:cursor], stats

def print_bounded_pair_stats(split_name, stats):
    print(
        "random-walk skip-gram "
        f"[{split_name}]: pairs={stats['generated_pairs']:,}/cap={stats['pair_cap']:,} "
        f"walks={stats['generated_walks']:,} capped={stats['capped']}"
    )

def _bounded_pair_chunks(stats, num_pairs):
    offsets = tuple((stats or {}).get('chunk_pair_offsets', (0, int(num_pairs))))
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != int(num_pairs):
        return ((0, int(num_pairs)),)
    return tuple(
        (int(start), int(end))
        for start, end in zip(offsets[:-1], offsets[1:])
        if int(end) > int(start)
    )

def train_skipgram_fast_positive(num_nodes, centers_np=None, contexts_np=None, embedding_dim=64, batch_size=1024,
                                 epochs=5, device='cpu', pair_builder=None, split_name='rw'):
    model = SkipGramModel(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_pairs = 0
        if pair_builder is not None:
            epoch_centers, epoch_contexts, epoch_stats = pair_builder()
            print_bounded_pair_stats(f"{split_name} epoch={epoch + 1}", epoch_stats)
        else:
            epoch_centers, epoch_contexts = centers_np, contexts_np
            epoch_stats = None
        if epoch_centers is None or epoch_contexts is None:
            raise ValueError("SkipGram pair corpus is missing.")
        for chunk_start, chunk_end in _bounded_pair_chunks(epoch_stats, len(epoch_centers)):
            order = np.random.permutation(chunk_end - chunk_start) + chunk_start
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                c_batch = torch.from_numpy(epoch_centers[indices]).long().to(device)
                ctx_batch = torch.from_numpy(epoch_contexts[indices]).long().to(device)
                optimizer.zero_grad()
                output = model(c_batch, ctx_batch)
                loss = -output.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * c_batch.size(0)
                total_pairs += c_batch.size(0)
        denom = max(total_pairs, 1)
        print(f"Fast positive SkipGram Epoch {epoch+1}/{epochs}, Loss: {total_loss/denom:.4f}")
        if pair_builder is not None:
            del epoch_centers, epoch_contexts
    return model.embeddings.weight.data.cpu().numpy()

def make_negative_sampling_distribution(degree, negative_power=0.75):
    weights = np.asarray(degree, dtype=np.float64)
    weights = np.power(np.maximum(weights, 0.0), float(negative_power))
    if weights.size == 0:
        return torch.empty(0, dtype=torch.float32)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full(weights.shape, 1.0 / max(weights.size, 1), dtype=np.float64)
    else:
        weights = weights / total
    return torch.from_numpy(weights.astype(np.float32, copy=False))

def train_skipgram_sgns(num_nodes, centers_np=None, contexts_np=None, degree=None, embedding_dim=64, batch_size=1024,
                        epochs=5, device='cpu', negative_samples=5, negative_power=0.75,
                        pair_builder=None, split_name='rw'):
    model = SkipGramModel(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    neg_probs = make_negative_sampling_distribution(degree, negative_power=negative_power).to(device)
    negative_samples = max(0, int(negative_samples))
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_pairs = 0
        if pair_builder is not None:
            epoch_centers, epoch_contexts, epoch_stats = pair_builder()
            print_bounded_pair_stats(f"{split_name} epoch={epoch + 1}", epoch_stats)
        else:
            epoch_centers, epoch_contexts = centers_np, contexts_np
            epoch_stats = None
        if epoch_centers is None or epoch_contexts is None:
            raise ValueError("SkipGram pair corpus is missing.")
        for chunk_start, chunk_end in _bounded_pair_chunks(epoch_stats, len(epoch_centers)):
            order = np.random.permutation(chunk_end - chunk_start) + chunk_start
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                c_batch = torch.from_numpy(epoch_centers[indices]).long().to(device)
                ctx_batch = torch.from_numpy(epoch_contexts[indices]).long().to(device)
                optimizer.zero_grad()
                center_emb = model.embeddings(c_batch)
                context_emb = model.out_embeddings(ctx_batch)
                pos_scores = (center_emb * context_emb).sum(dim=1)
                pos_loss = -F.logsigmoid(pos_scores).mean()
                if negative_samples > 0 and neg_probs.numel() > 0:
                    neg_nodes = torch.multinomial(
                        neg_probs,
                        num_samples=c_batch.size(0) * negative_samples,
                        replacement=True,
                    ).view(c_batch.size(0), negative_samples)
                    neg_emb = model.out_embeddings(neg_nodes)
                    neg_scores = torch.bmm(neg_emb.neg(), center_emb.unsqueeze(2)).squeeze(2)
                    neg_loss = -F.logsigmoid(neg_scores).sum(dim=1).mean()
                else:
                    neg_loss = torch.zeros((), dtype=pos_loss.dtype, device=device)
                loss = pos_loss + neg_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * c_batch.size(0)
                total_pairs += c_batch.size(0)
        denom = max(total_pairs, 1)
        print(f"Fast SGNS Epoch {epoch+1}/{epochs}, Loss: {total_loss/denom:.4f}")
        if pair_builder is not None:
            del epoch_centers, epoch_contexts
    return model.embeddings.weight.data.cpu().numpy()

def generate_node_embeddings_fast_positive(edge_index, num_nodes, device='cpu',
                                           walk_length=10, walks_per_node=20,
                                           window_size=2, embedding_dim=64,
                                           rw_epochs=5, restart_prob=0.0,
                                           split_name='rw', rw_max_pairs=20_000_000):
    _, neighbors, _, _, _ = preprocess_rw_graph(edge_index, num_nodes, split_name=split_name)
    def pair_builder():
        return build_bounded_fast_pairs(
            neighbors, num_nodes, walk_length=walk_length,
            walks_per_node=walks_per_node, window_size=window_size,
            max_pairs=rw_max_pairs, restart_prob=restart_prob,
        )
    return train_skipgram_fast_positive(
        num_nodes,
        embedding_dim=embedding_dim,
        epochs=rw_epochs,
        device=device,
        pair_builder=pair_builder,
        split_name=split_name,
    )

def generate_node_embeddings_fast_sgns(edge_index, num_nodes, device='cpu',
                                       walk_length=10, walks_per_node=20,
                                       window_size=2, embedding_dim=64,
                                       rw_epochs=5, restart_prob=0.0,
                                       negative_samples=5, negative_power=0.75,
                                       split_name='rw', rw_max_pairs=20_000_000):
    _, neighbors, _, degree, _ = preprocess_rw_graph(edge_index, num_nodes, split_name=split_name)
    def pair_builder():
        return build_bounded_fast_pairs(
            neighbors, num_nodes, walk_length=walk_length,
            walks_per_node=walks_per_node, window_size=window_size,
            max_pairs=rw_max_pairs, restart_prob=restart_prob,
        )
    return train_skipgram_sgns(
        num_nodes,
        degree=degree,
        embedding_dim=embedding_dim,
        epochs=rw_epochs,
        device=device,
        negative_samples=negative_samples,
        negative_power=negative_power,
        pair_builder=pair_builder,
        split_name=split_name,
    )

def sparse_ppmi(matrix):
    matrix = matrix.tocoo(copy=True)
    if matrix.nnz == 0:
        return matrix.tocsr()
    matrix.data = np.asarray(matrix.data, dtype=np.float64)
    total = float(matrix.data.sum())
    if not np.isfinite(total) or total <= 0.0:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    row_sum = np.asarray(matrix.tocsr().sum(axis=1)).reshape(-1)
    col_sum = np.asarray(matrix.tocsc().sum(axis=0)).reshape(-1)
    denom = row_sum[matrix.row] * col_sum[matrix.col]
    valid = denom > 0
    values = np.zeros_like(matrix.data, dtype=np.float64)
    values[valid] = np.log((matrix.data[valid] * total) / denom[valid])
    keep = values > 0
    if not np.any(keep):
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    return sparse.csr_matrix(
        (values[keep].astype(np.float32, copy=False), (matrix.row[keep], matrix.col[keep])),
        shape=matrix.shape,
        dtype=np.float32,
    )

def generate_node_embeddings_ppmi_svd(edge_index, num_nodes, embedding_dim=64,
                                      context_hops=5, split_name='rw'):
    _, _, adjacency, _, _ = preprocess_rw_graph(edge_index, num_nodes, split_name=split_name)
    if num_nodes <= 0:
        return np.zeros((0, embedding_dim), dtype=np.float32)
    if num_nodes == 1 or adjacency.nnz == 0:
        return np.zeros((num_nodes, embedding_dim), dtype=np.float32)
    row_sum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_degree = np.divide(1.0, row_sum, out=np.zeros_like(row_sum, dtype=np.float64), where=row_sum > 0)
    transition = sparse.diags(inv_degree.astype(np.float32, copy=False)).dot(adjacency).tocsr()
    context_hops = max(1, int(context_hops))
    power = transition.copy()
    cooc = power.copy()
    for _ in range(2, context_hops + 1):
        power = power.dot(transition).tocsr()
        cooc = cooc + power
    ppmi = sparse_ppmi(cooc)
    if ppmi.nnz == 0:
        embedding = np.zeros((num_nodes, min(embedding_dim, max(1, num_nodes - 1))), dtype=np.float32)
    else:
        n_components = min(int(embedding_dim), max(1, num_nodes - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        embedding = svd.fit_transform(ppmi).astype(np.float32, copy=False)
    embedding = StandardScaler().fit_transform(embedding).astype(np.float32, copy=False)
    embedding = normalize(embedding, norm='l2', axis=1).astype(np.float32, copy=False)
    if embedding.shape[1] < embedding_dim:
        pad = np.zeros((num_nodes, embedding_dim - embedding.shape[1]), dtype=np.float32)
        embedding = np.concatenate([embedding, pad], axis=1)
    return embedding[:, :embedding_dim].astype(np.float32, copy=False)

def generate_rw_embedding_by_impl(edge_index, num_nodes, device, rw_impl,
                                  walk_length, walks_per_node, window_size,
                                  embedding_dim, rw_epochs, negative_samples=5,
                                  negative_power=0.75, restart_prob=0.0,
                                  context_hops=5, split_name='rw',
                                  rw_max_pairs=20_000_000):
    if rw_impl == 'original':
        processed_edge_index, _, _, _, _ = preprocess_rw_graph(edge_index, num_nodes, split_name=split_name)
        walks = generate_random_walks(processed_edge_index.to(device), num_nodes, walk_length=walk_length,
                                      walks_per_node=walks_per_node, device=device)
        pairs = prepare_training_pairs(walks, window_size=window_size)
        return train_skipgram(num_nodes, pairs, embedding_dim=embedding_dim, epochs=rw_epochs, device=device)
    if rw_impl == 'fast_positive':
        return generate_node_embeddings_fast_positive(
            edge_index,
            num_nodes,
            device=device,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            window_size=window_size,
            embedding_dim=embedding_dim,
            rw_epochs=rw_epochs,
            restart_prob=restart_prob,
            split_name=split_name,
            rw_max_pairs=rw_max_pairs,
        )
    if rw_impl == 'fast_sgns':
        return generate_node_embeddings_fast_sgns(
            edge_index,
            num_nodes,
            device=device,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            window_size=window_size,
            embedding_dim=embedding_dim,
            rw_epochs=rw_epochs,
            restart_prob=restart_prob,
            negative_samples=negative_samples,
            negative_power=negative_power,
            split_name=split_name,
            rw_max_pairs=rw_max_pairs,
        )
    if rw_impl == 'ppmi_svd':
        return generate_node_embeddings_ppmi_svd(
            edge_index,
            num_nodes,
            embedding_dim=embedding_dim,
            context_hops=context_hops,
            split_name=split_name
        )
    raise ValueError(f"Unknown rw_impl: {rw_impl}")

def normalize_rw_branch(train_emb, test_emb, l2norm=True):
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_emb)
    test_z = scaler.transform(test_emb)

    if l2norm:
        train_norm = np.linalg.norm(train_z, axis=1, keepdims=True)
        test_norm = np.linalg.norm(test_z, axis=1, keepdims=True)
        train_z = train_z / np.maximum(train_norm, 1e-12)
        test_z = test_z / np.maximum(test_norm, 1e-12)

    return train_z.astype(np.float32, copy=False), test_z.astype(np.float32, copy=False)

def print_rw_diagnostics(args, train_rw, test_rw, extra=None):
    print("RW diagnostics:")
    print(f"  rw_impl={args.rw_impl}")
    print(f"  train_rw_shape={tuple(train_rw.shape)}")
    print(f"  test_rw_shape={tuple(test_rw.shape)}")
    print(f"  train_rw_mean={float(np.mean(train_rw)):.6f}")
    print(f"  train_rw_std={float(np.std(train_rw)):.6f}")
    print(f"  test_rw_mean={float(np.mean(test_rw)):.6f}")
    print(f"  test_rw_std={float(np.std(test_rw)):.6f}")
    print(f"  train_nan_count={int(np.isnan(train_rw).sum())}")
    print(f"  test_nan_count={int(np.isnan(test_rw).sum())}")
    if extra:
        for key, value in extra.items():
            print(f"  {key}={value}")

def generate_hybrid_fast_ppmi_embeddings(args, edge_index_train, tr_n, edge_index_test, te_n, rw_seed, device):
    stoch_dim = int(args.rw_hybrid_stoch_dim)
    ppmi_dim = int(args.rw_hybrid_ppmi_dim)
    total_dim = int(args.rw_embedding_dim)
    if stoch_dim + ppmi_dim != total_dim:
        raise ValueError(
            "rw_hybrid_stoch_dim + rw_hybrid_ppmi_dim must equal rw_embedding_dim "
            f"({stoch_dim} + {ppmi_dim} != {total_dim})"
        )

    set_global_seed(int(rw_seed))
    print(f"Generating train stochastic RW branch: rw_seed={rw_seed}, impl=fast_positive, dim={stoch_dim}")
    train_stoch = generate_rw_embedding_by_impl(
        edge_index_train, tr_n, device, 'fast_positive',
        args.rw_walk_length, args.rw_walks_per_node, args.rw_window_size,
        stoch_dim, args.rw_epochs,
        negative_samples=args.rw_negative_samples,
        negative_power=args.rw_negative_power,
        restart_prob=args.rw_restart_prob,
        context_hops=args.rw_context_hops,
        split_name='train_hybrid_stoch',
        rw_max_pairs=args.rw_max_pairs,
    )
    print(f"Generating test stochastic RW branch: rw_seed={rw_seed}, impl=fast_positive, dim={stoch_dim}")
    test_stoch = generate_rw_embedding_by_impl(
        edge_index_test, te_n, device, 'fast_positive',
        args.rw_walk_length, args.rw_walks_per_node, args.rw_window_size,
        stoch_dim, args.rw_epochs,
        negative_samples=args.rw_negative_samples,
        negative_power=args.rw_negative_power,
        restart_prob=args.rw_restart_prob,
        context_hops=args.rw_context_hops,
        split_name='test_hybrid_stoch',
        rw_max_pairs=args.rw_max_pairs,
    )

    print(f"Generating train deterministic RW branch: impl=ppmi_svd, dim={ppmi_dim}, hops={args.rw_hybrid_ppmi_hops}")
    train_ppmi = generate_rw_embedding_by_impl(
        edge_index_train, tr_n, device, 'ppmi_svd',
        args.rw_walk_length, args.rw_walks_per_node, args.rw_window_size,
        ppmi_dim, args.rw_epochs,
        negative_samples=args.rw_negative_samples,
        negative_power=args.rw_negative_power,
        restart_prob=args.rw_restart_prob,
        context_hops=args.rw_hybrid_ppmi_hops,
        split_name='train_hybrid_ppmi'
    )
    print(f"Generating test deterministic RW branch: impl=ppmi_svd, dim={ppmi_dim}, hops={args.rw_hybrid_ppmi_hops}")
    test_ppmi = generate_rw_embedding_by_impl(
        edge_index_test, te_n, device, 'ppmi_svd',
        args.rw_walk_length, args.rw_walks_per_node, args.rw_window_size,
        ppmi_dim, args.rw_epochs,
        negative_samples=args.rw_negative_samples,
        negative_power=args.rw_negative_power,
        restart_prob=args.rw_restart_prob,
        context_hops=args.rw_hybrid_ppmi_hops,
        split_name='test_hybrid_ppmi'
    )

    l2norm = parse_bool(args.rw_hybrid_l2norm)
    train_stoch, test_stoch = normalize_rw_branch(train_stoch, test_stoch, l2norm=l2norm)
    train_ppmi, test_ppmi = normalize_rw_branch(train_ppmi, test_ppmi, l2norm=l2norm)
    train_rw = np.concatenate([train_stoch, train_ppmi], axis=1).astype(np.float32, copy=False)
    test_rw = np.concatenate([test_stoch, test_ppmi], axis=1).astype(np.float32, copy=False)
    extra = {
        'stoch_dim': stoch_dim,
        'ppmi_dim': ppmi_dim,
        'stoch_norm_mean': f"{float(np.mean(np.linalg.norm(train_stoch, axis=1))):.6f}",
        'ppmi_norm_mean': f"{float(np.mean(np.linalg.norm(train_ppmi, axis=1))):.6f}",
    }
    print_rw_diagnostics(args, train_rw, test_rw, extra=extra)
    return train_rw, test_rw

def prune_edge_index_fanout(edge_index, num_nodes, fanout=100, seed=0, split_name='rw'):
    fanout = int(fanout)
    if fanout <= 0:
        return edge_index
    edge_cpu = edge_index.detach().cpu().numpy()
    if edge_cpu.size == 0:
        return edge_index
    src = edge_cpu[0].astype(np.int64, copy=False)
    dst = edge_cpu[1].astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed))
    order = np.argsort(src, kind='mergesort')
    src_sorted = src[order]
    dst_sorted = dst[order]
    unique_src, starts = np.unique(src_sorted, return_index=True)
    ends = np.r_[starts[1:], src_sorted.size]
    keep_src = []
    keep_dst = []
    total_kept = 0
    for node, start, end in zip(unique_src, starts, ends):
        if node < 0 or node >= int(num_nodes):
            continue
        degree = int(end - start)
        if degree > fanout:
            local_idx = rng.choice(degree, size=fanout, replace=False)
            chosen_src = src_sorted[start:end][local_idx]
            chosen_dst = dst_sorted[start:end][local_idx]
        else:
            chosen_src = src_sorted[start:end]
            chosen_dst = dst_sorted[start:end]
        keep_src.append(chosen_src)
        keep_dst.append(chosen_dst)
        total_kept += int(chosen_src.size)
    if keep_src:
        pruned = np.vstack([np.concatenate(keep_src), np.concatenate(keep_dst)]).astype(np.int64, copy=False)
    else:
        pruned = np.empty((2, 0), dtype=np.int64)
    print(
        f"RW graph pruning [{split_name}]: fanout={fanout}, "
        f"edges_before={edge_index.size(1)}, edges_after={total_kept}"
    )
    return torch.from_numpy(pruned).long().to(edge_index.device)

def generate_or_load_rw_embeddings(args, edge_index_train, tr_n, edge_index_test, te_n, rw_seed, device):
    os.makedirs(args.rw_cache_dir, exist_ok=True)
    reference_cache = getattr(args, 'rw_reference_cache', None)
    if reference_cache:
        reference_cache = str(reference_cache).format(rw_seed=int(rw_seed))
        if not os.path.isfile(reference_cache):
            raise FileNotFoundError(f"Reference random-walk cache not found: {reference_cache}")
        with np.load(reference_cache, allow_pickle=False) as z:
            if 'train_rw' not in z or 'test_rw' not in z:
                raise ValueError(
                    f"Reference random-walk cache must contain train_rw and test_rw: {reference_cache}"
                )
            train_rw = np.asarray(z['train_rw'], dtype=np.float32).copy()
            test_rw = np.asarray(z['test_rw'], dtype=np.float32).copy()
            reference_payload = None
            if 'payload' in z:
                raw_payload = z['payload']
                if np.asarray(raw_payload).size == 1:
                    try:
                        reference_payload = json.loads(str(np.asarray(raw_payload).item()))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        reference_payload = None

        expected_train_shape = (int(tr_n), int(args.rw_embedding_dim))
        expected_test_shape = (int(te_n), int(args.rw_embedding_dim))
        if train_rw.shape != expected_train_shape or test_rw.shape != expected_test_shape:
            raise ValueError(
                "Reference random-walk cache shape mismatch: "
                f"train={train_rw.shape}, expected_train={expected_train_shape}, "
                f"test={test_rw.shape}, expected_test={expected_test_shape}"
            )
        if not np.isfinite(train_rw).all() or not np.isfinite(test_rw).all():
            raise ValueError(f"Reference random-walk cache contains NaN or Inf: {reference_cache}")
        if isinstance(reference_payload, dict):
            expected_payload = {
                'rw_seed': int(rw_seed),
                'walk_length': int(args.rw_walk_length),
                'walks_per_node': int(args.rw_walks_per_node),
                'window_size': int(args.rw_window_size),
                'embedding_dim': int(args.rw_embedding_dim),
                'rw_epochs': int(args.rw_epochs),
                'rw_impl': str(args.rw_impl),
                'rw_max_pairs': int(args.rw_max_pairs),
            }
            mismatches = {
                key: (reference_payload.get(key), expected)
                for key, expected in expected_payload.items()
                if key in reference_payload and reference_payload.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Reference random-walk cache configuration mismatch: {mismatches}"
                )
        print(f"Loading explicit reference random-walk cache for rw_seed={rw_seed}: {reference_cache}")
        print_rw_diagnostics(args, train_rw, test_rw)
        return train_rw, test_rw

    payload = {
        'train_hash': _file_hash(args.train),
        'test_hash': _file_hash(args.test),
        'rw_seed': int(rw_seed),
        'walk_length': int(args.rw_walk_length),
        'walks_per_node': int(args.rw_walks_per_node),
        'window_size': int(args.rw_window_size),
        'embedding_dim': int(args.rw_embedding_dim),
        'rw_epochs': int(args.rw_epochs),
        'rw_impl': args.rw_impl,
        'rw_max_pairs': int(args.rw_max_pairs),
        'rw_negative_samples': int(args.rw_negative_samples),
        'rw_negative_power': float(args.rw_negative_power),
        'rw_restart_prob': float(args.rw_restart_prob),
        'rw_context_hops': int(args.rw_context_hops),
        'rw_hybrid_stoch_dim': int(args.rw_hybrid_stoch_dim),
        'rw_hybrid_ppmi_dim': int(args.rw_hybrid_ppmi_dim),
        'rw_hybrid_ppmi_hops': int(args.rw_hybrid_ppmi_hops),
        'rw_hybrid_l2norm': parse_bool(args.rw_hybrid_l2norm),
        'rw_graph_scope': getattr(args, 'rw_graph_scope', 'full'),
        'rw_prune_fanout': int(getattr(args, 'rw_prune_fanout', 50)),
    }
    cache_path = os.path.join(args.rw_cache_dir, f"rw_embedding_{_stable_hash_payload(payload)}.npz")
    legacy_payload = dict(payload)
    legacy_payload.pop('rw_graph_scope', None)
    legacy_payload.pop('rw_prune_fanout', None)
    legacy_cache_path = os.path.join(args.rw_cache_dir, f"rw_embedding_{_stable_hash_payload(legacy_payload)}.npz")
    original_positive_payload = {
        'train_hash': payload['train_hash'],
        'test_hash': payload['test_hash'],
        'rw_seed': payload['rw_seed'],
        'walk_length': payload['walk_length'],
        'walks_per_node': payload['walks_per_node'],
        'window_size': payload['window_size'],
        'embedding_dim': payload['embedding_dim'],
        'rw_epochs': payload['rw_epochs'],
        'rw_impl': payload['rw_impl'],
        'rw_max_pairs': payload['rw_max_pairs'],
    }
    original_positive_cache_path = os.path.join(
        args.rw_cache_dir,
        f"original_positive_rw_{_stable_hash_payload(original_positive_payload)}.npz"
    )
    if args.rw_cache_mode == 'per_rw_seed' and os.path.exists(cache_path):
        print(f"Loading random-walk cache for rw_seed={rw_seed}: {cache_path}")
        z = np.load(cache_path)
        train_rw, test_rw = z['train_rw'], z['test_rw']
        print_rw_diagnostics(args, train_rw, test_rw)
        return train_rw, test_rw
    allow_legacy_rw_cache = str(getattr(args, 'rw_graph_scope', 'full')).strip().lower() == 'full'
    if args.rw_cache_mode == 'per_rw_seed' and allow_legacy_rw_cache and os.path.exists(legacy_cache_path):
        print(f"Loading legacy random-walk cache for rw_seed={rw_seed}: {legacy_cache_path}")
        z = np.load(legacy_cache_path)
        train_rw, test_rw = z['train_rw'], z['test_rw']
        print_rw_diagnostics(args, train_rw, test_rw)
        return train_rw, test_rw
    if (args.rw_cache_mode == 'per_rw_seed'
            and args.rw_impl == 'fast_positive'
            and allow_legacy_rw_cache
            and os.path.exists(original_positive_cache_path)):
        print(f"Loading original positive random-walk cache for rw_seed={rw_seed}: {original_positive_cache_path}")
        z = np.load(original_positive_cache_path)
        train_rw, test_rw = z['train_rw'], z['test_rw']
        print_rw_diagnostics(args, train_rw, test_rw)
        return train_rw, test_rw

    set_global_seed(int(rw_seed))
    if getattr(args, 'rw_graph_scope', 'full') == 'pruned':
        edge_index_train = prune_edge_index_fanout(
            edge_index_train,
            tr_n,
            fanout=int(getattr(args, 'rw_prune_fanout', 50)),
            seed=int(rw_seed),
            split_name='train'
        )
        edge_index_test = prune_edge_index_fanout(
            edge_index_test,
            te_n,
            fanout=int(getattr(args, 'rw_prune_fanout', 50)),
            seed=int(rw_seed) + 1000003,
            split_name='test'
        )
    if args.rw_impl == 'hybrid_fast_ppmi':
        tr_node_emb, te_node_emb = generate_hybrid_fast_ppmi_embeddings(
            args, edge_index_train, tr_n, edge_index_test, te_n, rw_seed, device
        )
        if args.rw_cache_mode == 'per_rw_seed':
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, train_rw=tr_node_emb, test_rw=te_node_emb, payload=json.dumps(payload))
            print(f"Saved random-walk cache: {cache_path}")
        return tr_node_emb, te_node_emb

    print(f"Generating train random-walk embedding: rw_seed={rw_seed}, impl={args.rw_impl}")
    tr_node_emb = generate_rw_embedding_by_impl(
        edge_index_train, tr_n, device, args.rw_impl,
        args.rw_walk_length, args.rw_walks_per_node, args.rw_window_size,
        args.rw_embedding_dim, args.rw_epochs,
        negative_samples=args.rw_negative_samples,
        negative_power=args.rw_negative_power,
        restart_prob=args.rw_restart_prob,
        context_hops=args.rw_context_hops,
        split_name='train',
        rw_max_pairs=args.rw_max_pairs,
    )
    print(f"Generating test random-walk embedding: rw_seed={rw_seed}, impl={args.rw_impl}")
    te_node_emb = generate_rw_embedding_by_impl(
        edge_index_test, te_n, device, args.rw_impl,
        args.rw_walk_length, args.rw_walks_per_node, args.rw_window_size,
        args.rw_embedding_dim, args.rw_epochs,
        negative_samples=args.rw_negative_samples,
        negative_power=args.rw_negative_power,
        restart_prob=args.rw_restart_prob,
        context_hops=args.rw_context_hops,
        split_name='test',
        rw_max_pairs=args.rw_max_pairs,
    )
    if args.rw_cache_mode == 'per_rw_seed':
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, train_rw=tr_node_emb, test_rw=te_node_emb, payload=json.dumps(payload))
        print(f"Saved random-walk cache: {cache_path}")
    print_rw_diagnostics(args, tr_node_emb, te_node_emb)
    return tr_node_emb, te_node_emb

class MultiLayerGAT(nn.Module):
    pass

    def __init__(self,
                 in_channels,
                 hidden_channels=128,
                 out_channels=2,
                 num_layers=3,
                 heads=4,
                 dropout=0.3,
                 use_batchnorm=True,
                 use_layernorm=False,
                 residual=True,
                 use_mlp_head=True,
                 mlp_layers=2,
                 activation='leaky_relu',
                 weight_init='xavier',
                 use_strength_edge_weight=False):
        super(MultiLayerGAT, self).__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        self.residual = residual
        self.activation = activation
        self.use_mlp_head = use_mlp_head
        self.use_strength_edge_weight = bool(use_strength_edge_weight)

        self.gradient_checkpointing = os.environ.get('K2_GRADIENT_CHECKPOINT', '0') == '1'

        if isinstance(heads, int):
            heads = [heads] * num_layers
        else:
            if len(heads) != num_layers:
                raise ValueError("若提供 heads 列表，其长度必须等于 num_layers")

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev_channels = in_channels
        for i in range(num_layers):
            h = heads[i]
            out_h = hidden_channels
            concat = True if (i < num_layers - 1) else False

            layer = GATConv(prev_channels, out_h, heads=h, concat=True, dropout=dropout)
            self.gat_layers.append(layer)

            if use_batchnorm:
                self.norms.append(nn.BatchNorm1d(out_h * h))
            elif use_layernorm:
                self.norms.append(nn.LayerNorm(out_h * h))
            else:
                self.norms.append(nn.Identity())
            prev_channels = out_h * h

        self.res_mappings = nn.ModuleList()
        prev = in_channels
        for i in range(num_layers):
            curr = hidden_channels * heads[i]
            if self.residual:
                if prev != curr:
                    self.res_mappings.append(nn.Linear(prev, curr))
                else:
                    self.res_mappings.append(nn.Identity())
            else:
                self.res_mappings.append(nn.Identity())
            prev = curr

        self.weighted_neighbor_mappings = nn.ModuleList()
        if self.use_strength_edge_weight:
            prev = in_channels
            for i in range(num_layers):
                curr = hidden_channels * heads[i]
                self.weighted_neighbor_mappings.append(nn.Linear(prev, curr, bias=False))
                prev = curr

        head_in = hidden_channels * heads[-1]
        if use_mlp_head:
            mlp = []
            hidden = max(64, head_in // 2)
            for i in range(max(1, mlp_layers - 1)):
                mlp.append(nn.Linear(head_in if i == 0 else hidden, hidden))
                if use_batchnorm:
                    mlp.append(nn.BatchNorm1d(hidden))
                elif use_layernorm:
                    mlp.append(nn.LayerNorm(hidden))
                mlp.append(nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU())
                mlp.append(nn.Dropout(dropout))
            mlp.append(nn.Linear(hidden if mlp_layers > 1 else head_in, out_channels))
            self.classifier = nn.Sequential(*mlp)
        else:
            self.classifier = nn.Linear(head_in, out_channels)

        self.global_pool = global_mean_pool

        self._init_weights(method=weight_init)

    def _init_weights(self, method='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                else:
                    try:
                        nn.init.xavier_uniform_(m.weight)
                    except:
                        pass
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _weighted_neighbor_mean(x, edge_index, edge_weight):
        if edge_weight is None:
            raise ValueError('Strength-edge model requires aligned edge_weight.')
        edge_weight = edge_weight.to(device=x.device, dtype=x.dtype).view(-1)
        if edge_weight.numel() != edge_index.size(1):
            raise ValueError(
                f'edge_weight alignment mismatch: {edge_weight.numel()} != {edge_index.size(1)}'
            )
        src, dst = edge_index[0], edge_index[1]
        aggregated = torch.zeros_like(x)
        denominator = torch.zeros((x.size(0),), dtype=x.dtype, device=x.device)
        aggregated.index_add_(0, dst, x[src] * edge_weight.view(-1, 1))
        denominator.index_add_(0, dst, edge_weight)
        return aggregated / denominator.clamp_min(1e-12).view(-1, 1)

    def encode(self, x, edge_index, batch=None, edge_weight=None):
        h = x
        for i, gat in enumerate(self.gat_layers):
            def encode_one_layer(layer_input, layer_index=i, layer_module=gat):
                weighted_message = None
                if self.use_strength_edge_weight:
                    weighted_message = self.weighted_neighbor_mappings[layer_index](
                        self._weighted_neighbor_mean(layer_input, edge_index, edge_weight)
                    )
                out = layer_module(layer_input, edge_index)
                if weighted_message is not None:
                    out = out + weighted_message
                norm = self.norms[layer_index]
                if not isinstance(norm, nn.Identity):
                    try:
                        out = norm(out)
                    except Exception:

                        out = out

                if self.activation == 'leaky_relu':
                    out = F.leaky_relu(out, negative_slope=0.2)
                else:
                    out = F.relu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)

                if self.residual:
                    res_map = self.res_mappings[layer_index]
                    res = res_map(layer_input)
                    out = out + res
                return out

            if self.training and self.gradient_checkpointing:
                h = torch_checkpoint(
                    encode_one_layer, h, use_reentrant=False
                )
            else:
                h = encode_one_layer(h)

        if batch is not None:
            h = self.global_pool(h, batch)

        return h

    def forward(self, x, edge_index, batch=None, max_x=None, edge_weight=None):
        h = self.encode(x, edge_index, batch=batch, edge_weight=edge_weight)
        logits = self.classifier(h)
        return logits

class PriorLogitGAT(nn.Module):
    def __init__(self,
                 in_channels,
                 max_feat_dim,
                 hidden_channels=128,
                 out_channels=2,
                 num_layers=3,
                 heads=4,
                 dropout=0.3,
                 use_batchnorm=True,
                 use_layernorm=False,
                 residual=True,
                 use_mlp_head=True,
                 mlp_layers=2,
                 activation='leaky_relu',
                 weight_init='xavier',
                 prior_hidden_dim=16,
                 prior_dropout=0.1,
                 prior_alpha=0.1,
                 prior_target_class=1,
                 use_strength_edge_weight=False):
        super().__init__()
        self.gnn = MultiLayerGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_layernorm=use_layernorm,
            residual=residual,
            use_mlp_head=use_mlp_head,
            mlp_layers=mlp_layers,
            activation=activation,
            weight_init=weight_init,
            use_strength_edge_weight=use_strength_edge_weight
        )
        self.prior_alpha = float(prior_alpha)
        self.prior_target_class = int(prior_target_class)
        self.prior_mlp = nn.Sequential(
            nn.Linear(int(max_feat_dim), int(prior_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(prior_dropout)),
            nn.Linear(int(prior_hidden_dim), 1)
        )

    def forward(self, x, edge_index, batch=None, max_x=None, edge_weight=None):
        logits = self.gnn(
            x, edge_index, batch=batch, edge_weight=edge_weight
        )
        if max_x is None:
            return logits
        prior = self.prior_mlp(max_x).view(-1)
        logits = logits.clone()
        logits[:, self.prior_target_class] = (
            logits[:, self.prior_target_class] + self.prior_alpha * prior
        )
        return logits

class DualAdjGATMinimal(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=128,
                 out_channels=2,
                 num_layers=3,
                 heads=4,
                 dropout=0.3,
                 use_batchnorm=True,
                 use_layernorm=False,
                 residual=True,
                 use_mlp_head=True,
                 mlp_layers=2,
                 activation='leaky_relu',
                 weight_init='xavier',
                 dual_self_hidden_dim=128,
                 dual_gate_hidden_dim=64,
                 dual_dropout=0.3,
                 use_strength_edge_weight=False,
                 history_feature_dim=0,
                 history_feature_mode='none',
                 history_residual_scale=1.0,
                 history_pretrained_checkpoint=''):
        super().__init__()

        self.history_feature_dim = int(history_feature_dim)
        self.history_feature_mode = str(history_feature_mode)
        self.history_residual_scale = float(history_residual_scale)
        self.history_pretrained = False
        if self.history_feature_mode not in {
            'none', 'concat', 'residual', 'gated_residual',
            'history_base_structural_residual',
        }:
            raise ValueError(f'Unsupported history feature mode: {self.history_feature_mode}')
        residual_modes = {
            'residual', 'gated_residual', 'history_base_structural_residual'
        }
        if self.history_feature_mode in residual_modes and self.history_feature_dim <= 0:
            raise ValueError('history_feature_dim must be positive in residual mode')
        encoder_in_channels = (
            int(in_channels) - self.history_feature_dim
            if self.history_feature_mode in residual_modes
            else int(in_channels)
        )

        self.gat_encoder = MultiLayerGAT(
            in_channels=encoder_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_layernorm=use_layernorm,
            residual=residual,
            use_mlp_head=use_mlp_head,
            mlp_layers=mlp_layers,
            activation=activation,
            weight_init=weight_init,
            use_strength_edge_weight=use_strength_edge_weight
        )

        gat_heads = heads[-1] if isinstance(heads, (list, tuple)) else heads
        gat_out_dim = int(hidden_channels) * int(gat_heads)
        self.self_mlp = nn.Sequential(
            nn.Linear(encoder_in_channels, int(dual_self_hidden_dim)),
            nn.BatchNorm1d(int(dual_self_hidden_dim)) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU(),
            nn.Dropout(float(dual_dropout)),
            nn.Linear(int(dual_self_hidden_dim), gat_out_dim),
            nn.BatchNorm1d(gat_out_dim) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(gat_out_dim * 2, int(dual_gate_hidden_dim)),
            nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU(),
            nn.Dropout(float(dual_dropout)),
            nn.Linear(int(dual_gate_hidden_dim), gat_out_dim),
            nn.Sigmoid()
        )

        classifier_hidden = max(64, gat_out_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(gat_out_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(classifier_hidden, out_channels)
        )
        if self.history_feature_mode in {'residual', 'gated_residual'}:
            history_hidden = max(16, min(64, self.history_feature_dim * 2))
            self.history_residual_head = nn.Sequential(
                nn.LayerNorm(self.history_feature_dim),
                nn.Linear(self.history_feature_dim, history_hidden),
                nn.ReLU(),
                nn.Dropout(float(dual_dropout)),
                nn.Linear(history_hidden, out_channels),
            )
            if self.history_feature_mode == 'gated_residual':
                self.history_gate_head = nn.Sequential(
                    nn.LayerNorm(gat_out_dim + self.history_feature_dim),
                    nn.Linear(gat_out_dim + self.history_feature_dim, max(16, int(dual_gate_hidden_dim))),
                    nn.ReLU(),
                    nn.Linear(max(16, int(dual_gate_hidden_dim)), 1),
                    nn.Sigmoid(),
                )
        elif self.history_feature_mode == 'history_base_structural_residual':

            if history_pretrained_checkpoint:
                self.history_pretrained = True
                history_hidden = 64
                self.history_pretrained_net = nn.Sequential(
                    nn.Linear(self.history_feature_dim, 64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
                )
                payload = torch.load(history_pretrained_checkpoint, map_location='cpu', weights_only=False)
                state = payload.get('model_state_dict', payload) if isinstance(payload, dict) else payload
                if isinstance(state, dict) and state and all(str(k).startswith('net.') for k in state):
                    state = {str(k)[4:]: v for k, v in state.items()}
                self.history_pretrained_net.load_state_dict(state)
            else:
                history_hidden = max(32, min(128, self.history_feature_dim * 2))
                self.history_base_encoder = nn.Sequential(
                    nn.LayerNorm(self.history_feature_dim),
                    nn.Linear(self.history_feature_dim, history_hidden),
                    nn.ReLU(), nn.Dropout(float(dual_dropout)),
                )
                self.history_base_head = nn.Linear(history_hidden, out_channels)
            residual_hidden = max(32, min(128, int(dual_gate_hidden_dim)))
            self.structural_residual_head = nn.Sequential(
                nn.LayerNorm(gat_out_dim + history_hidden),
                nn.Linear(gat_out_dim + history_hidden, residual_hidden),
                nn.ReLU(),
                nn.Dropout(float(dual_dropout)),
                nn.Linear(residual_hidden, out_channels),
            )
            self.structural_residual_gate = nn.Sequential(
                nn.LayerNorm(gat_out_dim + history_hidden),
                nn.Linear(gat_out_dim + history_hidden, max(16, int(dual_gate_hidden_dim))),
                nn.ReLU(),
                nn.Linear(max(16, int(dual_gate_hidden_dim)), 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.structural_residual_head[-1].weight)
            nn.init.zeros_(self.structural_residual_head[-1].bias)
            gate_linear = self.structural_residual_gate[-2]
            nn.init.constant_(gate_linear.bias, -2.0)

    def forward(self, x, edge_index, batch=None, max_x=None, edge_weight=None):
        history_x = None
        encoder_x = x
        if self.history_feature_mode in {
            'residual', 'gated_residual', 'history_base_structural_residual'
        }:
            history_x = x[:, -self.history_feature_dim:]
            encoder_x = x[:, :-self.history_feature_dim]
        h_gat = self.gat_encoder.encode(
            encoder_x, edge_index, batch=batch, edge_weight=edge_weight
        )
        h_self = self.self_mlp(encoder_x)
        if batch is not None:
            h_self = self.gat_encoder.global_pool(h_self, batch)

        gate = self.gate_mlp(torch.cat([h_gat, h_self], dim=-1))
        h = gate * h_gat + (1.0 - gate) * h_self
        logits = self.classifier(h)
        if history_x is not None:
            history_context = history_x
            if batch is not None:
                history_context = self.gat_encoder.global_pool(history_context, batch)
            if self.history_feature_mode == 'history_base_structural_residual':
                if self.history_pretrained:
                    positive_logit = self.history_pretrained_net(history_context).squeeze(-1)
                    history_hidden = self.history_pretrained_net[:6](history_context)
                    history_logits = torch.stack([torch.zeros_like(positive_logit), positive_logit], dim=1)
                else:
                    history_hidden = self.history_base_encoder(history_context)
                    history_logits = self.history_base_head(history_hidden)
                joint = torch.cat([h, history_hidden], dim=-1)
                structural_gate = self.structural_residual_gate(joint)
                structural_residual = self.structural_residual_head(joint)
                logits = history_logits + structural_gate * structural_residual
            else:
                history_logits = self.history_residual_head(history_context)
                if self.history_feature_mode == 'gated_residual':
                    history_gate = self.history_gate_head(torch.cat([h, history_context], dim=-1))
                    history_logits = history_gate * history_logits
                logits = logits + self.history_residual_scale * history_logits
        return logits

def sanitize_score_array(scores, context="score"):
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    bad_mask = ~np.isfinite(arr)
    if not np.any(bad_mask):
        return arr
    raise FloatingPointError(
        f"Non-finite {context}: "
        f"nan={int(np.isnan(arr).sum())}, "
        f"+inf={int(np.isposinf(arr).sum())}, "
        f"-inf={int(np.isneginf(arr).sum())}; refusing to evaluate"
    )

def compute_auc_pr(y_true, y_score_pos):
    pass

    if y_true is None or len(y_true) == 0:
        return np.nan, np.nan, np.nan

    y_score_pos = sanitize_score_array(y_score_pos, context="AUC-PR scores")
    baseline = float(np.mean(y_true))
    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count == 0:
        return np.nan, np.nan, baseline
    if neg_count == 0:

        return 1.0, (1.0 / baseline) if baseline > 0 else np.nan, baseline

    try:

        auc_pr = float(average_precision_score(y_true, y_score_pos))
    except Exception:
        auc_pr = np.nan

    improvement = auc_pr / baseline if baseline > 0 and not np.isnan(auc_pr) else np.nan
    return auc_pr, improvement, baseline

def model_forward(model, data, device):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    max_x = getattr(data, 'max_x', None)
    if max_x is not None:
        max_x = max_x.to(device)
    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    output = model(x, edge_index, max_x=max_x, edge_weight=edge_weight)
    require_finite_tensor(output, "model logits")
    return output

def require_finite_tensor(value, context):
    if not torch.isfinite(value).all().item():
        finite = value.detach()[torch.isfinite(value.detach())]
        raise FloatingPointError(
            f"Non-finite {context}: shape={tuple(value.shape)} "
            f"finite={int(finite.numel())}/{value.numel()}"
        )

def compute_pairwise_ranking_aux_loss(out_batch, y_batch, margin=0.0, max_pairs=4096):
    pos_mask = y_batch == 1
    neg_mask = y_batch == 0
    if int(pos_mask.sum().item()) == 0 or int(neg_mask.sum().item()) == 0:
        return torch.zeros((), dtype=out_batch.dtype, device=out_batch.device)
    scores = out_batch[:, 1] - out_batch[:, 0]
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    max_pairs = max(1, int(max_pairs))
    num_pairs = min(max_pairs, int(pos_scores.numel()) * int(neg_scores.numel()))
    pos_idx = torch.randint(pos_scores.numel(), (num_pairs,), device=out_batch.device)
    neg_idx = torch.randint(neg_scores.numel(), (num_pairs,), device=out_batch.device)
    return F.relu(float(margin) - pos_scores[pos_idx] + neg_scores[neg_idx]).mean()

def train(model, data, optimizer, criterion, train_mask, scaler=None, batch_size=2048,
          priority_mask=None, priority_frac=0.5, priority_mode='hard',
          pairwise_aux_loss=False, pairwise_aux_weight=0.0,
          pairwise_aux_margin=0.0, pairwise_aux_max_pairs=4096):
    model.train()
    device = next(model.parameters()).device

    if scaler is None:
        scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_idx = torch.where(train_mask)[0].to(device)
    n = train_idx.size(0)
    if n == 0:
        return 0.0, 0.0, scaler

    total_loss_accum = 0.0
    sup_loss_accum = 0.0
    cnt = 0

    y_dev = data.y.to(device)

    optimizer.zero_grad()

    for idx_cpu in make_seed_batches(
        train_idx,
        train_mask,
        priority_mask=priority_mask,
        batch_size=batch_size,
        priority_frac=priority_frac,
        priority_mode=priority_mode,
        device=device,
    ):
        idx = idx_cpu.to(device)
        if idx.numel() == 0:
            continue

        with autocast(enabled=bool(scaler.is_enabled())):
            out_batch = model_forward(model, data, device)[idx]
            sup_loss = criterion(out_batch, y_dev[idx])
            require_finite_tensor(sup_loss, "training loss")
            total_loss = sup_loss
            if pairwise_aux_loss and float(pairwise_aux_weight) > 0.0:
                pairwise_loss = compute_pairwise_ranking_aux_loss(
                    out_batch,
                    y_dev[idx],
                    margin=pairwise_aux_margin,
                    max_pairs=pairwise_aux_max_pairs
                )
                total_loss = sup_loss + float(pairwise_aux_weight) * pairwise_loss
            require_finite_tensor(total_loss, "total training loss")

        scaler.scale(total_loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        if not torch.isfinite(grad_norm).all().item():
            raise FloatingPointError("Non-finite training gradient norm")
        for parameter in model.parameters():
            if parameter.grad is not None:
                require_finite_tensor(parameter.grad, "training gradient")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss_accum += float(total_loss.detach().cpu().item())
        sup_loss_accum += float(sup_loss.detach().cpu().item())
        cnt += 1

    return total_loss_accum / cnt if cnt else 0.0, sup_loss_accum / max(cnt, 1), scaler

def local_num_neighbors_schedule(local_num_neighbors=50, local_num_hops=1):
    hops = max(1, int(local_num_hops))
    fanout = max(1, int(local_num_neighbors))
    if hops == 1:
        return [fanout]
    if hops == 2 and fanout >= 2:
        return [max(1, fanout // 2), max(1, min(20, fanout))]
    return [fanout] * hops

def require_neighbor_loader():
    return

def has_neighbor_sampling_backend():
    return (
        NeighborLoader is not None
        and (
            importlib.util.find_spec('pyg_lib') is not None
            or importlib.util.find_spec('torch_sparse') is not None
        )
    )

def local_sampler_backend(local_sampler='auto', local_fallback_sample=True):
    sampler = str(local_sampler or 'auto').strip().lower()
    if sampler not in ('auto', 'neighborloader', 'fallback'):
        raise ValueError(f"Unsupported local_sampler={local_sampler}")
    if sampler == 'neighborloader':
        if not has_neighbor_sampling_backend():
            if local_fallback_sample:
                return 'fallback'
            raise ImportError("local_sampler=neighborloader requires pyg-lib or torch-sparse.")
        return 'neighborloader'
    if sampler == 'fallback':
        return 'fallback'
    return 'neighborloader' if has_neighbor_sampling_backend() else 'fallback'

def local_coverage_stats(sampled_nodes, sampled_edges, num_batches, full_num_nodes, full_num_edges):
    avg_nodes = float(sampled_nodes / max(int(num_batches), 1))
    avg_edges = float(sampled_edges / max(int(num_batches), 1))
    return {
        'avg_sampled_nodes_per_batch': avg_nodes,
        'avg_sampled_edges_per_batch': avg_edges,
        'local_coverage_ratio_nodes': float(avg_nodes / max(int(full_num_nodes), 1)),
        'local_coverage_ratio_edges': float(avg_edges / max(int(full_num_edges), 1)),
    }

def warn_if_near_full_local_sampling(stats, context='local sampling'):
    node_ratio = float(stats.get('local_coverage_ratio_nodes', 0.0) or 0.0)
    edge_ratio = float(stats.get('local_coverage_ratio_edges', 0.0) or 0.0)
    if node_ratio > 0.8 or edge_ratio > 0.8:
        print("WARNING: local sampling is near-full graph; reduce batch size or fanout.")
        print(
            f"WARNING: {context}: avg batch coverage nodes={node_ratio:.1%}, "
            f"edges={edge_ratio:.1%}."
        )

def get_fallback_adj_cache(data):
    cache = getattr(data, '_fallback_adj_cache', None)
    if cache is not None:
        return cache
    edge_index = data.edge_index.detach().cpu().long()
    num_nodes = int(data.num_nodes)
    sample_adj_sets = [set() for _ in range(num_nodes)]
    out_adj_lists = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for edge_id, (u, v) in enumerate(zip(src, dst)):
        u = int(u)
        v = int(v)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            sample_adj_sets[u].add(v)
            sample_adj_sets[v].add(u)
            out_adj_lists[u].append((v, edge_id))
    sample_adj = [torch.as_tensor(sorted(nei), dtype=torch.long) for nei in sample_adj_sets]
    out_adj = [tuple(nei) for nei in out_adj_lists]
    cache = {
        'sample_adj': sample_adj,
        'out_adj': out_adj,
        'num_nodes': num_nodes,
    }
    data._fallback_adj_cache = cache
    return cache

def make_fanout_local_batch(data, seed_nodes, local_num_neighbors=50, local_num_hops=1):
    if not isinstance(seed_nodes, torch.Tensor):
        seed_nodes = torch.as_tensor(seed_nodes, dtype=torch.long)
    seed_nodes = seed_nodes.detach().cpu().long()
    cache = get_fallback_adj_cache(data)
    sample_adj = cache['sample_adj']
    out_adj = cache['out_adj']
    num_nodes = cache['num_nodes']
    fanout_schedule = local_num_neighbors_schedule(local_num_neighbors, local_num_hops)
    sampled = set(int(v) for v in seed_nodes.tolist() if 0 <= int(v) < num_nodes)
    frontier = set(sampled)
    for fanout in fanout_schedule:
        next_frontier = set()
        fanout = max(1, int(fanout))
        for node in frontier:
            neighbors = sample_adj[node]
            if neighbors.numel() == 0:
                continue
            if neighbors.numel() > fanout:
                pick = neighbors[torch.randperm(neighbors.numel())[:fanout]]
            else:
                pick = neighbors
            for n in pick.tolist():
                n = int(n)
                if n not in sampled:
                    next_frontier.add(n)
                sampled.add(n)
        frontier = next_frontier
        if not frontier:
            break
    subset = torch.as_tensor(sorted(sampled), dtype=torch.long)
    local_index = {int(node): i for i, node in enumerate(subset.tolist())}
    mapping = torch.as_tensor([local_index.get(int(node), -1) for node in seed_nodes.tolist()], dtype=torch.long)
    if bool((mapping < 0).any()):
        raise RuntimeError("fanout local fallback failed to include all seed nodes")
    local_src = []
    local_dst = []
    local_edge_ids = []
    sampled_lookup = set(local_index)
    for src_node in subset.tolist():
        src_node = int(src_node)
        src_local = local_index[src_node]
        for dst_node, edge_id in out_adj[src_node]:
            if dst_node in sampled_lookup:
                local_src.append(src_local)
                local_dst.append(local_index[int(dst_node)])
                local_edge_ids.append(int(edge_id))
    if local_src:
        sub_edge_index = torch.as_tensor([local_src, local_dst], dtype=torch.long)
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = Data(
        x=data.x[subset],
        edge_index=sub_edge_index,
        y=data.y[subset],
    )
    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is not None:
        edge_ids = torch.as_tensor(local_edge_ids, dtype=torch.long)
        batch.edge_weight = edge_weight.detach().cpu()[edge_ids]
    max_x = getattr(data, 'max_x', None)
    batch.max_x = max_x[subset] if max_x is not None else None
    return batch, mapping.detach().long(), subset

def make_seed_batches(train_idx, train_mask, priority_mask=None, batch_size=2048,
                      priority_frac=0.5, priority_mode='hard', device=None):
    if device is None:
        device = train_idx.device
    batch_size = max(1, int(batch_size))
    n = train_idx.size(0)
    if n == 0:
        return []
    if priority_mask is not None:
        priority_idx_all = torch.where(priority_mask & train_mask)[0].to(device)
    else:
        priority_idx_all = torch.tensor([], dtype=torch.long, device=device)
    empty_priority = torch.zeros_like(train_mask) if priority_mask is None else priority_mask
    nonpriority_idx_all = torch.where(train_mask & (~empty_priority))[0].to(device)

    if priority_mode == 'hard' and priority_idx_all.numel() > 0:
        priority_idx_all = priority_idx_all[torch.randperm(priority_idx_all.numel(), device=device)]
        nonpriority_idx_all = nonpriority_idx_all[torch.randperm(nonpriority_idx_all.numel(), device=device)]
        desired_priority = int(math.ceil(batch_size * float(priority_frac)))
        desired_priority = min(max(desired_priority, 0), batch_size)
        desired_nonpriority = batch_size - desired_priority
        batches = []
        pr_pos = 0
        non_pos = 0
        while pr_pos < priority_idx_all.numel() or non_pos < nonpriority_idx_all.numel():
            pr_take = min(desired_priority, int(priority_idx_all.numel() - pr_pos))
            non_take = min(desired_nonpriority, int(nonpriority_idx_all.numel() - non_pos))
            spare = batch_size - pr_take - non_take
            if spare > 0 and pr_pos + pr_take < priority_idx_all.numel():
                extra = min(spare, int(priority_idx_all.numel() - pr_pos - pr_take))
                pr_take += extra
                spare -= extra
            if spare > 0 and non_pos + non_take < nonpriority_idx_all.numel():
                extra = min(spare, int(nonpriority_idx_all.numel() - non_pos - non_take))
                non_take += extra
            parts = []
            if pr_take > 0:
                parts.append(priority_idx_all[pr_pos:pr_pos + pr_take])
                pr_pos += pr_take
            if non_take > 0:
                parts.append(nonpriority_idx_all[non_pos:non_pos + non_take])
                non_pos += non_take
            if parts:
                idx = torch.cat(parts, dim=0)
                idx = idx[torch.randperm(idx.numel(), device=device)]
                batches.append(idx.detach().cpu())
        return batches

    if priority_mode == 'soft' and priority_idx_all.numel() > 0:
        priority_weight = 3.0
        sample_weights = torch.ones(train_idx.size(0), device=device)
        priority_set = set(int(x.item()) for x in priority_idx_all)
        for i, idx in enumerate(train_idx):
            if int(idx.item()) in priority_set:
                sample_weights[i] = priority_weight
        sample_weights_for_draw = sample_weights
    else:
        sample_weights_for_draw = None

    perm = torch.randperm(train_idx.size(0), device=device)
    train_idx = train_idx[perm]
    batches = []
    for start in range(0, n, batch_size):
        if priority_mode == 'soft' and sample_weights_for_draw is not None:
            k = min(batch_size, train_idx.size(0))
            select_pos = torch.multinomial(sample_weights_for_draw, k, replacement=True)
            idx = train_idx[select_pos].to(device)
        else:
            idx = train_idx[start:start + batch_size]
        if idx.numel() > 0:
            batches.append(idx.detach().cpu())
    return batches

def train_local(model, data, optimizer, criterion, train_mask, scaler=None,
                local_train_batch_size=64, local_num_neighbors=50, local_num_hops=1,
                local_sampler='auto', local_fallback_sample=True,
                effective_batch_size=256,
                priority_mask=None, priority_frac=0.5, priority_mode='hard',
                pairwise_aux_loss=False, pairwise_aux_weight=0.0,
                pairwise_aux_margin=0.0, pairwise_aux_max_pairs=4096):
    require_neighbor_loader()
    model.train()
    device = next(model.parameters()).device
    if scaler is None:
        scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_idx = torch.where(train_mask)[0].to(device)
    if train_idx.numel() == 0:
        return 0.0, 0.0, scaler, {'num_batches': 0, 'sampled_nodes': 0, 'sampled_edges': 0}

    seed_batches = make_seed_batches(
        train_idx,
        train_mask,
        priority_mask=priority_mask,
        batch_size=int(local_train_batch_size),
        priority_frac=priority_frac,
        priority_mode=priority_mode,
        device=device,
    )
    num_neighbors = local_num_neighbors_schedule(local_num_neighbors, local_num_hops)
    sampler_backend = local_sampler_backend(local_sampler, local_fallback_sample)
    use_neighbor_loader = sampler_backend == 'neighborloader'
    total_loss_accum = 0.0
    sup_loss_accum = 0.0
    cnt = 0
    seed_nodes_seen = 0
    sampled_nodes = 0
    sampled_edges = 0
    max_sampled_nodes = 0
    max_sampled_edges = 0
    optimizer.zero_grad()
    accum_samples = 0

    for seeds_cpu in seed_batches:
        if use_neighbor_loader:
            local_batches = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                input_nodes=seeds_cpu,
                batch_size=int(seeds_cpu.numel()),
                shuffle=False,
            )
        else:
            batch_cpu, seed_positions, _ = make_fanout_local_batch(
                data,
                seeds_cpu,
                local_num_neighbors=local_num_neighbors,
                local_num_hops=local_num_hops,
            )
            local_batches = [(batch_cpu, seed_positions)]
        for local_batch in local_batches:
            if use_neighbor_loader:
                batch = local_batch
                seed_positions = None
            else:
                batch, seed_positions = local_batch
            batch = batch.to(device)
            if seed_positions is not None:
                seed_positions = seed_positions.to(device)
                seed_count = int(seed_positions.numel())
            else:
                seed_count = int(getattr(batch, 'batch_size', seeds_cpu.numel()))
            seed_nodes_seen += int(seed_count)
            sampled_nodes += int(batch.num_nodes)
            sampled_edges += int(batch.edge_index.size(1))
            max_sampled_nodes = max(max_sampled_nodes, int(batch.num_nodes))
            max_sampled_edges = max(max_sampled_edges, int(batch.edge_index.size(1)))
            with autocast(enabled=bool(scaler.is_enabled())):
                out_all = model_forward(model, batch, device)
                if seed_positions is not None:
                    out_seed = out_all[seed_positions]
                    y_seed = batch.y[seed_positions]
                else:
                    out_seed = out_all[:seed_count]
                    y_seed = batch.y[:seed_count]
                sup_loss = criterion(out_seed, y_seed)
                require_finite_tensor(sup_loss, "local training loss")
                total_loss = sup_loss
                if pairwise_aux_loss and float(pairwise_aux_weight) > 0.0:
                    pairwise_loss = compute_pairwise_ranking_aux_loss(
                        out_seed,
                        y_seed,
                        margin=pairwise_aux_margin,
                        max_pairs=pairwise_aux_max_pairs
                    )
                    total_loss = sup_loss + float(pairwise_aux_weight) * pairwise_loss
                require_finite_tensor(total_loss, "local total training loss")

            scaler.scale(total_loss * (float(seed_count) / float(max(effective_batch_size, 1)))).backward()
            accum_samples += int(seed_count)
            if accum_samples < int(effective_batch_size):
                total_loss_accum += float(total_loss.detach().cpu().item())
                sup_loss_accum += float(sup_loss.detach().cpu().item())
                cnt += 1
                continue
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if not torch.isfinite(grad_norm).all().item():
                raise FloatingPointError("Non-finite local training gradient norm")
            for parameter in model.parameters():
                if parameter.grad is not None:
                    require_finite_tensor(parameter.grad, "local training gradient")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accum_samples = 0
            total_loss_accum += float(total_loss.detach().cpu().item())
            sup_loss_accum += float(sup_loss.detach().cpu().item())
            cnt += 1

    if accum_samples:
        scale = float(effective_batch_size) / float(accum_samples)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        if not torch.isfinite(grad_norm).all().item():
            raise FloatingPointError("Non-finite local training gradient norm")
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); accum_samples = 0

    coverage = local_coverage_stats(
        sampled_nodes,
        sampled_edges,
        cnt,
        int(data.num_nodes),
        int(data.edge_index.size(1)),
    )
    stats = {
        'num_batches': int(cnt),
        'sampler_backend': sampler_backend,
        'local_sampler_used': sampler_backend,
        'seed_nodes_seen': int(seed_nodes_seen),
        'sampled_nodes': int(sampled_nodes),
        'sampled_edges': int(sampled_edges),
        'avg_seed_nodes_per_batch': float(seed_nodes_seen / max(cnt, 1)),
        'max_sampled_nodes_per_batch': int(max_sampled_nodes),
        'max_sampled_edges_per_batch': int(max_sampled_edges),
        **coverage,
    }
    warn_if_near_full_local_sampling(stats, context='local training')
    return total_loss_accum / cnt if cnt else 0.0, sup_loss_accum / max(cnt, 1), scaler, stats

def predict_seed_logits_local(model, data, input_nodes, local_eval_batch_size=64,
                              local_num_neighbors=50, local_num_hops=1,
                              local_sampler='auto', local_fallback_sample=True,
                              device=None, context='local prediction'):
    require_neighbor_loader()
    if device is None:
        device = next(model.parameters()).device
    if not isinstance(input_nodes, torch.Tensor):
        input_nodes = torch.as_tensor(input_nodes, dtype=torch.long)
    input_nodes = input_nodes.detach().cpu().long()
    if input_nodes.numel() == 0:
        return np.empty((0, 2), dtype=np.float32), {
            'target_nodes_covered': 0,
            'sampler_backend': local_sampler_backend(local_sampler, local_fallback_sample),
            'local_sampler_used': local_sampler_backend(local_sampler, local_fallback_sample),
            'avg_seed_nodes_per_batch': 0.0,
            'avg_sampled_nodes_per_batch': 0.0,
            'avg_sampled_edges_per_batch': 0.0,
            'max_sampled_nodes_per_batch': 0,
            'max_sampled_edges_per_batch': 0,
            'local_coverage_ratio_nodes': 0.0,
            'local_coverage_ratio_edges': 0.0,
        }

    num_neighbors = local_num_neighbors_schedule(local_num_neighbors, local_num_hops)
    sampler_backend = local_sampler_backend(local_sampler, local_fallback_sample)
    use_neighbor_loader = sampler_backend == 'neighborloader'
    if use_neighbor_loader:
        local_batches = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=int(local_eval_batch_size),
            shuffle=False,
        )
    else:
        local_batches = []
        batch_size = max(1, int(local_eval_batch_size))
        for start in range(0, int(input_nodes.numel()), batch_size):
            seed_nodes = input_nodes[start:start + batch_size]
            batch_cpu, seed_positions, _ = make_fanout_local_batch(
                data,
                seed_nodes,
                local_num_neighbors=local_num_neighbors,
                local_num_hops=local_num_hops,
            )
            local_batches.append((batch_cpu, seed_positions, seed_nodes))
    model.eval()
    logits_parts = []
    covered_nodes = []
    sampled_nodes = 0
    sampled_edges = 0
    seed_nodes_seen = 0
    max_sampled_nodes = 0
    max_sampled_edges = 0
    num_batches = 0
    with torch.no_grad():
        for local_batch in local_batches:
            if use_neighbor_loader:
                batch = local_batch
                seed_positions = None
                seed_nodes = None
            else:
                batch, seed_positions, seed_nodes = local_batch
            batch = batch.to(device)
            if seed_positions is not None:
                seed_positions = seed_positions.to(device)
                seed_count = int(seed_positions.numel())
                out_seed = model_forward(model, batch, device)[seed_positions]
            else:
                seed_count = int(getattr(batch, 'batch_size', 0))
                out_seed = model_forward(model, batch, device)[:seed_count]
            seed_nodes_seen += int(seed_count)
            logits_parts.append(out_seed.detach().cpu())
            if seed_nodes is not None:
                covered_nodes.extend(seed_nodes.detach().cpu().tolist())
            elif hasattr(batch, 'n_id'):
                covered_nodes.extend(batch.n_id[:seed_count].detach().cpu().tolist())
            else:
                covered_nodes.extend(input_nodes[len(covered_nodes):len(covered_nodes) + seed_count].tolist())
            sampled_nodes += int(batch.num_nodes)
            sampled_edges += int(batch.edge_index.size(1))
            max_sampled_nodes = max(max_sampled_nodes, int(batch.num_nodes))
            max_sampled_edges = max(max_sampled_edges, int(batch.edge_index.size(1)))
            num_batches += 1

    logits = torch.cat(logits_parts, dim=0).numpy() if logits_parts else np.empty((0, 2), dtype=np.float32)
    expected = [int(v) for v in input_nodes.tolist()]
    if len(covered_nodes) != len(expected) or set(covered_nodes) != set(expected):
        missing = sorted(set(expected) - set(covered_nodes))
        raise RuntimeError(
            f"{context}: local sampler did not cover all seed nodes "
            f"(covered={len(set(covered_nodes))}, expected={len(set(expected))}, missing_first={missing[:10]})"
        )
    if covered_nodes != expected:
        order = {node: i for i, node in enumerate(covered_nodes)}
        try:
            logits = logits[[order[node] for node in expected]]
        except KeyError as exc:
            raise RuntimeError(f"{context}: local sampler output cannot be aligned to target seed order") from exc
    coverage = local_coverage_stats(
        sampled_nodes,
        sampled_edges,
        num_batches,
        int(data.num_nodes),
        int(data.edge_index.size(1)),
    )
    stats = {
        'target_nodes_covered': int(len(expected)),
        'sampler_backend': sampler_backend,
        'local_sampler_used': sampler_backend,
        'avg_seed_nodes_per_batch': float(seed_nodes_seen / max(num_batches, 1)),
        'max_sampled_nodes_per_batch': int(max_sampled_nodes),
        'max_sampled_edges_per_batch': int(max_sampled_edges),
        **coverage,
    }
    warn_if_near_full_local_sampling(stats, context=context)
    return logits, stats

def metrics_from_binary_scores(y_true, y_pos_probs, threshold=0.5, pos_ratio=None):
    y_true = np.asarray(y_true)
    y_pos_probs = sanitize_score_array(y_pos_probs, context="local positive probabilities")
    if len(y_true) == 0:
        return {
            "class0_accuracy": 0,
            "class1_accuracy": 0,
            "f1_score": 0,
            "f2_score": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "precision": [], "recall": [], "auc_pr": 0,
            "average_precision": 0,
            "mcc": np.nan,
            "y_true": [], "y_probs": [], "evaluation_gnn": 0,
            "threshold": threshold,
            "baseline": 0.0,
            "improvement": np.nan,
        }
    if pos_ratio is not None and pos_ratio > 0:
        n_targets = len(y_pos_probs)
        k = int(np.ceil(n_targets * float(pos_ratio)))
        k = min(max(k, 0), n_targets)
        if k == 0:
            y_pred_labels = np.zeros_like(y_true, dtype=int)
            actual_threshold = 1.0
        else:
            order = np.argsort(-y_pos_probs)
            topk_idx = order[:k]
            y_pred_labels = np.zeros_like(y_true, dtype=int)
            y_pred_labels[topk_idx] = 1
            actual_threshold = float(np.sort(y_pos_probs)[-k]) if k > 0 else 1.0
    else:
        y_pred_labels = (y_pos_probs > threshold).astype(int)
        actual_threshold = threshold

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels, labels=[0, 1]).ravel()
    class0_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    class1_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    f2 = fbeta_score(y_true, y_pred_labels, beta=2, zero_division=0)
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred_labels))
    except Exception:
        mcc = np.nan

    auc_pr, improvement, baseline = compute_auc_pr(y_true, y_pos_probs)
    if len(np.unique(y_true)) < 2:
        if len(y_true) > 0 and y_true[0] == 0:
            precision_arr = np.array([1.0])
            recall_arr = np.array([1.0])
        else:
            precision_arr = np.array([0.0])
            recall_arr = np.array([1.0])
        average_precision = 1.0 if (len(y_true) > 0 and y_true[0] == 0) else 0.0
    else:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pos_probs)
        try:
            average_precision = average_precision_score(y_true, y_pos_probs)
        except Exception:
            average_precision = np.nan
    evaluation_gnn = improvement if not np.isnan(improvement) else 0.0
    return {
        "class0_accuracy": class0_acc,
        "class1_accuracy": class1_acc,
        "f1_score": f1,
        "f2_score": f2,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "precision": precision_arr,
        "recall": recall_arr,
        "auc_pr": auc_pr,
        "average_precision": average_precision,
        "mcc": mcc,
        "y_true": y_true,
        "y_probs": y_pos_probs,
        "evaluation_gnn": evaluation_gnn,
        "threshold": actual_threshold,
        "baseline": baseline,
        "improvement": improvement,
    }

def build_temporal_activity_aligned_scores(dataset, split, triplets, indices, args):
    alpha = float(getattr(args, 'temporal_activity_alpha', 0.0))
    if alpha <= 0:
        return None
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--temporal_activity_alpha must be in [0, 1].")

    from pipelines.native import mlshol as mlshol_protocol
    mlshol_protocol.DATA_ROOT = Path(
        getattr(args, 'protocol_data_root', None) or Path(__file__).resolve().parent
    ).resolve()
    mlshol_protocol.EVENT_ORDER = str(getattr(args, 'protocol_event_order', 'raw'))

    prior_args = argparse.Namespace(
        temporal_activity_recent_fraction=float(args.temporal_activity_recent_fraction),
        temporal_activity_aggregation=str(args.temporal_activity_aggregation),
        temporal_activity_entity=str(args.temporal_activity_entity),
        maxsimplex_prior_weight=0.0,
    )
    if split == 'early':
        context_fraction = 0.4
    elif split == 'train':
        context_fraction = 0.6
    else:
        context_fraction = float(getattr(args, 'temporal_activity_context_ratio', 0.8))
        if not 0.0 < context_fraction <= 1.0:
            raise ValueError('--temporal_activity_context_ratio must be in (0, 1].')
    _, data_simplices = mlshol_protocol.get_timestamped_simplices(
        str(mlshol_protocol.raw_paths(str(dataset))[0]),
        str(mlshol_protocol.raw_paths(str(dataset))[1]),
        str(mlshol_protocol.raw_paths(str(dataset))[2]),
    )
    context_end = int(context_fraction * len(data_simplices))
    recent_count = max(
        1,
        int(round(context_end * float(args.temporal_activity_recent_fraction))),
    )
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    node_last = {}
    edge_last = {}
    if str(getattr(args, 'temporal_activity_measure', 'count')) == 'last_event':
        for event_index, simplex in enumerate(data_simplices[:context_end]):
            vertices = sorted(set(map(int, simplex)))
            for node in vertices:
                node_last[node] = event_index
            for edge in combinations(vertices, 2):
                edge_last[tuple(sorted(edge))] = event_index
    for simplex in data_simplices[max(0, context_end - recent_count):context_end]:
        vertices = sorted(set(map(int, simplex)))
        for node in vertices:
            node_counts[node] += 1
        for edge in combinations(vertices, 2):
            edge_counts[tuple(sorted(edge))] += 1

    aligned_indices = np.asarray(indices, dtype=np.int64)
    aligned_triangles = [
        tuple(sorted(map(int, triplets[int(index)])))
        for index in aligned_indices
    ]
    protocol_triangles = []
    try:
        protocol_paths = mlshol_protocol.split_paths(
            str(dataset), str(split), prior_args
        )
        protocol_triangles = [
            tuple(sorted(map(int, triangle)))
            for triangle in mlshol_protocol.read_triangles(protocol_paths['open'])
        ]
    except (FileNotFoundError, ValueError):

        protocol_triangles = []
    rank_domain = list(dict.fromkeys(protocol_triangles + aligned_triangles))
    raw_scores = []
    for triangle in rank_domain:
        if str(args.temporal_activity_entity) == 'node':
            counts = np.asarray(
                [
                    (node_last.get(int(node), -1) if str(getattr(args, 'temporal_activity_measure', 'count')) == 'last_event'
                     else node_counts[int(node)])
                    for node in triangle
                ], dtype=np.float64
            )
        elif str(args.temporal_activity_entity) == 'edge':
            counts = np.asarray(
                [
                    (edge_last.get(tuple(sorted(edge)), -1) if str(getattr(args, 'temporal_activity_measure', 'count')) == 'last_event'
                     else edge_counts[tuple(sorted(edge))])
                    for edge in combinations(triangle, 2)
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Unsupported temporal activity entity: {args.temporal_activity_entity}"
            )
        aggregation = str(args.temporal_activity_aggregation)
        if aggregation == 'min':
            raw_score = float(counts.min())
        elif aggregation == 'product':
            raw_score = float(np.log1p(counts).sum())
        elif aggregation == 'mean':
            raw_score = float(counts.mean())
        else:
            raise ValueError(
                f"Unsupported temporal activity aggregation: {aggregation}"
            )
        raw_scores.append(raw_score)
    rank_values = rankdata(np.asarray(raw_scores, dtype=np.float64), method='average')
    rank_values = rank_values / float(max(len(rank_values), 1))
    activity_map = {
        triangle: float(rank_values[index])
        for index, triangle in enumerate(rank_domain)
    }
    scores = np.asarray([activity_map[triangle] for triangle in aligned_triangles], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Temporal activity prior contains NaN or Inf.")
    print(
        "Temporal activity prior: "
        f"alpha={alpha:.3f}, measure={getattr(args, 'temporal_activity_measure', 'count')}, recent_fraction={args.temporal_activity_recent_fraction}, "
        f"aggregation={args.temporal_activity_aggregation}, "
        f"entity={args.temporal_activity_entity}, split={split}, targets={scores.size}"
    )
    return scores

def build_maxsimplex_rank_aligned_scores(triplets, indices, feature_path, column='tri_all_edges_shared_count'):
    pass

    feature_dict, feature_dim = load_edge_maxsimplex_features(feature_path)

    with open(feature_path, 'r', encoding='utf-8', newline='') as handle:
        header = handle.readline().rstrip('\n\r').split('\t')
    if column not in header:
        raise ValueError(
            f"Maximal-simplex prior column {column!r} not found in {feature_path}"
        )
    column_index = header.index(column) - 1
    if column_index < 0 or column_index >= feature_dim:
        raise ValueError(f"Invalid maximal-simplex prior column index for {column!r}")
    aligned = np.asarray(indices, dtype=np.int64)
    raw = np.asarray([
        float(feature_dict.get(tuple(sorted(map(int, triplets[int(index)]))),
                               np.zeros(feature_dim, dtype=np.float32))[column_index])
        for index in aligned
    ], dtype=np.float64)
    if raw.size == 0:
        return raw
    scores = rankdata(raw, method='average') / float(raw.size)
    if not np.isfinite(scores).all():
        raise ValueError('Maximal-simplex rank prior contains NaN/Inf.')
    print(
        'Maximal-simplex rank prior: '
        f"column={column}, targets={scores.size}, "
        f"raw_min={raw.min():.6g}, raw_max={raw.max():.6g}"
    )
    return scores.astype(np.float64, copy=False)

def test_local(model, data, target_mask, threshold=0.5, pos_ratio=None, pos_label=0,
               device=None, local_eval_batch_size=64,
               local_num_neighbors=50, local_num_hops=1,
               local_sampler='auto', local_fallback_sample=True):
    target_indices = torch.where(target_mask)[0]
    logits, stats = predict_seed_logits_local(
        model,
        data,
        target_indices,
        local_eval_batch_size=local_eval_batch_size,
        local_num_neighbors=local_num_neighbors,
        local_num_hops=local_num_hops,
        local_sampler=local_sampler,
        local_fallback_sample=local_fallback_sample,
        device=device,
        context='target local evaluation',
    )
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_true = data.y[target_mask].cpu().numpy()
    metrics = metrics_from_binary_scores(y_true, probs[:, 1] if probs.size else np.asarray([]), threshold=threshold, pos_ratio=pos_ratio)
    metrics.update(stats)
    expected_targets = int(target_indices.numel())
    if int(metrics.get('target_nodes_covered', 0)) != expected_targets:
        raise RuntimeError(
            "target local evaluation coverage mismatch: "
            f"target_nodes_covered={metrics.get('target_nodes_covered', 0)}, expected={expected_targets}"
        )
    return metrics

def test(model, data, target_mask, threshold=0.5, pos_ratio=None, pos_label=0, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        logits = model_forward(model, data, device)
        probs = F.softmax(logits, dim=1)

    y_true = data.y[target_mask].cpu().numpy()
    y_pos_probs = sanitize_score_array(
        probs[target_mask][:, 1].cpu().numpy(),
        context="test positive probabilities"
    )

    if len(y_true) == 0:
        return {
            "class0_accuracy": 0,
            "class1_accuracy": 0,
            "f1_score": 0,
            "f2_score": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "precision": [], "recall": [], "auc_pr": 0,
            "average_precision": 0,
            "mcc": np.nan,
            "y_true": [], "y_probs": [], "evaluation_gnn": 0,
            "threshold": threshold
        }

    if pos_ratio is not None and pos_ratio > 0:
        n_targets = len(y_pos_probs)
        k = int(np.ceil(n_targets * float(pos_ratio)))
        k = min(max(k, 0), n_targets)
        if k == 0:
            y_pred_labels = np.zeros_like(y_true, dtype=int)
            actual_threshold = 1.0
        else:
            order = np.argsort(-y_pos_probs)
            topk_idx = order[:k]
            y_pred_labels = np.zeros_like(y_true, dtype=int)
            y_pred_labels[topk_idx] = 1
            actual_threshold = float(np.sort(y_pos_probs)[-k]) if k > 0 else 1.0
    else:
        y_pred_labels = (y_pos_probs > threshold).astype(int)
        actual_threshold = threshold

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels, labels=[0, 1]).ravel()
    class0_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    class1_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    f2 = fbeta_score(y_true, y_pred_labels, beta=2, zero_division=0)
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred_labels))
    except Exception:
        mcc = np.nan

    auc_pr, improvement, baseline = compute_auc_pr(y_true, y_pos_probs)

    if len(np.unique(y_true)) < 2:

        if len(y_true) > 0 and y_true[0] == 0:
            precision_arr = np.array([1.0])
            recall_arr = np.array([1.0])
        else:
            precision_arr = np.array([0.0])
            recall_arr = np.array([1.0])
        average_precision = 1.0 if (len(y_true) > 0 and y_true[0] == 0) else 0.0
    else:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pos_probs)
        try:
            average_precision = average_precision_score(y_true, y_pos_probs)
        except Exception:
            average_precision = np.nan

    evaluation_gnn = improvement if not np.isnan(improvement) else 0.0

    return {
        "class0_accuracy": class0_acc,
        "class1_accuracy": class1_acc,
        "f1_score": f1,
        "f2_score": f2,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "precision": precision_arr,
        "recall": recall_arr,
        "auc_pr": auc_pr,
        "average_precision": average_precision,
        "mcc": mcc,
        "y_true": y_true,
        "y_probs": y_pos_probs,
        "evaluation_gnn": evaluation_gnn,
        "threshold": actual_threshold,
        "baseline": baseline,
        "improvement": improvement
    }

def save_predictions(preds_logits, targets, target_indices, triplet_network, filename, probs, pos_label=0, pos_ratio=None, threshold=None):
    n = len(probs)
    if pos_ratio is not None and pos_ratio > 0:
        k = int(np.ceil(n * float(pos_ratio)))
        k = min(max(k, 0), n)
        if k == 0:
            y_pred = np.zeros(n, dtype=int)
        else:
            order = np.argsort(-probs)
            topk = order[:k]
            y_pred = np.zeros(n, dtype=int)
            y_pred[topk] = 1
    else:
        th = threshold if threshold is not None else 0.5
        y_pred = (probs > th).astype(int)

    with open(filename, 'w') as f:
        f.write("NodeID,Triplet,TrueLabel,PredLabel,Probability\n")
        for i, idx in enumerate(target_indices):
            node_id = int(idx)
            triplet = triplet_network.get_triplet_by_node(node_id)
            true_label = int(targets[i].item())
            pred_label = int(y_pred[i])
            prob = float(probs[i])
            triplet_str = "|".join(map(str, triplet)) if triplet else "None"
            f.write(f'{node_id},"{triplet_str}",{true_label},{pred_label},{prob:.6f}\n')

def safe_average_precision_metric(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = sanitize_score_array(y_score, context="average precision scores")
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0

def minmax_norm(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)

def zscore_norm(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    mask = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(mask):
        return out
    mean = float(np.mean(values[mask]))
    std = float(np.std(values[mask]))
    if std <= 0:
        return out
    out[mask] = (values[mask] - mean) / std
    return out

def rank_scores_desc(scores):
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return ranks / max(len(scores), 1)

def rank_normalize_desc(scores):
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return ranks / max(len(scores), 1)

def safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return np.nan
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def compute_mcc_at_pos_rate(y_true, y_score, pos_rate):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = sanitize_score_array(y_score, context="MCC scores")
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    pos_rate = float(pos_rate)
    pos_rate = max(0.0, min(1.0, pos_rate))

    n = y_true.size
    k = int(np.ceil(n * pos_rate))
    k = min(max(k, 0), n)

    y_pred = np.zeros(n, dtype=np.int64)
    if k > 0:
        order = np.argsort(-y_score)
        y_pred[order[:k]] = 1

    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return np.nan

def format_float_for_print(value):
    try:
        value = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6f}"

def softmax_weights(values, temperature=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    temp = max(float(temperature), 1e-9)
    z = values / temp
    z = z - np.max(z)
    weights = np.exp(z)
    denom = float(np.sum(weights))
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(values) / max(values.size, 1)
    return weights / denom

def normalize_selection_metric(value):
    value = str(value).strip().lower()
    aliases = {
        'val_f2': 'val_f2',
        'valf2': 'val_f2',
        'val_f1': 'val_f1',
        'val_ap': 'val_ap',
        'valap': 'val_ap',
        'val_composite': 'val_composite',
        'val_f2_epoch_composite': 'val_f2_epoch_composite',
        'selector_val_ap': 'selector_val_ap',
        'selector_val_f2': 'selector_val_f2',
        'target_pred_pos_rate': 'target_pred_pos_rate',
        'checkpoint_epoch': 'checkpoint_epoch',
        'val_f2_target_rate': 'val_f2_target_rate',
        'val_f2_target_rate_epoch': 'val_f2_target_rate_epoch',
    }
    return aliases.get(value, value)

def evaluate_binary_selector_scores(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = sanitize_score_array(y_score, context="selector scores")
    if y_true.size == 0:
        return {
            'AP': np.nan,
            'AUC_PR': np.nan,
            'F2': np.nan,
            'F1': np.nan,
            'threshold': np.nan,
            'positive_rate': np.nan,
            'num_nodes': 0,
            'num_pos': 0,
            'num_neg': 0,
            'valid': False,
        }
    positive_rate = float(np.mean(y_true == 1))
    num_pos = int(np.sum(y_true == 1))
    num_neg = int(np.sum(y_true == 0))
    if len(np.unique(y_true)) < 2:
        return {
            'AP': np.nan,
            'AUC_PR': np.nan,
            'F2': np.nan,
            'F1': np.nan,
            'threshold': np.nan,
            'positive_rate': positive_rate,
            'num_nodes': int(y_true.size),
            'num_pos': num_pos,
            'num_neg': num_neg,
            'valid': False,
        }
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    if f2_scores.size > 1:
        best_idx = int(np.argmax(f2_scores[:-1]))
        best_threshold = float(thresholds[best_idx]) if thresholds.size > 0 else np.nan
        best_f2 = float(f2_scores[best_idx])
    else:
        best_threshold = np.nan
        best_f2 = 0.0
    best_f1 = float(np.max(f1_scores[:-1])) if f1_scores.size > 1 else 0.0
    return {
        'AP': safe_average_precision_metric(y_true, y_score),
        'AUC_PR': float(auc(recalls[::-1], precisions[::-1])),
        'F2': best_f2,
        'F1': best_f1,
        'threshold': best_threshold,
        'positive_rate': positive_rate,
        'num_nodes': int(y_true.size),
        'num_pos': num_pos,
        'num_neg': num_neg,
        'valid': num_pos > 0 and num_neg > 0,
    }

def evaluate_selector_from_logits(logits, selector_val_mask, selector_val_y):
    if selector_val_mask is None or selector_val_y is None:
        return evaluate_binary_selector_scores([], [])
    if not isinstance(selector_val_mask, torch.Tensor):
        selector_val_mask = torch.as_tensor(selector_val_mask, dtype=torch.bool)
    if not isinstance(selector_val_y, torch.Tensor):
        selector_val_y = torch.as_tensor(selector_val_y, dtype=torch.long)
    if selector_val_mask.numel() == 0 or not bool(selector_val_mask.any()):
        return evaluate_binary_selector_scores([], [])
    probs = sanitize_score_array(
        F.softmax(logits[selector_val_mask.to(logits.device)], dim=1)[:, 1].detach().cpu().numpy(),
        context="selector validation probabilities"
    )
    y_true = selector_val_y[selector_val_mask.cpu()].detach().cpu().numpy()
    return evaluate_binary_selector_scores(y_true, probs)

def evaluate_selector_info_from_logits(logits, selector_info):
    if selector_info is None:
        return evaluate_binary_selector_scores([], [])
    indices = np.asarray(selector_info.get('indices', []), dtype=np.int64)
    labels = np.asarray(selector_info.get('labels', []), dtype=np.int64)
    if indices.size == 0 or labels.size == 0 or indices.size != labels.size:
        return evaluate_binary_selector_scores([], [])
    selector_idx = torch.as_tensor(indices, dtype=torch.long, device=logits.device)
    scores = sanitize_score_array(
        F.softmax(logits, dim=1)[:, 1][selector_idx].detach().cpu().numpy(),
        context="selector info probabilities"
    )
    return evaluate_binary_selector_scores(labels, scores)

def save_split_indices(split_dir, seed, train_indices, val_indices):
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, f'train_indices_seed_{seed}.npy'), np.asarray(train_indices, dtype=np.int64))
    np.save(os.path.join(split_dir, f'val_indices_seed_{seed}.npy'), np.asarray(val_indices, dtype=np.int64))

def load_split_indices(split_dir, seed):
    train_path = os.path.join(split_dir, f'train_indices_seed_{seed}.npy')
    val_path = os.path.join(split_dir, f'val_indices_seed_{seed}.npy')
    if os.path.exists(train_path) and os.path.exists(val_path):
        return np.load(train_path), np.load(val_path)
    return None, None

def validate_stratified_split_indices(train_indices, val_indices, labels, pos_label, val_ratio):
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    labels = np.asarray(labels)
    train_n = int(labels.size)
    if train_indices.ndim != 1 or val_indices.ndim != 1:
        return False, "indices are not one-dimensional"
    if train_indices.size + val_indices.size != train_n:
        return False, "train and validation sizes do not cover all nodes"
    if (
        np.any(train_indices < 0)
        or np.any(train_indices >= train_n)
        or np.any(val_indices < 0)
        or np.any(val_indices >= train_n)
    ):
        return False, "indices are outside the current graph"
    if np.unique(train_indices).size != train_indices.size:
        return False, "training indices contain duplicates"
    if np.unique(val_indices).size != val_indices.size:
        return False, "validation indices contain duplicates"
    if np.intersect1d(train_indices, val_indices, assume_unique=False).size:
        return False, "training and validation indices overlap"

    positive = labels == int(pos_label)
    expected_val_pos = int(np.sum(positive) * float(val_ratio))
    expected_val_neg = int(np.sum(~positive) * float(val_ratio))
    actual_val_pos = int(np.sum(positive[val_indices]))
    actual_val_neg = int(val_indices.size - actual_val_pos)
    if actual_val_pos != expected_val_pos or actual_val_neg != expected_val_neg:
        return False, (
            "cached class counts do not match the current stratified protocol "
            f"(actual_pos={actual_val_pos}, expected_pos={expected_val_pos}, "
            f"actual_neg={actual_val_neg}, expected_neg={expected_val_neg})"
        )
    return True, ""

def save_priority_split_indices(split_dir, seed, priority_train_indices, priority_val_indices):
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, f'priority_train_indices_seed_{seed}.npy'), np.asarray(priority_train_indices, dtype=np.int64))
    np.save(os.path.join(split_dir, f'priority_val_indices_seed_{seed}.npy'), np.asarray(priority_val_indices, dtype=np.int64))

def load_priority_split_indices(split_dir, seed):
    train_path = os.path.join(split_dir, f'priority_train_indices_seed_{seed}.npy')
    val_path = os.path.join(split_dir, f'priority_val_indices_seed_{seed}.npy')
    if os.path.exists(train_path) and os.path.exists(val_path):
        return np.load(train_path), np.load(val_path)
    return None, None

def build_priority_holdout_selector(priority_mask, train_n, val_seed=2026, priority_val_frac=0.3,
                                    selector_val_pos_rate=0.035, split_dir=None, reuse_splits=False):
    if priority_mask is None:
        priority_mask = torch.zeros(train_n, dtype=torch.bool)
    if not isinstance(priority_mask, torch.Tensor):
        pm = torch.zeros(train_n, dtype=torch.bool)
        for n in priority_mask:
            if 0 <= int(n) < train_n:
                pm[int(n)] = True
        priority_mask = pm
    priority_mask = priority_mask.cpu().bool()
    rng = np.random.default_rng(int(val_seed) if val_seed is not None else None)
    priority_nodes = torch.where(priority_mask)[0].cpu().numpy()
    priority_train_loaded, priority_val_loaded = (None, None)
    if split_dir is not None and reuse_splits:
        priority_train_loaded, priority_val_loaded = load_priority_split_indices(split_dir, int(val_seed) if val_seed is not None else 0)
    if priority_train_loaded is not None and priority_val_loaded is not None:
        priority_val = np.asarray(priority_val_loaded, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(val_seed) if val_seed is not None else None)
        rng.shuffle(priority_nodes)
        num_priority_val = int(round(len(priority_nodes) * float(priority_val_frac)))
        num_priority_val = min(max(num_priority_val, 1 if len(priority_nodes) > 0 else 0), len(priority_nodes))
        priority_val = priority_nodes[:num_priority_val]

    priority_train_mask = priority_mask.clone()
    if len(priority_val) > 0:
        priority_train_mask[torch.as_tensor(priority_val, dtype=torch.long)] = False
    priority_train_indices = torch.where(priority_train_mask)[0].cpu().numpy()
    if split_dir is not None and (priority_train_loaded is None or priority_val_loaded is None):
        save_priority_split_indices(
            split_dir,
            int(val_seed) if val_seed is not None else 0,
            priority_train_indices,
            priority_val,
        )

    selector_val_mask = torch.zeros(train_n, dtype=torch.bool)
    selector_val_y = torch.full((train_n,), -1, dtype=torch.long)
    if len(priority_val) > 0:
        selector_val_mask[torch.as_tensor(priority_val, dtype=torch.long)] = True
        selector_val_y[torch.as_tensor(priority_val, dtype=torch.long)] = 1

    neg_pool = torch.where(~priority_mask)[0].cpu().numpy()
    rng.shuffle(neg_pool)
    pos_rate = max(min(float(selector_val_pos_rate), 0.99), 1e-6)
    num_val_neg = int(round(len(priority_val) * (1.0 - pos_rate) / pos_rate))
    num_val_neg = min(max(num_val_neg, 0), len(neg_pool))
    neg_val = neg_pool[:num_val_neg]
    if len(neg_val) > 0:
        selector_val_mask[torch.as_tensor(neg_val, dtype=torch.long)] = True
        selector_val_y[torch.as_tensor(neg_val, dtype=torch.long)] = 0
    selector_indices = np.concatenate([np.asarray(priority_val, dtype=np.int64), np.asarray(neg_val, dtype=np.int64)])
    selector_labels = np.concatenate([
        np.ones(len(priority_val), dtype=np.int64),
        np.zeros(len(neg_val), dtype=np.int64),
    ])
    selector_info = {
        'indices': selector_indices,
        'labels': selector_labels,
        'pos': int(selector_labels.sum()) if selector_labels.size > 0 else 0,
        'neg': int(selector_labels.size - selector_labels.sum()) if selector_labels.size > 0 else 0,
    }

    info = {
        'priority_total': int(priority_mask.sum().item()),
        'priority_train': int(priority_train_mask.sum().item()),
        'priority_val': int(len(priority_val)),
        'selector_val_pos': int(len(priority_val)),
        'selector_val_neg': int(len(neg_val)),
        'selector_val_negatives': int(len(neg_val)),
        'selector_val_count': int(selector_val_mask.sum().item()),
        'selector_val_nodes': int(selector_val_mask.sum().item()),
        'selector_val_pos_rate': float(len(priority_val) / max(1, int(selector_val_mask.sum().item()))),
        'selector_val_valid': bool(len(priority_val) > 0 and len(neg_val) > 0 and int(selector_val_mask.sum().item()) > 0),
    }
    return priority_train_mask, selector_val_mask, selector_val_y, selector_info, info

def adapt_gatconv_checkpoint_keys(checkpoint_state, model_state):
    pass
    adapted = checkpoint_state.copy()
    model_keys = set(model_state.keys())
    converted = []

    for key in list(adapted.keys()):
        suffix = '.lin.weight'
        if not key.endswith(suffix) or key in model_keys:
            continue
        prefix = key[:-len(suffix)]
        src_key = f'{prefix}.lin_src.weight'
        dst_key = f'{prefix}.lin_dst.weight'
        if src_key in model_keys and dst_key in model_keys:
            weight = adapted.pop(key)
            adapted[src_key] = weight
            adapted[dst_key] = weight
            converted.append((key, f'{src_key}, {dst_key}'))

    for model_key in model_keys:
        suffix = '.lin.weight'
        if not model_key.endswith(suffix) or model_key in adapted:
            continue
        prefix = model_key[:-len(suffix)]
        src_key = f'{prefix}.lin_src.weight'
        dst_key = f'{prefix}.lin_dst.weight'
        if src_key in adapted and dst_key in adapted:
            adapted[model_key] = adapted[src_key]
            adapted.pop(src_key)
            adapted.pop(dst_key)
            converted.append((f'{src_key}, {dst_key}', model_key))

    if converted:
        print(
            "Adapted GATConv checkpoint keys for torch-geometric compatibility: "
            f"{len(converted)} layer(s)"
        )
    return adapted

def parse_temporal_alpha_grid(value):
    values = []
    for token in str(value).split(','):
        token = token.strip()
        if token:
            alpha = float(token)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f'temporal alpha outside [0,1]: {alpha}')
            values.append(alpha)
    if not values:
        raise ValueError('temporal alpha grid is empty')
    return sorted(set(values))

def run_experiment(seed, base_result_folder, data_train, data_test,
                   triplet_network, target_mask, target_nodes, pos_ratio=None, pos_label=0,
                   priority_mask=None, priority_frac=0.5, priority_mode='hard',
                   model_cfg=None, rw_seed=None, val_seed=None,
                   save_checkpoint_variants=False, fixed_checkpoint_epoch=25,
                   selector_val_mask=None, selector_val_y=None, selector_info=None,
                   deterministic=False, reuse_splits=False, split_dir=None):
    set_global_seed(seed, deterministic=deterministic)

    exp_folder = os.path.join(base_result_folder, f"seed_{seed}")
    os.makedirs(exp_folder, exist_ok=True)

    device = resolve_device(model_cfg.get('device', 'auto') if model_cfg is not None else 'auto')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f"使用设备: {device}")

    target_indices = torch.where(target_mask)[0]

    print(f"1. [种子 {seed}] 初始化模型...")

    if model_cfg is None:
        model_cfg = {}

    common_model_kwargs = dict(
        in_channels=data_train.x.size(1),
        hidden_channels=int(model_cfg.get('hidden_channels', 128)),
        out_channels=2,
        num_layers=int(model_cfg.get('num_layers', 3)),
        heads=int(model_cfg.get('heads', 4)),
        dropout=float(model_cfg.get('dropout', 0.3)),
        use_batchnorm=bool(model_cfg.get('use_batchnorm', True)),
        use_layernorm=bool(model_cfg.get('use_layernorm', False)),
        residual=bool(model_cfg.get('residual', True)),
        use_mlp_head=bool(model_cfg.get('use_mlp_head', True)),
        mlp_layers=int(model_cfg.get('mlp_layers', 2)),
        activation=model_cfg.get('activation', 'leaky_relu'),
        weight_init=model_cfg.get('weight_init', 'xavier'),
        use_strength_edge_weight=bool(model_cfg.get('use_strength_edge_weight', False))
    )
    model_type = model_cfg.get('model_type', 'gat')
    if model_cfg.get('edge_maxsimplex_fusion') == 'prior_logit':
        model = PriorLogitGAT(
            **common_model_kwargs,
            max_feat_dim=int(model_cfg['max_feat_dim']),
            prior_hidden_dim=int(model_cfg.get('prior_hidden_dim', 16)),
            prior_dropout=float(model_cfg.get('prior_dropout', 0.1)),
            prior_alpha=float(model_cfg.get('prior_alpha', 0.1)),
            prior_target_class=1
        ).to(device)
    elif model_type == 'dual_adj_gat':
        model = DualAdjGATMinimal(
            **common_model_kwargs,
            dual_self_hidden_dim=int(model_cfg.get('dual_self_hidden_dim', 128)),
            dual_gate_hidden_dim=int(model_cfg.get('dual_gate_hidden_dim', 64)),
            dual_dropout=float(model_cfg.get('dual_dropout', common_model_kwargs.get('dropout', 0.3))),
            history_feature_dim=int(model_cfg.get('history_feature_dim', 0)),
            history_feature_mode=str(model_cfg.get('history_feature_mode', 'none')),
            history_residual_scale=float(model_cfg.get('history_residual_scale', 1.0)),
            history_pretrained_checkpoint=str(model_cfg.get('history_pretrained_checkpoint', '') or ''),
        ).to(device)
    else:
        model = MultiLayerGAT(**common_model_kwargs).to(device)

    pretrained_checkpoint = model_cfg.get('pretrained_checkpoint')
    pretrained_checkpoint_path = None
    if pretrained_checkpoint:
        pretrained_checkpoint_path = str(pretrained_checkpoint).format(
            seed=int(seed),
            model_seed=int(seed),
            rw_seed=int(rw_seed) if rw_seed is not None else 0,
        )
        if not os.path.isfile(pretrained_checkpoint_path):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found for model_seed={seed}: "
                f"{pretrained_checkpoint_path}"
            )
        try:
            checkpoint_state = torch.load(
                pretrained_checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            checkpoint_state = torch.load(pretrained_checkpoint_path, map_location=device)
        if isinstance(checkpoint_state, dict) and 'state_dict' in checkpoint_state:
            checkpoint_state = checkpoint_state['state_dict']
        checkpoint_state = adapt_gatconv_checkpoint_keys(
            checkpoint_state,
            model.state_dict(),
        )
        model.load_state_dict(checkpoint_state, strict=True)
        print(
            f"Loaded pretrained checkpoint for model_seed={seed}: "
            f"{pretrained_checkpoint_path}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg.get('lr', 0.005)), weight_decay=float(model_cfg.get('weight_decay', 1e-4)))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"2. [种子 {seed}] 创建分层验证集...")
    val_ratio = 0.2
    train_n = data_train.num_nodes
    supervision_mask = model_cfg.get("supervision_mask")
    if supervision_mask is None:
        supervision_mask = torch.ones(train_n, dtype=torch.bool)
    elif not isinstance(supervision_mask, torch.Tensor):
        supervision_mask = torch.as_tensor(supervision_mask, dtype=torch.bool)
    supervision_mask = supervision_mask.detach().cpu().bool()
    if supervision_mask.numel() != train_n:
        raise ValueError(
            "supervision_mask length mismatch: "
            f"{supervision_mask.numel()} != {train_n}"
        )
    eligible_idx = torch.where(supervision_mask)[0].cpu().numpy()
    eligible_y = data_train.y[supervision_mask]
    pos_local = torch.where(eligible_y == pos_label)[0].cpu().numpy()
    neg_local = torch.where(eligible_y != pos_label)[0].cpu().numpy()
    pos_idx = eligible_idx[pos_local]
    neg_idx = eligible_idx[neg_local]
    cached_train_idx, cached_val_idx = (None, None)
    if split_dir is not None and reuse_splits:
        cached_train_idx, cached_val_idx = load_split_indices(split_dir, seed)
        if cached_train_idx is not None and cached_val_idx is not None:
            split_is_valid, split_error = validate_stratified_split_indices(
                cached_train_idx,
                cached_val_idx,
                data_train.y.cpu().numpy(),
                pos_label,
                val_ratio,
            )
            if not split_is_valid:
                print(
                    f"WARNING: ignoring stale cached split for seed={seed}: "
                    f"{split_error}"
                )
                cached_train_idx, cached_val_idx = (None, None)
    if cached_train_idx is not None and cached_val_idx is not None:
        val_indices = np.asarray(cached_val_idx, dtype=np.int64)
        train_indices = np.asarray(cached_train_idx, dtype=np.int64)
        val_pos = np.array([], dtype=np.int64)
        val_neg = np.array([], dtype=np.int64)
    elif val_seed is None:
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        val_pos = pos_idx[:int(len(pos_idx) * val_ratio)]
        val_neg = neg_idx[:int(len(neg_idx) * val_ratio)]
        val_indices = np.concatenate([val_pos, val_neg]) if len(val_pos) + len(val_neg) > 0 else np.array([], dtype=np.int64)
        train_indices = np.setdiff1d(
            eligible_idx, val_indices, assume_unique=False
        )
    else:
        val_rng = np.random.default_rng(int(val_seed))
        val_rng.shuffle(pos_idx)
        val_rng.shuffle(neg_idx)
        val_pos = pos_idx[:int(len(pos_idx) * val_ratio)]
        val_neg = neg_idx[:int(len(neg_idx) * val_ratio)]
        val_indices = np.concatenate([val_pos, val_neg]) if len(val_pos) + len(val_neg) > 0 else np.array([], dtype=np.int64)
        train_indices = np.setdiff1d(
            eligible_idx, val_indices, assume_unique=False
        )
    if bool(model_cfg.get('no_validation', False)):

        val_indices = np.array([], dtype=np.int64)
        train_indices = np.asarray(eligible_idx, dtype=np.int64)
    val_mask = torch.zeros(train_n, dtype=torch.bool)
    if len(val_indices) > 0:
        val_mask[torch.tensor(val_indices, dtype=torch.long)] = True
    train_mask = torch.zeros(train_n, dtype=torch.bool)
    train_mask[torch.as_tensor(train_indices, dtype=torch.long)] = True
    if split_dir is not None and (cached_train_idx is None or cached_val_idx is None):
        save_split_indices(split_dir, seed, train_indices, val_indices)

    if priority_mask is None:
        priority_mask = torch.zeros(train_n, dtype=torch.bool)
    else:
        if not isinstance(priority_mask, torch.Tensor):
            pm = torch.zeros(train_n, dtype=torch.bool)
            for n in priority_mask:
                if 0 <= n < train_n:
                    pm[n] = True
            priority_mask = pm
        else:
            if priority_mask.numel() != train_n:
                pm = torch.zeros(train_n, dtype=torch.bool)
                print("警告: priority_mask 长度与训练节点数不匹配，已重置为空 mask")
                priority_mask = pm

    print(f"Priority mask: 在训练集中优先节点数 = {int((priority_mask & train_mask).sum().item())}")

    print(f"3. [种子 {seed}] 训练集分布统计...")
    train_labels = data_train.y[train_mask].cpu().numpy()
    train_counts = np.bincount(train_labels) if train_labels.size > 0 else np.array([0, 0])
    if train_counts.size == 1:
        train_counts = np.array([train_counts[0], 0])
    print(f"训练集中分布: label0={train_counts[0]}, label1={train_counts[1]}")

    print(f"4. [种子 {seed}] 开始训练...")
    train_losses, val_losses, test_metrics = [], [], []
    epoch_metric_rows = []
    best_val_f2 = 0
    best_val_ap = -1.0
    best_val_ap_smooth = -1.0
    best_val_mcc = -np.inf
    best_val_mcc_epoch = -1
    best_selector_val_ap = -1.0
    best_selector_val_auc_pr = np.nan
    best_selector_val_f2 = np.nan
    best_selector_val_f1 = np.nan
    best_selector_val_epoch = -1
    val_ap_smooth = None
    best_val_ap_epoch = -1
    best_val_ap_smooth_epoch = -1
    best_val_f2_epoch = -1
    best_val_f1_at_f2 = 0.0
    best_val_loss_at_f2 = np.nan
    train_loss_at_best_f2 = np.nan
    best_val_ap_at_f2 = np.nan
    best_val_ap_margin_at_f2 = np.nan
    best_val_threshold = 0.0
    current_threshold = 0.
    model_selection_by = normalize_selection_metric(model_cfg.get('model_selection_by', 'val_f2'))
    best_model_selection_score = -np.inf
    best_model_selection_metric = model_selection_by
    selected_model_threshold = np.nan
    selected_checkpoint_epoch_index = -1
    selected_completed_epochs = 0
    selected_optimizer_steps = 0
    best_model_state = None
    threshold_history = []
    if pretrained_checkpoint_path is not None:
        selected_model_threshold = float(model_cfg.get('pretrained_threshold', 0.5))
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_model_selection_score = np.nan
        best_model_selection_metric = 'pretrained_checkpoint'
        selected_checkpoint_epoch_index = -1
        selected_completed_epochs = 0

    tr_counts = np.bincount(data_train.y[train_mask].cpu().numpy(), minlength=2)
    if tr_counts.size == 1:
        tr_counts = np.array([tr_counts[0], 0])
    w0 = 1
    w1 = float(tr_counts[0]) / max(1, tr_counts[1]) if tr_counts[1] > 0 else 1.0
    class_weight = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    amp_enabled = bool(model_cfg.get('amp', True)) and device.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled)
    print(f"Mixed precision training: enabled={amp_enabled}")

    epochs = int(model_cfg.get('epochs', 150))
    eval_interval = int(model_cfg.get('eval_interval', 25))
    graph_scope = str(model_cfg.get('graph_scope', 'full')).strip().lower()
    local_num_neighbors = int(model_cfg.get('local_num_neighbors', 50))
    local_num_hops = int(model_cfg.get('local_num_hops', 1))
    local_eval_batch_size = int(model_cfg.get('local_eval_batch_size', 64))
    local_train_batch_size = int(model_cfg.get('local_train_batch_size', 64))
    local_sampler = str(model_cfg.get('local_sampler', 'auto'))
    local_fallback_sample = bool(model_cfg.get('local_fallback_sample', True))
    local_train_stats_accum = {'num_batches': 0, 'sampled_nodes': 0, 'sampled_edges': 0}
    last_local_eval_stats = {
        'sampler_backend': 'full',
        'local_sampler_used': 'full',
        'avg_seed_nodes_per_batch': 0.0,
        'avg_sampled_nodes_per_batch': 0.0,
        'avg_sampled_edges_per_batch': 0.0,
        'max_sampled_nodes_per_batch': 0,
        'max_sampled_edges_per_batch': 0,
        'target_nodes_covered': int(target_indices.numel()) if graph_scope == 'full' else 0,
        'local_coverage_ratio_nodes': 0.0,
        'local_coverage_ratio_edges': 0.0,
    }
    if graph_scope not in ('full', 'local'):
        raise ValueError(f"Unsupported graph_scope={graph_scope}")

    training_epochs = 0 if pretrained_checkpoint_path is not None else epochs
    if pretrained_checkpoint_path is not None:
        print("Pretrained evaluation mode: skipping optimizer training.")
    training_started = time.perf_counter()
    for epoch in range(training_epochs):
        if graph_scope == 'local':
            total_loss, sup_loss, scaler, local_train_stats = train_local(
                model, data_train, optimizer, criterion, train_mask, scaler=scaler,
                local_train_batch_size=local_train_batch_size,
                effective_batch_size=int(model_cfg.get('effective_batch_size', 256)),
                local_num_neighbors=local_num_neighbors,
                local_num_hops=local_num_hops,
                local_sampler=local_sampler,
                local_fallback_sample=local_fallback_sample,
                priority_mask=priority_mask, priority_frac=priority_frac, priority_mode=priority_mode,
                pairwise_aux_loss=bool(model_cfg.get('pairwise_aux_loss', False)),
                pairwise_aux_weight=float(model_cfg.get('pairwise_aux_weight', 0.0)),
                pairwise_aux_margin=float(model_cfg.get('pairwise_aux_margin', 0.0)),
                pairwise_aux_max_pairs=int(model_cfg.get('pairwise_aux_max_pairs', 4096))
            )
            local_train_stats_accum['num_batches'] += int(local_train_stats.get('num_batches', 0))
            local_train_stats_accum['sampled_nodes'] += int(local_train_stats.get('sampled_nodes', 0))
            local_train_stats_accum['sampled_edges'] += int(local_train_stats.get('sampled_edges', 0))
        else:
            total_loss, sup_loss, scaler = train(
                model, data_train, optimizer, criterion, train_mask, scaler=scaler, batch_size=int(model_cfg.get('batch_size', 2048)),
                priority_mask=priority_mask, priority_frac=priority_frac, priority_mode=priority_mode,
                pairwise_aux_loss=bool(model_cfg.get('pairwise_aux_loss', False)),
                pairwise_aux_weight=float(model_cfg.get('pairwise_aux_weight', 0.0)),
                pairwise_aux_margin=float(model_cfg.get('pairwise_aux_margin', 0.0)),
                pairwise_aux_max_pairs=int(model_cfg.get('pairwise_aux_max_pairs', 4096))
            )
        train_losses.append(total_loss)

        model.eval()
        if graph_scope == 'local':
            val_mask_cpu = val_mask
            if val_mask_cpu.any():
                val_indices_tensor = torch.where(val_mask_cpu)[0]
                val_logits_np, _ = predict_seed_logits_local(
                    model,
                    data_train,
                    val_indices_tensor,
                    local_eval_batch_size=local_eval_batch_size,
                    local_num_neighbors=local_num_neighbors,
                    local_num_hops=local_num_hops,
                    local_sampler=local_sampler,
                    local_fallback_sample=local_fallback_sample,
                    device=device,
                    context='validation local evaluation',
                )
                val_logits_t = torch.from_numpy(val_logits_np).to(device)
                val_probs_class1 = sanitize_score_array(
                    F.softmax(val_logits_t, dim=1)[:, 1].detach().cpu().numpy(),
                    context="validation positive probabilities"
                )
                val_true = data_train.y[val_mask_cpu].cpu().numpy()
                val_loss = criterion(val_logits_t, data_train.y[val_mask_cpu].to(device)).item()
            else:
                val_probs_class1 = np.array([0.5])
                val_true = np.array([0])
                val_loss = total_loss
            selector_metrics = evaluate_binary_selector_scores([], [])
        else:
            with torch.no_grad():
                val_out = model_forward(model, data_train, device)
            if selector_info is not None:
                selector_metrics = evaluate_selector_info_from_logits(val_out, selector_info)
            else:
                selector_metrics = evaluate_selector_from_logits(val_out, selector_val_mask, selector_val_y)
            val_mask_cpu = val_mask
            if val_mask_cpu.any():
                val_probs_class1 = sanitize_score_array(
                    F.softmax(val_out[val_mask_cpu.to(device)], dim=1)[:, 1].cpu().numpy(),
                    context="validation positive probabilities"
                )
                val_true = data_train.y[val_mask_cpu].cpu().numpy()
            else:
                val_probs_class1 = np.array([0.5])
                val_true = np.array([0])
            val_loss = criterion(val_out[val_mask_cpu.to(device)], data_train.y[val_mask_cpu].to(device)).item() if val_mask_cpu.any() else total_loss
        val_binary = (val_true == int(pos_label)).astype(int)
        val_positive_rate = float(np.mean(val_binary)) if val_binary.size > 0 else 0.0
        val_ap = safe_average_precision_metric(val_binary, val_probs_class1)
        val_ap_margin = float(val_ap) - val_positive_rate
        val_mcc = compute_mcc_at_pos_rate(val_binary, val_probs_class1, val_positive_rate)
        if np.isfinite(val_mcc) and val_mcc > best_val_mcc:
            best_val_mcc = float(val_mcc)
            best_val_mcc_epoch = int(epoch)
        if val_ap_smooth is None:
            val_ap_smooth = val_ap
        else:
            val_ap_smooth = 0.8 * float(val_ap_smooth) + 0.2 * float(val_ap)

        best_f1 = 0.0
        use_pr_search = (len(np.unique(val_true)) > 1) and (min(np.bincount(val_true)) >= 3 if len(val_true)>0 else False)
        if use_pr_search:
            precisions, recalls, thresholds = precision_recall_curve(val_true, val_probs_class1)
            f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
            if f2_scores.size > 1:
                best_idx = np.argmax(f2_scores[:-1])
                current_threshold = thresholds[best_idx] if thresholds.size > 0 else current_threshold
                best_f2 = f2_scores[best_idx]
                best_f1 = float(np.max(f1_scores[:-1])) if f1_scores.size > 1 else 0.0
            else:
                best_f2 = 0
        else:
            best_f2 = 0

        if best_f2 > best_val_f2:
            best_val_f2 = best_f2
            best_val_f2_epoch = int(epoch)
            best_val_f1_at_f2 = float(best_f1)
            best_val_loss_at_f2 = float(val_loss)
            train_loss_at_best_f2 = float(total_loss)
            best_val_ap_at_f2 = float(val_ap)
            best_val_ap_margin_at_f2 = float(val_ap_margin)
            best_val_threshold = float(current_threshold)
            print(f"Epoch {epoch:03d}, F2={best_f2:.4f} *, 阈值={current_threshold:.4f}")
        else:
            print(f"Epoch {epoch:03d}, F2={best_f2:.4f}, 阈值={current_threshold:.4f}")

        if val_ap > best_val_ap:
            best_val_ap = float(val_ap)
            best_val_ap_epoch = int(epoch)
        if float(val_ap_smooth) > best_val_ap_smooth:
            best_val_ap_smooth = float(val_ap_smooth)
            best_val_ap_smooth_epoch = int(epoch)
        if np.isfinite(selector_metrics.get('AP', np.nan)) and float(selector_metrics['AP']) > best_selector_val_ap:
            best_selector_val_ap = float(selector_metrics['AP'])
            best_selector_val_auc_pr = float(selector_metrics.get('AUC_PR', np.nan))
            best_selector_val_f2 = float(selector_metrics.get('F2', np.nan))
            best_selector_val_f1 = float(selector_metrics.get('F1', np.nan))
            best_selector_val_epoch = int(epoch)

        if bool(model_cfg.get('no_validation', False)):
            fixed_epoch = max(1, int(fixed_checkpoint_epoch))
            if (epoch + 1) == fixed_epoch or (epoch + 1) == training_epochs:
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_model_selection_score = float(epoch + 1)
                best_model_selection_metric = 'pre_frozen_epoch'
                selected_checkpoint_epoch_index = int(epoch)
                selected_completed_epochs = int(epoch + 1)
                selected_optimizer_steps = int(local_train_stats_accum.get('num_batches', 0))
            checkpoint_score = np.nan
        elif model_selection_by == 'val_f2':
            checkpoint_score = float(best_f2)
        elif model_selection_by == 'val_f1':
            checkpoint_score = float(best_f1)
        elif model_selection_by == 'val_ap':
            checkpoint_score = float(val_ap)
        elif model_selection_by == 'val_composite':
            checkpoint_score = float(0.6 * best_f2 + 0.4 * val_ap_margin)
        elif model_selection_by == 'val_f2_epoch_composite':
            checkpoint_score = float(best_f2 - 0.001 * epoch)
        elif model_selection_by == 'selector_val_ap':
            checkpoint_score = float(selector_metrics.get('AP', np.nan))
        elif model_selection_by == 'selector_val_f2':
            checkpoint_score = float(selector_metrics.get('F2', np.nan))
        else:
            checkpoint_score = np.nan
        if not bool(model_cfg.get('no_validation', False)) and not np.isfinite(checkpoint_score):
            checkpoint_score = float(best_f2)
            best_model_selection_metric = 'val_f2_fallback'
        if not bool(model_cfg.get('no_validation', False)) and checkpoint_score > best_model_selection_score:
            best_model_selection_score = float(checkpoint_score)
            selected_model_threshold = float(current_threshold)
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            selected_checkpoint_epoch_index = int(epoch)
            selected_completed_epochs = int(epoch + 1)
            selected_optimizer_steps = int(local_train_stats_accum.get('num_batches', 0))

        threshold_history.append((epoch, float(current_threshold), float(best_f2)))

        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if bool(model_cfg.get('evaluate_test_during_training', True)) and (
            epoch % eval_interval == 0 or epoch == epochs - 1
        ):
            if graph_scope == 'local':
                metrics = test_local(
                    model,
                    data_test,
                    target_mask,
                    threshold=current_threshold,
                    pos_ratio=pos_ratio,
                    pos_label=pos_label,
                    device=device,
                    local_eval_batch_size=local_eval_batch_size,
                    local_num_neighbors=local_num_neighbors,
                    local_num_hops=local_num_hops,
                    local_sampler=local_sampler,
                    local_fallback_sample=local_fallback_sample,
                )
                last_local_eval_stats = {
                    'sampler_backend': metrics.get('sampler_backend', local_sampler),
                    'local_sampler_used': metrics.get('local_sampler_used', metrics.get('sampler_backend', local_sampler)),
                    'avg_seed_nodes_per_batch': float(metrics.get('avg_seed_nodes_per_batch', 0.0)),
                    'avg_sampled_nodes_per_batch': float(metrics.get('avg_sampled_nodes_per_batch', 0.0)),
                    'avg_sampled_edges_per_batch': float(metrics.get('avg_sampled_edges_per_batch', 0.0)),
                    'max_sampled_nodes_per_batch': int(metrics.get('max_sampled_nodes_per_batch', 0)),
                    'max_sampled_edges_per_batch': int(metrics.get('max_sampled_edges_per_batch', 0)),
                    'target_nodes_covered': int(metrics.get('target_nodes_covered', 0)),
                    'local_coverage_ratio_nodes': float(metrics.get('local_coverage_ratio_nodes', 0.0)),
                    'local_coverage_ratio_edges': float(metrics.get('local_coverage_ratio_edges', 0.0)),
                }
            else:
                metrics = test(model, data_test, target_mask, threshold=current_threshold, pos_ratio=pos_ratio, pos_label=pos_label, device=device)
            test_metrics.append({
                'epoch': epoch,
                'class0_accuracy': metrics['class0_accuracy'],
                'class1_accuracy': metrics['class1_accuracy'],
                'f1_score': metrics['f1_score'],
                'f2_score': metrics['f2_score'],
                'confusion_matrix': metrics['confusion_matrix'],
                'auc_pr': metrics['auc_pr'],
                'average_precision': metrics['average_precision'],
                'evaluation_gnn': metrics['evaluation_gnn'],
                'threshold': metrics['threshold']
            })
            baseline = float(metrics.get('baseline', 0.0) or 0.0)
            y_true_target = np.asarray(metrics.get('y_true', []))

    training_seconds = time.perf_counter() - training_started

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_eval_threshold = (
        float(selected_model_threshold)
        if np.isfinite(selected_model_threshold)
        else float(current_threshold)
    )
    configured_temporal_alpha = float(model_cfg.get('temporal_activity_alpha', 0.0))
    selected_temporal_alpha = configured_temporal_alpha
    temporal_alpha_validation_ap = np.nan
    if (
        bool(model_cfg.get('validation_select_temporal_alpha', False))
        and model_cfg.get('temporal_activity_train_scores') is not None
        and val_mask.any()
    ):
        if graph_scope == 'local':
            val_indices_tensor = torch.where(val_mask)[0]
            val_logits_np, _ = predict_seed_logits_local(
                model, data_train, val_indices_tensor,
                local_eval_batch_size=local_eval_batch_size,
                local_num_neighbors=local_num_neighbors,
                local_num_hops=local_num_hops,
                local_sampler=local_sampler,
                local_fallback_sample=local_fallback_sample,
                device=device,
                context='temporal-alpha validation evaluation',
            )
            val_logits_final = torch.from_numpy(val_logits_np)
            val_gnn_scores = F.softmax(val_logits_final, dim=1)[:, 1].cpu().numpy()
        else:
            with torch.no_grad():
                val_logits_final = model_forward(model, data_train, device)[val_mask.to(device)]
            val_gnn_scores = F.softmax(val_logits_final, dim=1)[:, 1].cpu().numpy()
        val_labels_for_alpha = (data_train.y[val_mask].cpu().numpy() == int(pos_label)).astype(np.int64)
        full_prior_scores = np.asarray(
            model_cfg.get('temporal_activity_train_scores'), dtype=np.float64
        )
        if full_prior_scores.shape != (train_n,):
            raise ValueError(
                'Temporal train-prior alignment mismatch: '
                f'{full_prior_scores.shape} != {(train_n,)}'
            )
        val_prior_scores = full_prior_scores[val_mask.cpu().numpy()]
        if not np.isfinite(val_prior_scores).all():
            raise ValueError('Temporal train-prior contains NaN/Inf on validation candidates.')
        best_alpha_key = None
        for alpha in parse_temporal_alpha_grid(model_cfg.get('temporal_alpha_grid', '0,1')):
            blended_validation = (
                (1.0 - float(alpha)) * np.asarray(val_gnn_scores, dtype=np.float64)
                + float(alpha) * val_prior_scores
            )
            validation_ap = safe_average_precision_metric(
                val_labels_for_alpha, blended_validation
            )

            alpha_key = (float(validation_ap), -float(alpha))
            if best_alpha_key is None or alpha_key > best_alpha_key:
                best_alpha_key = alpha_key
                selected_temporal_alpha = float(alpha)
                temporal_alpha_validation_ap = float(validation_ap)
        print(
            'Validation-selected temporal blend: '
            f'configured_alpha={configured_temporal_alpha:.3f}, '
            f'selected_alpha={selected_temporal_alpha:.3f}, '
            f'validation_AP={temporal_alpha_validation_ap:.6f}'
        )
    inference_started = time.perf_counter()
    if graph_scope == 'local':
        metrics = test_local(
            model,
            data_test,
            target_mask,
            threshold=final_eval_threshold,
            pos_ratio=pos_ratio,
            pos_label=pos_label,
            device=device,
            local_eval_batch_size=local_eval_batch_size,
            local_num_neighbors=local_num_neighbors,
            local_num_hops=local_num_hops,
            local_sampler=local_sampler,
            local_fallback_sample=local_fallback_sample,
        )
        last_local_eval_stats = {
            'sampler_backend': metrics.get('sampler_backend', local_sampler),
            'local_sampler_used': metrics.get(
                'local_sampler_used',
                metrics.get('sampler_backend', local_sampler),
            ),
            'avg_seed_nodes_per_batch': float(metrics.get('avg_seed_nodes_per_batch', 0.0)),
            'avg_sampled_nodes_per_batch': float(metrics.get('avg_sampled_nodes_per_batch', 0.0)),
            'avg_sampled_edges_per_batch': float(metrics.get('avg_sampled_edges_per_batch', 0.0)),
            'max_sampled_nodes_per_batch': int(metrics.get('max_sampled_nodes_per_batch', 0)),
            'max_sampled_edges_per_batch': int(metrics.get('max_sampled_edges_per_batch', 0)),
            'target_nodes_covered': int(metrics.get('target_nodes_covered', 0)),
            'local_coverage_ratio_nodes': float(metrics.get('local_coverage_ratio_nodes', 0.0)),
            'local_coverage_ratio_edges': float(metrics.get('local_coverage_ratio_edges', 0.0)),
        }
    else:
        metrics = test(
            model,
            data_test,
            target_mask,
            threshold=final_eval_threshold,
            pos_ratio=pos_ratio,
            pos_label=pos_label,
            device=device,
        )
    inference_seconds = time.perf_counter() - inference_started
    raw_gnn_metrics = metrics
    temporal_activity_scores = model_cfg.get('temporal_activity_scores')
    temporal_activity_alpha = float(selected_temporal_alpha)
    temporal_prior_metrics = None
    if temporal_activity_scores is not None and temporal_activity_alpha > 0:
        y_true_for_blend = np.asarray(metrics.get('y_true', []), dtype=np.int64)
        gnn_scores_for_blend = np.asarray(metrics.get('y_probs', []), dtype=np.float64)
        temporal_activity_scores = np.asarray(temporal_activity_scores, dtype=np.float64)
        if temporal_activity_scores.shape != gnn_scores_for_blend.shape:
            raise ValueError(
                "Temporal activity score alignment mismatch: "
                f"prior={temporal_activity_scores.shape}, "
                f"gnn={gnn_scores_for_blend.shape}"
            )
        temporal_prior_metrics = metrics_from_binary_scores(
            y_true_for_blend,
            temporal_activity_scores,
            threshold=0.5,
            pos_ratio=pos_ratio,
        )
        blended_scores = (
            (1.0 - temporal_activity_alpha) * gnn_scores_for_blend
            + temporal_activity_alpha * temporal_activity_scores
        )
        metrics = metrics_from_binary_scores(
            y_true_for_blend,
            blended_scores,
            threshold=0.5,
            pos_ratio=pos_ratio,
        )
        print(
            "Temporal activity blend: "
            f"gnn_AP={raw_gnn_metrics['auc_pr']:.6f}, "
            f"prior_AP={temporal_prior_metrics['auc_pr']:.6f}, "
            f"blended_AP={metrics['auc_pr']:.6f}, "
            f"baseline={metrics['baseline']:.6f}, "
            f"lift={metrics['improvement']:.6f}"
        )
    baseline = float(metrics.get('baseline', 0.0) or 0.0)
    y_true_target = np.asarray(metrics.get('y_true', []))
    y_pos_probs = np.asarray(metrics.get('y_probs', []))
    target_positives = int(np.sum(y_true_target == 1)) if y_true_target.size else 0
    target_total = int(y_true_target.size)
    target_threshold = float(metrics.get('threshold', final_eval_threshold))
    target_pred_pos_rate = float(np.mean(y_pos_probs >= target_threshold)) if y_pos_probs.size else np.nan
    peak_gpu_allocated_gb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)
        if device.type == 'cuda' else 0.0
    )
    peak_gpu_reserved_gb = (
        float(torch.cuda.max_memory_reserved(device)) / (1024 ** 3)
        if device.type == 'cuda' else 0.0
    )
    print(f"target positives={target_positives}, target total={target_total}")

    if val_mask.any():
        if graph_scope == 'local':
            val_indices_tensor = torch.where(val_mask)[0]
            val_logits_final_np, _ = predict_seed_logits_local(
                model,
                data_train,
                val_indices_tensor,
                local_eval_batch_size=local_eval_batch_size,
                local_num_neighbors=local_num_neighbors,
                local_num_hops=local_num_hops,
                local_sampler=local_sampler,
                local_fallback_sample=local_fallback_sample,
                device=device,
                context='saved validation score channel',
            )
            y_score_val = F.softmax(torch.from_numpy(val_logits_final_np), dim=1)[:, 1].numpy()
        else:
            with torch.no_grad():
                val_logits_final = model_forward(model, data_train, device)[val_mask.to(device)]
            y_score_val = F.softmax(val_logits_final, dim=1)[:, 1].detach().cpu().numpy()
        y_true_val = (data_train.y[val_mask].cpu().numpy() == int(pos_label)).astype(np.int64)
    else:
        y_true_val = np.asarray([], dtype=np.int64)
        y_score_val = np.asarray([], dtype=np.float64)
    if val_mask.any():
        val_triplet_rows = []
        for node_id in torch.where(val_mask)[0].cpu().numpy():
            triplet = triplet_network.get_triplet_by_node(int(node_id))
            if triplet is None or len(triplet) != 3:
                val_triplet_rows.append((-1, -1, -1))
            else:
                val_triplet_rows.append(tuple(map(int, triplet)))
        val_triplets = np.asarray(val_triplet_rows, dtype=np.int64)
    else:
        val_triplets = np.empty((0, 3), dtype=np.int64)

    return {
        'seed': seed,
        'class0_accuracy': metrics['class0_accuracy'],
        'class1_accuracy': metrics['class1_accuracy'],
        'f1_score': metrics['f1_score'],
        'f2_score': metrics['f2_score'],
        'test_mcc': float(metrics.get('mcc', np.nan)),
        'auc_pr': metrics['auc_pr'],
        'average_precision': metrics['average_precision'],
        'random_baseline': baseline,
        'auc_pr_over_random_baseline': float(metrics.get('improvement', np.nan)),
        'confusion_matrix': metrics['confusion_matrix'],
        'y_true': metrics['y_true'],
        'y_probs': metrics['y_probs'],
        'y_true_target': np.asarray(y_true_target),
        'y_score_target': np.asarray(y_pos_probs),
        'y_score_target_unfused': np.asarray(raw_gnn_metrics.get('y_probs', []), dtype=np.float64),
        'y_true_val': np.asarray(y_true_val),
        'y_score_val': np.asarray(y_score_val),
        'val_triplets': val_triplets,
        'target_positives': target_positives,
        'target_total': target_total,
        'evaluation_gnn': metrics['evaluation_gnn'],
        'rw_seed': rw_seed,
        'training_seconds': float(training_seconds),
        'inference_seconds': float(inference_seconds),
        'completed_epochs': int(training_epochs),
        'planned_epochs': int(epochs),
        'best_epoch': int(best_val_ap_epoch),
        'selected_checkpoint_epoch_index': int(selected_checkpoint_epoch_index),
        'selected_completed_epochs': int(selected_completed_epochs),
        'selected_optimizer_steps': int(selected_optimizer_steps),
        'mean_epoch_seconds': float(training_seconds / max(training_epochs, 1)),
        'peak_gpu_allocated_gb': peak_gpu_allocated_gb,
        'peak_gpu_reserved_gb': peak_gpu_reserved_gb,
        'checkpoint_selection_source': 'pre_frozen_epoch' if bool(model_cfg.get('no_validation', False)) else 'validation',
        'TEST_USED_FOR_CHECKPOINT_SELECTION': False,
        'raw_gnn_auc_pr': float(raw_gnn_metrics.get('auc_pr', np.nan)),
        'raw_gnn_lift': float(raw_gnn_metrics.get('improvement', np.nan)),
        'prediction_std': float(np.std(y_pos_probs)) if y_pos_probs.size else np.nan,
        'raw_gnn_prediction_std': float(
            np.std(np.asarray(raw_gnn_metrics.get('y_probs', []), dtype=np.float64))
        ) if len(raw_gnn_metrics.get('y_probs', [])) else np.nan,
        'candidate_alignment_hash': model_cfg.get('candidate_alignment_hash', ''),
        'activity_score_hash': model_cfg.get('activity_score_audit', {}).get('score_hash', ''),
        'activity_score_min': model_cfg.get('activity_score_audit', {}).get('min', np.nan),
        'activity_score_max': model_cfg.get('activity_score_audit', {}).get('max', np.nan),
        'activity_score_mean': model_cfg.get('activity_score_audit', {}).get('mean', np.nan),
        'activity_score_std': model_cfg.get('activity_score_audit', {}).get('std', np.nan),
        'strength_feature': bool(model_cfg.get('use_strength_feature', False)),
        'strength_edge': bool(model_cfg.get('use_strength_edge_weight', False)),
        'temporal_activity_alpha': temporal_activity_alpha,
        'configured_temporal_activity_alpha': configured_temporal_alpha,
        'temporal_alpha_validation_ap': temporal_alpha_validation_ap,
        'temporal_prior_kind': str(model_cfg.get('temporal_prior_kind', 'recent_activity')),
        'temporal_activity_prior_auc_pr': (
            float(temporal_prior_metrics.get('auc_pr', np.nan))
            if temporal_prior_metrics is not None
            else np.nan
        ),
        'temporal_activity_prior_lift': (
            float(temporal_prior_metrics.get('improvement', np.nan))
            if temporal_prior_metrics is not None
            else np.nan
        ),
        'best_val_ap': float(best_val_ap),
        'best_val_ap_epoch': int(best_val_ap_epoch),
        'best_val_ap_smooth': float(best_val_ap_smooth),
        'best_val_ap_smooth_epoch': int(best_val_ap_smooth_epoch),
        'best_val_mcc': float(best_val_mcc) if np.isfinite(best_val_mcc) else np.nan,
        'best_val_mcc_epoch': int(best_val_mcc_epoch),
        'best_selector_val_ap': float(best_selector_val_ap) if best_selector_val_ap >= 0 else np.nan,
        'best_selector_val_auc_pr': float(best_selector_val_auc_pr),
        'best_selector_val_f2': float(best_selector_val_f2),
        'best_selector_val_f1': float(best_selector_val_f1),
        'best_selector_val_epoch': int(best_selector_val_epoch),
        'selector_val_ap': float(best_selector_val_ap) if best_selector_val_ap >= 0 else np.nan,
        'selector_val_auc_pr': float(best_selector_val_auc_pr),
        'selector_val_f2': float(best_selector_val_f2),
        'selector_val_f1': float(best_selector_val_f1),
        'selector_val_count': int(selector_info.get('labels', np.array([])).size) if selector_info is not None else int(selector_metrics.get('num_nodes', 0)) if 'selector_metrics' in locals() else 0,
        'selector_val_pos': int(selector_info.get('pos', 0)) if selector_info is not None else int(selector_metrics.get('num_pos', 0)) if 'selector_metrics' in locals() else 0,
        'selector_val_neg': int(selector_info.get('neg', 0)) if selector_info is not None else int(selector_metrics.get('num_neg', 0)) if 'selector_metrics' in locals() else 0,
        'best_val_f2': float(best_val_f2),
        'best_val_f2_epoch': int(best_val_f2_epoch),
        'best_val_f1': float(best_val_f1_at_f2),
        'best_val_loss': float(best_val_loss_at_f2) if not np.isnan(best_val_loss_at_f2) else np.nan,
        'train_loss_at_checkpoint': float(train_loss_at_best_f2) if not np.isnan(train_loss_at_best_f2) else np.nan,
        'best_val_threshold': float(best_val_threshold),
        'model_selection_by': model_selection_by,
        'model_selection_score': float(best_model_selection_score),
        'model_selection_metric': best_model_selection_metric,
        'best_val_ap_at_f2': float(best_val_ap_at_f2) if not np.isnan(best_val_ap_at_f2) else np.nan,
        'best_val_ap_margin': float(best_val_ap_margin_at_f2) if not np.isnan(best_val_ap_margin_at_f2) else np.nan,
        'target_pred_pos_rate': target_pred_pos_rate,
        'graph_scope': graph_scope,
        'local_num_hops': int(local_num_hops),
        'local_num_neighbors': int(local_num_neighbors),
        'local_train_batch_size': int(local_train_batch_size),
        'local_eval_batch_size': int(local_eval_batch_size),
        'local_sampler': last_local_eval_stats.get('sampler_backend', local_sampler),
        'local_sampler_used': last_local_eval_stats.get('local_sampler_used', last_local_eval_stats.get('sampler_backend', local_sampler)),
        'avg_seed_nodes_per_batch': float(last_local_eval_stats.get('avg_seed_nodes_per_batch', 0.0)),
        'avg_sampled_nodes_per_batch': float(last_local_eval_stats.get('avg_sampled_nodes_per_batch', 0.0)),
        'avg_sampled_edges_per_batch': float(last_local_eval_stats.get('avg_sampled_edges_per_batch', 0.0)),
        'max_sampled_nodes_per_batch': int(last_local_eval_stats.get('max_sampled_nodes_per_batch', 0)),
        'max_sampled_edges_per_batch': int(last_local_eval_stats.get('max_sampled_edges_per_batch', 0)),
        'target_nodes_covered': int(last_local_eval_stats.get('target_nodes_covered', 0)),
        'local_coverage_ratio_nodes': float(last_local_eval_stats.get('local_coverage_ratio_nodes', 0.0)),
        'local_coverage_ratio_edges': float(last_local_eval_stats.get('local_coverage_ratio_edges', 0.0)),
        'model_type': model_cfg.get('model_type', 'gat'),
        'dual_self_hidden_dim': model_cfg.get('dual_self_hidden_dim', np.nan),
        'dual_gate_hidden_dim': model_cfg.get('dual_gate_hidden_dim', np.nan),
        'dual_dropout': model_cfg.get('dual_dropout', np.nan),
        'pairwise_aux_loss': model_cfg.get('pairwise_aux_loss', False),
        'pairwise_aux_weight': model_cfg.get('pairwise_aux_weight', 0.0),
        'pairwise_aux_margin': model_cfg.get('pairwise_aux_margin', 0.0),
        'pairwise_aux_max_pairs': model_cfg.get('pairwise_aux_max_pairs', 4096),
        'edge_maxsimplex_fusion': model_cfg.get('edge_maxsimplex_fusion', 'none'),
        'edge_maxsimplex_ablation': model_cfg.get('edge_maxsimplex_ablation', 'real'),
        'prior_alpha': model_cfg.get('prior_alpha', np.nan),
        'prior_hidden_dim': model_cfg.get('prior_hidden_dim', np.nan),
        'prior_dropout': model_cfg.get('prior_dropout', np.nan),
        'max_feat_dim': model_cfg.get('max_feat_dim', 0),
    }

def load_priority_mask_for_train(args, train_n):
    if args.priority_mask_npz is not None:
        try:
            z = np.load(args.priority_mask_npz, allow_pickle=True)
            arr = z['priority'] if 'priority' in z else z[list(z.keys())[0]]
            if arr.dtype == bool:
                mask = torch.from_numpy(arr.astype(np.bool_))
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                mask = torch.zeros(train_n, dtype=torch.bool)
                for i in arr:
                    if 0 <= int(i) < train_n:
                        mask[int(i)] = True
            else:
                mask = torch.zeros(train_n, dtype=torch.bool)
            print(f"Loaded priority mask from npz with {int(mask.sum().item())} True")
            return mask
        except Exception as e:
            print(f"priority_mask_npz load failed: {e}")
            return torch.zeros(train_n, dtype=torch.bool)

    if args.priority_nodes is not None and os.path.exists(args.priority_nodes):
        nodes = []
        with open(args.priority_nodes, 'r') as f:
            for line in f:
                parts = re.split(r'[,\s]+', line.strip())
                for p in parts:
                    if p == '':
                        continue
                    try:
                        nodes.append(int(p))
                    except Exception:
                        pass
        pm = torch.zeros(train_n, dtype=torch.bool)
        for n in nodes:
            if 0 <= n < train_n:
                pm[n] = True
        print(f"Loaded priority nodes from file: {len(nodes)} nodes with {int(pm.sum().item())} in training set")
        return pm

    if args.priority_nodes is not None:
        print(f"priority_nodes file does not exist: {args.priority_nodes}")
    return torch.zeros(train_n, dtype=torch.bool)

def safe_float_value(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default

def infer_dataset_name(args):
    if getattr(args, 'dataset', None):
        return str(args.dataset)
    for candidate in (getattr(args, 'target', None), getattr(args, 'train', None), getattr(args, 'test', None)):
        if not candidate:
            continue
        name = infer_dataset_name_from_target(candidate)
        if name:
            return name
        stem = Path(candidate).stem
        for suffix in ('_openTriangles_test', '_train_1', '_all_1'):
            if stem.endswith(suffix):
                return stem[:-len(suffix)].replace('classified_network_', '')
    return 'unknown'

def compact_run_parameters(args):
    keys = [
        'input_mode', 'model_type', 'rw_impl', 'rw_seeds', 'model_seeds', 'rw_walks_per_node',
        'rw_epochs', 'rw_embedding_dim', 'rw_max_pairs', 'rw_reference_cache', 'rw_negative_samples', 'rw_negative_power',
        'rw_restart_prob', 'rw_context_hops', 'rw_graph_scope', 'rw_prune_fanout',
        'graph_scope', 'local_num_hops', 'local_num_neighbors', 'local_train_batch_size',
        'local_eval_batch_size', 'epochs', 'batch_size', 'lr', 'weight_decay', 'amp',
        'model_num_layers', 'model_hidden_channels', 'model_heads', 'model_dropout',
        'pretrained_checkpoint', 'pretrained_threshold',
        'dual_self_hidden_dim', 'dual_gate_hidden_dim', 'dual_dropout',
        'edge_maxsimplex_fusion', 'edge_maxsimplex_ablation', 'prior_alpha',
        'prior_hidden_dim', 'prior_dropout', 'device', 'train', 'test', 'target',
        'priority_nodes', 'temporal_activity_alpha', 'temporal_prior_kind',
        'temporal_prior_maxsimplex_weight', 'temporal_prior_maxsimplex_column',
        'validation_select_temporal_alpha', 'temporal_alpha_grid',
        'temporal_activity_recent_fraction', 'temporal_activity_context_ratio', 'temporal_activity_measure', 'temporal_activity_aggregation',
        'temporal_activity_entity', 'results'
    ]
    params = {}
    for key in keys:
        if hasattr(args, key):
            params[key] = getattr(args, key)
    return params

def summarize_minimal_results(rows, args):
    auc_values = np.asarray([safe_float_value(r.get('auc_pr', np.nan)) for r in rows], dtype=np.float64)
    baseline_values = np.asarray([safe_float_value(r.get('random_baseline', np.nan)) for r in rows], dtype=np.float64)
    lift_values = np.asarray([safe_float_value(r.get('auc_pr_over_random_baseline', np.nan)) for r in rows], dtype=np.float64)
    return {
        'dataset': infer_dataset_name(args),
        'mean_AUC_PR': float(np.nanmean(auc_values)) if auc_values.size else np.nan,
        'randomline': float(np.nanmean(baseline_values)) if baseline_values.size else np.nan,
        'liftx': float(np.nanmean(lift_values)) if lift_values.size else np.nan,
        'run_parameters': json.dumps(compact_run_parameters(args), ensure_ascii=False, sort_keys=True),
    }

def write_minimal_summary(base_result_folder, rows, args):
    summary = summarize_minimal_results(rows, args)
    csv_path = os.path.join(base_result_folder, 'summary.csv')
    txt_path = os.path.join(base_result_folder, 'summary.txt')
    fieldnames = ['dataset', 'mean_AUC_PR', 'randomline', 'liftx', 'run_parameters']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for key in fieldnames:
            f.write(f"{key}: {summary[key]}\n")
    print("\nFinal summary")
    print(f"dataset={summary['dataset']}")
    print(f"mean_AUC_PR={summary['mean_AUC_PR']:.6f}")
    print(f"randomline={summary['randomline']:.6f}")
    print(f"liftx={summary['liftx']:.6f}")
    print(f"run_parameters={summary['run_parameters']}")
    print(f"summary saved to: {csv_path}")
    return summary

def write_seed_results(base_result_folder, rows):
    csv_path = os.path.join(base_result_folder, 'seed_results.csv')
    fieldnames = [
        'seed',
        'rw_seed',
        'auc_pr',
        'random_baseline',
        'auc_pr_over_random_baseline',
        'raw_gnn_auc_pr',
        'raw_gnn_lift',
        'prediction_std',
        'raw_gnn_prediction_std',
        'candidate_alignment_hash',
        'activity_score_hash',
        'activity_score_min',
        'activity_score_max',
        'activity_score_mean',
        'activity_score_std',
        'strength_feature',
        'strength_edge',
        'temporal_activity_alpha',
        'configured_temporal_activity_alpha',
        'temporal_alpha_validation_ap',
        'temporal_prior_kind',
        'temporal_activity_prior_auc_pr',
        'temporal_activity_prior_lift',
        'rw_seconds',
        'training_seconds',
        'inference_seconds',
        'completed_epochs',
        'planned_epochs',
        'best_epoch',
        'selected_checkpoint_epoch_index',
        'selected_completed_epochs',
        'selected_optimizer_steps',
        'mean_epoch_seconds',
        'peak_gpu_allocated_gb',
        'peak_gpu_reserved_gb',
        'checkpoint_selection_source',
        'TEST_USED_FOR_CHECKPOINT_SELECTION',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, np.nan) for key in fieldnames})
    print(f"per-seed results saved to: {csv_path}")
    return csv_path

def apply_large_tags_defaults(args):
    if getattr(args, 'preset', 'original_f2') != 'original_f2':
        raise ValueError(f"Unsupported preset: {args.preset}")
    return args

def load_sampling_stats_for_args(args):
    return None

def sampling_result_fields(sampling_stats):
    if not sampling_stats:
        return {'sampled_dataset': False}
    fields = dict(sampling_stats)
    fields.setdefault('sampled_dataset', True)
    return fields

def infer_single_rw_config_name(args):
    return (
        f"rw_{args.rw_impl}_w{int(args.rw_walks_per_node)}"
        f"_e{int(args.rw_epochs)}_d{int(args.rw_embedding_dim)}"
    )

def build_rw_param_sweep_configs(args):
    return [{
        'name': infer_single_rw_config_name(args),
        'rw_walks_per_node': int(args.rw_walks_per_node),
        'rw_epochs': int(args.rw_epochs),
        'rw_embedding_dim': int(args.rw_embedding_dim),
    }]

def apply_rw_sweep_config(args, rw_cfg):
    for key, value in rw_cfg.items():
        if key != 'name' and hasattr(args, key):
            setattr(args, key, value)
    return args

def main_rwsearch():
    parser = argparse.ArgumentParser(description="Dual-Mapping GAT with configurable random-walk seed search")
    parser.add_argument('--run_mode', type=str, default='exact_fast', choices=['exact_fast', 'debug_selector'])
    parser.add_argument('--preset', type=str, default='original_f2', choices=['original_f2'])
    parser.add_argument('--input_mode', type=str, default='protocol', choices=['gpickle', 'raw', 'protocol'])
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name used when input_mode=raw/protocol.')
    parser.add_argument('--processed_dir', type=str, default=None, help='Optional processed protocol root used when input_mode=raw/protocol.')
    parser.add_argument('--protocol_root', type=str, default=None, help='Optional protocol triangle root; overrides processed_dir search order.')
    parser.add_argument('--generated_gpickle_dir', type=str, default=None, help='Where input_mode=raw/protocol writes generated gpickle files.')
    parser.add_argument('--generated_gpickle_feature_mode', type=str, default='ids', choices=['local', 'ids'])
    parser.add_argument('--generated_gpickle_closed_label', type=int, choices=[0, 1], default=1)
    parser.add_argument(
        '--protocol_graph_triangle_scope',
        type=str,
        default='legacy_closed_open',
        choices=['legacy_closed_open', 'open_only'],
        help='legacy_closed_open keeps historical closed and open triangle nodes; open_only builds candidate-only graphs from openTriangles and labels new_closures as positives.',
    )
    parser.add_argument('--include_new_closures_as_nodes', type=parse_bool, default=True)
    parser.add_argument(
        '--protocol_use_cli_target_paths',
        type=parse_bool,
        default=False,
        help='When false, protocol mode always uses processed/<dataset>/train|test target files.'
    )
    parser.add_argument('--nverts', type=str, default=None)
    parser.add_argument('--simplices', type=str, default=None)
    parser.add_argument('--times', type=str, default=None)
    parser.add_argument(
        '--protocol-event-order',
        choices=['raw', 'timestamp'],
        default='raw',
        help='Ordering used by context-only strength and activity calculations.',
    )
    parser.add_argument(
        '--protocol-data-root',
        type=str,
        default=None,
        help='Isolated root containing dataset/ protocol assets.',
    )
    parser.add_argument('--train', type=str, default="classified_network_train_1.gpickle")
    parser.add_argument('--test', type=str, default="classified_network_all_1.gpickle")
    parser.add_argument('--target', type=str, default="dataset/test/email-Enron_openTriangles_test.txt")
    parser.add_argument('--results', type=str, default="results_email_enron_dual_adj_min_protocol")
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--pos_ratio', type=str, default=None)
    parser.add_argument('--pos_label', type=int, default=None)
    parser.add_argument('--priority_nodes', type=str, default="dataset/train/email-Enron_new_closures_train.txt")
    parser.add_argument('--priority_mask_npz', type=str, default=None)
    parser.add_argument('--priority_frac', type=float, default=0.5)
    parser.add_argument('--priority_mode', type=str, default='hard', choices=['hard','soft'])
    parser.add_argument(
        '--supervision_scope',
        type=str,
        default='all_nodes',
        choices=['all_nodes', 'open_candidates'],
        help='Restrict supervised loss/validation to temporal open-triangle candidates.',
    )
    parser.add_argument(
        '--temporal_activity_alpha',
        type=float,
        default=0.0,
        help='Blend weight for a context-only recent-activity rank prior at final evaluation.',
    )
    parser.add_argument('--temporal_activity_recent_fraction', type=float, default=0.15)
    parser.add_argument(
        '--temporal_activity_measure', type=str, default='count',
        choices=['count', 'last_event'],
        help='count uses recent event frequency; last_event ranks the latest raw-event index of the weakest candidate face.',
    )
    parser.add_argument(
        '--temporal_activity_context_ratio', type=float, default=0.8,
        help='Raw-event context cutoff used for the final activity prior. 1.0 is transductive and uses post-cutoff raw events.',
    )
    parser.add_argument(
        '--temporal_prior_kind', type=str, default='recent_activity',
        choices=['recent_activity', 'candidate_law', 'maxsimplex_rank', 'activity_maxsimplex'],
        help='History-only score prior: recent activity, legacy candidate law, direct maximal-simplex rank, or their weighted combination.',
    )
    parser.add_argument('--candidate_law_cache_dir', type=str, default=None)
    parser.add_argument(
        '--temporal_prior_maxsimplex_column', type=str,
        default='tri_all_edges_shared_count',
        help='Column used by --temporal_prior_kind maxsimplex_rank/activity_maxsimplex.',
    )
    parser.add_argument(
        '--temporal_prior_maxsimplex_weight', type=float,
        default=0.5,
        help='Maximal-simplex weight for --temporal_prior_kind activity_maxsimplex.',
    )
    parser.add_argument('--validation_select_temporal_alpha', type=parse_bool, default=False)
    parser.add_argument(
        '--temporal_alpha_grid', type=str,
        default='0,0.05,0.1,0.2,0.35,0.5,0.7,0.85,1.0',
    )
    parser.add_argument(
        '--temporal_activity_aggregation',
        type=str,
        default='product',
        choices=['min', 'product', 'mean'],
    )
    parser.add_argument(
        '--temporal_activity_entity',
        type=str,
        default='node',
        choices=['node', 'edge'],
    )
    parser.add_argument(
        '--use-strength-feature',
        type=parse_bool,
        default=False,
        help='Append leakage-safe log1p pair-count features; default preserves legacy input exactly.',
    )
    parser.add_argument(
        '--use-strength-edge-weight',
        type=parse_bool,
        default=False,
        help='Add weighted historical-neighbor messages without changing the derived node/edge sets.',
    )
    parser.add_argument('--train_strength_context_ratio', type=float, default=0.6)
    parser.add_argument('--test_strength_context_ratio', type=float, default=0.8)
    parser.add_argument(
        '--history-feature-npz', type=str, default=None,
        help='Leakage-audited candidate history vectors aligned to train/test triplets.',
    )
    parser.add_argument(
        '--history-feature-mode', choices=[
            'none', 'concat', 'residual', 'gated_residual',
            'history_base_structural_residual',
        ], default='none',
    )
    parser.add_argument('--history-residual-scale', type=float, default=1.0)
    parser.add_argument('--history-pretrained-checkpoint', type=str, default=None,
                        help='Shared HistoryMLP checkpoint for history_base_structural_residual.')
    parser.add_argument(
        '--use-temporal-score-prior',
        type=parse_bool,
        default=False,
        help='Enable the historical score-level temporal activity blend.',
    )
    parser.add_argument('--use_edge_maxsimplex_features', type=str, default='false')
    parser.add_argument('--train_edge_maxsimplex_txt', type=str, default=None)
    parser.add_argument('--test_edge_maxsimplex_txt', type=str, default=None)
    parser.add_argument('--edge_maxsimplex_raw_dir', type=str, default=None)
    parser.add_argument('--train_edge_maxsimplex_context_ratio', type=float, default=0.6)
    parser.add_argument('--test_edge_maxsimplex_context_ratio', type=float, default=0.8)
    parser.add_argument('--edge_maxsimplex_fusion', type=str, default='none', choices=['none', 'concat', 'prior_logit'])
    parser.add_argument('--edge_maxsimplex_ablation', type=str, default='real', choices=['real', 'zero', 'shuffle'])
    parser.add_argument('--prior_alpha', type=float, default=0.1)
    parser.add_argument('--prior_hidden_dim', type=int, default=16)
    parser.add_argument('--prior_dropout', type=float, default=0.1)
    parser.add_argument('--maxsimplex_shuffle_seed', type=int, default=2026)
    parser.add_argument('--seeds', type=str, default="48")
    parser.add_argument('--model_seeds', type=str, default="48")
    parser.add_argument('--val_seed', type=int, default=2026)
    parser.add_argument('--no_fixed_val_seed', dest='val_seed', action='store_const', const=None)
    parser.add_argument('--rw_seed', type=int, default=0)
    parser.add_argument('--rw_seeds', type=str, default="0")
    parser.add_argument('--rw_cache_mode', type=str, default='per_rw_seed', choices=['none', 'per_rw_seed'])
    parser.add_argument('--rw_cache_dir', type=str, default='.cache_rw')
    parser.add_argument(
        '--rw_reference_cache',
        type=str,
        default=None,
        help=(
            'Load a frozen train_rw/test_rw NPZ directly. The path may contain '
            '{rw_seed}; graph-file hashes are intentionally ignored after strict '
            'shape and RW-configuration validation.'
        ),
    )
    parser.add_argument('--rw_impl', type=str, default='fast_positive', choices=['original', 'fast_positive', 'fast_sgns', 'ppmi_svd', 'hybrid_fast_ppmi'])
    parser.add_argument('--disable-rw', action='store_true',
                        help='Use an all-zero RW channel for audited no-RW Dual-Mapping runs.')
    parser.add_argument('--rw_walk_length', type=int, default=10)
    parser.add_argument('--rw_walks_per_node', type=int, default=30)
    parser.add_argument('--rw_window_size', type=int, default=2)
    parser.add_argument('--rw_embedding_dim', type=int, default=64)
    parser.add_argument('--rw_epochs', type=int, default=10)
    parser.add_argument(
        '--rw_max_pairs',
        type=int,
        default=20_000_000,
        help='Hard cap on preallocated random-walk positive pairs.',
    )
    parser.add_argument('--rw_negative_samples', type=int, default=5)
    parser.add_argument('--rw_negative_power', type=float, default=0.75)
    parser.add_argument('--rw_restart_prob', type=float, default=0.0)
    parser.add_argument('--rw_context_hops', type=int, default=5)
    parser.add_argument('--graph_scope', type=str, default='full', choices=['full', 'local'])
    parser.add_argument('--local_num_neighbors', type=int, default=50)
    parser.add_argument('--local_num_hops', type=int, default=1)
    parser.add_argument('--local_eval_batch_size', type=int, default=64)
    parser.add_argument('--local_train_batch_size', type=int, default=64)
    parser.add_argument('--local_sampler', type=str, default='auto', choices=['auto', 'neighborloader', 'fallback'])
    parser.add_argument('--local_fallback_sample', type=parse_bool, default=True)
    parser.add_argument('--rw_graph_scope', type=str, default='full', choices=['full', 'pruned'])
    parser.add_argument('--rw_prune_fanout', type=int, default=50)
    parser.add_argument('--rw_hybrid_stoch_dim', type=int, default=32)
    parser.add_argument('--rw_hybrid_ppmi_dim', type=int, default=32)
    parser.add_argument('--rw_hybrid_ppmi_hops', type=int, default=5)
    parser.add_argument('--rw_hybrid_l2norm', type=parse_bool, default=True)
    parser.add_argument('--rw_param_sweep', type=str, default='false')
    parser.add_argument('--model_num_layers', type=int, default=8)
    parser.add_argument('--model_hidden_channels', type=int, default=128)
    parser.add_argument('--model_heads', type=int, default=2)
    parser.add_argument('--model_dropout', type=float, default=0.1)
    parser.add_argument(
        '--pretrained_checkpoint',
        type=str,
        default=None,
        help=(
            'Evaluate a frozen model without retraining. The path may contain '
            '{model_seed}, {seed}, and {rw_seed}.'
        ),
    )
    parser.add_argument(
        '--pretrained_threshold',
        type=float,
        default=0.5,
        help='Classification threshold used only for F1/F2 in pretrained evaluation; AUC-PR is unaffected.',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='dual_adj_gat',
        choices=['gat', 'dual_adj_gat'],
        help='gat: original MultiLayerGAT; dual_adj_gat: GAT branch + self feature branch + gated fusion'
    )
    parser.add_argument('--dual_self_hidden_dim', type=int, default=128)
    parser.add_argument('--dual_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--dual_dropout', type=float, default=0.3)
    parser.add_argument('--dual_stability_preset', type=str, default='none', choices=['none', 'small'])
    parser.add_argument('--model_use_batchnorm', action='store_true')
    parser.add_argument('--model_use_layernorm', action='store_true')
    parser.add_argument('--model_residual', action='store_true')
    parser.add_argument('--model_use_mlp_head', action='store_true')
    parser.add_argument('--model_mlp_layers', type=int, default=2)
    parser.add_argument('--model_activation', type=str, default='leaky_relu', choices=['leaky_relu','relu'])
    parser.add_argument('--model_weight_init', type=str, default='xavier', choices=['xavier','kaiming'])
    parser.add_argument('--pairwise_aux_loss', type=parse_bool, default=False)
    parser.add_argument('--pairwise_aux_weight', type=float, default=0.0)
    parser.add_argument('--pairwise_aux_margin', type=float, default=0.0)
    parser.add_argument('--pairwise_aux_max_pairs', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--effective-batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--test_eval_interval', type=int, default=25)
    parser.add_argument(
        '--evaluate-test-during-training',
        type=parse_bool,
        default=True,
        help='When false, freeze the validation-selected checkpoint before the first test evaluation.',
    )
    parser.add_argument(
        '--no-validation',
        type=parse_bool,
        default=False,
        help='Final refit mode: train on all eligible training candidates and use the pre-frozen epoch.',
    )
    parser.add_argument('--model_selection_by', type=str, default='val_f2', choices=['val_ap', 'val_f1', 'val_f2', 'val_composite', 'val_f2_epoch_composite', 'selector_val_ap', 'selector_val_f2', 'target_pred_pos_rate', 'checkpoint_epoch', 'val_f2_target_rate', 'val_f2_target_rate_epoch'])
    parser.add_argument('--selection_metric', type=str, default=None)
    parser.add_argument(
        '--ensemble_method',
        type=str,
        default='prob_weighted',
        choices=['rank_topk', 'rank_mean', 'rank_max', 'rank_weighted_by_val_f2', 'prob_weighted_by_val_f2', 'prob_weighted', 'raw_mean']
    )
    parser.add_argument('--ensemble_topk', type=int, default=5)
    parser.add_argument('--ensemble_select_by', type=str, default='val_f2', choices=['val_ap', 'val_f1', 'val_f2', 'val_composite', 'val_f2_epoch_composite', 'selector_val_ap', 'selector_val_f2', 'target_pred_pos_rate', 'checkpoint_epoch', 'val_f2_target_rate', 'val_f2_target_rate_epoch'])
    parser.add_argument('--ensemble_temperature', type=float, default=0.01)
    parser.add_argument('--max_target_pred_pos_rate_for_ensemble', type=float, default=0.85)
    parser.add_argument('--min_target_pred_pos_rate_for_ensemble', type=float, default=0.0)
    parser.add_argument('--deterministic', type=str, default='false')
    parser.add_argument(
        '--amp',
        type=parse_bool,
        default=True,
        help='Enable CUDA automatic mixed precision. Disable for large graphs if FP16 gradients overflow.',
    )
    parser.add_argument('--reuse_splits', type=str, default='true')
    parser.add_argument('--selector_val', type=str, default='none', choices=['none', 'priority_holdout'])
    parser.add_argument('--selector_removes_priority_from_training', type=str, default='false')
    parser.add_argument('--priority_val_frac', type=float, default=0.3)
    parser.add_argument('--selector_val_pos_rate', type=float, default=0.035)
    parser.add_argument('--save_checkpoint_variants', type=str, default='false')
    parser.add_argument('--fixed_checkpoint_epoch', type=int, default=25)
    parser.add_argument('--selection_dev_rw_seeds', type=str, default=None)
    parser.add_argument('--selection_eval_rw_seeds', type=str, default=None)
    parser.add_argument('--enable_selection_rule_search', type=str, default='false')
    args = parser.parse_args()
    apply_large_tags_defaults(args)
    if not parse_bool(args.use_temporal_score_prior):
        if float(args.temporal_activity_alpha) != 0.0:
            print(
                'Temporal alpha was supplied but --use-temporal-score-prior is false; '
                'score-level prior is disabled.'
            )
        args.temporal_activity_alpha = 0.0
    elif not (0.0 <= float(args.temporal_activity_alpha) <= 1.0):
        raise ValueError('--temporal_activity_alpha must be in [0, 1].')
    if args.selection_metric is not None:
        args.model_selection_by = normalize_selection_metric(args.selection_metric)
        args.ensemble_select_by = normalize_selection_metric(args.selection_metric)
    else:
        args.model_selection_by = normalize_selection_metric(args.model_selection_by)
        args.ensemble_select_by = normalize_selection_metric(args.ensemble_select_by)
    if args.rw_impl == 'hybrid_fast_ppmi':
        hybrid_dim = int(args.rw_hybrid_stoch_dim) + int(args.rw_hybrid_ppmi_dim)
        if hybrid_dim != int(args.rw_embedding_dim):
            raise ValueError(
                "For rw_impl=hybrid_fast_ppmi, rw_hybrid_stoch_dim + rw_hybrid_ppmi_dim "
                f"must equal rw_embedding_dim ({args.rw_hybrid_stoch_dim} + "
                f"{args.rw_hybrid_ppmi_dim} != {args.rw_embedding_dim})"
            )
    if parse_bool(args.enable_selection_rule_search):
        print("Validation-rule search is disabled for exact_fast runs; ignoring --enable_selection_rule_search.")
    args.enable_selection_rule_search = 'false'
    if args.run_mode == 'exact_fast':

        pass
    deterministic = parse_bool(args.deterministic)
    reuse_splits = parse_bool(args.reuse_splits)
    selector_removes_priority_from_training = parse_bool(args.selector_removes_priority_from_training)
    enable_selection_rule_search = parse_bool(args.enable_selection_rule_search)
    sampling_stats = load_sampling_stats_for_args(args)
    sample_fields = sampling_result_fields(sampling_stats)
    if sample_fields.get('sampled_dataset', False):
        if int(sample_fields.get('sample_target_positives', 0)) <= 0:
            raise RuntimeError("sampled dataset target positives=0; refusing to continue.")
        print(
            "Sampled dataset mode: "
            f"sample_name={sample_fields.get('sample_name', '')}, "
            f"sample_mode={sample_fields.get('sample_mode', '')}, "
            f"debug_only={sample_fields.get('sample_debug_only', False)}, "
            f"target_total={sample_fields.get('sample_target_total', np.nan)}, "
            f"target_positives={sample_fields.get('sample_target_positives', np.nan)}, "
            f"random_baseline_AUC_PR={sample_fields.get('sample_random_baseline_AUC_PR', np.nan)}"
        )
        if sample_fields.get('sample_debug_only', False):
            print("WARNING: debug_only=True; this result is not a formal benchmark result.")
            if sample_fields.get('sample_leakage_warning', ''):
                print(f"WARNING: {sample_fields.get('sample_leakage_warning')}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_result_folder = args.results or f'experiment_results_rwsearch_{timestamp}'
    os.makedirs(base_result_folder, exist_ok=True)
    if args.input_mode in {'raw', 'protocol'}:
        print(f"input_mode={args.input_mode}: generating protocol Dual-Mapping gpickle files before training.")
        _generate_protocol_gpickles_for_raw_mode(args, base_result_folder)

    ensure_edge_maxsimplex_feature_files(args)
    validate_edge_maxsimplex_args(args)
    split_dir = os.path.join(base_result_folder, 'splits')
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(".cache_graph", exist_ok=True)
    os.makedirs(args.rw_cache_dir, exist_ok=True)
    pos_ratio = parse_pos_ratio(args.pos_ratio)

    tr_feat, tr_lbl, tr_trip, tr_edges, tr_n = load_g_as_arrays(args.train)
    te_feat, te_lbl, te_trip, te_edges, te_n = load_g_as_arrays(args.test)
    original_topology_audit = {
        'train_nodes': int(tr_n),
        'test_nodes': int(te_n),
        'train_edges': int(len(tr_edges)),
        'test_edges': int(len(te_edges)),
        'train_triplet_hash': _controlled_array_hash(np.asarray([tuple(x) for x in tr_trip], dtype=np.int64)),
        'test_triplet_hash': _controlled_array_hash(np.asarray([tuple(x) for x in te_trip], dtype=np.int64)),
        'train_edge_hash': _controlled_array_hash(np.asarray(tr_edges, dtype=np.int64)),
        'test_edge_hash': _controlled_array_hash(np.asarray(te_edges, dtype=np.int64)),
    }
    tr_strength_feature = te_strength_feature = None
    tr_strength_edge_weight = te_strength_edge_weight = None
    strength_audit = {}
    if parse_bool(args.use_strength_feature) or parse_bool(args.use_strength_edge_weight):
        (
            tr_strength_feature,
            te_strength_feature,
            tr_strength_edge_weight,
            te_strength_edge_weight,
            strength_audit,
        ) = build_controlled_strength_inputs(
            args.dataset, tr_trip, te_trip, tr_edges, te_edges,
            event_order=args.protocol_event_order,
            data_root=args.protocol_data_root,
            train_context_fraction=args.train_strength_context_ratio,
            test_context_fraction=args.test_strength_context_ratio,
        )
    if parse_bool(args.use_strength_feature):
        tr_feat = np.concatenate([tr_feat, tr_strength_feature], axis=1).astype(np.float32)
        te_feat = np.concatenate([te_feat, te_strength_feature], axis=1).astype(np.float32)
    history_feature_dim = 0
    history_audit = {'enabled': False, 'mode': str(args.history_feature_mode)}
    if args.history_feature_mode != 'none':
        if not args.history_feature_npz:
            raise ValueError('--history-feature-npz is required when history features are enabled')
        with np.load(args.history_feature_npz, allow_pickle=False) as payload:
            history_train_candidates = np.asarray(payload['train_candidates'], dtype=np.int64)
            history_test_candidates = np.asarray(payload['test_candidates'], dtype=np.int64)
            expected_train = np.sort(
                np.asarray([tuple(map(int, row)) for row in tr_trip], dtype=np.int64), axis=1
            )
            expected_test = np.sort(
                np.asarray([tuple(map(int, row)) for row in te_trip], dtype=np.int64), axis=1
            )
            stored_train = np.asarray(payload['train_x'], dtype=np.float32)
            stored_test = np.asarray(payload['test_x'], dtype=np.float32)
            if stored_train.shape[0] != history_train_candidates.shape[0] or stored_test.shape[0] != history_test_candidates.shape[0]:
                raise ValueError('History feature row count mismatch')
            train_map = {tuple(row): vector for row, vector in zip(history_train_candidates, stored_train)}
            test_map = {tuple(row): vector for row, vector in zip(history_test_candidates, stored_test)}
            try:
                history_train = np.stack([train_map[tuple(row)] for row in expected_train])
                history_test = np.stack([test_map[tuple(row)] for row in expected_test])
            except KeyError as error:
                raise ValueError(f'History candidate alignment mismatch: {error}') from error
            if history_train.shape[1] != history_test.shape[1] or history_train.shape[1] <= 0:
                raise ValueError('Invalid history feature dimensions')
            if not np.isfinite(history_train).all() or not np.isfinite(history_test).all():
                raise ValueError('Non-finite history feature input')
            history_feature_dim = int(history_train.shape[1])
            feature_names = [str(value) for value in payload['feature_names'].tolist()]
        tr_feat = np.concatenate([tr_feat, history_train], axis=1).astype(np.float32)
        te_feat = np.concatenate([te_feat, history_test], axis=1).astype(np.float32)
        history_audit = {
            'enabled': True,
            'mode': str(args.history_feature_mode),
            'feature_dim': history_feature_dim,
            'feature_names': feature_names,
            'npz': str(Path(args.history_feature_npz).resolve()),
            'npz_md5': _file_hash(args.history_feature_npz),
        }
    controlled_audit = {
        'dataset': args.dataset,
        'protocol_event_order': args.protocol_event_order,
        'protocol_data_root': args.protocol_data_root,
        'use_strength_feature': bool(parse_bool(args.use_strength_feature)),
        'use_strength_edge_weight': bool(parse_bool(args.use_strength_edge_weight)),
        'use_temporal_score_prior': bool(parse_bool(args.use_temporal_score_prior)),
        'temporal_prior_kind': str(args.temporal_prior_kind),
        'temporal_activity_alpha': float(args.temporal_activity_alpha),
        'validation_select_temporal_alpha': bool(parse_bool(args.validation_select_temporal_alpha)),
        'use_edge_maxsimplex_features': bool(parse_bool(args.use_edge_maxsimplex_features)),
        'original_topology': original_topology_audit,
        'weighted_topology_assertions': {
            'node_set_unchanged': True,
            'edge_set_unchanged': True,
        },
        'strength': strength_audit,
        'history_features': history_audit,
        'original_gpickle_md5': {
            'train': _file_hash(args.train),
            'test': _file_hash(args.test),
        },
        'rw_reference_cache_md5': (
            _file_hash(args.rw_reference_cache)
            if args.rw_reference_cache and os.path.isfile(args.rw_reference_cache)
            else None
        ),
    }
    with open(os.path.join(base_result_folder, 'controlled_audit.json'), 'w', encoding='utf-8') as handle:
        json.dump(controlled_audit, handle, indent=2)
    supervision_mask = torch.ones(tr_n, dtype=torch.bool)
    if args.supervision_scope == 'open_candidates':
        train_open_path = _resolve_protocol_triangle_path(
            args, args.dataset, 'train', 'openTriangles'
        )
        train_open = {
            tuple(sorted(map(int, triangle)))
            for triangle in load_target_triplets(train_open_path)
        }
        train_future_positive = {
            tuple(sorted(map(int, triangle)))
            for triangle in load_target_triplets(args.priority_nodes)
        }
        supervision_np = np.asarray(
            [
                tuple(sorted(map(int, triangle))) in train_open
                for triangle in tr_trip
            ],
            dtype=bool,
        )
        tr_lbl = np.asarray(
            [
                int(tuple(sorted(map(int, triangle))) in train_future_positive)
                for triangle in tr_trip
            ],
            dtype=np.int64,
        )
        supervision_mask = torch.from_numpy(supervision_np)
        print(
            "Open-candidate supervision: "
            f"eligible={int(supervision_np.sum())} "
            f"positives={int(tr_lbl[supervision_np].sum())}"
        )
    print(f"train: nodes={tr_n}, edges={tr_edges.shape[0]}, feature_dim={tr_feat.shape[1]}")
    print(f"test: nodes={te_n}, edges={te_edges.shape[0]}, feature_dim={te_feat.shape[1]}")
    max_feat_dim = 0
    tr_max_feat = None
    te_max_feat = None
    edge_max_stats = {
        'max_feat_dim': 0,
        'train_max_feat_nonzero_rate': np.nan,
        'test_max_feat_nonzero_rate': np.nan,
        'train_max_feat_all_zero_rows': 0,
        'test_max_feat_all_zero_rows': 0,
    }
    if (
        args.edge_maxsimplex_fusion != 'none'
        or str(args.temporal_prior_kind) in ('maxsimplex_rank', 'activity_maxsimplex')
    ):
        if not args.train_edge_maxsimplex_txt or not args.test_edge_maxsimplex_txt:
            raise ValueError(
                'maxsimplex_rank prior requires explicit train/test maximal-simplex feature files'
            )
        tr_max_feat, te_max_feat, max_feat_dim, edge_max_stats = load_aligned_edge_maxsimplex_matrices(tr_trip, te_trip, args)
        tr_max_feat, te_max_feat = apply_edge_maxsimplex_ablation(tr_max_feat, te_max_feat, args)
        edge_max_stats.update(maxsimplex_matrix_stats(tr_max_feat, 'train'))
        edge_max_stats.update(maxsimplex_matrix_stats(te_max_feat, 'test'))
        edge_max_stats['edge_maxsimplex_fusion'] = args.edge_maxsimplex_fusion
        edge_max_stats['edge_maxsimplex_ablation'] = args.edge_maxsimplex_ablation

    triplet_network = TripletNetworkFast(te_trip)
    target_triplets = load_target_triplets(args.target)
    target_nodes = set()
    for triplet in target_triplets:
        node_id = triplet_network.find_node(triplet)
        if node_id is not None:
            target_nodes.add(node_id)
    target_nodes = sorted(target_nodes)
    print(f"target triplets={len(target_triplets)}, target nodes found={len(target_nodes)}")
    if sample_fields.get('sampled_dataset', False):
        unique_target_triplets = {tuple(sorted(map(int, tri))) for tri in target_triplets}
        if len(target_nodes) != len(unique_target_triplets):
            raise RuntimeError(
                "sampled dataset target nodes found incomplete: "
                f"target_nodes_found={len(target_nodes)}, unique_target_triplets={len(unique_target_triplets)}"
            )
    if te_max_feat is not None:
        validate_open_candidate_maxsimplex_values(
            te_trip,
            target_nodes,
            te_max_feat,
            args.test_edge_maxsimplex_txt,
            "tri_all_edges_shared_count",
        )
    target_mask = create_target_mask_from_nodes(target_nodes, te_n)
    aligned_target_triplets = np.asarray(
        [tuple(map(int, te_trip[index])) for index in target_nodes], dtype=np.int64
    )
    candidate_alignment_hash = _controlled_array_hash(aligned_target_triplets)
    temporal_activity_scores = None
    temporal_activity_train_scores = None
    prior_kind = str(getattr(args, 'temporal_prior_kind', 'recent_activity'))
    prior_model_audit = {}
    if parse_bool(args.use_temporal_score_prior):
        protocol_dataset = args.dataset or _infer_protocol_dataset_name(args)
        if prior_kind == 'candidate_law':
            from candidate_law_prior import build_or_load_candidate_law_prior
            candidate_cache = (
                args.candidate_law_cache_dir
                or os.path.join(args.rw_cache_dir, 'candidate_law')
            )
            prior_device = resolve_device(args.device)
            temporal_activity_train_scores, temporal_activity_scores, prior_model_audit = (
                build_or_load_candidate_law_prior(
                    protocol_dataset,
                    tr_trip,
                    tr_lbl,
                    supervision_mask.cpu().numpy(),
                    aligned_target_triplets,
                    args.protocol_data_root or Path(__file__).resolve().parent,
                    args.protocol_event_order,
                    args.val_seed if args.val_seed is not None else 2026,
                    candidate_cache,
                    device=prior_device,
                    seed=41,
                )
            )
            print(
                'Candidate-law prior: '
                f"validation_AP={prior_model_audit.get('validation_ap', np.nan):.6f}, "
                f"feature_dim={prior_model_audit.get('feature_dim', 0)}, "
                f"targets={len(temporal_activity_scores)}"
            )
        elif prior_kind == 'recent_activity':
            target_indices_np = torch.where(target_mask)[0].cpu().numpy()
            temporal_activity_scores = build_temporal_activity_aligned_scores(
                protocol_dataset, 'test', te_trip, target_indices_np, args
            )
            eligible_train_indices = torch.where(supervision_mask)[0].cpu().numpy()
            eligible_train_scores = build_temporal_activity_aligned_scores(
                protocol_dataset, 'train', tr_trip, eligible_train_indices, args
            )
            temporal_activity_train_scores = np.full(tr_n, np.nan, dtype=np.float64)
            temporal_activity_train_scores[eligible_train_indices] = eligible_train_scores
        elif prior_kind == 'maxsimplex_rank':
            if not args.train_edge_maxsimplex_txt or not args.test_edge_maxsimplex_txt:
                raise ValueError(
                    'maxsimplex_rank prior requires --train_edge_maxsimplex_txt '
                    'and --test_edge_maxsimplex_txt'
                )
            target_indices_np = torch.where(target_mask)[0].cpu().numpy()
            temporal_activity_scores = build_maxsimplex_rank_aligned_scores(
                te_trip,
                target_indices_np,
                args.test_edge_maxsimplex_txt,
                args.temporal_prior_maxsimplex_column,
            )
            eligible_train_indices = torch.where(supervision_mask)[0].cpu().numpy()
            eligible_train_scores = build_maxsimplex_rank_aligned_scores(
                tr_trip,
                eligible_train_indices,
                args.train_edge_maxsimplex_txt,
                args.temporal_prior_maxsimplex_column,
            )
            temporal_activity_train_scores = np.full(tr_n, np.nan, dtype=np.float64)
            temporal_activity_train_scores[eligible_train_indices] = eligible_train_scores
        elif prior_kind == 'activity_maxsimplex':
            if not args.train_edge_maxsimplex_txt or not args.test_edge_maxsimplex_txt:
                raise ValueError(
                    'activity_maxsimplex prior requires explicit train/test '
                    'maximal-simplex feature files'
                )
            if float(args.temporal_activity_alpha) <= 0.0:
                raise ValueError(
                    'activity_maxsimplex prior requires --temporal_activity_alpha > 0'
                )
            maxsimplex_weight = float(args.temporal_prior_maxsimplex_weight)
            if not 0.0 <= maxsimplex_weight <= 1.0:
                raise ValueError('--temporal_prior_maxsimplex_weight must be in [0, 1].')

            target_indices_np = torch.where(target_mask)[0].cpu().numpy()
            activity_target = build_temporal_activity_aligned_scores(
                protocol_dataset, 'test', te_trip, target_indices_np, args
            )
            maxsimplex_target = build_maxsimplex_rank_aligned_scores(
                te_trip,
                target_indices_np,
                args.test_edge_maxsimplex_txt,
                args.temporal_prior_maxsimplex_column,
            )
            eligible_train_indices = torch.where(supervision_mask)[0].cpu().numpy()
            activity_train = build_temporal_activity_aligned_scores(
                protocol_dataset, 'train', tr_trip, eligible_train_indices, args
            )
            maxsimplex_train = build_maxsimplex_rank_aligned_scores(
                tr_trip,
                eligible_train_indices,
                args.train_edge_maxsimplex_txt,
                args.temporal_prior_maxsimplex_column,
            )
            temporal_activity_scores = (
                (1.0 - maxsimplex_weight) * activity_target
                + maxsimplex_weight * maxsimplex_target
            )
            temporal_activity_train_scores = np.full(tr_n, np.nan, dtype=np.float64)
            temporal_activity_train_scores[eligible_train_indices] = (
                (1.0 - maxsimplex_weight) * activity_train
                + maxsimplex_weight * maxsimplex_train
            )
            prior_model_audit = {
                'combination': 'weighted_arithmetic_mean',
                'activity_weight': float(1.0 - maxsimplex_weight),
                'maxsimplex_weight': float(maxsimplex_weight),
                'maxsimplex_column': str(args.temporal_prior_maxsimplex_column),
            }
            print(
                'Combined activity/maximal-simplex prior: '
                f'activity_weight={1.0 - maxsimplex_weight:.3f}, '
                f'maxsimplex_weight={maxsimplex_weight:.3f}'
            )
        else:
            raise ValueError(f'Unsupported temporal_prior_kind={prior_kind}')
    activity_score_audit = {
        'candidate_alignment_hash': candidate_alignment_hash,
        'num_candidates': int(len(target_nodes)),
        'enabled': bool(temporal_activity_scores is not None),
        'prior_kind': prior_kind,
        'context_ratio_used_for_final_score': float(getattr(args, 'temporal_activity_context_ratio', 0.8)),
        'post_cutoff_raw_events_used': bool(float(getattr(args, 'temporal_activity_context_ratio', 0.8)) > 0.8),
        'measure': str(getattr(args, 'temporal_activity_measure', 'count')),
        'prior_model': prior_model_audit,
    }
    if temporal_activity_scores is not None:
        activity_array = np.asarray(temporal_activity_scores, dtype=np.float64)
        activity_score_audit.update({
            'score_hash': _controlled_array_hash(activity_array),
            'min': float(activity_array.min()) if activity_array.size else np.nan,
            'max': float(activity_array.max()) if activity_array.size else np.nan,
            'mean': float(activity_array.mean()) if activity_array.size else np.nan,
            'std': float(activity_array.std()) if activity_array.size else np.nan,
        })
    with open(os.path.join(base_result_folder, 'activity_score_audit.json'), 'w', encoding='utf-8') as handle:
        json.dump(activity_score_audit, handle, indent=2)

    pos_label = args.pos_label
    if pos_label is None:
        target_labels = te_lbl[target_mask.cpu().numpy()]
        class_counts = np.bincount(target_labels) if target_labels.size > 0 else np.array([0, 0])
        if class_counts.size == 1:
            class_counts = np.array([class_counts[0], 0])
        pos_label = 0 if class_counts[0] < class_counts[1] else 1
    print(f"pos_label={pos_label}")
    if args.edge_maxsimplex_fusion == 'prior_logit' and int(pos_label) != 1:
        raise ValueError("Current prior_logit implementation assumes positive class is label 1.")

    priority_mask = load_priority_mask_for_train(args, tr_n)
    priority_sampling_mask = priority_mask
    selector_val_mask = None
    selector_val_y = None
    selector_info = None
    selector_val_info = {'selector_val': args.selector_val}
    if args.selector_val == 'priority_holdout':
        priority_holdout_train_mask, selector_val_mask, selector_val_y, selector_info, selector_val_info = build_priority_holdout_selector(
            priority_mask,
            tr_n,
            val_seed=args.val_seed,
            priority_val_frac=args.priority_val_frac,
            selector_val_pos_rate=args.selector_val_pos_rate,
            split_dir=split_dir,
            reuse_splits=reuse_splits
        )
        priority_sampling_mask = priority_holdout_train_mask if selector_removes_priority_from_training else priority_mask
        selector_val_info['selector_val'] = args.selector_val
        selector_val_info['selector_removes_priority_from_training'] = bool(selector_removes_priority_from_training)
        print(
            "Selector validation priority_holdout: "
            f"priority_total={selector_val_info['priority_total']}, "
            f"priority_train={selector_val_info['priority_train']}, "
            f"priority_val={selector_val_info['priority_val']}, "
            f"selector_neg={selector_val_info['selector_val_negatives']}, "
            f"selector_nodes={selector_val_info['selector_val_nodes']}, "
            f"selector_pos_rate={selector_val_info['selector_val_pos_rate']:.4f}"
        )
        if not selector_val_info.get('selector_val_valid', False):
            print("WARNING: selector validation invalid; skip selector-based selection.")
        if selector_info is not None:
            np.save(os.path.join(split_dir, 'selector_val_indices.npy'), np.asarray(selector_info.get('indices', []), dtype=np.int64))
            np.save(os.path.join(split_dir, 'selector_val_labels.npy'), np.asarray(selector_info.get('labels', []), dtype=np.int64))
    with open(os.path.join(base_result_folder, 'selector_val_info.json'), 'w', encoding='utf-8') as f:
        json.dump(selector_val_info, f, indent=2)
    with open(os.path.join(base_result_folder, 'edge_maxsimplex_config.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'use_edge_maxsimplex_features': parse_bool(args.use_edge_maxsimplex_features),
            'edge_maxsimplex_fusion': args.edge_maxsimplex_fusion,
            'edge_maxsimplex_ablation': args.edge_maxsimplex_ablation,
            'train_edge_maxsimplex_txt': args.train_edge_maxsimplex_txt,
            'test_edge_maxsimplex_txt': args.test_edge_maxsimplex_txt,
            'edge_maxsimplex_raw_dir': args.edge_maxsimplex_raw_dir,
            'train_edge_maxsimplex_context_ratio': args.train_edge_maxsimplex_context_ratio,
            'test_edge_maxsimplex_context_ratio': args.test_edge_maxsimplex_context_ratio,
            'prior_alpha': args.prior_alpha,
            'prior_hidden_dim': args.prior_hidden_dim,
            'prior_dropout': args.prior_dropout,
            'maxsimplex_shuffle_seed': args.maxsimplex_shuffle_seed,
            **edge_max_stats,
        }, f, indent=2)

    device = resolve_device(args.device)
    edge_index_train = torch.from_numpy(tr_edges.T.copy()).long().to(device)
    edge_index_test = torch.from_numpy(te_edges.T.copy()).long().to(device)
    if parse_bool(args.use_strength_edge_weight):
        train_edge_attr = torch.from_numpy(tr_strength_edge_weight).float().to(device)
        test_edge_attr = torch.from_numpy(te_strength_edge_weight).float().to(device)
        edge_index_train, train_edge_attr = add_self_loops(
            edge_index_train, edge_attr=train_edge_attr, fill_value=1.0, num_nodes=tr_n
        )
        edge_index_test, test_edge_attr = add_self_loops(
            edge_index_test, edge_attr=test_edge_attr, fill_value=1.0, num_nodes=te_n
        )
    else:
        edge_index_train, _ = add_self_loops(edge_index_train)
        edge_index_test, _ = add_self_loops(edge_index_test)
        train_edge_attr = None
        test_edge_attr = None

    model_cfg = {
        'num_layers': args.model_num_layers,
        'hidden_channels': args.model_hidden_channels,
        'heads': args.model_heads,
        'dropout': args.model_dropout,
        'device': args.device,
        'model_type': args.model_type,
        'dual_self_hidden_dim': args.dual_self_hidden_dim,
        'dual_gate_hidden_dim': args.dual_gate_hidden_dim,
        'dual_dropout': args.dual_dropout,
        'use_batchnorm': args.model_use_batchnorm,
        'use_layernorm': args.model_use_layernorm,
        'residual': args.model_residual,
        'use_mlp_head': args.model_use_mlp_head,
        'mlp_layers': args.model_mlp_layers,
        'activation': args.model_activation,
        'weight_init': args.model_weight_init,
        'pairwise_aux_loss': parse_bool(args.pairwise_aux_loss),
        'pairwise_aux_weight': args.pairwise_aux_weight,
        'pairwise_aux_margin': args.pairwise_aux_margin,
        'pairwise_aux_max_pairs': args.pairwise_aux_max_pairs,
        'supervision_mask': supervision_mask,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'amp': bool(args.amp),
        'epochs': args.epochs,
        'no_validation': parse_bool(args.no_validation),
        'eval_interval': args.test_eval_interval if args.test_eval_interval is not None else args.eval_interval,
        'evaluate_test_during_training': bool(args.evaluate_test_during_training),
        'batch_size': args.batch_size,
        'effective_batch_size': args.effective_batch_size,
        'model_selection_by': args.model_selection_by,
        'graph_scope': args.graph_scope,
        'local_num_neighbors': args.local_num_neighbors,
        'local_num_hops': args.local_num_hops,
        'local_eval_batch_size': args.local_eval_batch_size,
        'local_train_batch_size': args.local_train_batch_size,
        'local_sampler': args.local_sampler,
        'local_fallback_sample': parse_bool(args.local_fallback_sample),
        'edge_maxsimplex_fusion': args.edge_maxsimplex_fusion,
        'edge_maxsimplex_ablation': args.edge_maxsimplex_ablation,
        'prior_alpha': args.prior_alpha,
        'prior_hidden_dim': args.prior_hidden_dim,
        'prior_dropout': args.prior_dropout,
        'max_feat_dim': max_feat_dim,
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'pretrained_threshold': args.pretrained_threshold,
        'temporal_activity_scores': temporal_activity_scores,
        'temporal_activity_train_scores': temporal_activity_train_scores,
        'temporal_activity_alpha': args.temporal_activity_alpha,
        'validation_select_temporal_alpha': parse_bool(args.validation_select_temporal_alpha),
        'temporal_alpha_grid': args.temporal_alpha_grid,
        'temporal_prior_kind': prior_kind,
        'temporal_prior_maxsimplex_weight': args.temporal_prior_maxsimplex_weight,
        'temporal_prior_maxsimplex_column': args.temporal_prior_maxsimplex_column,
        'candidate_alignment_hash': candidate_alignment_hash,
        'activity_score_audit': activity_score_audit,
        'use_strength_feature': bool(parse_bool(args.use_strength_feature)),
        'use_strength_edge_weight': bool(parse_bool(args.use_strength_edge_weight)),
        'use_temporal_score_prior': bool(parse_bool(args.use_temporal_score_prior)),
        'history_feature_dim': history_feature_dim,
        'history_feature_mode': args.history_feature_mode,
        'history_residual_scale': args.history_residual_scale,
        'history_pretrained_checkpoint': args.history_pretrained_checkpoint,
    }

    rw_param_sweep = parse_bool(args.rw_param_sweep)
    rw_seeds = parse_seeds(args.rw_seeds) if args.rw_seeds is not None else [int(args.rw_seed)]
    model_seed_arg = args.model_seeds if args.model_seeds is not None else args.seeds
    model_seeds = parse_seeds(model_seed_arg)
    if rw_param_sweep:
        rw_sweep_configs = build_rw_param_sweep_configs(args)
        print(
            "RW parameter sweep enabled under fixed rw_seed(s): "
            f"rw_seeds={rw_seeds}, model_seeds={model_seeds}"
        )
    else:
        rw_sweep_configs = [{
            'name': infer_single_rw_config_name(args),
            'rw_walks_per_node': int(args.rw_walks_per_node),
            'rw_epochs': int(args.rw_epochs),
            'rw_embedding_dim': int(args.rw_embedding_dim),
        }]
    save_checkpoint_variants = False
    all_results = []

    for rw_cfg in rw_sweep_configs:
        apply_rw_sweep_config(args, rw_cfg)
        cfg_name = rw_cfg['name']
        cfg_folder = os.path.join(base_result_folder, cfg_name) if rw_param_sweep else base_result_folder
        os.makedirs(cfg_folder, exist_ok=True)
        print(
            f"\n=== RW config: {cfg_name} impl={args.rw_impl} "
            f"(walks_per_node={args.rw_walks_per_node}, "
            f"rw_epochs={args.rw_epochs}, embedding_dim={args.rw_embedding_dim}, "
            f"negative_samples={args.rw_negative_samples}, restart_prob={args.rw_restart_prob}, "
            f"context_hops={args.rw_context_hops}, "
            f"hybrid_stoch_dim={args.rw_hybrid_stoch_dim}, "
            f"hybrid_ppmi_dim={args.rw_hybrid_ppmi_dim}, "
            f"hybrid_ppmi_hops={args.rw_hybrid_ppmi_hops}, "
            f"hybrid_l2norm={parse_bool(args.rw_hybrid_l2norm)}) ==="
        )
        for rw_seed in rw_seeds:
            rw_folder = os.path.join(cfg_folder, f"rw_seed_{rw_seed}")
            os.makedirs(rw_folder, exist_ok=True)
            rw_started = time.perf_counter()
            if args.disable_rw:
                tr_node_emb = np.zeros((tr_n, int(args.rw_embedding_dim)), dtype=np.float32)
                te_node_emb = np.zeros((te_n, int(args.rw_embedding_dim)), dtype=np.float32)
                print('RW disabled: using zero channel; no random-walk construction called.')
            else:
                tr_node_emb, te_node_emb = generate_or_load_rw_embeddings(
                    args, edge_index_train, tr_n, edge_index_test, te_n, rw_seed, device
                )
            rw_seconds = time.perf_counter() - rw_started
            if args.edge_maxsimplex_fusion == 'concat':
                gnn_train_feat = np.concatenate([tr_feat, tr_max_feat], axis=1)
                gnn_test_feat = np.concatenate([te_feat, te_max_feat], axis=1)
                base_train = np.concatenate([gnn_train_feat, tr_node_emb], axis=1)
                base_test = np.concatenate([gnn_test_feat, te_node_emb], axis=1)
            else:
                base_train = np.concatenate([tr_feat, tr_node_emb], axis=1)
                base_test = np.concatenate([te_feat, te_node_emb], axis=1)
            if args.disable_rw and args.history_feature_mode == 'history_base_structural_residual':
                X_train = base_train.astype(np.float32)
                X_test = base_test.astype(np.float32)
            else:
                base_scaler = StandardScaler().fit(base_train)
                X_train = base_scaler.transform(base_train).astype(np.float32)
                X_test = base_scaler.transform(base_test).astype(np.float32)
            data_train = Data(x=torch.from_numpy(X_train), edge_index=edge_index_train.cpu(), y=torch.from_numpy(tr_lbl).long())
            data_test = Data(x=torch.from_numpy(X_test), edge_index=edge_index_test.cpu(), y=torch.from_numpy(te_lbl).long())
            data_train.edge_weight = train_edge_attr.cpu() if train_edge_attr is not None else None
            data_test.edge_weight = test_edge_attr.cpu() if test_edge_attr is not None else None
            if args.edge_maxsimplex_fusion == 'prior_logit':
                max_scaler = StandardScaler().fit(tr_max_feat)
                Z_train = max_scaler.transform(tr_max_feat).astype(np.float32)
                Z_test = max_scaler.transform(te_max_feat).astype(np.float32)
                data_train.max_x = torch.from_numpy(Z_train)
                data_test.max_x = torch.from_numpy(Z_test)
            else:
                data_train.max_x = None
                data_test.max_x = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            rw_results = []
            for model_seed in model_seeds:
                print(f"\n=== rw_config={cfg_name}, rw_seed={rw_seed}, model_seed={model_seed} ===")
                try:
                    res = run_experiment(
                        seed=model_seed,
                        base_result_folder=rw_folder,
                        data_train=data_train,
                        data_test=data_test,
                        triplet_network=triplet_network,
                        target_mask=target_mask,
                        target_nodes=target_nodes,
                        pos_ratio=pos_ratio,
                        pos_label=pos_label,
                        priority_mask=priority_sampling_mask,
                        priority_frac=args.priority_frac,
                        priority_mode=args.priority_mode,
                        model_cfg=model_cfg,
                        rw_seed=rw_seed,
                        val_seed=args.val_seed,
                        save_checkpoint_variants=save_checkpoint_variants,
                        fixed_checkpoint_epoch=args.fixed_checkpoint_epoch,
                        selector_val_mask=selector_val_mask,
                        selector_val_y=selector_val_y,
                        selector_info=selector_info,
                        deterministic=deterministic,
                        reuse_splits=reuse_splits,
                        split_dir=split_dir
                    )
                except RuntimeError as exc:
                    if 'out of memory' in str(exc).lower() and str(args.dataset).lower() in {'tags-ask-ubuntu', 'tags-math-sx'}:
                        print(
                            "CUDA OOM fallback suggestions: "
                            "1) local_train_batch_size=32, local_eval_batch_size=32; "
                            "2) local_num_neighbors=30; "
                            "3) model_hidden_channels=64, model_heads=2; "
                            "4) dual_self_hidden_dim=64, dual_gate_hidden_dim=32; "
                            "5) rw_graph_scope=pruned, rw_prune_fanout=50."
                        )
                    raise
                if res:
                    res['rw_seconds'] = float(rw_seconds)

                    prediction_path = os.path.join(rw_folder, f'target_scores_seed_{model_seed}.npz')
                    target_triplets_for_save = np.asarray(
                        [triplet_network.get_triplet_by_node(int(node_id)) for node_id in target_nodes],
                        dtype=np.int64,
                    )
                    np.savez_compressed(
                        prediction_path,
                        y_true_target=np.asarray(res.get('y_true_target', []), dtype=np.int64),
                        y_score_target=np.asarray(res.get('y_score_target', []), dtype=np.float64),
                        y_score_target_unfused=np.asarray(res.get('y_score_target_unfused', []), dtype=np.float64),
                        y_true_val=np.asarray(res.get('y_true_val', []), dtype=np.int64),
                        y_score_val=np.asarray(res.get('y_score_val', []), dtype=np.float64),
                        val_triplets=np.asarray(res.get('val_triplets', []), dtype=np.int64),
                        target_node_ids=np.asarray(target_nodes, dtype=np.int64),
                        target_triplets=target_triplets_for_save,
                        candidate_alignment_hash=np.asarray(
                            str(res.get('candidate_alignment_hash', ''))
                        ),
                    )
                    Path(os.path.join(rw_folder, 'target_scores.npz')).write_bytes(
                        Path(prediction_path).read_bytes()
                    )
                    res['rw_config'] = cfg_name
                    res['rw_impl'] = args.rw_impl
                    res['rw_walks_per_node'] = int(args.rw_walks_per_node)
                    res['rw_epochs'] = int(args.rw_epochs)
                    res['rw_embedding_dim'] = int(args.rw_embedding_dim)
                    res['rw_negative_samples'] = int(args.rw_negative_samples)
                    res['rw_negative_power'] = float(args.rw_negative_power)
                    res['rw_restart_prob'] = float(args.rw_restart_prob)
                    res['rw_context_hops'] = int(args.rw_context_hops)
                    res['rw_graph_scope'] = args.rw_graph_scope
                    res['rw_prune_fanout'] = int(args.rw_prune_fanout)
                    res.update(sample_fields)
                    res['rw_hybrid_stoch_dim'] = int(args.rw_hybrid_stoch_dim)
                    res['rw_hybrid_ppmi_dim'] = int(args.rw_hybrid_ppmi_dim)
                    res['rw_hybrid_ppmi_hops'] = int(args.rw_hybrid_ppmi_hops)
                    res['rw_hybrid_l2norm'] = parse_bool(args.rw_hybrid_l2norm)
                    rw_results.append(res)
                    all_results.append(res)

    if all_results:
        write_seed_results(base_result_folder, all_results)
        write_minimal_summary(base_result_folder, all_results, args)
    print("\n" + "=" * 50)
    print("Run complete:", base_result_folder)
    print("=" * 50)
