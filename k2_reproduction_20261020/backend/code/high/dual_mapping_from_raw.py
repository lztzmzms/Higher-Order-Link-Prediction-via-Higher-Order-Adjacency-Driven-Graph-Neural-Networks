#!/usr/bin/env python
"""Dual-Mapping triangle closure prediction from raw timestamped simplices.

This script intentionally does not read or write gpickle files.  It rebuilds
the temporal closure task directly from the three raw files:

    nverts.txt, simplices.txt, times.txt

Protocol:
    0-20 context -> 20-40 future closure is a supervised training task.
    0-40 context -> 40-60 future closure is a supervised training task.
    0-60 context -> 60-80 future closure is the validation task.
    0-80 context -> 80-100 future closure is the final test task.

Dual-Mapping:
    open triangle -> derived graph node
    shared lower face (edge) -> derived graph edge
    local topological features + GraphSAGE message passing -> closure score
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from interaction_strength import (
    build_boundary_strength,
    candidate_strength_matrix,
    dataset_strength_statistics,
    strength_value,
    write_rows,
)

try:
    from scipy import sparse  # noqa: F401  # kept to fail early if scipy is absent
except Exception:  # pragma: no cover - dependency check only
    sparse = None

try:
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import rankdata
except Exception as exc:  # pragma: no cover - dependency check only
    raise RuntimeError("scikit-learn is required for AP evaluation and scaling.") from exc


Triangle = Tuple[int, int, int]
Edge = Tuple[int, int]


# ---------------------------------------------------------------------------
# 1. argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-Mapping GNN for raw timestamped simplex triangle closure."
    )
    parser.add_argument("--nverts", default="email-Enron-nverts.txt", help="Path to nverts txt file.")
    parser.add_argument("--simplices", default="email-Enron-simplices.txt", help="Path to flattened simplices txt file.")
    parser.add_argument("--times", default="email-Enron-times.txt", help="Path to timestamps txt file.")
    parser.add_argument(
        "--event-order",
        choices=["raw", "timestamp"],
        default="raw",
        help="Event order for temporal windows; raw reproduces the historical randomline split.",
    )
    parser.add_argument("--task_dim", type=int, default=2, help="Only task_dim=2 is supported.")
    parser.add_argument("--protocol", choices=["rolling", "multi_window"], default="multi_window")
    parser.add_argument("--split", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument(
        "--model",
        choices=["heuristic_rw_residual_gnn", "heuristic_residual_gnn", "gnn"],
        default="heuristic_rw_residual_gnn",
    )
    parser.add_argument("--encoder", choices=["sage", "mlp", "gat"], default="sage")
    parser.add_argument("--global_encoder", default="gnn", help="Accepted for compatibility; GNN is used.")
    parser.add_argument("--use_random_walk", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--rw_mode", choices=["aligned_union"], default="aligned_union")
    parser.add_argument("--rw_dim", type=int, default=64)
    parser.add_argument("--rw_walk_length", type=int, default=20)
    parser.add_argument("--rw_walks_per_node", type=int, default=20)
    parser.add_argument("--rw_window_size", type=int, default=5)
    parser.add_argument("--rw_epochs", type=int, default=5)
    parser.add_argument("--rw_negative_samples", type=int, default=5)
    parser.add_argument(
        "--rw_max_pairs",
        type=int,
        default=20_000_000,
        help="Hard cap on skip-gram positive pairs; bounds CPU/GPU memory on large datasets.",
    )
    parser.add_argument("--embedding_seed", type=int, default=42)
    parser.add_argument("--priority_walk_multiplier", type=int, default=5)
    parser.add_argument(
        "--priority_feature_mode",
        choices=["train_only", "train_val_for_final"],
        default="train_val_for_final",
    )
    parser.add_argument("--use_rw_struct_features", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--rw_steps", type=int, default=4)
    parser.add_argument("--rw_num_walks", type=int, default=200, help="Kept for CLI compatibility; exact sparse RW stats are used.")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--feature_proj_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--loss", choices=["bce", "bce_ranking", "pu_ranking"], default="pu_ranking")
    parser.add_argument("--pos_weight", default="1.0")
    parser.add_argument("--unlabeled_weight", type=float, default=0.03)
    parser.add_argument("--lambda_rank", type=float, default=0.05)
    parser.add_argument("--hard_negative_top", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage2_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--residual_lr", type=float, default=0.0003)
    parser.add_argument("--max_residual_alpha", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--final_retrain", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--seeds", default="48,53,56")
    parser.add_argument("--results", default="results_dual_mapping_email_enron_priority_rw")
    parser.add_argument("--processed_dir", default="processed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--warn_open_triangles", type=int, default=2_000_000)
    parser.add_argument("--warn_dual_edges", type=int, default=5_000_000)
    parser.add_argument(
        "--dual-edge-weight",
        choices=["binary", "boundary-count", "boundary-log", "boundary-decay"],
        default="binary",
        help="Representation-level derived-graph weighting; binary preserves the original model.",
    )
    parser.add_argument(
        "--dual-shared-face-reduce", choices=["sum", "mean"], default="sum",
        help="Combine multiple shared lower faces (mean is an ablation).",
    )
    parser.add_argument(
        "--information-mode",
        choices=["legacy", "topology", "strength", "topology+strength", "full"],
        default="legacy",
        help="A/B/C/D information-source ablation; legacy preserves the pre-revision feature path.",
    )
    parser.add_argument(
        "--strength-decay-lambda", type=float, default=0.0,
        help="Decay rate in timestamp units; <=0 chooses a unit-safe 15%-span half-life.",
    )
    parser.add_argument("--node-activity-recent-fraction", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_seeds(seed_text: str) -> List[int]:
    seeds = []
    for chunk in seed_text.split(","):
        chunk = chunk.strip()
        if chunk:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    return seeds


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    fallback = Path("dataset") / "raw_data" / "email-Enron" / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Input file not found: {path_text}")


# ---------------------------------------------------------------------------
# 2. load raw timestamped simplices
# ---------------------------------------------------------------------------


def read_int_lines(path: Path) -> List[int]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                values.append(int(text))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} is not an integer: {text!r}") from exc
    return values


def load_timestamped_simplices(nverts_path, simplices_path, times_path, event_order="raw"):
    nverts_path = resolve_input_path(str(nverts_path))
    simplices_path = resolve_input_path(str(simplices_path))
    times_path = resolve_input_path(str(times_path))

    nverts = read_int_lines(nverts_path)
    flat_vertices = read_int_lines(simplices_path)
    times = read_int_lines(times_path)

    assert len(nverts) == len(times), "nverts and times must be one-to-one."
    assert sum(nverts) == len(flat_vertices), "simplices length must equal sum(nverts)."

    simplices = []
    offset = 0
    duplicate_vertex_simplices = 0
    singleton_simplices = 0
    dropped_empty_simplices = 0
    for i, nv in enumerate(nverts):
        raw_verts = flat_vertices[offset : offset + nv]
        offset += nv

        verts = tuple(sorted(set(raw_verts)))
        if len(verts) != len(raw_verts):
            duplicate_vertex_simplices += 1
        if len(verts) == 0:
            dropped_empty_simplices += 1
            continue
        if len(verts) == 1:
            singleton_simplices += 1

        simplices.append(
            {
                "id": i,
                "time": times[i],
                "vertices": verts,
                "dim": len(verts) - 1,
                "size": len(verts),
            }
        )

    if event_order == "timestamp":
        simplices.sort(key=lambda x: (x["time"], x["id"]))
    elif event_order != "raw":
        raise ValueError(f"Unsupported event order: {event_order}")
    stats = {
        "raw_rows": len(nverts),
        "kept_simplices": len(simplices),
        "duplicate_vertex_simplices": duplicate_vertex_simplices,
        "singleton_simplices": singleton_simplices,
        "dropped_empty_simplices": dropped_empty_simplices,
        "nverts_path": str(nverts_path),
        "simplices_path": str(simplices_path),
        "times_path": str(times_path),
    }
    return simplices, stats


# ---------------------------------------------------------------------------
# 3. temporal split
# ---------------------------------------------------------------------------


def temporal_split(simplices, split=(0.2, 0.4, 0.6, 0.8)):
    n = len(simplices)
    indices = [int(frac * n) for frac in split]
    contexts = {f"S{int(frac * 100)}": simplices[:idx] for frac, idx in zip(split, indices)}
    futures = {}
    prev_idx = 0
    prev_frac = 0.0
    for frac, idx in zip(split, indices):
        futures[f"future_{int(prev_frac * 100)}_{int(frac * 100)}"] = simplices[prev_idx:idx]
        prev_idx = idx
        prev_frac = frac
    futures[f"future_{int(prev_frac * 100)}_100"] = simplices[prev_idx:]
    contexts["S100"] = simplices
    return contexts, futures


# ---------------------------------------------------------------------------
# 4. build observed simplicial state
# ---------------------------------------------------------------------------


def build_observed_state(S_context):
    observed_edges: Set[Edge] = set()
    observed_triangles: Set[Triangle] = set()
    observed_simplices_by_size = defaultdict(set)

    for simplex in S_context:
        V = simplex["vertices"]

        # Downward closure: any observed higher simplex contains all lower faces.
        for r in range(1, len(V) + 1):
            for face in combinations(V, r):
                observed_simplices_by_size[r].add(tuple(sorted(face)))

        for e in combinations(V, 2):
            observed_edges.add(tuple(sorted(e)))

        for tri in combinations(V, 3):
            observed_triangles.add(tuple(sorted(tri)))

    return {
        "edges": observed_edges,
        "triangles": observed_triangles,
        "simplices_by_size": observed_simplices_by_size,
    }


def build_adjacency(edges: Iterable[Edge]) -> Dict[int, Set[int]]:
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def build_edge_triangle_support(triangles: Iterable[Triangle]) -> Dict[Edge, int]:
    support = defaultdict(int)
    for tri in triangles:
        a, b, c = tri
        support[tuple(sorted((a, b)))] += 1
        support[tuple(sorted((a, c)))] += 1
        support[tuple(sorted((b, c)))] += 1
    return support


def build_edge_upper_adjacency(observed_triangles: Iterable[Triangle]):
    edge_upper_neighbors = defaultdict(set)
    edge_upper_weight = defaultdict(int)
    for tri in observed_triangles:
        faces = triangle_faces(tri)
        for ei, ej in combinations(faces, 2):
            edge_upper_neighbors[ei].add(ej)
            edge_upper_neighbors[ej].add(ei)
            edge_upper_weight[(ei, ej)] += 1
            edge_upper_weight[(ej, ei)] += 1
    return edge_upper_neighbors, edge_upper_weight


# ---------------------------------------------------------------------------
# 5. enumerate open triangles
# ---------------------------------------------------------------------------


def enumerate_open_triangles(observed_edges, observed_triangles):
    adj = build_adjacency(observed_edges)
    open_triangles: List[Triangle] = []

    for u in sorted(adj):
        for v in sorted(x for x in adj[u] if x > u):
            common = adj[u].intersection(adj[v])
            for w in sorted(x for x in common if x > v):
                tri = (u, v, w)
                if tri not in observed_triangles:
                    open_triangles.append(tri)

    return open_triangles


# ---------------------------------------------------------------------------
# 6. label candidates using future window
# ---------------------------------------------------------------------------


def triangles_closed_in_future(future_simplices):
    future_triangles: Set[Triangle] = set()
    for simplex in future_simplices:
        V = simplex["vertices"]
        if len(V) >= 3:
            for tri in combinations(V, 3):
                future_triangles.add(tuple(sorted(tri)))
    return future_triangles


def label_candidates(candidates: Sequence[Triangle], future_triangles: Set[Triangle]) -> np.ndarray:
    return np.asarray([1 if tri in future_triangles else 0 for tri in candidates], dtype=np.int64)


# ---------------------------------------------------------------------------
# 7. build dual graph
# ---------------------------------------------------------------------------


def triangle_faces(tri: Triangle) -> Tuple[Edge, Edge, Edge]:
    a, b, c = tri
    return (
        tuple(sorted((a, b))),
        tuple(sorted((a, c))),
        tuple(sorted((b, c))),
    )


def build_dual_graph(
    triangles: Sequence[Triangle],
    boundary_strength=None,
    weight_mode: str = "binary",
    shared_face_reduce: str = "sum",
):
    face_to_tris: Dict[Edge, List[int]] = defaultdict(list)
    for i, tri in enumerate(triangles):
        for face in triangle_faces(tri):
            face_to_tris[face].append(i)

    edges: Set[Tuple[int, int]] = set()
    edge_weight = defaultdict(float)
    shared_face_count = defaultdict(int)

    for face, ids in face_to_tris.items():
        if len(ids) < 2:
            continue
        for i, j in combinations(ids, 2):
            u, v = sorted((i, j))
            edges.add((u, v))
            edge_weight[(u, v)] += strength_value(
                None if boundary_strength is None else boundary_strength.get(face),
                weight_mode,
            )
            shared_face_count[(u, v)] += 1

    if edges:
        edge_pairs = np.asarray(sorted(edges), dtype=np.int64)
        edge_index = np.vstack(
            [
                np.concatenate([edge_pairs[:, 0], edge_pairs[:, 1]]),
                np.concatenate([edge_pairs[:, 1], edge_pairs[:, 0]]),
            ]
        )
        weights = np.asarray(
            [
                edge_weight[tuple(edge)]
                / (shared_face_count[tuple(edge)] if shared_face_reduce == "mean" else 1.0)
                for edge in edge_pairs
            ],
            dtype=np.float32,
        )
        # edge_index is directed, so edge_weight must be aligned and directed too.
        directed_weights = np.concatenate([weights, weights]).astype(np.float32)
    else:
        edge_pairs = np.empty((0, 2), dtype=np.int64)
        edge_index = np.empty((2, 0), dtype=np.int64)
        weights = np.empty((0,), dtype=np.float32)
        directed_weights = np.empty((0,), dtype=np.float32)

    return {
        "edge_index": edge_index,
        "edge_pairs": edge_pairs,
        "edge_weight": weights,
        "directed_edge_weight": directed_weights,
        "face_to_tris": face_to_tris,
        "num_nodes": len(triangles),
    }


def dual_degrees(num_nodes: int, edge_pairs: np.ndarray) -> np.ndarray:
    degrees = np.zeros(num_nodes, dtype=np.float32)
    for u, v in edge_pairs:
        degrees[u] += 1.0
        degrees[v] += 1.0
    return degrees


# ---------------------------------------------------------------------------
# 8. build local structural features
# ---------------------------------------------------------------------------


COUNT_FEATURES = {
    "edge_degree_ab",
    "edge_degree_ac",
    "edge_degree_bc",
    "edge_degree_min",
    "edge_degree_mean",
    "edge_degree_max",
    "edge_degree_sum",
    "node_degree_a",
    "node_degree_b",
    "node_degree_c",
    "node_degree_min",
    "node_degree_mean",
    "node_degree_max",
    "node_degree_sum",
    "common_neighbors_ab",
    "common_neighbors_ac",
    "common_neighbors_bc",
    "cn_min",
    "cn_mean",
    "cn_max",
    "cn_sum",
    "face_support_ab",
    "face_support_ac",
    "face_support_bc",
    "face_support_min",
    "face_support_mean",
    "face_support_max",
    "face_support_sum",
    "SCN",
    "wSCN",
    "SPA",
    "wSPA",
    "dual_degree",
}


FEATURE_NAMES = [
    "edge_degree_ab",
    "edge_degree_ac",
    "edge_degree_bc",
    "edge_degree_min",
    "edge_degree_mean",
    "edge_degree_max",
    "edge_degree_sum",
    "node_degree_a",
    "node_degree_b",
    "node_degree_c",
    "node_degree_min",
    "node_degree_mean",
    "node_degree_max",
    "node_degree_sum",
    "common_neighbors_ab",
    "common_neighbors_ac",
    "common_neighbors_bc",
    "cn_min",
    "cn_mean",
    "cn_max",
    "cn_sum",
    "jaccard_ab",
    "jaccard_ac",
    "jaccard_bc",
    "jaccard_min",
    "jaccard_mean",
    "jaccard_max",
    "face_support_ab",
    "face_support_ac",
    "face_support_bc",
    "face_support_min",
    "face_support_mean",
    "face_support_max",
    "face_support_sum",
    "SCN",
    "SRA",
    "SAA",
    "SPA",
    "wSCN",
    "wSRA",
    "wSAA",
    "wSPA",
    "dual_degree",
    "log_dual_degree",
]

HEURISTIC_FEATURE_NAMES = [
    "cn_sum",
    "jaccard_mean",
    "SCN",
    "SRA",
    "SAA",
    "SPA",
    "wSCN",
    "wSRA",
    "wSAA",
    "wSPA",
    "face_support_sum",
    "edge_degree_sum",
]


def rw_feature_names(rw_steps: int) -> List[str]:
    steps = range(2, max(2, rw_steps) + 1)
    names = []
    names.extend([f"rw_return_prob_{step}" for step in steps])
    names.extend([f"rw_landing_entropy_{step}" for step in steps])
    names.extend(["pagerank", "personalized_pagerank_approx", "degree_normalized_pagerank"])
    return names


def local_feature_names() -> List[str]:
    return [name for name in FEATURE_NAMES if name not in HEURISTIC_FEATURE_NAMES]


def summarize_triplet(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.min()), float(arr.mean()), float(arr.max()), float(arr.sum())


def jaccard_for_pair(adj: Dict[int, Set[int]], u: int, v: int) -> float:
    nu = set(adj.get(u, set()))
    nv = set(adj.get(v, set()))
    union = (nu | nv) - {u, v}
    if not union:
        return 0.0
    inter = (nu & nv) - {u, v}
    return len(inter) / len(union)


def build_features(
    candidates: Sequence[Triangle],
    observed_state,
    dual_graph,
    return_raw: bool = False,
) -> np.ndarray:
    adj = build_adjacency(observed_state["edges"])
    node_degree = {node: len(neigh) for node, neigh in adj.items()}
    face_support = build_edge_triangle_support(observed_state["triangles"])
    edge_upper_neighbors, edge_upper_weight = build_edge_upper_adjacency(observed_state["triangles"])

    ddeg = dual_degrees(dual_graph["num_nodes"], dual_graph["edge_pairs"])

    rows = []
    raw_rows = []
    for i, tri in enumerate(candidates):
        a, b, c = tri
        faces = triangle_faces(tri)

        # Lower edge line-degree in the observed 1-skeleton.
        edge_degrees = [
            max(0, node_degree.get(u, 0) + node_degree.get(v, 0) - 2)
            for u, v in faces
        ]
        edge_min, edge_mean, edge_max, edge_sum = summarize_triplet(edge_degrees)

        node_degrees = [node_degree.get(a, 0), node_degree.get(b, 0), node_degree.get(c, 0)]
        node_min, node_mean, node_max, node_sum = summarize_triplet(node_degrees)

        common_neighbors = [
            len((adj.get(u, set()) & adj.get(v, set())) - {u, v})
            for u, v in faces
        ]
        cn_min, cn_mean, cn_max, cn_sum = summarize_triplet(common_neighbors)

        jaccards = [jaccard_for_pair(adj, u, v) for u, v in faces]
        jac_min, jac_mean, jac_max, _ = summarize_triplet(jaccards)

        supports = [face_support.get(face, 0) for face in faces]
        fs_min, fs_mean, fs_max, fs_sum = summarize_triplet(supports)

        gammas = [set(edge_upper_neighbors.get(face, set())) for face in faces]
        gamma_degrees = [len(gamma) for gamma in gammas]

        scn = 0.0
        sra = 0.0
        saa = 0.0
        spa = 0.0
        wscn = 0.0
        wsra = 0.0
        wsaa = 0.0
        wspa = 0.0
        for left, right in combinations(range(3), 2):
            common = gammas[left] & gammas[right]
            scn += len(common)
            for common_edge in common:
                degree = float(len(edge_upper_neighbors.get(common_edge, set())))
                if degree > 0:
                    sra += 1.0 / degree
                    saa += 1.0 / math.log(degree + 1.0)
            spa += gamma_degrees[left] * gamma_degrees[right]

            weighted_common = 0.0
            for common_edge in common:
                weight_left = edge_upper_weight.get((faces[left], common_edge), 0)
                weight_right = edge_upper_weight.get((faces[right], common_edge), 0)
                common_weight = min(weight_left, weight_right)
                weighted_common += common_weight
                degree = float(len(edge_upper_neighbors.get(common_edge, set())))
                if degree > 0:
                    wsra += common_weight / degree
                    wsaa += common_weight / math.log(degree + 1.0)
            wscn += weighted_common

            weighted_degree_left = sum(edge_upper_weight.get((faces[left], nbr), 0) for nbr in gammas[left])
            weighted_degree_right = sum(edge_upper_weight.get((faces[right], nbr), 0) for nbr in gammas[right])
            wspa += weighted_degree_left * weighted_degree_right

        row_map = {
            "edge_degree_ab": edge_degrees[0],
            "edge_degree_ac": edge_degrees[1],
            "edge_degree_bc": edge_degrees[2],
            "edge_degree_min": edge_min,
            "edge_degree_mean": edge_mean,
            "edge_degree_max": edge_max,
            "edge_degree_sum": edge_sum,
            "node_degree_a": node_degrees[0],
            "node_degree_b": node_degrees[1],
            "node_degree_c": node_degrees[2],
            "node_degree_min": node_min,
            "node_degree_mean": node_mean,
            "node_degree_max": node_max,
            "node_degree_sum": node_sum,
            "common_neighbors_ab": common_neighbors[0],
            "common_neighbors_ac": common_neighbors[1],
            "common_neighbors_bc": common_neighbors[2],
            "cn_min": cn_min,
            "cn_mean": cn_mean,
            "cn_max": cn_max,
            "cn_sum": cn_sum,
            "jaccard_ab": jaccards[0],
            "jaccard_ac": jaccards[1],
            "jaccard_bc": jaccards[2],
            "jaccard_min": jac_min,
            "jaccard_mean": jac_mean,
            "jaccard_max": jac_max,
            "face_support_ab": supports[0],
            "face_support_ac": supports[1],
            "face_support_bc": supports[2],
            "face_support_min": fs_min,
            "face_support_mean": fs_mean,
            "face_support_max": fs_max,
            "face_support_sum": fs_sum,
            "SCN": scn,
            "SRA": sra,
            "SAA": saa,
            "SPA": spa,
            "wSCN": wscn,
            "wSRA": wsra,
            "wSAA": wsaa,
            "wSPA": wspa,
            "dual_degree": ddeg[i],
            "log_dual_degree": math.log1p(ddeg[i]),
        }

        row = []
        raw_row = []
        for name in FEATURE_NAMES:
            value = float(row_map[name])
            raw_row.append(value)
            if name in COUNT_FEATURES:
                value = math.log1p(max(0.0, value))
            row.append(value)
        rows.append(row)
        raw_rows.append(raw_row)

    if not rows:
        empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return (empty, empty) if return_raw else empty
    features = np.asarray(rows, dtype=np.float32)
    raw_features = np.asarray(raw_rows, dtype=np.float32)
    return (features, raw_features) if return_raw else features


def build_random_walk_structural_features(edge_index: np.ndarray, num_nodes: int, rw_steps: int = 4) -> np.ndarray:
    names = rw_feature_names(rw_steps)
    if num_nodes == 0:
        return np.empty((0, len(names)), dtype=np.float32)
    if edge_index.size == 0:
        return np.zeros((num_nodes, len(names)), dtype=np.float32)
    if sparse is None:
        raise RuntimeError("scipy is required for sparse random-walk structural features.")

    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    data = np.ones(len(src), dtype=np.float64)
    adj = sparse.csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    p = sparse.diags(inv_degree).dot(adj).tocsr()

    current = p.copy()
    return_probs = {}
    entropies = {}
    for step in range(1, max(2, rw_steps) + 1):
        if step > 1:
            current = current.dot(p).tocsr()
        if step >= 2:
            return_probs[step] = current.diagonal().astype(np.float64)
            entropy = np.zeros(num_nodes, dtype=np.float64)
            for row_id in range(num_nodes):
                row = current.data[current.indptr[row_id] : current.indptr[row_id + 1]]
                if row.size:
                    row = row[row > 0]
                    entropy[row_id] = -float(np.sum(row * np.log(row + 1e-12)))
            entropies[step] = entropy

    alpha = 0.15
    pr = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    teleport = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    for _ in range(50):
        pr = alpha * teleport + (1.0 - alpha) * p.T.dot(pr)
    pr = pr / max(pr.sum(), 1e-12)

    ppr_diag = np.full(num_nodes, alpha, dtype=np.float64)
    coeff = alpha
    for step in range(1, max(2, rw_steps) + 1):
        coeff *= (1.0 - alpha)
        if step in return_probs:
            ppr_diag += coeff * return_probs[step]
    degree_norm_pr = pr / np.log1p(degree + 1.0)

    columns = []
    for step in range(2, max(2, rw_steps) + 1):
        columns.append(return_probs[step])
    for step in range(2, max(2, rw_steps) + 1):
        columns.append(entropies[step])
    columns.extend([pr, ppr_diag, degree_norm_pr])
    return np.vstack(columns).T.astype(np.float32)


def select_feature_block(features: np.ndarray, names: Sequence[str], selected_names: Sequence[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    cols = [index[name] for name in selected_names]
    if len(cols) == 0:
        return np.empty((features.shape[0], 0), dtype=np.float32)
    return features[:, cols].astype(np.float32)


def task_rank_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.shape[0] == 0:
        return scores.astype(np.float32)
    ranks = np.zeros_like(scores, dtype=np.float32)
    denom = max(1, scores.shape[0])
    for col in range(scores.shape[1]):
        ranks[:, col] = rankdata(scores[:, col], method="average") / denom
    return ranks


def task_zscore(scores: np.ndarray) -> np.ndarray:
    if scores.shape[0] == 0:
        return scores.astype(np.float32)
    mean = scores.mean(axis=0, keepdims=True)
    std = scores.std(axis=0, keepdims=True)
    return ((scores - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def build_heuristic_feature_block(raw_features: np.ndarray) -> np.ndarray:
    raw = select_feature_block(raw_features, FEATURE_NAMES, HEURISTIC_FEATURE_NAMES)
    log_values = np.log1p(np.maximum(raw, 0.0)).astype(np.float32)
    ranks = task_rank_normalize(raw)
    zscores = task_zscore(raw)
    return np.hstack([raw, log_values, ranks, zscores]).astype(np.float32)


def heuristic_block_names() -> List[str]:
    names = []
    for prefix in ["raw", "log1p", "rank", "zscore"]:
        names.extend([f"{prefix}_{name}" for name in HEURISTIC_FEATURE_NAMES])
    return names


def build_closure_task(
    context_simplices,
    future_simplices,
    task_name: str,
    use_rw_struct_features=True,
    rw_steps=4,
    args=None,
):
    """Build one independent temporal closure task.

    The state, candidates, derived graph, and features come only from the
    context window.  The future window is used only to create labels.
    """
    state = build_observed_state(context_simplices)
    candidates = enumerate_open_triangles(state["edges"], state["triangles"])
    future_closed = triangles_closed_in_future(future_simplices)
    labels = label_candidates(candidates, future_closed)
    if context_simplices:
        cutoff = max(float(item["time"]) for item in context_simplices) + 1e-9
    else:
        cutoff = 0.0
    decay_lambda = float(getattr(args, "strength_decay_lambda", 0.0))
    boundary_strength = build_boundary_strength(context_simplices, cutoff, decay_lambda)
    dual_graph = build_dual_graph(
        candidates,
        boundary_strength,
        getattr(args, "dual_edge_weight", "binary"),
        getattr(args, "dual_shared_face_reduce", "sum"),
    )
    features, raw_features = build_features(candidates, state, dual_graph, return_raw=True)
    rw_features = (
        build_random_walk_structural_features(dual_graph["edge_index"], dual_graph["num_nodes"], rw_steps)
        if use_rw_struct_features
        else np.empty((len(candidates), 0), dtype=np.float32)
    )
    heuristic_raw_features = select_feature_block(raw_features, FEATURE_NAMES, HEURISTIC_FEATURE_NAMES)
    heuristic_features = build_heuristic_feature_block(raw_features)
    local_features = select_feature_block(features, FEATURE_NAMES, local_feature_names())
    strength_features, strength_feature_names, strength_rows = candidate_strength_matrix(
        candidates, boundary_strength
    )
    recent_fraction = float(getattr(args, "node_activity_recent_fraction", 0.15))
    recent_count = max(1, int(round(len(context_simplices) * recent_fraction)))
    node_counts = defaultdict(float)
    for event in context_simplices[-recent_count:]:
        for node in set(map(int, event["vertices"])):
            node_counts[node] += 1.0
    node_activity_features = np.asarray(
        [
            [
                min(node_counts[node] for node in tri),
                np.mean([node_counts[node] for node in tri]),
                max(node_counts[node] for node in tri),
                sum(math.log1p(node_counts[node]) for node in tri),
            ]
            for tri in candidates
        ],
        dtype=np.float32,
    )
    information_mode = getattr(args, "information_mode", "full")
    empty = np.empty((len(candidates), 0), dtype=np.float32)
    if information_mode == "strength":
        local_features = strength_features
        heuristic_features = empty
    elif information_mode in {"topology+strength", "full"}:
        local_features = np.hstack([local_features, strength_features]).astype(np.float32)

    return {
        "name": task_name,
        "context_size": len(context_simplices),
        "future_size": len(future_simplices),
        "state": state,
        "candidates": candidates,
        "labels": labels,
        "priority_mask": labels == 1,
        "dual_graph": dual_graph,
        "edge_index": dual_graph["edge_index"],
        "edge_weight": dual_graph["directed_edge_weight"],
        "features": features,
        "raw_features": raw_features,
        "local_features": local_features,
        "heuristic_raw_features": heuristic_raw_features,
        "heuristic_features": heuristic_features,
        "rw_struct_features": rw_features,
        "rw_features": rw_features,
        "strength_features": strength_features,
        "strength_feature_names": strength_feature_names,
        "strength_rows": strength_rows,
        "node_activity_features": node_activity_features,
        "boundary_strength": boundary_strength,
        "cutoff": cutoff,
    }


def build_global_union_graph(tasks: Sequence[dict]):
    global_triangles = sorted(set().union(*(set(task["candidates"]) for task in tasks)))
    global_tri_to_id = {tri: i for i, tri in enumerate(global_triangles)}
    global_edges = set()
    for task in tasks:
        candidates = task["candidates"]
        for u, v in task["dual_graph"]["edge_pairs"]:
            gu = global_tri_to_id[candidates[int(u)]]
            gv = global_tri_to_id[candidates[int(v)]]
            if gu != gv:
                global_edges.add(tuple(sorted((gu, gv))))
    adjacency = [set() for _ in global_triangles]
    for u, v in global_edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    return global_triangles, global_tri_to_id, sorted(global_edges), adjacency


def priority_global_ids(priority_tasks: Sequence[dict], global_tri_to_id: Dict[Triangle, int]) -> Set[int]:
    ids = set()
    for task in priority_tasks:
        for tri, is_priority in zip(task["candidates"], task["priority_mask"]):
            if bool(is_priority) and tri in global_tri_to_id:
                ids.add(global_tri_to_id[tri])
    return ids


def union_graph_hash(num_nodes: int, edges: Sequence[Tuple[int, int]], priority_ids: Set[int], args) -> str:
    payload = {
        "event_order": args.event_order,
        "num_nodes": num_nodes,
        "edges": edges,
        "priority_ids": sorted(priority_ids),
        "rw_dim": args.rw_dim,
        "rw_walk_length": args.rw_walk_length,
        "rw_walks_per_node": args.rw_walks_per_node,
        "rw_window_size": args.rw_window_size,
        "rw_epochs": args.rw_epochs,
        "rw_negative_samples": args.rw_negative_samples,
        "embedding_seed": args.embedding_seed,
        "priority_walk_multiplier": args.priority_walk_multiplier,
        "rw_max_pairs": args.rw_max_pairs,
        "rw_sampler_version": 2,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def generate_random_walks(adjacency: Sequence[Set[int]], start_nodes: Sequence[int], walk_length: int, seed: int):
    rng = random.Random(seed)
    walks = []
    for start in start_nodes:
        walk = [int(start)]
        cur = int(start)
        for _ in range(max(0, walk_length - 1)):
            neigh = list(adjacency[cur])
            if not neigh:
                break
            cur = int(rng.choice(neigh))
            walk.append(cur)
        walks.append(walk)
    return walks


def train_skipgram_embedding(
    num_nodes: int,
    center_ids: np.ndarray,
    context_ids: np.ndarray,
    args,
) -> np.ndarray:
    torch, nn, F = require_torch()
    set_seed(args.embedding_seed, torch)
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    if len(center_ids) == 0:
        rng = np.random.default_rng(args.embedding_seed)
        return rng.normal(0.0, 0.01, size=(num_nodes, args.rw_dim)).astype(np.float32)

    centers = torch.from_numpy(center_ids).to(device=device, dtype=torch.long)
    contexts = torch.from_numpy(context_ids).to(device=device, dtype=torch.long)
    emb_in = nn.Embedding(num_nodes, args.rw_dim).to(device)
    emb_out = nn.Embedding(num_nodes, args.rw_dim).to(device)
    nn.init.xavier_uniform_(emb_in.weight)
    nn.init.xavier_uniform_(emb_out.weight)
    optimizer = torch.optim.Adam([emb_in.weight, emb_out.weight], lr=0.01)
    batch_size = 4096
    num_pairs = centers.numel()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.embedding_seed)

    for _ in range(int(args.rw_epochs)):
        perm = torch.randperm(num_pairs, device=device, generator=generator)
        for start in range(0, num_pairs, batch_size):
            idx = perm[start : start + batch_size]
            c = centers[idx]
            ctx = contexts[idx]
            c_emb = emb_in(c)
            pos_emb = emb_out(ctx)
            pos_score = (c_emb * pos_emb).sum(dim=-1)
            loss_pos = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            neg = torch.randint(
                low=0,
                high=num_nodes,
                size=(idx.numel(), int(args.rw_negative_samples)),
                device=device,
                generator=generator,
            )
            neg_emb = emb_out(neg)
            neg_score = torch.einsum("bd,bkd->bk", c_emb, neg_emb)
            loss_neg = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            loss = loss_pos + loss_neg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    embedding = emb_in.weight.detach().cpu().numpy().astype(np.float32)
    del centers, contexts, emb_in, emb_out, optimizer
    if use_cuda:
        torch.cuda.empty_cache()
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / np.maximum(norms, 1e-6)


def skipgram_pairs_per_full_walk(walk_length: int, window: int) -> int:
    return 2 * sum(max(0, walk_length - distance) for distance in range(1, window + 1))


def build_bounded_skipgram_pairs(
    num_nodes: int,
    adjacency: Sequence[Set[int]],
    priority_ids: Set[int],
    args,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate random-walk pairs with a strict allocation cap.

    Small graphs retain the original complete walk multiset.  When the
    requested multiset would exceed ``rw_max_pairs``, every node receives one
    start walk first and the remaining starts are sampled with the original
    priority-node bias.  Pair arrays are preallocated, so peak memory is known
    before training and no multi-gigabyte Python pair list is constructed.
    """
    walk_length = int(args.rw_walk_length)
    window = int(args.rw_window_size)
    walks_per_node = int(args.rw_walks_per_node)
    priority_extra = max(0, int(args.priority_walk_multiplier) - 1) * walks_per_node
    max_pairs = max(1, int(args.rw_max_pairs))
    full_pairs_per_walk = max(1, skipgram_pairs_per_full_walk(walk_length, window))
    requested_walks = num_nodes * walks_per_node + len(priority_ids) * priority_extra
    requested_pairs_upper = requested_walks * full_pairs_per_walk
    rng = random.Random(int(args.embedding_seed))

    if requested_pairs_upper <= max_pairs:
        start_nodes = []
        for node in range(num_nodes):
            start_nodes.extend([node] * walks_per_node)
        for node in sorted(priority_ids):
            start_nodes.extend([node] * priority_extra)
        capped = False
    else:
        max_walks = max(num_nodes, max_pairs // full_pairs_per_walk)
        start_nodes = list(range(num_nodes))
        rng.shuffle(start_nodes)
        remaining = max(0, max_walks - len(start_nodes))
        priority_nodes = sorted(priority_ids)
        base_mass = max(1, num_nodes * walks_per_node)
        priority_mass = len(priority_nodes) * priority_extra
        priority_probability = priority_mass / (base_mass + priority_mass)
        for _ in range(remaining):
            if priority_nodes and rng.random() < priority_probability:
                start_nodes.append(rng.choice(priority_nodes))
            else:
                start_nodes.append(rng.randrange(num_nodes))
        capped = True

    allocation_pairs = min(max_pairs, requested_pairs_upper)
    centers = np.empty(allocation_pairs, dtype=np.int64)
    contexts = np.empty(allocation_pairs, dtype=np.int64)
    cursor = 0
    if capped:
        walk_iter = (
            generate_random_walks(adjacency, [start], walk_length, rng.randrange(2**31))[0]
            for start in start_nodes
        )
    else:
        # Preserve the original continuous-RNG walk sequence exactly whenever
        # the complete pair set fits under the memory cap.
        walk_iter = iter(
            generate_random_walks(adjacency, start_nodes, walk_length, int(args.embedding_seed))
        )
    generated_walks = 0
    for walk in walk_iter:
        generated_walks += 1
        for i, center in enumerate(walk):
            lo = max(0, i - window)
            hi = min(len(walk), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                if cursor >= allocation_pairs:
                    break
                centers[cursor] = center
                contexts[cursor] = walk[j]
                cursor += 1
            if cursor >= allocation_pairs:
                break
        if cursor >= allocation_pairs:
            break

    stats = {
        "requested_walks": requested_walks,
        "generated_walks": generated_walks,
        "requested_pairs_upper": requested_pairs_upper,
        "generated_pairs": cursor,
        "pair_cap": max_pairs,
        "capped": capped,
    }
    return centers[:cursor], contexts[:cursor], stats


def build_aligned_random_walk_embedding(
    num_nodes: int,
    adjacency: Sequence[Set[int]],
    edges: Sequence[Tuple[int, int]],
    priority_ids: Set[int],
    args,
) -> np.ndarray:
    cache_dir = Path(".cache_rw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = union_graph_hash(num_nodes, edges, priority_ids, args)
    cache_path = cache_dir / f"aligned_union_rw_{cache_key}.npz"
    if cache_path.exists():
        return np.load(cache_path)["embedding"].astype(np.float32)

    centers, contexts, sampling_stats = build_bounded_skipgram_pairs(
        num_nodes, adjacency, priority_ids, args
    )
    print(
        "  random-walk skip-gram: "
        f"pairs={sampling_stats['generated_pairs']:,}/cap={sampling_stats['pair_cap']:,} "
        f"walks={sampling_stats['generated_walks']:,} capped={sampling_stats['capped']}"
    )
    embedding = train_skipgram_embedding(num_nodes, centers, contexts, args)
    np.savez_compressed(cache_path, embedding=embedding)
    return embedding


def bfs_distances(adjacency: Sequence[Set[int]], source: int) -> np.ndarray:
    dist = np.full(len(adjacency), np.inf, dtype=np.float32)
    queue = [source]
    dist[source] = 0.0
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for nbr in adjacency[node]:
            if not np.isfinite(dist[nbr]):
                dist[nbr] = dist[node] + 1.0
                queue.append(nbr)
    return dist


def build_priority_features(num_nodes: int, adjacency: Sequence[Set[int]], priority_ids: Set[int]) -> np.ndarray:
    if num_nodes == 0:
        return np.empty((0, 5), dtype=np.float32)
    priority = np.zeros(num_nodes, dtype=np.float32)
    for node in priority_ids:
        priority[node] = 1.0
    if not priority_ids:
        return np.zeros((num_nodes, 6), dtype=np.float32)

    top3 = np.full((num_nodes, 3), np.inf, dtype=np.float32)
    for source in sorted(priority_ids):
        dist = bfs_distances(adjacency, source)
        combined = np.concatenate([top3, dist[:, None]], axis=1)
        combined.sort(axis=1)
        top3 = combined[:, :3]
    min_dist = top3[:, 0]
    finite_mask = np.isfinite(top3)
    finite_sum = np.where(finite_mask, top3, 0.0).sum(axis=1)
    finite_count = finite_mask.sum(axis=1)
    mean_top3 = np.divide(finite_sum, finite_count, out=np.full(num_nodes, 99.0, dtype=np.float32), where=finite_count > 0)
    min_dist = np.where(np.isfinite(min_dist), min_dist, 99.0)
    mean_top3 = np.where(np.isfinite(mean_top3), mean_top3, 99.0)

    one_hop = np.zeros(num_nodes, dtype=np.float32)
    two_hop = np.zeros(num_nodes, dtype=np.float32)
    for node in range(num_nodes):
        neigh1 = adjacency[node]
        one_hop[node] = sum(1 for nbr in neigh1 if nbr in priority_ids)
        neigh2 = set()
        for nbr in neigh1:
            neigh2.update(adjacency[nbr])
        neigh2.discard(node)
        two_hop[node] = sum(1 for nbr in neigh2 if nbr in priority_ids)

    if sparse is None:
        rw_priority = priority.copy()
    else:
        rows = []
        cols = []
        for u, nbrs in enumerate(adjacency):
            for v in nbrs:
                rows.append(u)
                cols.append(v)
        data = np.ones(len(rows), dtype=np.float64)
        adj = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        degree = np.asarray(adj.sum(axis=1)).ravel()
        inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
        p = sparse.diags(inv_degree).dot(adj).tocsr()
        seed = priority.astype(np.float64)
        seed = seed / max(seed.sum(), 1e-12)
        score = seed.copy()
        alpha = 0.15
        for _ in range(30):
            score = alpha * seed + (1.0 - alpha) * p.T.dot(score)
        rw_priority = score.astype(np.float32)

    return np.vstack(
        [
            priority,
            np.log1p(min_dist),
            np.log1p(mean_top3),
            np.log1p(one_hop),
            np.log1p(two_hop),
            rw_priority,
        ]
    ).T.astype(np.float32)


def attach_aligned_rw_and_priority_features(tasks: Sequence[dict], priority_tasks: Sequence[dict], args):
    global_triangles, global_tri_to_id, global_edges, adjacency = build_global_union_graph(tasks)
    priority_ids = priority_global_ids(priority_tasks, global_tri_to_id)
    if args.use_random_walk:
        global_rw = build_aligned_random_walk_embedding(len(global_triangles), adjacency, global_edges, priority_ids, args)
    else:
        global_rw = np.zeros((len(global_triangles), 0), dtype=np.float32)
    global_priority = build_priority_features(len(global_triangles), adjacency, priority_ids)

    for task in tasks:
        ids = np.asarray([global_tri_to_id[tri] for tri in task["candidates"]], dtype=np.int64)
        task["aligned_rw_features"] = global_rw[ids].astype(np.float32)
        task["priority_features"] = global_priority[ids].astype(np.float32)
        task["rw_features"] = np.hstack([task["aligned_rw_features"], task["rw_struct_features"]]).astype(np.float32)
    return {
        "num_global_triangles": len(global_triangles),
        "num_global_edges": len(global_edges),
        "num_priority_nodes": len(priority_ids),
    }


def apply_information_mode_features(tasks: Sequence[dict], args) -> None:
    """Add the D-only node-activity channel without removing shared RW/priority."""
    for task in tasks:
        if args.information_mode == "full":
            task["priority_features"] = np.hstack(
                [task["priority_features"], task["node_activity_features"]]
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# 9. model definition
# ---------------------------------------------------------------------------


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required for training. Use an environment with torch "
            "or pass --skip_training to only build and save the protocol data."
        ) from exc
    return torch, nn, F


def make_model_classes():
    torch, nn, F = require_torch()

    class GraphSAGELayer(nn.Module):
        def __init__(self, dim: int, dropout: float):
            super().__init__()
            self.lin_self = nn.Linear(dim, dim)
            self.lin_neigh = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, h, edge_index, edge_weight=None):
            num_nodes = h.size(0)
            if edge_index.numel() == 0:
                neigh = torch.zeros_like(h)
            else:
                src, dst = edge_index[0], edge_index[1]
                if edge_weight is None:
                    edge_weight = torch.ones_like(src, dtype=h.dtype)
                else:
                    edge_weight = edge_weight.to(dtype=h.dtype)
                neigh = torch.zeros_like(h)
                neigh.index_add_(0, dst, h[src] * edge_weight.unsqueeze(-1))
                deg = torch.zeros(num_nodes, device=h.device, dtype=h.dtype)
                deg.index_add_(0, dst, edge_weight)
                neigh = neigh / deg.clamp_min(1.0).unsqueeze(-1)

            out = self.lin_self(h) + self.lin_neigh(neigh)
            out = self.norm(out)
            out = F.relu(out)
            return self.dropout(out)

    class FeatureBlockEncoder(nn.Module):
        def __init__(self, local_dim: int, heuristic_dim: int, rw_dim: int, priority_dim: int, proj_dim: int, dropout: float):
            super().__init__()
            self.proj_dim = proj_dim
            self.local_proj = self._make_proj(local_dim, proj_dim, dropout)
            self.heuristic_proj = self._make_proj(heuristic_dim, proj_dim, dropout)
            self.rw_proj = self._make_proj(rw_dim, proj_dim, dropout)
            self.priority_proj = self._make_proj(priority_dim, proj_dim, dropout)
            self.norm = nn.LayerNorm(proj_dim * 4)

        @staticmethod
        def _make_proj(in_dim: int, out_dim: int, dropout: float):
            if in_dim <= 0:
                return None
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def _apply_proj(self, proj, x):
            if proj is None:
                return x.new_zeros((x.size(0), self.proj_dim))
            return proj(x)

        def forward(self, local_x, heuristic_x, rw_x, priority_x):
            h_local = self._apply_proj(self.local_proj, local_x)
            h_heur = self._apply_proj(self.heuristic_proj, heuristic_x)
            h_rw = self._apply_proj(self.rw_proj, rw_x)
            h_priority = self._apply_proj(self.priority_proj, priority_x)
            return self.norm(torch.cat([h_local, h_heur, h_rw, h_priority], dim=-1))

    class DualMappingGNN(nn.Module):
        def __init__(
            self,
            local_dim: int,
            heuristic_dim: int,
            rw_dim: int,
            priority_dim: int,
            feature_proj_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 2,
            encoder: str = "sage",
            dropout: float = 0.4,
            max_residual_alpha: float = 0.2,
        ):
            super().__init__()
            if encoder == "gat":
                print("WARNING: --encoder gat requested; this version uses residual GraphSAGE instead.")
                encoder = "sage"
            self.encoder = encoder
            self.model_mode = "heuristic_residual_gnn"
            self.block_encoder = FeatureBlockEncoder(
                local_dim=local_dim,
                heuristic_dim=heuristic_dim,
                rw_dim=rw_dim,
                priority_dim=priority_dim,
                proj_dim=feature_proj_dim,
                dropout=dropout,
            )
            block_dim = feature_proj_dim * 4
            self.input_linear = nn.Sequential(
                nn.Linear(block_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gnn_layers = nn.ModuleList(
                [GraphSAGELayer(hidden_dim, dropout) for _ in range(max(0, num_layers))]
            )
            self.residual_projection = nn.Identity()
            self.base_head = nn.Sequential(
                nn.Linear(heuristic_dim + rw_dim + priority_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.alpha_param = nn.Parameter(torch.tensor(0.0))
            self.max_residual_alpha = float(max_residual_alpha)
            self.use_residual = True

        def forward(self, local_x, heuristic_x, rw_x, priority_x, edge_index, edge_weight=None):
            base_input = torch.cat([heuristic_x, rw_x, priority_x], dim=-1)
            base_logit = self.base_head(base_input).squeeze(-1)
            if not self.use_residual:
                return base_logit
            x = self.block_encoder(local_x, heuristic_x, rw_x, priority_x)
            h0 = self.input_linear(x)
            h = h0
            if self.encoder == "sage":
                for layer in self.gnn_layers:
                    h = layer(h, edge_index, edge_weight)
                h = h + self.residual_projection(h0)
            residual = self.residual_head(h).squeeze(-1)
            alpha = torch.sigmoid(self.alpha_param) * self.max_residual_alpha
            return base_logit + alpha * residual

        def set_stage1_mode(self):
            self.use_residual = False
            for param in self.parameters():
                param.requires_grad = False
            for param in self.base_head.parameters():
                param.requires_grad = True

        def set_stage2_mode(self):
            self.use_residual = True
            for param in self.parameters():
                param.requires_grad = True

    return torch, nn, F, DualMappingGNN


# ---------------------------------------------------------------------------
# 10. training and evaluation
# ---------------------------------------------------------------------------


def set_seed(seed: int, torch=None):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0 or int(y_true.sum()) == 0:
        return 0.0
    return float(average_precision_score(y_true, score))


def heuristic_ap(task) -> Dict[str, float]:
    labels = task["labels"]
    values = task["heuristic_raw_features"]
    name_to_col = {name: i for i, name in enumerate(HEURISTIC_FEATURE_NAMES)}
    specs = {
        "CN": "cn_sum",
        "Jaccard": "jaccard_mean",
        "SCN": "SCN",
        "SRA": "SRA",
        "SAA": "SAA",
        "SPA": "SPA",
        "wSCN": "wSCN",
        "wSRA": "wSRA",
        "wSAA": "wSAA",
        "wSPA": "wSPA",
        "face_support_sum": "face_support_sum",
        "edge_degree_sum": "edge_degree_sum",
    }
    scores = {}
    for output_name, feature_name in specs.items():
        scores[output_name] = safe_average_precision(labels, values[:, name_to_col[feature_name]])
    fusion_names = ["cn_sum", "jaccard_mean", "SPA", "edge_degree_sum", "face_support_sum"]
    fusion_cols = [name_to_col[name] for name in fusion_names]
    fusion_ranks = task_rank_normalize(values[:, fusion_cols])
    scores["rank_fusion"] = safe_average_precision(labels, fusion_ranks.mean(axis=1))
    return scores


def all_heuristic_ap(tasks: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    return {task["name"]: heuristic_ap(task) for task in tasks}


def print_heuristic_baselines(heuristics: Dict[str, Dict[str, float]]):
    print("\nHeuristic baseline AP:")
    for task_name, scores in heuristics.items():
        print(f"  {task_name}:")
        for name in [
            "CN",
            "Jaccard",
            "SCN",
            "SRA",
            "SAA",
            "SPA",
            "wSCN",
            "wSRA",
            "wSAA",
            "wSPA",
            "face_support_sum",
            "edge_degree_sum",
            "rank_fusion",
        ]:
            print(f"    {name}: {scores[name]:.6f}")


def choose_device(device_arg: str, torch):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ranking_loss(logits, labels, hard_negative_top: int, F):
    pos_logits = logits[labels > 0.5]
    neg_logits = logits[labels <= 0.5]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return logits.new_tensor(0.0)
    k = min(int(hard_negative_top), int(3 * pos_logits.numel()), neg_logits.numel())
    hard_neg = torch_topk(neg_logits, k)
    diffs = pos_logits.view(-1, 1) - hard_neg.view(1, -1)
    return -F.logsigmoid(diffs).mean()


def pu_ranking_loss(
    logits,
    labels,
    pos_weight=1.0,
    unlabeled_weight=0.03,
    lambda_rank=0.05,
    F=None,
):
    torch, _, _ = require_torch()
    pos_logits = logits[labels > 0.5]
    unl_logits = logits[labels <= 0.5]
    loss = logits.new_tensor(0.0)
    if pos_logits.numel() > 0:
        loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_logits.new_ones(pos_logits.shape))
        loss = loss + float(pos_weight) * loss_pos
    if unl_logits.numel() > 0:
        loss_unl = F.binary_cross_entropy_with_logits(unl_logits, unl_logits.new_zeros(unl_logits.shape))
        loss = loss + float(unlabeled_weight) * loss_unl
    if pos_logits.numel() > 0 and unl_logits.numel() > 0:
        k = min(unl_logits.numel(), max(256, 3 * pos_logits.numel()))
        perm = torch.randperm(unl_logits.numel(), device=logits.device)[:k]
        sampled_unl = unl_logits[perm]
        loss_rank = -F.logsigmoid(pos_logits[:, None] - sampled_unl[None, :]).mean()
        loss = loss + float(lambda_rank) * loss_rank
    return loss


def torch_topk(values, k: int):
    if k <= 0 or k >= values.numel():
        return values
    return values.topk(k=k, largest=True).values


def resolve_pos_weight(pos_weight_arg, train_labels: Sequence[np.ndarray]) -> float:
    labels = np.concatenate([np.asarray(y).ravel() for y in train_labels])
    num_pos = float(labels.sum())
    num_neg = float(len(labels) - labels.sum())
    if num_pos <= 0 or num_neg <= 0:
        return 1.0
    text = str(pos_weight_arg).strip().lower()
    if text == "auto_sqrt":
        return math.sqrt(num_neg / num_pos)
    if text == "auto":
        return num_neg / num_pos
    return float(pos_weight_arg)


def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    train_inputs: Sequence[dict],
    val_input: dict,
    test_input: dict,
    result_dir: Path,
):
    torch, nn, F, DualMappingGNN = make_model_classes()
    set_seed(seed, torch)
    device = choose_device(args.device, torch)

    model = DualMappingGNN(
        local_dim=train_inputs[0]["local"].shape[1],
        heuristic_dim=train_inputs[0]["heuristic"].shape[1],
        rw_dim=train_inputs[0]["rw"].shape[1],
        priority_dim=train_inputs[0]["priority"].shape[1],
        feature_proj_dim=args.feature_proj_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        encoder=args.encoder,
        dropout=args.dropout,
        max_residual_alpha=args.max_residual_alpha,
    ).to(device)

    def to_tensors(item):
        return {
            "local": torch.tensor(item["local"], dtype=torch.float32, device=device),
            "heuristic": torch.tensor(item["heuristic"], dtype=torch.float32, device=device),
            "rw": torch.tensor(item["rw"], dtype=torch.float32, device=device),
            "priority": torch.tensor(item["priority"], dtype=torch.float32, device=device),
            "labels": torch.tensor(item["labels"], dtype=torch.float32, device=device),
            "edge_index": torch.tensor(item["edge_index"], dtype=torch.long, device=device),
            "edge_weight": torch.tensor(item["edge_weight"], dtype=torch.float32, device=device),
        }

    train_tensors = [to_tensors(item) for item in train_inputs]
    val_t = to_tensors(val_input)
    test_t = to_tensors(test_input)
    pos_weight_value = resolve_pos_weight(args.pos_weight, [item["labels"] for item in train_inputs])
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def compute_loss(item):
        logits = model(item["local"], item["heuristic"], item["rw"], item["priority"], item["edge_index"], item["edge_weight"])
        if args.loss == "pu_ranking":
            loss = pu_ranking_loss(
                logits,
                item["labels"],
                pos_weight=pos_weight_value,
                unlabeled_weight=args.unlabeled_weight,
                lambda_rank=args.lambda_rank,
                F=F,
            )
            return loss, loss.detach(), logits.new_tensor(0.0)
        loss_bce = criterion(logits, item["labels"])
        loss_rank = logits.new_tensor(0.0)
        loss = loss_bce
        if args.loss == "bce_ranking":
            loss_rank = ranking_loss(logits, item["labels"], args.hard_negative_top, F)
            loss = loss_bce + float(args.lambda_rank) * loss_rank
        return loss, loss_bce.detach(), loss_rank.detach()

    def evaluate_scores():
        model.eval()
        with torch.no_grad():
            train_scores_local = [
                torch.sigmoid(model(item["local"], item["heuristic"], item["rw"], item["priority"], item["edge_index"], item["edge_weight"]))
                .detach()
                .cpu()
                .numpy()
                for item in train_tensors
            ]
            val_scores_local = torch.sigmoid(
                model(val_t["local"], val_t["heuristic"], val_t["rw"], val_t["priority"], val_t["edge_index"], val_t["edge_weight"])
            ).detach().cpu().numpy()
            test_scores_local = torch.sigmoid(
                model(test_t["local"], test_t["heuristic"], test_t["rw"], test_t["priority"], test_t["edge_index"], test_t["edge_weight"])
            ).detach().cpu().numpy()
        return np.concatenate(train_scores_local), val_scores_local, test_scores_local

    def run_stage(stage_name: str, epochs: int, lr: float, patience: int):
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=args.weight_decay,
        )
        best = {
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "val_ap": -1.0,
            "epoch": 0,
        }
        history_rows = []
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            losses = []
            bce_losses = []
            rank_losses = []
            for item in train_tensors:
                loss, loss_bce, loss_rank = compute_loss(item)
                losses.append(loss)
                bce_losses.append(loss_bce)
                rank_losses.append(loss_rank)
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

            _, val_scores_local, _ = evaluate_scores()
            val_ap = safe_average_precision(val_input["labels"], val_scores_local)
            row = {
                "stage": stage_name,
                "epoch": epoch,
                "loss": float(loss.detach().cpu().item()),
                "loss_bce": float(torch.stack(bce_losses).mean().cpu().item()),
                "loss_rank": float(torch.stack(rank_losses).mean().cpu().item()),
                "val_ap": val_ap,
                "pos_weight": float(pos_weight_value),
                "alpha": float((torch.sigmoid(model.alpha_param.detach()) * model.max_residual_alpha).cpu().item()),
            }
            history_rows.append(row)
            if val_ap > best["val_ap"]:
                best = {
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "val_ap": val_ap,
                    "epoch": epoch,
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
            if args.verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"seed={seed} {stage_name} epoch={epoch:04d} loss={row['loss']:.6f} val_AP={val_ap:.6f}")
            if bad_epochs >= patience:
                break
        return best, history_rows

    model.set_stage1_mode()
    stage1_best, stage1_history = run_stage("stage1_heuristic_head", args.stage1_epochs, args.lr, args.early_stop_patience)
    model.load_state_dict(stage1_best["state"])
    model.use_residual = False
    stage1_train_scores, stage1_val_scores, stage1_test_scores = evaluate_scores()
    stage1_train_ap = safe_average_precision(np.concatenate([item["labels"] for item in train_inputs]), stage1_train_scores)
    stage1_val_ap = safe_average_precision(val_input["labels"], stage1_val_scores)
    stage1_test_ap = safe_average_precision(test_input["labels"], stage1_test_scores)

    model.set_stage2_mode()
    stage2_best, stage2_history = run_stage("stage2_residual_gnn", args.stage2_epochs, args.residual_lr, args.early_stop_patience)

    if stage2_best["val_ap"] > stage1_best["val_ap"]:
        selected_stage = "stage2_residual_gnn"
        selected_best = stage2_best
    else:
        selected_stage = "stage1_heuristic_head"
        selected_best = stage1_best

    model.load_state_dict(selected_best["state"])
    model.use_residual = selected_stage == "stage2_residual_gnn"
    train_scores, val_scores, test_scores = evaluate_scores()

    train_labels = np.concatenate([item["labels"] for item in train_inputs])
    train_ap = safe_average_precision(train_labels, train_scores)
    val_ap = safe_average_precision(val_input["labels"], val_scores)
    test_ap = safe_average_precision(test_input["labels"], test_scores)
    best_epoch = int(selected_best["epoch"])
    best_val_ap = float(selected_best["val_ap"])
    history = stage1_history + stage2_history

    seed_dir = result_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "best_by_val_ap.pt")
    with (seed_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (seed_dir / "model_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_stage": selected_stage,
                "use_residual": selected_stage == "stage2_residual_gnn",
                "stage1_best_epoch": int(stage1_best["epoch"]),
                "stage1_best_val_ap": float(stage1_best["val_ap"]),
                "stage2_best_epoch": int(stage2_best["epoch"]),
                "stage2_best_val_ap": float(stage2_best["val_ap"]),
                "pos_weight": float(pos_weight_value),
            },
            handle,
            indent=2,
        )
    np.save(seed_dir / f"train_scores_seed_{seed}.npy", train_scores)
    np.save(seed_dir / f"val_scores_seed_{seed}.npy", val_scores)
    np.save(seed_dir / f"test_scores_seed_{seed}.npy", test_scores)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_ap": float(best_val_ap),
        "selected_stage": selected_stage,
        "heuristic_head_train_ap": float(stage1_train_ap),
        "heuristic_head_val_ap": float(stage1_val_ap),
        "heuristic_head_test_ap": float(stage1_test_ap),
        "train_ap": float(train_ap),
        "val_ap": float(val_ap),
        "test_ap": float(test_ap),
        "test_ap_before_final_retrain": float(test_ap),
        "train_scores": train_scores,
        "val_scores": val_scores,
        "test_scores": test_scores,
    }


# ---------------------------------------------------------------------------
# 11. save outputs
# ---------------------------------------------------------------------------


def save_pickle(path: Path, obj):
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_dual_graph_npz(path: Path, dual_graph):
    np.savez_compressed(
        path,
        edge_index=dual_graph["edge_index"],
        edge_pairs=dual_graph["edge_pairs"],
        edge_weight=dual_graph["edge_weight"],
        num_nodes=np.asarray([dual_graph["num_nodes"]], dtype=np.int64),
    )


def positive_rate(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(labels.mean())


def task_stats(task):
    labels = task["labels"]
    return {
        "name": task["name"],
        "context_size": task["context_size"],
        "future_size": task["future_size"],
        "observed_edges": len(task["state"]["edges"]),
        "observed_closed_triangles": len(task["state"]["triangles"]),
        "num_candidates": len(task["candidates"]),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "positive_rate": positive_rate(labels),
        "dual_nodes": len(task["candidates"]),
        "dual_edges": len(task["dual_graph"]["edge_pairs"]),
    }


def task_overlap(left, right) -> int:
    return len(set(left["candidates"]) & set(right["candidates"]))


def protocol_overlap_stats(train_tasks: Sequence[dict], val_task, test_task):
    stats = {
        "overlap_val_test": task_overlap(val_task, test_task),
    }
    for task in train_tasks:
        stats[f"overlap_{task['name']}_val"] = task_overlap(task, val_task)
        stats[f"overlap_{task['name']}_test"] = task_overlap(task, test_task)
    return stats


def print_protocol_stats(load_stats, simplices, contexts, train_tasks, val_task, test_task):
    vertices = set()
    for simplex in simplices:
        vertices.update(simplex["vertices"])

    print("\nProtocol statistics")
    print(f"Total timestamped simplices: {load_stats['raw_rows']}")
    print(f"Usable simplices after dedup/drop: {len(simplices)}")
    print(f"Singleton simplices retained for temporal split: {load_stats['singleton_simplices']}")
    print(f"Number of vertices: {len(vertices)}")
    for key in sorted(contexts, key=lambda x: int(x[1:])):
        print(f"{key} size: {len(contexts[key])}")

    display_tasks = [(f"Train task {idx + 1}", task) for idx, task in enumerate(train_tasks)]
    display_tasks.extend([("Validation task", val_task), ("Test task", test_task)])
    for title, task in display_tasks:
        stats = task_stats(task)
        print(f"\n{title}: {stats['name']}")
        print(f"  context simplices: {stats['context_size']}")
        print(f"  future simplices: {stats['future_size']}")
        print(f"  observed edges: {stats['observed_edges']}")
        print(f"  observed closed triangles: {stats['observed_closed_triangles']}")
        print(f"  open triangle candidates: {stats['num_candidates']}")
        print(f"  positives: {stats['num_positive']}")
        print(f"  negatives: {stats['num_negative']}")
        print(f"  positive_rate: {stats['positive_rate']:.6f}")
        print(f"  derived graph nodes / edges: {stats['dual_nodes']} / {stats['dual_edges']}")

    overlap = protocol_overlap_stats(train_tasks, val_task, test_task)
    print("\nCandidate overlap:")
    for name, value in overlap.items():
        print(f"  {name}: {value}")
    print("  Feature leakage check: no vertex/triangle ID one-hot, no future labels/features, no test candidates in loss.")


def warnings_for_protocol(args, train_tasks, val_task, test_task):
    warnings = []
    for task in list(train_tasks) + [val_task, test_task]:
        name = task["name"]
        labels = task["labels"]
        candidates = task["candidates"]
        dual = task["dual_graph"]
        if positive_rate(labels) == 0:
            warnings.append(f"{name} positive rate = 0")
        if len(candidates) > args.warn_open_triangles:
            warnings.append(f"{name} open triangles too many: {len(candidates)}")
        if len(dual["edge_pairs"]) > args.warn_dual_edges:
            warnings.append(f"{name} derived graph too dense: {len(dual['edge_pairs'])} edges")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return warnings


def save_score_distribution(path: Path, labels: np.ndarray, scores: np.ndarray):
    bins = np.linspace(0.0, 1.0, 21)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        positives = int(labels[mask].sum()) if count else 0
        rows.append(
            {
                "score_min": float(lo),
                "score_max": float(hi),
                "count": count,
                "positives": positives,
                "positive_rate": positives / count if count else 0.0,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_top_predictions(path: Path, candidates: Sequence[Triangle], labels: np.ndarray, scores: np.ndarray, k=100):
    order = np.argsort(-scores)[: min(k, len(scores))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "triangle", "label", "score"])
        for rank, idx in enumerate(order, start=1):
            writer.writerow([rank, " ".join(map(str, candidates[idx])), int(labels[idx]), float(scores[idx])])


def fit_block_scalers(train_tasks: Sequence[dict]):
    scalers = {}
    for block in ["local", "heuristic", "rw", "priority"]:
        train_block = np.vstack([task[f"{block}_features"] for task in train_tasks])
        if train_block.shape[1] == 0:
            scalers[block] = None
        else:
            scaler = StandardScaler()
            scaler.fit(train_block)
            scalers[block] = scaler
    return scalers


def transform_task_blocks(task: dict, scalers: Dict[str, StandardScaler]) -> dict:
    def transform(block):
        values = task[f"{block}_features"]
        scaler = scalers[block]
        if scaler is None:
            return values.astype(np.float32)
        return scaler.transform(values).astype(np.float32)

    return {
        "name": task["name"],
        "local": transform("local"),
        "heuristic": transform("heuristic"),
        "rw": transform("rw"),
        "priority": transform("priority"),
        "labels": task["labels"],
        "edge_index": task["edge_index"],
        "edge_weight": task["edge_weight"],
    }


def main():
    args = parse_args()
    if args.task_dim != 2:
        raise ValueError("Only --task_dim 2 triangle closure is supported in this first version.")
    if args.protocol == "multi_window":
        if len(args.split) != 4 or not (0.0 < args.split[0] < args.split[1] < args.split[2] < args.split[3] < 1.0):
            raise ValueError("--protocol multi_window requires --split 0.2 0.4 0.6 0.8.")
    else:
        if len(args.split) != 3 or not (0.0 < args.split[0] < args.split[1] < args.split[2] < 1.0):
            raise ValueError("--protocol rolling requires three split fractions, e.g. 0.4 0.6 0.8.")
    if args.global_encoder not in {"gnn", "node2vec"}:
        print(f"WARNING: --global_encoder {args.global_encoder!r} accepted but ignored; GNN context is used.")
    elif args.global_encoder == "node2vec":
        print("WARNING: node2vec is not used in this first version; supervised GNN learns context directly.")

    start = time.time()
    result_dir = Path(args.results)
    processed_dir = Path(args.processed_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw timestamped simplices...")
    simplices, load_stats = load_timestamped_simplices(
        args.nverts, args.simplices, args.times, event_order=args.event_order
    )
    print(f"Loaded {len(simplices)} simplices in {args.event_order} event order.")
    if load_stats["duplicate_vertex_simplices"]:
        print(f"WARNING: duplicate vertices were deduplicated in {load_stats['duplicate_vertex_simplices']} simplices.")
    if load_stats["singleton_simplices"]:
        print(f"Recorded {load_stats['singleton_simplices']} singleton simplices; they are retained for temporal split.")
    if load_stats["dropped_empty_simplices"]:
        print(f"WARNING: dropped {load_stats['dropped_empty_simplices']} empty simplices.")

    print("Building temporal splits...")
    contexts, futures = temporal_split(simplices, args.split)

    print("Building temporal closure tasks...")
    if args.protocol == "multi_window":
        train_tasks = [
            build_closure_task(
                contexts["S20"],
                futures["future_20_40"],
                "train_0_20_to_20_40",
                args.use_rw_struct_features,
                args.rw_steps,
                args,
            ),
            build_closure_task(
                contexts["S40"],
                futures["future_40_60"],
                "train_0_40_to_40_60",
                args.use_rw_struct_features,
                args.rw_steps,
                args,
            ),
        ]
        val_task = build_closure_task(
            contexts["S60"], futures["future_60_80"], "val_0_60_to_60_80", args.use_rw_struct_features, args.rw_steps, args
        )
        test_task = build_closure_task(
            contexts["S80"], futures["future_80_100"], "test_0_80_to_80_100", args.use_rw_struct_features, args.rw_steps, args
        )
    else:
        train_tasks = [
            build_closure_task(
                contexts["S40"],
                futures["future_40_60"],
                "train_0_40_to_40_60",
                args.use_rw_struct_features,
                args.rw_steps,
                args,
            )
        ]
        val_task = build_closure_task(
            contexts["S60"], futures["future_60_80"], "val_0_60_to_60_80", args.use_rw_struct_features, args.rw_steps, args
        )
        test_task = build_closure_task(
            contexts["S80"], futures["future_80_100"], "test_0_80_to_80_100", args.use_rw_struct_features, args.rw_steps, args
        )
    all_tasks = train_tasks + [val_task, test_task]
    print("Building aligned global feature blocks from train tasks...")
    global_feature_stats = attach_aligned_rw_and_priority_features(all_tasks, train_tasks, args)
    apply_information_mode_features(all_tasks, args)

    # Leakage audit and interpretable strength artifacts for the final cutoff.
    audit_rows = []
    for edge_id, (edge, record) in enumerate(sorted(test_task["boundary_strength"].items())):
        audit_rows.append(
            {
                "edge_id": edge_id,
                "u": edge[0],
                "v": edge[1],
                "cutoff": test_task["cutoff"],
                "count": record.count,
                "log_count": record.log_count,
                "decay": record.decay,
                "recency": record.recency,
                "last_time": record.last_time,
                "strictly_pre_cutoff": int(record.last_time < test_task["cutoff"]),
            }
        )
    write_rows(result_dir / "interaction_strength_audit.csv", audit_rows)
    candidate_rows = []
    for label, row in zip(test_task["labels"], test_task["strength_rows"]):
        candidate_rows.append(
            {"task": test_task["name"], **row, "label_for_audit_only": int(label)}
        )
    write_rows(result_dir / "candidate_strength_features.csv", candidate_rows)
    dataset_stats = dataset_strength_statistics(
        contexts["S80"], test_task["cutoff"], args.strength_decay_lambda
    )
    write_rows(
        result_dir / "dataset_strength_statistics.csv",
        [{"dataset": Path(args.nverts).name.replace("-nverts.txt", ""), **dataset_stats}],
    )
    overlap_stats = protocol_overlap_stats(train_tasks, val_task, test_task)
    heuristic_scores = all_heuristic_ap(all_tasks)
    print_heuristic_baselines(heuristic_scores)

    print("Saving protocol artifacts...")
    save_pickle(processed_dir / "simplices_sorted.pkl", simplices)
    if "S20" in contexts:
        save_pickle(processed_dir / "K20_state.pkl", train_tasks[0]["state"])
    save_pickle(processed_dir / "K40_state.pkl", train_tasks[-1]["state"])
    save_pickle(processed_dir / "K60_state.pkl", val_task["state"])
    save_pickle(processed_dir / "K80_state.pkl", test_task["state"])
    for task in all_tasks:
        prefix = task["name"]
        save_pickle(processed_dir / f"{prefix}_candidates.pkl", task["candidates"])
        np.save(processed_dir / f"{prefix}_labels.npy", task["labels"])
        save_dual_graph_npz(processed_dir / f"{prefix}_dual_graph.npz", task["dual_graph"])
        np.save(processed_dir / f"{prefix}_features.npy", task["features"])
        np.save(processed_dir / f"{prefix}_raw_features.npy", task["raw_features"])
        np.save(processed_dir / f"{prefix}_local_features.npy", task["local_features"])
        np.save(processed_dir / f"{prefix}_heuristic_raw_features.npy", task["heuristic_raw_features"])
        np.save(processed_dir / f"{prefix}_heuristic_features.npy", task["heuristic_features"])
        np.save(processed_dir / f"{prefix}_rw_struct_features.npy", task["rw_struct_features"])
        np.save(processed_dir / f"{prefix}_aligned_rw_features.npy", task["aligned_rw_features"])
        np.save(processed_dir / f"{prefix}_rw_features.npy", task["rw_features"])
        np.save(processed_dir / f"{prefix}_priority_features.npy", task["priority_features"])
    with (processed_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(FEATURE_NAMES, handle, indent=2)
    with (processed_dir / "feature_block_names.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "local": local_feature_names(),
                "heuristic_raw": HEURISTIC_FEATURE_NAMES,
                "heuristic": heuristic_block_names(),
                "rw": rw_feature_names(args.rw_steps) if args.use_rw_struct_features else [],
                "aligned_rw": [f"aligned_rw_{i}" for i in range(args.rw_dim)] if args.use_random_walk else [],
                "priority": [
                    "is_priority",
                    "dist_to_priority_min",
                    "dist_to_priority_mean_top3",
                    "priority_neighbor_count_1hop",
                    "priority_neighbor_count_2hop",
                    "random_walk_priority_score",
                ] + (
                    ["node_activity_min", "node_activity_mean", "node_activity_max", "node_activity_log_sum"]
                    if args.information_mode == "full"
                    else []
                ),
            },
            handle,
            indent=2,
        )
    with (processed_dir / "load_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(load_stats, handle, indent=2)
    with (processed_dir / "protocol_overlap.json").open("w", encoding="utf-8") as handle:
        json.dump(overlap_stats, handle, indent=2)
    with (processed_dir / "heuristic_ap.json").open("w", encoding="utf-8") as handle:
        json.dump(heuristic_scores, handle, indent=2)
    with (processed_dir / "global_feature_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(global_feature_stats, handle, indent=2)

    print_protocol_stats(
        load_stats,
        simplices,
        contexts,
        train_tasks,
        val_task,
        test_task,
    )
    protocol_warnings = warnings_for_protocol(args, train_tasks, val_task, test_task)

    if args.skip_training:
        print("\nTraining skipped by --skip_training.")
        summary = {
            "load_stats": load_stats,
            "warnings": protocol_warnings,
            "tasks": {
                "train": [task_stats(task) for task in train_tasks],
                "validation": task_stats(val_task),
                "test": task_stats(test_task),
            },
            "overlap": overlap_stats,
            "heuristic_ap": heuristic_scores,
            "global_feature_stats": global_feature_stats,
            "elapsed_sec": time.time() - start,
        }
        with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return

    if any(len(task["labels"]) == 0 for task in train_tasks) or len(val_task["labels"]) == 0 or len(test_task["labels"]) == 0:
        raise RuntimeError("No candidates available for training or testing.")
    for block in ["local_features", "heuristic_features", "rw_features", "priority_features"]:
        dims = [task[block].shape[1] for task in all_tasks]
        if len(set(dims)) != 1:
            raise RuntimeError(f"Train, validation, and test {block} dimensions do not match: {dims}")

    print("\nScaling feature blocks from train tasks only...")
    scalers = fit_block_scalers(train_tasks)
    train_inputs = [transform_task_blocks(task, scalers) for task in train_tasks]
    val_input = transform_task_blocks(val_task, scalers)
    test_input = transform_task_blocks(test_task, scalers)
    with (processed_dir / "feature_block_scalers.pkl").open("wb") as handle:
        pickle.dump(scalers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    final_train_inputs = None
    final_val_input = None
    final_test_input = None
    final_global_feature_stats = None
    if args.final_retrain:
        print("Rebuilding aligned global feature blocks from train+validation tasks for final retrain...")
        final_train_tasks = train_tasks + [val_task]
        final_global_feature_stats = attach_aligned_rw_and_priority_features(all_tasks, final_train_tasks, args)
        apply_information_mode_features(all_tasks, args)
        final_scalers = fit_block_scalers(final_train_tasks)
        final_train_inputs = [transform_task_blocks(task, final_scalers) for task in final_train_tasks]
        final_val_input = transform_task_blocks(val_task, final_scalers)
        final_test_input = transform_task_blocks(test_task, final_scalers)
        with (processed_dir / "feature_block_scalers_final_retrain.pkl").open("wb") as handle:
            pickle.dump(final_scalers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training Dual-Mapping GNN with multi-window temporal validation...")
    seed_results = []
    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        print(f"\nSeed {seed}")
        result = train_one_seed(
            seed,
            args,
            train_inputs,
            val_input,
            test_input,
            result_dir,
        )
        result["test_scores_before_final_retrain"] = result["test_scores"]
        result["test_ap_before_final_retrain"] = result["test_ap"]
        if args.final_retrain:
            final_result = train_one_seed(
                seed,
                args,
                final_train_inputs,
                final_val_input,
                final_test_input,
                result_dir / "final_retrain",
            )
            result["final_retrain"] = final_result
            result["test_scores_after_final_retrain"] = final_result["test_scores"]
            result["test_ap_after_final_retrain"] = final_result["test_ap"]
            result["test_scores"] = final_result["test_scores"]
            result["test_ap"] = final_result["test_ap"]
        seed_results.append(result)
        print(
            f"  best_epoch={result['best_epoch']} "
            f"train_AP={result['train_ap']:.6f} "
            f"val_AP={result['val_ap']:.6f} "
            f"test_AP_before_final_retrain={result['test_ap_before_final_retrain']:.6f} "
            f"test_AP={result['test_ap']:.6f}"
        )

    test_scores_before_stack = np.vstack([r["test_scores_before_final_retrain"] for r in seed_results])
    ensemble_scores_before = test_scores_before_stack.mean(axis=0)
    ensemble_ap_before = safe_average_precision(test_task["labels"], ensemble_scores_before)
    test_scores_stack = np.vstack([r["test_scores"] for r in seed_results])
    ensemble_scores = test_scores_stack.mean(axis=0)
    ensemble_ap = safe_average_precision(test_task["labels"], ensemble_scores)
    np.save(result_dir / "test_scores_ensemble_before_final_retrain.npy", ensemble_scores_before)
    np.save(result_dir / "test_scores_ensemble.npy", ensemble_scores)

    mean_ap_before = float(np.mean([r["test_ap_before_final_retrain"] for r in seed_results]))
    mean_ap = float(np.mean([r["test_ap"] for r in seed_results]))
    train_ap_mean = float(np.mean([r["train_ap"] for r in seed_results]))
    val_ap_mean = float(np.mean([r["val_ap"] for r in seed_results]))
    test_pr = positive_rate(test_task["labels"])
    test_ap_lift = ensemble_ap / test_pr if test_pr > 0 else 0.0

    train_labels_all = np.concatenate([task["labels"] for task in train_tasks])
    train_candidates_all = sum(len(task["candidates"]) for task in train_tasks)
    print("\nTrain task:")
    print(f"  candidates: {train_candidates_all}")
    print(f"  positives: {int(train_labels_all.sum())}")
    print(f"  positive_rate: {positive_rate(train_labels_all):.6f}")
    print(f"  train_AP: {train_ap_mean:.6f}")

    print("\nValidation:")
    print(f"  candidates: {len(val_task['candidates'])}")
    print(f"  positives: {int(val_task['labels'].sum())}")
    print(f"  positive_rate: {positive_rate(val_task['labels']):.6f}")
    print(f"  best_val_AP: {val_ap_mean:.6f}")

    print("\nTest:")
    print(f"  candidates: {len(test_task['candidates'])}")
    print(f"  positives: {int(test_task['labels'].sum())}")
    print(f"  positive_rate: {test_pr:.6f}")
    print(f"  test_AP_before_final_retrain: {ensemble_ap_before:.6f}")
    print(f"  test_AP_after_final_retrain: {ensemble_ap:.6f}")
    print(f"  test_AP_lift: {test_ap_lift:.6f}")

    print("\nSeeds:")
    for result in seed_results:
        print(
            f"  seed {result['seed']} val_AP: {result['val_ap']:.6f} "
            f"test_before: {result['test_ap_before_final_retrain']:.6f} "
            f"test_after: {result['test_ap']:.6f}"
        )
    print(f"  mean_AP_before_final_retrain: {mean_ap_before:.6f}")
    print(f"  mean_AP_after_final_retrain: {mean_ap:.6f}")
    print(f"  ensemble_AP: {ensemble_ap:.6f}")

    test_heuristics = heuristic_scores[test_task["name"]]
    print("\nHeuristic baseline on test:")
    for name in [
        "CN",
        "Jaccard",
        "SCN",
        "SRA",
        "SAA",
        "SPA",
        "wSCN",
        "wSRA",
        "wSAA",
        "wSPA",
        "face_support_sum",
        "edge_degree_sum",
        "rank_fusion",
    ]:
        print(f"  {name}_AP: {test_heuristics[name]:.6f}")
    best_heuristic_ap = max(test_heuristics.values())
    model_minus_best_heuristic = ensemble_ap - best_heuristic_ap
    print("\nComparison:")
    print(f"  best_heuristic_AP: {best_heuristic_ap:.6f}")
    print(f"  model_minus_best_heuristic: {model_minus_best_heuristic:.6f}")
    if ensemble_ap < best_heuristic_ap:
        warning = "learned model underperforms heuristic baseline; report heuristic baseline as stronger model or reduce GNN residual weight."
        print(f"WARNING: {warning}")
        protocol_warnings.append(warning)

    best_val_scores = max(seed_results, key=lambda r: r["val_ap"])["val_scores"]
    distribution_shift = val_ap_mean >= 0.5 and ensemble_ap < max(0.2, 0.5 * val_ap_mean)
    if distribution_shift:
        print("WARNING: validation AP is high while test AP is low; saving score diagnostics.")
        save_score_distribution(result_dir / "val_score_distribution.csv", val_task["labels"], best_val_scores)
        save_score_distribution(result_dir / "test_score_distribution.csv", test_task["labels"], ensemble_scores)
        save_top_predictions(
            result_dir / "top_100_test_predictions.csv",
            test_task["candidates"],
            test_task["labels"],
            ensemble_scores,
        )

    summary = {
        "run_schema_version": 2,
        "load_stats": load_stats,
        "interaction_strength": {
            "dual_edge_weight": args.dual_edge_weight,
            "shared_face_reduce": args.dual_shared_face_reduce,
            "information_mode": args.information_mode,
            "decay_lambda": args.strength_decay_lambda,
            "strict_pre_cutoff": True,
        },
        "warnings": protocol_warnings,
        "tasks": {
            "train": [task_stats(task) for task in train_tasks],
            "validation": task_stats(val_task),
            "test": task_stats(test_task),
        },
        "overlap": overlap_stats,
        "global_feature_stats": global_feature_stats,
        "final_global_feature_stats": final_global_feature_stats,
        "heuristic_ap": heuristic_scores,
        "heuristic_baseline": {
            "test_CN_AP": test_heuristics["CN"],
            "test_Jaccard_AP": test_heuristics["Jaccard"],
            "test_SCN_AP": test_heuristics["SCN"],
            "test_SRA_AP": test_heuristics["SRA"],
            "test_SAA_AP": test_heuristics["SAA"],
            "test_SPA_AP": test_heuristics["SPA"],
            "test_wSCN_AP": test_heuristics["wSCN"],
            "test_wSRA_AP": test_heuristics["wSRA"],
            "test_wSAA_AP": test_heuristics["wSAA"],
            "test_wSPA_AP": test_heuristics["wSPA"],
            "test_face_support_sum_AP": test_heuristics["face_support_sum"],
            "test_edge_degree_sum_AP": test_heuristics["edge_degree_sum"],
            "test_rank_fusion_AP": test_heuristics["rank_fusion"],
            "best_heuristic_AP": best_heuristic_ap,
        },
        "model": {
            "heuristic_head_AP": float(np.mean([r["heuristic_head_test_ap"] for r in seed_results])),
            "residual_gnn_AP_before_final_retrain": ensemble_ap_before,
            "residual_gnn_AP_after_final_retrain": ensemble_ap,
            "ensemble_AP": ensemble_ap,
            "AP_lift": test_ap_lift,
            "model_minus_best_heuristic": model_minus_best_heuristic,
        },
        "train": {
            "AP_mean_over_seeds": train_ap_mean,
        },
        "validation": {
            "AP_mean_over_seeds": val_ap_mean,
        },
        "test": {
            "positive_rate": test_pr,
            "AP_ensemble_before_final_retrain": ensemble_ap_before,
            "AP_ensemble": ensemble_ap,
            "AP_lift": test_ap_lift,
        },
        "seeds": [
            {
                "seed": int(r["seed"]),
                "best_epoch": int(r["best_epoch"]),
                "selected_stage": r["selected_stage"],
                "heuristic_head_AP": float(r["heuristic_head_test_ap"]),
                "train_AP": float(r["train_ap"]),
                "val_AP": float(r["val_ap"]),
                "test_AP_before_final_retrain": float(r["test_ap_before_final_retrain"]),
                "test_AP": float(r["test_ap"]),
            }
            for r in seed_results
        ],
        "mean_AP_before_final_retrain": mean_ap_before,
        "mean_AP": mean_ap,
        "ensemble_AP": ensemble_ap,
        "elapsed_sec": time.time() - start,
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
