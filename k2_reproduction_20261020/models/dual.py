"""Original structural features and GraphSAGE classes (experiment residual backend)."""
from __future__ import annotations
import math
import random
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from models.interaction_strength import strength_value

Triangle = Tuple[int, int, int]

Edge = Tuple[int, int]

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

def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
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
