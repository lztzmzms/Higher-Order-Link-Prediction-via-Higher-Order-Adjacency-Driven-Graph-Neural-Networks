import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from sklearn import metrics as met
from scipy.stats import rankdata
from tqdm import tqdm

from mp.models import SparseSIN
from root_dir import ROOT_DIR
from simplicial_complex.data_loading import DataLoader
from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi


DEFAULT_DATASETS = [
    "contact-high-school",
    "contact-primary-school",
    "email-Enron",
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-ask-ubuntu",
]


class SkipGramNS(nn.Module):
    """Dual-Mapping aligned-union random-walk embedding model."""

    def __init__(self, num_nodes, dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(num_nodes, dim)
        self.out_embeddings = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)

    def forward(self, centers, contexts, negative_samples):
        center_emb = self.in_embeddings(centers)
        context_emb = self.out_embeddings(contexts)
        positive_scores = (center_emb * context_emb).sum(dim=-1)
        positive_loss = -F.logsigmoid(positive_scores).mean()

        negative_nodes = torch.randint(
            0,
            self.in_embeddings.num_embeddings,
            (centers.numel(), negative_samples),
            device=centers.device,
        )
        negative_emb = self.out_embeddings(negative_nodes)
        negative_scores = torch.bmm(
            negative_emb.neg(),
            center_emb.unsqueeze(-1),
        ).squeeze(-1)
        negative_loss = -F.logsigmoid(negative_scores).mean()
        return positive_loss + negative_loss


class SparseSINWithDualRW(nn.Module):
    """SparseSIN encoder fused with an aligned Dual-Mapping RW vector."""

    def __init__(self, args):
        super().__init__()
        self.fusion_mode = args.rw_fusion
        self.sin = SparseSIN(
            num_input_features=args.num_input_features,
            num_layers=args.num_layers,
            num_classes=2,
            hidden=args.hidden,
            max_dim=args.max_dim,
            dropout_rate=args.dropout_rate,
            readout=args.readout,
            final_readout=args.final_readout,
        )
        sin_dim = 2 * args.hidden
        self.rw_scale = float(args.rw_scale)
        rw_input_norm = (
            nn.LayerNorm(args.rw_dim)
            if args.rw_input_normalization == "layer_norm"
            else nn.Identity()
        )
        self.rw_projection = nn.Sequential(
            rw_input_norm,
            nn.Linear(args.rw_dim, args.rw_proj_dim),
            nn.ReLU(),
            nn.Dropout(args.rw_dropout),
        )
        fusion_dim = args.rw_proj_dim if self.fusion_mode == "rw_only" else sin_dim + args.rw_proj_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, args.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(args.rw_dropout),
            nn.Linear(args.fusion_hidden, 2),
        )

    def forward(self, data, return_logits=False):
        rw_feature = data.cochains[0].rw_feature
        rw_embedding = self.rw_projection(rw_feature) * self.rw_scale
        if self.fusion_mode == "rw_only":
            fused = rw_embedding
        else:
            sin_embedding = self.sin(data, return_embedding=True)
            fused = torch.cat([sin_embedding, rw_embedding], dim=1)
        logits = self.classifier(fused)
        return logits if return_logits else F.softmax(logits, dim=1)


def combinations_to_labels(distance_sum):
    offset = 0
    triples = []
    labels = []

    def generate_partitions(total, result, count, output):
        if total < 0:
            return
        if total == 0:
            part = result[:count]
            if len(part) == 3:
                output.append(part)
            return
        value = 1 if count == 0 else result[count - 1]
        while value <= total:
            result[count] = value
            generate_partitions(total - value, result, count + 1, output)
            value += 1

    for radius in range(3, distance_sum + 1):
        partitions = []
        generate_partitions(radius, [None] * radius, 0, partitions)
        for index, part in enumerate(partitions):
            triples.append(part)
            labels.append(1 + offset + index + 1)
        offset += len(partitions)
    return triples, labels


def remove_target_link(k_hop_subgraph, tri):
    graph = nx.Graph(k_hop_subgraph)
    graph.remove_edges_from(list(combinations(tri, 2)))
    return graph


def trnl_node_labeling(k_hop_subgraph, tri):
    node_1, node_2, node_3 = tuple(tri)
    graph = remove_target_link(k_hop_subgraph, tri)
    node_list = list(graph.nodes)
    index_1 = node_list.index(node_1)
    index_2 = node_list.index(node_2)
    index_3 = node_list.index(node_3)

    dist2node_1 = [nx.shortest_path_length(graph, source=node_1, target=node) for node in node_list]
    dist2node_2 = [nx.shortest_path_length(graph, source=node_2, target=node) for node in node_list]
    dist2node_3 = [nx.shortest_path_length(graph, source=node_3, target=node) for node in node_list]
    dist2node_1[index_2] = 0
    dist2node_1[index_3] = 0
    dist2node_2[index_1] = 0
    dist2node_2[index_3] = 0
    dist2node_3[index_1] = 0
    dist2node_3[index_2] = 0

    labels = [None] * len(node_list)
    for index in range(len(node_list)):
        distances = sorted([dist2node_1[index], dist2node_2[index], dist2node_3[index]])
        triples, trnl_labels = combinations_to_labels(sum(distances))
        labels[index] = [trnl_labels[triples.index(distances)]] if distances in triples else [0]
    labels[index_1] = [1]
    labels[index_2] = [1]
    labels[index_3] = [1]
    return labels


def extract_k_hop_subgraph(
    graph,
    num_hops,
    tri,
    sample_ratio=1.0,
    max_nodes_per_step=None,
    enforce_node_cap=False,
):
    visited = set(tri)
    current_level = set(tri)
    for _ in range(1, num_hops + 1):
        next_level = set()
        for node in current_level:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited and neighbor not in next_level:
                    next_level.add(neighbor)
        if sample_ratio < 1.0:
            next_level = set(random.sample(list(next_level), int(sample_ratio * len(next_level))))
        if max_nodes_per_step is not None and len(next_level) > max_nodes_per_step:
            next_level = set(random.sample(list(next_level), max_nodes_per_step))
        if not next_level:
            break
        if enforce_node_cap:
            visited.update(next_level)
        else:
            for node in current_level:
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
        current_level = next_level
    return nx.subgraph(graph, visited)


def construct_labeled_networkx_graph(k_hop_subgraph, old_simplices, tri, positives, labelling="TRNL"):
    if labelling != "TRNL":
        raise ValueError(f"Only TRNL is supported by this runner, got {labelling}")
    node_list = list(k_hop_subgraph.nodes)
    edge_list = list(k_hop_subgraph.edges)
    embeddings = trnl_node_labeling(k_hop_subgraph, tri)
    nx.set_node_attributes(
        k_hop_subgraph,
        {node_list[index]: {"feature": embeddings[index]} for index in range(len(embeddings))},
    )

    simplices = []
    node_set = set(node_list)
    for old_simplex in old_simplices:
        if len(old_simplex) >= 3:
            simplex = sorted(set(old_simplex) & node_set)
            if len(simplex) >= 3:
                simplices.append(tuple(simplex))
        simplices = list(set(simplices))
    simplices = [list(simplex) for simplex in simplices]
    for edge in edge_list:
        simplices.insert(0, sorted(list(edge)))

    original = simplices.copy()
    for i in range(len(original)):
        for simplex in original[i + 1 :]:
            if set(original[i]) < set(simplex):
                simplices[i] = [-1]
            if set(original[i]) > set(simplex):
                simplices[original.index(simplex)] = [-1]
    simplices = [simplex for simplex in simplices if simplex != [-1]]
    simplices.sort(key=len)
    return nx.Graph(k_hop_subgraph, y=1 if tri in positives else 0, simplices=simplices)


def to_pyg_data(k_hop_subgraph):
    node_list = list(k_hop_subgraph.nodes)
    node_feature = [k_hop_subgraph.nodes[node]["feature"] for node in node_list]
    directed = k_hop_subgraph.to_directed() if not nx.is_directed(k_hop_subgraph) else k_hop_subgraph
    graph = nx.convert_node_labels_to_integers(directed)
    nodes_new = list(graph.nodes)
    edges_new = list(graph.edges)
    edge_index = torch.tensor(edges_new, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_feature, dtype=torch.float)
    simplices = [list(simplex) for simplex in k_hop_subgraph.graph["simplices"]]
    for simplex in simplices:
        for i in range(len(simplex)):
            simplex[i] = nodes_new[node_list.index(simplex[i])]
        simplex.sort()
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([k_hop_subgraph.graph["y"]], dtype=torch.long),
        node_ids=torch.tensor(nodes_new),
        num_nodes=len(node_list),
        simplices=simplices,
    )


def read_int_lines(path):
    values = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                values.append(int(text))
    return values


def get_timestamped_simplices(root_nverts, root_timestamp, root_simplices):
    nverts = read_int_lines(root_nverts)
    timestamps = read_int_lines(root_timestamp)
    flat_simplices = read_int_lines(root_simplices)
    split_simplices = []
    offset = 0
    for count in nverts:
        simplex = flat_simplices[offset : offset + count]
        simplex.sort()
        split_simplices.append(simplex)
        offset += count
    return timestamps, split_simplices


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def read_triangles(path):
    triangles = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            tri = tuple(sorted((int(parts[0]), int(parts[1]), int(parts[2]))))
            triangles.append(tri)
    return triangles


def node_label_transform(graph):
    mapping = {node: int(node) for node in graph.nodes()}
    return nx.relabel_nodes(graph, mapping, copy=False)


def sample_uniform(items, cap, seed):
    items = list(items)
    if cap is None or len(items) <= cap:
        return items
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(items), size=cap, replace=False)
    return [items[int(i)] for i in indexes]


def sample_stratified(candidates, positives, cap, positive_fraction, seed):
    candidates = list(candidates)
    if cap is None or len(candidates) <= cap:
        return candidates
    rng = np.random.default_rng(seed)
    positive_items = [item for item in candidates if item in positives]
    negative_items = [item for item in candidates if item not in positives]
    desired_pos = min(len(positive_items), int(round(cap * positive_fraction)))
    desired_neg = min(len(negative_items), cap - desired_pos)
    if desired_pos + desired_neg < cap:
        remaining_pos = len(positive_items) - desired_pos
        extra_pos = min(remaining_pos, cap - desired_pos - desired_neg)
        desired_pos += extra_pos
    chosen = []
    if desired_pos:
        chosen.extend([positive_items[int(i)] for i in rng.choice(len(positive_items), size=desired_pos, replace=False)])
    if desired_neg:
        chosen.extend([negative_items[int(i)] for i in rng.choice(len(negative_items), size=desired_neg, replace=False)])
    rng.shuffle(chosen)
    return chosen


def effective_sampling_seed(args):
    return args.seed if args.sampling_seed is None else int(args.sampling_seed)


def raw_paths(dataset):
    raw_dir = Path(ROOT_DIR) / "dataset" / "raw_data" / dataset
    return (
        raw_dir / f"{dataset}-nverts.txt",
        raw_dir / f"{dataset}-times.txt",
        raw_dir / f"{dataset}-simplices.txt",
    )


def get_old_simplices(dataset, split):
    nverts_path, times_path, simplices_path = raw_paths(dataset)
    _, data_simplices = get_timestamped_simplices(
        str(nverts_path), str(times_path), str(simplices_path)
    )
    if split == "early":
        fraction = 0.4
    elif split == "train":
        fraction = 0.6
    else:
        fraction = 0.8
    end = int(fraction * len(data_simplices))
    return data_simplices[:end]


def resolve_maxsimplex_feature_path(dataset, split, feature_dir):
    """Resolve TSV-formatted maximal-simplex features, regardless of suffix."""
    feature_dir = Path(feature_dir)
    stem = f"{dataset}_{split}_edge_maxsimplex_features"
    candidates = [
        feature_dir / f"{stem}{suffix}"
        for suffix in (".tsv", ".txt", ".bin")
    ]
    for path in candidates:
        if path.is_file():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Maximal-simplex prior feature file not found. Tried: {tried}"
    )


def build_temporal_activity_map(dataset, split, args):
    """Context-only activity prior for each open-triangle candidate.

    This uses no future labels.  It measures whether all three vertices were
    active in the most recent portion of the observed event stream, then
    converts the score to a within-task rank so it transfers across windows.
    """
    if split == "early":
        context_fraction = 0.4
    elif split == "train":
        context_fraction = 0.6
    else:
        context_fraction = 0.8
    _, data_simplices = get_timestamped_simplices(
        str(raw_paths(dataset)[0]),
        str(raw_paths(dataset)[1]),
        str(raw_paths(dataset)[2]),
    )
    context_end = int(context_fraction * len(data_simplices))
    recent_count = max(
        1, int(round(context_end * args.temporal_activity_recent_fraction))
    )
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    for simplex in data_simplices[max(0, context_end - recent_count) : context_end]:
        vertices = sorted(set(map(int, simplex)))
        for node in vertices:
            node_counts[node] += 1
        for edge in combinations(vertices, 2):
            edge_counts[tuple(sorted(edge))] += 1

    candidates = read_triangles(split_paths(dataset, split, args)["open"])
    raw_scores = []
    for triangle in candidates:
        if args.temporal_activity_entity == "node":
            counts = np.asarray(
                [node_counts[int(node)] for node in triangle], dtype=np.float64
            )
        elif args.temporal_activity_entity == "edge":
            counts = np.asarray(
                [
                    edge_counts[tuple(sorted(edge))]
                    for edge in combinations(triangle, 2)
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Unsupported temporal activity entity: {args.temporal_activity_entity}"
            )
        if args.temporal_activity_aggregation == "min":
            score = float(counts.min())
        elif args.temporal_activity_aggregation == "product":
            score = float(np.log1p(counts).sum())
        elif args.temporal_activity_aggregation == "mean":
            score = float(counts.mean())
        else:
            raise ValueError(
                "Unsupported temporal activity aggregation: "
                f"{args.temporal_activity_aggregation}"
            )
        raw_scores.append(score)
    activity_ranks = rankdata(np.asarray(raw_scores), method="average") / max(
        len(raw_scores), 1
    )
    if args.maxsimplex_prior_weight > 0:
        if split == "early":
            raise ValueError(
                "Maximal-simplex prior requires an explicit early feature file"
            )
        feature_path = resolve_maxsimplex_feature_path(
            dataset,
            split,
            args.maxsimplex_feature_dir,
        )
        print(f"Loading maximal-simplex prior features: {feature_path}", flush=True)
        values = {}
        with feature_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if args.maxsimplex_prior_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Column {args.maxsimplex_prior_column!r} not found in "
                    f"{feature_path}"
                )
            for row in reader:
                triangle = tuple(
                    sorted(map(int, row["triplet"].replace("|", " ").split()))
                )
                values[triangle] = float(row[args.maxsimplex_prior_column])
        maximal_raw = np.asarray([values[triangle] for triangle in candidates])
        maximal_ranks = rankdata(maximal_raw, method="average") / max(
            len(candidates), 1
        )
        weight = float(args.maxsimplex_prior_weight)
        activity_ranks = (activity_ranks + weight * maximal_ranks) / (1.0 + weight)
    return {
        triangle: np.float32(activity_ranks[index])
        for index, triangle in enumerate(candidates)
    }


def split_paths(dataset, split, args=None):
    if split == "early":
        if args is None or not getattr(args, "temporal_early_dir", None):
            raise ValueError("Temporal early protocol has not been prepared")
        split_dir = Path(args.temporal_early_dir)
        return {
            "graph": split_dir / f"{dataset}_G_early.graphml",
            "open": split_dir / f"{dataset}_openTriangles_early.txt",
            "positive": split_dir / f"{dataset}_new_closures_early.txt",
        }
    split_dir = Path(ROOT_DIR) / "dataset" / split
    return {
        "graph": split_dir / f"{dataset}_G_{split}.graphml",
        "open": split_dir / f"{dataset}_openTriangles_{split}.txt",
        "positive": split_dir / f"{dataset}_new_closures_{split}.txt",
    }


def dataset_source_fingerprint(dataset, split, args=None):
    paths = list(split_paths(dataset, split, args).values()) + list(raw_paths(dataset))
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(str(path.name).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def write_triangle_lines(path, triangles):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for triangle in triangles:
            handle.write(f"{triangle[0]} {triangle[1]} {triangle[2]}\n")


def ensure_temporal_early_protocol(dataset, args):
    raw_fingerprint = hashlib.sha256()
    for path in raw_paths(dataset):
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                raw_fingerprint.update(chunk)
    digest = raw_fingerprint.hexdigest()[:16]
    base_cache = Path(args.cache_dir) if args.cache_dir else Path(args.output_dir) / "cache"
    output_dir = base_cache / "temporal_protocol" / f"{dataset}_{digest}"
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / f"{dataset}_G_early.graphml"
    open_path = output_dir / f"{dataset}_openTriangles_early.txt"
    positive_path = output_dir / f"{dataset}_new_closures_early.txt"
    close_path = output_dir / f"{dataset}_closeTriangles_early.txt"
    expected = (graph_path, open_path, positive_path, close_path)
    if all(path.exists() for path in expected):
        args.temporal_early_dir = str(output_dir)
        return

    _, simplices = get_timestamped_simplices(
        str(raw_paths(dataset)[0]),
        str(raw_paths(dataset)[1]),
        str(raw_paths(dataset)[2]),
    )
    context_end = int(0.4 * len(simplices))
    future_end = int(0.6 * len(simplices))
    context = simplices[:context_end]
    future = simplices[context_end:future_end]

    nodes = set()
    edges = set()
    closed = set()
    for simplex in context:
        vertices = tuple(sorted(set(map(int, simplex))))
        nodes.update(vertices)
        edges.update(tuple(sorted(edge)) for edge in combinations(vertices, 2))
        closed.update(tuple(sorted(triangle)) for triangle in combinations(vertices, 3))

    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(sorted(edges))
    adjacency = {node: set(graph.neighbors(node)) for node in graph.nodes}
    candidates = []
    for left in sorted(adjacency):
        for middle in sorted(node for node in adjacency[left] if node > left):
            for right in sorted(node for node in adjacency[left] & adjacency[middle] if node > middle):
                triangle = (left, middle, right)
                if triangle not in closed:
                    candidates.append(triangle)

    future_closed = set()
    for simplex in future:
        vertices = tuple(sorted(set(map(int, simplex))))
        future_closed.update(
            tuple(sorted(triangle))
            for triangle in combinations(vertices, 3)
        )
    positives = sorted(set(candidates) & future_closed)

    nx.write_graphml(graph, graph_path)
    write_triangle_lines(open_path, sorted(candidates))
    write_triangle_lines(positive_path, positives)
    write_triangle_lines(close_path, sorted(closed))
    args.temporal_early_dir = str(output_dir)
    print(
        f"Temporal early protocol: candidates={len(candidates)} "
        f"positives={len(positives)} baseline={len(positives) / max(len(candidates), 1):.6f}",
        flush=True,
    )


def dual_edges_for_candidates(candidates, global_index):
    face_to_nodes = defaultdict(list)
    for triangle in candidates:
        node_id = global_index[triangle]
        for face in combinations(triangle, 2):
            face_to_nodes[tuple(sorted(face))].append(node_id)

    edges = set()
    for node_ids in face_to_nodes.values():
        for left, right in combinations(sorted(set(node_ids)), 2):
            if left != right:
                edges.add((left, right))
    return edges


def build_csr_adj(edges, num_nodes):
    neighbors = [set() for _ in range(num_nodes)]
    for left, right in edges:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    for node in range(num_nodes):
        if not neighbors[node]:
            neighbors[node].add(node)

    degrees = np.asarray([len(values) for values in neighbors], dtype=np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(degrees)
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    offset = 0
    for values in neighbors:
        ordered = np.asarray(sorted(values), dtype=np.int64)
        indices[offset : offset + len(ordered)] = ordered
        offset += len(ordered)
    return indptr, indices


def generate_walks_csr_chunk(indptr, indices, starts, walk_length, rng):
    walks = np.empty((len(starts), walk_length + 1), dtype=np.int64)
    walks[:, 0] = starts
    current = starts.copy()
    for step in range(1, walk_length + 1):
        next_nodes = np.empty_like(current)
        for index, node in enumerate(current):
            begin = indptr[node]
            end = indptr[node + 1]
            next_nodes[index] = indices[rng.integers(begin, end)]
        walks[:, step] = next_nodes
        current = next_nodes
    return walks


def iter_pairs_from_walks(walks, window_size, batch_size):
    centers = []
    contexts = []
    for walk in walks:
        for index, center in enumerate(walk):
            left = max(0, index - window_size)
            right = min(len(walk), index + window_size + 1)
            for context_index in range(left, right):
                if index == context_index:
                    continue
                centers.append(int(center))
                contexts.append(int(walk[context_index]))
                if len(centers) >= batch_size:
                    yield (
                        np.asarray(centers, dtype=np.int64),
                        np.asarray(contexts, dtype=np.int64),
                    )
                    centers.clear()
                    contexts.clear()
    if centers:
        yield (
            np.asarray(centers, dtype=np.int64),
            np.asarray(contexts, dtype=np.int64),
        )


def train_aligned_dual_rw(
    indptr,
    indices,
    num_nodes,
    priority_mask,
    args,
    device,
):
    seed_everything(args.rw_seed)
    rng = np.random.default_rng(args.rw_seed)
    model = SkipGramNS(num_nodes, args.rw_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.rw_lr)
    nodes = np.arange(num_nodes, dtype=np.int64)
    start_nodes = nodes
    priority_indices = np.flatnonzero(priority_mask)
    if priority_indices.size and args.rw_priority_multiplier > 1:
        start_nodes = np.concatenate(
            [
                nodes,
                np.tile(priority_indices, args.rw_priority_multiplier - 1),
            ]
        )
    print(
        f"Dual-RW aligned union: nodes={num_nodes} edges={len(indices) // 2} "
        f"priority_nodes={len(priority_indices)} multiplier={args.rw_priority_multiplier}",
        flush=True,
    )

    model.train()
    for epoch in range(1, args.rw_epochs + 1):
        total_loss = 0.0
        total_batches = 0
        for _ in range(args.rw_walks_per_node):
            shuffled = rng.permutation(start_nodes)
            for start in range(0, len(shuffled), args.rw_chunk_starts):
                starts = shuffled[start : start + args.rw_chunk_starts]
                walks = generate_walks_csr_chunk(
                    indptr,
                    indices,
                    starts,
                    args.rw_walk_length,
                    rng,
                )
                for center_np, context_np in iter_pairs_from_walks(
                    walks,
                    args.rw_window_size,
                    args.rw_batch_size,
                ):
                    centers = torch.from_numpy(center_np).long().to(device)
                    contexts = torch.from_numpy(context_np).long().to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(centers, contexts, args.rw_negative_samples)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    total_batches += 1
        print(
            f"Dual-RW epoch={epoch}/{args.rw_epochs} "
            f"loss={total_loss / max(total_batches, 1):.6f}",
            flush=True,
        )

    embedding = model.in_embeddings.weight.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / np.maximum(norms, 1e-12)


def dual_rw_cache_path(dataset, args):
    if args.rw_cache_dir:
        directory = Path(args.rw_cache_dir)
    elif args.cache_dir:
        directory = Path(args.cache_dir) / "dual_rw"
    else:
        directory = Path(args.output_dir) / "dual_rw"
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": 2,
        "dataset": dataset,
        "train_source": dataset_source_fingerprint(dataset, "train", args),
        "test_source": dataset_source_fingerprint(dataset, "test", args),
        "temporal_validation": args.temporal_validation,
        "early_source": (
            dataset_source_fingerprint(dataset, "early", args)
            if args.temporal_validation
            else None
        ),
        "rw_seed": args.rw_seed,
        "rw_dim": args.rw_dim,
        "walk_length": args.rw_walk_length,
        "walks_per_node": args.rw_walks_per_node,
        "window_size": args.rw_window_size,
        "epochs": args.rw_epochs,
        "negative_samples": args.rw_negative_samples,
        "priority_multiplier": args.rw_priority_multiplier,
        "rw_lr": args.rw_lr,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return directory / f"aligned_dual_rw_{dataset}_{digest}.npz", payload


def load_or_build_aligned_dual_rw(dataset, args, device):
    if args.rw_reference_cache:
        path = Path(args.rw_reference_cache)
        expected_payload = None
    else:
        path, expected_payload = dual_rw_cache_path(dataset, args)

    if path.exists() and not args.rebuild_rw_cache:
        with np.load(path, allow_pickle=False) as cached:
            triangles = np.asarray(cached["triangles"], dtype=np.int64)
            embedding = np.asarray(cached["embedding"], dtype=np.float32)
            payload = json.loads(str(np.asarray(cached["payload"]).item()))
        if expected_payload is not None and payload != expected_payload:
            raise ValueError(f"Dual-RW cache configuration mismatch: {path}")
        if triangles.shape != (embedding.shape[0], 3) or embedding.shape[1] != args.rw_dim:
            raise ValueError(
                f"Dual-RW cache shape mismatch: triangles={triangles.shape}, "
                f"embedding={embedding.shape}"
            )
        print(f"Loading aligned Dual-Mapping RW cache: {path}", flush=True)
    else:
        if args.rw_reference_cache:
            raise FileNotFoundError(f"Dual-RW reference cache not found: {path}")
        train_candidates = read_triangles(split_paths(dataset, "train", args)["open"])
        test_candidates = read_triangles(split_paths(dataset, "test", args)["open"])
        early_candidates = (
            read_triangles(split_paths(dataset, "early", args)["open"])
            if args.temporal_validation
            else []
        )
        all_triangles = sorted(set(early_candidates) | set(train_candidates) | set(test_candidates))
        global_index = {triangle: index for index, triangle in enumerate(all_triangles)}
        edges = dual_edges_for_candidates(train_candidates, global_index)
        edges.update(dual_edges_for_candidates(test_candidates, global_index))
        if early_candidates:
            edges.update(dual_edges_for_candidates(early_candidates, global_index))
        indptr, indices = build_csr_adj(sorted(edges), len(all_triangles))

        priority_split = "early" if args.temporal_validation else "train"
        train_positives = set(read_triangles(split_paths(dataset, priority_split, args)["positive"]))
        priority_mask = np.asarray(
            [triangle in train_positives for triangle in all_triangles],
            dtype=bool,
        )
        embedding = train_aligned_dual_rw(
            indptr,
            indices,
            len(all_triangles),
            priority_mask,
            args,
            device,
        )
        triangles = np.asarray(all_triangles, dtype=np.int64)
        np.savez_compressed(
            path,
            triangles=triangles,
            embedding=embedding,
            payload=json.dumps(expected_payload, sort_keys=True),
        )
        print(f"Saved aligned Dual-Mapping RW cache: {path}", flush=True)

    if not np.isfinite(embedding).all():
        raise ValueError(f"Dual-RW cache contains NaN or Inf: {path}")
    feature_map = {
        tuple(map(int, triangle)): embedding[index]
        for index, triangle in enumerate(triangles)
    }
    return feature_map, str(path)


def load_dual_mapping_full_rw(dataset, args):
    """Load the exact full-graph Dual-Mapping RW vectors for train/test.

    The original Dual-Mapping graph contains historical closed triangles as
    well as the open triangle candidates.  Its cache stores separate train
    and test coordinate matrices.  We align those matrices to triangle IDs
    from the corresponding gpickle files and expose only the candidate rows
    needed by SparseSIN.
    """
    if not args.rw_reference_cache:
        raise ValueError(
            "--rw-source dual_mapping_full requires --rw-reference-cache "
            "containing train_rw and test_rw"
        )
    path = Path(args.rw_reference_cache)
    if not path.exists():
        raise FileNotFoundError(f"Dual-Mapping full RW cache not found: {path}")
    with np.load(path, allow_pickle=False) as cached:
        if "train_rw" not in cached or "test_rw" not in cached:
            raise ValueError(
                f"Dual-Mapping full RW cache must contain train_rw and test_rw: {path}"
            )
        train_rw = np.asarray(cached["train_rw"], dtype=np.float32)
        test_rw = np.asarray(cached["test_rw"], dtype=np.float32)

    gpickle_dir = Path(ROOT_DIR) / "gpickle" / dataset
    train_graph_path = (
        gpickle_dir / f"classified_network_{dataset}_train_1.gpickle"
    )
    test_graph_path = (
        gpickle_dir / f"classified_network_{dataset}_all_1.gpickle"
    )
    for graph_path in (train_graph_path, test_graph_path):
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Full Dual-Mapping graph required for RW alignment: {graph_path}"
            )

    import pickle

    with train_graph_path.open("rb") as handle:
        train_graph = pickle.load(handle)
    with test_graph_path.open("rb") as handle:
        test_graph = pickle.load(handle)
    if train_rw.shape != (train_graph.number_of_nodes(), args.rw_dim):
        raise ValueError(
            "Train full-RW shape mismatch: "
            f"cache={train_rw.shape}, graph_nodes={train_graph.number_of_nodes()}, "
            f"rw_dim={args.rw_dim}"
        )
    if test_rw.shape != (test_graph.number_of_nodes(), args.rw_dim):
        raise ValueError(
            "Test full-RW shape mismatch: "
            f"cache={test_rw.shape}, graph_nodes={test_graph.number_of_nodes()}, "
            f"rw_dim={args.rw_dim}"
        )

    if args.rw_full_normalization == "train_standardize":
        mean = train_rw.mean(axis=0, keepdims=True)
        std = train_rw.std(axis=0, keepdims=True)
        train_rw = (train_rw - mean) / np.maximum(std, 1e-6)
        test_rw = (test_rw - mean) / np.maximum(std, 1e-6)
    elif args.rw_full_normalization == "l2":
        train_rw = train_rw / np.maximum(
            np.linalg.norm(train_rw, axis=1, keepdims=True), 1e-12
        )
        test_rw = test_rw / np.maximum(
            np.linalg.norm(test_rw, axis=1, keepdims=True), 1e-12
        )
    elif args.rw_full_normalization != "none":
        raise ValueError(
            f"Unsupported full-RW normalization: {args.rw_full_normalization}"
        )

    def graph_feature_map(graph, matrix):
        result = {}
        for node_id in range(graph.number_of_nodes()):
            triangle = tuple(sorted(map(int, graph.nodes[node_id]["triplet"])))
            result[triangle] = np.asarray(matrix[node_id], dtype=np.float32)
        return result

    feature_maps = {
        "train": graph_feature_map(train_graph, train_rw),
        "test": graph_feature_map(test_graph, test_rw),
    }
    if args.temporal_validation:
        raise ValueError(
            "dual_mapping_full currently supports the original train/test "
            "protocol; temporal early full-graph vectors were not supplied"
        )
    if not np.isfinite(train_rw).all() or not np.isfinite(test_rw).all():
        raise ValueError(f"Dual-Mapping full RW cache contains NaN or Inf: {path}")
    print(
        "Loading original full-graph Dual-Mapping RW cache: "
        f"{path} train={train_rw.shape} test={test_rw.shape} "
        f"normalization={args.rw_full_normalization}",
        flush=True,
    )
    return feature_maps, str(path)


def split_sampling_parameters(split, args):
    if split == "early":
        return (
            args.sampling_mode,
            args.train_positive_fraction,
            args.train_raw_cap,
            args.max_train_complexes,
            200,
            1200,
        )
    if split == "train" and args.temporal_validation:
        return (
            args.validation_sampling_mode,
            None,
            args.validation_raw_cap,
            args.max_validation_complexes,
            300,
            1300,
        )
    if split == "train":
        return (
            args.sampling_mode,
            args.train_positive_fraction,
            args.train_raw_cap,
            args.max_train_complexes,
            100,
            100,
        )
    return (
        args.test_sampling_mode,
        args.test_positive_fraction,
        args.test_raw_cap,
        args.max_test_complexes,
        1100,
        1100,
    )


def select_sampled_candidates(dataset, split, args):
    paths = split_paths(dataset, split, args)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing protocol files for {dataset}/{split}: {missing}")

    candidates = read_triangles(paths["open"])
    positives = set(read_triangles(paths["positive"]))

    (
        sampling_mode,
        configured_positive_fraction,
        raw_cap,
        complex_cap,
        stratified_seed_offset,
        uniform_seed_offset,
    ) = split_sampling_parameters(split, args)
    sampling_seed = effective_sampling_seed(args)
    if sampling_mode == "stratified":
        if configured_positive_fraction is None:
            fraction = len(positives) / max(len(candidates), 1)
        else:
            fraction = configured_positive_fraction
        sampled_candidates = sample_stratified(
            candidates,
            positives,
            min(raw_cap, complex_cap) if raw_cap is not None and complex_cap is not None else complex_cap or raw_cap,
            fraction,
            sampling_seed + stratified_seed_offset,
        )
    elif sampling_mode == "uniform":
        sampled_candidates = sample_uniform(
            candidates,
            raw_cap,
            sampling_seed + uniform_seed_offset - 100,
        )
        sampled_candidates = sample_uniform(
            sampled_candidates,
            complex_cap,
            sampling_seed + uniform_seed_offset,
        )
    else:
        raise ValueError(f"Unsupported {split} sampling mode: {sampling_mode}")
    return paths, candidates, positives, sampled_candidates, sampling_mode


def valid_sampled_triangles(dataset, split, args):
    paths, _, _, sampled_candidates, _ = select_sampled_candidates(
        dataset, split, args
    )
    graph = node_label_transform(nx.read_graphml(paths["graph"]))
    valid_triangles = []
    for tri in sampled_candidates:
        try:
            subgraph = extract_k_hop_subgraph(
                graph,
                args.num_hops,
                tri,
                sample_ratio=args.sample_ratio,
                max_nodes_per_step=args.max_nodes_per_step,
                enforce_node_cap=args.enforce_node_cap,
            )
            if nx.is_connected(remove_target_link(subgraph, tri)):
                valid_triangles.append(tri)
        except Exception:
            if args.strict:
                raise
    return valid_triangles


def build_pyg_samples(dataset, split, args):
    (
        paths,
        candidates,
        positives,
        sampled_candidates,
        sampling_mode,
    ) = select_sampled_candidates(dataset, split, args)
    graph = node_label_transform(nx.read_graphml(paths["graph"]))
    old_simplices = get_old_simplices(dataset, split)

    pyg_samples = []
    valid_triangles = []
    skipped = 0
    positive_count = 0
    iterator = tqdm(sampled_candidates, desc=f"{dataset} {split} graphs", leave=False)
    for tri in iterator:
        try:
            subgraph = extract_k_hop_subgraph(
                graph,
                args.num_hops,
                tri,
                sample_ratio=args.sample_ratio,
                max_nodes_per_step=args.max_nodes_per_step,
                enforce_node_cap=args.enforce_node_cap,
            )
            subgraph_without_target = remove_target_link(subgraph, tri)
            if not nx.is_connected(subgraph_without_target):
                skipped += 1
                continue

            labeled = construct_labeled_networkx_graph(
                subgraph_without_target,
                old_simplices,
                tri,
                list(positives),
                labelling=args.labelling_method,
            )
            data = to_pyg_data(labeled)
            y = 1 if tri in positives else 0
            data.y = torch.tensor([y], dtype=torch.long)
            if args.rw_fusion != "none":
                feature_map = args.dual_rw_feature_maps[split]
                if tri not in feature_map:
                    raise KeyError(
                        f"Triangle missing from {args.rw_source} Dual-RW cache "
                        f"for split={split}: {tri}"
                    )
                data.rw_feature = torch.from_numpy(
                    np.asarray(feature_map[tri], dtype=np.float32)
                ).view(1, -1)
            positive_count += y
            pyg_samples.append(data)
            valid_triangles.append(tri)
        except Exception:
            skipped += 1
            if args.strict:
                raise

    stats = {
        "candidate_count": len(candidates),
        "positive_candidate_count": len(positives),
        "candidate_positive_fraction": len(positives) / max(len(candidates), 1),
        "sampling_mode": sampling_mode,
        "sampled_candidate_count": len(sampled_candidates),
        "valid_graph_count": len(pyg_samples),
        "skipped_graph_count": skipped,
        "valid_positive_count": positive_count,
        "valid_negative_count": len(pyg_samples) - positive_count,
        "valid_positive_fraction": positive_count / max(len(pyg_samples), 1),
    }
    return pyg_samples, stats, valid_triangles


def cache_path(dataset, split, args):
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sampling_mode, configured_positive_fraction, raw_cap, complex_cap, _, _ = (
        split_sampling_parameters(split, args)
    )
    positive_fraction = (
        "auto"
        if configured_positive_fraction is None
        else f"{configured_positive_fraction:g}"
    )
    source_fingerprint = dataset_source_fingerprint(dataset, split, args)
    if args.rw_fusion != "none":
        rw_payload = {
            "path": str(Path(args.resolved_rw_cache).resolve()),
            "source": args.rw_source,
            "dim": args.rw_dim,
            "full_normalization": args.rw_full_normalization,
        }
        rw_cache_token = hashlib.sha256(
            json.dumps(rw_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        rw_tag = f"rw-{args.rw_source}-d{args.rw_dim}-{rw_cache_token}"
    else:
        rw_tag = "rw-none"
    cache_schema = "v4" if split == "test" else "v3"
    name = (
        f"{cache_schema}_{dataset}_{split}_seed{effective_sampling_seed(args)}_hop{args.num_hops}"
        f"_maxstep{args.max_nodes_per_step}_cap{int(args.enforce_node_cap)}"
        f"_{sampling_mode}_pf{positive_fraction}"
        f"_raw{raw_cap}_n{complex_cap}_ratio{args.sample_ratio:g}"
        f"_dim{args.max_dim}_{args.init_method}_{args.labelling_method}_{rw_tag}"
        f"_src{source_fingerprint}.pt"
    )
    return cache_dir / name


def structural_cache_path(dataset, split, args):
    """RW-independent complex cache so feature searches do not rebuild GUDHI."""
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sampling_mode, configured_positive_fraction, raw_cap, complex_cap, _, _ = (
        split_sampling_parameters(split, args)
    )
    positive_fraction = (
        "auto"
        if configured_positive_fraction is None
        else f"{configured_positive_fraction:g}"
    )
    source_fingerprint = dataset_source_fingerprint(dataset, split, args)
    name = (
        f"v5struct_{dataset}_{split}_seed{effective_sampling_seed(args)}_hop{args.num_hops}"
        f"_maxstep{args.max_nodes_per_step}_cap{int(args.enforce_node_cap)}"
        f"_{sampling_mode}_pf{positive_fraction}"
        f"_raw{raw_cap}_n{complex_cap}_ratio{args.sample_ratio:g}"
        f"_dim{args.max_dim}_{args.init_method}_{args.labelling_method}"
        f"_src{source_fingerprint}.pt"
    )
    return cache_dir / name


def legacy_structural_cache_path(dataset, split, args):
    """Find a pre-v5 cache with exactly the same structural sampling key."""
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output_dir) / "cache"
    sampling_mode, configured_positive_fraction, raw_cap, complex_cap, _, _ = (
        split_sampling_parameters(split, args)
    )
    positive_fraction = (
        "auto"
        if configured_positive_fraction is None
        else f"{configured_positive_fraction:g}"
    )
    source_fingerprint = dataset_source_fingerprint(dataset, split, args)
    suffix = (
        f"_{dataset}_{split}_seed{effective_sampling_seed(args)}_hop{args.num_hops}"
        f"_maxstep{args.max_nodes_per_step}_cap{int(args.enforce_node_cap)}"
        f"_{sampling_mode}_pf{positive_fraction}"
        f"_raw{raw_cap}_n{complex_cap}_ratio{args.sample_ratio:g}"
        f"_dim{args.max_dim}_{args.init_method}_{args.labelling_method}"
        f"_src{source_fingerprint}.pt"
    )
    matches = sorted(cache_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def attach_aux_features(complexes, triangles, split, args):
    if len(complexes) != len(triangles):
        raise ValueError(
            f"Complex/triangle alignment mismatch for {split}: "
            f"{len(complexes)} != {len(triangles)}"
        )
    for triangle, complex_data in zip(triangles, complexes):
        triangle = tuple(map(int, triangle))
        if args.rw_fusion != "none":
            feature_map = args.dual_rw_feature_maps[split]
            if triangle not in feature_map:
                raise KeyError(
                    f"Triangle missing from {args.rw_source} RW map for {split}: "
                    f"{triangle}"
                )
            complex_data.cochains[0].rw_feature = torch.from_numpy(
                np.asarray(feature_map[triangle], dtype=np.float32)
            ).view(1, -1)
        if args.temporal_activity_alpha > 0:
            activity_map = args.temporal_activity_maps[split]
            if triangle not in activity_map:
                raise KeyError(
                    f"Triangle missing from temporal activity map for {split}: "
                    f"{triangle}"
                )
            complex_data.cochains[0].temporal_activity_feature = torch.tensor(
                [[float(activity_map[triangle])]], dtype=torch.float32
            )


def load_or_build_complexes(dataset, split, args):
    path = structural_cache_path(dataset, split, args)
    read_path = path
    if not path.exists() and not args.rebuild_cache:
        read_path = legacy_structural_cache_path(dataset, split, args) or path
    if read_path.exists() and not args.rebuild_cache:
        payload = torch.load(read_path, map_location="cpu", weights_only=False)
        triangles = payload.get("triangles")
        if triangles is None:
            triangles = valid_sampled_triangles(dataset, split, args)
            torch.save(
                {
                    "complexes": payload["complexes"],
                    "stats": payload["stats"],
                    "triangles": triangles,
                },
                path,
            )
            print(
                f"Migrated RW-independent structural cache: {read_path} -> {path}",
                flush=True,
            )
        attach_aux_features(payload["complexes"], triangles, split, args)
        return payload["complexes"], payload["stats"]

    pyg_samples, stats, triangles = build_pyg_samples(dataset, split, args)
    complexes = convert_graph_dataset_with_gudhi(
        pyg_samples,
        expansion_dim=args.max_dim,
        include_down_adj=False,
        init_method=args.init_method,
    )
    # Keep the large structural cache independent of RW parameters.
    for complex_data in complexes:
        if hasattr(complex_data.cochains[0], "rw_feature"):
            delattr(complex_data.cochains[0], "rw_feature")
    torch.save(
        {"complexes": complexes, "stats": stats, "triangles": triangles},
        path,
    )
    attach_aux_features(complexes, triangles, split, args)
    return complexes, stats


def make_class_weights(complexes, device):
    labels = [int(item.y.view(-1)[0].item()) for item in complexes]
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    total = counts.sum()
    weights = total / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float, device=device)


def compute_loss(logits, labels, args, class_weights=None):
    labels = labels.view(-1).long()
    if args.loss_mode in {"bce", "cross_entropy"}:
        return F.cross_entropy(logits, labels)
    if args.loss_mode in {"weighted_bce", "weighted_cross_entropy"}:
        return F.cross_entropy(logits, labels, weight=class_weights)
    if args.loss_mode == "focal":
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        row_index = torch.arange(labels.numel(), device=labels.device)
        pt = prob[row_index, labels].clamp(1e-6, 1 - 1e-6)
        ce = -log_prob[row_index, labels]
        focal = (1 - pt).pow(args.focal_gamma) * ce
        if class_weights is not None:
            focal = focal * class_weights[labels]
        return focal.mean()
    raise ValueError(f"Unsupported loss mode: {args.loss_mode}")


def evaluate_model(model, loader, device, args, class_weights=None):
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true = []
    y_prob = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data, return_logits=True)
            labels = data.y.view(-1).long()
            loss = compute_loss(logits, labels, args, class_weights)
            total_loss += loss.item() * labels.numel()
            total_items += labels.numel()
            probabilities = F.softmax(logits, dim=1)
            probs = probabilities[:, 1].detach().cpu().numpy()
            if args.temporal_activity_alpha > 0:
                activity = (
                    data.cochains[0]
                    .temporal_activity_feature.view(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                alpha = float(args.temporal_activity_alpha)
                probs = (1.0 - alpha) * probs + alpha * activity
            preds = (probs >= 0.5).astype(np.int64)
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    baseline = float(y_true_np.mean()) if len(y_true_np) else 0.0

    if len(np.unique(y_true_np)) == 2:
        precision, recall, _ = met.precision_recall_curve(y_true_np, y_prob_np)
        trapezoid_auc_pr = float(met.auc(recall, precision))
        # Non-interpolated step integration is the official AUC-PR used for
        # reporting. Unlike trapezoidal interpolation, a constant scorer has
        # AUC-PR equal to the positive prevalence instead of approximately 0.5.
        auc_pr = float(met.average_precision_score(y_true_np, y_prob_np))
        f1 = float(met.f1_score(y_true_np, y_pred_np, zero_division=0))
        precision_score = float(met.precision_score(y_true_np, y_pred_np, zero_division=0))
        recall_score = float(met.recall_score(y_true_np, y_pred_np, zero_division=0))
    else:
        trapezoid_auc_pr = float("nan")
        auc_pr = float("nan")
        f1 = float("nan")
        precision_score = float("nan")
        recall_score = float("nan")

    acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) else float("nan")
    class0_mask = y_true_np == 0
    class1_mask = y_true_np == 1
    acc0 = float((y_pred_np[class0_mask] == 0).mean()) if class0_mask.any() else float("nan")
    acc1 = float((y_pred_np[class1_mask] == 1).mean()) if class1_mask.any() else float("nan")
    return {
        "loss": total_loss / max(total_items, 1),
        "auc_pr": auc_pr,
        "ap": auc_pr,
        "trapezoid_auc_pr_diagnostic": trapezoid_auc_pr,
        "baseline": baseline,
        "lift": auc_pr / baseline if baseline > 0 and not np.isnan(auc_pr) else float("nan"),
        "accuracy": acc,
        "class0_accuracy": acc0,
        "class1_accuracy": acc1,
        "f1": f1,
        "precision": precision_score,
        "recall": recall_score,
        "count": int(len(y_true_np)),
        "positives": int(y_true_np.sum()) if len(y_true_np) else 0,
        "negatives": int(len(y_true_np) - y_true_np.sum()) if len(y_true_np) else 0,
        "score_min": float(y_prob_np.min()) if len(y_prob_np) else float("nan"),
        "score_max": float(y_prob_np.max()) if len(y_prob_np) else float("nan"),
        "score_std": float(y_prob_np.std()) if len(y_prob_np) else float("nan"),
        "unique_score_count": int(len(np.unique(y_prob_np))),
    }


def stratified_train_validation_split(complexes, val_fraction, seed):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    labels = np.asarray(
        [int(item.y.view(-1)[0].item()) for item in complexes],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise RuntimeError(
                f"Cannot create a stratified validation split: class {label} "
                f"has only {len(indices)} sample(s)."
            )
        val_count = min(max(1, int(round(len(indices) * val_fraction))), len(indices) - 1)
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return (
        [complexes[index] for index in train_indices],
        [complexes[index] for index in val_indices],
    )


def build_classifier(args, device):
    if args.rw_fusion == "none":
        return SparseSIN(
            num_input_features=args.num_input_features,
            num_layers=args.num_layers,
            num_classes=2,
            hidden=args.hidden,
            max_dim=args.max_dim,
            dropout_rate=args.dropout_rate,
            readout=args.readout,
            final_readout=args.final_readout,
        ).to(device)
    return SparseSINWithDualRW(args).to(device)


def train_fixed_epochs(model, loader, epochs, args, device, class_weights, prefix):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=0.5,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data, return_logits=True)
            labels = data.y.view(-1).long()
            loss = compute_loss(logits, labels, args, class_weights)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            total_items += labels.numel()
        scheduler.step()
        if epoch == epochs or epoch % args.eval_every == 0:
            print(
                f"{prefix} epoch={epoch}/{epochs} "
                f"train_loss={total_loss / max(total_items, 1):.6f}",
                flush=True,
            )


def train_dataset(dataset, args, device):
    if args.temporal_validation:
        ensure_temporal_early_protocol(dataset, args)

    if args.rw_fusion != "none":
        if args.rw_source == "aligned_union":
            feature_map, args.resolved_rw_cache = load_or_build_aligned_dual_rw(
                dataset,
                args,
                device,
            )
            args.dual_rw_feature_maps = {
                "early": feature_map,
                "train": feature_map,
                "test": feature_map,
            }
        elif args.rw_source == "dual_mapping_full":
            (
                args.dual_rw_feature_maps,
                args.resolved_rw_cache,
            ) = load_dual_mapping_full_rw(dataset, args)
        else:
            raise ValueError(f"Unsupported RW source: {args.rw_source}")
        # RW generation has its own seed. Restore the model/data seed before
        # constructing cached samples or initializing the classifier.
        seed_everything(args.seed)
    else:
        args.dual_rw_feature_maps = {}
        args.resolved_rw_cache = ""
    if args.temporal_activity_alpha > 0:
        activity_splits = ["train", "test"]
        if args.temporal_validation:
            activity_splits.append("early")
        args.temporal_activity_maps = {
            split: build_temporal_activity_map(dataset, split, args)
            for split in activity_splits
        }
    else:
        args.temporal_activity_maps = {}

    if args.temporal_validation:
        fit_complexes, train_stats = load_or_build_complexes(dataset, "early", args)
        val_complexes, validation_stats = load_or_build_complexes(dataset, "train", args)
        train_complexes = fit_complexes
    else:
        train_complexes, train_stats = load_or_build_complexes(dataset, "train", args)
        validation_stats = None

    if args.validation_only:
        test_complexes = []
        test_stats = {
            "candidate_positive_fraction": float("nan"),
            "valid_graph_count": 0,
            "valid_positive_count": 0,
        }
    else:
        test_complexes, test_stats = load_or_build_complexes(dataset, "test", args)
    if not train_complexes or (not args.validation_only and not test_complexes):
        raise RuntimeError(f"{dataset}: no valid train/test complexes")

    if args.temporal_validation:
        if not fit_complexes or not val_complexes:
            raise RuntimeError(f"{dataset}: no valid temporal fit/validation complexes")
    else:
        fit_complexes, val_complexes = stratified_train_validation_split(
            train_complexes,
            args.val_fraction,
            args.val_seed,
        )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        fit_complexes,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_complexes, batch_size=args.batch_size, shuffle=False)
    test_loader = (
        None
        if args.validation_only
        else DataLoader(test_complexes, batch_size=args.batch_size, shuffle=False)
    )

    model = build_classifier(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    class_weights = (
        make_class_weights(fit_complexes, device)
        if args.loss_mode in {"weighted_bce", "weighted_cross_entropy", "focal"}
        else None
    )

    best = None
    best_state = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_items = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data, return_logits=True)
            labels = data.y.view(-1).long()
            loss = compute_loss(logits, labels, args, class_weights)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * labels.numel()
            train_items += labels.numel()
        scheduler.step()

        if epoch == args.epochs or epoch % args.eval_every == 0:
            metrics = evaluate_model(model, val_loader, device, args, class_weights)
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss / max(train_items, 1)
            if best is None or metrics["auc_pr"] > best["auc_pr"]:
                best = metrics
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{dataset} epoch={epoch} train_loss={metrics['train_loss']:.6f} "
                f"val_auc_pr={metrics['auc_pr']:.6f} val_lift={metrics['lift']:.6f} "
                f"val_acc={metrics['accuracy']:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError(f"{dataset}: validation did not produce a selectable checkpoint")
    model.load_state_dict(best_state)
    final_retrain_epochs = 0
    if args.validation_only:
        final_metrics = evaluate_model(model, val_loader, device, args, class_weights)
        evaluation_scope = "validation"
    elif args.temporal_validation:
        final_retrain_epochs = int(best["epoch"])
        final_complexes = fit_complexes + val_complexes
        seed_everything(args.seed)
        final_generator = torch.Generator()
        final_generator.manual_seed(args.seed)
        final_loader = DataLoader(
            final_complexes,
            batch_size=args.batch_size,
            shuffle=True,
            generator=final_generator,
        )
        final_class_weights = (
            make_class_weights(final_complexes, device)
            if args.loss_mode in {"weighted_bce", "weighted_cross_entropy", "focal"}
            else None
        )
        model = build_classifier(args, device)
        train_fixed_epochs(
            model,
            final_loader,
            final_retrain_epochs,
            args,
            device,
            final_class_weights,
            f"{dataset} final_retrain",
        )
        final_metrics = evaluate_model(
            model,
            test_loader,
            device,
            args,
            final_class_weights,
        )
        evaluation_scope = "test"
    else:
        final_metrics = evaluate_model(model, test_loader, device, args, class_weights)
        evaluation_scope = "test"
    final_metrics["train_seconds"] = time.time() - start
    final_metrics["train_valid_graph_count"] = train_stats["valid_graph_count"]
    final_metrics["test_valid_graph_count"] = test_stats["valid_graph_count"]
    final_metrics["train_valid_positive_count"] = train_stats["valid_positive_count"]
    final_metrics["test_valid_positive_count"] = test_stats["valid_positive_count"]
    final_metrics["fit_graph_count"] = len(fit_complexes)
    final_metrics["validation_graph_count"] = len(val_complexes)
    final_metrics["validation_positive_count"] = sum(
        int(item.y.view(-1)[0].item()) for item in val_complexes
    )
    final_metrics["temporal_validation"] = args.temporal_validation
    final_metrics["final_retrain_epochs"] = final_retrain_epochs
    final_metrics["validation_sampling"] = (
        json.dumps(validation_stats, sort_keys=True)
        if validation_stats is not None
        else ""
    )
    final_metrics["dataset"] = dataset
    final_metrics["seed"] = args.seed
    final_metrics["sampling_seed"] = effective_sampling_seed(args)
    final_metrics["lr"] = args.lr
    final_metrics["hidden"] = args.hidden
    final_metrics["num_layers"] = args.num_layers
    final_metrics["epochs"] = args.epochs
    final_metrics["dropout_rate"] = args.dropout_rate
    final_metrics["readout"] = args.readout
    final_metrics["final_readout"] = args.final_readout
    final_metrics["val_seed"] = args.val_seed
    final_metrics["val_fraction"] = args.val_fraction
    final_metrics["grad_clip"] = args.grad_clip
    final_metrics["sampling_mode"] = args.sampling_mode
    final_metrics["test_sampling_mode"] = args.test_sampling_mode
    final_metrics["train_positive_fraction"] = args.train_positive_fraction
    final_metrics["test_positive_fraction"] = test_stats["candidate_positive_fraction"]
    final_metrics["loss_mode"] = args.loss_mode
    final_metrics["focal_gamma"] = args.focal_gamma
    final_metrics["evaluation_scope"] = evaluation_scope
    final_metrics["rw_fusion"] = args.rw_fusion
    final_metrics["rw_source"] = args.rw_source
    final_metrics["rw_cache"] = args.resolved_rw_cache
    final_metrics["rw_seed"] = args.rw_seed
    final_metrics["rw_dim"] = args.rw_dim
    final_metrics["rw_proj_dim"] = args.rw_proj_dim
    final_metrics["fusion_hidden"] = args.fusion_hidden
    final_metrics["rw_dropout"] = args.rw_dropout
    final_metrics["rw_scale"] = args.rw_scale
    final_metrics["rw_input_normalization"] = args.rw_input_normalization
    final_metrics["rw_full_normalization"] = args.rw_full_normalization
    final_metrics["rw_walk_length"] = args.rw_walk_length
    final_metrics["rw_walks_per_node"] = args.rw_walks_per_node
    final_metrics["rw_window_size"] = args.rw_window_size
    final_metrics["rw_epochs"] = args.rw_epochs
    final_metrics["rw_negative_samples"] = args.rw_negative_samples
    final_metrics["rw_priority_multiplier"] = args.rw_priority_multiplier
    final_metrics["temporal_activity_alpha"] = args.temporal_activity_alpha
    final_metrics[
        "temporal_activity_recent_fraction"
    ] = args.temporal_activity_recent_fraction
    final_metrics[
        "temporal_activity_aggregation"
    ] = args.temporal_activity_aggregation
    final_metrics["temporal_activity_entity"] = args.temporal_activity_entity
    final_metrics["maxsimplex_prior_weight"] = args.maxsimplex_prior_weight
    final_metrics["maxsimplex_prior_column"] = args.maxsimplex_prior_column
    final_metrics["best_val_epoch"] = best["epoch"]
    final_metrics["best_val_auc_pr"] = best["auc_pr"]
    final_metrics["best_val_lift"] = best["lift"]
    final_metrics["sampling"] = json.dumps({"train": train_stats, "test": test_stats}, sort_keys=True)
    return final_metrics


def write_results(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "seed",
        "sampling_seed",
        "lr",
        "hidden",
        "num_layers",
        "epochs",
        "dropout_rate",
        "readout",
        "final_readout",
        "val_seed",
        "val_fraction",
        "grad_clip",
        "sampling_mode",
        "test_sampling_mode",
        "train_positive_fraction",
        "test_positive_fraction",
        "loss_mode",
        "focal_gamma",
        "evaluation_scope",
        "rw_fusion",
        "rw_source",
        "rw_cache",
        "rw_seed",
        "rw_dim",
        "rw_proj_dim",
        "fusion_hidden",
        "rw_dropout",
        "rw_scale",
        "rw_input_normalization",
        "rw_full_normalization",
        "rw_walk_length",
        "rw_walks_per_node",
        "rw_window_size",
        "rw_epochs",
        "rw_negative_samples",
        "rw_priority_multiplier",
        "temporal_activity_alpha",
        "temporal_activity_recent_fraction",
        "temporal_activity_aggregation",
        "temporal_activity_entity",
        "maxsimplex_prior_weight",
        "maxsimplex_prior_column",
        "temporal_validation",
        "final_retrain_epochs",
        "auc_pr",
        "ap",
        "trapezoid_auc_pr_diagnostic",
        "baseline",
        "lift",
        "accuracy",
        "class0_accuracy",
        "class1_accuracy",
        "f1",
        "precision",
        "recall",
        "loss",
        "count",
        "positives",
        "negatives",
        "score_min",
        "score_max",
        "score_std",
        "unique_score_count",
        "best_val_epoch",
        "best_val_auc_pr",
        "best_val_lift",
        "train_valid_graph_count",
        "test_valid_graph_count",
        "train_valid_positive_count",
        "test_valid_positive_count",
        "fit_graph_count",
        "validation_graph_count",
        "validation_positive_count",
        "validation_sampling",
        "train_seconds",
        "sampling",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary(rows, output_path):
    valid_rows = [row for row in rows if row.get("baseline") not in (None, "")]
    if not valid_rows:
        return

    summaries = []
    for dataset in sorted({row["dataset"] for row in valid_rows}):
        dataset_rows = [row for row in valid_rows if row["dataset"] == dataset]

        def summarize(field):
            values = np.asarray([float(row[field]) for row in dataset_rows], dtype=np.float64)
            return float(values.mean()), float(values.std(ddof=0))

        summary = {
            "dataset": dataset,
            "runs": len(dataset_rows),
            "seeds": ",".join(str(row["seed"]) for row in dataset_rows),
        }
        for field in ("auc_pr", "baseline", "lift", "best_val_auc_pr", "best_val_lift"):
            mean_value, std_value = summarize(field)
            summary[f"mean_{field}"] = mean_value
            summary[f"std_{field}"] = std_value
        summaries.append(summary)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(
            f"{summary['dataset']} {summary['runs']}-run summary: "
            f"mean_baseline={summary['mean_baseline']:.6f} "
            f"mean_auc_pr={summary['mean_auc_pr']:.6f} "
            f"mean_lift={summary['mean_lift']:.6f} "
            f"mean_best_val_lift={summary['mean_best_val_lift']:.6f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["email-Enron"])
    parser.add_argument(
        "--output-dir",
        default=str(Path(ROOT_DIR) / "exp" / "results" / "email_enron_mlshol_dual_rw_seed44"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--temporal-validation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[44])
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--val-seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-input-features", type=int, default=1)
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--readout", choices=["sum", "mean"], default="mean")
    parser.add_argument("--final-readout", choices=["sum", "mean"], default="mean")
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-nodes-per-step", type=int, default=75)
    parser.add_argument("--enforce-node-cap", dest="enforce_node_cap", action="store_true", default=True)
    parser.add_argument("--no-enforce-node-cap", dest="enforce_node_cap", action="store_false")
    parser.add_argument("--train-raw-cap", type=int, default=10000)
    parser.add_argument("--test-raw-cap", type=int, default=2500)
    parser.add_argument("--validation-raw-cap", type=int, default=10000)
    parser.add_argument("--max-train-complexes", type=int, default=200)
    parser.add_argument("--max-validation-complexes", type=int, default=1000)
    parser.add_argument("--max-test-complexes", type=int, default=1000)
    parser.add_argument("--sampling-mode", choices=["uniform", "stratified"], default="stratified")
    parser.add_argument("--validation-sampling-mode", choices=["uniform", "stratified"], default="uniform")
    parser.add_argument("--test-sampling-mode", choices=["uniform", "stratified"], default="uniform")
    parser.add_argument("--train-positive-fraction", type=float, default=0.2)
    parser.add_argument("--test-positive-fraction", type=float, default=None)
    parser.add_argument(
        "--loss-mode",
        choices=["cross_entropy", "weighted_cross_entropy", "bce", "weighted_bce", "focal"],
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--labelling-method", default="TRNL")
    parser.add_argument("--init-method", default="sum")
    parser.add_argument("--rw-fusion", choices=["none", "concat", "rw_only"], default="concat")
    parser.add_argument(
        "--rw-source",
        choices=["aligned_union", "dual_mapping_full"],
        default="aligned_union",
    )
    parser.add_argument("--rw-cache-dir", default=None)
    parser.add_argument("--rw-reference-cache", default=None)
    parser.add_argument("--rw-seed", type=int, default=42)
    parser.add_argument("--rw-dim", type=int, default=64)
    parser.add_argument("--rw-proj-dim", type=int, default=32)
    parser.add_argument("--fusion-hidden", type=int, default=32)
    parser.add_argument("--rw-dropout", type=float, default=0.5)
    parser.add_argument("--rw-scale", type=float, default=1.0)
    parser.add_argument(
        "--rw-input-normalization",
        choices=["layer_norm", "none"],
        default="layer_norm",
    )
    parser.add_argument(
        "--rw-full-normalization",
        choices=["train_standardize", "l2", "none"],
        default="train_standardize",
    )
    parser.add_argument("--rw-walk-length", type=int, default=20)
    parser.add_argument("--rw-walks-per-node", type=int, default=20)
    parser.add_argument("--rw-window-size", type=int, default=5)
    parser.add_argument("--rw-epochs", type=int, default=5)
    parser.add_argument("--rw-negative-samples", type=int, default=5)
    parser.add_argument("--rw-priority-multiplier", type=int, default=5)
    parser.add_argument("--rw-lr", type=float, default=0.01)
    parser.add_argument("--rw-batch-size", type=int, default=8192)
    parser.add_argument("--rw-chunk-starts", type=int, default=4096)
    parser.add_argument("--temporal-activity-alpha", type=float, default=0.0)
    parser.add_argument(
        "--temporal-activity-recent-fraction", type=float, default=0.15
    )
    parser.add_argument(
        "--temporal-activity-aggregation",
        choices=["min", "product", "mean"],
        default="product",
    )
    parser.add_argument(
        "--temporal-activity-entity",
        choices=["node", "edge"],
        default="node",
    )
    parser.add_argument("--maxsimplex-prior-weight", type=float, default=0.0)
    parser.add_argument(
        "--maxsimplex-prior-column",
        default="tri_all_edges_shared_count",
    )
    parser.add_argument(
        "--maxsimplex-feature-dir",
        default=str(Path(ROOT_DIR) / ".cache_features"),
        help=(
            "Directory containing <dataset>_<split>_edge_maxsimplex_features "
            "with a .tsv, .txt, or .bin suffix"
        ),
    )
    parser.add_argument("--rebuild-rw-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    rows = []
    for seed in args.seeds:
        args.seed = seed
        for dataset in args.datasets:
            # Dataset runs are independent of argument order.
            seed_everything(seed)
            print(f"=== {dataset} | seed={seed} ===", flush=True)
            try:
                row = train_dataset(dataset, args, device)
                rows.append(row)
                write_results(rows, Path(args.output_dir) / "results.csv")
            except Exception as exc:
                rows.append({"dataset": dataset, "seed": seed, "sampling": f"ERROR: {exc}"})
                write_results(rows, Path(args.output_dir) / "results.csv")
                print(f"{dataset} seed={seed} failed: {exc}", flush=True)
                if args.strict:
                    raise

    write_summary(rows, Path(args.output_dir) / "summary.csv")
    print(f"wrote {Path(args.output_dir) / 'results.csv'}", flush=True)
    print(f"wrote {Path(args.output_dir) / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
