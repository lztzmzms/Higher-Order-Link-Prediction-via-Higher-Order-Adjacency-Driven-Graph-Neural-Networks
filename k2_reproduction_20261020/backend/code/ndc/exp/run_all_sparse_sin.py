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
from interaction_strength import (
    build_boundary_strength,
    candidate_strength_matrix,
    dataset_strength_statistics,
    strength_value,
    write_rows,
)
from k2_provenance import validate_feature_provenance


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

# Defaults preserve the historical source-order protocol. Controlled
# timestamp experiments can redirect all protocol assets without touching it.
DATA_ROOT = Path(ROOT_DIR)
EVENT_ORDER = "raw"


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
        self.use_strength_feature = bool(args.use_strength_feature)
        self.history_feature_mode = str(args.history_feature_mode)
        self.history_feature_dim = int(getattr(args, "history_feature_dim", 0))
        self.history_residual_scale = float(args.history_residual_scale)
        self.history_pretrained_checkpoint = str(getattr(args, "history_pretrained_checkpoint", "") or "")
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
        if self.history_feature_mode == "history_only":
            history_hidden = max(32, min(128, self.history_feature_dim * 2))
            self.history_only_encoder = nn.Sequential(
                nn.LayerNorm(self.history_feature_dim),
                nn.Linear(self.history_feature_dim, history_hidden),
                nn.ReLU(),
                nn.Dropout(args.rw_dropout),
            )
            self.history_only_head = nn.Linear(history_hidden, 2)
        if self.use_strength_feature:
            self.strength_projection = nn.Sequential(
                nn.LayerNorm(7),
                nn.Linear(7, args.rw_proj_dim),
                nn.ReLU(),
                nn.Dropout(args.rw_dropout),
            )
        if self.history_feature_mode == "concat":
            self.history_projection = nn.Sequential(
                nn.LayerNorm(self.history_feature_dim),
                nn.Linear(self.history_feature_dim, args.rw_proj_dim),
                nn.ReLU(),
                nn.Dropout(args.rw_dropout),
            )
        elif self.history_feature_mode in {"residual", "gated_residual"}:
            self.history_residual_head = nn.Sequential(
                nn.LayerNorm(self.history_feature_dim),
                nn.Linear(self.history_feature_dim, args.fusion_hidden),
                nn.ReLU(),
                nn.Dropout(args.rw_dropout),
                nn.Linear(args.fusion_hidden, 2),
            )
        if self.fusion_mode == "rw_only":
            fusion_dim = args.rw_proj_dim
        elif self.fusion_mode == "none":
            # A no-RW run can still use the history head and/or interaction
            # strength channel.  The previous process-control path returned
            # bare SparseSIN here, silently dropping those requested inputs.
            fusion_dim = sin_dim
        else:
            fusion_dim = sin_dim + args.rw_proj_dim
        if self.use_strength_feature:
            fusion_dim += args.rw_proj_dim
        if self.history_feature_mode == "concat":
            fusion_dim += args.rw_proj_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, args.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(args.rw_dropout),
            nn.Linear(args.fusion_hidden, 2),
        )
        if self.history_feature_mode == "gated_residual":
            self.history_gate_head = nn.Sequential(
                nn.LayerNorm(fusion_dim + self.history_feature_dim),
                nn.Linear(fusion_dim + self.history_feature_dim, args.fusion_hidden),
                nn.ReLU(),
                nn.Linear(args.fusion_hidden, 1),
                nn.Sigmoid(),
            )
        if self.history_feature_mode == "history_base_structural_residual":
            # History is the base ranking; the structural branch is a
            # zero-initialized, candidate-conditioned correction.  This keeps
            # the first joint forward pass exactly equal to the history head.
            self.history_pretrained = bool(self.history_pretrained_checkpoint)
            if self.history_pretrained:
                # This is deliberately the same 64->64->64->1 HistoryMLP
                # used by run_v13_history_pretrain.py.  The checkpoint is
                # loaded below and its positive logit is represented as [0,z]
                # so softmax([0,z]) == sigmoid(z).
                history_hidden = 64
                self.history_pretrained_net = nn.Sequential(
                    nn.Linear(self.history_feature_dim, 64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 1),
                )
                checkpoint = torch.load(self.history_pretrained_checkpoint, map_location="cpu", weights_only=False)
                state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                if isinstance(state, dict) and state and all(str(key).startswith("net.") for key in state):
                    state = {str(key)[4:]: value for key, value in state.items()}
                self.history_pretrained_net.load_state_dict(state)
                self.history_pretrained_net.to(next(self.sin.parameters()).device)
            else:
                history_hidden = max(32, min(128, self.history_feature_dim * 2))
                self.history_base_encoder = nn.Sequential(
                    nn.LayerNorm(self.history_feature_dim),
                    nn.Linear(self.history_feature_dim, history_hidden),
                    nn.ReLU(),
                    nn.Dropout(args.rw_dropout),
                )
                self.history_base_head = nn.Linear(history_hidden, 2)
            residual_hidden = max(32, min(128, int(args.fusion_hidden)))
            self.structural_residual_head = nn.Sequential(
                nn.LayerNorm(fusion_dim + history_hidden),
                nn.Linear(fusion_dim + history_hidden, residual_hidden),
                nn.ReLU(),
                nn.Dropout(args.rw_dropout),
                nn.Linear(residual_hidden, 2),
            )
            self.structural_residual_gate = nn.Sequential(
                nn.LayerNorm(fusion_dim + history_hidden),
                nn.Linear(fusion_dim + history_hidden, residual_hidden),
                nn.ReLU(),
                nn.Linear(residual_hidden, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.structural_residual_head[-1].weight)
            nn.init.zeros_(self.structural_residual_head[-1].bias)
            nn.init.constant_(self.structural_residual_gate[-2].bias, -2.0)

    def forward(self, data, return_logits=False):
        if self.history_feature_mode == "history_only":
            history = data.cochains[0].history_feature
            logits = self.history_only_head(self.history_only_encoder(history))
            return logits if return_logits else F.softmax(logits, dim=1)
        if self.fusion_mode == "none":
            sin_embedding = self.sin(data, return_embedding=True)
            fused = sin_embedding
        else:
            rw_feature = data.cochains[0].rw_feature
            rw_embedding = self.rw_projection(rw_feature) * self.rw_scale
            if self.fusion_mode == "rw_only":
                fused = rw_embedding
            else:
                sin_embedding = self.sin(data, return_embedding=True)
                fused = torch.cat([sin_embedding, rw_embedding], dim=1)
        if self.use_strength_feature:
            strength = data.cochains[0].interaction_strength_feature
            strength_embedding = self.strength_projection(strength)
            fused = torch.cat([fused, strength_embedding], dim=1)
        if self.history_feature_mode == "concat":
            history = data.cochains[0].history_feature
            fused = torch.cat([fused, self.history_projection(history)], dim=1)
        logits = self.classifier(fused)
        if self.history_feature_mode in {"residual", "gated_residual"}:
            history = data.cochains[0].history_feature
            history_logits = self.history_residual_head(history)
            if self.history_feature_mode == "gated_residual":
                gate = self.history_gate_head(torch.cat([fused, history], dim=1))
                history_logits = gate * history_logits
            logits = logits + self.history_residual_scale * history_logits
        elif self.history_feature_mode == "history_base_structural_residual":
            history = data.cochains[0].history_feature
            if self.history_pretrained:
                history_hidden = self.history_pretrained_net[:6](history)
                positive_logit = self.history_pretrained_net[6](history_hidden).squeeze(-1)
                history_logits = torch.stack([torch.zeros_like(positive_logit), positive_logit], dim=1)
            else:
                history_hidden = self.history_base_encoder(history)
                history_logits = self.history_base_head(history_hidden)
            joint = torch.cat([fused, history_hidden], dim=1)
            structural_gate = self.structural_residual_gate(joint)
            structural_residual = self.structural_residual_head(joint)
            logits = history_logits + structural_gate * structural_residual
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


def trnl_node_labeling(k_hop_subgraph, tri, disconnected_policy="skip"):
    node_1, node_2, node_3 = tuple(tri)
    graph = remove_target_link(k_hop_subgraph, tri)
    node_list = list(graph.nodes)
    index_1 = node_list.index(node_1)
    index_2 = node_list.index(node_2)
    index_3 = node_list.index(node_3)

    distance_maps = [
        nx.single_source_shortest_path_length(graph, source=source)
        for source in (node_1, node_2, node_3)
    ]
    if disconnected_policy == "skip" and any(
        len(distance_map) != len(node_list) for distance_map in distance_maps
    ):
        raise nx.NetworkXNoPath("target-removed local graph is disconnected")
    dist2node_1 = [distance_maps[0].get(node) for node in node_list]
    dist2node_2 = [distance_maps[1].get(node) for node in node_list]
    dist2node_3 = [distance_maps[2].get(node) for node in node_list]
    dist2node_1[index_2] = 0
    dist2node_1[index_3] = 0
    dist2node_2[index_1] = 0
    dist2node_2[index_3] = 0
    dist2node_3[index_1] = 0
    dist2node_3[index_2] = 0

    labels = [None] * len(node_list)
    for index in range(len(node_list)):
        if any(distance is None for distance in (
            dist2node_1[index], dist2node_2[index], dist2node_3[index]
        )):
            labels[index] = [0]
            continue
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


def construct_labeled_networkx_graph(
    k_hop_subgraph,
    old_simplices,
    tri,
    positives,
    labelling="TRNL",
    disconnected_policy="skip",
    label_graph=None,
    observed_2_skeleton=False,
):
    if labelling != "TRNL":
        raise ValueError(f"Only TRNL is supported by this runner, got {labelling}")
    node_list = list(k_hop_subgraph.nodes)
    edge_list = list(k_hop_subgraph.edges)
    # The K=2 task already observes all three target edges at cutoff.  When
    # the ``keep_target_links`` ablation is enabled, message passing receives
    # the original local graph while TRNL node labels are still computed on a
    # target-edge-removed view.  This isolates the effect of preserving the
    # known edges without changing the distance-label definition.
    embeddings = trnl_node_labeling(
        k_hop_subgraph if label_graph is None else label_graph,
        tri,
        disconnected_policy=disconnected_policy,
    )
    nx.set_node_attributes(
        k_hop_subgraph,
        {node_list[index]: {"feature": embeddings[index]} for index in range(len(embeddings))},
    )

    observed_simplices = set()
    node_set = set(node_list)
    for old_simplex in old_simplices:
        if len(old_simplex) >= 3:
            simplex = sorted(set(old_simplex) & node_set)
            if len(simplex) >= 3:
                observed_simplices.add(tuple(simplex))
    # Keep only inclusion-maximal observed simplices.  The old implementation
    # compared every ordered pair and called ``original.index`` inside the
    # nested loop, which became quadratic twice over on dense local histories.
    # Sorting by decreasing size lets us compare each simplex only with the
    # already retained maximal set while preserving the same downward-closure
    # pruning semantics.
    simplex_keys = {tuple(sorted(simplex)) for simplex in observed_simplices}
    simplex_keys.update(tuple(sorted(edge)) for edge in edge_list)
    if observed_2_skeleton:
        # Materialize only the downward-closed 1/2-skeleton.  This avoids
        # inserting a large historical simplex into GUDHI before truncating
        # it to dimension two.
        skeleton = set()
        for key in simplex_keys:
            vertices = tuple(key)
            for size in (2, 3):
                if len(vertices) >= size:
                    skeleton.update(combinations(vertices, size))
        simplices = [list(simplex) for simplex in sorted(skeleton, key=lambda value: (len(value), value))]
    else:
        maximal = []
        for key in sorted(simplex_keys, key=lambda value: (-len(value), value)):
            key_set = set(key)
            if not any(key_set < set(parent) for parent in maximal):
                maximal.append(key)
        simplices = [list(simplex) for simplex in sorted(maximal, key=lambda value: (len(value), value))]
    return nx.Graph(k_hop_subgraph, y=1 if tri in positives else 0, simplices=simplices)


def to_pyg_data(k_hop_subgraph):
    node_list = list(k_hop_subgraph.nodes)
    node_index = {node: index for index, node in enumerate(node_list)}
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
            simplex[i] = nodes_new[node_index[simplex[i]]]
        simplex.sort()
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([k_hop_subgraph.graph["y"]], dtype=torch.long),
        node_ids=torch.tensor(nodes_new),
        original_node_ids=torch.tensor([int(node) for node in node_list], dtype=torch.long),
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
    rows = []
    offset = 0
    for row_id, (count, timestamp) in enumerate(zip(nverts, timestamps)):
        simplex = flat_simplices[offset : offset + count]
        simplex.sort()
        rows.append((int(timestamp), row_id, simplex))
        offset += count
    if EVENT_ORDER == "timestamp":
        rows.sort(key=lambda row: (row[0], row[1]))
    elif EVENT_ORDER != "raw":
        raise ValueError(f"Unsupported event order: {EVENT_ORDER}")
    return [row[0] for row in rows], [row[2] for row in rows]


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
    raw_dir = Path(DATA_ROOT) / "dataset" / "raw_data" / dataset
    return (
        raw_dir / f"{dataset}-nverts.txt",
        raw_dir / f"{dataset}-times.txt",
        raw_dir / f"{dataset}-simplices.txt",
    )


def get_old_simplices(dataset, split, args=None):
    nverts_path, times_path, simplices_path = raw_paths(dataset)
    _, data_simplices = get_timestamped_simplices(
        str(nverts_path), str(times_path), str(simplices_path)
    )
    if split == "early":
        fraction = 0.4
    elif split == "train":
        fraction = float(getattr(args, "train_history_context_ratio", 0.6))
    else:
        fraction = float(getattr(args, "test_history_context_ratio", 0.8))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{dataset}/{split}: history context ratio must be in (0, 1], got {fraction}")
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


def validate_maxsimplex_feature_path(dataset, split, feature_path, args):
    """Validate the feature cache against the exact raw protocol cutoff."""
    raw_dir = Path(DATA_ROOT) / "dataset" / "raw_data" / dataset
    gpickle_dir = Path(DATA_ROOT) / "gpickle" / dataset
    gpickle_name = (
        f"classified_network_{dataset}_train_1.gpickle"
        if split == "train"
        else f"classified_network_{dataset}_all_1.gpickle"
    )
    gpickle_path = gpickle_dir / gpickle_name
    if not gpickle_path.is_file():
        raise FileNotFoundError(
            f"Maximal-simplex provenance requires the matching gpickle: {gpickle_path}"
        )
    ratio = (
        float(getattr(args, "train_history_context_ratio", 0.6))
        if split == "train"
        else float(getattr(args, "test_history_context_ratio", 0.8))
    )
    return validate_feature_provenance(
        feature_path,
        raw_dir,
        dataset,
        EVENT_ORDER,
        ratio,
        gpickle_path,
    )


def build_temporal_activity_map(dataset, split, args):
    """Context-only activity prior for each open-triangle candidate.

    By default this uses the observed context only.  An explicit context
    ratio of 1.0 is a transductive raw-event diagnostic and must be reported
    separately because it includes post-cutoff events, although it still
    never reads the future-label file while constructing the score.
    """
    if split == "early":
        context_fraction = 0.4
    elif split == "train":
        context_fraction = 0.6
    else:
        context_fraction = float(
            getattr(
                args,
                "temporal_activity_context_ratio",
                getattr(args, "test_history_context_ratio", 0.8),
            )
        )
        if not 0.0 < context_fraction <= 1.0:
            raise ValueError("temporal_activity_context_ratio must be in (0, 1]")
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
    node_last = {}
    edge_last = {}
    if args.temporal_activity_measure == "last_event":
        for event_index, simplex in enumerate(data_simplices[:context_end]):
            vertices = sorted(set(map(int, simplex)))
            for node in vertices:
                node_last[node] = event_index
            for edge in combinations(vertices, 2):
                edge_last[tuple(sorted(edge))] = event_index
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
                [
                    node_last.get(int(node), -1)
                    if args.temporal_activity_measure == "last_event"
                    else node_counts[int(node)]
                    for node in triangle
                ], dtype=np.float64
            )
        elif args.temporal_activity_entity == "edge":
            counts = np.asarray(
                [
                    edge_last.get(tuple(sorted(edge)), -1)
                    if args.temporal_activity_measure == "last_event"
                    else edge_counts[tuple(sorted(edge))]
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
        provenance = validate_maxsimplex_feature_path(dataset, split, feature_path, args)
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
        missing = [triangle for triangle in candidates if triangle not in values]
        if missing:
            raise ValueError(
                f"{dataset}/{split}: maximal-simplex cache is missing "
                f"{len(missing)} protocol candidates"
            )
        maximal_raw = np.asarray([values[triangle] for triangle in candidates], dtype=np.float64)
        if not np.isfinite(maximal_raw).all():
            raise FloatingPointError(
                f"{dataset}/{split}: maximal-simplex prior contains NaN or Inf"
            )
        if args.maxsimplex_prior_column == "tri_all_edges_shared_count":
            nonzero = int(np.sum(maximal_raw != 0.0))
            if nonzero:
                raise ValueError(
                    f"{dataset}/{split}: {nonzero}/{len(candidates)} open candidates "
                    "have tri_all_edges_shared_count != 0; cache/protocol mismatch"
                )
        maximal_ranks = rankdata(maximal_raw, method="average") / max(
            len(candidates), 1
        )
        weight = float(args.maxsimplex_prior_weight)
        activity_ranks = (activity_ranks + weight * maximal_ranks) / (1.0 + weight)
    return {
        triangle: np.float32(activity_ranks[index])
        for index, triangle in enumerate(candidates)
    }


def build_interaction_strength_map(dataset, split, args):
    """Build boundary strength from timestamped events strictly before T."""
    if split == "early":
        context_fraction = 0.4
    elif split == "train":
        context_fraction = (
            float(args.train_strength_context_ratio)
            if args.train_strength_context_ratio is not None else 0.6
        )
    else:
        context_fraction = (
            float(args.test_strength_context_ratio)
            if args.test_strength_context_ratio is not None
            else float(getattr(args, "test_history_context_ratio", 0.8))
        )
    timestamps, simplices = get_timestamped_simplices(
        str(raw_paths(dataset)[0]), str(raw_paths(dataset)[1]), str(raw_paths(dataset)[2])
    )
    context_end = int(context_fraction * len(simplices))
    if EVENT_ORDER == "raw":
        # Raw-order is an event-index protocol.  Use a cutoff above every
        # retained context timestamp so no valid raw-history event is dropped
        # merely because its timestamp is out of order.
        cutoff = (
            float(max(timestamps[:context_end]) + 1)
            if context_end
            else 0.0
        )
    else:
        cutoff = float(timestamps[context_end]) if context_end < len(timestamps) else float(max(timestamps) + 1)
    events = [
        {"time": timestamp, "vertices": vertices}
        for timestamp, vertices in zip(timestamps[:context_end], simplices[:context_end])
    ]
    strength = build_boundary_strength(events, cutoff, args.strength_decay_lambda)
    return strength, cutoff, events


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
    split_dir = Path(DATA_ROOT) / "dataset" / split
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


def skipgram_pairs_per_walk(walk_positions, window_size):
    walk_positions = max(1, int(walk_positions))
    window_size = max(0, min(int(window_size), walk_positions - 1))
    return 2 * sum(walk_positions - distance for distance in range(1, window_size + 1))


def build_bounded_aligned_rw_pairs(indptr, indices, num_nodes, priority_mask, args):
    """Build a preallocated random-walk pair corpus with a strict cap."""
    rng = np.random.default_rng(args.rw_seed)
    nodes = np.arange(num_nodes, dtype=np.int64)
    priority_indices = np.flatnonzero(priority_mask).astype(np.int64, copy=False)
    start_pool = nodes
    if priority_indices.size and args.rw_priority_multiplier > 1:
        start_pool = np.concatenate(
            [nodes, np.tile(priority_indices, args.rw_priority_multiplier - 1)]
        )

    pairs_per_walk = skipgram_pairs_per_walk(
        int(args.rw_walk_length) + 1,
        int(args.rw_window_size),
    )
    requested_walks = int(len(start_pool) * int(args.rw_walks_per_node))
    requested_pairs = int(requested_walks * pairs_per_walk)
    pair_cap = max(1, int(args.rw_max_pairs))
    allocation_pairs = min(pair_cap, requested_pairs)
    centers = np.empty(allocation_pairs, dtype=np.int64)
    contexts = np.empty(allocation_pairs, dtype=np.int64)

    if pairs_per_walk <= 0 or requested_walks <= 0:
        return centers[:0], contexts[:0], {
            "requested_walks": requested_walks,
            "generated_walks": 0,
            "requested_pairs": requested_pairs,
            "generated_pairs": 0,
            "pair_cap": pair_cap,
            "capped": requested_pairs > pair_cap,
        }

    max_walks = min(
        requested_walks,
        (allocation_pairs + pairs_per_walk - 1) // pairs_per_walk,
    )
    if requested_pairs <= pair_cap:
        starts = np.concatenate(
            [rng.permutation(start_pool) for _ in range(int(args.rw_walks_per_node))]
        )
    elif max_walks >= num_nodes:
        first = rng.permutation(nodes)
        remaining = max_walks - num_nodes
        extra = (
            rng.choice(start_pool, size=remaining, replace=True)
            if remaining > 0
            else np.empty(0, dtype=np.int64)
        )
        starts = np.concatenate([first, extra])
    else:
        starts = rng.permutation(nodes)[:max_walks]

    cursor = 0
    generated_walks = 0
    chunk_size = max(1, int(args.rw_chunk_starts))
    for start in range(0, len(starts), chunk_size):
        chunk_starts = starts[start:start + chunk_size]
        walks = generate_walks_csr_chunk(
            indptr,
            indices,
            chunk_starts,
            int(args.rw_walk_length),
            rng,
        )
        generated_walks += len(walks)
        for distance in range(1, min(int(args.rw_window_size), walks.shape[1] - 1) + 1):
            left = walks[:, :-distance].reshape(-1)
            right = walks[:, distance:].reshape(-1)
            for center_values, context_values in ((left, right), (right, left)):
                remaining = allocation_pairs - cursor
                if remaining <= 0:
                    break
                count = min(remaining, len(center_values))
                centers[cursor:cursor + count] = center_values[:count]
                contexts[cursor:cursor + count] = context_values[:count]
                cursor += count
            if cursor >= allocation_pairs:
                break
        if cursor >= allocation_pairs:
            break

    stats = {
        "requested_walks": requested_walks,
        "generated_walks": int(generated_walks),
        "requested_pairs": requested_pairs,
        "generated_pairs": int(cursor),
        "pair_cap": pair_cap,
        "capped": requested_pairs > pair_cap,
    }
    return centers[:cursor], contexts[:cursor], stats


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
            [nodes, np.tile(priority_indices, args.rw_priority_multiplier - 1)]
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
        epoch_pairs = 0
        generated_walks = 0
        pair_cap = max(1, int(args.rw_max_pairs))
        capped = False
        for _ in range(args.rw_walks_per_node):
            shuffled = rng.permutation(start_nodes)
            for start in range(0, len(shuffled), args.rw_chunk_starts):
                starts = shuffled[start:start + args.rw_chunk_starts]
                walks = generate_walks_csr_chunk(
                    indptr,
                    indices,
                    starts,
                    args.rw_walk_length,
                    rng,
                )
                generated_walks += len(walks)
                for center_np, context_np in iter_pairs_from_walks(
                    walks,
                    args.rw_window_size,
                    args.rw_batch_size,
                ):
                    remaining = pair_cap - epoch_pairs
                    if remaining <= 0:
                        capped = True
                        break
                    if len(center_np) > remaining:
                        center_np = center_np[:remaining]
                        context_np = context_np[:remaining]
                        capped = True
                    centers = torch.from_numpy(center_np).long().to(device)
                    contexts = torch.from_numpy(context_np).long().to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(centers, contexts, args.rw_negative_samples)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    total_batches += 1
                    epoch_pairs += len(center_np)
                    if epoch_pairs >= pair_cap:
                        capped = True
                        break
                if capped:
                    break
            if capped:
                break
        print(
            "random-walk skip-gram "
            f"[epoch={epoch}]: pairs={epoch_pairs:,}/cap={pair_cap:,} "
            f"walks={generated_walks:,} capped={capped}",
            flush=True,
        )
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
        "max_pairs": args.rw_max_pairs,
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
        # Keep the cache write resilient if a caller cleaned its parent folder.
        path.parent.mkdir(parents=True, exist_ok=True)
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
        payload = None
        if "payload" in cached and np.asarray(cached["payload"]).size == 1:
            try:
                payload = json.loads(str(np.asarray(cached["payload"]).item()))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = None

    gpickle_dir = Path(DATA_ROOT) / "gpickle" / dataset
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

    if not isinstance(payload, dict):
        raise ValueError(f"Dual-Mapping full RW cache has no readable provenance payload: {path}")

    def file_md5(file_path):
        digest = hashlib.md5()
        with Path(file_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    expected_hashes = {
        "train_hash": file_md5(train_graph_path),
        "test_hash": file_md5(test_graph_path),
    }
    hash_mismatches = {
        key: (payload.get(key), expected)
        for key, expected in expected_hashes.items()
        if payload.get(key) != expected
    }
    if hash_mismatches:
        raise ValueError(
            "Dual-Mapping full RW cache provenance does not match the packaged graphs: "
            f"{hash_mismatches}"
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
            if (
                nx.is_connected(remove_target_link(subgraph, tri))
                or args.disconnected_local_policy == "unreachable_zero"
            ):
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
    old_simplices = get_old_simplices(dataset, split, args)
    # Index the fixed history prefix once.  The previous implementation
    # scanned every historical simplex for every candidate, which made the
    # complete 60% NDC run spend most of its time before GUDHI was called.
    history_simplices_by_node = defaultdict(list)
    for old_simplex in old_simplices:
        simplex = tuple(sorted(set(old_simplex)))
        if len(simplex) >= 3:
            for node in simplex:
                history_simplices_by_node[int(node)].append(simplex)

    pyg_samples = []
    valid_triangles = []
    skipped = 0
    positive_count = 0
    iterator = tqdm(
        sampled_candidates,
        desc=f"{dataset} {split} graphs",
        leave=False,
        disable=True,
    )
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
            if (
                not nx.is_connected(subgraph_without_target)
                and args.disconnected_local_policy == "skip"
            ):
                skipped += 1
                continue

            message_graph = subgraph if args.keep_target_links else subgraph_without_target
            local_old_simplices = set()
            local_nodes = set(subgraph.nodes)
            for node in local_nodes:
                for simplex in history_simplices_by_node.get(int(node), ()):
                    if sum(int(vertex in local_nodes) for vertex in simplex) >= 3:
                        local_old_simplices.add(simplex)
            labeled = construct_labeled_networkx_graph(
                message_graph,
                local_old_simplices,
                tri,
                list(positives),
                labelling=args.labelling_method,
                disconnected_policy=args.disconnected_local_policy,
                label_graph=subgraph_without_target if args.keep_target_links else None,
                observed_2_skeleton=args.observed_2_skeleton,
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
    keep_tag = f"_keep{int(args.keep_target_links)}" if args.keep_target_links else ""
    cache_schema = "v4" if split == "test" else "v3"
    name = (
        f"{cache_schema}_{dataset}_{split}_seed{effective_sampling_seed(args)}_hop{args.num_hops}"
        f"_maxstep{args.max_nodes_per_step}_cap{int(args.enforce_node_cap)}"
        f"_{sampling_mode}_pf{positive_fraction}"
        f"_raw{raw_cap}_n{complex_cap}_ratio{args.sample_ratio:g}"
        f"_dim{args.max_dim}_{args.init_method}_{args.labelling_method}"
        f"_disc{args.disconnected_local_policy}_{rw_tag}"
        f"_obs2{int(args.observed_2_skeleton)}"
        f"{keep_tag}"
        f"_src{source_fingerprint}.pt"
    )
    # Windows still rejects long paths in some PyTorch file operations even
    # when the process itself supports long paths.  Keep the full cache key in
    # the digest while using a compact filename on disk.
    full_name = name
    if os.name == "nt" and len(str(cache_dir / name)) >= 240:
        token = hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
        name = f"processed_{dataset}_{split}_{token}.pt"
    # Retain existing caches; new cache files use the complete key's digest.
    legacy_path = cache_dir / name
    portable_name = "c_" + hashlib.sha256(full_name.encode("utf-8")).hexdigest()[:24] + ".pt"
    portable_path = cache_dir / portable_name
    return legacy_path if legacy_path.is_file() else portable_path


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
    keep_tag = f"_keep{int(args.keep_target_links)}" if args.keep_target_links else ""
    name = (
        f"v6strength_{dataset}_{split}_seed{effective_sampling_seed(args)}_hop{args.num_hops}"
        f"_maxstep{args.max_nodes_per_step}_cap{int(args.enforce_node_cap)}"
        f"_{sampling_mode}_pf{positive_fraction}"
        f"_raw{raw_cap}_n{complex_cap}_ratio{args.sample_ratio:g}"
        f"_dim{args.max_dim}_{args.init_method}_{args.labelling_method}"
        f"_disc{args.disconnected_local_policy}"
        f"_obs2{int(args.observed_2_skeleton)}"
        f"{keep_tag}"
        f"_src{source_fingerprint}.pt"
    )
    full_name = name
    if os.name == "nt" and len(str(cache_dir / name)) >= 240:
        token = hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
        name = f"structural_{dataset}_{split}_{token}.pt"
    # Retain existing caches; new cache files use the complete key's digest.
    legacy_path = cache_dir / name
    portable_name = "c_" + hashlib.sha256(full_name.encode("utf-8")).hexdigest()[:24] + ".pt"
    portable_path = cache_dir / portable_name
    return legacy_path if legacy_path.is_file() else portable_path


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
    if split == "test":
        args.test_candidate_triplets_for_export = np.asarray(triangles, dtype=np.int64)
    if len(complexes) != len(triangles):
        raise ValueError(
            f"Complex/triangle alignment mismatch for {split}: "
            f"{len(complexes)} != {len(triangles)}"
        )
    for triangle, complex_data in zip(triangles, complexes):
        triangle = tuple(map(int, triangle))
        strength_map = args.interaction_strength_maps[split]
        cochain0 = complex_data.cochains[0]
        if not hasattr(cochain0, "original_node_ids"):
            raise RuntimeError(
                "Structural cache predates interaction-strength node alignment; "
                "rerun with --rebuild-cache."
            )
        if args.node_embedding_maps:
            mapping = args.node_embedding_maps["train" if split in ("train", "early") else "test"]
            node_ids = [int(value) for value in cochain0.original_node_ids.view(-1).tolist()]
            if any(node not in mapping for node in node_ids):
                raise KeyError(f"Missing node embedding for {split}")
            cochain0.x = torch.from_numpy(np.stack([mapping[node] for node in node_ids]).astype(np.float32))
            for dim in range(1, len(complex_data.cochains)):
                boundary = complex_data.cochains[dim].boundary_index
                if boundary is None:
                    raise ValueError(f"Missing dimension-{dim} boundary for embedding initialization")
                previous = complex_data.cochains[dim - 1].x
                count = complex_data.cochains[dim].x.size(0)
                values = torch.zeros((count, previous.size(1)), dtype=previous.dtype)
                values.index_add_(0, boundary[1], previous[boundary[0]])
                complex_data.cochains[dim].x = values / float(dim)
        upper_index = cochain0.upper_index
        if upper_index is not None:
            original_ids = cochain0.original_node_ids.view(-1).tolist()
            weights = [
                strength_value(
                    strength_map.get(
                        tuple(sorted((int(original_ids[int(src)]), int(original_ids[int(dst)]))))
                    ),
                    args.adjacency_mode,
                )
                for src, dst in upper_index.t().tolist()
            ]
            cochain0.upper_weight = torch.tensor(weights, dtype=torch.float32)
        matrix, names, _ = candidate_strength_matrix([triangle], strength_map)
        log_indices = [index for index, name in enumerate(names) if name.startswith("log_strength_")]
        log_matrix = matrix[:, log_indices]
        if args.use_strength_feature:
            mean = np.asarray(args.strength_feature_mean, dtype=np.float32).reshape(1, -1)
            std = np.asarray(args.strength_feature_std, dtype=np.float32).reshape(1, -1)
            log_matrix = (log_matrix - mean) / std
        cochain0.interaction_strength_feature = torch.from_numpy(
            np.asarray(log_matrix, dtype=np.float32)
        )
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
        if args.temporal_activity_alpha > 0 or args.validation_select_activity_alpha:
            activity_map = args.temporal_activity_maps[split]
            if triangle not in activity_map:
                raise KeyError(
                    f"Triangle missing from temporal activity map for {split}: "
                    f"{triangle}"
                )
            complex_data.cochains[0].temporal_activity_feature = torch.tensor(
                [[float(activity_map[triangle])]], dtype=torch.float32
            )
        if args.history_feature_mode != "none":
            history_map = args.history_feature_maps[split]
            if triangle not in history_map:
                raise KeyError(f"Triangle missing from history feature map for {split}: {triangle}")
            cochain0.history_feature = torch.from_numpy(
                np.asarray(history_map[triangle], dtype=np.float32)
            ).view(1, -1)


def load_or_build_complexes(dataset, split, args):
    path = structural_cache_path(dataset, split, args)
    read_path = path
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
    complex_limit = max(1, args.max_dim)
    for sample in pyg_samples:
        sample.simplices = [simplex for simplex in sample.simplices if len(simplex) <= complex_limit + 1]
    complexes = convert_graph_dataset_with_gudhi(
        pyg_samples,
        expansion_dim=complex_limit,
        include_down_adj=False,
        init_method=args.init_method,
        expand_cliques=not args.observed_2_skeleton,
    )
    # Keep the large structural cache independent of RW parameters.
    for complex_data in complexes:
        if hasattr(complex_data.cochains[0], "rw_feature"):
            delattr(complex_data.cochains[0], "rw_feature")
        if hasattr(complex_data.cochains[0], "history_feature"):
            delattr(complex_data.cochains[0], "history_feature")
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


def require_finite_tensor(value, context):
    if not torch.isfinite(value).all().item():
        finite = value.detach()[torch.isfinite(value.detach())]
        raise FloatingPointError(
            f"Non-finite {context}: shape={tuple(value.shape)} "
            f"finite={int(finite.numel())}/{value.numel()}"
        )


def compute_loss(logits, labels, args, class_weights=None):
    require_finite_tensor(logits, "MLSHOL logits")
    labels = labels.view(-1).long()
    if args.loss_mode in {"bce", "cross_entropy"}:
        loss = F.cross_entropy(logits, labels)
        require_finite_tensor(loss, "MLSHOL loss")
        return loss
    if args.loss_mode in {"weighted_bce", "weighted_cross_entropy"}:
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        require_finite_tensor(loss, "MLSHOL weighted loss")
        return loss
    if args.loss_mode == "focal":
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        row_index = torch.arange(labels.numel(), device=labels.device)
        pt = prob[row_index, labels].clamp(1e-6, 1 - 1e-6)
        ce = -log_prob[row_index, labels]
        focal = (1 - pt).pow(args.focal_gamma) * ce
        if class_weights is not None:
            focal = focal * class_weights[labels]
        loss = focal.mean()
        require_finite_tensor(loss, "MLSHOL focal loss")
        return loss
    raise ValueError(f"Unsupported loss mode: {args.loss_mode}")


def compute_pairwise_aux_loss(logits, labels, margin=0.0, max_pairs=4096):
    """A small ranking regularizer used only on fit batches."""
    labels = labels.view(-1).long()
    positive = labels == 1
    negative = labels == 0
    if not bool(positive.any()) or not bool(negative.any()):
        return logits.new_zeros(())
    scores = logits[:, 1] - logits[:, 0]
    pos_scores = scores[positive]
    neg_scores = scores[negative]
    pair_count = min(int(max_pairs), int(pos_scores.numel()) * int(neg_scores.numel()))
    if pair_count <= 0:
        return logits.new_zeros(())
    pos_index = torch.randint(pos_scores.numel(), (pair_count,), device=logits.device)
    neg_index = torch.randint(neg_scores.numel(), (pair_count,), device=logits.device)
    return F.softplus(float(margin) - pos_scores[pos_index] + neg_scores[neg_index]).mean()


def evaluate_model(model, loader, device, args, class_weights=None, fusion_alpha=None, return_arrays=False):
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true = []
    y_prob = []
    y_pred = []
    activity_values = []
    unfused_values = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data, return_logits=True)
            require_finite_tensor(logits, "MLSHOL evaluation logits")
            labels = data.y.view(-1).long()
            loss = compute_loss(logits, labels, args, class_weights)
            total_loss += loss.item() * labels.numel()
            total_items += labels.numel()
            probabilities = F.softmax(logits, dim=1)
            require_finite_tensor(probabilities, "MLSHOL probabilities")
            probs = probabilities[:, 1].detach().cpu().numpy()
            if return_arrays:
                unfused_values.extend(probs.tolist())
            if not np.isfinite(probs).all():
                raise FloatingPointError("Non-finite MLSHOL positive probabilities")
            if args.temporal_activity_alpha > 0 or args.validation_select_activity_alpha:
                activity = (
                    data.cochains[0]
                    .temporal_activity_feature.view(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                activity_values.extend(activity.tolist())
                alpha = float(args.temporal_activity_alpha if fusion_alpha is None else fusion_alpha)
                if not args.validation_select_activity_alpha or fusion_alpha is not None:
                    probs = (1.0 - alpha) * probs + alpha * activity
                    if not np.isfinite(probs).all():
                        raise FloatingPointError("Non-finite MLSHOL fused probabilities")
            preds = (probs >= 0.5).astype(np.int64)
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    baseline = float(y_true_np.mean()) if len(y_true_np) else 0.0
    selected_activity_alpha = float(
        args.temporal_activity_alpha if fusion_alpha is None else fusion_alpha
    )
    if args.validation_select_activity_alpha and fusion_alpha is None:
        activity_np = np.asarray(activity_values, dtype=np.float64)
        best_ap = -1.0
        best_probs = y_prob_np
        for alpha in np.linspace(0.0, 1.0, 21):
            fused = (1.0 - alpha) * y_prob_np + alpha * activity_np
            ap = float(met.average_precision_score(y_true_np, fused))
            if ap > best_ap:
                best_ap = ap
                selected_activity_alpha = float(alpha)
                best_probs = fused
        y_prob_np = best_probs
        y_pred_np = (y_prob_np >= 0.5).astype(np.int64)

    if len(np.unique(y_true_np)) == 2:
        # Average precision is the only PR-area metric used by this runner.
        auc_pr = float(met.average_precision_score(y_true_np, y_prob_np))
        roc_auc = float(met.roc_auc_score(y_true_np, y_prob_np))
        f1 = float(met.f1_score(y_true_np, y_pred_np, zero_division=0))
        precision_score = float(met.precision_score(y_true_np, y_pred_np, zero_division=0))
        recall_score = float(met.recall_score(y_true_np, y_pred_np, zero_division=0))
    else:
        auc_pr = float("nan")
        roc_auc = float("nan")
        f1 = float("nan")
        precision_score = float("nan")
        recall_score = float("nan")

    acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) else float("nan")
    class0_mask = y_true_np == 0
    class1_mask = y_true_np == 1
    acc0 = float((y_pred_np[class0_mask] == 0).mean()) if class0_mask.any() else float("nan")
    acc1 = float((y_pred_np[class1_mask] == 1).mean()) if class1_mask.any() else float("nan")
    result = {
        "loss": total_loss / max(total_items, 1),
        "auc_pr": auc_pr,
        "ap": auc_pr,
        "roc_auc": roc_auc,
        "baseline": baseline,
        "lift": auc_pr / baseline if baseline > 0 and not np.isnan(auc_pr) else float("nan"),
        "selected_activity_alpha": selected_activity_alpha,
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
    if return_arrays:
        result["_y_true"] = y_true_np
        result["_y_score"] = y_prob_np
        result["_y_score_unfused"] = np.asarray(unfused_values, dtype=np.float64)
    return result


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
    if args.rw_fusion == "none" and args.history_feature_mode == "none" and not args.use_strength_feature:
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
            require_finite_tensor(logits, "MLSHOL training logits")
            labels = data.y.view(-1).long()
            loss = compute_loss(logits, labels, args, class_weights)
            if args.pairwise_aux_loss and args.pairwise_aux_weight > 0:
                loss = loss + args.pairwise_aux_weight * compute_pairwise_aux_loss(
                    logits,
                    labels,
                    margin=args.pairwise_aux_margin,
                    max_pairs=args.pairwise_aux_max_pairs,
                )
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    require_finite_tensor(parameter.grad, "MLSHOL training gradient")
            if args.grad_clip is not None and args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm).all().item():
                    raise FloatingPointError("Non-finite MLSHOL gradient norm")
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
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    feature_started = time.perf_counter()
    if args.temporal_validation:
        ensure_temporal_early_protocol(dataset, args)

    args.history_feature_maps = {}
    args.history_feature_dim = 0
    if args.history_feature_mode != "none":
        if not args.history_feature_npz:
            raise ValueError("--history-feature-npz is required when history features are enabled")
        with np.load(args.history_feature_npz, allow_pickle=False) as payload:
            train_candidates = np.asarray(payload["train_candidates"], dtype=np.int64)
            test_candidates = np.asarray(payload["test_candidates"], dtype=np.int64)
            train_x = np.asarray(payload["train_x"], dtype=np.float32)
            test_x = np.asarray(payload["test_x"], dtype=np.float32)
            if train_x.shape[0] != train_candidates.shape[0] or test_x.shape[0] != test_candidates.shape[0]:
                raise ValueError("History feature row/candidate mismatch")
            if train_x.shape[1] != test_x.shape[1] or train_x.shape[1] <= 0:
                raise ValueError("Invalid history feature dimensions")
            if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
                raise ValueError("Non-finite history feature input")
            args.history_feature_dim = int(train_x.shape[1])
            train_map = {
                tuple(map(int, triangle)): vector
                for triangle, vector in zip(train_candidates, train_x)
            }
            test_map = {
                tuple(map(int, triangle)): vector
                for triangle, vector in zip(test_candidates, test_x)
            }
            args.history_feature_names = [str(value) for value in payload["feature_names"].tolist()]
        args.history_feature_maps = {"train": train_map, "test": test_map, "early": train_map}
        audit = {
            "mode": args.history_feature_mode,
            "feature_dim": args.history_feature_dim,
            "feature_names": args.history_feature_names,
            "npz": str(Path(args.history_feature_npz).resolve()),
            "train_candidates": len(train_map),
            "test_candidates": len(test_map),
        }
        (Path(args.output_dir) / f"{dataset}_history_feature_audit.json").parent.mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / f"{dataset}_history_feature_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )

    args.node_embedding_maps = {}
    if args.node_embedding_train_npz or args.node_embedding_test_npz:
        if not (args.node_embedding_train_npz and args.node_embedding_test_npz):
            raise ValueError("Both node embedding files are required")
        for split, path in (("train", args.node_embedding_train_npz), ("test", args.node_embedding_test_npz)):
            with np.load(path, allow_pickle=False) as payload:
                nodes = np.asarray(payload["nodes"], dtype=np.int64)
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
            if vectors.shape != (len(nodes), args.num_input_features) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid {split} node embedding shape or values")
            args.node_embedding_maps[split] = dict(zip(nodes.tolist(), vectors))
    strength_splits = ["train", "test"] + (["early"] if args.temporal_validation else [])
    strength_payloads = {
        split: build_interaction_strength_map(dataset, split, args)
        for split in strength_splits
    }
    args.interaction_strength_maps = {
        split: payload[0] for split, payload in strength_payloads.items()
    }
    train_strength_candidates = read_triangles(split_paths(dataset, "train", args)["open"])
    train_strength_matrix, train_strength_names, _ = candidate_strength_matrix(
        train_strength_candidates, args.interaction_strength_maps["train"]
    )
    train_log_indices = [
        index for index, name in enumerate(train_strength_names)
        if name.startswith("log_strength_")
    ]
    train_log_strength = train_strength_matrix[:, train_log_indices]
    args.strength_feature_mean = train_log_strength.mean(axis=0).astype(np.float32)
    args.strength_feature_std = train_log_strength.std(axis=0).astype(np.float32)
    args.strength_feature_std = np.where(
        args.strength_feature_std > 1e-12,
        args.strength_feature_std,
        np.float32(1.0),
    ).astype(np.float32)
    test_strength, test_cutoff, test_events = strength_payloads["test"]
    audit_rows = [
        {
            "dataset": dataset,
            "edge_id": edge_id,
            "u": edge[0],
            "v": edge[1],
            "cutoff": test_cutoff,
            "count": record.count,
            "log_count": record.log_count,
            "decay": record.decay,
            "recency": record.recency,
            "last_time": record.last_time,
            "strictly_pre_cutoff": int(record.last_time < test_cutoff),
        }
        for edge_id, (edge, record) in enumerate(sorted(test_strength.items()))
    ]
    write_rows(Path(args.output_dir) / f"{dataset}_interaction_strength_audit.csv", audit_rows)
    test_candidates = read_triangles(split_paths(dataset, "test", args)["open"])
    test_positives = set(read_triangles(split_paths(dataset, "test", args)["positive"]))
    _, _, candidate_rows = candidate_strength_matrix(test_candidates, test_strength)
    write_rows(
        Path(args.output_dir) / f"{dataset}_candidate_strength_features.csv",
        [
            {"dataset": dataset, **row, "label_for_audit_only": int(triangle in test_positives)}
            for triangle, row in zip(test_candidates, candidate_rows)
        ],
    )
    write_rows(
        Path(args.output_dir) / f"{dataset}_dataset_strength_statistics.csv",
        [{"dataset": dataset, **dataset_strength_statistics(test_events, test_cutoff, args.strength_decay_lambda)}],
    )
    feature_seconds = time.perf_counter() - feature_started

    rw_started = time.perf_counter()
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
    rw_seconds = time.perf_counter() - rw_started
    if args.temporal_activity_alpha > 0 or args.validation_select_activity_alpha:
        activity_splits = ["train", "test"]
        if args.temporal_validation:
            activity_splits.append("early")
        args.temporal_activity_maps = {
            split: build_temporal_activity_map(dataset, split, args)
            for split in activity_splits
        }
    else:
        args.temporal_activity_maps = {}

    complex_started = time.perf_counter()
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
    complex_seconds = time.perf_counter() - complex_started
    if not train_complexes or (not args.validation_only and not test_complexes):
        raise RuntimeError(f"{dataset}: no valid train/test complexes")

    if args.temporal_validation:
        if not fit_complexes or not val_complexes:
            raise RuntimeError(f"{dataset}: no valid temporal fit/validation complexes")
    elif args.no_validation:
        # Final refit mode: the caller has already frozen the configuration and
        # epoch budget on an external development window.  Do not create a
        # second hidden split or select a checkpoint from labels in this task.
        fit_complexes = list(train_complexes)
        val_complexes = []
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
    val_loader = DataLoader(
        val_complexes if val_complexes else fit_complexes,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = (
        None
        if args.validation_only
        else DataLoader(test_complexes, batch_size=args.batch_size, shuffle=False)
    )

    model = build_classifier(args, device)
    loaded_checkpoint = None
    if args.load_checkpoint:
        loaded_checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        if isinstance(loaded_checkpoint, dict) and "model_state_dict" in loaded_checkpoint:
            loaded_checkpoint = loaded_checkpoint["model_state_dict"]
        model.load_state_dict(loaded_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    class_weights = (
        make_class_weights(fit_complexes, device)
        if args.loss_mode in {"weighted_bce", "weighted_cross_entropy", "focal"}
        else None
    )

    best = None
    best_state = None
    local_branch_grad_norm = None
    local_branch_checked = False
    if args.rw_fusion == "rw_only":
        raise RuntimeError("MLSHOL_LOCAL_BRANCH_DISCONNECTED: rw_only is forbidden")
    training_started = time.perf_counter()
    if args.load_checkpoint:
        best = {
            "epoch": args.epochs,
            "auc_pr": float("nan"),
            "lift": float("nan"),
            "train_loss": float("nan"),
        }
        best_state = copy.deepcopy(model.state_dict())
        training_seconds = 0.0
        print(
            f"{dataset} checkpoint_loaded={args.load_checkpoint} "
            "checkpoint_selection=external_pre_frozen_epoch",
            flush=True,
        )
    else:
      for epoch in range(1, args.epochs + 1):
          if (
              args.history_feature_mode == "history_base_structural_residual"
              and getattr(model, "history_pretrained", False)
              and hasattr(model, "history_pretrained_net")
          ):
              freeze = epoch <= int(getattr(args, "history_freeze_epochs", 0))
              for parameter in model.history_pretrained_net.parameters():
                  parameter.requires_grad_(not freeze)
              print(
                  f"history_pretrained_frozen={freeze} epoch={epoch}",
                  flush=True,
              )
          model.train()
          if (
              args.history_feature_mode == "history_base_structural_residual"
              and getattr(model, "history_pretrained", False)
              and hasattr(model, "history_pretrained_net")
              and freeze
          ):
              # model.train() recursively re-enables dropout.  A frozen
              # pretrained history head must also stay in eval mode so the
              # residual starts from the exact saved H ranking.
              model.history_pretrained_net.eval()
          train_loss = 0.0
          train_items = 0
          for data in train_loader:
              data = data.to(device)
              optimizer.zero_grad()
              logits = model(data, return_logits=True)
              labels = data.y.view(-1).long()
              loss = compute_loss(logits, labels, args, class_weights)
              if args.pairwise_aux_loss and args.pairwise_aux_weight > 0:
                  loss = loss + args.pairwise_aux_weight * compute_pairwise_aux_loss(
                      logits,
                      labels,
                      margin=args.pairwise_aux_margin,
                      max_pairs=args.pairwise_aux_max_pairs,
                  )
              loss.backward()
              if not local_branch_checked:
                if not hasattr(model, "sin"):
                    raise RuntimeError("MLSHOL_LOCAL_BRANCH_DISCONNECTED: missing local encoder")
                local_branch_grad_norm = float(sum(
                    parameter.grad.detach().pow(2).sum().item()
                    for parameter in model.sin.parameters()
                    if parameter.grad is not None
                ) ** 0.5)
                if not np.isfinite(local_branch_grad_norm) or local_branch_grad_norm <= 0:
                    # The history-base/structural-residual head deliberately
                    # starts with a zero residual projection.  Its first
                    # backward pass therefore has zero gradient through the
                    # structural branch by construction; after optimizer.step
                    # the next batch must have a nonzero gradient.  Defer the
                    # connectivity assertion for that one warm-up batch.
                    if args.history_feature_mode == "history_base_structural_residual":
                        print(
                            "MLSHOL_LOCAL_BRANCH_GRAD_DEFERRED_ZERO_RESIDUAL=YES",
                            flush=True,
                        )
                    else:
                        raise RuntimeError(
                            "MLSHOL_LOCAL_BRANCH_DISCONNECTED: local branch gradient is zero"
                        )
                else:
                    local_branch_checked = True
                    print(
                        "MLSHOL_LOCAL_OUTPUT_ENTERS_CLASSIFIER=YES "
                        "RW_ONLY_CLASSIFIER=NO "
                        f"LOCAL_BRANCH_GRAD_NORM={local_branch_grad_norm:.12g}",
                        flush=True,
                    )
              if args.grad_clip is not None and args.grad_clip > 0:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
              optimizer.step()
              train_loss += loss.item() * labels.numel()
              train_items += labels.numel()
          scheduler.step()

          if epoch == args.epochs or epoch % args.eval_every == 0:
              if args.no_validation:
                  best = {
                    "epoch": epoch,
                    "auc_pr": float("nan"),
                    "lift": float("nan"),
                    "train_loss": train_loss / max(train_items, 1),
                }
                  best_state = copy.deepcopy(model.state_dict())
                  print(
                    f"{dataset} epoch={epoch} train_loss={best['train_loss']:.6f} "
                    "checkpoint_selection=pre_frozen_epoch",
                    flush=True,
                )
              else:
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

    if not args.load_checkpoint:
        training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError(f"{dataset}: validation did not produce a selectable checkpoint")
    model.load_state_dict(best_state)
    if args.save_checkpoint:
        checkpoint_path = Path(args.save_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dataset": dataset,
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "protocol": "0-60 history / 60-80 development",
            },
            checkpoint_path,
        )
        print(f"checkpoint_saved={checkpoint_path}", flush=True)
    final_retrain_epochs = 0
    inference_started = time.perf_counter()
    if args.validation_only:
        final_metrics = evaluate_model(model, val_loader, device, args, class_weights)
        evaluation_scope = "validation"
    elif args.no_validation:
        final_metrics = evaluate_model(
            model,
            test_loader,
            device,
            args,
            class_weights,
            return_arrays=True,
        )
        evaluation_scope = "test"
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
            fusion_alpha=best.get("selected_activity_alpha"),
            return_arrays=True,
        )
        evaluation_scope = "test"
    else:
        final_metrics = evaluate_model(
            model,
            test_loader,
            device,
            args,
            class_weights,
            fusion_alpha=best.get("selected_activity_alpha"),
            return_arrays=True,
        )
        evaluation_scope = "test"
    inference_seconds = time.perf_counter() - inference_started
    if evaluation_scope == "test" and "_y_true" in final_metrics:
        per_seed_score_path = Path(args.output_dir) / f"{dataset}_test_scores_seed_{args.seed}.npz"
        np.savez_compressed(
            per_seed_score_path,
            y_true=final_metrics.pop("_y_true"),
            y_score=final_metrics.pop("_y_score"),
            y_score_unfused=final_metrics.pop("_y_score_unfused"),
            target_triplets=args.test_candidate_triplets_for_export,
        )
        (Path(args.output_dir) / f"{dataset}_test_scores.npz").write_bytes(
            per_seed_score_path.read_bytes()
        )
    final_metrics["train_seconds"] = training_seconds
    final_metrics["feature_seconds"] = feature_seconds
    final_metrics["rw_seconds"] = rw_seconds
    final_metrics["subgraph_extraction_seconds"] = complex_seconds
    final_metrics["inference_seconds"] = inference_seconds
    final_metrics["completed_epochs"] = args.epochs
    final_metrics["planned_epochs"] = args.epochs
    final_metrics["mean_epoch_seconds"] = training_seconds / max(args.epochs, 1)
    final_metrics["peak_gpu_allocated_gb"] = (
        float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    final_metrics["peak_gpu_reserved_gb"] = (
        float(torch.cuda.max_memory_reserved(device)) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    final_metrics["checkpoint_selection_source"] = (
        "pre_frozen_epoch" if args.no_validation else "validation_auc_pr"
    )
    final_metrics["TEST_USED_FOR_CHECKPOINT_SELECTION"] = False
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
    final_metrics["MLSHOL_FUSION_MODE"] = args.rw_fusion
    final_metrics["MLSHOL_DISCONNECTED_LOCAL_POLICY"] = args.disconnected_local_policy
    final_metrics["observed_2_skeleton"] = bool(args.observed_2_skeleton)
    final_metrics["train_history_context_ratio"] = args.train_history_context_ratio
    final_metrics["test_history_context_ratio"] = args.test_history_context_ratio
    final_metrics["MLSHOL_LOCAL_OUTPUT_ENTERS_CLASSIFIER"] = bool(
        args.history_feature_mode != "history_only" and args.rw_fusion != "rw_only"
    )
    final_metrics["RW_ONLY_CLASSIFIER"] = False
    final_metrics["LOCAL_BRANCH_GRAD_NORM"] = local_branch_grad_norm
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
    final_metrics["rw_max_pairs"] = args.rw_max_pairs
    final_metrics["rw_priority_multiplier"] = args.rw_priority_multiplier
    final_metrics["temporal_activity_alpha"] = args.temporal_activity_alpha
    final_metrics["adjacency_mode"] = args.adjacency_mode
    final_metrics["strength_feature"] = bool(args.use_strength_feature)
    final_metrics["strength_edge"] = bool(args.use_strength_edge_weight)
    final_metrics["activity_prior"] = bool(args.use_temporal_score_prior)
    final_metrics["strength_decay_lambda"] = args.strength_decay_lambda
    final_metrics[
        "temporal_activity_recent_fraction"
    ] = args.temporal_activity_recent_fraction
    final_metrics["temporal_activity_context_ratio"] = args.temporal_activity_context_ratio
    final_metrics["temporal_activity_measure"] = args.temporal_activity_measure
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
        "MLSHOL_FUSION_MODE",
        "MLSHOL_DISCONNECTED_LOCAL_POLICY",
        "observed_2_skeleton",
        "train_history_context_ratio",
        "test_history_context_ratio",
        "MLSHOL_LOCAL_OUTPUT_ENTERS_CLASSIFIER",
        "RW_ONLY_CLASSIFIER",
        "LOCAL_BRANCH_GRAD_NORM",
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
        "rw_max_pairs",
        "rw_priority_multiplier",
        "temporal_activity_alpha",
        "selected_activity_alpha",
        "adjacency_mode",
        "strength_feature",
        "strength_edge",
        "activity_prior",
        "strength_decay_lambda",
        "temporal_activity_recent_fraction",
        "temporal_activity_aggregation",
        "temporal_activity_entity",
        "maxsimplex_prior_weight",
        "maxsimplex_prior_column",
        "temporal_validation",
        "final_retrain_epochs",
        "auc_pr",
        "ap",
        "roc_auc",
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
        "feature_seconds",
        "rw_seconds",
        "subgraph_extraction_seconds",
        "inference_seconds",
        "completed_epochs",
        "planned_epochs",
        "mean_epoch_seconds",
        "peak_gpu_allocated_gb",
        "peak_gpu_reserved_gb",
        "checkpoint_selection_source",
        "TEST_USED_FOR_CHECKPOINT_SELECTION",
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
    parser.add_argument("--data-root", default=str(ROOT_DIR))
    parser.add_argument(
        "--event-order", choices=["raw", "timestamp"], default="raw"
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(ROOT_DIR) / "exp" / "results" / "email_enron_mlshol_dual_rw_seed44"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Use all sampled training complexes and the pre-frozen epoch budget; never split or select on this task.",
    )
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
    parser.add_argument(
        "--observed-2-skeleton",
        action="store_true",
        help=(
            "Use the cutoff-observed downward-closed 2-skeleton and disable "
            "GUDHI clique expansion."
        ),
    )
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
    parser.add_argument("--pairwise-aux-loss", action="store_true")
    parser.add_argument("--pairwise-aux-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-aux-margin", type=float, default=0.0)
    parser.add_argument("--pairwise-aux-max-pairs", type=int, default=4096)
    parser.add_argument("--labelling-method", default="TRNL")
    parser.add_argument("--node-embedding-train-npz", default=None)
    parser.add_argument("--node-embedding-test-npz", default=None)
    parser.add_argument(
        "--disconnected-local-policy",
        choices=["skip", "unreachable_zero"],
        default="skip",
        help="TRNL policy for disconnected target-removed sampled neighborhoods.",
    )
    parser.add_argument(
        "--keep-target-links",
        action="store_true",
        help=(
            "Keep the three known K=2 target edges in the message-passing graph; "
            "TRNL labels and connectivity checks still use the target-removed view."
        ),
    )
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
    parser.add_argument(
        "--rw-max-pairs",
        type=int,
        default=20_000_000,
        help="Hard cap on preallocated random-walk positive pairs.",
    )
    parser.add_argument("--rw-priority-multiplier", type=int, default=5)
    parser.add_argument("--rw-lr", type=float, default=0.01)
    parser.add_argument("--rw-batch-size", type=int, default=8192)
    parser.add_argument("--rw-chunk-starts", type=int, default=4096)
    parser.add_argument("--temporal-activity-alpha", type=float, default=0.0)
    parser.add_argument("--use-strength-feature", action="store_true")
    parser.add_argument("--use-strength-edge-weight", action="store_true")
    parser.add_argument("--train-strength-context-ratio", type=float, default=None)
    parser.add_argument("--test-strength-context-ratio", type=float, default=None)
    parser.add_argument(
        "--train-history-context-ratio", type=float, default=0.6,
        help="History prefix used for train structural complexes.",
    )
    parser.add_argument(
        "--test-history-context-ratio", type=float, default=0.8,
        help="History prefix used for test/validation structural complexes.",
    )
    parser.add_argument("--history-feature-npz", default=None)
    parser.add_argument(
        "--history-pretrained-checkpoint",
        default=None,
        help="Load an exact HistoryMLP checkpoint for history_base_structural_residual.",
    )
    parser.add_argument(
        "--history-freeze-epochs",
        type=int,
        default=0,
        help="Freeze the loaded HistoryMLP for this many joint-training epochs.",
    )
    parser.add_argument(
        "--history-feature-mode", choices=[
            "none", "concat", "residual", "gated_residual",
            "history_base_structural_residual", "history_only",
        ], default="none"
    )
    parser.add_argument("--history-residual-scale", type=float, default=1.0)
    parser.add_argument("--use-temporal-score-prior", action="store_true")
    parser.add_argument(
        "--validation-select-activity-alpha",
        action="store_true",
        help="Select the convex model/activity weight on validation AP only, then freeze it for test.",
    )
    parser.add_argument(
        "--adjacency-mode",
        choices=["binary", "weighted-unit", "weighted-count", "weighted-log", "weighted-decay"],
        default="binary",
        help="Representation-level historical edge weighting used inside SparseSIN messages.",
    )
    parser.add_argument(
        "--strength-decay-lambda", type=float, default=0.0,
        help="Decay rate in timestamp units; <=0 chooses a unit-safe 15%-span half-life.",
    )
    parser.add_argument(
        "--temporal-activity-recent-fraction", type=float, default=0.15
    )
    parser.add_argument(
        "--temporal-activity-context-ratio", type=float, default=0.8,
        help="Final activity context cutoff; 1.0 is transductive and uses post-cutoff raw events.",
    )
    parser.add_argument(
        "--temporal-activity-measure", choices=["count", "last_event"], default="count",
        help="count uses recent frequency; last_event ranks latest raw-event indices.",
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
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="Save the frozen model state after training for later sharded inference.",
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="Load an externally frozen model state and skip optimization.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, cuda:1, ...",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global DATA_ROOT, EVENT_ORDER
    DATA_ROOT = Path(args.data_root).resolve()
    EVENT_ORDER = str(args.event_order)
    if not args.use_temporal_score_prior:
        if args.temporal_activity_alpha != 0.0:
            print(
                "Temporal alpha supplied while --use-temporal-score-prior is off; "
                "score-level prior is disabled.",
                flush=True,
            )
        args.temporal_activity_alpha = 0.0
        args.validation_select_activity_alpha = False
    if args.use_strength_edge_weight and args.adjacency_mode == "binary":
        raise ValueError(
            "--use-strength-edge-weight requires a weighted --adjacency-mode"
        )
    if not args.use_strength_edge_weight and args.adjacency_mode != "binary":
        raise ValueError(
            "weighted --adjacency-mode requires --use-strength-edge-weight"
        )
    requested_device = str(args.device).strip().lower()
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    device = torch.device(requested_device)
    print(f"device={device}", flush=True)
    rows = []
    failed = []
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
                import traceback
                traceback.print_exc()
                rows.append({"dataset": dataset, "seed": seed, "sampling": f"ERROR: {exc}"})
                failed.append(f"{dataset} seed={seed}: {exc}")
                write_results(rows, Path(args.output_dir) / "results.csv")
                print(f"{dataset} seed={seed} failed: {exc}", flush=True)
                print(
                    "Dataset failure was recorded; continuing with the remaining "
                    "datasets.",
                    flush=True,
                )

    write_summary(rows, Path(args.output_dir) / "summary.csv")
    print(f"wrote {Path(args.output_dir) / 'results.csv'}", flush=True)
    print(f"wrote {Path(args.output_dir) / 'summary.csv'}", flush=True)
    if failed and args.strict:
        raise RuntimeError("One or more dataset runs failed:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
