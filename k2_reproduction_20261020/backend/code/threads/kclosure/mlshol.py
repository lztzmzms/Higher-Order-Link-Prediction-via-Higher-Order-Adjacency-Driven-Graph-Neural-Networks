"""MLSHOL k=3 local-enclosing-complex smoke utilities."""

from __future__ import annotations

import math
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from kclosure.mrnl import generalized_mrnl_node_labeling, remove_target_edges
from mp.models import SparseSIN
from simplicial_complex.data_loading import DataLoader
from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _rss_bytes():
    return int(psutil.Process(os.getpid()).memory_info().rss) if psutil else 0


def _one_hop_nodes(graph: nx.Graph, target: Sequence[int], radius: int = 1) -> set:
    visited = set(target)
    frontier = set(target)
    for _ in range(radius):
        frontier = {neighbor for node in frontier for neighbor in graph.neighbors(node)} - visited
        visited.update(frontier)
    return visited


def prepare_local_sample_context(task):
    """Build immutable full-history lookup structures once per task.

    The previous implementation rebuilt the full NetworkX graph and scanned
    every observed simplex for every sampled candidate.  That is equivalent
    mathematically, but becomes prohibitive on threads-math-sx.  Anchoring each
    simplex at its smallest node lets a local sample inspect only simplices
    incident to its local node set while preserving the original sorted order.
    """
    state = task["state"]
    graph = nx.Graph()
    graph.add_edges_from(state["simplices_by_size"][2])
    simplices_by_anchor = {}
    for size in range(2, 5):
        anchored = {}
        for simplex in sorted(state["simplices_by_size"][size]):
            anchored.setdefault(simplex[0], []).append(simplex)
        simplices_by_anchor[size] = anchored
    return {
        "graph": graph,
        "simplices_by_anchor": simplices_by_anchor,
    }


def _local_simplices(context, local_nodes: set) -> List[Tuple[int, ...]]:
    selected = []
    lookup = context["simplices_by_anchor"]
    for size in range(2, 5):
        rows = []
        by_anchor = lookup[size]
        for node in local_nodes:
            for simplex in by_anchor.get(node, ()):
                if all(vertex in local_nodes for vertex in simplex):
                    rows.append(simplex)
        rows.sort()
        selected.extend(rows)
    return selected


def build_local_pyg_sample(task, candidate, label: int, radius: int = 1, context=None):
    state = task["state"]
    context = context or prepare_local_sample_context(task)
    graph = context["graph"]
    local_nodes = _one_hop_nodes(graph, candidate, radius=radius)
    local = graph.subgraph(local_nodes).copy()
    labeling_graph = remove_target_edges(local, candidate)
    if not nx.is_connected(labeling_graph):
        raise nx.NetworkXNoPath("disconnected local graph after target-edge removal")

    nodes, labels, vectors = generalized_mrnl_node_labeling(local, candidate)
    if nodes != list(labeling_graph.nodes):
        raise AssertionError("MRNL node ordering changed unexpectedly")
    node_to_local = {node: index for index, node in enumerate(nodes)}
    edge_pairs = [
        (node_to_local[u], node_to_local[v])
        for u, v in labeling_graph.edges
    ]
    directed = edge_pairs + [(v, u) for u, v in edge_pairs]
    edge_index = (
        torch.tensor(directed, dtype=torch.long).t().contiguous()
        if directed else torch.empty((2, 0), dtype=torch.long)
    )

    simplices: List[List[int]] = [
        [node_to_local[node] for node in simplex]
        for simplex in _local_simplices(context, local_nodes)
    ]
    data = Data(
        x=torch.tensor(labels, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([int(label)], dtype=torch.long),
        num_nodes=len(nodes),
        simplices=simplices,
        original_node_ids=torch.tensor(nodes, dtype=torch.long),
    )
    data.mrnl_distance_vectors = vectors
    return data


def build_complexes(task, cap: int | None = None, radius: int = 1):
    samples = []
    candidates = []
    skipped = []
    context = prepare_local_sample_context(task)
    candidate_rows = task["candidates"] if cap is None else task["candidates"][:cap]
    label_rows = task["labels"] if cap is None else task["labels"][:cap]
    for candidate, label in zip(candidate_rows, label_rows):
        try:
            samples.append(build_local_pyg_sample(task, candidate, int(label), radius=radius, context=context))
            candidates.append(candidate)
        except nx.NetworkXNoPath:
            skipped.append(candidate)
    complexes = convert_graph_dataset_with_gudhi(
        samples,
        expansion_dim=3,
        include_down_adj=False,
        init_method="sum",
        expand_cliques=False,
    )
    return complexes, candidates, skipped


def assert_b3(complex_data):
    if complex_data.dimension < 3:
        raise AssertionError("k=3 local enclosing complex contains no 3-simplices")
    num_triangles = int(complex_data.cochains[2].x.size(0))
    num_tetrahedra = int(complex_data.cochains[3].x.size(0))
    boundary_index = complex_data.cochains[3].boundary_index
    if boundary_index is None:
        raise AssertionError("B3 incidence is missing")
    if tuple(boundary_index.shape) != (2, 4 * num_tetrahedra):
        raise AssertionError(
            f"B3 incidence shape mismatch: {tuple(boundary_index.shape)} != (2, {4 * num_tetrahedra})"
        )
    if boundary_index.numel() and int(boundary_index[0].max()) >= num_triangles:
        raise AssertionError("B3 references a non-existent triangular face")
    values = torch.ones(boundary_index.size(1), dtype=torch.float32)
    b3 = torch.sparse_coo_tensor(
        boundary_index, values, size=(num_triangles, num_tetrahedra)
    ).coalesce()
    assert tuple(b3.shape) == (num_triangles, num_tetrahedra)
    counts = torch.bincount(boundary_index[1], minlength=num_tetrahedra)
    if not torch.all(counts == 4):
        raise AssertionError("every tetrahedron must have exactly four triangular boundaries")
    return {
        "B3_shape": [num_triangles, num_tetrahedra],
        "B3_incidence_entries": int(boundary_index.size(1)),
    }


def _prediction_summary(values: np.ndarray):
    rounded = np.round(values.astype(np.float64), decimals=12)
    return {
        "prediction_min": float(values.min()),
        "prediction_max": float(values.max()),
        "prediction_mean": float(values.mean()),
        "prediction_std": float(values.std()),
        "unique_prediction_count": int(np.unique(rounded).size),
        "predictions_finite": bool(np.isfinite(values).all()),
        "predictions_non_constant": bool(values.std() > 0 and np.unique(rounded).size > 1),
    }


def _predict(model, loader, device):
    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch, return_logits=True)
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            labels.extend(batch.y.view(-1).cpu().numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def smoke_train(
    train_complexes,
    eval_complexes,
    checkpoint_path: Path,
    seed: int = 44,
    epochs: int = 3,
    batch_size: int = 16,
    config=None,
):
    total_started = time.perf_counter()
    config = dict(config or {})
    if not train_complexes or not eval_complexes:
        raise RuntimeError("MLSHOL smoke requires non-empty train and evaluation complexes")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rss_peak = _rss_bytes()
    train_loader = DataLoader(train_complexes, batch_size=batch_size, shuffle=False, max_dim=3)
    eval_loader = DataLoader(eval_complexes, batch_size=batch_size, shuffle=False, max_dim=3)
    model = SparseSIN(
        num_input_features=1,
        num_layers=int(config.get("num_layers", 4)),
        num_classes=2,
        hidden=int(config.get("hidden_dim", 6)),
        max_dim=3,
        dropout_rate=float(config.get("dropout", 0.5)),
        readout="mean",
        final_readout="mean",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)), weight_decay=float(config.get("weight_decay", 0.0)))
    pos_weight_mode = str(config.get("pos_weight", "natural"))
    n_positive = int(sum(int(item.y.view(-1)[0]) for item in train_complexes))
    n_negative = int(len(train_complexes) - n_positive)
    if pos_weight_mode == "sqrt":
        pos_weight_value = math.sqrt(n_negative / max(n_positive, 1))
    elif pos_weight_mode == "full":
        pos_weight_value = n_negative / max(n_positive, 1)
    elif pos_weight_mode in {"natural", "1", "1.0"}:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(pos_weight_mode)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight_value], dtype=torch.float32, device=device))
    losses = []
    gradient_norm = 0.0
    forward_ok = False
    backward_ok = False
    training_started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch, return_logits=True)
            forward_ok = bool(torch.isfinite(logits).all())
            loss = criterion(logits, batch.y.view(-1).long())
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MLSHOL smoke loss")
            loss.backward()
            sq_norm = sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in model.parameters() if parameter.grad is not None
            )
            gradient_norm = math.sqrt(sq_norm)
            backward_ok = gradient_norm > 0
            optimizer.step()
            epoch_loss += float(loss.item()) * int(batch.y.numel())
            rss_peak = max(rss_peak, _rss_bytes())
        losses.append(epoch_loss / len(train_complexes))
    training_seconds = time.perf_counter() - training_started

    inference_started = time.perf_counter()
    y_true, before = _predict(model, eval_loader, device)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "seed": seed, "simplex_dim": 3}, checkpoint_path)
    reloaded = SparseSIN(
        num_input_features=1, num_layers=int(config.get("num_layers", 4)), num_classes=2, hidden=int(config.get("hidden_dim", 6)),
        max_dim=3, dropout_rate=float(config.get("dropout", 0.5)), readout="mean", final_readout="mean",
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    reloaded.load_state_dict(payload["model_state"])
    _, after = _predict(reloaded, eval_loader, device)
    inference_seconds = time.perf_counter() - inference_started
    checkpoint_reproduces = bool(np.allclose(before, after, rtol=0, atol=1e-7))
    summary = _prediction_summary(before)
    random_baseline = float(y_true.mean()) if len(y_true) else 0.0
    ap = float(average_precision_score(y_true, before)) if y_true.sum() else 0.0
    summary.update({
        "SMOKE_ONLY": True,
        "device": str(device),
        "forward_ok": forward_ok,
        "backward_ok": backward_ok,
        "nan_or_inf": not summary["predictions_finite"],
        "gradient_norm": gradient_norm,
        "loss_epoch_1": losses[0],
        "loss_final_smoke_epoch": losses[-1],
        "epochs": epochs,
        "random_baseline": random_baseline,
        "average_precision": ap,
        "lift": ap / random_baseline if random_baseline > 0 else None,
        "roc_auc": float(roc_auc_score(y_true, before)) if len(np.unique(y_true)) > 1 else None,
        "pos_weight_mode": pos_weight_mode,
        "pos_weight": float(pos_weight_value),
        "config": config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reload_reproduces_predictions": checkpoint_reproduces,
        "peak_CPU_RAM_bytes": rss_peak,
        "peak_VRAM_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "total_wall_clock_seconds": time.perf_counter() - total_started,
        "status": "PASS" if forward_ok and backward_ok and summary["predictions_non_constant"] and checkpoint_reproduces else "SMOKE_FAIL_CONSTANT_PREDICTION",
    })
    return summary
