"""Dimension-generic Dual-Mapping smoke utilities."""

from __future__ import annotations

import math
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from dual_mapping_from_raw import (
    build_bounded_skipgram_pairs,
    build_random_walk_structural_features,
    make_model_classes,
    train_skipgram_embedding,
)
from kclosure.core import structural_features as build_local_structural_features

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _rss_bytes():
    return int(psutil.Process(os.getpid()).memory_info().rss) if psutil else 0


def _adjacency(task):
    adjacency = [set() for _ in task["candidates"]]
    edges = task["derived_graph"]["edge_pairs"]
    for left, right in edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    return adjacency


def build_feature_blocks(task, aligned_rw: np.ndarray, fallback, seed: int = 42, structural_features=None, use_rw: bool = True, rw_steps: int = 4):
    """Use unweighted topology and the legacy random-walk hyperparameters."""
    local, local_names = build_local_structural_features(task)
    graph = task["derived_graph"]
    if use_rw:
        rw_struct = build_random_walk_structural_features(
            graph["edge_index"], graph["num_nodes"], rw_steps=rw_steps
        )
        rw = np.hstack((aligned_rw, rw_struct)).astype(np.float32)
    else:
        rw = np.zeros((len(local), 1), dtype=np.float32)
    heuristic = np.asarray(structural_features, dtype=np.float32) if structural_features is not None else np.empty((len(local), 0), dtype=np.float32)
    return {
        "local": local,
        "heuristic": heuristic,
        "rw": rw,
        "priority": np.empty((len(local), 0), dtype=np.float32),
        "local_feature_names": local_names,
        "random_walk_feature_fallback": fallback,
            "random_walk_parameters": {
            "embedding_dim": 64,
            "walk_length": 20,
            "walks_per_node": 20,
            "window_size": 5,
            "epochs": 5,
            "seed": seed,
            "use_rw": bool(use_rw),
            "normalization": "row_l2",
        },
    }


def build_aligned_embeddings(tasks, seed: int = 42):
    """Train one legacy-parameter walk space over the union of task graphs."""
    global_candidates = sorted(set().union(*(set(task["candidates"]) for task in tasks)))
    global_id = {candidate: index for index, candidate in enumerate(global_candidates)}
    global_edges = set()
    for task in tasks:
        for left, right in task["derived_graph"]["edge_pairs"]:
            u = global_id[task["candidates"][int(left)]]
            v = global_id[task["candidates"][int(right)]]
            if u != v:
                global_edges.add((u, v) if u < v else (v, u))
    adjacency = [set() for _ in global_candidates]
    for left, right in global_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    args = Namespace(
        embedding_seed=seed,
        rw_dim=64,
        rw_epochs=5,
        rw_negative_samples=5,
        rw_walk_length=20,
        rw_window_size=5,
        rw_walks_per_node=20,
        priority_walk_multiplier=1,
        rw_max_pairs=2_000_000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if not global_candidates:
        embedding = np.empty((0, args.rw_dim), dtype=np.float32)
        fallback = "empty_derived_graph"
    elif not global_edges:
        embedding = np.zeros((len(global_candidates), args.rw_dim), dtype=np.float32)
        fallback = "zero_embedding_no_derived_edges"
    else:
        centers, contexts, _ = build_bounded_skipgram_pairs(
            len(global_candidates), adjacency, set(), args
        )
        embedding = train_skipgram_embedding(
            len(global_candidates), centers, contexts, args
        )
        fallback = None
    aligned = []
    for task in tasks:
        ids = np.asarray([global_id[candidate] for candidate in task["candidates"]], dtype=np.int64)
        aligned.append(embedding[ids].astype(np.float32))
    return aligned, fallback


def _tensor_inputs(task, blocks, device):
    return {
        "local_x": torch.from_numpy(blocks["local"]).to(device),
        "heuristic_x": torch.from_numpy(blocks["heuristic"]).to(device),
        "rw_x": torch.from_numpy(blocks["rw"]).to(device),
        "priority_x": torch.from_numpy(blocks["priority"]).to(device),
        "edge_index": torch.from_numpy(task["derived_graph"]["edge_index"]).long().to(device),
        "labels": torch.from_numpy(task["labels"].astype(np.float32)).to(device),
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


def smoke_train(
    train_task,
    eval_task,
    checkpoint_path: Path,
    seed: int = 44,
    epochs: int = 3,
    config=None,
    structural_features=None,
):
    """Full-candidate Dual-Mapping training with frozen, auditable config knobs."""
    total_started = time.perf_counter()
    config = dict(config or {})
    use_rw = bool(config.get("use_rw", True))
    rw_steps = int(config.get("rw_steps", 4))
    if len(train_task["candidates"]) == 0 or len(eval_task["candidates"]) == 0:
        raise RuntimeError("Dual-Mapping smoke requires non-empty candidate graphs")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rss_peak = _rss_bytes()
    feature_started = time.perf_counter()
    if use_rw:
        aligned, fallback = build_aligned_embeddings((train_task, eval_task), seed=seed)
    else:
        aligned = [np.zeros((len(task["candidates"]), 0), dtype=np.float32) for task in (train_task, eval_task)]
        fallback = "disabled_by_config"
    train_aux = structural_features[0] if structural_features is not None else None
    eval_aux = structural_features[1] if structural_features is not None else None
    train_blocks = build_feature_blocks(train_task, aligned[0], fallback, seed=seed, structural_features=train_aux, use_rw=use_rw, rw_steps=rw_steps)
    eval_blocks = build_feature_blocks(eval_task, aligned[1], fallback, seed=seed, structural_features=eval_aux, use_rw=use_rw, rw_steps=rw_steps)

    # Fit feature normalization on training candidates only.
    for block_name in ("local", "rw", "heuristic"):
        mean = train_blocks[block_name].mean(axis=0, keepdims=True)
        std = train_blocks[block_name].std(axis=0, keepdims=True)
        train_blocks[block_name] = ((train_blocks[block_name] - mean) / np.maximum(std, 1e-6)).astype(np.float32)
        eval_blocks[block_name] = ((eval_blocks[block_name] - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    feature_computation_seconds = time.perf_counter() - feature_started

    train = _tensor_inputs(train_task, train_blocks, device)
    evaluation = _tensor_inputs(eval_task, eval_blocks, device)
    _, _, _, DualMappingGNN = make_model_classes()
    model_kwargs = dict(
        local_dim=train["local_x"].shape[1],
        heuristic_dim=train["heuristic_x"].shape[1],
        rw_dim=train["rw_x"].shape[1],
        priority_dim=train["priority_x"].shape[1],
        feature_proj_dim=64,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        encoder="sage",
        dropout=float(config.get("dropout", 0.5)),
        max_residual_alpha=float(config.get("max_residual_alpha", 0.2)),
    )
    model = DualMappingGNN(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)), weight_decay=float(config.get("weight_decay", 0.005)))
    pos_weight_mode = str(config.get("pos_weight", "natural"))
    n_positive = int(np.asarray(train_task["labels"]).sum())
    n_negative = int(len(train_task["labels"]) - n_positive)
    if pos_weight_mode == "sqrt":
        pos_weight_value = math.sqrt(n_negative / max(n_positive, 1))
    elif pos_weight_mode == "full":
        pos_weight_value = n_negative / max(n_positive, 1)
    elif pos_weight_mode in {"natural", "1", "1.0"}:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(pos_weight_mode)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    losses = []
    gradient_norm = 0.0
    forward_ok = False
    backward_ok = False
    training_started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(
            train["local_x"], train["heuristic_x"], train["rw_x"],
            train["priority_x"], train["edge_index"], None,
        )
        forward_ok = bool(torch.isfinite(logits).all())
        loss = criterion(logits, train["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Dual-Mapping smoke loss")
        loss.backward()
        gradient_norm = math.sqrt(sum(
            float(parameter.grad.detach().pow(2).sum().item())
            for parameter in model.parameters() if parameter.grad is not None
        ))
        backward_ok = gradient_norm > 0
        optimizer.step()
        losses.append(float(loss.item()))
        rss_peak = max(rss_peak, _rss_bytes())
    training_seconds = time.perf_counter() - training_started

    inference_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        before = torch.sigmoid(model(
            evaluation["local_x"], evaluation["heuristic_x"], evaluation["rw_x"],
            evaluation["priority_x"], evaluation["edge_index"], None,
        )).cpu().numpy().astype(np.float64)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "seed": seed, "simplex_dim": 3}, checkpoint_path)
    reloaded = DualMappingGNN(**model_kwargs).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    reloaded.eval()
    with torch.no_grad():
        after = torch.sigmoid(reloaded(
            evaluation["local_x"], evaluation["heuristic_x"], evaluation["rw_x"],
            evaluation["priority_x"], evaluation["edge_index"], None,
        )).cpu().numpy().astype(np.float64)
    inference_seconds = time.perf_counter() - inference_started

    checkpoint_reproduces = bool(np.allclose(before, after, rtol=0, atol=1e-7))
    summary = _prediction_summary(before)
    labels = eval_task["labels"]
    random_baseline = float(labels.mean()) if len(labels) else 0.0
    ap = float(average_precision_score(labels, before)) if labels.sum() else 0.0
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
        "roc_auc": float(roc_auc_score(labels, before)) if len(np.unique(labels)) > 1 else None,
        "candidate_subsampling": False,
        "class_balancing": pos_weight_mode != "natural",
        "pos_weight_mode": pos_weight_mode,
        "pos_weight": float(pos_weight_value),
        "config": config,
        "weighted_adjacency": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reload_reproduces_predictions": checkpoint_reproduces,
        "random_walk_feature_fallback": eval_blocks["random_walk_feature_fallback"],
        "random_walk_parameters": eval_blocks["random_walk_parameters"],
        "peak_CPU_RAM_bytes": rss_peak,
        "peak_VRAM_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "feature_computation_seconds": feature_computation_seconds,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "total_wall_clock_seconds": time.perf_counter() - total_started,
        "status": "PASS" if forward_ok and backward_ok and summary["predictions_non_constant"] and checkpoint_reproduces else "SMOKE_FAIL_CONSTANT_PREDICTION",
    })
    return summary
