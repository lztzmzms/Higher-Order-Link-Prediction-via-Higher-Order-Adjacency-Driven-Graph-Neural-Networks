#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="protocol_data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--shared-rw-cache", default=None)
    parser.add_argument("--model-layers", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--config", default=None, help="Frozen experiment configuration JSON.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()
    results_root = Path(args.results_root).resolve()
    config = load_json(Path(args.config).resolve() if args.config else root / "configs" / "model_selection.json")
    spec = config["dual"][args.dataset]
    out = results_root / "dual" / args.dataset / (args.run_name or spec["name"])
    out.mkdir(parents=True, exist_ok=True)

    split_dir = data_root / "shared_splits" / args.dataset
    split = np.load(split_dir / "fit_val_indices.npz")
    fit = np.asarray(split["fit_indices"], dtype=np.int64)
    val = np.asarray(split["val_indices"], dtype=np.int64)
    labels = np.asarray(split["train_labels"], dtype=np.int64)
    seed_values = config["model_seeds"]
    os.environ["K2_SHARED_SPLIT_DIR"] = str(split_dir)
    os.environ["K2_SHARED_DATASET"] = args.dataset
    from pipelines.native import candidate_law_prior

    graph_scope = spec.get("graph_triangle_scope", "open_only")
    shared_train = data_root / "dataset" / "train" / f"{args.dataset}_openTriangles_train.txt"
    shared_test = data_root / "dataset" / "test" / f"{args.dataset}_openTriangles_test.txt"
    train_closed = data_root / "dataset" / "train" / f"{args.dataset}_closeTriangles_train.txt"

    def read_triangles(path):
        return [tuple(sorted(map(int, line.split()))) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    open_triangles = read_triangles(shared_train)
    if graph_scope == "open_only":
        open_graph_indices = np.arange(len(open_triangles), dtype=np.int64)
    else:
        graph_triangles = []
        graph_index = {}
        for triangle in read_triangles(train_closed) + open_triangles:
            if triangle not in graph_index:
                graph_index[triangle] = len(graph_triangles)
                graph_triangles.append(triangle)
        open_graph_indices = np.asarray([graph_index[triangle] for triangle in open_triangles], dtype=np.int64)
    fit_graph = open_graph_indices[fit]
    val_graph = open_graph_indices[val]

    model_split_dir = out / "splits"
    model_split_dir.mkdir(parents=True, exist_ok=True)
    for seed in seed_values:
        np.save(model_split_dir / f"train_indices_seed_{seed}.npy", fit_graph)
        np.save(model_split_dir / f"val_indices_seed_{seed}.npy", val_graph)

    def exact_shared_candidate_law_split(candidate_labels, supervision_mask, val_seed, val_ratio=0.2):
        candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
        supervision_mask = np.asarray(supervision_mask, dtype=bool)
        if not np.array_equal(candidate_labels[open_graph_indices], labels):
            raise RuntimeError("candidate-law train labels/order do not match the shared manifest")
        expected_mask = np.zeros(len(candidate_labels), dtype=bool)
        expected_mask[open_graph_indices] = True
        if not np.array_equal(supervision_mask, expected_mask):
            raise RuntimeError("candidate-law supervision mask is not the complete shared candidate set")
        return fit_graph.copy(), val_graph.copy()

    candidate_law_prior._split_indices = exact_shared_candidate_law_split

    from pipelines.native import dual

    if spec.get("strength_edge_mode") == "unit":


        def unit_weights(triplets, edges, pair_counts, scale=1.0):
            return np.ones(len(edges), dtype=np.float32)
        dual._derived_edge_strength_weights = unit_weights
        original_builder = dual.build_controlled_strength_inputs
        def unit_builder(*builder_args, **builder_kwargs):
            values = list(original_builder(*builder_args, **builder_kwargs))
            values[-1]["edge_weight_mode"] = "unit; same branch as count-weighted control"
            return tuple(values)
        dual.build_controlled_strength_inputs = unit_builder

    def shared_split_validator(train_indices, val_indices, candidate_labels, pos_label, val_ratio):
        train_indices = np.asarray(train_indices, dtype=np.int64)
        val_indices = np.asarray(val_indices, dtype=np.int64)
        y = np.asarray(candidate_labels, dtype=np.int64)
        valid = (
            train_indices.ndim == val_indices.ndim == 1
            and len(train_indices) + len(val_indices) == len(open_graph_indices)
            and np.array_equal(
                np.sort(np.concatenate([train_indices, val_indices])),
                np.sort(open_graph_indices),
            )
            and np.unique(train_indices).size == len(train_indices)
            and np.unique(val_indices).size == len(val_indices)
            and np.intersect1d(train_indices, val_indices).size == 0
            and np.unique(y[train_indices]).size == 2
            and np.unique(y[val_indices]).size == 2
        )
        return valid, "shared protocol indices must partition all candidates and retain both classes"

    dual.validate_stratified_split_indices = shared_split_validator


    train_positive = data_root / "dataset" / "train" / f"{args.dataset}_new_closures_train.txt"

    positive_triangles = set(read_triangles(train_positive))
    rebuilt_labels = np.asarray([int(triangle in positive_triangles) for triangle in open_triangles], dtype=np.int64)
    if not np.array_equal(rebuilt_labels, labels):
        raise RuntimeError(
            "Dual full training labels/order do not match the shared protocol manifest"
        )
    if int(labels[val].sum()) <= 0:
        raise RuntimeError("Shared validation split contains no positive examples")

    rw_cache = Path(args.shared_rw_cache).resolve() if args.shared_rw_cache else out / "rw_cache"
    rw_cache.mkdir(parents=True, exist_ok=True)
    temporal_alpha = float(spec.get("temporal_activity_alpha", 0.0))
    select_temporal_alpha = bool(
        spec.get("validation_select_temporal_alpha", spec["prior"] != "none")
    )
    argv = [
        "node_classfication_embedding_original_rwsearch.py",
        "--input_mode", "protocol",
        "--dataset", args.dataset,
        "--processed_dir", str(data_root / "dataset"),
        "--protocol_root", str(data_root / "dataset"),
        "--protocol-data-root", str(data_root),
        "--generated_gpickle_dir", str(data_root / "gpickle" / args.dataset),
        "--protocol_graph_triangle_scope", graph_scope,
        "--generated_gpickle_feature_mode", "ids",
        "--target", str(shared_test),
        "--priority_nodes", str(train_positive),
        "--results", str(out),
        "--rw_cache_dir", str(rw_cache),
        "--model_type", "dual_adj_gat",
        "--seeds", ",".join(map(str, seed_values)),
        "--model_seeds", ",".join(map(str, seed_values)),
        "--rw_seeds", str(config["rw_seed"]),
        "--rw_impl", "fast_positive",
        "--rw_walk_length", str(spec.get("rw_walk_length", 10)),
        "--rw_walks_per_node", str(spec.get("rw_walks_per_node", 20)),
        "--rw_window_size", str(spec.get("rw_window_size", 2)),
        "--rw_epochs", str(spec.get("rw_epochs", 5)),
        "--rw_embedding_dim", "64",
        "--rw_max_pairs", "20000000",
        "--epochs", str(spec["epochs"]),
        "--model_selection_by", "val_ap",
        "--evaluate-test-during-training", "false",
        "--graph_scope", "local",
        "--local_num_neighbors", "30",
        "--local_num_hops", "1",
        "--local_train_batch_size", "64",
        "--local_eval_batch_size", "64",
        "--supervision_scope", "open_candidates",
        "--pos_label", "1",
        "--use-strength-feature", str(bool(spec["strength_feature"])).lower(),
        "--use-strength-edge-weight", str(bool(spec["strength_edge"])).lower(),
        "--train_strength_context_ratio", "0.6",
        "--test_strength_context_ratio", "0.8",
        "--use-temporal-score-prior", str(spec["prior"] != "none").lower(),
        "--temporal_prior_kind", spec["prior"] if spec["prior"] != "none" else "recent_activity",
        "--validation_select_temporal_alpha", str(select_temporal_alpha).lower(),
        "--temporal_activity_alpha", str(temporal_alpha),
        "--use_edge_maxsimplex_features", "false",
        "--rw_param_sweep", "false",
        "--deterministic", "true",
        "--amp", "false",
        "--device", args.device,
        "--reuse_splits", "true",
        "--val_seed", "2026",
    ]
    if spec["name"] == "D_small_regularized":
        argv += [
            "--model_num_layers", "2",
            "--model_hidden_channels", "64",
            "--model_heads", "2",
            "--model_dropout", "0.3",
            "--dual_self_hidden_dim", "64",
            "--dual_gate_hidden_dim", "32",
            "--dual_dropout", "0.3",
            "--lr", "0.001",
            "--weight_decay", "0.005",
        ]
    else:
        argv += [
            "--model_num_layers", "4",
            "--model_hidden_channels", "128",
            "--model_heads", "2",
            "--model_dropout", "0.2",
            "--dual_self_hidden_dim", "128",
            "--dual_gate_hidden_dim", "64",
            "--dual_dropout", "0.2",
            "--lr", "0.002",
            "--weight_decay", "0.001",
        ]
    if args.model_layers is not None:
        argv[argv.index("--model_num_layers") + 1] = str(args.model_layers)
    if args.epochs_override is not None:
        argv[argv.index("--epochs") + 1] = str(args.epochs_override)
    sys.argv = argv
    dual.main_rwsearch()
