#!/usr/bin/env python3
"""Run one dataset's selected MLSHoL configuration on shared K=2 indices."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="protocol_data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=None, help="Frozen experiment configuration JSON.")
    parser.add_argument("--run-name", default=None,
                        help="Short output directory name; does not change model settings.")
    parser.add_argument("--max-nodes-per-step", type=int, default=None,
                        help="Override the local complex size for memory-limited runs.")
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--node-embedding-train-npz", default=None)
    parser.add_argument("--node-embedding-test-npz", default=None)
    parser.add_argument("--history-feature-npz", default=None)
    parser.add_argument("--history-feature-mode", default="none")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rw-cache-dir", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()
    results_root = Path(args.results_root).resolve()
    config = json.loads((Path(args.config).resolve() if args.config else root / "configs" / "model_selection.json").read_text(encoding="utf-8"))
    spec = config["mlshol"][args.dataset]
    dual_spec = config["dual"][args.dataset]
    out = results_root / "mlshol" / args.dataset / (args.run_name or spec["name"])
    out.mkdir(parents=True, exist_ok=True)
    split_dir = data_root / "shared_splits" / args.dataset
    split = np.load(split_dir / "fit_val_indices.npz")
    fit = np.asarray(split["fit_indices"], dtype=np.int64)
    val = np.asarray(split["val_indices"], dtype=np.int64)
    labels = np.asarray(split["train_labels"], dtype=np.int64)
    os.environ["K2_SHARED_SPLIT_DIR"] = str(split_dir)
    os.environ["K2_SHARED_DATASET"] = args.dataset

    import exp.run_all_sparse_sin as mlshol

    def exact_shared_split(complexes, val_fraction, seed):
        if len(complexes) != len(labels):
            raise RuntimeError(
                "MLSHoL complex count/order differs from shared candidates: "
                f"{len(complexes)} != {len(labels)}"
            )
        observed = np.asarray([int(item.y.view(-1)[0].item()) for item in complexes], dtype=np.int64)
        if not np.array_equal(observed, labels):
            raise RuntimeError("MLSHoL labels/order do not match the shared candidate manifest")
        return [complexes[int(index)] for index in fit], [complexes[int(index)] for index in val]

    mlshol.stratified_train_validation_split = exact_shared_split
    argv = [
        "exp.run_all_sparse_sin",
        "--datasets", args.dataset,
        "--data-root", str(data_root),
        "--event-order", "raw",
        "--output-dir", str(out),
        "--cache-dir", str(Path(args.cache_dir).resolve() if args.cache_dir else out / "cache"),
        "--seeds", *map(str, config["model_seeds"]),
        "--sampling-seed", "2026",
        "--val-seed", "2026",
        "--val-fraction", "0.2",
        "--train-raw-cap", "30001",
        "--max-train-complexes", "30001",
        "--validation-raw-cap", "5001",
        "--max-validation-complexes", "5001",
        "--test-raw-cap", "50001",
        "--max-test-complexes", "50001",
        "--sampling-mode", "uniform",
        "--validation-sampling-mode", "uniform",
        "--test-sampling-mode", "uniform",
        "--train-history-context-ratio", "0.6",
        "--test-history-context-ratio", "0.8",
        "--num-hops", "1",
        "--sample-ratio", "1.0",
        "--max-nodes-per-step", "35",
        "--rw-fusion", "concat",
        "--rw-source", spec.get("rw_source", "aligned_union"),
        "--rw-cache-dir", str(Path(args.rw_cache_dir).resolve() if args.rw_cache_dir else out / "cache" / "rw"),
        "--rw-seed", str(config["rw_seed"]),
        "--rw-walks-per-node", "20",
        "--rw-epochs", "5",
        "--rw-dim", "64",
        "--epochs", str(spec["epochs"]),
        "--eval-every", "5",
        "--batch-size", "64",
        "--loss-mode", spec["loss_mode"],
        "--focal-gamma", str(spec["focal_gamma"]),
        "--disconnected-local-policy", "unreachable_zero",
        "--strict",
        "--device", args.device,
    ]
    if spec["strength_feature"]:
        argv.append("--use-strength-feature")
    if spec.get("observed_2_skeleton", False):
        argv.append("--observed-2-skeleton")
    if spec.get("strength_edge", False):
        argv += ["--use-strength-edge-weight", "--adjacency-mode", spec.get("adjacency_mode", "weighted-log")]
    if spec.get("activity_prior", False):
        argv += [
            "--use-temporal-score-prior",
            "--temporal-activity-alpha", str(spec.get("temporal_activity_alpha", 0.0)),
        ]
        if spec.get("validation_select_activity_alpha", False):
            argv.append("--validation-select-activity-alpha")
    if "max_nodes_per_step" in spec:
        index = argv.index("--max-nodes-per-step")
        argv[index + 1] = str(spec["max_nodes_per_step"])
    if args.max_nodes_per_step is not None:
        index = argv.index("--max-nodes-per-step")
        argv[index + 1] = str(args.max_nodes_per_step)
    if args.history_feature_mode != "none":
        if not args.history_feature_npz:
            raise ValueError("--history-feature-npz is required for history mode")
        argv += ["--history-feature-mode", args.history_feature_mode,
                 "--history-feature-npz", str(Path(args.history_feature_npz).resolve())]
    if "rw_scale" in spec:
        argv += ["--rw-scale", str(spec["rw_scale"])]
    if "rw_input_normalization" in spec:
        argv += ["--rw-input-normalization", spec["rw_input_normalization"]]
    if "rw_full_normalization" in spec:
        argv += ["--rw-full-normalization", spec["rw_full_normalization"]]
    if spec.get("rw_source") == "dual_mapping_full":
        dual_rw_dir = Path(dual_spec["revision_shared_rw_cache"]).resolve() if "revision_shared_rw_cache" in dual_spec else results_root / "dual" / args.dataset / dual_spec["name"] / "rw_cache"
        reference_caches = sorted(dual_rw_dir.glob("rw_embedding_*.npz"))
        if len(reference_caches) != 1:
            raise RuntimeError(
                "Expected exactly one freshly generated Dual full-graph RW cache at "
                f"{dual_rw_dir}; found {len(reference_caches)}"
            )
        argv += [
            "--rw-reference-cache", str(reference_caches[0]),
        ]
    argv += ["--max-dim", str(args.max_dim)]
    argv += ["--save-checkpoint", str(out / "best_checkpoint.pt")]
    if args.epochs_override is not None:
        argv[argv.index("--epochs") + 1] = str(args.epochs_override)
    if args.node_embedding_train_npz:
        if not args.node_embedding_test_npz:
            raise ValueError("Both train and test node embeddings are required")
        argv += ["--node-embedding-train-npz", args.node_embedding_train_npz,
                 "--node-embedding-test-npz", args.node_embedding_test_npz,
                 "--num-input-features", "64"]
    sys.argv = argv
    mlshol.main()


if __name__ == "__main__":
    main()
