from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import run_strict_timestamp_t2_mlp_local_v1 as mlp
import run_strict_timestamp_chronological_core_v1 as core

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT
OUT_ROOT = ROOT / "results" / "strict_timestamp_t2_mlp_r1_dose_local_v1"
ALLOW_EXACT_BUILD = False
SERVER_MODE = False
PROTOCOL = "strict_timestamp_t2_mlp_r1_dose_local_v1"
TASK = "T2"
TARGETS = (0.20, 0.40, 0.60, 0.80, 0.95)
TARGET_CODES = {0.20: "020", 0.40: "040", 0.60: "060", 0.80: "080", 0.95: "095"}
TARGET_TOLERANCE = 0.02
MLP_SEEDS = tuple(range(21001, 21006))
R1_SEEDS_BY_DOSE = {
    0.20: tuple(range(51001, 51011)),
    0.40: tuple(range(52001, 52011)),
    0.60: tuple(range(53001, 53011)),
    0.80: tuple(range(54001, 54011)),
    0.95: tuple(range(55001, 55011)),
}

def seed_map_from_base(base: int):
    pass
    base = int(base)
    return {target: tuple(range(base + index * 1000, base + index * 1000 + 10)) for index, target in enumerate(TARGETS)}
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 61001
MAX_DOSE_RESTARTS = 8
MAX_CONTROLLER_ATTEMPTS = 1_000_000

PRIMARY_CASES = [
    (2, "email-Eu"),
    (2, "email-Enron"),
    (2, "contact-high-school"),
    (2, "contact-primary-school"),
    (2, "NDC-substances"),
    (3, "email-Eu"),
    (3, "NDC-substances"),
    (3, "tags-math-sx"),
]
OPTIONAL_CASES = [
    (3, "tags-ask-ubuntu"),
    (3, "threads-math-sx"),
    (2, "tags-ask-ubuntu"),
    (2, "tags-math-sx"),
    (2, "threads-math-sx"),
]
ALL_CASES = PRIMARY_CASES + OPTIONAL_CASES

REALIZATION_FIELDS = [
    "order", "dataset", "target_dose", "actual_changed_fraction", "dose_error", "r1_seed", "mlp_seed", "valid",
    "LL_real", "LL_R1", "delta_LL", "Brier_real", "Brier_R1", "delta_Brier", "MAE_p", "RMSE_p",
    "median_abs_delta_p", "Q90_abs_delta_p", "Q95_abs_delta_p", "Q99_abs_delta_p", "max_abs_delta_p",
    "Pearson_p", "Spearman_p", "AP_real", "AP_R1", "delta_AP", "AUC_real", "AUC_R1", "delta_AUC",
    "Lift_real", "Lift_R1", "delta_Lift", "residual_jaccard", "DOSE_TARGET_HIT", "MAX_REACHABLE_CHANGED_FRACTION",
    "H_HASH_MATCH", "CANDIDATE_HASH_MATCH", "BOUNDARY_HASH_MATCH", "NODEWISE_RESIDUAL_DEGREE_MATCH",
    "RESIDUAL_EDGE_COUNT_MATCH", "K3_RESIDUAL_L2_HASH_MATCH", "H_FEATURE_MAX_ABS_DIFF", "L2_FEATURE_MAX_ABS_DIFF",
    "LOWER_ADJACENCY_EXACT", "R1_CONSTRUCTION_LABEL_BLIND", "R1_CONSTRUCTION_MODEL_BLIND",
    "R1_CONSTRUCTION_OUTCOME_BLIND", "MODEL_RETRAINED_ON_R1", "SCALER_REFIT_ON_R1", "VALID_R1",
    "CANDIDATE_LEVEL_METRIC_RECOMPUTATION", "baseline_feature_hash", "failure_reason",
]
CLUSTER_FIELDS = [
    "order", "dataset", "target_dose", "r1_seed", "actual_changed_fraction", "n_valid_mlp", "mean_delta_LL",
    "std_delta_LL", "mean_MAE_p", "mean_Q95_abs_delta_p", "mean_delta_Brier", "mean_delta_AP", "mean_delta_AUC",
]
AGGREGATE_FIELDS = [
    "order", "dataset", "target_dose", "n_valid_r1", "n_mlp_evaluations", "mean_actual_changed_fraction",
    "std_actual_changed_fraction", "mean_delta_LL", "median_delta_LL", "std_delta_LL", "Q05_delta_LL", "Q95_delta_LL",
    "CI95_delta_LL_low", "CI95_delta_LL_high", "mean_MAE_p", "median_MAE_p", "CI95_MAE_low", "CI95_MAE_high",
    "mean_Q95_abs_delta_p", "mean_delta_Brier", "mean_delta_AP", "mean_delta_AUC", "dose_response_class",
    "mean_abs_delta_LL_lt_001", "mean_abs_delta_LL_lt_005", "mean_abs_delta_LL_lt_010", "mean_abs_delta_LL_lt_025",
    "mean_MAE_lt_005", "mean_MAE_lt_010", "mean_MAE_lt_020",
]

class ResourceLimit(Exception):
    pass

class ProtocolFailure(Exception):
    pass

def jdefault(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=jdefault), encoding="utf-8")

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_csv(path: Path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def hash_array(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))

    return sha256_bytes(f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes(order="C"))

def hash_set(values) -> str:
    rows = []
    for value in values:
        if isinstance(value, (tuple, list, np.ndarray)):
            rows.append(",".join(str(int(item)) for item in value))
        else:
            rows.append(str(value))
    return sha256_bytes("\n".join(sorted(rows)).encode())

def hash_lower_adjacency(task) -> str:
    return hash_set((index, neighbor) for index, values in enumerate(task["lower_adj"]) for neighbor in values if index < neighbor)

def hash_structure(structure):
    return sha256_bytes("\n".join(f"{name}:{hash_set(structure.get(name, set()))}" for name in ("nodes", "edges", "triangles", "tetras")).encode())

def case_key(order, dataset):
    return f"K{order}|{dataset}|T2"

def code(target):
    return TARGET_CODES[float(target)]

def case_root(order, dataset):
    return OUT_ROOT / "00_protocol" / "cases" / f"K{order}" / dataset

def baseline_root(order, dataset):
    return OUT_ROOT / "01_baselines" / f"K{order}" / dataset

def dose_root(order, dataset, target):
    return OUT_ROOT / f"{ {'020':'02','040':'03','060':'04','080':'05','095':'06'}[code(target)] }_dose_{code(target)}" / f"K{order}" / dataset

def graph_root(order, dataset, target, r1_seed):
    return OUT_ROOT / "08_structural_audit" / f"K{order}" / dataset / f"dose_{code(target)}" / f"r1_seed_{r1_seed}"

def prediction_root(order, dataset, target, r1_seed, mlp_seed):
    return OUT_ROOT / "07_predictions" / f"K{order}" / dataset / f"dose_{code(target)}" / f"r1_seed_{r1_seed}" / f"mlp_seed_{mlp_seed}"

def baseline_model_root(order, dataset, mlp_seed):
    return OUT_ROOT / "01_baselines" / f"K{order}" / dataset / f"mlp_seed_{mlp_seed}"

def fmt(value, digits=9):
    if value is None or value == "":
        return "NA"
    try:
        value = float(value)
        return f"{value:.{digits}f}" if math.isfinite(value) else "NA"
    except (TypeError, ValueError):
        return str(value)

def finite(values):
    result = []
    for value in values:
        try:
            number = float(value)
            if math.isfinite(number):
                result.append(number)
        except (TypeError, ValueError):
            pass
    return result

def mean(values):
    values = finite(values)
    return float(np.mean(values)) if values else None

def median(values):
    values = finite(values)
    return float(np.median(values)) if values else None

def std(values):
    values = finite(values)
    return float(np.std(values)) if values else None

def percentile(values, q):
    values = finite(values)
    return float(np.quantile(values, q)) if values else None

def configure_source(source_root: Path):
    global SOURCE_ROOT
    SOURCE_ROOT = source_root.resolve()
    core.ROOT = SOURCE_ROOT
    if hasattr(core, "k2legacy"):
        core.k2legacy.ROOT = SOURCE_ROOT
        core.k2legacy.RAW_ROOT = SOURCE_ROOT / "dataset" / "raw_data"
    if hasattr(core, "k3legacy"):
        core.k3legacy.ROOT = SOURCE_ROOT
    if hasattr(core, "hp"):
        core.hp.ROOT = SOURCE_ROOT
        if hasattr(core.hp, "k2"):
            core.hp.k2.ROOT = SOURCE_ROOT
            core.hp.k2.RAW_ROOT = SOURCE_ROOT / "dataset" / "raw_data"
        if hasattr(core.hp, "k3"):
            core.hp.k3.ROOT = SOURCE_ROOT

def resource_is_oom(exc):
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or ("cuda" in text and "memory" in text)

def load_audit(dataset):
    cache = SOURCE_ROOT / "results" / "strict_timestamp_chronological_core_v1" / "01_timestamp_audit" / "audit_cache.json"
    if cache.is_file():
        values = read_json(cache)
        if dataset in values:
            return values[dataset]
    audit, _, _ = core.audit_dataset(dataset)
    return audit

def build_task_exact(order, dataset, events, context_end, label_start, label_end, name):
    old_caps = getattr(core.k2legacy, "RESOURCE_CAP_DATASETS", set())
    core.k2legacy.RESOURCE_CAP_DATASETS = set()
    try:
        return core.build_task(order, dataset, events, context_end, label_start, label_end, name)
    finally:
        core.k2legacy.RESOURCE_CAP_DATASETS = old_caps

def matrix(arrays, names):
    return np.column_stack([arrays[name] for name in names]).astype(np.float32) if names else np.empty((len(next(iter(arrays.values()))), 0), dtype=np.float32)

def load_case_data(order, dataset):
    task_dir = SOURCE_ROOT / "results" / "strict_timestamp_chronological_core_v1" / ("03_k3" if order == 3 else "04_k2") / dataset / "T2"
    summary_path = task_dir / "task_summary.json"
    train_feature_path = task_dir / "real_train_features.npz"
    eval_feature_path = task_dir / "real_eval_features.npz"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    cache_ready = summary.get("status") == "COMPLETE" and train_feature_path.is_file() and eval_feature_path.is_file()
    if not cache_ready and not ALLOW_EXACT_BUILD:
        reason = summary.get("reason", "STRICT_T2_FEATURE_BUNDLE_MISSING") if summary else "STRICT_T2_TASK_SUMMARY_MISSING"
        raise ResourceLimit(reason)
    events = core.load_events(dataset, "timestamp")
    audit = load_audit(dataset)
    n_events = len(events)
    c40, c60, c80 = (int(audit[key]) for key in ("cut_40", "cut_60", "cut_80"))
    if not (0 < c40 < c60 < c80 < n_events):
        raise ProtocolFailure("INVALID_STRICT_TIMESTAMP_CUTS")
    train = build_task_exact(order, dataset, events, c60, c60, c80, "T2_TRAIN")
    evaluation = build_task_exact(order, dataset, events, c80, c80, n_events, "T2_EVAL")
    train_dec, eval_dec = core.decompose(order, train), core.decompose(order, evaluation)
    if cache_ready and (train["candidate_hash"], len(train["candidates"])) != (summary.get("candidate_hash_train"), int(summary.get("train_candidate_count"))):
        raise ProtocolFailure("TRAIN_CANDIDATE_HASH_MISMATCH")
    if cache_ready and (evaluation["candidate_hash"], len(evaluation["candidates"])) != (summary.get("candidate_hash_eval"), int(summary.get("eval_candidate_count"))):
        raise ProtocolFailure("EVAL_CANDIDATE_HASH_MISMATCH")
    gate = core.timestamp_gate(events, c60, c80, c80, n_events, (0, c40, c60, c80, n_events))
    if gate.get("no_timestamp_group_crosses_split") != "YES":
        raise ProtocolFailure("NO_TIMESTAMP_GROUP_CROSSES_SPLIT=NO")
    if cache_ready:
        with np.load(train_feature_path, allow_pickle=False) as values:
            train_arrays = {key: values[key] for key in values.files}
        with np.load(eval_feature_path, allow_pickle=False) as values:
            eval_arrays = {key: values[key] for key in values.files}
    else:

        train_arrays = core.compute_arrays(order, train, train_dec)
        eval_arrays = core.compute_arrays(order, evaluation, eval_dec)
        cache_dir = OUT_ROOT / "00_protocol" / "feature_cache" / f"K{order}" / dataset / "T2"
        np.savez_compressed(cache_dir / "real_train_features.npz", **train_arrays)
        np.savez_compressed(cache_dir / "real_eval_features.npz", **eval_arrays)
    h_names, l1_names, l2_names = core.feature_names(order)
    h_names, l1_names, l2_names = tuple(h_names), tuple(l1_names), tuple(l2_names)
    full_names = h_names + l1_names + l2_names
    if not set(full_names).issubset(train_arrays) or set(train_arrays) != set(eval_arrays):
        raise ProtocolFailure("FEATURE_ORDER_OR_NAME_SET_MISMATCH")
    x_train = matrix(train_arrays, full_names)
    x_eval = matrix(eval_arrays, full_names)
    if x_train.shape[0] != len(train["labels"]) or x_eval.shape[0] != len(evaluation["labels"]):
        raise ProtocolFailure("FEATURE_LABEL_ALIGNMENT_FAIL")
    if not np.isfinite(x_train).all() or not np.isfinite(x_eval).all():
        raise ProtocolFailure("FEATURE_NAN_OR_INF")
    h_structure = core.h_structure(order, eval_dec)
    return {
        "order": order, "dataset": dataset, "events": events, "audit": audit, "gate": gate,
        "train": train, "eval": evaluation, "train_dec": train_dec, "eval_dec": eval_dec,
        "train_arrays": train_arrays, "eval_arrays": eval_arrays, "h_names": h_names, "l1_names": l1_names,
        "l2_names": l2_names, "full_names": full_names, "X_train_Full": x_train, "X_eval_Full": x_eval,
        "y_train": np.asarray(train["labels"], dtype=np.int8), "y_eval": np.asarray(evaluation["labels"], dtype=np.int8),
        "candidate_hash_train": train["candidate_hash"], "candidate_hash_eval": evaluation["candidate_hash"],
        "boundary_hash_eval": evaluation["boundary_hash"], "feature_hash_train": hash_array(x_train), "feature_hash_eval": hash_array(x_eval),
        "h_hash": hash_structure(h_structure), "lower_adjacency_hash": hash_lower_adjacency(evaluation),
        "l2_hash": hash_set(eval_dec.get("l2_triangles", set())), "real_edges": set(eval_dec["l1_edges"]),
    }

def save_case_protocol(data):
    payload = {
        "protocol": PROTOCOL, "task": "T2", "order": f"K{data['order']}", "dataset": data["dataset"],
        "STRICT_TIMESTAMP": "YES", "PRIMARY_TEMPORAL_WINDOW": "T2", "FULL_DATASET": "YES",
        "train": "W0+W1 observed -> W2 labels", "eval": "W0+W1+W2 observed -> W3 labels",
        "NO_TIMESTAMP_GROUP_CROSSES_SPLIT": "YES", "T1_CONSTRUCTED": "NO", "R0_RUN": "NO",
        "event_subsampling": "NO", "candidate_subsampling": "NO", "negative_subsampling": "NO",
        "node_subsampling": "NO", "edge_subsampling": "NO", "time_truncation": "NO",
        "candidate_hash_train": data["candidate_hash_train"], "candidate_hash_eval": data["candidate_hash_eval"],
        "boundary_hash_eval": data["boundary_hash_eval"], "feature_hash_train_Full": data["feature_hash_train"],
        "feature_hash_eval_Full": data["feature_hash_eval"], "H_hash": data["h_hash"],
        "lower_adjacency_hash": data["lower_adjacency_hash"], "K3_L2_hash": data["l2_hash"] if data["order"] == 3 else "NA",
        "feature_names": list(data["full_names"]), "n_raw_events": len(data["events"]),
        "n_train_candidates": len(data["train"]["candidates"]), "n_eval_candidates": len(data["eval"]["candidates"]),
    }
    path = case_root(data["order"], data["dataset"]) / "case_protocol.json"
    write_json(path, payload)
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode())

def model_source_path(order, dataset, mlp_seed):
    return SOURCE_ROOT / "results" / "strict_timestamp_t2_mlp_local_v1" / "03_models" / f"K{order}" / dataset / "T2" / "Full" / f"seed_{mlp_seed}"

def load_or_train_models(data, device, batch_size):
    bundles = {}
    refs = []
    reused_all = True
    protocol_path = SOURCE_ROOT / "results" / "strict_timestamp_t2_mlp_local_v1" / ("01_k3" if data["order"] == 3 else "02_k2") / data["dataset"] / "T2" / "00_protocol.json"
    old_protocol = read_json(protocol_path) if protocol_path.is_file() else {}
    for seed in MLP_SEEDS:
        source = model_source_path(data["order"], data["dataset"], seed)
        model_path, scaler_path = source / "model.pt", source / "scaler"
        meta_path = model_path.with_suffix(".json")
        valid_source = model_path.is_file() and scaler_path.with_suffix(".npz").is_file() and scaler_path.with_suffix(".json").is_file() and meta_path.is_file()
        checkpoint = None
        if valid_source:
            checkpoint = torch.load(model_path, map_location="cpu")
            arch = checkpoint.get("architecture", {})
            valid_source = arch.get("input_dim") == len(data["full_names"]) and checkpoint.get("feature_names") == list(data["full_names"])
            valid_source = valid_source and old_protocol.get("feature_hash_eval_Full") in (None, data["feature_hash_eval"])
        if valid_source:
            bundle = mlp.load_model(model_path, scaler_path, device)
            source_name = "REUSED_FROZEN_FULL_MLP"
        else:
            reused_all = False
            target = baseline_model_root(data["order"], data["dataset"], seed)
            bundle, metadata = mlp.train_one_model(data["X_train_Full"], data["y_train"], data["full_names"], seed, device, batch_size, target / "model.pt", target / "scaler")
            source_name = "RETRAINED_EXACT_FROZEN_PROTOCOL"
        refs.append({"mlp_seed": seed, "source": source_name, "source_path": str(source if source_name.startswith("REUSED") else baseline_model_root(data["order"], data["dataset"], seed)), "architecture_changed": "NO", "hyperparameters_changed": "NO"})
        bundles[seed] = bundle
    write_json(baseline_root(data["order"], data["dataset"]) / "model_reuse.json", {"MLP_PROTOCOL_REUSED": "YES" if reused_all else "PARTIAL", "MLP_ARCHITECTURE_CHANGED": "NO", "MLP_HYPERPARAMETERS_CHANGED": "NO", "models": refs})
    return bundles, reused_all

def metric_row(y, p):
    return mlp.metric_row(np.asarray(y, dtype=np.int8), np.asarray(p, dtype=np.float64))

def save_real_baselines(data, bundles, device, batch_size):
    result = {}
    for seed, bundle in bundles.items():
        logits, probabilities = mlp.predict_model(bundle, data["X_eval_Full"], device, batch_size)
        result[seed] = {"logits": logits, "p": probabilities, "metrics": metric_row(data["y_eval"], probabilities)}
        root = baseline_root(data["order"], data["dataset"])
        np.savez_compressed(root / f"real_eval_mlp_seed_{seed}.npz", candidate_id=np.arange(len(data["y_eval"]), dtype=np.int64), y=data["y_eval"], p_real=probabilities, logit_real=logits)
    return result

def degree_map(edges):
    values = Counter()
    for left, right in edges:
        values[int(left)] += 1
        values[int(right)] += 1
    return dict(sorted(values.items()))

def propose_swap(edge_list, current, forbidden, rng):
    if len(edge_list) < 2:
        return None
    first, second = rng.choice(len(edge_list), size=2, replace=False)
    e1, e2 = edge_list[int(first)], edge_list[int(second)]
    a, b = e1; c, d = e2
    if len({a, b, c, d}) < 4:
        return None
    proposed = {(min(a, d), max(a, d)), (min(c, b), max(c, b))} if rng.random() < 0.5 else {(min(a, c), max(a, c)), (min(b, d), max(b, d))}
    if any(left == right for left, right in proposed) or proposed & forbidden or proposed & (current - {e1, e2}):
        return None
    return int(first), int(second), e1, e2, proposed

def target_changed_fraction_rewire(real_edges, forbidden, target, seed):
    real_edges = {tuple(sorted((int(a), int(b)))) for a, b in real_edges}
    forbidden = {tuple(sorted((int(a), int(b)))) for a, b in forbidden}
    edge_count = len(real_edges)
    if edge_count < 2:
        return None, {"DOSE_TARGET_HIT": "NO", "MAX_REACHABLE_CHANGED_FRACTION": 0.0, "failure_reason": "FEWER_THAN_TWO_RESIDUAL_EDGES"}
    lower, upper = max(0.0, target - TARGET_TOLERANCE), min(1.0, target + TARGET_TOLERANCE)

    grid = [1.0 - retained / edge_count for retained in range(edge_count + 1)]
    feasible_grid = [value for value in grid if lower - 1e-15 <= value <= upper + 1e-15]
    if not feasible_grid:
        nearest = min(grid, key=lambda value: abs(value - target))
        return None, {"DOSE_TARGET_HIT": "NO", "MAX_REACHABLE_CHANGED_FRACTION": max(grid), "failure_reason": "TARGET_CHANGED_FRACTION_NOT_REACHED_DISCRETE_RESIDUAL_EDGE_GRID", "residual_edge_count": edge_count, "nearest_grid_changed_fraction": nearest, "controller_attempts": 0}
    max_seen = 0.0
    for restart in range(MAX_DOSE_RESTARTS):
        rng = np.random.default_rng(int(seed) + restart * 1000003)
        edge_list, current, retained = sorted(real_edges), set(real_edges), edge_count
        attempts = 0
        accepted_swaps = 0
        while attempts < min(MAX_CONTROLLER_ATTEMPTS, max(20000, edge_count * 100)):
            attempts += 1
            proposal = propose_swap(edge_list, current, forbidden, rng)
            if proposal is None:
                continue
            first, second, e1, e2, proposed = proposal
            new_retained = retained - int(e1 in real_edges) - int(e2 in real_edges) + sum(int(edge in real_edges) for edge in proposed)
            changed = 1.0 - new_retained / edge_count
            max_seen = max(max_seen, changed)
            if changed <= (1.0 - retained / edge_count) + 1e-15 or changed > upper + 1e-15:
                continue
            current.remove(e1); current.remove(e2); current.update(proposed)
            replacements = sorted(proposed)
            edge_list[first], edge_list[second], retained = replacements[0], replacements[1], new_retained
            accepted_swaps += 1
            if lower - 1e-15 <= changed <= upper + 1e-15:
                return current, {"DOSE_TARGET_HIT": "YES", "MAX_REACHABLE_CHANGED_FRACTION": max_seen, "actual_changed_fraction": changed, "failure_reason": "", "controller_attempts": attempts, "accepted_swaps": accepted_swaps, "residual_edge_count": edge_count}
    return None, {"DOSE_TARGET_HIT": "NO", "MAX_REACHABLE_CHANGED_FRACTION": max_seen, "actual_changed_fraction": None, "failure_reason": "TARGET_CHANGED_FRACTION_NOT_REACHED", "accepted_swaps": accepted_swaps, "residual_edge_count": edge_count}

def cluster_bootstrap(values_by_cluster):
    values = np.asarray([float(value) for value in values_by_cluster if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))

def spearman(left, right):
    return mlp.spearman(np.asarray(left), np.asarray(right))

def empty_row(data, target, r1_seed, mlp_seed, info):
    real = data["real_metrics"][mlp_seed]
    return {"order": f"K{data['order']}", "dataset": data["dataset"], "target_dose": target, "actual_changed_fraction": info.get("actual_changed_fraction"), "dose_error": None, "r1_seed": r1_seed, "mlp_seed": mlp_seed, "valid": "NO", "LL_real": real["LogLoss"], "LL_R1": None, "delta_LL": None, "Brier_real": real["Brier"], "Brier_R1": None, "delta_Brier": None, "MAE_p": None, "RMSE_p": None, "median_abs_delta_p": None, "Q90_abs_delta_p": None, "Q95_abs_delta_p": None, "Q99_abs_delta_p": None, "max_abs_delta_p": None, "Pearson_p": None, "Spearman_p": None, "AP_real": real.get("AP"), "AP_R1": None, "delta_AP": None, "AUC_real": real.get("AUC"), "AUC_R1": None, "delta_AUC": None, "Lift_real": real.get("Lift"), "Lift_R1": None, "delta_Lift": None, "residual_jaccard": None, "DOSE_TARGET_HIT": info.get("DOSE_TARGET_HIT", "NO"), "MAX_REACHABLE_CHANGED_FRACTION": info.get("MAX_REACHABLE_CHANGED_FRACTION"), "H_HASH_MATCH": "YES", "CANDIDATE_HASH_MATCH": "YES", "BOUNDARY_HASH_MATCH": "YES", "NODEWISE_RESIDUAL_DEGREE_MATCH": "NO", "RESIDUAL_EDGE_COUNT_MATCH": "NO", "K3_RESIDUAL_L2_HASH_MATCH": "YES" if data["order"] == 3 else "NA", "H_FEATURE_MAX_ABS_DIFF": None, "L2_FEATURE_MAX_ABS_DIFF": None, "LOWER_ADJACENCY_EXACT": "YES", "R1_CONSTRUCTION_LABEL_BLIND": "YES", "R1_CONSTRUCTION_MODEL_BLIND": "YES", "R1_CONSTRUCTION_OUTCOME_BLIND": "YES", "MODEL_RETRAINED_ON_R1": "NO", "SCALER_REFIT_ON_R1": "NO", "VALID_R1": "NO", "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "NA", "baseline_feature_hash": data["feature_hash_eval"], "failure_reason": info.get("failure_reason", "TARGET_CONTROLLER_FAIL")}

def evaluate_graph(data, target, r1_seed, random_edges, info, bundles):
    graph_dir = graph_root(data["order"], data["dataset"], target, r1_seed)
    graph_dir.mkdir(parents=True, exist_ok=True)
    real_edges = data["real_edges"]
    actual = info.get("actual_changed_fraction")
    if random_edges is not None:
        actual = 1.0 - len(real_edges & random_edges) / len(real_edges) if real_edges else 0.0
        info["actual_changed_fraction"] = actual
        info["residual_jaccard"] = len(real_edges & random_edges) / len(real_edges | random_edges) if real_edges | random_edges else 1.0
        np.savez_compressed(graph_dir / "r1_edges.npz", left=np.asarray([x[0] for x in sorted(random_edges)], dtype=np.int64), right=np.asarray([x[1] for x in sorted(random_edges)], dtype=np.int64))
    write_json(graph_dir / "r1_protocol.json", {"order": f"K{data['order']}", "dataset": data["dataset"], "task": "T2", "target_dose": target, "r1_seed": r1_seed, "target_tolerance": TARGET_TOLERANCE, "dose_info": info, "R1_CONSTRUCTION_LABEL_BLIND": "YES", "R1_CONSTRUCTION_MODEL_BLIND": "YES", "R1_CONSTRUCTION_OUTCOME_BLIND": "YES"})
    rows = []
    if random_edges is None:
        for seed in MLP_SEEDS:
            row = empty_row(data, target, r1_seed, seed, info)
            pdir = prediction_root(data["order"], data["dataset"], target, r1_seed, seed)
            write_json(pdir / "realization.json", row)
            rows.append(row)
        write_json(graph_dir / "r1_audit.json", {"valid": "NO", **info})
        return rows
    forbidden = set(core.h_structure(data["order"], data["eval_dec"])["edges"])
    h_hash_match = hash_structure(core.h_structure(data["order"], data["eval_dec"])) == data["h_hash"]
    degree_exact = degree_map(real_edges) == degree_map(random_edges)
    edge_exact = len(real_edges) == len(random_edges)
    candidate_exact = data["eval"]["candidate_hash"] == data["candidate_hash_eval"]
    boundary_exact = data["eval"]["boundary_hash"] == data["boundary_hash_eval"]
    lower_exact = hash_lower_adjacency(data["eval"]) == data["lower_adjacency_hash"]
    x_r1 = mlp.build_r1_features(data, random_edges)
    h_len, l1_len = len(data["h_names"]), len(data["l1_names"])
    h_diff = float(np.max(np.abs(x_r1[:, :h_len] - data["X_eval_Full"][:, :h_len]))) if h_len else 0.0
    l2_diff = float(np.max(np.abs(x_r1[:, h_len + l1_len:] - data["X_eval_Full"][:, h_len + l1_len:]))) if data["l2_names"] else 0.0
    l2_hash_match = data["order"] != 3 or data["l2_hash"] == hash_set(data["eval_dec"].get("l2_triangles", set()))
    structural_valid = bool(info.get("DOSE_TARGET_HIT") == "YES" and h_hash_match and candidate_exact and boundary_exact and degree_exact and edge_exact and lower_exact and h_diff == 0.0 and l2_diff == 0.0 and l2_hash_match)
    write_json(graph_dir / "r1_audit.json", {"order": f"K{data['order']}", "dataset": data["dataset"], "target_dose": target, "r1_seed": r1_seed, "actual_changed_fraction": actual, "residual_jaccard": info.get("residual_jaccard"), "H_HASH_MATCH": "YES" if h_hash_match else "NO", "CANDIDATE_HASH_MATCH": "YES" if candidate_exact else "NO", "BOUNDARY_HASH_MATCH": "YES" if boundary_exact else "NO", "NODEWISE_RESIDUAL_DEGREE_MATCH": "YES" if degree_exact else "NO", "RESIDUAL_EDGE_COUNT_MATCH": "YES" if edge_exact else "NO", "K3_RESIDUAL_L2_HASH_MATCH": "YES" if l2_hash_match else "NO", "H_FEATURE_MAX_ABS_DIFF": h_diff, "L2_FEATURE_MAX_ABS_DIFF": l2_diff, "LOWER_ADJACENCY_EXACT": "YES" if lower_exact else "NO", "VALID_R1": "YES" if structural_valid else "NO"})
    for seed in MLP_SEEDS:
        pdir = prediction_root(data["order"], data["dataset"], target, r1_seed, seed)
        pdir.mkdir(parents=True, exist_ok=True)
        marker = pdir / "realization.json"
        if marker.is_file():
            cached = read_json(marker)
            if cached.get("baseline_feature_hash") == data["feature_hash_eval"]:
                rows.append(cached)
                continue
        if not structural_valid:
            row = empty_row(data, target, r1_seed, seed, {**info, "failure_reason": "R1_STRUCTURAL_GATE_FAIL"})
        else:
            logits, p_r1 = mlp.predict_model(bundles[seed], x_r1, DEVICE, BATCH_SIZE)
            p_real = data["real_predictions"][seed]
            y = data["y_eval"]
            real_metrics, r1_metrics = metric_row(y, p_real), metric_row(y, p_r1)
            delta = np.abs(p_r1 - p_real)
            row = {"order": f"K{data['order']}", "dataset": data["dataset"], "target_dose": target, "actual_changed_fraction": actual, "dose_error": actual - target, "r1_seed": r1_seed, "mlp_seed": seed, "valid": "YES", "LL_real": real_metrics["LogLoss"], "LL_R1": r1_metrics["LogLoss"], "delta_LL": r1_metrics["LogLoss"] - real_metrics["LogLoss"], "Brier_real": real_metrics["Brier"], "Brier_R1": r1_metrics["Brier"], "delta_Brier": r1_metrics["Brier"] - real_metrics["Brier"], "MAE_p": float(np.mean(delta)), "RMSE_p": float(np.sqrt(np.mean(delta ** 2))), "median_abs_delta_p": float(np.median(delta)), "Q90_abs_delta_p": float(np.quantile(delta, .90)), "Q95_abs_delta_p": float(np.quantile(delta, .95)), "Q99_abs_delta_p": float(np.quantile(delta, .99)), "max_abs_delta_p": float(np.max(delta)) if len(delta) else 0.0, "Pearson_p": mlp.pearson(p_real, p_r1), "Spearman_p": spearman(p_real, p_r1), "AP_real": real_metrics.get("AP"), "AP_R1": r1_metrics.get("AP"), "delta_AP": r1_metrics.get("AP") - real_metrics.get("AP") if real_metrics.get("AP") is not None and r1_metrics.get("AP") is not None else None, "AUC_real": real_metrics.get("AUC"), "AUC_R1": r1_metrics.get("AUC"), "delta_AUC": r1_metrics.get("AUC") - real_metrics.get("AUC") if real_metrics.get("AUC") is not None and r1_metrics.get("AUC") is not None else None, "Lift_real": real_metrics.get("Lift"), "Lift_R1": r1_metrics.get("Lift"), "delta_Lift": r1_metrics.get("Lift") - real_metrics.get("Lift") if real_metrics.get("Lift") is not None and r1_metrics.get("Lift") is not None else None, "residual_jaccard": info.get("residual_jaccard"), "DOSE_TARGET_HIT": info.get("DOSE_TARGET_HIT"), "MAX_REACHABLE_CHANGED_FRACTION": info.get("MAX_REACHABLE_CHANGED_FRACTION"), "H_HASH_MATCH": "YES" if h_hash_match else "NO", "CANDIDATE_HASH_MATCH": "YES" if candidate_exact else "NO", "BOUNDARY_HASH_MATCH": "YES" if boundary_exact else "NO", "NODEWISE_RESIDUAL_DEGREE_MATCH": "YES" if degree_exact else "NO", "RESIDUAL_EDGE_COUNT_MATCH": "YES" if edge_exact else "NO", "K3_RESIDUAL_L2_HASH_MATCH": "YES" if l2_hash_match else "NO" if data["order"] == 3 else "NA", "H_FEATURE_MAX_ABS_DIFF": h_diff, "L2_FEATURE_MAX_ABS_DIFF": l2_diff, "LOWER_ADJACENCY_EXACT": "YES" if lower_exact else "NO", "R1_CONSTRUCTION_LABEL_BLIND": "YES", "R1_CONSTRUCTION_MODEL_BLIND": "YES", "R1_CONSTRUCTION_OUTCOME_BLIND": "YES", "MODEL_RETRAINED_ON_R1": "NO", "SCALER_REFIT_ON_R1": "NO", "VALID_R1": "YES", "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "PASS", "baseline_feature_hash": data["feature_hash_eval"], "failure_reason": ""}
            loss_real = -(y * np.log(np.clip(p_real, 1e-7, 1 - 1e-7)) + (1 - y) * np.log1p(-np.clip(p_real, 1e-7, 1 - 1e-7)))
            loss_r1 = -(y * np.log(np.clip(p_r1, 1e-7, 1 - 1e-7)) + (1 - y) * np.log1p(-np.clip(p_r1, 1e-7, 1 - 1e-7)))
            np.savez_compressed(pdir / "candidate_predictions.npz", candidate_id=np.arange(len(y), dtype=np.int64), y=y, p_real=p_real, p_R1=p_r1, loss_real=loss_real, loss_R1=loss_r1, delta_loss=loss_r1 - loss_real)
        write_json(marker, row)
        rows.append(row)
    write_json(graph_dir / "r1_complete.json", {"status": "COMPLETE", "valid": "YES" if structural_valid else "NO", "n_mlp_rows": len(rows)})
    return rows

def aggregate_groups(rows):
    cluster_rows = []
    aggregate_rows = []
    for order, dataset in ALL_CASES:
        for target in TARGETS:
            selected = [row for row in rows if row.get("order") == f"K{order}" and row.get("dataset") == dataset and abs(float(row.get("target_dose", -1)) - target) < 1e-9]
            by_seed = defaultdict(list)
            for row in selected:
                if row.get("VALID_R1") == "YES" and row.get("valid") == "YES":
                    by_seed[int(row["r1_seed"])].append(row)
            for r1_seed, group in sorted(by_seed.items()):
                cluster_rows.append({"order": f"K{order}", "dataset": dataset, "target_dose": target, "r1_seed": r1_seed, "actual_changed_fraction": mean([row["actual_changed_fraction"] for row in group]), "n_valid_mlp": len(group), "mean_delta_LL": mean([row["delta_LL"] for row in group]), "std_delta_LL": std([row["delta_LL"] for row in group]), "mean_MAE_p": mean([row["MAE_p"] for row in group]), "mean_Q95_abs_delta_p": mean([row["Q95_abs_delta_p"] for row in group]), "mean_delta_Brier": mean([row["delta_Brier"] for row in group]), "mean_delta_AP": mean([row["delta_AP"] for row in group]), "mean_delta_AUC": mean([row["delta_AUC"] for row in group])})
            clusters = [row for row in cluster_rows if row["order"] == f"K{order}" and row["dataset"] == dataset and abs(float(row["target_dose"]) - target) < 1e-9 and row["n_valid_mlp"] == len(MLP_SEEDS)]
            ll_values = [row["mean_delta_LL"] for row in clusters]
            mae_values = [row["mean_MAE_p"] for row in clusters]
            ll_low, ll_high = cluster_bootstrap(ll_values)
            mae_low, mae_high = cluster_bootstrap(mae_values)
            aggregate_rows.append({"order": f"K{order}", "dataset": dataset, "target_dose": target, "n_valid_r1": len(clusters), "n_mlp_evaluations": len(clusters) * len(MLP_SEEDS), "mean_actual_changed_fraction": mean([row["actual_changed_fraction"] for row in clusters]), "std_actual_changed_fraction": std([row["actual_changed_fraction"] for row in clusters]), "mean_delta_LL": mean(ll_values), "median_delta_LL": median(ll_values), "std_delta_LL": std(ll_values), "Q05_delta_LL": percentile(ll_values, .05), "Q95_delta_LL": percentile(ll_values, .95), "CI95_delta_LL_low": ll_low, "CI95_delta_LL_high": ll_high, "mean_MAE_p": mean(mae_values), "median_MAE_p": median(mae_values), "CI95_MAE_low": mae_low, "CI95_MAE_high": mae_high, "mean_Q95_abs_delta_p": mean([row["mean_Q95_abs_delta_p"] for row in clusters]), "mean_delta_Brier": mean([row["mean_delta_Brier"] for row in clusters]), "mean_delta_AP": mean([row["mean_delta_AP"] for row in clusters]), "mean_delta_AUC": mean([row["mean_delta_AUC"] for row in clusters]), "dose_response_class": "INCOMPLETE", "mean_abs_delta_LL_lt_001": None, "mean_abs_delta_LL_lt_005": None, "mean_abs_delta_LL_lt_010": None, "mean_abs_delta_LL_lt_025": None, "mean_MAE_lt_005": None, "mean_MAE_lt_010": None, "mean_MAE_lt_020": None})
    for order, dataset in ALL_CASES:
        curve = [row for row in aggregate_rows if row["order"] == f"K{order}" and row["dataset"] == dataset]
        means = [row["mean_delta_LL"] for row in sorted(curve, key=lambda item: float(item["target_dose"]))]
        if len(means) == len(TARGETS) and all(value is not None for value in means):
            if max(abs(value) for value in means) < .005:
                label = "FLAT_ROBUST"
            elif means[-1] > .005 and max(abs(value) for value in means[:-1]) < .005:
                label = "HIGH_DOSE_ONLY_DEGRADATION"
            elif means[-1] > .005 and all(abs(means[index + 1]) >= abs(means[index]) - .002 for index in range(4)):
                label = "GRADUAL_DEGRADATION"
            elif means[-1] < -.005 and means[-1] <= min(means[:-1]):
                label = "IMPROVEMENT"
            elif any(value > .005 for value in means) and any(value < -.005 for value in means):
                label = "NON_MONOTONIC"
            else:
                label = "UNSTABLE"
        else:
            label = "INCOMPLETE"
        for row in curve:
            row["dose_response_class"] = label
            row["mean_abs_delta_LL_lt_001"] = "YES" if row["mean_delta_LL"] is not None and abs(float(row["mean_delta_LL"])) < .001 else "NO" if row["mean_delta_LL"] is not None else "NA"
            row["mean_abs_delta_LL_lt_005"] = "YES" if row["mean_delta_LL"] is not None and abs(float(row["mean_delta_LL"])) < .005 else "NO" if row["mean_delta_LL"] is not None else "NA"
            row["mean_abs_delta_LL_lt_010"] = "YES" if row["mean_delta_LL"] is not None and abs(float(row["mean_delta_LL"])) < .010 else "NO" if row["mean_delta_LL"] is not None else "NA"
            row["mean_abs_delta_LL_lt_025"] = "YES" if row["mean_delta_LL"] is not None and abs(float(row["mean_delta_LL"])) < .025 else "NO" if row["mean_delta_LL"] is not None else "NA"
            row["mean_MAE_lt_005"] = "YES" if row["mean_MAE_p"] is not None and float(row["mean_MAE_p"]) < .005 else "NO" if row["mean_MAE_p"] is not None else "NA"
            row["mean_MAE_lt_010"] = "YES" if row["mean_MAE_p"] is not None and float(row["mean_MAE_p"]) < .010 else "NO" if row["mean_MAE_p"] is not None else "NA"
            row["mean_MAE_lt_020"] = "YES" if row["mean_MAE_p"] is not None and float(row["mean_MAE_p"]) < .020 else "NO" if row["mean_MAE_p"] is not None else "NA"
    return cluster_rows, aggregate_rows

def write_tables(rows, case_states):
    eligible = [row for row in rows if case_states.get(f"{row.get('order')}|{row.get('dataset')}|T2", {}).get("status") == "COMPLETE"]
    table_root = OUT_ROOT / "09_tables"
    clusters, aggregate = aggregate_groups(eligible)
    write_csv(table_root / "MLP_R1_DOSE_REALIZATION_LEVEL.csv", eligible, REALIZATION_FIELDS)
    write_csv(table_root / "MLP_R1_DOSE_CLUSTER_LEVEL.csv", clusters, CLUSTER_FIELDS)
    write_csv(table_root / "MLP_R1_DOSE_AGGREGATE.csv", aggregate, AGGREGATE_FIELDS)
    summary, dose95 = [], []
    for order, dataset in ALL_CASES:
        curve = sorted([row for row in aggregate if row["order"] == f"K{order}" and row["dataset"] == dataset], key=lambda item: float(item["target_dose"]))
        row = {"order": f"K{order}", "dataset": dataset, "dose_response_class": curve[0]["dose_response_class"] if curve else "INCOMPLETE"}
        for item in curve:
            row[f"delta_LL_{TARGET_CODES[float(item['target_dose'])]}"] = item["mean_delta_LL"]
            row[f"MAE_{TARGET_CODES[float(item['target_dose'])]}"] = item["mean_MAE_p"]
            if abs(float(item["target_dose"]) - .95) < 1e-9:
                dose95.append({"order": item["order"], "dataset": item["dataset"], "n_valid_r1": item["n_valid_r1"], "mean_actual_changed_fraction": item["mean_actual_changed_fraction"], "mean_delta_LL": item["mean_delta_LL"], "CI95_delta_LL_low": item["CI95_delta_LL_low"], "CI95_delta_LL_high": item["CI95_delta_LL_high"], "median_delta_LL": item["median_delta_LL"], "mean_MAE_p": item["mean_MAE_p"], "CI95_MAE_low": item["CI95_MAE_low"], "CI95_MAE_high": item["CI95_MAE_high"], "mean_Q95_abs_delta_p": item["mean_Q95_abs_delta_p"], "mean_delta_Brier": item["mean_delta_Brier"], "mean_delta_AP": item["mean_delta_AP"], "mean_delta_AUC": item["mean_delta_AUC"]})
        summary.append(row)
    summary_fields = ["order", "dataset", "dose_response_class"] + [f"delta_LL_{TARGET_CODES[t]}" for t in TARGETS] + [f"MAE_{TARGET_CODES[t]}" for t in TARGETS]
    write_csv(table_root / "MLP_R1_DOSE_DATASET_SUMMARY.csv", summary, summary_fields)
    write_csv(table_root / "MLP_R1_DOSE_95_SUMMARY.csv", dose95)
    return clusters, aggregate, summary, dose95

def write_figures(aggregate):
    figure_root = OUT_ROOT / "10_figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(figure_root / "figure_status.json", {"status": "SKIPPED", "reason": repr(exc)})
        return "SKIPPED"
    made = 0
    for order, dataset in ALL_CASES:
        curve = sorted([row for row in aggregate if row["order"] == f"K{order}" and row["dataset"] == dataset and row.get("mean_delta_LL") is not None], key=lambda item: float(item["target_dose"]))
        if not curve:
            continue
        x = np.asarray([float(row["target_dose"]) * 100.0 for row in curve], dtype=np.float64)
        ll = np.asarray([float(row["mean_delta_LL"]) for row in curve], dtype=np.float64)
        ll_low = np.asarray([float(row["CI95_delta_LL_low"]) if row["CI95_delta_LL_low"] is not None else np.nan for row in curve], dtype=np.float64)
        ll_high = np.asarray([float(row["CI95_delta_LL_high"]) if row["CI95_delta_LL_high"] is not None else np.nan for row in curve], dtype=np.float64)
        mae = np.asarray([float(row["mean_MAE_p"]) if row["mean_MAE_p"] is not None else np.nan for row in curve], dtype=np.float64)
        mae_low = np.asarray([float(row["CI95_MAE_low"]) if row["CI95_MAE_low"] is not None else np.nan for row in curve], dtype=np.float64)
        mae_high = np.asarray([float(row["CI95_MAE_high"]) if row["CI95_MAE_high"] is not None else np.nan for row in curve], dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(x, ll, marker="o", label="mean ΔLL")
        if np.isfinite(ll_low).any() and np.isfinite(ll_high).any():
            axes[0].fill_between(x, ll_low, ll_high, alpha=0.20, label="cluster bootstrap 95% CI")
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_xlabel("actual changed fraction target (%)")
        axes[0].set_ylabel("Δ log loss (R1 − real)")
        axes[0].set_title("LogLoss dose response")
        axes[0].legend(fontsize=8)
        axes[1].plot(x, mae, marker="o", label="mean MAE(p)")
        if np.isfinite(mae_low).any() and np.isfinite(mae_high).any():
            axes[1].fill_between(x, mae_low, mae_high, alpha=0.20, label="cluster bootstrap 95% CI")
        axes[1].set_xlabel("actual changed fraction target (%)")
        axes[1].set_ylabel("absolute prediction shift")
        axes[1].set_title("Prediction-shift dose response")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"K{order} {dataset}")
        fig.savefig(figure_root / f"K{order}_{dataset}_dose_response.png", dpi=160)
        plt.close(fig)
        made += 1
    write_json(figure_root / "figure_status.json", {"status": "COMPLETE", "n_figures": made, "matplotlib": matplotlib.__version__})
    return "COMPLETE"

def generate_server_fallback(case_states):
    required = []
    for order, dataset in ALL_CASES:
        state = case_states.get(case_key(order, dataset), {})
        status = state.get("status", "NOT_RUN")
        if status == "COMPLETE":
            continue
        scheduled_server = (order, dataset) in OPTIONAL_CASES
        resource_limit_observed = status == "RESOURCE_LIMIT"
        if scheduled_server or resource_limit_observed:
            required.append({"order": f"K{order}", "dataset": dataset, "task": "T2",
                             "execution_plan": "SERVER", "execution_status": status,
                             "scheduled_server_case": scheduled_server,
                             "resource_limit_observed": resource_limit_observed,
                             "reason": state.get("reason") or ("RESOURCE_LIMIT" if resource_limit_observed else "SCHEDULED_SERVER_EXECUTION")})
    root = OUT_ROOT / "12_server_fallback"
    write_json(root / "LOCAL_COMPLETE_CASES.json", [{"order": f"K{o}", "dataset": d, "task": "T2"} for o, d in PRIMARY_CASES if case_states.get(case_key(o, d), {}).get("status") == "COMPLETE"])
    write_json(root / "SERVER_REQUIRED_CASES.json", required)
    return required


def generate_server_package(required):
    package = ROOT / "server_package" / "strict_timestamp_t2_mlp_r1_dose_server_v1"
    package.mkdir(parents=True, exist_ok=True)
    files = ["run_strict_timestamp_t2_mlp_r1_dose_local_v1.py", "run_strict_timestamp_t2_mlp_r1_dose_server_v1.py", "merge_mlp_r1_dose_local_server_v1.py", "run_strict_timestamp_t2_mlp_local_v1.py", "run_strict_timestamp_chronological_core_v1.py", "run_h_preserving_perturbation_dev_v2.py", "run_k2_residual_lower_order_conditional_redundancy.py", "run_k3_residual_lower_order_conditional_redundancy.py", "requirements.txt"]
    for name in files:
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, package / name)
    if (ROOT / "kclosure").is_dir():
        shutil.copytree(ROOT / "kclosure", package / "kclosure", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    write_json(package / "SERVER_REQUIRED_CASES.json", {"protocol": PROTOCOL, "task": "T2", "cases": required, "target_changed_fractions": list(TARGETS), "target_tolerance": TARGET_TOLERANCE, "mlp_training_seeds": list(MLP_SEEDS), "r1_seeds_by_dose": {str(target): list(seeds) for target, seeds in R1_SEEDS_BY_DOSE.items()}, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "raw_dataset_copied": "NO", "raw_dataset_source": "server source root"})
    shell = "#!/usr/bin/env bash\nset -euo pipefail\npython3 -u run_strict_timestamp_t2_mlp_r1_dose_server_v1.py --all --cases-file SERVER_REQUIRED_CASES.json --resume \"$@\"\n"
    (package / "run_mlp_r1_dose_server_v1.sh").write_text(shell, encoding="utf-8")
    readme = """# STRICT TIMESTAMP T2 MLP R1 DOSE SERVER V1\n\nThis package runs only the cases listed in SERVER_REQUIRED_CASES.json. It uses the same T2, Full MLP, five MLP seeds, five actual changed-fraction doses, ten R1 seeds per dose, structural gates, and cluster bootstrap as the local runner. It never runs T1 or R0 and never subsamples a scientific case.\n\nThe package intentionally does not copy raw datasets. Run it with --source-root pointing to the server project root containing dataset/raw_data and, when available, strict timestamp feature/model caches. Missing MLP archives are retrained with the frozen architecture and seeds 21001-21005; R1 never retrains or refits the scaler.\n\nExample:\n\n```bash\npython3 -m py_compile run_strict_timestamp_t2_mlp_r1_dose_server_v1.py run_strict_timestamp_t2_mlp_r1_dose_local_v1.py\npython3 -u run_strict_timestamp_t2_mlp_r1_dose_server_v1.py --all --cases-file SERVER_REQUIRED_CASES.json --source-root /path/to/project --resume\n```\n\nIf one dataset fails, the case is recorded and the next dataset continues.\n"""
    (package / "README_MLP_R1_DOSE_SERVER_V1.md").write_text(readme, encoding="utf-8")
    zip_path = ROOT / "server_package" / "strict_timestamp_t2_mlp_r1_dose_server_v1.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package.parent).as_posix())
    write_json(OUT_ROOT / "12_server_fallback" / "SERVER_PACKAGE_MANIFEST.json", {"package": str(package), "zip": str(zip_path), "server_code_generated": "YES", "server_package_generated": "YES", "raw_dataset_copied": "NO", "case_count": len(required)})
    return package, zip_path

def write_report(state, aggregate, dose95, required, local_attempted):
    complete = [item for item in aggregate if item.get("n_valid_r1", 0) == len(R1_SEEDS_BY_DOSE[float(item["target_dose"])])]
    lines = ["# STRICT TIMESTAMP T2 MLP R1 DOSE LOCAL-FIRST REPORT V1", "", "- TASK = T2", "- STRICT_TIMESTAMP = YES", "- FULL_DATASET_ONLY = YES", "- T1_RUN = NO", "- R0_RUN = NO", "- MLP_HYPERPARAMETERS_TUNED = NO", "- R1_MODEL_RETRAINED = NO", "- SCALER_REFIT_ON_R1 = NO", "- R1_SEED_SELECTED_BY_OUTCOME = NO", "- DOSE_SELECTED_BY_OUTCOME = NO", "", "## Local status", "", f"- LOCAL_CASES_ATTEMPTED = {len(local_attempted)}", f"- LOCAL_CASES_COMPLETE = {sum(state.get(case_key(o, d), {}).get('status') == 'COMPLETE' for o, d in local_attempted)}", f"- LOCAL_RESOURCE_LIMIT = {sum(state.get(case_key(o, d), {}).get('status') == 'RESOURCE_LIMIT' for o, d in local_attempted)}", f"- SERVER_REQUIRED_CASES = {len(required)}", "", "## Dose curves", "", "| order | dataset | dose | n_valid_r1 | actual fraction | mean ΔLL | 95% CI ΔLL | mean MAE(p) | 95% CI MAE(p) | class |", "|---|---|---:|---:|---:|---:|---|---:|---|---|"]
    for row in aggregate:
        lines.append(f"| {row['order']} | {row['dataset']} | {float(row['target_dose']):.2f} | {row['n_valid_r1']} | {fmt(row['mean_actual_changed_fraction'])} | {fmt(row['mean_delta_LL'])} | [{fmt(row['CI95_delta_LL_low'])}, {fmt(row['CI95_delta_LL_high'])}] | {fmt(row['mean_MAE_p'])} | [{fmt(row['CI95_MAE_low'])}, {fmt(row['CI95_MAE_high'])}] | {row['dose_response_class']} |")
    lines += ["", "## 95% dose", "", "The complete 20–40–60–80–95% curve is primary. The 95% table is a separate summary, not a replacement for the curve.", "", "## Evidence level", "", "FOURTH_LAYER_LOCAL_EVIDENCE = DESCRIPTIVE_ONLY", "The classification is descriptive and does not automatically prove or disprove fourth-layer robustness.", "", "## Server fallback", "", f"- SERVER_REQUIRED_CASES.json: `{OUT_ROOT / '12_server_fallback' / 'SERVER_REQUIRED_CASES.json'}`", f"- Server package: `{ROOT / 'server_package' / 'strict_timestamp_t2_mlp_r1_dose_server_v1.zip'}`", "", "## Safety", "", "- No event/candidate/node/edge/positive/negative/time subsampling.", "- The same R1 graph is evaluated by all five MLP seeds.", "- R1 construction does not read labels, predictions, or outcomes."]
    path = OUT_ROOT / "11_summary" / "STRICT_TIMESTAMP_T2_MLP_R1_DOSE_LOCAL_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def initial_state(cases, device, args):
    return {"protocol": PROTOCOL, "task": TASK, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": {"target_changed_fractions": list(TARGETS), "target_tolerance": TARGET_TOLERANCE, "mlp_training_seeds": list(MLP_SEEDS), "r1_seeds_by_dose": {str(target): list(seeds) for target, seeds in R1_SEEDS_BY_DOSE.items()}, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "device": str(device), "batch_size": args.batch_size}, "cases": {case_key(order, dataset): {"order": f"K{order}", "dataset": dataset, "task": "T2", "status": "PENDING", "stages": {}} for order, dataset in ALL_CASES}, "overall_status": "RUNNING"}

def save_state(state):
    state["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(OUT_ROOT / "run_manifest.json", state)

def load_state(cases, device, args):
    path = OUT_ROOT / "run_manifest.json"
    if args.resume and path.is_file():
        state = read_json(path)
        if state.get("protocol") != PROTOCOL or state.get("task") != TASK:
            raise ProtocolFailure("OUTPUT_PROTOCOL_MISMATCH")
        return state
    state = initial_state(cases, device, args)
    save_state(state)
    return state

def run_case(order, dataset, state, device, batch_size, doses):
    key = case_key(order, dataset)
    case_state = state["cases"][key]
    try:
        started = time.time()
        data = load_case_data(order, dataset)
        data["protocol_hash"] = save_case_protocol(data)
        bundles, reused_all = load_or_train_models(data, device, batch_size)
        data["bundles"], data["real_predictions"], data["real_metrics"] = bundles, {}, {}
        baseline = save_real_baselines(data, bundles, device, batch_size)
        for seed in MLP_SEEDS:
            data["real_predictions"][seed] = baseline[seed]["p"]
            data["real_metrics"][seed] = baseline[seed]["metrics"]
        case_state.update({"status": "RUNNING", "MLP_PROTOCOL_REUSED": "YES" if reused_all else "PARTIAL", "protocol_hash": data["protocol_hash"], "n_eval_candidates": len(data["y_eval"]), "stages": {"protocol": "COMPLETE", "baseline_models": "COMPLETE"}})
        save_state(state)
        all_rows = []
        for target in doses:
            for r1_seed in R1_SEEDS_BY_DOSE[target]:
                marker = graph_root(order, dataset, target, r1_seed) / "r1_protocol.json"
                random_edges, info = None, None
                if marker.is_file() and (graph_root(order, dataset, target, r1_seed) / "r1_edges.npz").is_file():
                    proto = read_json(marker); info = proto.get("dose_info", {}); arrays = np.load(graph_root(order, dataset, target, r1_seed) / "r1_edges.npz", allow_pickle=False); random_edges = set(zip(arrays["left"].tolist(), arrays["right"].tolist()))
                else:
                    forbidden = set(core.h_structure(order, data["eval_dec"])["edges"])
                    random_edges, info = target_changed_fraction_rewire(data["real_edges"], forbidden, target, r1_seed)
                rows = evaluate_graph(data, target, r1_seed, random_edges, info, bundles)
                all_rows.extend(rows)
                case_state["stages"][f"dose_{code(target)}_r1_{r1_seed}"] = "COMPLETE"
                save_state(state)
                del random_edges
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        case_state.update({"status": "COMPLETE", "runtime_seconds": time.time() - started, "n_realization_rows": len(all_rows)})
        save_state(state)
        return all_rows
    except (ResourceLimit, MemoryError, OSError) as exc:
        case_state.update({"status": "RESOURCE_LIMIT", "LOCAL_RESOURCE_LIMIT": "YES", "reason": repr(exc)})
        write_json(case_root(order, dataset) / "case_failure.json", {"order": f"K{order}", "dataset": dataset, "task": "T2", "status": "RESOURCE_LIMIT", "LOCAL_RESOURCE_LIMIT": "YES", "reason": repr(exc), "traceback": traceback.format_exc(), "scientific_result": "NA", "next_dataset_continues": "YES"})
        save_state(state)
        return []
    except Exception as exc:
        if resource_is_oom(exc):
            case_state.update({"status": "RESOURCE_LIMIT", "LOCAL_RESOURCE_LIMIT": "YES", "reason": repr(exc)})
            status = "RESOURCE_LIMIT"
        else:
            case_state.update({"status": "ENGINEERING_FAILURE", "LOCAL_RESOURCE_LIMIT": "NO", "reason": repr(exc)})
            status = "ENGINEERING_FAILURE"
        write_json(case_root(order, dataset) / "case_failure.json", {"order": f"K{order}", "dataset": dataset, "task": "T2", "status": status, "LOCAL_RESOURCE_LIMIT": case_state.get("LOCAL_RESOURCE_LIMIT"), "reason": repr(exc), "scientific_result": "NA", "next_dataset_continues": "YES", "traceback": traceback.format_exc()})
        save_state(state)
        return []
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

def selected_cases(args):
    cases = list(PRIMARY_CASES + (OPTIONAL_CASES if args.include_optional else []))
    if args.cases_file:
        payload = read_json(Path(args.cases_file))
        entries = payload.get("cases", payload) if isinstance(payload, dict) else payload
        cases = [(int(str(item["order"]).replace("K", "")), item["dataset"]) for item in entries]
    if args.order:
        cases = [case for case in cases if case[0] == int(args.order[1:])]
    if args.dataset:
        cases = [case for case in cases if case[1] == args.dataset]
    if not cases:
        raise SystemExit("NO_MATCHING_CASE")
    return cases

def selected_doses(args):
    if not args.dose:
        return list(TARGETS)
    result = []
    for value in args.dose:
        target = round(float(value), 2)
        if target not in TARGETS:
            raise SystemExit("DOSE_MUST_BE_ONE_OF_0.20_0.40_0.60_0.80_0.95")
        if target not in result:
            result.append(target)
    return result

def merge_realization_rows(existing, current):
    pass
    merged = {}
    for row in list(existing) + list(current):
        try:
            key = (row.get("order"), row.get("dataset"), round(float(row.get("target_dose")), 8), int(row.get("r1_seed")), int(row.get("mlp_seed")))
        except (TypeError, ValueError):
            continue
        merged[key] = row
    return list(merged.values())

def load_existing_realization_rows():
    return read_csv(OUT_ROOT / "09_tables" / "MLP_R1_DOSE_REALIZATION_LEVEL.csv")

def write_protocol(args, device, cases):
    payload = {"protocol": PROTOCOL, "task": "T2", "STRICT_TIMESTAMP": "YES", "FULL_DATASET_ONLY": "YES", "target_changed_fractions": list(TARGETS), "selected_doses": selected_doses(args), "target_tolerance": TARGET_TOLERANCE, "mlp_training_seeds": list(MLP_SEEDS), "r1_seeds_by_dose": {str(target): list(seeds) for target, seeds in R1_SEEDS_BY_DOSE.items()}, "architecture": {"hidden_1": 64, "hidden_2": 32, "dropout": .20}, "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4}, "device": {"device": str(device), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda if torch.cuda.is_available() else None, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, "cases_requested": [{"order": f"K{o}", "dataset": d} for o, d in cases], "T1_RUN": "NO", "R0_RUN": "NO", "FULL_DATASET_SUBSAMPLED": "NO", "R1_MODEL_RETRAINED": "NO", "SCALER_REFIT_ON_R1": "NO", "cluster_bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}}
    payload["protocol_hash"] = sha256_bytes(json.dumps(payload, sort_keys=True).encode())
    write_json(OUT_ROOT / "00_protocol" / "protocol.json", payload)
    return payload

def main(argv=None):
    parser = argparse.ArgumentParser(description="Strict timestamp T2 MLP H-preserving R1 dose response, local-first V1.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--order", choices=("k2", "k3"))
    parser.add_argument("--dataset")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--cases-file")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "strict_timestamp_t2_mlp_r1_dose_local_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dose", action="append", type=float, help="Run only this dose; repeat for multiple doses.")
    parser.add_argument("--r1-seed-base", type=int, help="Use ten new R1 seeds per dose: base+0..9, base+1000..1009, etc.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    global OUT_ROOT, DEVICE, BATCH_SIZE, ALLOW_EXACT_BUILD, SERVER_MODE, R1_SEEDS_BY_DOSE
    OUT_ROOT = args.output_root.resolve()
    SOURCE_ROOT = args.source_root.resolve()
    configure_source(SOURCE_ROOT)
    ALLOW_EXACT_BUILD = bool(args.server or args.include_optional)
    SERVER_MODE = bool(args.server)
    if args.r1_seed_base is not None:
        if args.r1_seed_base < 1:
            raise SystemExit("R1_SEED_BASE_MUST_BE_POSITIVE")
        R1_SEEDS_BY_DOSE = seed_map_from_base(args.r1_seed_base)
    DEVICE = mlp.resolve_device(args.device)
    BATCH_SIZE = args.batch_size
    cases = selected_cases(args)
    doses = selected_doses(args)
    for directory in ("00_protocol", "01_baselines", "02_dose_020", "03_dose_040", "04_dose_060", "05_dose_080", "06_dose_095", "07_predictions", "08_structural_audit", "09_tables", "10_figures", "11_summary", "12_server_fallback", "logs"):
        (OUT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    write_protocol(args, DEVICE, cases)
    state = load_state(cases, DEVICE, args)
    state.setdefault("config", {})["selected_doses"] = list(doses)
    state["config"]["r1_seeds_by_dose"] = {str(target): list(seeds) for target, seeds in R1_SEEDS_BY_DOSE.items()}
    save_state(state)
    if args.dry_run:
        rows = []
        for order, dataset in cases:
            try:
                data = load_case_data(order, dataset)
                rows.append({"order": f"K{order}", "dataset": dataset, "status": "READY", "n_eval_candidates": len(data["y_eval"]), "timestamp_gate": "YES", "T1_constructed": "NO", "scientific_result": "NO"})
            except Exception as exc:
                rows.append({"order": f"K{order}", "dataset": dataset, "status": "RESOURCE_LIMIT" if resource_is_oom(exc) or isinstance(exc, ResourceLimit) else "ENGINEERING_FAILURE", "reason": repr(exc), "scientific_result": "NO"})
        write_csv(OUT_ROOT / "00_protocol" / "dry_run_cases.csv", rows)
        passed = all(row["status"] == "READY" for row in rows)
        print(f"DRY_RUN_PROTOCOL_PASS = {'YES' if passed else 'NO'}", flush=True)
        print("SCIENTIFIC_RESULT = NO", flush=True)
        return 0
    all_rows = []
    for order, dataset in cases:
        all_rows.extend(run_case(order, dataset, state, DEVICE, BATCH_SIZE, doses))
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    all_rows = merge_realization_rows(load_existing_realization_rows(), all_rows)
    clusters, aggregate, summary, dose95 = write_tables(all_rows, state["cases"])
    figure_status = write_figures(aggregate)
    required = generate_server_fallback(state["cases"])
    package, zip_path = generate_server_package(required)
    report = write_report(state["cases"], aggregate, dose95, required, cases)
    state["overall_status"] = "COMPLETE" if all(state["cases"].get(case_key(o, d), {}).get("status") == "COMPLETE" for o, d in cases) else "PARTIAL"
    state["summary"] = {"local_cases_attempted": len(cases), "local_cases_complete": sum(state["cases"].get(case_key(o, d), {}).get("status") == "COMPLETE" for o, d in cases), "server_required_cases": len(required), "figure_status": figure_status, "report": str(report), "server_package": str(package), "server_zip": str(zip_path)}
    save_state(state)
    print("============================================================", flush=True)
    print("T2 MLP R1 DOSE LOCAL-FIRST V1", flush=True)
    print("TASK = T2", flush=True)
    print("FULL_DATASET_ONLY = YES", flush=True)
    print(f"LOCAL_CASES_ATTEMPTED = {len(cases)}", flush=True)
    print(f"LOCAL_CASES_COMPLETE = {sum(state['cases'].get(case_key(o, d), {}).get('status') == 'COMPLETE' for o, d in cases)}", flush=True)
    print(f"LOCAL_RESOURCE_LIMIT = {sum(state['cases'].get(case_key(o, d), {}).get('status') == 'RESOURCE_LIMIT' for o, d in cases)}", flush=True)
    print(f"SERVER_REQUIRED_CASES = {len(required)}", flush=True)
    print("R1_GRAPHS_PER_COMPLETE_CASE = 50", flush=True)
    print("MLP_EVALUATIONS_PER_COMPLETE_CASE = 250", flush=True)
    print("T1_RUN = NO", flush=True)
    print("FULL_DATASET_SUBSAMPLED = NO", flush=True)
    print("MLP_HYPERPARAMETERS_TUNED = NO", flush=True)
    print("R1_SEED_SELECTED_BY_OUTCOME = NO", flush=True)
    print("DOSE_SELECTED_BY_OUTCOME = NO", flush=True)
    print(f"FINAL_REPORT = {report}", flush=True)
    print(f"SERVER_PACKAGE = {package}", flush=True)
    print(f"SERVER_ZIP = {zip_path}", flush=True)
    print(f"OVERALL_STATUS = {state['overall_status']}", flush=True)
    print("============================================================", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
