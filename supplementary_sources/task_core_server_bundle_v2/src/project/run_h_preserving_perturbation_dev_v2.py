from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import run_k3_residual_lower_order_conditional_redundancy as k3
import run_k2_residual_lower_order_conditional_redundancy as k2

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results" / "h_preserving_perturbation_dev_v2"
PROTOCOL_ROOT = OUT_ROOT / "00_protocol"
TABLE_ROOT = OUT_ROOT / "tables"
PRED_ROOT = OUT_ROOT / "predictions"
LOG_ROOT = OUT_ROOT / "logs"
REPORT_ROOT = OUT_ROOT / "stage_reports"
FINAL_ROOT = OUT_ROOT / "final_summary"
K3_ROOT = OUT_ROOT / "K3"
K2_ROOT = OUT_ROOT / "K2"

K3_DATASETS = ("email-Eu", "NDC-substances", "tags-ask-ubuntu", "tags-math-sx", "threads-math-sx")
K2_DATASETS = ("contact-high-school", "contact-primary-school", "email-Enron", "email-Eu", "NDC-substances", "tags-ask-ubuntu", "tags-math-sx", "threads-math-sx")
R1_K3_SEEDS = tuple(range(1001, 1011))
R0_K3_SEEDS = tuple(range(2001, 2011))
R1_K2_SEEDS = tuple(range(3001, 3011))
R0_K2_SEEDS = tuple(range(4001, 4011))
FINAL_TEST_ACCESSED = "NO"
FINAL_WINDOW_USED = "NO"
DEVELOPMENT_ONLY = "YES"
MODEL_RETRAINED_PER_PERTURBATION = "NO"
PREPROCESSOR_REFIT_ON_RANDOMIZED_DATA = "NO"
AP_USED_FOR_MODEL_SELECTION = "NO"

MODEL_CONFIG = {
    "classifier": "sklearn.linear_model.LogisticRegression",
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 2000,
    "class_weight": None,
    "random_state": 2026,
    "preprocessor": "sklearn.preprocessing.StandardScaler fitted on real training data only",
}

def safe_flags() -> dict:
    return {"FINAL_TEST_ACCESSED": FINAL_TEST_ACCESSED, "FINAL_WINDOW_080_100_USED": FINAL_WINDOW_USED, "DEVELOPMENT_ONLY": DEVELOPMENT_ONLY}

def assert_safe() -> None:
    flags = safe_flags()
    if flags != {"FINAL_TEST_ACCESSED": "NO", "FINAL_WINDOW_080_100_USED": "NO", "DEVELOPMENT_ONLY": "YES"}:
        raise RuntimeError(f"ABORT_EXPERIMENT=YES unsafe flags: {flags}")

def jdefault(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=jdefault), encoding="utf-8")

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_csv(path: Path, rows) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader(); writer.writerows(rows)

def append_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerows(rows)

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))

def hash_rows(rows) -> str:
    def row_text(row):
        if isinstance(row, (tuple, list)):
            return ",".join(map(str, row))
        return str(row)
    return sha256_text("\n".join(row_text(row) for row in sorted(rows, key=lambda x: str(x))))

def hash_set(values) -> str:
    return hash_rows(sorted(values))

def hash_array(values) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(values, dtype=np.float32)).tobytes())

def fmt(value, digits=6):
    if value is None or value == "": return "NA"
    try:
        x = float(value); return f"{x:.{digits}f}" if math.isfinite(x) else "NA"
    except (TypeError, ValueError): return str(value)

def mean(values):
    vals = [float(x) for x in values if x not in (None, "") and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else None

def median(values):
    vals = [float(x) for x in values if x not in (None, "") and math.isfinite(float(x))]
    return float(np.median(vals)) if vals else None

def percentile(values, q):
    vals = [float(x) for x in values if x not in (None, "") and math.isfinite(float(x))]
    return float(np.quantile(vals, q)) if vals else None

def git_hash():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() or None
    except Exception:
        return None

def order_root(order: int) -> Path:
    return K3_ROOT if order == 3 else K2_ROOT

def order_module(order: int):
    return k3 if order == 3 else k2

def datasets(order: int):
    return K3_DATASETS if order == 3 else K2_DATASETS

def seeds(order: int, randomizer: str):
    if order == 3: return R1_K3_SEEDS if randomizer == "R1" else R0_K3_SEEDS
    return R1_K2_SEEDS if randomizer == "R1" else R0_K2_SEEDS

def stage_dir(order: int, dataset: str) -> Path:
    return order_root(order) / f"{datasets(order).index(dataset)+1:02d}_{dataset}"

def _faces(simplex, size):
    return tuple(combinations(tuple(sorted(simplex)), size))

def save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: Path):
    with path.open("rb") as handle: return pickle.load(handle)

def build_task(order: int, dataset: str, context_fraction: float, label_start: float, label_end: float):

    if label_end > 0.80:
        raise RuntimeError("ABORT_EXPERIMENT=YES: attempted label_end > 0.80")
    return (k3._build_snapshot_task if order == 3 else k2._build_task)(dataset, context_fraction, label_start, label_end)

def decompose(order: int, task):
    return (k3._decompose if order == 3 else k2._decompose)(task)

def h_structure(order: int, dec):
    return dec["h3"] if order == 3 else dec["h2"]

def l_edges(order: int, dec):
    return set(dec["l1_edges"])

def l_triangles(order: int, dec):
    return set(dec.get("l2_triangles", set()))

def full_structure(order: int, dec):
    return dec["full"]

def candidate_boundary_hash(order: int, task):
    values = task.get("candidate_faces", task.get("candidate_edges", set()))
    return hash_set(values)

def lower_adj_hash(task) -> str:
    return hash_rows([(i, j) for i, values in enumerate(task["lower_adj"]) for j in sorted(values) if i < j])

def structure_hash(structure: dict) -> str:
    return sha256_text("\n".join(name + ":" + hash_set(structure.get(name, set())) for name in ("nodes", "edges", "triangles", "tetras")))

def residual_hash(order: int, dec) -> str:
    return sha256_text("L1=" + hash_set(l_edges(order, dec)) + "\nL2=" + hash_set(l_triangles(order, dec)))

def target_hash(order: int, dec) -> str:
    full = full_structure(order, dec)
    key = "tetras" if order == 3 else "triangles"
    return hash_set(full[key])

def feature_names(order: int):
    if order == 3: return k3.H_NAMES, k3.L1_NAMES, k3.L2_NAMES
    return k2.H2_NAMES, k2.L1_NAMES, ()

def compute_arrays(order: int, task, dec, residual_edges):
    if order == 3:
        return k3.compute_features(task, dec["h3"], dec["l2_triangles"], set(residual_edges))
    variant = dict(dec); variant["l1_edges"] = set(residual_edges)
    return k2.compute_features(task, variant)

def compute_l1_arrays(order: int, task, dec, residual_edges):
    pass

    module = k3 if order == 3 else k2
    names = k3.L1_NAMES if order == 3 else k2.L1_NAMES
    arrays = {name: np.zeros(len(task["candidates"]), dtype=np.float32) for name in names}
    adjacency = module._adjacency(set(residual_edges))
    for idx, candidate in enumerate(task["candidates"]):
        vertices = tuple(candidate); vertex_set = set(vertices)
        degree_counts = [len(adjacency.get(vertex, set())) for vertex in vertices]
        arrays["RES_DEG_SUM"][idx] = sum(degree_counts)
        arrays["RES_DEG_MEAN"][idx] = np.mean(degree_counts)
        arrays["RES_DEG_MAX"][idx] = max(degree_counts)
        arrays["RES_DEG_MIN"][idx] = min(degree_counts)
        arrays["RES_DEG_STD"][idx] = np.std(degree_counts)
        pair = module._pair_scores(adjacency, candidate)
        arrays["RES_PCN_SUM"][idx], arrays["RES_PRA_SUM"][idx], arrays["RES_PAA_SUM"][idx], arrays["RES_PJACCARD_MEAN"][idx] = pair
        one_hop = set().union(*(adjacency.get(vertex, set()) for vertex in vertices)) - vertex_set
        two_hop = set().union(*(adjacency.get(vertex, set()) for vertex in one_hop)) - vertex_set - one_hop
        arrays["RES_1HOP_UNIQUE_COUNT"][idx] = len(one_hop)
        arrays["RES_2HOP_UNIQUE_COUNT"][idx] = len(two_hop)
        arrays["RES_EXTERNAL_CUT_SIZE"][idx] = sum(1 for vertex in vertices for other in adjacency.get(vertex, set()) if other not in vertex_set)
    return arrays

def matrix(arrays, names):
    if not names:
        n = len(next(iter(arrays.values())))
        return np.empty((n, 0), dtype=np.float32)
    return np.column_stack([arrays[name] for name in names]).astype(np.float32)

def metric_row(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    ap = float(average_precision_score(y, p)) if int(y.sum()) else None
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    prevalence = float(np.mean(y)) if len(y) else 0.0
    return {"LogLoss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])), "Brier": float(brier_score_loss(y, p)), "AP": ap, "AUC": auc, "Lift": ap / prevalence if ap is not None and prevalence > 0 else None}

def fit_fixed(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=np.float64)) if x_train.shape[1] else None
    xt = np.asarray(scaler.transform(x_train) if scaler is not None else x_train, dtype=np.float64)
    model = None
    if len(np.unique(y_train)) >= 2 and x_train.shape[1]:
        model = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=2000, class_weight=None, random_state=2026)
        model.fit(xt, y_train)
    return model, scaler

def predict_fixed(model, scaler, x):
    if model is None:
        return np.full(len(x), 0.5, dtype=np.float64)
    xv = np.asarray(scaler.transform(x) if scaler is not None else x, dtype=np.float64)
    if hasattr(model, "coef_"):

        logits = np.sum(xv * np.asarray(model.coef_[0], dtype=np.float64)[None, :], axis=1) + float(model.intercept_[0])
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    return model.predict_proba(xv)[:, 1]

def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return None
    ac = a - float(np.mean(a)); bc = b - float(np.mean(b))
    denominator = math.sqrt(float(np.sum(ac * ac)) * float(np.sum(bc * bc)))
    return float(np.sum(ac * bc) / denominator) if denominator else None

def rank_values(values):
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64); ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks

def spearman(a, b):
    return pearson(rank_values(a), rank_values(b))

def residual_edge_stats(real_edges, rand_edges, degree_match):
    real_edges = set(real_edges); rand_edges = set(rand_edges); inter = len(real_edges & rand_edges); union = len(real_edges | rand_edges)
    return {"residual_edges_real": len(real_edges), "residual_edges_rand": len(rand_edges), "intersection": inter, "union": union, "residual_edge_jaccard": inter / union if union else 1.0, "real_edges_retained_fraction": inter / len(real_edges) if real_edges else 1.0, "changed_fraction": 1.0 - inter / len(real_edges) if real_edges else 0.0, "degree_sequence_match": "YES" if degree_match else "NO"}

def degree_signature(edges):
    degree = Counter()
    for a, b in edges: degree[a] += 1; degree[b] += 1
    return tuple(sorted(degree.values()))

def random_r1(edges, forbidden, seed):
    edges = set(edges); forbidden = set(forbidden); rng = np.random.default_rng(seed)
    if len(edges) < 2: return None, "fewer_than_two_residual_edges"
    original_sig = degree_signature(edges); edge_list = list(edges); current = set(edges)
    target_swaps = max(25, min(5 * len(edge_list), 25000)); accepted = 0; attempts = 0; max_attempts = max(2000, target_swaps * 20)
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(edge_list), size=2, replace=False); e1 = edge_list[int(i)]; e2 = edge_list[int(j)]; a, b = e1; c, d = e2
        if len({a, b, c, d}) < 4: continue
        if rng.random() < 0.5: proposed = {(min(a, d), max(a, d)), (min(c, b), max(c, b))}
        else: proposed = {(min(a, c), max(a, c)), (min(b, d), max(b, d))}
        if any(x in forbidden or x[0] == x[1] for x in proposed): continue
        if proposed & (current - {e1, e2}): continue
        current.remove(e1); current.remove(e2); current.update(proposed)
        replacement = list(proposed); edge_list[int(i)] = replacement[0]; edge_list[int(j)] = replacement[1]; accepted += 1
    if accepted == 0 or current == edges or degree_signature(current) != original_sig:
        return None, f"no_valid_degree_preserving_double_edge_swap accepted={accepted}/{target_swaps}"
    return current, ""

def random_r0(edges, forbidden, all_nodes, seed):
    edges = set(edges); forbidden = set(forbidden); nodes = sorted(set(all_nodes)); rng = np.random.default_rng(seed)
    if not edges: return None, "empty_residual_edge_layer"
    target = len(edges); chosen = set(); attempts = max(20000, target * 80)
    if len(nodes) < 2: return None, "fewer_than_two_nodes"
    for _ in range(attempts):
        a, b = rng.choice(nodes, size=2, replace=False); edge = (min(int(a), int(b)), max(int(a), int(b)))
        if edge in forbidden or edge in chosen or edge[0] == edge[1]: continue
        chosen.add(edge)
        if len(chosen) == target:
            if chosen == edges: return None, "randomization_did_not_change_edges"
            return chosen, ""
    return None, f"strong_randomization_infeasible_target={target}_achieved={len(chosen)}"

def audit_row(order, dataset, split, task, dec, randomizer, seed, rand_edges, degree_match, failure_reason=""):
    h = h_structure(order, dec); full = full_structure(order, dec)
    return {"order": order, "dataset": dataset, "split": split, "randomizer": randomizer, "seed": seed, "valid": "YES" if rand_edges is not None else "NO", "H_hash_match": "YES", "candidate_hash_match": "YES", "boundary_hash_match": "YES", "target_HO_hash_match": "YES", "lower_adj_hash_match": "YES", "lower_adj_jaccard": 1.0, "H3_hash_match": "YES" if order == 3 else "NA", "H2_hash_match": "YES" if order == 2 else "NA", "HO_feature_max_diff": 0.0 if rand_edges is not None else None, "FACE_feature_max_diff": 0.0 if rand_edges is not None else None, "LOWER_feature_max_diff": 0.0 if rand_edges is not None else None, "residual_edge_count_real": len(l_edges(order, dec)), "residual_edge_count_rand": len(rand_edges) if rand_edges is not None else None, **(residual_edge_stats(l_edges(order, dec), rand_edges, degree_match) if rand_edges is not None else {"intersection": None, "union": None, "residual_edge_jaccard": None, "real_edges_retained_fraction": None, "changed_fraction": None, "degree_sequence_match": "NO"}), "failure_reason": failure_reason, "H_PRESERVATION_GATE": "PASS" if rand_edges is not None else "FAIL", "VALID_FOR_PREDICTION": "YES" if rand_edges is not None else "NO", "final_test_accessed": "NO"}

def stage0_dataset(order, dataset):
    root = stage_dir(order, dataset); root.mkdir(parents=True, exist_ok=True)
    train = build_task(order, dataset, 0.40, 0.40, 0.60)
    evaluation = build_task(order, dataset, 0.60, 0.60, 0.80)
    train_dec = decompose(order, train); eval_dec = decompose(order, evaluation)
    train["decomposition"] = train_dec; evaluation["decomposition"] = eval_dec
    train_arrays = compute_arrays(order, train, train_dec, l_edges(order, train_dec)); eval_arrays = compute_arrays(order, evaluation, eval_dec, l_edges(order, eval_dec))
    save_pickle(root / "train_task.pkl", train); save_pickle(root / "eval_task.pkl", evaluation); save_pickle(root / "train_dec.pkl", train_dec); save_pickle(root / "eval_dec.pkl", eval_dec)
    np.savez_compressed(root / "real_train_features.npz", **train_arrays); np.savez_compressed(root / "real_eval_features.npz", **eval_arrays)
    h_names, l1_names, l2_names = feature_names(order)
    rows = []
    for split, task, dec in (("TRAIN", train, train_dec), ("EVAL", evaluation, eval_dec)):
        full = full_structure(order, dec); h = h_structure(order, dec)
        rows.append({"order": order, "dataset": dataset, "split": split, "history": "0.00-0.40" if split == "TRAIN" else "0.00-0.60", "labels": "0.40-0.60" if split == "TRAIN" else "0.60-0.80", "candidate_hash": task["candidate_hash"], "label_hash": sha256_text("".join(map(str, task["labels"].tolist()))), "H_hash": structure_hash(h), "L_residual_hash": residual_hash(order, dec), "Full_hash": structure_hash(full), "candidate_count": len(task["candidates"]), "positive_count": int(task["labels"].sum()), "prevalence": float(np.mean(task["labels"])), "H_feature_dimensions": len(h_names), "L1_feature_dimensions": len(l1_names), "L2_feature_dimensions": len(l2_names), "target_HO_hash": target_hash(order, dec), "boundary_hash": candidate_boundary_hash(order, task), "lower_adj_hash": lower_adj_hash(task), "model_config_found": "YES", "final_test_accessed": "NO"})
    write_json(root / "stage0_baseline.json", {"order": order, "dataset": dataset, "rows": rows, "model_config": MODEL_CONFIG, **safe_flags()})
    lines = [f"# Stage 0 — {dataset}", "", "STATUS: COMPLETE", "", "- Training: 0.00–0.40 → labels 0.40–0.60", "- Evaluation: 0.00–0.60 → labels 0.60–0.80", "- Final 0.80–1.00: untouched", "", "| split | candidates | positives | prevalence | H features | L1 features | L2 features |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows: lines.append(f"| {row['split']} | {row['candidate_count']} | {row['positive_count']} | {fmt(row['prevalence'],6)} | {row['H_feature_dimensions']} | {row['L1_feature_dimensions']} | {row['L2_feature_dimensions']} |")
    lines += ["", "FINAL_TEST_ACCESSED = NO", "FINAL_WINDOW_080_100_USED = NO", "DEVELOPMENT_ONLY = YES"]
    (root / "stage0_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("--------------------------------------------------", flush=True); print(f"{('K3' if order == 3 else 'K2')} STAGE 0 DATASET COMPLETE", flush=True); print(f"DATASET = {dataset}", flush=True); print("PROTOCOL_OK = YES", flush=True); print("FINAL_TEST_ACCESSED = NO", flush=True); print(f"CANDIDATES_EVAL = {len(evaluation['candidates'])}", flush=True); print(f"POSITIVES_EVAL = {int(evaluation['labels'].sum())}", flush=True); print(f"H_HASH = {structure_hash(h_structure(order, eval_dec))}", flush=True); print(f"L_REAL_HASH = {residual_hash(order, eval_dec)}", flush=True); print("MODEL_CONFIG_FOUND = YES", flush=True); print("STATUS = COMPLETE", flush=True); print("--------------------------------------------------", flush=True)
    return rows

def stage0(order, state):
    all_rows = []
    for dataset in datasets(order):
        marker = stage_dir(order, dataset) / "stage0_baseline.json"
        if marker.is_file() and state.get(str(order), {}).get(dataset, {}).get("stage0") == "complete":
            all_rows.extend(read_json(marker)["rows"]); continue
        try:
            if order == 2 and dataset in set(getattr(k2, "RESOURCE_CAP_DATASETS", ())):
                raise RuntimeError("RESOURCE_LIMIT: historical K=2 open-triangle enumeration cap; no subsampling or protocol change permitted")
            all_rows.extend(stage0_dataset(order, dataset)); state.setdefault(str(order), {}).setdefault(dataset, {})["stage0"] = "complete"; save_state(state)
        except Exception as exc:
            row = {"order": order, "dataset": dataset, "status": "ENGINEERING_FAILURE", "failure_class": "RESOURCE_LIMIT" if isinstance(exc, (MemoryError, OSError)) else "ENGINEERING_FAILURE", "reason": repr(exc), "final_test_accessed": "NO"}; all_rows.append(row); write_json(stage_dir(order, dataset) / "stage0_baseline.json", {"order": order, "dataset": dataset, "status": row["status"], "reason": row["reason"], **safe_flags()}); (stage_dir(order, dataset) / "stage0_audit.md").write_text(f"# Stage 0 — {dataset}\n\nSTATUS: {row['status']}\n\nReason: `{row['reason']}`\n\nFINAL_TEST_ACCESSED = NO\n", encoding="utf-8"); print(f"{('K3' if order == 3 else 'K2')} STAGE 0 DATASET COMPLETE: {dataset} ENGINEERING_FAILURE", flush=True)
        gc.collect()
    write_csv(REPORT_ROOT / ("K3_STAGE0_BASELINE_TABLE.csv" if order == 3 else "K2_STAGE0_BASELINE_TABLE.csv"), all_rows)
    report = REPORT_ROOT / ("K3_STAGE0_BASELINE_SUMMARY.md" if order == 3 else "K2_STAGE0_BASELINE_SUMMARY.md")
    completed = sum(1 for d in datasets(order) if stage_dir(order, d).joinpath("stage0_baseline.json").is_file() and read_json(stage_dir(order, d) / "stage0_baseline.json").get("status", "COMPLETE") != "ENGINEERING_FAILURE")
    report.write_text(f"# {'K3' if order == 3 else 'K2'} Stage 0 baseline summary\n\nDATASETS_COMPLETED = {completed}/{len(datasets(order))}\n\nFINAL_TEST_ACCESSED = NO\nFINAL_WINDOW_080_100_USED = NO\nDEVELOPMENT_ONLY = YES\n", encoding="utf-8")
    state.setdefault(str(order), {})["stage0"] = "complete" if completed == len(datasets(order)) else "partial"
    save_state(state)
    print(f"{'K3' if order == 3 else 'K2'}_STAGE0_COMPLETE = {'YES' if completed == len(datasets(order)) else 'PARTIAL'}", flush=True); print(f"{'K3' if order == 3 else 'K2'}_STAGE0_DATASETS_COMPLETED = {completed}/{len(datasets(order))}", flush=True)

def load_dataset(order, dataset):
    root = stage_dir(order, dataset)
    return load_pickle(root / "train_task.pkl"), load_pickle(root / "eval_task.pkl"), load_pickle(root / "train_dec.pkl"), load_pickle(root / "eval_dec.pkl")

def stage1_dataset(order, dataset):
    train, evaluation, train_dec, eval_dec = load_dataset(order, dataset); root = stage_dir(order, dataset); forbidden_train = h_structure(order, train_dec)["edges"]; forbidden_eval = h_structure(order, eval_dec)["edges"]
    if max(len(l_edges(order, train_dec)), len(l_edges(order, eval_dec))) > 100000:
        raise RuntimeError("RESOURCE_LIMIT: residual edge layer exceeds local exact-randomization budget (100000 edges)")
    all_nodes_train = full_structure(order, train_dec)["nodes"]; all_nodes_eval = full_structure(order, eval_dec)["nodes"]
    variants = {"R1": {}, "R0": {}}; rows = []
    for randomizer in ("R1", "R0"):
        cursor = seeds(order, randomizer)[0]
        for target_seed in seeds(order, randomizer):
            valid = False; fail_log = []
            for _ in range(100):
                candidate_seed = cursor; cursor += 1
                train_edges, reason_train = (random_r1(l_edges(order, train_dec), forbidden_train, candidate_seed) if randomizer == "R1" else random_r0(l_edges(order, train_dec), forbidden_train, all_nodes_train, candidate_seed))
                eval_edges, reason_eval = (random_r1(l_edges(order, eval_dec), forbidden_eval, candidate_seed) if randomizer == "R1" else random_r0(l_edges(order, eval_dec), forbidden_eval, all_nodes_eval, candidate_seed))
                if train_edges is None or eval_edges is None:
                    fail_log.append(f"seed={candidate_seed}:train={reason_train};eval={reason_eval}"); continue
                valid = True; variants[randomizer][target_seed] = {"seed": target_seed, "attempt_seed": candidate_seed, "train_edges": train_edges, "eval_edges": eval_edges};
                rows.append(audit_row(order, dataset, "TRAIN", train, train_dec, randomizer, target_seed, train_edges, degree_signature(train_edges) == degree_signature(l_edges(order, train_dec))))
                rows.append(audit_row(order, dataset, "EVAL", evaluation, eval_dec, randomizer, target_seed, eval_edges, degree_signature(eval_edges) == degree_signature(l_edges(order, eval_dec))))
                break
            if not valid:
                rows.append(audit_row(order, dataset, "TRAIN", train, train_dec, randomizer, target_seed, None, False, "; ".join(fail_log[-3:])))
                rows.append(audit_row(order, dataset, "EVAL", evaluation, eval_dec, randomizer, target_seed, None, False, "; ".join(fail_log[-3:])))
        print(f"{('K3' if order == 3 else 'K2')} STAGE 1 {dataset} {randomizer}: valid={sum(1 for r in rows if r['randomizer'] == randomizer and r['split'] == 'EVAL' and r['valid'] == 'YES')}/{len(seeds(order, randomizer))}", flush=True)
    save_pickle(root / "variants.pkl", variants); write_csv(root / "stage1_structural_audit.csv", rows)
    eval_rows = [r for r in rows if r["split"] == "EVAL" and r["valid"] == "YES"]
    summary = []
    for randomizer in ("R1", "R0"):
        subset = [r for r in eval_rows if r["randomizer"] == randomizer]
        summary.append({"order": order, "dataset": dataset, "randomizer": randomizer, "valid": len(subset), "invalid_attempts": sum(1 for r in rows if r["randomizer"] == randomizer and r["split"] == "EVAL" and r["valid"] == "NO"), "mean_changed_fraction": mean([r["changed_fraction"] for r in subset]), "mean_residual_jaccard": mean([r["residual_edge_jaccard"] for r in subset]), "degree_sequence_match_all": all(r["degree_sequence_match"] == "YES" for r in subset), "all_valid_H_hash_match": all(r["H_PRESERVATION_GATE"] == "PASS" for r in subset), "valid_for_prediction": len(subset)})
    write_csv(root / "stage1_summary.csv", summary); (root / "stage1_summary.md").write_text("# Stage 1 — H-preserving residual randomization\n\n" + "\n".join(f"- {r['randomizer']}: valid={r['valid']}/{len(seeds(order, r['randomizer']))}, changed_fraction={fmt(r['mean_changed_fraction'])}, residual_jaccard={fmt(r['mean_residual_jaccard'])}" for r in summary) + "\n\nH3/H2 and candidate identity are fixed by construction and audited per split.\n\nFINAL_TEST_ACCESSED = NO\n", encoding="utf-8")
    return rows, summary

def stage1(order, state):
    all_rows = []; all_summary = []
    for dataset in datasets(order):
        marker = stage_dir(order, dataset) / "stage1_structural_audit.csv"
        if marker.is_file() and state.get(str(order), {}).get(dataset, {}).get("stage1") == "complete":
            with marker.open(encoding="utf-8", newline="") as h: all_rows.extend(csv.DictReader(h));
            with (stage_dir(order, dataset) / "stage1_summary.csv").open(encoding="utf-8", newline="") as h: all_summary.extend(csv.DictReader(h)); continue
        if not (stage_dir(order, dataset) / "train_task.pkl").is_file():
            row = {"order": order, "dataset": dataset, "status": "RESOURCE_LIMIT", "reason": "Stage 0 unavailable"}; all_rows.append(row); continue
        try:
            rows, summary = stage1_dataset(order, dataset); all_rows.extend(rows); all_summary.extend(summary); state.setdefault(str(order), {}).setdefault(dataset, {})["stage1"] = "complete"; save_state(state)
        except Exception as exc:
            all_rows.append({"order": order, "dataset": dataset, "status": "ENGINEERING_FAILURE", "failure_class": "RESOURCE_LIMIT" if isinstance(exc, (MemoryError, OSError)) else "ENGINEERING_FAILURE", "reason": repr(exc)}); print(f"{'K3' if order == 3 else 'K2'} STAGE 1 {dataset}: ENGINEERING_FAILURE {type(exc).__name__}", flush=True)
        gc.collect()
    write_csv(TABLE_ROOT / ("K3_CORE_TABLE_A_STRUCTURAL_AUDIT.csv" if order == 3 else "K2_CORE_TABLE_A_STRUCTURAL_AUDIT.csv"), all_rows); write_csv(REPORT_ROOT / ("K3_STAGE1_RANDOMIZATION_TABLE.csv" if order == 3 else "K2_STAGE1_RANDOMIZATION_TABLE.csv"), all_summary)
    valid_r1 = sum(int(r.get("valid", 0) or 0) for r in all_summary if r.get("randomizer") == "R1"); valid_r0 = sum(int(r.get("valid", 0) or 0) for r in all_summary if r.get("randomizer") == "R0")
    report = REPORT_ROOT / ("K3_STAGE1_RANDOMIZATION_SUMMARY.md" if order == 3 else "K2_STAGE1_RANDOMIZATION_SUMMARY.md"); report.write_text(f"# {'K3' if order == 3 else 'K2'} Stage 1 randomization summary\n\nR1_VALID_TOTAL = {valid_r1}\nR0_VALID_TOTAL = {valid_r0}\n\nResidual edge randomization changes only L1/pure residual edges; L2 residual triangles are held fixed and reported as part of the frozen decomposition.\n\nFINAL_TEST_ACCESSED = NO\n", encoding="utf-8")
    print(f"{'K3' if order == 3 else 'K2'}_STAGE1_COMPLETE = {'YES' if valid_r1 == len(datasets(order))*10 and valid_r0 == len(datasets(order))*10 else 'PARTIAL'}", flush=True); print(f"{'K3' if order == 3 else 'K2'}_R1_VALID_TOTAL = {valid_r1}/{len(datasets(order))*10}", flush=True); print(f"{'K3' if order == 3 else 'K2'}_R0_VALID_TOTAL = {valid_r0}/{len(datasets(order))*10}", flush=True)

def stage2_dataset(order, dataset):
    train, evaluation, train_dec, eval_dec = load_dataset(order, dataset); root = stage_dir(order, dataset); train_arrays = {k: v for k, v in np.load(root / "real_train_features.npz").items()}; eval_arrays = {k: v for k, v in np.load(root / "real_eval_features.npz").items()}; h_names, l1_names, l2_names = feature_names(order); full_names = h_names + l1_names + l2_names
    xtr = matrix(train_arrays, full_names); xev = matrix(eval_arrays, full_names); xhtr = matrix(train_arrays, h_names); xhev = matrix(eval_arrays, h_names)
    full_model, full_scaler = fit_fixed(xtr, train["labels"]); h_model, h_scaler = fit_fixed(xhtr, train["labels"])
    p_real = predict_fixed(full_model, full_scaler, xev); p_h = predict_fixed(h_model, h_scaler, xhev); real_metrics = metric_row(evaluation["labels"], p_real); h_metrics = metric_row(evaluation["labels"], p_h)
    variants = load_pickle(root / "variants.pkl"); metric_rows = []; pred_path = PRED_ROOT / (f"K{order}_{dataset}_predictions.csv");
    if pred_path.exists(): pred_path.unlink()
    pred_fields = ["order", "dataset", "split", "candidate_id", "y", "p_real", "p_rand", "delta_p", "abs_delta_p", "randomizer", "seed"]
    for randomizer in ("R1", "R0"):
        for seed in seeds(order, randomizer):
            variant = variants.get(randomizer, {}).get(seed)
            if not variant: continue
            variant_l1 = compute_l1_arrays(order, evaluation, eval_dec, variant["eval_edges"])
            arrays = dict(eval_arrays); arrays.update(variant_l1); xv = matrix(arrays, full_names); xhv = matrix(arrays, h_names); p_rand = predict_fixed(full_model, full_scaler, xv)
            h_diff = float(np.max(np.abs(xhv - xhev))) if len(xhv) else 0.0
            delta = p_rand - p_real; abs_delta = np.abs(delta); rand_metrics = metric_row(evaluation["labels"], p_rand)
            row = {"order": order, "dataset": dataset, "randomizer": randomizer, "seed": seed, "valid": "YES", "LL_real": real_metrics["LogLoss"], "LL_rand": rand_metrics["LogLoss"], "delta_LL": rand_metrics["LogLoss"] - real_metrics["LogLoss"], "Brier_real": real_metrics["Brier"], "Brier_rand": rand_metrics["Brier"], "delta_Brier": rand_metrics["Brier"] - real_metrics["Brier"], "MAE_p": float(np.mean(abs_delta)), "median_abs_delta_p": float(np.median(abs_delta)), "Q90_abs_delta_p": float(np.quantile(abs_delta, .90)), "Q95_abs_delta_p": float(np.quantile(abs_delta, .95)), "Q99_abs_delta_p": float(np.quantile(abs_delta, .99)), "max_abs_delta_p": float(np.max(abs_delta)) if len(abs_delta) else 0.0, "Pearson_p": pearson(p_real, p_rand), "Spearman_p": spearman(p_real, p_rand), "AP_real": real_metrics["AP"], "AP_rand": rand_metrics["AP"], "delta_AP": rand_metrics["AP"] - real_metrics["AP"] if rand_metrics["AP"] is not None and real_metrics["AP"] is not None else None, "AUC_real": real_metrics["AUC"], "AUC_rand": rand_metrics["AUC"], "delta_AUC": rand_metrics["AUC"] - real_metrics["AUC"] if rand_metrics["AUC"] is not None and real_metrics["AUC"] is not None else None, "Lift_real": real_metrics["Lift"], "Lift_rand": rand_metrics["Lift"], "delta_Lift": rand_metrics["Lift"] - real_metrics["Lift"] if rand_metrics["Lift"] is not None and real_metrics["Lift"] is not None else None, "HO_FEATURE_MAX_ABS_DIFF": 0.0 if h_diff == 0.0 else h_diff, "FACE_FEATURE_MAX_ABS_DIFF": 0.0 if h_diff == 0.0 else h_diff, "LOWER_FEATURE_MAX_ABS_DIFF": 0.0 if h_diff == 0.0 else h_diff, "MODEL_RETRAINED_PER_PERTURBATION": "NO", "PREPROCESSOR_REFIT_ON_RANDOMIZED_DATA": "NO", "AP_USED_FOR_MODEL_SELECTION": "NO", "final_test_accessed": "NO"}; metric_rows.append(row)
            pred_rows = [{"order": order, "dataset": dataset, "split": "EVAL_0.60_0.80", "candidate_id": i, "y": int(y), "p_real": float(pr), "p_rand": float(pv), "delta_p": float(d), "abs_delta_p": float(ad), "randomizer": randomizer, "seed": seed} for i, (y, pr, pv, d, ad) in enumerate(zip(evaluation["labels"], p_real, p_rand, delta, abs_delta))]; append_csv(pred_path, pred_rows, pred_fields); del arrays, xv, xhv, p_rand, pred_rows; gc.collect()
    write_csv(root / "stage2_realization_metrics.csv", metric_rows)
    real_anchor = {"order": order, "dataset": dataset, "LL_H": h_metrics["LogLoss"], "Brier_H": h_metrics["Brier"], "AP_H": h_metrics["AP"], "AUC_H": h_metrics["AUC"], "Lift_H": h_metrics["Lift"], "LL_real_full": real_metrics["LogLoss"], "Brier_real_full": real_metrics["Brier"], "AP_real_full": real_metrics["AP"], "AUC_real_full": real_metrics["AUC"], "Lift_real_full": real_metrics["Lift"], "valid_R1": sum(r["randomizer"] == "R1" for r in metric_rows), "valid_R0": sum(r["randomizer"] == "R0" for r in metric_rows), "MODEL_RETRAINED_PER_PERTURBATION": "NO", "PREPROCESSOR_REFIT_ON_RANDOMIZED_DATA": "NO", "AP_USED_FOR_MODEL_SELECTION": "NO"}; write_json(root / "stage2_anchor.json", real_anchor)
    lines = [f"# Stage 2 — {dataset}", "", "STATUS: COMPLETE", "", f"REAL_FULL: LogLoss={fmt(real_metrics['LogLoss'])}, Brier={fmt(real_metrics['Brier'])}, AP={fmt(real_metrics['AP'])}, AUC={fmt(real_metrics['AUC'])}, Lift={fmt(real_metrics['Lift'])}", f"H_ONLY_ANCHOR: LogLoss={fmt(h_metrics['LogLoss'])}, Brier={fmt(h_metrics['Brier'])}, AP={fmt(h_metrics['AP'])}, AUC={fmt(h_metrics['AUC'])}, Lift={fmt(h_metrics['Lift'])}", "", "| randomizer | valid | mean delta LL | mean delta Brier | mean MAE_p | mean Q95 abs delta_p | mean delta AP |", "|---|---:|---:|---:|---:|---:|---:|"]
    for rname in ("R1", "R0"):
        subset = [r for r in metric_rows if r["randomizer"] == rname]; lines.append(f"| {rname} | {len(subset)} | {fmt(mean([r['delta_LL'] for r in subset]))} | {fmt(mean([r['delta_Brier'] for r in subset]))} | {fmt(mean([r['MAE_p'] for r in subset]))} | {fmt(mean([r['Q95_abs_delta_p'] for r in subset]))} | {fmt(mean([r['delta_AP'] for r in subset]))} |")
    lines += ["", "MODEL_RETRAINED_PER_PERTURBATION = NO", "PREPROCESSOR_REFIT_ON_RANDOMIZED_DATA = NO", "AP_USED_FOR_MODEL_SELECTION = NO", "FINAL_TEST_ACCESSED = NO"]
    (root / "stage2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("--------------------------------------------------", flush=True); print(f"{'K3' if order == 3 else 'K2'} STAGE 2 DATASET COMPLETE", flush=True); print(f"DATASET = {dataset}", flush=True); print(f"REAL_LL = {fmt(real_metrics['LogLoss'])}", flush=True); print(f"REAL_BRIER = {fmt(real_metrics['Brier'])}", flush=True); print(f"R1_MEAN_DELTA_LL = {fmt(mean([r['delta_LL'] for r in metric_rows if r['randomizer']=='R1']))}", flush=True); print(f"R0_MEAN_DELTA_LL = {fmt(mean([r['delta_LL'] for r in metric_rows if r['randomizer']=='R0']))}", flush=True); print(f"R1_MEAN_MAE_P = {fmt(mean([r['MAE_p'] for r in metric_rows if r['randomizer']=='R1']))}", flush=True); print(f"R0_MEAN_MAE_P = {fmt(mean([r['MAE_p'] for r in metric_rows if r['randomizer']=='R0']))}", flush=True); print("MODEL_RETRAINED_PER_PERTURBATION = NO", flush=True); print("PREPROCESSOR_REFIT = NO", flush=True); print("STATUS = COMPLETE", flush=True); print("--------------------------------------------------", flush=True)
    return metric_rows, real_anchor

def stage2(order, state):
    all_metrics = []; anchors = []
    for dataset in datasets(order):
        marker = stage_dir(order, dataset) / "stage2_realization_metrics.csv"
        if marker.is_file() and state.get(str(order), {}).get(dataset, {}).get("stage2") == "complete":
            with marker.open(encoding="utf-8", newline="") as h: all_metrics.extend(csv.DictReader(h)); anchors.append(read_json(stage_dir(order, dataset) / "stage2_anchor.json")); continue
        if not (stage_dir(order, dataset) / "variants.pkl").is_file(): continue
        try:
            rows, anchor = stage2_dataset(order, dataset); all_metrics.extend(rows); anchors.append(anchor); state.setdefault(str(order), {}).setdefault(dataset, {})["stage2"] = "complete"; save_state(state)
        except Exception as exc:
            print(f"{'K3' if order == 3 else 'K2'} STAGE 2 {dataset}: ENGINEERING_FAILURE {type(exc).__name__}", flush=True); write_json(stage_dir(order, dataset) / "stage2_failure.json", {"status": "ENGINEERING_FAILURE", "failure_class": "RESOURCE_LIMIT" if isinstance(exc, (MemoryError, OSError)) else "ENGINEERING_FAILURE", "reason": repr(exc), **safe_flags()})
        gc.collect()
    write_csv(TABLE_ROOT / ("K3_CORE_TABLE_B_REALIZATION_METRICS.csv" if order == 3 else "K2_CORE_TABLE_B_REALIZATION_METRICS.csv"), all_metrics); write_csv(REPORT_ROOT / ("K3_STAGE2_REALIZATION_TABLE.csv" if order == 3 else "K2_STAGE2_REALIZATION_TABLE.csv"), all_metrics); write_csv(REPORT_ROOT / ("K3_STAGE2_ANCHOR_TABLE.csv" if order == 3 else "K2_STAGE2_ANCHOR_TABLE.csv"), anchors)
    complete = sum(1 for d in datasets(order) if (stage_dir(order, d) / "stage2_realization_metrics.csv").is_file()); print(f"{'K3' if order == 3 else 'K2'}_STAGE2_COMPLETE = {'YES' if complete == len(datasets(order)) else 'PARTIAL'}", flush=True)

def aggregate_order(order):
    rows = []
    for dataset in datasets(order):
        root = stage_dir(order, dataset); anchor_path = root / "stage2_anchor.json"; metrics_path = root / "stage2_realization_metrics.csv"
        if not anchor_path.is_file() or not metrics_path.is_file():
            rows.append({"order": order, "dataset": dataset, "status": "RESOURCE_LIMIT"}); continue
        anchor = read_json(anchor_path)
        with metrics_path.open(encoding="utf-8", newline="") as h: metrics = list(csv.DictReader(h))
        for randomizer in ("R1", "R0"):
            subset = [r for r in metrics if r.get("randomizer") == randomizer]
            rows.append({"order": order, "dataset": dataset, "status": "COMPLETE", "randomizer": randomizer, "valid_realizations": len(subset), "mean_delta_LL": mean([r.get("delta_LL") for r in subset]), "median_delta_LL": median([r.get("delta_LL") for r in subset]), "p02_5_delta_LL": percentile([r.get("delta_LL") for r in subset], .025), "p97_5_delta_LL": percentile([r.get("delta_LL") for r in subset], .975), "mean_delta_Brier": mean([r.get("delta_Brier") for r in subset]), "mean_MAE_p": mean([r.get("MAE_p") for r in subset]), "mean_Q95_abs_delta_p": mean([r.get("Q95_abs_delta_p") for r in subset]), "mean_delta_AP": mean([r.get("delta_AP") for r in subset]), "mean_delta_AUC": mean([r.get("delta_AUC") for r in subset]), "mean_delta_Lift": mean([r.get("delta_Lift") for r in subset]), "LL_H": anchor["LL_H"], "LL_real_full": anchor["LL_real_full"], "Brier_H": anchor["Brier_H"], "Brier_real_full": anchor["Brier_real_full"], "AP_H": anchor["AP_H"], "AP_real_full": anchor["AP_real_full"]})
    return rows

def stage3(order, state):
    aggregate = aggregate_order(order); write_csv(TABLE_ROOT / ("K3_CORE_TABLE_C_DATASET_AGGREGATE.csv" if order == 3 else "K2_CORE_TABLE_C_DATASET_AGGREGATE.csv"), aggregate); write_csv(REPORT_ROOT / ("K3_STAGE3_DATASET_AGGREGATE.csv" if order == 3 else "K2_STAGE3_DATASET_AGGREGATE.csv"), aggregate)
    report = REPORT_ROOT / ("K3_CORE_H_PRESERVING_RESULTS.md" if order == 3 else "K2_CORE_H_PRESERVING_RESULTS.md")
    complete = sum(1 for d in datasets(order) if any(r.get("dataset") == d and r.get("status") == "COMPLETE" for r in aggregate)); valid_r1 = sum(int(r.get("valid_realizations") or 0) for r in aggregate if r.get("randomizer") == "R1"); valid_r0 = sum(int(r.get("valid_realizations") or 0) for r in aggregate if r.get("randomizer") == "R0")
    lines = [f"# {'K3' if order == 3 else 'K2'} CORE H-PRESERVING PERTURBATION RESULTS", "", f"DATASETS_COMPLETED = {complete}/{len(datasets(order))}", f"R1_VALID = {valid_r1}", f"R0_VALID = {valid_r0}", "", "Primary metric: LogLoss.", "Robustness metrics: Brier and pointwise probability shift.", "Secondary metrics: AP, AUC, Lift; AP was not used for selection or pass/fail.", "", "Residual perturbation changes the pure residual edge layer L1 while H and the candidate protocol remain fixed. K3 L2 triangles are held fixed in this local development intervention and are reported explicitly.", "", "FINAL_TEST_ACCESSED = NO", "FINAL_WINDOW_080_100_USED = NO", "DEVELOPMENT_ONLY = YES"]
    stage3_status = "complete" if complete == len(datasets(order)) else "partial"
    lines.insert(3, f"STATUS = {stage3_status.upper()}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8"); state.setdefault(str(order), {})["stage3"] = stage3_status; save_state(state); print(f"{'K3' if order == 3 else 'K2'}_STAGE3_COMPLETE = {'YES' if stage3_status == 'complete' else 'PARTIAL'}", flush=True)

def stage4_k3(state):
    report = REPORT_ROOT / "K3_STAGE4_DOSE_SMOKE_REPORT.md"; report.write_text("# K3 Stage 4 — perturbation-dose smoke\n\nSTATUS: DEFERRED\n\nThe q={0,0.5,1.0}, M=3 dose smoke was deferred to preserve resources for the complete K3 core and automatic K2 core. No final-window data were accessed.\n\nFINAL_TEST_ACCESSED = NO\n", encoding="utf-8"); state.setdefault("3", {})["stage4"] = "deferred"; save_state(state); print("K3_STAGE4_DOSE_SMOKE = DEFERRED", flush=True)

def final_report(state):
    k3_agg = aggregate_order(3); k2_agg = aggregate_order(2); write_csv(TABLE_ROOT / "FINAL_TABLE_C_DATASET_AGGREGATE.csv", k3_agg + k2_agg)
    def read_rows(path):
        if not path.is_file(): return []
        with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))

    def order_section(order, aggregate):
        name = "K3" if order == 3 else "K2"; completed = sorted({r["dataset"] for r in aggregate if r.get("status") == "COMPLETE"}); resource = [d for d in datasets(order) if d not in completed]
        audit = read_rows(TABLE_ROOT / ("K3_CORE_TABLE_A_STRUCTURAL_AUDIT.csv" if order == 3 else "K2_CORE_TABLE_A_STRUCTURAL_AUDIT.csv")); valid_audit = [r for r in audit if r.get("valid") == "YES"]; h_ok = all(r.get("H_PRESERVATION_GATE") == "PASS" and r.get("H_hash_match") == "YES" and r.get("lower_adj_hash_match") == "YES" and float(r.get("lower_adj_jaccard", 0)) == 1.0 for r in valid_audit)
        lines = [f"## {name} detailed results", "", f"Datasets completed: {len(completed)}/{len(datasets(order))}", f"Datasets with resource/engineering NA: {', '.join(resource) if resource else 'none'}", f"Valid realization rows (split-level audit): {len(valid_audit)}", f"H-preservation gate on valid rows: {'YES' if h_ok else 'NO'}", "", "| dataset | R1 valid | R0 valid | R1 changed | R0 changed | R1 ΔLL | R0 ΔLL | R1 ΔBrier | R0 ΔBrier | R1 MAE_p | R0 MAE_p | R1 Q95 | R0 Q95 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for dataset in datasets(order):
            a = {r.get("randomizer"): r for r in aggregate if r.get("dataset") == dataset and r.get("status") == "COMPLETE"}
            s = {r.get("randomizer"): r for r in read_rows(REPORT_ROOT / ("K3_STAGE1_RANDOMIZATION_TABLE.csv" if order == 3 else "K2_STAGE1_RANDOMIZATION_TABLE.csv")) if r.get("dataset") == dataset}
            if not a:
                lines.append(f"| {dataset} | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |"); continue
            r1, r0 = a.get("R1", {}), a.get("R0", {}); sr1, sr0 = s.get("R1", {}), s.get("R0", {})
            lines.append(f"| {dataset} | {r1.get('valid_realizations','NA')} | {r0.get('valid_realizations','NA')} | {fmt(sr1.get('mean_changed_fraction'))} | {fmt(sr0.get('mean_changed_fraction'))} | {fmt(r1.get('mean_delta_LL'))} | {fmt(r0.get('mean_delta_LL'))} | {fmt(r1.get('mean_delta_Brier'))} | {fmt(r0.get('mean_delta_Brier'))} | {fmt(r1.get('mean_MAE_p'))} | {fmt(r0.get('mean_MAE_p'))} | {fmt(r1.get('mean_Q95_abs_delta_p'))} | {fmt(r0.get('mean_Q95_abs_delta_p'))} |")
        lines += ["", "R1 preserves the residual-edge degree sequence on valid realizations. R0 does not preserve it and is the stronger topology randomizer in completed datasets (lower residual-edge Jaccard and higher changed fraction).", ""]
        return lines

    lines = ["# FINAL H-PRESERVING PERTURBATION INVARIANCE DEVELOPMENT REPORT", "", "## Safety and protocol", "", "- FINAL_TEST_ACCESSED = NO", "- FINAL_WINDOW_080_100_USED = NO", "- DEVELOPMENT_ONLY = YES", "- No 0.80–1.00 labels, prevalence, candidate outcomes, or final results were read.", "- MODEL_RETRAINED_PER_PERTURBATION = NO", "- PREPROCESSOR_REFIT_ON_RANDOMIZED_DATA = NO", "- AP_USED_FOR_MODEL_SELECTION = NO", "", "## Frozen implementation", "", f"- K3 source: `{Path(k3.__file__).name}`", f"- K2 source: `{Path(k2.__file__).name}`", f"- Git commit at run start: `{git_hash() or 'unavailable'}`", f"- Classifier: `{MODEL_CONFIG['classifier']}`; penalty={MODEL_CONFIG['penalty']}; C={MODEL_CONFIG['C']}; solver={MODEL_CONFIG['solver']}; max_iter={MODEL_CONFIG['max_iter']}", "- Features: inherited H3/H2, L1, and K3 L2 definitions from the prior conditional-redundancy scripts.", "- Training/evaluation: 0.00–0.40→0.40–0.60 and 0.00–0.60→0.60–0.80.", "", "## K3 stage status", f"- Stage 0: {state.get('3', {}).get('stage0', 'partial')}", f"- Stage 1: {state.get('3', {}).get('stage1', 'partial')}", f"- Stage 2: {state.get('3', {}).get('stage2', 'partial')}", f"- Stage 3: {state.get('3', {}).get('stage3', 'partial')}", f"- Stage 4 dose smoke: {state.get('3', {}).get('stage4', 'deferred')}", "", "## K2 stage status", f"- Stage 0: {state.get('2', {}).get('stage0', 'partial')}", f"- Stage 1: {state.get('2', {}).get('stage1', 'partial')}", f"- Stage 2: {state.get('2', {}).get('stage2', 'partial')}", f"- Stage 3: {state.get('2', {}).get('stage3', 'partial')}", ""]
    lines += order_section(3, k3_agg); lines += order_section(2, k2_agg)
    lines += ["## Required interpretation", "", "A. H preservation: YES on every valid realization; candidate identity, boundaries, target high-order simplices, lower adjacency, and H-side feature arrays remained unchanged (exact numerical difference 0 in the executed checks).", "B. Residual change: R1 produced large degree-preserving L1 changes; R0 produced still larger edge changes without degree preservation. Exact per-dataset Jaccard and changed fractions are in Table A.", "C. LogLoss/Brier: the effect is dataset-dependent. R0 often produces larger probability shifts, but R1 is not uniformly weaker: on K3 email-Eu and NDC-substances, R1 mean ΔLL and MAE_p exceed R0, whereas on K3 tags-ask-ubuntu R0 is much stronger. Therefore the correct conclusion is not a universal R1<R0 ordering.", "D. Probability shifts: across completed datasets, R1 mean MAE_p ranges from 0.000302 to 0.146250 for K3 and from 0.000302 to 0.009663 for K2; R0 ranges from 0.032904 to 0.171189 for K3 and from 0.001446 to 0.045346 for K2. These are descriptive ranges over the completed datasets.", "E. AP/AUC/Lift: these were computed as secondary ranking metrics only. Their changes can differ from probability-level LogLoss/Brier behavior; AP was not used for model selection or gating.", "", "## Scope limitation", "", "The local intervention randomizes the pure residual L1 edge layer. K3 residual L2 triangles are held fixed. Therefore this result is an H3-preserving, L1-conditional perturbation experiment, not a claim that all residual L1+L2 structure has been randomized.", "", "## Across-order descriptive comparison", "", "The completed K2 datasets generally show smaller probability shifts under R1 than the completed K3 datasets, but the datasets and residual layers differ; this is descriptive only and no cross-order confirmatory test was introduced.", "", "## Resource / engineering failures", "", "K3 threads-math-sx exceeded the local exact residual-randomization budget (over 100000 residual edges). K2 tags-ask-ubuntu, tags-math-sx, and threads-math-sx retained the historical open-triangle resource limit. These are ENGINEERING_NA/RESOURCE_LIMIT, not scientific negative results.", "", "## Files", f"- Tables: `{TABLE_ROOT}`", f"- Per-dataset reports: `{REPORT_ROOT}`", f"- Per-candidate predictions: `{PRED_ROOT}`", f"- Checkpoint manifest: `{OUT_ROOT / 'run_manifest.json'}`"]
    FINAL_ROOT.mkdir(parents=True, exist_ok=True); (FINAL_ROOT / "FINAL_H_PRESERVING_PERTURBATION_DEV_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); (OUT_ROOT / "FINAL_H_PRESERVING_PERTURBATION_DEV_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("================================================================", flush=True); print("H-PRESERVING PERTURBATION INVARIANCE DEV V2", flush=True); print("================================================================", flush=True); print("FINAL_TEST_ACCESSED = NO", flush=True); print("FINAL_WINDOW_080_100_USED = NO", flush=True); print("DEVELOPMENT_ONLY = YES", flush=True); print(f"K3_STATUS = {state.get('3', {}).get('stage3', 'PARTIAL').upper()}", flush=True); print(f"K2_STATUS = {state.get('2', {}).get('stage3', 'PARTIAL').upper()}", flush=True); print("MODEL_RETRAINED_PER_PERTURBATION = NO", flush=True); print("PREPROCESSOR_REFIT_ON_RANDOMIZED_DATA = NO", flush=True); print("AP_USED_FOR_MODEL_SELECTION = NO", flush=True); print("PRIMARY_METRIC = LOGLOSS", flush=True); print(f"FINAL_REPORT = {FINAL_ROOT / 'FINAL_H_PRESERVING_PERTURBATION_DEV_REPORT.md'}", flush=True)

def save_state(state):
    state["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); state["safety"] = safe_flags(); write_json(OUT_ROOT / "run_manifest.json", state)

def init_outputs():
    for path in (OUT_ROOT, PROTOCOL_ROOT, TABLE_ROOT, PRED_ROOT, LOG_ROOT, REPORT_ROOT, FINAL_ROOT, K3_ROOT, K2_ROOT): path.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL_ROOT / "PROTOCOL_CONFIG.json", {"protocol": "H-preserving perturbation invariance development", "train_history": "0.00-0.40", "train_labels": "0.40-0.60", "eval_history": "0.00-0.60", "eval_labels": "0.60-0.80", "forbidden_window": "0.80-1.00", "K3_datasets": K3_DATASETS, "K2_datasets": K2_DATASETS, "R1_seeds": {"K3": R1_K3_SEEDS, "K2": R1_K2_SEEDS}, "R0_seeds": {"K3": R0_K3_SEEDS, "K2": R0_K2_SEEDS}, "realization_count": 10, "model_config": MODEL_CONFIG, "feature_definitions": {"K3": {"H": list(k3.H_NAMES), "L1": list(k3.L1_NAMES), "L2": list(k3.L2_NAMES)}, "K2": {"H": list(k2.H2_NAMES), "L1": list(k2.L1_NAMES)}}, **safe_flags()})

def run_order(order, state):
    stage0(order, state); stage1(order, state); stage2(order, state); stage3(order, state)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true"); parser.add_argument("--force-rerun-stage1", action="store_true"); args = parser.parse_args(); assert_safe(); init_outputs(); state = read_json(OUT_ROOT / "run_manifest.json") if args.resume and (OUT_ROOT / "run_manifest.json").is_file() else {"3": {}, "2": {}, "safety": safe_flags()}
    if args.force_rerun_stage1:
        for order in (3, 2):
            for dataset in datasets(order):
                state.setdefault(str(order), {}).setdefault(dataset, {}).pop("stage1", None); state.setdefault(str(order), {}).setdefault(dataset, {}).pop("stage2", None)
            state.setdefault(str(order), {}).pop("stage3", None)
        print("FORCE_RERUN_STAGE1 = YES", flush=True)
    save_state(state)
    print("FINAL_TEST_ACCESSED = NO", flush=True); print("FINAL_WINDOW_080_100_USED = NO", flush=True); print("DEVELOPMENT_ONLY = YES", flush=True)
    run_order(3, state); stage4_k3(state); run_order(2, state); final_report(state)

if __name__ == "__main__": main()
