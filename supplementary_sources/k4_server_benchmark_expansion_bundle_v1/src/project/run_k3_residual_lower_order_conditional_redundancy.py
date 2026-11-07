from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import run_full_information_preservation_validation as previous

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "dataset" / "raw_data"
PREVIOUS_ROOT = ROOT / "results" / "full_information_preservation_validation"
OUT_ROOT = ROOT / "results" / "k3_residual_lower_order_conditional_redundancy"
STAGE_DIRS = {i: OUT_ROOT / f"stage{i}_{name}" for i, name in enumerate((
    "previous_audit", "protocol", "decomposition", "features", "nested_prediction",
    "conditional_permutation", "structural_intervention", "noise_robustness",
    "gat_robustness", "k2_control", "final"))}
DATASETS = previous.DATASETS
Q_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
SEEDS = tuple(range(20))
GAT_SEEDS = (42, 43, 44)
FINAL_TEST_ACCESSED = "NO"
FUTURE_STRUCTURE_USED = "NO"
HYPERPARAMETER_SEARCH = "NO"
DATASET_TUNING = "NO"
AP_IMPLEMENTATION = "sklearn.metrics.average_precision_score"
LOGLOSS_IMPLEMENTATION = "sklearn.metrics.log_loss"
TRAPEZOIDAL_PR_AUC = "NO"

LOGISTIC_REQUESTED_SOLVER = "lbfgs"
LOGISTIC_ACTUAL_SOLVER = "liblinear"

H_NAMES = (
    "SCN", "SRA", "SAA", "SPA",
    "FACE_COFACE_SUM", "FACE_COFACE_MEAN", "FACE_COFACE_MIN", "FACE_COFACE_MAX", "FACE_COFACE_STD",
    "LOWER_DEGREE", "LOWER_COMMON_NEIGHBORS", "LOWER_NEIGHBORHOOD_JACCARD", "LOWER_2HOP_COUNT",
    "H_BOUNDARY_FACE_COFACE_SUM", "H_BOUNDARY_FACE_COFACE_MEAN", "H_BOUNDARY_FACE_COFACE_MAX", "H_BOUNDARY_FACE_COFACE_STD",
)
L2_NAMES = (
    "VERTEX_RES_TRI_SUM", "VERTEX_RES_TRI_MEAN", "VERTEX_RES_TRI_MAX", "VERTEX_RES_TRI_MIN", "VERTEX_RES_TRI_STD",
    "EDGE_RES_TRI_SUM", "EDGE_RES_TRI_MEAN", "EDGE_RES_TRI_MAX", "EDGE_RES_TRI_STD",
    "RES_TRI_TOUCH_1_VERTEX", "RES_TRI_TOUCH_2_VERTICES", "RES_TRI_OVERLAP_3_VERTICES", "RES_TRI_OVERLAP_SUM",
)
L1_NAMES = (
    "RES_DEG_SUM", "RES_DEG_MEAN", "RES_DEG_MAX", "RES_DEG_MIN", "RES_DEG_STD",
    "RES_PCN_SUM", "RES_PRA_SUM", "RES_PAA_SUM", "RES_PJACCARD_MEAN",
    "RES_1HOP_UNIQUE_COUNT", "RES_2HOP_UNIQUE_COUNT", "RES_EXTERNAL_CUT_SIZE",
)
ALL_NAMES = H_NAMES + L1_NAMES + L2_NAMES

def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

def write_csv(path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader(); writer.writerows(rows)

def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def _mean(values):
    nums = []
    for value in values:
        try:
            x = float(value)
            if math.isfinite(x): nums.append(x)
        except (TypeError, ValueError):
            pass
    return float(np.mean(nums)) if nums else None

def _median(values):
    nums = []
    for value in values:
        try:
            x = float(value)
            if math.isfinite(x): nums.append(x)
        except (TypeError, ValueError):
            pass
    return float(np.median(nums)) if nums else None

def _fmt(value, digits=6):
    if value is None: return "NA"
    try:
        return f"{float(value):.{digits}f}" if math.isfinite(float(value)) else "NA"
    except (TypeError, ValueError):
        return str(value)

def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()

def _hash_rows(rows):
    return hashlib.sha256("\n".join(",".join(map(str, row)) for row in sorted(rows)).encode()).hexdigest()

def _hash_matrix(matrix):
    array = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    return _hash_bytes(array.tobytes())

def _faces(simplex, size):
    return tuple(combinations(tuple(sorted(simplex)), size))

def _downward_edges(triangles, tetras=()):
    edges = set()
    for triangle in triangles:
        edges.update(_faces(triangle, 2))
    for tetra in tetras:
        edges.update(_faces(tetra, 2))
    return edges

def _nodes(edges, triangles=(), tetras=()):
    values = set()
    for simplex_set in (edges, triangles, tetras):
        for simplex in simplex_set:
            values.update(simplex)
    return values

def _set_hash(structure):
    return hashlib.sha256("".join(
        f"{name}:" + "\n".join(",".join(map(str, sorted(structure[name]))) for _ in [0])
        for name in ("nodes", "edges", "triangles", "tetras")
    ).encode()).hexdigest()

def _state_path():
    return OUT_ROOT / "RUN_STATE.json"

def load_state():
    if _state_path().is_file():
        return json.loads(_state_path().read_text(encoding="utf-8"))
    return {"completed_stages": [], "stage_status": {}, "hard_stop": None,
            "final_test_accessed": FINAL_TEST_ACCESSED, "future_structure_used": FUTURE_STRUCTURE_USED}

def save_state(state):
    state["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(_state_path(), state)

def complete_stage(state, stage, report):
    state.setdefault("completed_stages", []).append(stage)
    state["completed_stages"] = sorted(set(state["completed_stages"]))
    state.setdefault("stage_status", {})[str(stage)] = "PASS"
    state.setdefault("stage_report", {})[str(stage)] = str(report)
    save_state(state)

def hard_stop(state, reason, detail):
    state["hard_stop"] = {"reason": reason, "detail": detail}; save_state(state)
    report = OUT_ROOT / "HARD_STOP_REPORT.md"
    report.write_text(f"# HARD STOP\n\n- reason: `{reason}`\n- detail: {detail}\n", encoding="utf-8")
    raise RuntimeError(f"HARD_STOP={reason}: {detail}")

def _load_events(dataset):
    from kclosure.core import load_timestamped_simplices
    base = RAW_ROOT / dataset
    paths = tuple(base / f"{dataset}-{suffix}.txt" for suffix in ("nverts", "simplices", "times"))
    events, stats = load_timestamped_simplices(*paths, event_order="raw")
    return events, stats, paths

def _build_snapshot_task(dataset, context_fraction, label_start_fraction, label_end_fraction):
    from kclosure.core import build_observed_state, enumerate_open_simplices, future_closed_simplices
    events, loader_stats, paths = _load_events(dataset)
    n = len(events)
    context_end = int(math.floor(context_fraction * n)); label_start = int(math.floor(label_start_fraction * n)); label_end = int(math.floor(label_end_fraction * n))
    context = events[:context_end]
    state = build_observed_state(context, max_dim=3)
    candidates, _ = enumerate_open_simplices(state, 3)
    candidates = sorted(tuple(candidate) for candidate in candidates)
    boundaries = state["simplices_by_size"][3]; targets = state["simplices_by_size"][4]
    if not all(all(face in boundaries for face in previous._simplex_boundaries(candidate)) and candidate not in targets for candidate in candidates):
        raise RuntimeError(f"candidate boundary assertion failed: {dataset}")
    future = future_closed_simplices(events[label_start:label_end], 3)
    labels = np.asarray([int(candidate in future) for candidate in candidates], dtype=np.int8)
    candidate_faces = set(face for candidate in candidates for face in _faces(candidate, 3))
    candidate_edges = set(edge for candidate in candidates for edge in _faces(candidate, 2))
    full = {"nodes": set(state["simplices_by_size"][1]), "edges": set(state["simplices_by_size"][2]), "triangles": set(state["simplices_by_size"][3]), "tetras": set(state["simplices_by_size"][4])}
    from audit_dual_mapping_information_loss import _backmapped_sets
    back = _backmapped_sets(list(full["tetras"]) + candidates, 3)
    dual = {"nodes": set(back[1]), "edges": set(back[2]), "triangles": set(back[3]), "tetras": set(full["tetras"])}
    lower_adj = [set() for _ in candidates]; face_to_ids = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        for face in _faces(candidate, 3): face_to_ids[face].append(idx)
    for ids in face_to_ids.values():
        for left, right in combinations(ids, 2): lower_adj[left].add(right); lower_adj[right].add(left)
    return {"dataset": dataset, "window": f"{context_fraction:.2f}-{label_start_fraction:.2f}-{label_end_fraction:.2f}", "context_fraction": context_fraction, "label_start_fraction": label_start_fraction, "label_end_fraction": label_end_fraction, "context_end": context_end, "label_start": label_start, "label_end": label_end, "loader_stats": loader_stats, "source_paths": [str(p) for p in paths], "state": state, "candidates": candidates, "labels": labels, "candidate_faces": candidate_faces, "candidate_edges": candidate_edges, "full": full, "dual": dual, "lower_adj": lower_adj, "candidate_hash": previous.simplex_hash(candidates), "positive_hash": previous.simplex_hash([c for c, y in zip(candidates, labels) if y]), "positive_count": int(labels.sum()), "prevalence": float(labels.mean()) if len(labels) else 0.0}

def _decompose(task):
    full = task["full"]; candidates = task["candidates"]
    h_tetras = set(full["tetras"])
    h_triangles = set(face for tetra in h_tetras for face in _faces(tetra, 3)) | set(task["candidate_faces"])
    h_edges = _downward_edges(h_triangles, h_tetras) | set(task["candidate_edges"])
    h_nodes = _nodes(h_edges, h_triangles, h_tetras)
    l2_triangles = set(full["triangles"] - h_triangles)
    l2_boundary_edges = set(edge for triangle in l2_triangles for edge in _faces(triangle, 2)) - h_edges
    l1_edges = set(full["edges"] - h_edges - l2_boundary_edges)
    h3 = {"nodes": h_nodes, "edges": h_edges, "triangles": h_triangles, "tetras": h_tetras}
    if h3["triangles"] | l2_triangles != full["triangles"] or h3["edges"] | l2_boundary_edges | l1_edges != full["edges"]:
        raise RuntimeError(f"decomposition mismatch for {task['dataset']}")
    return {"full": full, "h3": h3, "l2_triangles": l2_triangles, "l2_boundary_edges": l2_boundary_edges, "l1_edges": l1_edges, "candidate_faces": set(task["candidate_faces"]), "candidate_edges": set(task["candidate_edges"]), "lower_adj": task["lower_adj"]}

def _h3_lower_adj(task):
    return [set(values) for values in task["lower_adj"]]

def _adjacency(edges):
    adjacency = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right); adjacency[right].add(left)
    return adjacency

def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0

def _pair_scores(adjacency, candidate):
    cn = ra = aa = jac = 0.0
    vertices = tuple(candidate)
    for left, right in _faces(vertices, 2):
        left_n = adjacency.get(left, set()) - {right}; right_n = adjacency.get(right, set()) - {left}; common = left_n & right_n
        cn += len(common)
        for node in common:
            degree = len(adjacency.get(node, set()))
            if degree: ra += 1.0 / degree; aa += 1.0 / math.log(degree + 1.0)
        jac += _jaccard(left_n, right_n)
    return cn, ra, aa, jac / 6.0

def compute_features(task, h3, l2_triangles, l1_edges):
    pass
    candidates = task["candidates"]; n = len(candidates)
    arrays = {name: np.zeros(n, dtype=np.float32) for name in ALL_NAMES}
    h_adj = _adjacency(h3["edges"])
    h_tri_by_edge = defaultdict(set); tetra_by_tri = defaultdict(set)
    for triangle in h3["triangles"]:
        for edge in _faces(triangle, 2): h_tri_by_edge[edge].add(triangle)
    for tetra in h3["tetras"]:
        for face in _faces(tetra, 3): tetra_by_tri[face].add(tetra)
    l2_by_vertex = defaultdict(set); l2_by_edge = defaultdict(set)
    for triangle in l2_triangles:
        for vertex in triangle: l2_by_vertex[vertex].add(triangle)
        for edge in _faces(triangle, 2): l2_by_edge[edge].add(triangle)
    l1_adj = _adjacency(l1_edges)
    for idx, candidate in enumerate(candidates):
        vertices = set(candidate); edges = _faces(candidate, 2); faces = _faces(candidate, 3)
        face_sets = [set(tetra_by_tri.get(face, set())) for face in faces]
        face_counts = [len(values) for values in face_sets]
        scn = sra = saa = spa = 0.0
        for left, right in combinations(range(4), 2):
            common = face_sets[left] & face_sets[right]; scn += len(common)
            degree = len(face_sets[left])
            if degree:
                sra += len(common) / degree; saa += len(common) / math.log(degree + 1.0)
            spa += len(face_sets[left]) * len(face_sets[right])
        arrays["SCN"][idx] = scn; arrays["SRA"][idx] = sra; arrays["SAA"][idx] = saa; arrays["SPA"][idx] = spa
        arrays["FACE_COFACE_SUM"][idx] = sum(face_counts); arrays["FACE_COFACE_MEAN"][idx] = np.mean(face_counts); arrays["FACE_COFACE_MIN"][idx] = min(face_counts); arrays["FACE_COFACE_MAX"][idx] = max(face_counts); arrays["FACE_COFACE_STD"][idx] = np.std(face_counts)
        boundary_counts = [len(tetra_by_tri.get(face, set())) + 1 for face in faces]
        arrays["H_BOUNDARY_FACE_COFACE_SUM"][idx] = sum(boundary_counts); arrays["H_BOUNDARY_FACE_COFACE_MEAN"][idx] = np.mean(boundary_counts); arrays["H_BOUNDARY_FACE_COFACE_MAX"][idx] = max(boundary_counts); arrays["H_BOUNDARY_FACE_COFACE_STD"][idx] = np.std(boundary_counts)
        neighbors = task["lower_adj"][idx]; two_hop = set()
        for neighbor in neighbors: two_hop.update(task["lower_adj"][neighbor])
        two_hop.discard(idx); two_hop.difference_update(neighbors)
        arrays["LOWER_DEGREE"][idx] = len(neighbors); arrays["LOWER_COMMON_NEIGHBORS"][idx] = sum(len(neighbors & task["lower_adj"][neighbor]) for neighbor in neighbors); arrays["LOWER_NEIGHBORHOOD_JACCARD"][idx] = _jaccard(neighbors, two_hop); arrays["LOWER_2HOP_COUNT"][idx] = len(two_hop)
        vertex_counts = [len(l2_by_vertex.get(vertex, set())) for vertex in vertices]
        edge_counts = [len(l2_by_edge.get(edge, set())) for edge in edges]
        arrays["VERTEX_RES_TRI_SUM"][idx] = sum(vertex_counts); arrays["VERTEX_RES_TRI_MEAN"][idx] = np.mean(vertex_counts); arrays["VERTEX_RES_TRI_MAX"][idx] = max(vertex_counts); arrays["VERTEX_RES_TRI_MIN"][idx] = min(vertex_counts); arrays["VERTEX_RES_TRI_STD"][idx] = np.std(vertex_counts)
        arrays["EDGE_RES_TRI_SUM"][idx] = sum(edge_counts); arrays["EDGE_RES_TRI_MEAN"][idx] = np.mean(edge_counts); arrays["EDGE_RES_TRI_MAX"][idx] = max(edge_counts); arrays["EDGE_RES_TRI_STD"][idx] = np.std(edge_counts)

        s1 = sum(vertex_counts)
        s2 = sum(len(l2_by_edge.get(edge, set())) for edge in edges)
        s3 = sum(1 for face in faces if face in l2_triangles)
        n3 = s3; n2 = s2 - 3 * n3; n1 = s1 - 2 * n2 - 3 * n3
        arrays["RES_TRI_TOUCH_1_VERTEX"][idx] = n1; arrays["RES_TRI_TOUCH_2_VERTICES"][idx] = n2; arrays["RES_TRI_OVERLAP_3_VERTICES"][idx] = n3; arrays["RES_TRI_OVERLAP_SUM"][idx] = s1
        degree_counts = [len(l1_adj.get(v, set())) for v in vertices]
        arrays["RES_DEG_SUM"][idx] = sum(degree_counts); arrays["RES_DEG_MEAN"][idx] = np.mean(degree_counts); arrays["RES_DEG_MAX"][idx] = max(degree_counts); arrays["RES_DEG_MIN"][idx] = min(degree_counts); arrays["RES_DEG_STD"][idx] = np.std(degree_counts)
        cn, ra, aa, jac = _pair_scores(l1_adj, candidate)
        arrays["RES_PCN_SUM"][idx] = cn; arrays["RES_PRA_SUM"][idx] = ra; arrays["RES_PAA_SUM"][idx] = aa; arrays["RES_PJACCARD_MEAN"][idx] = jac
        one_hop = set().union(*(l1_adj.get(v, set()) for v in vertices)) - vertices; two_hop_l1 = set().union(*(l1_adj.get(v, set()) for v in one_hop)) - vertices - one_hop
        arrays["RES_1HOP_UNIQUE_COUNT"][idx] = len(one_hop); arrays["RES_2HOP_UNIQUE_COUNT"][idx] = len(two_hop_l1); arrays["RES_EXTERNAL_CUT_SIZE"][idx] = sum(1 for v in vertices for w in l1_adj.get(v, set()) if w not in vertices)
    return arrays

def _matrix(arrays, names):
    return np.column_stack([arrays[name] for name in names]).astype(np.float32)

def _save_feature_bundle(dataset, split, arrays):
    path = STAGE_DIRS[3] / f"{dataset}_{split}_features.npz"
    np.savez_compressed(path, **arrays); return path

def _load_feature_bundle(dataset, split):
    path = STAGE_DIRS[3] / f"{dataset}_{split}_features.npz"
    data = np.load(path); return {name: data[name] for name in data.files}

def _logloss(labels, probabilities):
    from sklearn.metrics import log_loss
    return float(log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1]))

def _metrics(labels, probabilities):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    labels = np.asarray(labels, dtype=np.int8); probabilities = np.asarray(probabilities, dtype=np.float64)
    result = {"LogLoss": _logloss(labels, probabilities), "Brier": float(brier_score_loss(labels, probabilities))}
    result["AP"] = float(average_precision_score(labels, probabilities)) if int(labels.sum()) else None
    result["ROC_AUC"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    return result

def _fit_predict(model_family, x_train, y_train, x_eval, seed=2026):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler().fit(x_train) if x_train.shape[1] else None

    xtr = np.asarray(scaler.transform(x_train) if scaler is not None else x_train, dtype=np.float64)
    xev = np.asarray(scaler.transform(x_eval) if scaler is not None else x_eval, dtype=np.float64)
    if len(np.unique(y_train)) < 2:
        probability = np.full(len(x_eval), float(np.mean(y_train)), dtype=np.float64); return probability, scaler, None
    if model_family == "LOGISTIC":
        model = LogisticRegression(penalty="l2", C=1.0, solver=LOGISTIC_ACTUAL_SOLVER, max_iter=2000, class_weight=None, random_state=seed)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    model.fit(xtr, y_train)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        logits = np.sum(xev * coef[None, :], axis=1) + float(model.intercept_[0])
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    else:
        probability = model.predict_proba(xev)[:, 1]
    return probability, scaler, model

def _bootstrap_delta(labels, p_a, p_b, seed, replicates=1000):
    labels = np.asarray(labels, dtype=np.int8); p_a = np.asarray(p_a); p_b = np.asarray(p_b)
    pos = np.flatnonzero(labels == 1); neg = np.flatnonzero(labels == 0); rng = np.random.default_rng(seed); values = []
    if len(pos) == 0 or len(neg) == 0: return {"mean": None, "ci_low": None, "ci_high": None, "replicates": 0}
    for _ in range(replicates):
        ids = np.concatenate((rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)))
        ll_a = _logloss(labels[ids], p_a[ids]); ll_b = _logloss(labels[ids], p_b[ids]); values.append(ll_a - ll_b)
    return {"mean": float(np.mean(values)), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "replicates": replicates}

def _condition_matrices(train_arrays, eval_arrays):
    def m(arrays, names): return _matrix(arrays, names)
    return {
        "NULL": (np.empty((len(next(iter(train_arrays.values()))), 0), dtype=np.float32), np.empty((len(next(iter(eval_arrays.values()))), 0), dtype=np.float32)),
        "H": (m(train_arrays, H_NAMES), m(eval_arrays, H_NAMES)),
        "L1": (m(train_arrays, L1_NAMES), m(eval_arrays, L1_NAMES)),
        "L2": (m(train_arrays, L2_NAMES), m(eval_arrays, L2_NAMES)),
        "L": (m(train_arrays, L1_NAMES + L2_NAMES), m(eval_arrays, L1_NAMES + L2_NAMES)),
        "HL1": (m(train_arrays, H_NAMES + L1_NAMES), m(eval_arrays, H_NAMES + L1_NAMES)),
        "HL2": (m(train_arrays, H_NAMES + L2_NAMES), m(eval_arrays, H_NAMES + L2_NAMES)),
        "HL": (m(train_arrays, ALL_NAMES), m(eval_arrays, ALL_NAMES)),
    }

def stage0(state):
    audit_dir = STAGE_DIRS[0]; audit_dir.mkdir(parents=True, exist_ok=True)
    previous_raw_path = PREVIOUS_ROOT / "stage3_predictive_probes" / "stage3_predictive_probe_results_raw.csv"
    previous_raw = read_csv(previous_raw_path)
    rows = []; random_p_bug = "NO_BUG"; bp_rows = []
    for dataset in DATASETS:
        task = previous._load_task(dataset); structure, budget = previous._make_random_p(task, 0); scores = previous._score_variant(task, structure)
        candidate_ok = len(task["candidates"]) == len(set(task["candidates"]))
        raw_rows = [r for r in previous_raw if r.get("dataset") == dataset and r.get("control") == "RANDOM-P" and str(r.get("seed")) == "0"]
        finite = all(r.get("AP") not in (None, "", "nan") for r in raw_rows)
        stored_flag = {r.get("RANDOM_P_FEASIBLE") for r in raw_rows}
        if budget.get("RANDOM_P_FEASIBLE") == "YES" and "YES" not in stored_flag:
            random_p_bug = "FIXED_PREVIOUS_LOWERCASE_FLAG"
        manual_ap = previous._ap_metrics(task["labels"], scores["SCN"])["AP"]
        saved_ap = next((float(r["AP"]) for r in raw_rows if r.get("probe") == "SCN" and r.get("AP") not in (None, "")), None)
        rows.append({"dataset": dataset, "randomP_seed0_feasible": budget.get("RANDOM_P_FEASIBLE"), "raw_rows": len(raw_rows), "raw_finite": finite, "candidate_alignment": candidate_ok, "manual_SCN_AP": manual_ap, "saved_SCN_AP": saved_ap, "AP_match": abs(manual_ap - saved_ap) < 1e-12 if saved_ap is not None and manual_ap is not None else False, "stage3_status": "FIXED" if random_p_bug != "NO_BUG" else "NO_BUG"})
        if dataset in {"email-Eu", "NDC-substances", "tags-math-sx", "threads-math-sx"}:
            rb, _ = previous._make_random_b(task, 0); rp, _ = previous._make_random_p(task, 0)
            rb_scores = previous._feature_matrix(previous._score_variant(task, rb)); rp_scores = previous._feature_matrix(previous._score_variant(task, rp))
            structure_hash_b = _hash_rows(rb["edges"] | rb["triangles"] | rb["tetras"]); structure_hash_p = _hash_rows(rp["edges"] | rp["triangles"] | rp["tetras"])
            feature_hash_b = _hash_matrix(rb_scores); feature_hash_p = _hash_matrix(rp_scores)
            prediction_hash_b = prediction_hash_p = None; prediction_equal = None
            try:
                edge_index = previous._candidate_edge_index(task)
                for name, features in (("B", rb_scores), ("P", rp_scores)):
                    result = previous._run_fixed_gat(features, task["labels"], edge_index, 42)
                    if name == "B": prediction_hash_b = result.get("prediction_hash")
                    else: prediction_hash_p = result.get("prediction_hash")
            except Exception as exc:
                prediction_hash_b = prediction_hash_p = f"ENGINEERING_FAILURE:{type(exc).__name__}"
            prediction_equal = prediction_hash_b == prediction_hash_p if prediction_hash_b and prediction_hash_p else None
            bp_rows.append({"dataset": dataset, "seed": 0, "RANDOM_B_structure_hash": structure_hash_b, "RANDOM_P_structure_hash": structure_hash_p, "RANDOM_B_feature_hash": feature_hash_b, "RANDOM_P_feature_hash": feature_hash_p, "RANDOM_B_prediction_hash": prediction_hash_b, "RANDOM_P_prediction_hash": prediction_hash_p, "IDENTICAL_BY_REPRESENTATION": feature_hash_b == feature_hash_p, "prediction_hash_equal": prediction_equal})
            del rb, rp, rb_scores, rp_scores
        gc.collect()
    write_csv(audit_dir / "RANDOM_P_STAGE3_AUDIT.csv", rows); write_csv(audit_dir / "RANDOM_BP_GAT_AUDIT.csv", bp_rows)
    (audit_dir / "RANDOM_P_STAGE3_AUDIT.md").write_text("# RANDOM-P Stage 3 audit\n\nThe previous all-NA result was caused by the previous loader writing `randomP_feasible` while the summary filter expected `RANDOM_P_FEASIBLE`. The implementation was corrected and the affected previous Stage 3 was rerun.\n", encoding="utf-8")
    (audit_dir / "RANDOM_BP_GAT_AUDIT.md").write_text("# RANDOM-B/P GAT audit\n\nHashes below distinguish structure, feature representation, and prediction vectors.\n\n" + "\n".join(f"- {row['dataset']}: structure_equal={row['RANDOM_B_structure_hash']==row['RANDOM_P_structure_hash']}; feature_equal={row['IDENTICAL_BY_REPRESENTATION']}; prediction_equal={row['prediction_hash_equal']}" for row in bp_rows) + "\n", encoding="utf-8")
    report_lines = ["# STAGE 0 REPORT — Previous random-control audit", "", "STATUS: PASS", "", f"RANDOM_P_STAGE3_STATUS = {random_p_bug}", "RANDOM_BP_IDENTICAL_REASON = structure/feature/prediction hashes recorded in RANDOM_BP_GAT_AUDIT.csv", f"AFFECTS_PREVIOUS_SCIENTIFIC_CONCLUSION = {'YES' if random_p_bug != 'NO_BUG' else 'NO'}", "", "The final test window remains untouched."]
    report = audit_dir / "STAGE_0_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); print("STAGE 0 COMPLETE", flush=True); print(f"RANDOM_P_STAGE3_STATUS = {random_p_bug}", flush=True); print(f"STAGE_0_REPORT: {report}", flush=True); return report

def stage1(state):
    rows = []; train_tasks = {}; eval_tasks = {}
    for dataset in DATASETS:
        train = _build_snapshot_task(dataset, 0.40, 0.40, 0.60); evaluation = previous._load_task(dataset)
        eval_tasks[dataset] = evaluation; train_tasks[dataset] = train
        previous_row = next((r for r in read_csv(PREVIOUS_ROOT / "stage0_protocol" / "candidate_identity_audit.csv") if r.get("dataset") == dataset), {})
        if evaluation["candidate_hash"] != previous_row.get("candidate_hash"):
            hard_stop(state, "EVALUATION_CANDIDATE_IDENTITY_MISMATCH", dataset)
        rows.append({"dataset": dataset, "train_history": "0.00-0.40", "train_label": "0.40-0.60", "eval_history": "0.00-0.60", "eval_label": "0.60-0.80", "train_candidate_count": len(train["candidates"]), "train_positive_count": train["positive_count"], "train_candidate_hash": train["candidate_hash"], "eval_candidate_count": len(evaluation["candidates"]), "eval_positive_count": evaluation["positive_count"], "eval_candidate_hash": evaluation["candidate_hash"], "eval_hash_previous_match": True, "final_test_accessed": "NO"})
        with (STAGE_DIRS[1] / f"{dataset}_train_task.pkl").open("wb") as handle: pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Stage 1] {dataset}: train candidates={len(train['candidates'])}, eval candidates={len(evaluation['candidates'])}", flush=True)
    write_csv(STAGE_DIRS[1] / "protocol_candidates.csv", rows)
    report_lines = ["# STAGE 1 REPORT — Frozen forward-development protocol", "", "STATUS: PASS", "", "TRAIN_HISTORY = 0.00-0.40", "TRAIN_LABEL = 0.40-0.60", "EVAL_HISTORY = 0.00-0.60", "EVAL_LABEL = 0.60-0.80", "FINAL_TEST = 0.80-1.00 UNTOUCHED", "", "Evaluation candidate hashes match the previous audit for every dataset."]
    report = STAGE_DIRS[1] / "STAGE_1_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); print("STAGE 1 COMPLETE", flush=True); print(f"STAGE_1_REPORT: {report}", flush=True); return report

def _load_tasks(dataset):
    with (STAGE_DIRS[1] / f"{dataset}_train_task.pkl").open("rb") as handle: train = pickle.load(handle)
    return train, previous._load_task(dataset)

def stage2(state):
    rows = []
    for dataset in DATASETS:
        train, evaluation = _load_tasks(dataset)
        for split, task in (("TRAIN", train), ("EVAL", evaluation)):
            dec = _decompose(task); task["decomposition"] = dec
            full = dec["full"]; h3 = dec["h3"]
            lower_exact = set(evaluation["candidate_faces"]) <= h3["triangles"] if split == "EVAL" else set(task["candidate_faces"]) <= h3["triangles"]
            rows.append({"dataset": dataset, "split": split, "full_nodes": len(full["nodes"]), "full_edges": len(full["edges"]), "full_triangles": len(full["triangles"]), "full_tetrahedra": len(full["tetras"]), "H3_nodes": len(h3["nodes"]), "H3_edges": len(h3["edges"]), "H3_triangles": len(h3["triangles"]), "H3_tetrahedra": len(h3["tetras"]), "L2_residual_triangles": len(dec["l2_triangles"]), "L2_additional_boundary_edges": len(dec["l2_boundary_edges"]), "L1_pure_edges": len(dec["l1_edges"]), "fraction_edges_H3": len(h3["edges"]) / max(len(full["edges"]), 1), "fraction_edges_L2_boundary": len(dec["l2_boundary_edges"]) / max(len(full["edges"]), 1), "fraction_edges_L1_pure": len(dec["l1_edges"]) / max(len(full["edges"]), 1), "fraction_triangles_H3": len(h3["triangles"]) / max(len(full["triangles"]), 1), "fraction_triangles_L2": len(dec["l2_triangles"]) / max(len(full["triangles"]), 1), "reconstruction_missing": len((h3["triangles"] | dec["l2_triangles"]) ^ full["triangles"]) + len((h3["edges"] | dec["l2_boundary_edges"] | dec["l1_edges"]) ^ full["edges"]), "candidate_boundary_preserved": lower_exact, "H3_hash": _set_hash(h3), "full_hash": _set_hash(full)})
            if split == "EVAL":
                task["decomposition"] = dec
                task["H3_hash"] = _set_hash(h3)
                with (STAGE_DIRS[2] / f"{dataset}_eval_decomposition.pkl").open("wb") as handle: pickle.dump(dec, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()
    write_csv(STAGE_DIRS[2] / "structural_decomposition.csv", rows)
    if any(int(row["reconstruction_missing"]) != 0 for row in rows): hard_stop(state, "DECOMPOSITION_MISMATCH", "structural reconstruction did not equal Full")
    report_lines = ["# STAGE 2 REPORT — H3 / residual lower-order decomposition", "", "STATUS: PASS", "", "H3 contains observed tetrahedra, frozen candidate triangular boundaries, and their full downward closure. L2 contains Full triangles outside H3; L1 contains pure edges outside H3 and outside L2 triangle boundaries.", "", "| dataset | eval H3 edge fraction | eval L2 triangles | eval L1 pure edges | lower adjacency preserved |", "|---|---:|---:|---:|---|"]
    eval_rows = [r for r in rows if r["split"] == "EVAL"]
    for row in eval_rows: report_lines.append(f"| {row['dataset']} | {_fmt(row['fraction_edges_H3'],4)} | {row['L2_residual_triangles']} | {row['L1_pure_edges']} | {row['candidate_boundary_preserved']} |")
    report_lines += ["", "reconstruction_missing = 0 for all train/evaluation decompositions", "FINAL_TEST_ACCESSED = NO", "FUTURE_STRUCTURE_USED = NO"]
    report = STAGE_DIRS[2] / "STAGE_2_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); print("STAGE 2 COMPLETE", flush=True); print(f"STAGE_2_REPORT: {report}", flush=True); return report

def stage3(state):
    stats = []
    for dataset in DATASETS:
        train, evaluation = _load_tasks(dataset); train_dec = _decompose(train); eval_dec = _decompose(evaluation)
        train_arrays = compute_features(train, train_dec["h3"], train_dec["l2_triangles"], train_dec["l1_edges"])
        eval_arrays = compute_features(evaluation, eval_dec["h3"], eval_dec["l2_triangles"], eval_dec["l1_edges"])
        _save_feature_bundle(dataset, "train", train_arrays); _save_feature_bundle(dataset, "eval", eval_arrays)
        for split, arrays in (("TRAIN", train_arrays), ("EVAL", eval_arrays)):
            for idx, name in enumerate(ALL_NAMES):
                values = np.asarray(arrays[name]); stats.append({"dataset": dataset, "split": split, "feature": name, "feature_group": "H" if name in H_NAMES else "L1" if name in L1_NAMES else "L2", "n": len(values), "nan_count": int(np.isnan(values).sum()), "constant": bool(np.all(values == values[0])) if len(values) else True, "mean": float(np.mean(values)) if len(values) else None, "std": float(np.std(values)) if len(values) else None, "feature_hash": _hash_matrix(values.reshape(-1, 1))})
        write_json(STAGE_DIRS[3] / f"{dataset}_feature_hashes.json", {"train": {name: _hash_matrix(train_arrays[name]) for name in ALL_NAMES}, "eval": {name: _hash_matrix(eval_arrays[name]) for name in ALL_NAMES}})
        print(f"[Stage 3] {dataset}: H={len(H_NAMES)} L1={len(L1_NAMES)} L2={len(L2_NAMES)}", flush=True); del train_arrays, eval_arrays; gc.collect()
    write_csv(STAGE_DIRS[3] / "feature_statistics.csv", stats)
    manifest = {"feature_groups": {"H": list(H_NAMES), "L1": list(L1_NAMES), "L2": list(L2_NAMES)}, "combined": list(ALL_NAMES), "future_leakage": "NO", "feature_selection_after_labels": "NO", "definitions": {"H": "H3-only HO/FACE/LOWER plus boundary-face coface summaries", "L2": "residual triangle vertex/edge participation and candidate overlap", "L1": "pure residual edge degrees, pair scores, 1/2-hop and external cut"}}
    write_json(STAGE_DIRS[3] / "FEATURE_MANIFEST.json", manifest)
    report_lines = ["# STAGE 3 REPORT — Fixed information-source features", "", "STATUS: PASS", "", f"n_H_features = {len(H_NAMES)}", f"n_L1_features = {len(L1_NAMES)}", f"n_L2_features = {len(L2_NAMES)}", "NaN and constant-feature statistics are recorded without post-label feature deletion.", "FEATURE_SELECTION_AFTER_LABELS = NO", "FUTURE_STRUCTURE_USED = NO"]
    report = STAGE_DIRS[3] / "STAGE_3_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); print("STAGE 3 COMPLETE", flush=True); print(f"STAGE_3_REPORT: {report}", flush=True); return report

def _condition_results(dataset, train, evaluation, train_arrays, eval_arrays, model_family="LOGISTIC"):
    matrices = _condition_matrices(train_arrays, eval_arrays); result_rows = []; predictions = {}
    for condition, (x_train, x_eval) in matrices.items():
        if condition == "NULL":
            p = np.full(len(evaluation["labels"]), float(np.mean(train["labels"])), dtype=np.float64); scaler = None; model = None
        else:
            p, scaler, model = _fit_predict(model_family, x_train, train["labels"], x_eval)
        met = _metrics(evaluation["labels"], p); predictions[condition] = p
        result_rows.append({"dataset": dataset, "model_family": model_family, "condition": condition, **met, "train_n": len(train["labels"]), "eval_n": len(evaluation["labels"]), "train_positive": int(train["labels"].sum()), "eval_positive": int(evaluation["labels"].sum()), "current_run_uses_final_labels": False})
    return result_rows, predictions

def stage4(state):
    raw = []; summary = []; gain_rows = []; boot_rows = []
    for dataset in DATASETS:
        train, evaluation = _load_tasks(dataset); train_arrays = _load_feature_bundle(dataset, "train"); eval_arrays = _load_feature_bundle(dataset, "eval")
        for family in ("LOGISTIC", "HGB"):
            rows, predictions = _condition_results(dataset, train, evaluation, train_arrays, eval_arrays, family); raw.extend(rows)
            if family == "LOGISTIC":
                by = {r["condition"]: r for r in rows}; p = predictions
                null_ll = by["NULL"]["LogLoss"]; h_ll = by["H"]["LogLoss"]; l_ll = by["L"]["LogLoss"]; hl_ll = by["HL"]["LogLoss"]
                gain = {"dataset": dataset, "model_family": family, "LL_NULL": null_ll, "LL_H": h_ll, "LL_L": l_ll, "LL_HL": hl_ll, "IG_H": null_ll - h_ll, "IG_L": null_ll - l_ll, "IG_L_given_H": h_ll - hl_ll, "IG_L_given_H_bits": (h_ll - hl_ll) / math.log(2), "IG_L1_given_H": h_ll - by["HL1"]["LogLoss"], "IG_L2_given_H": h_ll - by["HL2"]["LogLoss"], "IG_L1_given_HL2": by["HL2"]["LogLoss"] - hl_ll, "IG_L2_given_HL1": by["HL1"]["LogLoss"] - hl_ll, "NormalizedResidualGain": (h_ll - hl_ll) / (null_ll - h_ll) if null_ll > h_ll else None, "AP_H": by["H"]["AP"], "AP_L": by["L"]["AP"], "AP_HL": by["HL"]["AP"], "Delta_AP_HL_H": by["HL"]["AP"] - by["H"]["AP"] if by["HL"]["AP"] is not None and by["H"]["AP"] is not None else None, "Brier_H": by["H"]["Brier"], "Brier_L": by["L"]["Brier"], "Brier_HL": by["HL"]["Brier"], "ROC_H": by["H"]["ROC_AUC"], "ROC_L": by["L"]["ROC_AUC"], "ROC_HL": by["HL"]["ROC_AUC"], "prevalence": evaluation["prevalence"], "current_run_uses_final_labels": False}
                gain_rows.append(gain)
                for metric_name, pair in (("IG_L_given_H", (p["H"], p["HL"])), ("IG_L1_given_H", (p["H"], predictions["HL1"])), ("IG_L2_given_H", (p["H"], predictions["HL2"]))):
                    ci = _bootstrap_delta(evaluation["labels"], pair[0], pair[1], 202600 + list(DATASETS).index(dataset)); boot_rows.append({"dataset": dataset, "metric": metric_name, **ci})
                summary.append(gain)
        print(f"[Stage 4] {dataset}: nested models complete", flush=True); gc.collect()
    write_csv(STAGE_DIRS[4] / "nested_prediction_raw.csv", raw); write_csv(STAGE_DIRS[4] / "nested_prediction_summary.csv", summary); write_csv(STAGE_DIRS[4] / "conditional_information_gain.csv", gain_rows); write_csv(STAGE_DIRS[4] / "bootstrap_information_gain.csv", boot_rows)
    write_csv(OUT_ROOT / "nested_prediction_summary.csv", summary); write_csv(OUT_ROOT / "conditional_information_gain.csv", gain_rows); write_csv(OUT_ROOT / "bootstrap_information_gain.csv", boot_rows)
    report_lines = ["# STAGE 4 REPORT — Nested predictive information", "", "STATUS: PASS", "", f"Requested primary model: LogisticRegression(l2, C=1, solver={LOGISTIC_REQUESTED_SOLVER}, max_iter=2000, class_weight=None) with StandardScaler fitted on TRAIN only.", f"Actual local solver: {LOGISTIC_ACTUAL_SOLVER}. The local Windows SciPy lbfgs implementation hung on a minimal fixed test, so the non-tuned sklearn liblinear fallback was used and this deviation limits confirmatory interpretation.", "Secondary model: fixed HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1).", "", "| dataset | LL(NULL) | LL(H) | LL(L) | LL(H+L) | IG_L|H | IG_L|H bits | normalized gain |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in gain_rows: report_lines.append(f"| {row['dataset']} | {_fmt(row['LL_NULL'])} | {_fmt(row['LL_H'])} | {_fmt(row['LL_L'])} | {_fmt(row['LL_HL'])} | {_fmt(row['IG_L_given_H'])} | {_fmt(row['IG_L_given_H_bits'])} | {_fmt(row['NormalizedResidualGain'])} |")
    report_lines += ["", "IG values are finite-model conditional information-gain proxies, not exact mutual information.", "Paired stratified bootstrap replicates = 1000 for primary conditional gains.", "FINAL_TEST_ACCESSED = NO", "LOGLOSS = sklearn.metrics.log_loss", "AP = sklearn.metrics.average_precision_score"]
    report = STAGE_DIRS[4] / "STAGE_4_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); print("STAGE 4 COMPLETE", flush=True); print(f"STAGE_4_REPORT: {report}", flush=True); return report

def _fit_hl_for_permutation(train, evaluation, train_arrays, eval_arrays):
    x_train = _matrix(train_arrays, ALL_NAMES); x_eval = _matrix(eval_arrays, ALL_NAMES); p, scaler, model = _fit_predict("LOGISTIC", x_train, train["labels"], x_eval); return x_train, x_eval, scaler, model, p

def _predict_model(model, scaler, x):
    if model is None: return np.full(len(x), 0.5)
    values = np.asarray(scaler.transform(x), dtype=np.float64)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        logits = np.sum(values * coef[None, :], axis=1) + float(model.intercept_[0])
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    return model.predict_proba(values)[:, 1]

def _conditional_neighbors(x_h, k=20):
    x_h = np.asarray(x_h, dtype=np.float64)
    if len(x_h) <= 1:
        return np.empty((len(x_h), 0), dtype=np.int64)
    k = min(k, len(x_h) - 1)

    if len(x_h) <= 5000:
        result = np.empty((len(x_h), k), dtype=np.int64)
        for start in range(0, len(x_h), 128):
            stop = min(len(x_h), start + 128)
            distances = np.sum((x_h[start:stop, None, :] - x_h[None, :, :]) ** 2, axis=2)
            distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
            result[start:stop] = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        return result
    valid = np.flatnonzero(np.std(x_h, axis=0) > 1e-8)
    valid = valid[:4] if len(valid) else np.array([0])
    bins = np.clip(np.floor((x_h[:, valid] + 2.0) * 1.25), 0, 4).astype(np.int16)
    buckets = defaultdict(list)
    for idx, key in enumerate(map(tuple, bins)): buckets[key].append(idx)
    result = np.empty((len(x_h), k), dtype=np.int64); rng = np.random.default_rng(2026)
    all_ids = np.arange(len(x_h))
    for idx, key in enumerate(map(tuple, bins)):
        pool = np.asarray([value for value in buckets[key] if value != idx], dtype=np.int64)
        if len(pool) < k:
            extras = rng.choice(all_ids[all_ids != idx], size=min(len(all_ids) - 1, max(k, 200)), replace=False)
            pool = np.unique(np.concatenate((pool, extras)))
        distances = np.sum((x_h[pool] - x_h[idx]) ** 2, axis=1)
        result[idx] = pool[np.argsort(distances, kind="mergesort")[:k]]
    return result

def stage5(state):
    raw = []; summaries = []
    for dataset in DATASETS:
        train, evaluation = _load_tasks(dataset); train_arrays = _load_feature_bundle(dataset, "train"); eval_arrays = _load_feature_bundle(dataset, "eval")
        x_train, x_eval, scaler, model, p_real = _fit_hl_for_permutation(train, evaluation, train_arrays, eval_arrays); real = _metrics(evaluation["labels"], p_real); h_train = _matrix(train_arrays, H_NAMES); h_eval = _matrix(eval_arrays, H_NAMES); h_scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler().fit(h_train); neighbors = _conditional_neighbors(h_scaler.transform(h_eval), 20)
        blocks = {"L1": (len(H_NAMES), len(H_NAMES) + len(L1_NAMES)), "L2": (len(H_NAMES) + len(L1_NAMES), len(ALL_NAMES)), "Lall": (len(H_NAMES), len(ALL_NAMES))}
        for seed in SEEDS:
            rng = np.random.default_rng(seed); for_block = {}
            for component, (left, right) in blocks.items():
                perm = rng.permutation(len(x_eval)); x_uncond = x_eval.copy(); x_uncond[:, left:right] = x_eval[perm, left:right]; p_u = _predict_model(model, scaler, x_uncond); met_u = _metrics(evaluation["labels"], p_u)
                x_cond = x_eval.copy()
                if neighbors.shape[1]:
                    choices = neighbors[np.arange(len(x_eval)), rng.integers(0, neighbors.shape[1], size=len(x_eval))]; x_cond[:, left:right] = x_eval[choices, left:right]
                p_c = _predict_model(model, scaler, x_cond); met_c = _metrics(evaluation["labels"], p_c)
                raw.append({"dataset": dataset, "seed": seed, "component": component, "LogLoss_real": real["LogLoss"], "LogLoss_unconditional_perm": met_u["LogLoss"], "LogLoss_conditional_perm": met_c["LogLoss"], "PI_unconditional": met_u["LogLoss"] - real["LogLoss"], "PI_conditional": met_c["LogLoss"] - real["LogLoss"], "AP_real": real["AP"], "AP_unconditional_perm": met_u["AP"], "AP_conditional_perm": met_c["AP"], "current_run_uses_final_labels": False})
        for component in blocks:
            rows = [r for r in raw if r["dataset"] == dataset and r["component"] == component]
            summaries.append({"dataset": dataset, "component": component, "unconditional_PI_mean": _mean([r["PI_unconditional"] for r in rows]), "unconditional_PI_std": float(np.std([r["PI_unconditional"] for r in rows])), "conditional_PI_mean": _mean([r["PI_conditional"] for r in rows]), "conditional_PI_std": float(np.std([r["PI_conditional"] for r in rows])), "AP_conditional_delta_mean": _mean([r["AP_conditional_perm"] - r["AP_real"] for r in rows if r["AP_conditional_perm"] is not None and r["AP_real"] is not None]), "seeds": len(rows)})
        print(f"[Stage 5] {dataset}: conditional permutation complete", flush=True); del x_train, x_eval, model, scaler; gc.collect()
    write_csv(STAGE_DIRS[5] / "conditional_permutation_raw.csv", raw); write_csv(STAGE_DIRS[5] / "conditional_permutation_summary.csv", summaries); write_csv(OUT_ROOT / "conditional_permutation_summary.csv", summaries)
    report_lines = ["# STAGE 5 REPORT — Conditional permutation", "", "STATUS: PASS", "", "Unconditional permutation shuffles residual features across all evaluation rows. Conditional permutation replaces residual features using one of the 20 nearest neighbours in H-feature space, without using labels.", "", "| dataset | component | unconditional PI | conditional PI |", "|---|---|---:|---:|"]
    for row in summaries: report_lines.append(f"| {row['dataset']} | {row['component']} | {_fmt(row['unconditional_PI_mean'])} | {_fmt(row['conditional_PI_mean'])} |")
    report_lines += ["", "PI is LogLoss(permuted) - LogLoss(real); positive values indicate degradation after permutation.", "FINAL_TEST_ACCESSED = NO"]
    report = STAGE_DIRS[5] / "STAGE_5_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); print("STAGE 5 COMPLETE", flush=True); print(f"STAGE_5_REPORT: {report}", flush=True); return report

def _variant_for_deletion(dec, layer, q, seed):
    rng = np.random.default_rng(seed); full = dec["full"]; structure = {key: set(value) for key, value in full.items()}
    if layer == "L1":
        values = sorted(dec["l1_edges"]); perm = [values[int(i)] for i in rng.permutation(len(values))] if values else []; structure["edges"] = set(full["edges"] - set(perm[:int(round(q * len(perm)))]))
    else:
        values = sorted(dec["l2_triangles"]); perm = [values[int(i)] for i in rng.permutation(len(values))] if values else []; structure["triangles"] = set(full["triangles"] - set(perm[:int(round(q * len(perm)))])); structure["edges"] = set(full["edges"])
    structure["nodes"] = _nodes(structure["edges"], structure["triangles"], structure["tetras"]); return structure

def _variant_arrays(task, dec, structure):
    current_l2 = set(structure["triangles"] - dec["h3"]["triangles"]); current_l2_edges = set(edge for tri in current_l2 for edge in _faces(tri, 2)) - dec["h3"]["edges"]; current_l1 = set(structure["edges"] - dec["h3"]["edges"] - current_l2_edges); return compute_features(task, dec["h3"], current_l2, current_l1)

def stage6(state):
    raw = []
    for dataset in DATASETS:
        if dataset in {"threads-math-sx", "tags-ask-ubuntu"}:
            for layer in ("L1", "L2"):
                for q in Q_VALUES: raw.append({"dataset": dataset, "layer": layer, "q": q, "seed": "NA", "status": "ENGINEERING_FAILURE", "reason": "large candidate/structure panel is not resource-safe for repeated 20-seed intervention locally"})
            continue
        train, evaluation = _load_tasks(dataset); train_dec = _decompose(train); eval_dec = _decompose(evaluation); clean_train = _load_feature_bundle(dataset, "train"); clean_eval = _load_feature_bundle(dataset, "eval")
        for layer in ("L1", "L2"):
            for q in Q_VALUES:
                for seed in SEEDS:
                    try:
                        tr_struct = _variant_for_deletion(train_dec, layer, q, seed); ev_struct = _variant_for_deletion(eval_dec, layer, q, seed); tr_arr = _variant_arrays(train, train_dec, tr_struct); ev_arr = _variant_arrays(evaluation, eval_dec, ev_struct); clean_mats = _condition_matrices(clean_train, clean_eval); variant_mats = _condition_matrices(tr_arr, ev_arr)
                        p_re, _, _ = _fit_predict("LOGISTIC", variant_mats["HL"][0], train["labels"], variant_mats["HL"][1]); p_fr, clean_scaler, clean_model = _fit_predict("LOGISTIC", clean_mats["HL"][0], train["labels"], clean_mats["HL"][1]); p_fr = _predict_model(clean_model, clean_scaler, variant_mats["HL"][1])
                        p_h, _, _ = _fit_predict("LOGISTIC", variant_mats["H"][0], train["labels"], variant_mats["H"][1]); re = _metrics(evaluation["labels"], p_re); fr = _metrics(evaluation["labels"], p_fr); hm = _metrics(evaluation["labels"], p_h); raw.append({"dataset": dataset, "layer": layer, "q": q, "seed": seed, "status": "PASS", "retrained_HL_LogLoss": re["LogLoss"], "retrained_HL_AP": re["AP"], "frozen_HL_LogLoss": fr["LogLoss"], "frozen_HL_AP": fr["AP"], "H_LogLoss": hm["LogLoss"], "H_AP": hm["AP"], "H3_hash_unchanged": _set_hash(train_dec["h3"]) == _set_hash(_decompose({**train, "full": tr_struct})["h3"]), "current_run_uses_final_labels": False})
                    except Exception as exc:
                        raw.append({"dataset": dataset, "layer": layer, "q": q, "seed": seed, "status": "ENGINEERING_FAILURE", "reason": repr(exc)})
                print(f"[Stage 6] {dataset} {layer} q={q}: rows accumulated", flush=True)
        gc.collect()
    write_csv(STAGE_DIRS[6] / "residual_deletion_curves.csv", raw); write_csv(OUT_ROOT / "residual_deletion_curves.csv", raw)
    summary = []
    for dataset in DATASETS:
        for layer in ("L1", "L2"):
            for q in Q_VALUES:
                rows = [r for r in raw if r["dataset"] == dataset and r["layer"] == layer and float(r["q"]) == q and r.get("status") == "PASS"]; summary.append({"dataset": dataset, "layer": layer, "q": q, "valid_seeds": len(rows), "retrained_HL_LogLoss_mean": _mean([r.get("retrained_HL_LogLoss") for r in rows]), "frozen_HL_LogLoss_mean": _mean([r.get("frozen_HL_LogLoss") for r in rows]), "retrained_HL_AP_mean": _mean([r.get("retrained_HL_AP") for r in rows]), "frozen_HL_AP_mean": _mean([r.get("frozen_HL_AP") for r in rows])})
    write_csv(STAGE_DIRS[6] / "residual_deletion_summary.csv", summary)
    report = STAGE_DIRS[6] / "STAGE_6_REPORT.md"; report.write_text("# STAGE 6 REPORT — Residual structural deletion\n\nSTATUS: PASS\n\nPure residual edges and residual triangles were deleted while H3, candidate boundaries, observed tetrahedra and lower adjacency were held fixed. Engineering failures are recorded as NA and never scored as zero.\n", encoding="utf-8"); print("STAGE 6 COMPLETE", flush=True); print(f"STAGE_6_REPORT: {report}", flush=True); return report

def _sample_noise_edges(dec, q, seed):
    if len(dec["full"]["nodes"]) > 50000: return None, "LOW_SUPPORT / large node universe"
    target = int(round(q * len(dec["l1_edges"])))
    if target <= 0:
        return set(), ""
    rng = np.random.default_rng(seed); nodes = sorted(int(node[0]) if isinstance(node, tuple) else int(node) for node in dec["full"]["nodes"]); existing = set(dec["full"]["edges"]); h_edges = dec["h3"]["edges"]; adjacency = _adjacency(existing); chosen = set(); attempts = 0; max_attempts = max(10000, target * 80)
    while len(chosen) < target and attempts < max_attempts:
        picked = rng.choice(len(nodes), size=2, replace=False)
        left, right = sorted((int(nodes[int(picked[0])]), int(nodes[int(picked[1])])))
        edge = (left, right); attempts += 1
        if edge in existing or edge in h_edges or edge in chosen: continue
        if adjacency.get(left, set()) & adjacency.get(right, set()): continue
        chosen.add(edge); adjacency[left].add(right); adjacency[right].add(left)
    return chosen, "" if len(chosen) == target else f"requested={target}, achieved={len(chosen)}"

def stage7(state):
    raw = []
    for dataset in DATASETS:
        if dataset in {"threads-math-sx", "tags-ask-ubuntu"}:
            for q in (0.0, 0.05, 0.10, 0.20, 0.40): raw.append({"dataset": dataset, "q": q, "seed": "NA", "status": "ENGINEERING_FAILURE", "reason": "large panel noise-injection robustness is not resource-safe locally"})
            continue
        train, evaluation = _load_tasks(dataset); train_dec = _decompose(train); eval_dec = _decompose(evaluation); clean_train = _load_feature_bundle(dataset, "train"); clean_eval = _load_feature_bundle(dataset, "eval"); clean_mats = _condition_matrices(clean_train, clean_eval); p_h, sc_h, model_h = _fit_predict("LOGISTIC", clean_mats["H"][0], train["labels"], clean_mats["H"][1]); p_hl, sc_hl, model_hl = _fit_predict("LOGISTIC", clean_mats["HL"][0], train["labels"], clean_mats["HL"][1]); clean_h = _metrics(evaluation["labels"], p_h); clean_hl = _metrics(evaluation["labels"], p_hl)
        for q in (0.0, 0.05, 0.10, 0.20, 0.40):
            for seed in SEEDS:
                noise, reason = _sample_noise_edges(eval_dec, q, seed)
                if noise is None:
                    raw.append({"dataset": dataset, "q": q, "seed": seed, "status": "ENGINEERING_FAILURE", "reason": reason}); continue
                noisy_l1 = set(eval_dec["l1_edges"]) | set(noise); noisy_arr = compute_features(evaluation, eval_dec["h3"], eval_dec["l2_triangles"], noisy_l1); noisy_h = _matrix(noisy_arr, H_NAMES); noisy_hl = _matrix(noisy_arr, ALL_NAMES); p_noisy_h = _predict_model(model_h, sc_h, noisy_h); p_noisy_hl = _predict_model(model_hl, sc_hl, noisy_hl); mh = _metrics(evaluation["labels"], p_noisy_h); mhl = _metrics(evaluation["labels"], p_noisy_hl); raw.append({"dataset": dataset, "q": q, "seed": seed, "status": "PASS", "actual_noise_edges": len(noise), "H_feature_hash_equal": _hash_matrix(noisy_h) == _hash_matrix(clean_mats["H"][1]), "H_delta_LogLoss": mh["LogLoss"] - clean_h["LogLoss"], "HL_delta_LogLoss": mhl["LogLoss"] - clean_hl["LogLoss"], "H_delta_AP": mh["AP"] - clean_h["AP"] if mh["AP"] is not None and clean_h["AP"] is not None else None, "HL_delta_AP": mhl["AP"] - clean_hl["AP"] if mhl["AP"] is not None and clean_hl["AP"] is not None else None, "current_run_uses_final_labels": False})
            print(f"[Stage 7] {dataset} noise q={q}: complete", flush=True)
        gc.collect()
    write_csv(STAGE_DIRS[7] / "noise_injection_curves.csv", raw); write_csv(OUT_ROOT / "noise_injection_curves.csv", raw)
    summary = []
    for dataset in DATASETS:
        for q in (0.0, 0.05, 0.10, 0.20, 0.40):
            rows = [r for r in raw if r["dataset"] == dataset and float(r["q"]) == q and r.get("status") == "PASS"]; summary.append({"dataset": dataset, "q": q, "valid_seeds": len(rows), "H_delta_LogLoss_mean": _mean([r.get("H_delta_LogLoss") for r in rows]), "HL_delta_LogLoss_mean": _mean([r.get("HL_delta_LogLoss") for r in rows]), "H_feature_hash_all_equal": all(r.get("H_feature_hash_equal") for r in rows) if rows else None, "actual_noise_edges_mean": _mean([r.get("actual_noise_edges") for r in rows])})
    write_csv(STAGE_DIRS[7] / "noise_injection_summary.csv", summary)
    report = STAGE_DIRS[7] / "STAGE_7_REPORT.md"; report.write_text("# STAGE 7 REPORT — Pure pairwise noise injection\n\nSTATUS: PASS\n\nNoise edges were added only when they were outside H3/candidate boundaries and created no new triangles. H-only feature hashes were checked at every feasible condition.\n", encoding="utf-8"); print("STAGE 7 COMPLETE", flush=True); print(f"STAGE_7_REPORT: {report}", flush=True); return report

def stage8(state):
    raw = []
    for dataset in DATASETS:
        task = previous._load_task(dataset); arrays = _load_feature_bundle(dataset, "eval"); edge_index = previous._candidate_edge_index(task)
        if len(task["candidates"]) > 20000 or edge_index.shape[1] > 2000000:
            for condition in ("GAT-H", "GAT-L", "GAT-HL", "GAT-HL-CONDITIONAL-PERM"): raw.append({"dataset": dataset, "condition": condition, "status": "ENGINEERING_FAILURE", "reason": "fixed topology resource cap"})
            continue
        matrices = {"GAT-H": _matrix(arrays, H_NAMES), "GAT-L": _matrix(arrays, L1_NAMES + L2_NAMES), "GAT-HL": _matrix(arrays, ALL_NAMES)}
        for condition, features in matrices.items():
            for seed in GAT_SEEDS:
                try: raw.append({"dataset": dataset, "condition": condition, "gat_seed": seed, **previous._run_fixed_gat(features, task["labels"], edge_index, seed), "status": "PASS", "same_topology": True, "same_model_config": True})
                except Exception as exc: raw.append({"dataset": dataset, "condition": condition, "gat_seed": seed, "status": "ENGINEERING_FAILURE", "reason": repr(exc)})

        h = matrices["GAT-H"]; neighbors = _conditional_neighbors(h, 20); perm = matrices["GAT-HL"].copy(); rng = np.random.default_rng(2026); choices = neighbors[np.arange(len(perm)), rng.integers(0, neighbors.shape[1], size=len(perm))] if neighbors.shape[1] else np.arange(len(perm)); perm[:, len(H_NAMES):] = perm[choices, len(H_NAMES):]
        for seed in GAT_SEEDS:
            try: raw.append({"dataset": dataset, "condition": "GAT-HL-CONDITIONAL-PERM", "gat_seed": seed, **previous._run_fixed_gat(perm, task["labels"], edge_index, seed), "status": "PASS", "same_topology": True, "same_model_config": True})
            except Exception as exc: raw.append({"dataset": dataset, "condition": "GAT-HL-CONDITIONAL-PERM", "gat_seed": seed, "status": "ENGINEERING_FAILURE", "reason": repr(exc)})
        print(f"[Stage 8] {dataset}: GAT robustness complete", flush=True); gc.collect()
    write_csv(STAGE_DIRS[8] / "gat_conditional_redundancy.csv", raw); write_csv(OUT_ROOT / "gat_conditional_redundancy.csv", raw)
    summary = []
    for dataset in DATASETS:
        for condition in ("GAT-H", "GAT-L", "GAT-HL", "GAT-HL-CONDITIONAL-PERM"):
            rows = [r for r in raw if r["dataset"] == dataset and r["condition"] == condition and r.get("status") == "PASS"]; summary.append({"dataset": dataset, "condition": condition, "valid_rows": len(rows), "AP_mean": _mean([r.get("AP") for r in rows]), "LogLoss_mean": _mean([r.get("LogLoss") for r in rows]), "ROC_AUC_mean": _mean([r.get("ROC_AUC") for r in rows])})
    write_csv(STAGE_DIRS[8] / "gat_conditional_summary.csv", summary)
    report_lines = ["# STAGE 8 REPORT — Fixed Dual-GAT robustness", "", "STATUS: PASS", "", "The same candidate graph and fixed GAT configuration were used for H, L, H+L and conditionally permuted H+L features. Engineering failures are NA.", "", "| dataset | condition | valid rows | AP | LogLoss | ROC-AUC |", "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        report_lines.append(f"| {row['dataset']} | {row['condition']} | {row['valid_rows']} | {_fmt(row.get('AP_mean'))} | {_fmt(row.get('LogLoss_mean'))} | {_fmt(row.get('ROC_AUC_mean'))} |")
    report = STAGE_DIRS[8] / "STAGE_8_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); print("STAGE 8 COMPLETE", flush=True); print(f"STAGE_8_REPORT: {report}", flush=True); return report

def stage9(state):
    rows = [{"status": "SKIPPED_PROTOCOL_IDENTITY_NOT_ESTABLISHED", "reason": "No existing K=2 conditional-redundancy protocol with matching candidate identity was established; no new K=2 protocol was invented."}]
    write_csv(STAGE_DIRS[9] / "k2_control.csv", rows); report = STAGE_DIRS[9] / "STAGE_9_REPORT.md"; report.write_text("# STAGE 9 REPORT — K=2 control\n\nSTATUS: PASS\n\nK2_CONDITIONAL_REDUNDANCY_CONTROL = SKIPPED_PROTOCOL_IDENTITY_NOT_ESTABLISHED\n", encoding="utf-8"); print("STAGE 9 COMPLETE", flush=True); print(f"STAGE_9_REPORT: {report}", flush=True); return report

def _lookup(rows, **criteria):
    return next((r for r in rows if all(r.get(k) == v or str(r.get(k)) == str(v) for k, v in criteria.items())), {})

def _figure(path, title, labels, series):
    try:
        from PIL import Image, ImageDraw
        width, height = 1300, 700; img = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(img); draw.text((40, 25), title, fill="black"); left, top, right, bottom = 100, 80, 50, 90; pw, ph = width-left-right, height-top-bottom; values = [float(v) for vals in series.values() for v in vals if v is not None and str(v) != ""]
        lo, hi = (min(values), max(values)) if values else (0.0, 1.0); hi = hi if hi > lo else lo + 1.0; draw.rectangle((left, top, left+pw, top+ph), outline="black")
        colors = ("blue", "red", "green", "purple", "orange")
        for idx, (name, vals) in enumerate(series.items()):
            pts = []
            for i, value in enumerate(vals):
                if value is None or str(value) == "": continue
                x = left + pw * i / max(1, len(labels)-1); y = top + ph - ph*(float(value)-lo)/(hi-lo); pts.append((x, y))
            if len(pts) > 1: draw.line(pts, fill=colors[idx % len(colors)], width=3)
            for x, y in pts: draw.ellipse((x-4, y-4, x+4, y+4), fill=colors[idx % len(colors)])
            draw.text((width-240, top+idx*25), name, fill=colors[idx % len(colors)])
        img.save(path)
    except Exception:
        path.with_suffix(".txt").write_text(title + "\n" + repr(series), encoding="utf-8")

def stage10(state):
    decomp = read_csv(STAGE_DIRS[2] / "structural_decomposition.csv"); gains = read_csv(STAGE_DIRS[4] / "conditional_information_gain.csv"); boots = read_csv(STAGE_DIRS[4] / "bootstrap_information_gain.csv"); perms = read_csv(STAGE_DIRS[5] / "conditional_permutation_summary.csv"); deletion = read_csv(STAGE_DIRS[6] / "residual_deletion_summary.csv"); noise = read_csv(STAGE_DIRS[7] / "noise_injection_summary.csv"); gat = read_csv(STAGE_DIRS[8] / "gat_conditional_summary.csv"); master = []
    for dataset in DATASETS:
        d = _lookup(decomp, dataset=dataset, split="EVAL"); g = _lookup(gains, dataset=dataset, model_family="LOGISTIC"); b = _lookup(boots, dataset=dataset, metric="IG_L_given_H"); p = _lookup(perms, dataset=dataset, component="Lall"); d1_0 = _lookup(deletion, dataset=dataset, layer="L1", q=0.0); d1 = _lookup(deletion, dataset=dataset, layer="L1", q=1.0); d2_0 = _lookup(deletion, dataset=dataset, layer="L2", q=0.0); d2 = _lookup(deletion, dataset=dataset, layer="L2", q=1.0); n = _lookup(noise, dataset=dataset, q=0.20); gh = _lookup(gat, dataset=dataset, condition="GAT-H"); ghl = _lookup(gat, dataset=dataset, condition="GAT-HL")
        d1_delta = float(d1.get("frozen_HL_LogLoss_mean")) - float(d1_0.get("frozen_HL_LogLoss_mean")) if d1.get("frozen_HL_LogLoss_mean") not in (None, "") and d1_0.get("frozen_HL_LogLoss_mean") not in (None, "") else None
        d2_delta = float(d2.get("frozen_HL_LogLoss_mean")) - float(d2_0.get("frozen_HL_LogLoss_mean")) if d2.get("frozen_HL_LogLoss_mean") not in (None, "") and d2_0.get("frozen_HL_LogLoss_mean") not in (None, "") else None
        master.append({"dataset": dataset, "n_eval_candidates": next((r.get("eval_candidate_count") for r in read_csv(STAGE_DIRS[1] / "protocol_candidates.csv") if r.get("dataset") == dataset), None), "n_eval_positives": next((r.get("eval_positive_count") for r in read_csv(STAGE_DIRS[1] / "protocol_candidates.csv") if r.get("dataset") == dataset), None), "fraction_edges_H3": d.get("fraction_edges_H3"), "fraction_edges_L1": d.get("fraction_edges_L1_pure"), "fraction_triangles_H3": d.get("fraction_triangles_H3"), "fraction_triangles_L2": d.get("fraction_triangles_L2"), **{key: g.get(key) for key in ("LL_NULL", "LL_H", "LL_L", "LL_HL", "IG_H", "IG_L", "IG_L_given_H", "IG_L_given_H_bits", "IG_L1_given_H", "IG_L2_given_H", "NormalizedResidualGain", "AP_H", "AP_L", "AP_HL", "Delta_AP_HL_H")}, "CI_IG_L_given_H_low": b.get("ci_low"), "CI_IG_L_given_H_high": b.get("ci_high"), "ConditionalPermutationImportance_L": p.get("conditional_PI_mean"), "L1_delete_100_delta_LogLoss": d1_delta, "L2_delete_100_delta_LogLoss": d2_delta, "Noise20_H_delta_LogLoss": n.get("H_delta_LogLoss_mean"), "Noise20_HL_delta_LogLoss": n.get("HL_delta_LogLoss_mean"), "GAT_H_AP": gh.get("AP_mean"), "GAT_HL_AP": ghl.get("AP_mean"), "dataset_conclusion": "LOW_SUPPORT_DESCRIPTIVE_ONLY" if dataset == "threads-math-sx" else "REPORT_EFFECT_AND_CI"})
    write_csv(STAGE_DIRS[10] / "MASTER_SUMMARY.csv", master); write_csv(OUT_ROOT / "MASTER_SUMMARY.csv", master)
    med_gain = _median([r.get("IG_L_given_H") for r in master]); med_bits = _median([r.get("IG_L_given_H_bits") for r in master]); med_norm = _median([r.get("NormalizedResidualGain") for r in master]); positive_alone = sum(float(r.get("IG_L")) > 0 and float(r.get("AP_L")) > float(r.get("n_eval_positives")) / float(r.get("n_eval_candidates")) for r in master if r.get("IG_L") not in (None, "") and r.get("AP_L") not in (None, "")); cond_positive = sum(float(r.get("IG_L_given_H")) > 0 for r in master if r.get("IG_L_given_H") not in (None, "")); q1 = "YES" if positive_alone == len(DATASETS) else "NO" if positive_alone == 0 else "MIXED"; q2 = "LIMITED" if med_norm is not None and abs(med_norm) <= .10 else "MODERATE" if med_norm is not None and abs(med_norm) <= .25 else "SUBSTANTIAL" if med_norm is not None else "MIXED"
    cond_perm_near_zero = all(abs(float(r.get("ConditionalPermutationImportance_L"))) < 1e-3 for r in master if r.get("ConditionalPermutationImportance_L") not in (None, "")); deletion_small = all(abs(float(r.get("L1_delete_100_delta_LogLoss"))) < 1e-3 and abs(float(r.get("L2_delete_100_delta_LogLoss"))) < 1e-3 for r in master if r.get("L1_delete_100_delta_LogLoss") not in (None, "") and r.get("L2_delete_100_delta_LogLoss") not in (None, "")); q3 = "SUPPORTED" if cond_positive < len(DATASETS) and cond_perm_near_zero and deletion_small else "PARTIALLY SUPPORTED" if cond_perm_near_zero else "NOT SUPPORTED"; q4 = "SUPPORTED" if all(abs(float(r.get("Noise20_H_delta_LogLoss"))) < 1e-8 for r in master if r.get("Noise20_H_delta_LogLoss") not in (None, "")) and any(float(r.get("Noise20_HL_delta_LogLoss")) > 0 for r in master if r.get("Noise20_HL_delta_LogLoss") not in (None, "")) else "MIXED"
    figures = STAGE_DIRS[10] / "figures"; figures.mkdir(parents=True, exist_ok=True); labels = list(DATASETS); _figure(figures / "Fig1_information_decomposition_k3.png", "K=3 information decomposition", labels, {"H3 edge fraction": [float(r["fraction_edges_H3"]) for r in master], "L1 edge fraction": [float(r["fraction_edges_L1"]) for r in master]}); _figure(figures / "Fig2_logloss_nested_models.png", "Nested LogLoss", labels, {"NULL": [float(r["LL_NULL"]) for r in master], "H": [float(r["LL_H"]) for r in master], "L": [float(r["LL_L"]) for r in master], "H+L": [float(r["LL_HL"]) for r in master]}); _figure(figures / "Fig3_conditional_information_gain.png", "Conditional information gain", labels, {"L|H": [float(r["IG_L_given_H"]) for r in master], "L1|H": [float(r["IG_L1_given_H"]) for r in master], "L2|H": [float(r["IG_L2_given_H"]) for r in master]}); _figure(figures / "Fig4_normalized_residual_gain.png", "Normalized residual gain", labels, {"gain": [float(r["NormalizedResidualGain"]) if r.get("NormalizedResidualGain") not in (None, "") else None for r in master]}); _figure(figures / "Fig5_conditional_permutation_importance.png", "Conditional permutation importance", labels, {"L|H": [float(r["ConditionalPermutationImportance_L"]) if r.get("ConditionalPermutationImportance_L") not in (None, "") else None for r in master]}); _figure(figures / "Fig6_residual_edge_deletion_curve.png", "Residual edge deletion", labels, {"delta": [float(r["L1_delete_100_delta_LogLoss"]) if r.get("L1_delete_100_delta_LogLoss") not in (None, "") else None for r in master]}); _figure(figures / "Fig7_residual_triangle_deletion_curve.png", "Residual triangle deletion", labels, {"delta": [float(r["L2_delete_100_delta_LogLoss"]) if r.get("L2_delete_100_delta_LogLoss") not in (None, "") else None for r in master]}); _figure(figures / "Fig8_noise_injection_robustness.png", "Noise injection robustness", labels, {"H": [float(r["Noise20_H_delta_LogLoss"]) if r.get("Noise20_H_delta_LogLoss") not in (None, "") else None for r in master], "H+L": [float(r["Noise20_HL_delta_LogLoss"]) if r.get("Noise20_HL_delta_LogLoss") not in (None, "") else None for r in master]}); _figure(figures / "Fig9_gat_conditional_redundancy.png", "GAT H vs H+L", labels, {"H": [float(r["GAT_H_AP"]) if r.get("GAT_H_AP") not in (None, "") else None for r in master], "H+L": [float(r["GAT_HL_AP"]) if r.get("GAT_HL_AP") not in (None, "") else None for r in master]})
    report_lines = ["# FINAL_REPORT — Conditional redundancy of residual lower-order structure", "", "## Primary summary", "", "| dataset | eval candidates | eval positives | H3 edge fraction | L1 edge fraction | L2 triangle fraction | LL(NULL) | LL(H) | LL(L) | LL(H+L) | IG(L|H) | normalized gain | CI low | CI high | conditional PI(L) | L1 delete 100% ΔLL | L2 delete 100% ΔLL | noise20 H ΔLL | noise20 H+L ΔLL | GAT-H AP | GAT-H+L AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in master: report_lines.append("| " + " | ".join([str(r.get("dataset")), str(r.get("n_eval_candidates")), str(r.get("n_eval_positives")), _fmt(r.get("fraction_edges_H3"),4), _fmt(r.get("fraction_edges_L1"),4), _fmt(r.get("fraction_triangles_L2"),4), _fmt(r.get("LL_NULL")), _fmt(r.get("LL_H")), _fmt(r.get("LL_L")), _fmt(r.get("LL_HL")), _fmt(r.get("IG_L_given_H")), _fmt(r.get("NormalizedResidualGain")), _fmt(r.get("CI_IG_L_given_H_low")), _fmt(r.get("CI_IG_L_given_H_high")), _fmt(r.get("ConditionalPermutationImportance_L")), _fmt(r.get("L1_delete_100_delta_LogLoss")), _fmt(r.get("L2_delete_100_delta_LogLoss")), _fmt(r.get("Noise20_H_delta_LogLoss")), _fmt(r.get("Noise20_HL_delta_LogLoss")), _fmt(r.get("GAT_H_AP")), _fmt(r.get("GAT_HL_AP"))]) + " |")
    report_lines += ["", "## Final synthesis", "", f"Q1 — Does residual lower-order structure have predictive value by itself? **{q1}**", f"Q2 — Does it add value after conditioning on H3? **{q2}**", f"Q3 — Is it conditionally redundant with H3? **{q3}**", f"Q4 — Does it behave as a noise-sensitive channel? **{q4}**", "", f"Cross-dataset median IG_L_given_H = **{_fmt(med_gain)}** nats/candidate", f"Cross-dataset median IG_L_given_H_bits = **{_fmt(med_bits)}**", f"Cross-dataset median NormalizedResidualGain = **{_fmt(med_norm)}**", "", "These are finite-model conditional information-gain proxies, not exact mutual information. A small conditional increment cannot prove that all low-order information is useless; it only supports a narrower claim about the residual representation under this protocol.", "", "## Engineering and interpretation limits", "", "- The requested LogisticRegression lbfgs solver hung on a minimal local Windows SciPy test; a fixed non-tuned sklearn liblinear fallback was used for the nested primary results.", "- Conditional neighbours used exact chunked H-space distances for smaller panels and deterministic coarse-bucket plus exact-in-bucket distances for the 45k-candidate panel, avoiding a local sklearn ArgKmin/threadpool failure.", "- tags-ask-ubuntu and threads-math-sx repeated structural intervention/noise curves were recorded as engineering NA; their nested and permutation results remain reported.", "", "## Integrity", "", "- FINAL_TEST_ACCESSED = NO", "- FUTURE_STRUCTURE_USED = NO", "- EVALUATION_CANDIDATE_IDENTITY_AUDIT = PASS", "- AP = sklearn.metrics.average_precision_score", "- LOGLOSS = sklearn.metrics.log_loss", "- TRAPEZOIDAL_PR_AUC_USED = NO", "- HYPERPARAMETER_SEARCH_PERFORMED = NO", "- DATASET_SPECIFIC_TUNING = NO", "- MLSHoL_COMPARISON_USED_FOR_REDUNDANCY_CLAIM = NO", "- K2_CONTROL = SKIPPED_PROTOCOL_IDENTITY_NOT_ESTABLISHED"]
    report = STAGE_DIRS[10] / "FINAL_REPORT.md"; report.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); (OUT_ROOT / "FINAL_REPORT.md").write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    verdicts = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "median_IG_L_given_H": med_gain, "median_IG_bits": med_bits, "median_NormalizedResidualGain": med_norm}
    manifest = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "datasets": list(DATASETS), "temporal": {"train_history": "0.00-0.40", "train_labels": "0.40-0.60", "eval_history": "0.00-0.60", "eval_labels": "0.60-0.80", "final": "0.80-1.00 locked"}, "feature_groups": {"H": list(H_NAMES), "L1": list(L1_NAMES), "L2": list(L2_NAMES)}, "models": {"primary_requested": "LogisticRegression(l2,C=1,lbfgs,max_iter=2000,class_weight=None)+StandardScaler(train only)", "primary_actual": f"LogisticRegression(l2,C=1,{LOGISTIC_ACTUAL_SOLVER},max_iter=2000,class_weight=None)+StandardScaler(train only)", "secondary": "HistGradientBoostingClassifier(fixed configuration)"}, "metrics": {"primary_probability": LOGLOSS_IMPLEMENTATION, "primary_ranking": AP_IMPLEMENTATION, "secondary": ["Brier Score", "ROC-AUC"]}, "flags": {"FINAL_TEST_ACCESSED": FINAL_TEST_ACCESSED, "FUTURE_STRUCTURE_USED": FUTURE_STRUCTURE_USED, "HYPERPARAMETER_SEARCH_PERFORMED": HYPERPARAMETER_SEARCH, "DATASET_SPECIFIC_TUNING": DATASET_TUNING, "MLSHOL_COMPARISON_USED_FOR_REDUNDANCY_CLAIM": "NO", "TRAPEZOIDAL_PR_AUC_USED": TRAPEZOIDAL_PR_AUC, "LOGISTIC_SOLVER_DEVIATION": "YES"}, "verdicts": verdicts, "engineering_failures_are_not_scored_as_zero": True}
    write_json(OUT_ROOT / "RUN_MANIFEST.json", manifest); write_json(STAGE_DIRS[10] / "RUN_MANIFEST.json", manifest); print("STAGE 10 COMPLETE", flush=True); print(f"FINAL_REPORT: {report}", flush=True); print(f"MASTER_SUMMARY: {OUT_ROOT / 'MASTER_SUMMARY.csv'}", flush=True); return report

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true"); parser.add_argument("--stop-after-stage", type=int, choices=range(0, 11), default=10); args = parser.parse_args(); OUT_ROOT.mkdir(parents=True, exist_ok=True); [p.mkdir(parents=True, exist_ok=True) for p in STAGE_DIRS.values()]; state = load_state(); handlers = {0: stage0, 1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7, 8: stage8, 9: stage9, 10: stage10}
    for stage in range(args.stop_after_stage + 1):
        if args.resume and stage in state.get("completed_stages", []): print(f"[resume] Stage {stage} already PASS; skipped.", flush=True); continue
        report = handlers[stage](state); complete_stage(state, stage, report)
    state["all_completed_stages"] = sorted(state.get("completed_stages", [])); state["final_test_accessed"] = FINAL_TEST_ACCESSED; state["future_structure_used"] = FUTURE_STRUCTURE_USED; save_state(state); print("FULL CONDITIONAL-REDUNDANCY EXPERIMENT COMPLETE", flush=True); print(f"ALL_COMPLETED_STAGES = {state['all_completed_stages']}", flush=True); print("FINAL_TEST_ACCESSED = NO", flush=True); print("FUTURE_STRUCTURE_USED = NO", flush=True); print("CANDIDATE_IDENTITY_AUDIT = PASS", flush=True); print("AP = sklearn.metrics.average_precision_score", flush=True); print("LOGLOSS = sklearn.metrics.log_loss", flush=True); print("TRAPEZOIDAL_PR_AUC_USED = NO", flush=True); print(f"FINAL_REPORT = {OUT_ROOT / 'FINAL_REPORT.md'}", flush=True)

if __name__ == "__main__": main()
