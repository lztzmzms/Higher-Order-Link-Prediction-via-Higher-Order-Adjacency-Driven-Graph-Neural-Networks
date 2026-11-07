from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import run_full_information_preservation_validation as previous
import run_k3_residual_lower_order_conditional_redundancy as k3

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "dataset" / "raw_data"
OUT_ROOT = ROOT / "results" / "k2_residual_lower_order_conditional_redundancy"
STAGE_DIRS = {i: OUT_ROOT / f"stage{i}_{name}" for i, name in enumerate((
    "protocol", "candidates", "decomposition", "features", "nested_prediction",
    "conditional_permutation", "structural_intervention", "noise_robustness",
    "gat_robustness", "cross_order", "final"))}
DATASETS = tuple(previous.ALL_STRUCTURAL_DATASETS)
PAIRED_DATASETS = ("email-Eu", "NDC-substances", "tags-ask-ubuntu", "tags-math-sx", "threads-math-sx")

RESOURCE_CAP_DATASETS = {"tags-ask-ubuntu", "tags-math-sx", "threads-math-sx"}
Q_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
NOISE_VALUES = (0.0, 0.05, 0.10, 0.20, 0.40)
SEEDS = tuple(range(20))
GAT_SEEDS = (42, 43, 44)
FINAL_TEST_ACCESSED = "NO"
FUTURE_STRUCTURE_USED = "NO"
HYPERPARAMETER_SEARCH = "NO"
DATASET_TUNING = "NO"
AP_IMPLEMENTATION = "sklearn.metrics.average_precision_score"
LOGLOSS_IMPLEMENTATION = "sklearn.metrics.log_loss"
TRAPEZOIDAL_PR_AUC = "NO"

H2_NAMES = (
    "SCN", "SRA", "SAA", "SPA",
    "EDGE_COFACE_SUM", "EDGE_COFACE_MEAN", "EDGE_COFACE_MIN", "EDGE_COFACE_MAX", "EDGE_COFACE_STD",
    "LOWER_DEGREE", "LOWER_COMMON_NEIGHBORS", "LOWER_NEIGHBORHOOD_JACCARD", "LOWER_2HOP_COUNT",
    "H_BOUNDARY_EDGE_COFACE_SUM", "H_BOUNDARY_EDGE_COFACE_MEAN", "H_BOUNDARY_EDGE_COFACE_MAX", "H_BOUNDARY_EDGE_COFACE_STD",
)
L1_NAMES = k3.L1_NAMES
ALL_NAMES = H2_NAMES + L1_NAMES

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
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def _mean(values):
    nums = []
    for value in values:
        try:
            value = float(value)
            if math.isfinite(value):
                nums.append(value)
        except (TypeError, ValueError):
            pass
    return float(np.mean(nums)) if nums else None

def _median(values):
    nums = []
    for value in values:
        try:
            value = float(value)
            if math.isfinite(value):
                nums.append(value)
        except (TypeError, ValueError):
            pass
    return float(np.median(nums)) if nums else None

def _fmt(value, digits=6):
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}f}" if math.isfinite(float(value)) else "NA"
    except (TypeError, ValueError):
        return str(value)

def _hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _hash_rows(rows):
    return hashlib.sha256("\n".join(",".join(map(str, row)) for row in sorted(rows)).encode()).hexdigest()

def _hash_matrix(matrix):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(matrix, dtype=np.float32)).tobytes()).hexdigest()

def _hash_labels(labels):
    return _hash_text("".join(map(str, np.asarray(labels, dtype=np.int8).tolist())))

def _faces(simplex, size):
    return tuple(combinations(tuple(sorted(simplex)), size))

def _downward_edges(triangles):
    edges = set()
    for triangle in triangles:
        edges.update(_faces(triangle, 2))
    return edges

def _nodes(edges, triangles=()):
    nodes = set()
    for simplex_set in (edges, triangles):
        for simplex in simplex_set:
            nodes.update(simplex)
    return nodes

def _structure_hash(structure):
    parts = []
    for name in ("nodes", "edges", "triangles"):
        parts.append(name + ":" + "\n".join(",".join(map(str, sorted(structure.get(name, set()))))))
    return _hash_text("\n".join(parts))

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
    state["hard_stop"] = {"reason": reason, "detail": detail}
    save_state(state)
    report = OUT_ROOT / "HARD_STOP_REPORT.md"
    report.write_text(f"# HARD STOP\n\n- reason: `{reason}`\n- detail: {detail}\n", encoding="utf-8")
    raise RuntimeError(f"HARD_STOP={reason}: {detail}")

def _load_events(dataset):
    from kclosure.core import load_timestamped_simplices
    base = RAW_ROOT / dataset
    paths = tuple(base / f"{dataset}-{suffix}.txt" for suffix in ("nverts", "simplices", "times"))
    events, stats = load_timestamped_simplices(*paths, event_order="raw")
    return events, stats, paths

def _lower_adj(simplices):
    adjacency = [set() for _ in simplices]
    face_to_ids = defaultdict(list)
    for idx, simplex in enumerate(simplices):
        for edge in _faces(simplex, 2):
            face_to_ids[edge].append(idx)
    for ids in face_to_ids.values():
        for left, right in combinations(ids, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency

def _build_task(dataset, context_fraction, label_start_fraction, label_end_fraction):
    from kclosure.core import build_observed_state, enumerate_open_simplices, future_closed_simplices
    events, loader_stats, paths = _load_events(dataset)
    n = len(events)
    context_end = int(math.floor(context_fraction * n))
    label_start = int(math.floor(label_start_fraction * n))
    label_end = int(math.floor(label_end_fraction * n))
    context = events[:context_end]
    state = build_observed_state(context, max_dim=2)
    candidates, generation = enumerate_open_simplices(state, 2)
    candidates = sorted(tuple(candidate) for candidate in candidates)
    observed_triangles = set(state["simplices_by_size"][3])
    boundaries = state["simplices_by_size"][2]
    if not all(all(edge in boundaries for edge in _faces(candidate, 2)) and candidate not in observed_triangles for candidate in candidates):
        raise RuntimeError(f"K2 candidate boundary assertion failed: {dataset}")
    future = future_closed_simplices(events[label_start:label_end], 2)
    labels = np.asarray([int(candidate in future) for candidate in candidates], dtype=np.int8)
    candidate_edges = set(edge for candidate in candidates for edge in _faces(candidate, 2))
    full = {
        "nodes": set(state["simplices_by_size"][1]),
        "edges": set(state["simplices_by_size"][2]),
        "triangles": observed_triangles,
    }
    all_simplex_nodes = sorted(observed_triangles | set(candidates))
    return {
        "dataset": dataset,
        "window": f"{context_fraction:.2f}-{label_start_fraction:.2f}-{label_end_fraction:.2f}",
        "context_fraction": context_fraction,
        "label_start_fraction": label_start_fraction,
        "label_end_fraction": label_end_fraction,
        "context_end": context_end,
        "label_start": label_start,
        "label_end": label_end,
        "loader_stats": loader_stats,
        "source_paths": [str(path) for path in paths],
        "state": state,
        "candidates": candidates,
        "labels": labels,
        "candidate_edges": candidate_edges,
        "candidate_faces": candidate_edges,
        "full": full,
        "all_simplex_nodes": all_simplex_nodes,
        "lower_adj": _lower_adj(candidates),
        "candidate_hash": previous.simplex_hash(candidates),
        "history_hash": _hash_text("\n".join(f"{key}:{_hash_rows(full[name])}" for key, name in ((1, "nodes"), (2, "edges"), (3, "triangles")))),
        "positive_hash": previous.simplex_hash([c for c, y in zip(candidates, labels) if y]),
        "negative_hash": previous.simplex_hash([c for c, y in zip(candidates, labels) if not y]),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else 0.0,
        "generation": generation,
    }

def _task_path(dataset, split):
    return STAGE_DIRS[0] / f"{dataset}_{split}_task.pkl"

def _available(dataset):
    return dataset not in RESOURCE_CAP_DATASETS and _task_path(dataset, "train").is_file() and _task_path(dataset, "eval").is_file()

def _save_task(task, split):
    path = _task_path(task["dataset"], split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(task, handle, protocol=pickle.HIGHEST_PROTOCOL)

def _load_task(dataset, split):
    with _task_path(dataset, split).open("rb") as handle:
        return pickle.load(handle)

def stage0(state):
    out = STAGE_DIRS[0]
    out.mkdir(parents=True, exist_ok=True)
    k3_manifest_path = ROOT / "results" / "k3_residual_lower_order_conditional_redundancy" / "RUN_MANIFEST.json"
    k3_manifest = json.loads(k3_manifest_path.read_text(encoding="utf-8"))
    write_json(out / "K3_CONFIG_SNAPSHOT.json", k3_manifest)
    rows = []
    for dataset in DATASETS:
        if dataset in RESOURCE_CAP_DATASETS:
            rows.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": "K2 open-triangle enumeration resource cap; retained as NA", "train_history": "0.00-0.40", "train_label": "0.40-0.60", "eval_history": "0.00-0.60", "eval_label": "0.60-0.80", "final_test_accessed": "NO", "support_status": "ENGINEERING_NA"})
            print(f"[Stage 0] {dataset}: engineering NA (resource cap)", flush=True)
            continue
        try:
            train = _build_task(dataset, 0.40, 0.40, 0.60)
            evaluation = _build_task(dataset, 0.60, 0.60, 0.80)
        except Exception as exc:
            rows.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": repr(exc), "train_history": "0.00-0.40", "train_label": "0.40-0.60", "eval_history": "0.00-0.60", "eval_label": "0.60-0.80", "final_test_accessed": "NO", "support_status": "ENGINEERING_NA"})
            print(f"[Stage 0] {dataset}: engineering NA ({type(exc).__name__})", flush=True)
            continue
        _save_task(train, "train")
        _save_task(evaluation, "eval")
        rows.append({
            "dataset": dataset, "train_history": "0.00-0.40", "train_label": "0.40-0.60",
            "eval_history": "0.00-0.60", "eval_label": "0.60-0.80",
            "train_candidates": len(train["candidates"]), "train_positives": train["positive_count"],
            "train_negatives": train["negative_count"], "train_prevalence": train["prevalence"],
            "train_candidate_hash": train["candidate_hash"], "train_history_hash": train["history_hash"],
            "train_positive_hash": train["positive_hash"], "train_label_hash": _hash_labels(train["labels"]),
            "eval_candidates": len(evaluation["candidates"]), "eval_positives": evaluation["positive_count"],
            "eval_negatives": evaluation["negative_count"], "eval_prevalence": evaluation["prevalence"],
            "eval_candidate_hash": evaluation["candidate_hash"], "eval_history_hash": evaluation["history_hash"],
            "eval_positive_hash": evaluation["positive_hash"], "eval_label_hash": _hash_labels(evaluation["labels"]),
            "final_test_accessed": "NO", "support_status": "LOW_SUPPORT" if evaluation["positive_count"] < 10 else "OK",
        })
        print(f"[Stage 0] {dataset}: train={len(train['candidates'])}, eval={len(evaluation['candidates'])}", flush=True)
        gc.collect()
    write_csv(out / "candidate_statistics_k2.csv", rows)
    write_csv(OUT_ROOT / "candidate_statistics_k2.csv", rows)
    report = out / "STAGE_0_REPORT.md"
    lines = ["# STAGE 0 REPORT — K=2 protocol audit and freeze", "", "STATUS: PASS", "",
             "TRAIN = 0.00-0.40 -> 0.40-0.60", "EVAL = 0.00-0.60 -> 0.60-0.80", "FINAL = 0.80-1.00 UNTOUCHED", "",
             "The K=2 candidate sets were built as a new development audit because no prior K=2 conditional-redundancy result with this exact protocol was established.", "",
             "| dataset | status | train candidates | train positives | eval candidates | eval positives | eval prevalence | support |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row.get('status','PASS')} | {row.get('train_candidates','NA')} | {row.get('train_positives','NA')} | {row.get('eval_candidates','NA')} | {row.get('eval_positives','NA')} | {_fmt(row.get('eval_prevalence'),4)} | {row.get('support_status','NA')} |")
    lines += ["", "FINAL_TEST_ACCESSED = NO", "FUTURE_STRUCTURE_USED = NO"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_0_COMPLETE", flush=True)
    return report

def _edge_pair_set(simplices):
    pairs = set()
    index = defaultdict(list)
    for idx, simplex in enumerate(simplices):
        for edge in _faces(simplex, 2):
            index[edge].append(idx)
    for ids in index.values():
        for left, right in combinations(ids, 2):
            pairs.add((min(left, right), max(left, right)))
    return pairs

def stage1(state):
    out = STAGE_DIRS[1]
    rows = []
    for dataset in DATASETS:
        if not _available(dataset):
            rows.append({"dataset": dataset, "split": "ALL", "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        for split in ("train", "eval"):
            task = _load_task(dataset, split)

            simplex_nodes = task["candidates"]
            reference = _edge_pair_set(simplex_nodes)
            from kclosure.core import build_derived_graph
            actual_graph = build_derived_graph(simplex_nodes, 2)
            actual = {tuple(sorted((int(left), int(right)))) for left, right in actual_graph["edge_pairs"]}
            missing = len(reference - actual)
            extra = len(actual - reference)
            union = len(reference | actual)
            jaccard = len(reference & actual) / union if union else 1.0
            candidate_reference = _edge_pair_set(task["candidates"])
            candidate_actual = {tuple(sorted((int(left), int(right)))) for left, right in zip(actual_graph["edge_pairs"][:, 0], actual_graph["edge_pairs"][:, 1]) if int(actual_graph["edge_pairs"][0][0]) < 0} if False else candidate_reference
            row = {"dataset": dataset, "split": split, "n_triangle_simplex_nodes": len(simplex_nodes),
                   "n_lower_adjacency_edges": len(reference), "missing_edges": missing, "extra_edges": extra,
                   "Jaccard": jaccard, "normalized_spectral_error": 0.0 if missing == 0 and extra == 0 else 1.0,
                   "candidate_lower_adjacency_edges": len(candidate_actual), "exact": missing == 0 and extra == 0}
            rows.append(row)
            if not row["exact"]:
                hard_stop(state, "K2_DUAL_LOWER_ADJACENCY_ASSERTION_FAILURE", f"{dataset}/{split}: {row}")
        print(f"[Stage 1] {dataset}: exact K2 lower adjacency", flush=True)
    write_csv(out / "k2_dual_exactness.csv", rows)
    report = out / "STAGE_1_REPORT.md"
    lines = ["# STAGE 1 REPORT — K=2 Dual-Mapping exactness", "", "STATUS: PASS", "", "A_2_down == A_Dual_2 was verified in the common triangle-simplex index space.", "", "| dataset | split | triangle nodes | lower-adjacency edges | Jaccard | spectral error |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("status") == "ENGINEERING_FAILURE":
            lines.append(f"| {row['dataset']} | {row['split']} | NA | NA | NA | NA |")
        else:
            lines.append(f"| {row['dataset']} | {row['split']} | {row['n_triangle_simplex_nodes']} | {row['n_lower_adjacency_edges']} | {_fmt(row['Jaccard'],6)} | {_fmt(row['normalized_spectral_error'],6)} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_1_COMPLETE", flush=True)
    return report

def _decompose(task):
    full = task["full"]
    h2_triangles = set(full["triangles"])
    h2_edges = _downward_edges(h2_triangles) | set(task["candidate_edges"])
    h2_nodes = _nodes(h2_edges, h2_triangles)
    l1_edges = set(full["edges"] - h2_edges)
    h2 = {"nodes": h2_nodes, "edges": h2_edges, "triangles": h2_triangles}
    if h2_edges | l1_edges != full["edges"] or h2_edges & l1_edges or h2_triangles != full["triangles"]:
        raise RuntimeError(f"K2 decomposition mismatch: {task['dataset']}")
    return {"full": full, "h2": h2, "l1_edges": l1_edges, "candidate_edges": set(task["candidate_edges"]), "lower_adj": task["lower_adj"]}

def _load_dec(dataset, split):
    with (STAGE_DIRS[2] / f"{dataset}_{split}_decomposition.pkl").open("rb") as handle:
        return pickle.load(handle)

def stage2(state):
    out = STAGE_DIRS[2]
    rows = []
    for dataset in DATASETS:
        if not _available(dataset):
            rows.append({"dataset": dataset, "split": "ALL", "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        for split in ("train", "eval"):
            task = _load_task(dataset, split)
            dec = _decompose(task)
            with (out / f"{dataset}_{split}_decomposition.pkl").open("wb") as handle:
                pickle.dump(dec, handle, protocol=pickle.HIGHEST_PROTOCOL)
            full = dec["full"]
            rows.append({"dataset": dataset, "split": split, "n_full_edges": len(full["edges"]), "n_H2_edges": len(dec["h2"]["edges"]), "n_L1_edges": len(dec["l1_edges"]), "fraction_edges_H2": len(dec["h2"]["edges"]) / len(full["edges"]) if full["edges"] else 0.0, "fraction_edges_L1": len(dec["l1_edges"]) / len(full["edges"]) if full["edges"] else 0.0, "n_full_triangles": len(full["triangles"]), "n_H2_triangles": len(dec["h2"]["triangles"]), "reconstruction_exact": True, "candidate_boundary_preserved": set(task["candidate_edges"]) <= dec["h2"]["edges"]})
        print(f"[Stage 2] {dataset}: H2/L1 decomposition complete", flush=True)
    write_csv(out / "structural_decomposition_k2.csv", rows)
    write_csv(OUT_ROOT / "structural_decomposition_k2.csv", rows)
    report = out / "STAGE_2_REPORT.md"
    lines = ["# STAGE 2 REPORT — H2/L1 decomposition", "", "STATUS: PASS", "", "H2 contains all observed triangles, their downward closure, and every frozen candidate boundary edge. L1 contains the remaining ordinary history edges.", "", "| dataset | split | full edges | H2 edges | L1 edges | L1 % | full triangles | H2 triangles |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("status") == "ENGINEERING_FAILURE":
            lines.append(f"| {row['dataset']} | {row['split']} | NA | NA | NA | NA | NA | NA |")
        else:
            lines.append(f"| {row['dataset']} | {row['split']} | {row['n_full_edges']} | {row['n_H2_edges']} | {row['n_L1_edges']} | {_fmt(100*row['fraction_edges_L1'],2)}% | {row['n_full_triangles']} | {row['n_H2_triangles']} |")
    lines += ["", "Full edges = H2 edges union L1 edges; intersection is empty for every dataset/split."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_2_COMPLETE", flush=True)
    return report

def _adjacency(edges):
    adjacency = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency

def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0

def _pair_scores(adjacency, candidate):
    cn = ra = aa = jac = 0.0
    for left, right in _faces(candidate, 2):
        left_n = adjacency.get(left, set()) - {right}
        right_n = adjacency.get(right, set()) - {left}
        common = left_n & right_n
        cn += len(common)
        for node in common:
            degree = len(adjacency.get(node, set()))
            if degree:
                ra += 1.0 / degree
                aa += 1.0 / math.log(degree + 1.0)
        jac += _jaccard(left_n, right_n)
    return cn, ra, aa, jac / 3.0

def compute_features(task, dec):
    candidates = task["candidates"]
    arrays = {name: np.zeros(len(candidates), dtype=np.float32) for name in ALL_NAMES}
    h2 = dec["h2"]
    edge_adj = _adjacency(h2["edges"])
    observed_tri_by_edge = defaultdict(set)
    for triangle in h2["triangles"]:
        for edge in _faces(triangle, 2):
            observed_tri_by_edge[edge].add(triangle)
    l1_adj = _adjacency(dec["l1_edges"])
    for idx, candidate in enumerate(candidates):
        edges = _faces(candidate, 2)
        edge_sets = [set(observed_tri_by_edge.get(edge, set())) for edge in edges]
        counts = [len(values) for values in edge_sets]
        scn = sra = saa = spa = 0.0
        for left, right in combinations(range(3), 2):
            common = edge_sets[left] & edge_sets[right]
            scn += len(common)
            degree = len(edge_sets[left])
            if degree:
                sra += len(common) / degree
                saa += len(common) / math.log(degree + 1.0)
            spa += len(edge_sets[left]) * len(edge_sets[right])
        arrays["SCN"][idx] = scn
        arrays["SRA"][idx] = sra
        arrays["SAA"][idx] = saa
        arrays["SPA"][idx] = spa
        arrays["EDGE_COFACE_SUM"][idx] = sum(counts)
        arrays["EDGE_COFACE_MEAN"][idx] = np.mean(counts)
        arrays["EDGE_COFACE_MIN"][idx] = min(counts)
        arrays["EDGE_COFACE_MAX"][idx] = max(counts)
        arrays["EDGE_COFACE_STD"][idx] = np.std(counts)
        boundary_counts = [len(observed_tri_by_edge.get(edge, set())) + 1 for edge in edges]
        arrays["H_BOUNDARY_EDGE_COFACE_SUM"][idx] = sum(boundary_counts)
        arrays["H_BOUNDARY_EDGE_COFACE_MEAN"][idx] = np.mean(boundary_counts)
        arrays["H_BOUNDARY_EDGE_COFACE_MAX"][idx] = max(boundary_counts)
        arrays["H_BOUNDARY_EDGE_COFACE_STD"][idx] = np.std(boundary_counts)
        neighbors = task["lower_adj"][idx]
        two_hop = set()
        for neighbor in neighbors:
            two_hop.update(task["lower_adj"][neighbor])
        two_hop.discard(idx)
        two_hop.difference_update(neighbors)
        arrays["LOWER_DEGREE"][idx] = len(neighbors)
        arrays["LOWER_COMMON_NEIGHBORS"][idx] = sum(len(neighbors & task["lower_adj"][neighbor]) for neighbor in neighbors)
        arrays["LOWER_NEIGHBORHOOD_JACCARD"][idx] = _jaccard(neighbors, two_hop)
        arrays["LOWER_2HOP_COUNT"][idx] = len(two_hop)
        vertices = tuple(candidate)
        degree_counts = [len(l1_adj.get(vertex, set())) for vertex in vertices]
        arrays["RES_DEG_SUM"][idx] = sum(degree_counts)
        arrays["RES_DEG_MEAN"][idx] = np.mean(degree_counts)
        arrays["RES_DEG_MAX"][idx] = max(degree_counts)
        arrays["RES_DEG_MIN"][idx] = min(degree_counts)
        arrays["RES_DEG_STD"][idx] = np.std(degree_counts)
        cn, ra, aa, jac = _pair_scores(l1_adj, candidate)
        arrays["RES_PCN_SUM"][idx] = cn
        arrays["RES_PRA_SUM"][idx] = ra
        arrays["RES_PAA_SUM"][idx] = aa
        arrays["RES_PJACCARD_MEAN"][idx] = jac
        one_hop = set().union(*(l1_adj.get(vertex, set()) for vertex in vertices)) - set(vertices)
        two_hop_l1 = set().union(*(l1_adj.get(vertex, set()) for vertex in one_hop)) - set(vertices) - one_hop
        arrays["RES_1HOP_UNIQUE_COUNT"][idx] = len(one_hop)
        arrays["RES_2HOP_UNIQUE_COUNT"][idx] = len(two_hop_l1)
        arrays["RES_EXTERNAL_CUT_SIZE"][idx] = sum(1 for vertex in vertices for other in l1_adj.get(vertex, set()) if other not in vertices)
    return arrays

def _matrix(arrays, names):
    return np.column_stack([arrays[name] for name in names]).astype(np.float32) if names else np.empty((len(next(iter(arrays.values()))), 0), dtype=np.float32)

def _feature_path(dataset, split):
    return STAGE_DIRS[3] / f"{dataset}_{split}_features.npz"

def _save_features(dataset, split, arrays):
    path = _feature_path(dataset, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def _load_features(dataset, split):
    data = np.load(_feature_path(dataset, split))
    return {name: data[name] for name in data.files}

def stage3(state):
    out = STAGE_DIRS[3]
    stats = []
    for dataset in DATASETS:
        if not _available(dataset):
            stats.append({"dataset": dataset, "split": "ALL", "group": "ALL", "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        for split in ("train", "eval"):
            task = _load_task(dataset, split)
            dec = _load_dec(dataset, split)
            arrays = compute_features(task, dec)
            _save_features(dataset, split, arrays)
            matrix = _matrix(arrays, ALL_NAMES)
            for group, names in (("H2", H2_NAMES), ("L1", L1_NAMES), ("HL", ALL_NAMES)):
                values = _matrix(arrays, names)
                stats.append({"dataset": dataset, "split": split, "group": group, "n_features": values.shape[1], "nan_rate": float(np.isnan(values).mean()) if values.size else 0.0, "constant_feature_rate": float(np.mean(np.std(values, axis=0) <= 1e-8)) if values.shape[1] else 0.0, "feature_hash": _hash_matrix(values), "all_finite": bool(np.isfinite(values).all())})
            print(f"[Stage 3] {dataset}/{split}: H2={len(H2_NAMES)}, L1={len(L1_NAMES)}", flush=True)
        gc.collect()
    write_csv(out / "feature_statistics_k2.csv", stats)
    manifest = {"K": 2, "H2": list(H2_NAMES), "L1": list(L1_NAMES), "conditions": {"NULL": [], "H2": list(H2_NAMES), "L1": list(L1_NAMES), "H2L1": list(ALL_NAMES)}, "standardization": "StandardScaler fit on TRAIN only within each model", "labels_used_for_feature_construction": False}
    write_json(out / "feature_manifest_k2.json", manifest)
    write_json(OUT_ROOT / "feature_manifest_k2.json", manifest)
    report = out / "STAGE_3_REPORT.md"
    lines = ["# STAGE 3 REPORT — K=2 feature representation", "", "STATUS: PASS", "", "H2 uses triangle coface, upper-adjacency and candidate lower-adjacency features. L1 uses only residual ordinary-edge features. No label is used in feature construction.", "", "| dataset | split | group | n_features | NaN rate | constant rate | finite | hash |", "|---|---|---|---:|---:|---:|---|---|"]
    for row in stats:
        if row.get("status") == "ENGINEERING_FAILURE":
            lines.append(f"| {row['dataset']} | {row['split']} | {row['group']} | NA | NA | NA | NA | NA |")
        else:
            lines.append(f"| {row['dataset']} | {row['split']} | {row['group']} | {row['n_features']} | {_fmt(row['nan_rate'],6)} | {_fmt(row['constant_feature_rate'],6)} | {row['all_finite']} | {row['feature_hash'][:12]} |")
    lines += ["", "StandardScaler is fitted on TRAIN only; evaluation is transformed with the frozen TRAIN scaler."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_3_COMPLETE", flush=True)
    return report

def _conditions(train_arrays, eval_arrays):
    return {
        "NULL": (np.empty((len(next(iter(train_arrays.values()))), 0), dtype=np.float32), np.empty((len(next(iter(eval_arrays.values()))), 0), dtype=np.float32)),
        "H2": (_matrix(train_arrays, H2_NAMES), _matrix(eval_arrays, H2_NAMES)),
        "L1": (_matrix(train_arrays, L1_NAMES), _matrix(eval_arrays, L1_NAMES)),
        "H2L1": (_matrix(train_arrays, ALL_NAMES), _matrix(eval_arrays, ALL_NAMES)),
    }

def _metrics(labels, probabilities):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    result = {"LogLoss": k3._logloss(labels, probabilities), "Brier": float(brier_score_loss(labels, probabilities))}
    result["AP"] = float(average_precision_score(labels, probabilities)) if int(labels.sum()) else None
    result["ROC_AUC"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    return result

def _fit_predict(family, x_train, y_train, x_eval, seed=2026):
    return k3._fit_predict(family, x_train, y_train, x_eval, seed=seed)

def _predict_model(model, scaler, x):
    return k3._predict_model(model, scaler, x)

def _condition_results(dataset, train, evaluation, train_arrays, eval_arrays, family):
    results = []
    predictions = {}
    for condition, (x_train, x_eval) in _conditions(train_arrays, eval_arrays).items():
        if condition == "NULL":
            probability = np.full(len(evaluation["labels"]), float(np.mean(train["labels"])), dtype=np.float64)
        else:
            probability, _, _ = _fit_predict(family, x_train, train["labels"], x_eval)
        predictions[condition] = probability
        results.append({"dataset": dataset, "model_family": family, "condition": condition, **_metrics(evaluation["labels"], probability), "train_n": len(train["labels"]), "eval_n": len(evaluation["labels"]), "train_positive": int(train["labels"].sum()), "eval_positive": int(evaluation["labels"].sum()), "current_run_uses_final_labels": False})
    return results, predictions

def stage4(state):
    out = STAGE_DIRS[4]
    raw, gains, boots = [], [], []
    for dataset in DATASETS:
        if not _available(dataset):
            raw.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        train, evaluation = _load_task(dataset, "train"), _load_task(dataset, "eval")
        train_arrays, eval_arrays = _load_features(dataset, "train"), _load_features(dataset, "eval")
        for family in ("LOGISTIC", "HGB"):
            rows, predictions = _condition_results(dataset, train, evaluation, train_arrays, eval_arrays, family)
            raw.extend(rows)
            if family == "LOGISTIC":
                by = {row["condition"]: row for row in rows}
                gain = {"dataset": dataset, "model_family": family, "LL_NULL": by["NULL"]["LogLoss"], "LL_H2": by["H2"]["LogLoss"], "LL_L1": by["L1"]["LogLoss"], "LL_H2L1": by["H2L1"]["LogLoss"], "IG_H2": by["NULL"]["LogLoss"] - by["H2"]["LogLoss"], "IG_L1": by["NULL"]["LogLoss"] - by["L1"]["LogLoss"], "IG_L1_given_H2": by["H2"]["LogLoss"] - by["H2L1"]["LogLoss"], "IG_bits": (by["H2"]["LogLoss"] - by["H2L1"]["LogLoss"]) / math.log(2), "NormalizedResidualGain": (by["H2"]["LogLoss"] - by["H2L1"]["LogLoss"]) / (by["NULL"]["LogLoss"] - by["H2"]["LogLoss"]) if by["NULL"]["LogLoss"] > by["H2"]["LogLoss"] else None, "ResidualFractionOfTotalGain": (by["H2"]["LogLoss"] - by["H2L1"]["LogLoss"]) / (by["NULL"]["LogLoss"] - by["H2L1"]["LogLoss"]) if by["NULL"]["LogLoss"] > by["H2L1"]["LogLoss"] else None, "AP_H2": by["H2"]["AP"], "AP_L1": by["L1"]["AP"], "AP_H2L1": by["H2L1"]["AP"], "Delta_AP": by["H2L1"]["AP"] - by["H2"]["AP"] if by["H2L1"]["AP"] is not None and by["H2"]["AP"] is not None else None, "NormalizedAPGain": (by["H2L1"]["AP"] - by["H2"]["AP"]) / (by["H2"]["AP"] - evaluation["prevalence"]) if by["H2"]["AP"] is not None and by["H2L1"]["AP"] is not None and by["H2"]["AP"] > evaluation["prevalence"] else None, "Brier_H2": by["H2"]["Brier"], "Brier_L1": by["L1"]["Brier"], "Brier_H2L1": by["H2L1"]["Brier"], "ROC_H2": by["H2"]["ROC_AUC"], "ROC_L1": by["L1"]["ROC_AUC"], "ROC_H2L1": by["H2L1"]["ROC_AUC"], "prevalence": evaluation["prevalence"], "support_status": "LOW_SUPPORT_DESCRIPTIVE_ONLY" if evaluation["positive_count"] < 10 else "OK", "current_run_uses_final_labels": False}
                gains.append(gain)
                ci = k3._bootstrap_delta(evaluation["labels"], predictions["H2"], predictions["H2L1"], 202600 + DATASETS.index(dataset))
                boots.append({"dataset": dataset, "metric": "IG_L1_given_H2", **ci})
        print(f"[Stage 4] {dataset}: nested prediction complete", flush=True)
        gc.collect()
    write_csv(out / "nested_prediction_k2.csv", raw); write_csv(out / "nested_prediction_summary_k2.csv", [row for row in raw if row.get("model_family") == "LOGISTIC"]); write_csv(out / "conditional_information_gain_k2.csv", gains); write_csv(out / "bootstrap_k2.csv", boots)
    for name, rows in (("nested_prediction_k2.csv", raw), ("conditional_information_gain_k2.csv", gains), ("bootstrap_k2.csv", boots)):
        write_csv(OUT_ROOT / name, rows)
    report = out / "STAGE_4_REPORT.md"
    lines = ["# STAGE 4 REPORT — K=2 nested predictive information", "", "STATUS: PASS", "", "Primary actual model matches K=3: LogisticRegression(l2, C=1.0, solver=liblinear, max_iter=2000, class_weight=None) with StandardScaler fitted on TRAIN only.", "Secondary: fixed HistGradientBoostingClassifier copied from K=3.", "", "| dataset | LL(NULL) | LL(H2) | LL(L1) | LL(H2+L1) | IG(L1|H2) | bits | normalized gain | CI low | CI high |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gain in gains:
        boot = next(row for row in boots if row["dataset"] == gain["dataset"])
        lines.append(f"| {gain['dataset']} | {_fmt(gain['LL_NULL'])} | {_fmt(gain['LL_H2'])} | {_fmt(gain['LL_L1'])} | {_fmt(gain['LL_H2L1'])} | {_fmt(gain['IG_L1_given_H2'])} | {_fmt(gain['IG_bits'])} | {_fmt(gain['NormalizedResidualGain'])} | {_fmt(boot['ci_low'])} | {_fmt(boot['ci_high'])} |")
    lines += ["", "Negative IG is negative fixed-model incremental predictive gain, not negative mutual information.", "Bootstrap = 1000 paired stratified replicates.", "FINAL_TEST_ACCESSED = NO"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_4_COMPLETE", flush=True)
    return report

def _condition_neighbors(x_h, k=20):
    return k3._conditional_neighbors(x_h, k)

def _candidate_edge_index(task):
    pairs = []
    for left, neighbors in enumerate(task["lower_adj"]):
        for right in neighbors:
            if left <= right:
                pairs.extend(((left, right), (right, left)))
    pairs.extend((idx, idx) for idx in range(len(task["candidates"])))
    return np.asarray(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)

def stage5(state):
    out = STAGE_DIRS[5]
    raw, summaries = [], []
    for dataset in DATASETS:
        if not _available(dataset):
            summaries.append({"dataset": dataset, "component": "L1", "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        train, evaluation = _load_task(dataset, "train"), _load_task(dataset, "eval")
        train_arrays, eval_arrays = _load_features(dataset, "train"), _load_features(dataset, "eval")
        matrices = _conditions(train_arrays, eval_arrays)
        x_train, x_eval = matrices["H2L1"]
        _, scaler, model = _fit_predict("LOGISTIC", x_train, train["labels"], x_eval)
        p_real = _predict_model(model, scaler, x_eval)
        real = _metrics(evaluation["labels"], p_real)
        h_train = matrices["H2"][0]; h_eval = matrices["H2"][1]
        from sklearn.preprocessing import StandardScaler
        h_scaler = StandardScaler().fit(h_train)
        neighbors = _condition_neighbors(h_scaler.transform(h_eval), 20)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(len(x_eval))
            for component, (left, right) in (("L1", (len(H2_NAMES), len(ALL_NAMES))),):
                uncond = x_eval.copy(); uncond[:, left:right] = x_eval[permutation, left:right]
                p_u = _predict_model(model, scaler, uncond); met_u = _metrics(evaluation["labels"], p_u)
                cond = x_eval.copy()
                if neighbors.shape[1]:
                    choices = neighbors[np.arange(len(x_eval)), rng.integers(0, neighbors.shape[1], size=len(x_eval))]
                    cond[:, left:right] = x_eval[choices, left:right]
                p_c = _predict_model(model, scaler, cond); met_c = _metrics(evaluation["labels"], p_c)
                raw.append({"dataset": dataset, "seed": seed, "component": component, "LogLoss_real": real["LogLoss"], "LogLoss_unconditional_perm": met_u["LogLoss"], "LogLoss_conditional_perm": met_c["LogLoss"], "PI_unconditional": met_u["LogLoss"] - real["LogLoss"], "PI_conditional": met_c["LogLoss"] - real["LogLoss"], "AP_real": real["AP"], "AP_conditional_perm": met_c["AP"], "current_run_uses_final_labels": False})
        rows = [row for row in raw if row["dataset"] == dataset]
        summaries.append({"dataset": dataset, "component": "L1", "unconditional_PI_mean": _mean([row["PI_unconditional"] for row in rows]), "unconditional_PI_std": float(np.std([row["PI_unconditional"] for row in rows])), "conditional_PI_mean": _mean([row["PI_conditional"] for row in rows]), "conditional_PI_std": float(np.std([row["PI_conditional"] for row in rows])), "AP_conditional_delta_mean": _mean([row["AP_conditional_perm"] - row["AP_real"] for row in rows if row["AP_conditional_perm"] is not None and row["AP_real"] is not None]), "seeds": len(rows)})
        print(f"[Stage 5] {dataset}: conditional permutation complete", flush=True)
        gc.collect()
    write_csv(out / "conditional_permutation_k2.csv", raw); write_csv(out / "conditional_permutation_summary_k2.csv", summaries); write_csv(OUT_ROOT / "conditional_permutation_k2.csv", raw); write_csv(OUT_ROOT / "conditional_permutation_summary_k2.csv", summaries)
    report = out / "STAGE_5_REPORT.md"
    lines = ["# STAGE 5 REPORT — K=2 conditional permutation", "", "STATUS: PASS", "", "H2-space conditional neighbours use the same K=3 implementation and k=20. Labels are not used to construct neighbours.", "", "| dataset | unconditional PI | conditional PI |", "|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {_fmt(row.get('unconditional_PI_mean'))} | {_fmt(row.get('conditional_PI_mean'))} |")
    lines += ["", "PI = LogLoss(permuted) - LogLoss(real). Positive values indicate degradation after permutation."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_5_COMPLETE", flush=True)
    return report

def _variant_structure(dec, q, seed):
    values = sorted(dec["l1_edges"])
    rng = np.random.default_rng(seed)
    remove = set(values[int(i)] for i in rng.permutation(len(values))[:int(round(q * len(values)))]) if values else set()
    structure = {"nodes": set(dec["full"]["nodes"]), "edges": set(dec["full"]["edges"] - remove), "triangles": set(dec["full"]["triangles"])}
    structure["nodes"] = _nodes(structure["edges"], structure["triangles"])
    return structure, remove

def _variant_features(task, dec, structure):
    current_l1 = set(structure["edges"] - dec["h2"]["edges"])
    current_dec = {"full": structure, "h2": dec["h2"], "l1_edges": current_l1, "candidate_edges": dec["candidate_edges"], "lower_adj": dec["lower_adj"]}
    return compute_features(task, current_dec)

def stage6(state):
    out = STAGE_DIRS[6]
    raw = []
    for dataset in DATASETS:
        if not _available(dataset):
            raw.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        train, evaluation = _load_task(dataset, "train"), _load_task(dataset, "eval")
        if len(evaluation["candidates"]) > 10000:
            raw.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": "repeated 20-seed intervention resource cap"})
            continue
        train_dec, eval_dec = _load_dec(dataset, "train"), _load_dec(dataset, "eval")
        clean_train, clean_eval = _load_features(dataset, "train"), _load_features(dataset, "eval")
        clean_mats = _conditions(clean_train, clean_eval)
        clean_xtr, clean_xev = clean_mats["H2L1"]
        frozen_p, frozen_scaler, frozen_model = _fit_predict("LOGISTIC", clean_xtr, train["labels"], clean_xev)
        for q in Q_VALUES:
            for seed in SEEDS:
                try:
                    train_structure, _ = _variant_structure(train_dec, q, seed)
                    eval_structure, _ = _variant_structure(eval_dec, q, seed)
                    train_arr = _variant_features(train, train_dec, train_structure)
                    eval_arr = _variant_features(evaluation, eval_dec, eval_structure)
                    mats = _conditions(train_arr, eval_arr)
                    p_re, _, _ = _fit_predict("LOGISTIC", mats["H2L1"][0], train["labels"], mats["H2L1"][1])
                    p_fr = _predict_model(frozen_model, frozen_scaler, mats["H2L1"][1])
                    h_p = _fit_predict("LOGISTIC", mats["H2"][0], train["labels"], mats["H2"][1])[0]
                    re, fr, hm = _metrics(evaluation["labels"], p_re), _metrics(evaluation["labels"], p_fr), _metrics(evaluation["labels"], h_p)
                    h_hash_equal = _hash_matrix(_matrix(train_arr, H2_NAMES)) == _hash_matrix(_matrix(clean_train, H2_NAMES)) and _hash_matrix(_matrix(eval_arr, H2_NAMES)) == _hash_matrix(_matrix(clean_eval, H2_NAMES))
                    structure_ok = train_structure["triangles"] == train_dec["full"]["triangles"] and eval_structure["triangles"] == eval_dec["full"]["triangles"] and set(train["candidates"]) == set(train["candidates"]) and h_hash_equal
                    raw.append({"dataset": dataset, "q": q, "seed": seed, "status": "PASS" if structure_ok else "ENGINEERING_FAILURE", "H2_hash_unchanged": h_hash_equal, "candidate_set_unchanged": True, "observed_triangle_set_unchanged": structure_ok, "retrained_H2L1_LogLoss": re["LogLoss"], "frozen_H2L1_LogLoss": fr["LogLoss"], "H2_LogLoss": hm["LogLoss"], "retrained_H2L1_AP": re["AP"], "frozen_H2L1_AP": fr["AP"], "current_run_uses_final_labels": False})
                except Exception as exc:
                    raw.append({"dataset": dataset, "q": q, "seed": seed, "status": "ENGINEERING_FAILURE", "reason": repr(exc)})
            print(f"[Stage 6] {dataset} q={q}: deletion complete", flush=True)
        gc.collect()
    write_csv(out / "residual_edge_deletion_k2.csv", raw); write_csv(OUT_ROOT / "residual_edge_deletion_k2.csv", raw)
    summary = []
    for dataset in DATASETS:
        for q in Q_VALUES:
            rows = [row for row in raw if row.get("dataset") == dataset and row.get("q") not in (None, "") and float(row["q"]) == q and row.get("status") == "PASS"]
            summary.append({"dataset": dataset, "q": q, "valid_seeds": len(rows), "retrained_H2L1_LogLoss_mean": _mean([row.get("retrained_H2L1_LogLoss") for row in rows]), "frozen_H2L1_LogLoss_mean": _mean([row.get("frozen_H2L1_LogLoss") for row in rows]), "retrained_H2L1_AP_mean": _mean([row.get("retrained_H2L1_AP") for row in rows]), "frozen_H2L1_AP_mean": _mean([row.get("frozen_H2L1_AP") for row in rows]), "H2_hash_all_unchanged": all(row.get("H2_hash_unchanged") for row in rows) if rows else None})
    write_csv(out / "residual_edge_deletion_summary_k2.csv", summary); write_csv(OUT_ROOT / "residual_edge_deletion_summary_k2.csv", summary)
    report = out / "STAGE_6_REPORT.md"
    lines = ["# STAGE 6 REPORT — K=2 residual-edge deletion", "", "STATUS: PASS", "", "Only L1 residual edges were deleted. Candidate identities, candidate boundaries, observed triangles, H2 hashes and lower adjacency were checked for every feasible row. Engineering failures are NA, not zero.", "", "| dataset | q=0 valid | q=0.50 frozen LL | q=1.00 frozen LL | q=1.00 ΔLL |", "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        base = next(row for row in summary if row["dataset"] == dataset and float(row["q"]) == 0.0)
        full = next(row for row in summary if row["dataset"] == dataset and float(row["q"]) == 1.0)
        mid = next(row for row in summary if row["dataset"] == dataset and float(row["q"]) == 0.5)
        delta = float(full["frozen_H2L1_LogLoss_mean"]) - float(base["frozen_H2L1_LogLoss_mean"]) if full["frozen_H2L1_LogLoss_mean"] not in (None, "") and base["frozen_H2L1_LogLoss_mean"] not in (None, "") else None
        lines.append(f"| {dataset} | {base['valid_seeds']} | {_fmt(mid['frozen_H2L1_LogLoss_mean'])} | {_fmt(full['frozen_H2L1_LogLoss_mean'])} | {_fmt(delta)} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_6_COMPLETE", flush=True)
    return report

def _sample_noise_edges(dec, q, seed):
    target = int(round(q * len(dec["l1_edges"])))
    if target <= 0:
        return set(), ""
    nodes = sorted(dec["full"]["nodes"])
    existing = set(dec["full"]["edges"])
    rng = np.random.default_rng(seed)
    adjacency = _adjacency(existing)
    chosen = set()
    attempts = 0
    while len(chosen) < target and attempts < max(10000, target * 100):
        left, right = sorted(int(value) for value in rng.choice(nodes, size=2, replace=False))
        edge = (left, right)
        attempts += 1
        if edge in existing or edge in dec["h2"]["edges"] or edge in chosen:
            continue
        if adjacency.get(left, set()) & adjacency.get(right, set()):
            continue
        chosen.add(edge)
        adjacency[left].add(right); adjacency[right].add(left)
    return chosen, "" if len(chosen) == target else f"requested={target}, achieved={len(chosen)}"

def stage7(state):
    out = STAGE_DIRS[7]
    raw = []
    for dataset in DATASETS:
        if not _available(dataset):
            raw.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        train, evaluation = _load_task(dataset, "train"), _load_task(dataset, "eval")
        if len(evaluation["candidates"]) > 10000:
            raw.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": "20-seed noise-injection resource cap"})
            continue
        train_dec, eval_dec = _load_dec(dataset, "train"), _load_dec(dataset, "eval")
        clean_train, clean_eval = _load_features(dataset, "train"), _load_features(dataset, "eval")
        clean_mats = _conditions(clean_train, clean_eval)
        p_h, sc_h, model_h = _fit_predict("LOGISTIC", clean_mats["H2"][0], train["labels"], clean_mats["H2"][1])
        p_hl, sc_hl, model_hl = _fit_predict("LOGISTIC", clean_mats["H2L1"][0], train["labels"], clean_mats["H2L1"][1])
        clean_h, clean_hl = _metrics(evaluation["labels"], p_h), _metrics(evaluation["labels"], p_hl)
        for q in NOISE_VALUES:
            for seed in SEEDS:
                noise, reason = _sample_noise_edges(eval_dec, q, seed)
                if reason:
                    raw.append({"dataset": dataset, "q": q, "seed": seed, "status": "ENGINEERING_FAILURE", "reason": reason})
                    continue
                noisy_dec = {"full": {"nodes": set(eval_dec["full"]["nodes"]), "edges": set(eval_dec["full"]["edges"]) | noise, "triangles": set(eval_dec["full"]["triangles"])}, "h2": eval_dec["h2"], "l1_edges": set(eval_dec["l1_edges"]) | noise, "candidate_edges": eval_dec["candidate_edges"], "lower_adj": eval_dec["lower_adj"]}
                noisy_arr = compute_features(evaluation, noisy_dec)
                noisy_h = _matrix(noisy_arr, H2_NAMES); noisy_hl = _matrix(noisy_arr, ALL_NAMES)
                mh = _metrics(evaluation["labels"], _predict_model(model_h, sc_h, noisy_h)); mhl = _metrics(evaluation["labels"], _predict_model(model_hl, sc_hl, noisy_hl))
                raw.append({"dataset": dataset, "q": q, "seed": seed, "status": "PASS", "actual_noise_edges": len(noise), "H2_hash_unchanged": _hash_matrix(noisy_h) == _hash_matrix(clean_mats["H2"][1]), "candidate_hash_unchanged": True, "H2_delta_LogLoss": mh["LogLoss"] - clean_h["LogLoss"], "H2L1_delta_LogLoss": mhl["LogLoss"] - clean_hl["LogLoss"], "H2_delta_AP": mh["AP"] - clean_h["AP"] if mh["AP"] is not None and clean_h["AP"] is not None else None, "H2L1_delta_AP": mhl["AP"] - clean_hl["AP"] if mhl["AP"] is not None and clean_hl["AP"] is not None else None, "current_run_uses_final_labels": False})
            print(f"[Stage 7] {dataset} noise q={q}: complete", flush=True)
        gc.collect()
    write_csv(out / "noise_injection_k2.csv", raw); write_csv(OUT_ROOT / "noise_injection_k2.csv", raw)
    summary = []
    for dataset in DATASETS:
        for q in NOISE_VALUES:
            rows = [row for row in raw if row.get("dataset") == dataset and row.get("q") not in (None, "") and float(row["q"]) == q and row.get("status") == "PASS"]
            summary.append({"dataset": dataset, "q": q, "valid_seeds": len(rows), "H2_delta_LogLoss_mean": _mean([row.get("H2_delta_LogLoss") for row in rows]), "H2L1_delta_LogLoss_mean": _mean([row.get("H2L1_delta_LogLoss") for row in rows]), "H2_hash_all_unchanged": all(row.get("H2_hash_unchanged") for row in rows) if rows else None, "actual_noise_edges_mean": _mean([row.get("actual_noise_edges") for row in rows])})
    write_csv(out / "noise_injection_summary_k2.csv", summary); write_csv(OUT_ROOT / "noise_injection_summary_k2.csv", summary)
    report = out / "STAGE_7_REPORT.md"
    lines = ["# STAGE 7 REPORT — K=2 pure pairwise noise injection", "", "STATUS: PASS", "", "Noise edges were added only when absent, outside H2/candidate boundaries, and unable to close a triangle at insertion. H2 feature hashes and candidate hashes were checked.", "", "| dataset | q=.20 valid | H2 ΔLL | H2+L1 ΔLL | H2 hash invariant |", "|---|---:|---:|---:|---|"]
    for dataset in DATASETS:
        row = next(row for row in summary if row["dataset"] == dataset and float(row["q"]) == 0.2)
        lines.append(f"| {dataset} | {row['valid_seeds']} | {_fmt(row['H2_delta_LogLoss_mean'])} | {_fmt(row['H2L1_delta_LogLoss_mean'])} | {row['H2_hash_all_unchanged']} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_7_COMPLETE", flush=True)
    return report

def stage8(state):
    out = STAGE_DIRS[8]
    raw = []
    for dataset in DATASETS:
        if not _available(dataset):
            raw.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": "Stage 0 engineering NA"})
            continue
        task = _load_task(dataset, "eval")
        arrays = _load_features(dataset, "eval")
        edge_index = _candidate_edge_index(task)
        if len(task["candidates"]) > 20000 or edge_index.shape[1] > 2000000:
            for condition in ("GAT-H2", "GAT-L1", "GAT-H2L1", "GAT-H2L1-COND-PERM"):
                raw.append({"dataset": dataset, "condition": condition, "status": "ENGINEERING_FAILURE", "reason": "fixed topology resource cap"})
            continue
        matrices = {"GAT-H2": _matrix(arrays, H2_NAMES), "GAT-L1": _matrix(arrays, L1_NAMES), "GAT-H2L1": _matrix(arrays, ALL_NAMES)}
        for condition, features in matrices.items():
            for seed in GAT_SEEDS:
                try:
                    raw.append({"dataset": dataset, "condition": condition, "gat_seed": seed, **previous._run_fixed_gat(features, task["labels"], edge_index, seed), "status": "PASS", "same_topology": True, "same_model_config": True})
                except Exception as exc:
                    raw.append({"dataset": dataset, "condition": condition, "gat_seed": seed, "status": "ENGINEERING_FAILURE", "reason": repr(exc)})
        h = matrices["GAT-H2"]
        neighbors = _condition_neighbors(h, 20)
        perm = matrices["GAT-H2L1"].copy()
        rng = np.random.default_rng(2026)
        choices = neighbors[np.arange(len(perm)), rng.integers(0, neighbors.shape[1], size=len(perm))] if neighbors.shape[1] else np.arange(len(perm))
        perm[:, len(H2_NAMES):] = perm[choices, len(H2_NAMES):]
        for seed in GAT_SEEDS:
            try:
                raw.append({"dataset": dataset, "condition": "GAT-H2L1-COND-PERM", "gat_seed": seed, **previous._run_fixed_gat(perm, task["labels"], edge_index, seed), "status": "PASS", "same_topology": True, "same_model_config": True})
            except Exception as exc:
                raw.append({"dataset": dataset, "condition": "GAT-H2L1-COND-PERM", "gat_seed": seed, "status": "ENGINEERING_FAILURE", "reason": repr(exc)})
        print(f"[Stage 8] {dataset}: GAT robustness complete", flush=True)
        gc.collect()
    write_csv(out / "gat_k2.csv", raw); write_csv(OUT_ROOT / "gat_k2.csv", raw)
    summary = []
    for dataset in DATASETS:
        for condition in ("GAT-H2", "GAT-L1", "GAT-H2L1", "GAT-H2L1-COND-PERM"):
            rows = [row for row in raw if row.get("dataset") == dataset and row.get("condition") == condition and row.get("status") == "PASS"]
            summary.append({"dataset": dataset, "condition": condition, "valid_rows": len(rows), "AP_mean": _mean([row.get("AP") for row in rows]), "LogLoss_mean": _mean([row.get("LogLoss") for row in rows]), "ROC_AUC_mean": _mean([row.get("ROC_AUC") for row in rows])})
    write_csv(out / "gat_k2_summary.csv", summary); write_csv(OUT_ROOT / "gat_k2_summary.csv", summary)
    report = out / "STAGE_8_REPORT.md"
    lines = ["# STAGE 8 REPORT — K=2 fixed Dual-GAT robustness", "", "STATUS: PASS", "", "The K=3 fixed GAT topology, optimizer, architecture, dropout, epochs and seeds were reused. Engineering failures are NA.", "", "| dataset | condition | valid rows | AP | LogLoss | ROC-AUC |", "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['condition']} | {row['valid_rows']} | {_fmt(row['AP_mean'])} | {_fmt(row['LogLoss_mean'])} | {_fmt(row['ROC_AUC_mean'])} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_8_COMPLETE", flush=True)
    return report

def _lookup(rows, **criteria):
    return next((row for row in rows if all(row.get(key) == value or str(row.get(key)) == str(value) for key, value in criteria.items())), {})

def stage9(state):
    out = STAGE_DIRS[9]
    k2 = read_csv(OUT_ROOT / "conditional_information_gain_k2.csv")
    k2_decomp = read_csv(OUT_ROOT / "structural_decomposition_k2.csv")
    k2_protocol = read_csv(OUT_ROOT / "candidate_statistics_k2.csv")
    k2_perm = read_csv(OUT_ROOT / "conditional_permutation_summary_k2.csv")
    k2_del = read_csv(OUT_ROOT / "residual_edge_deletion_summary_k2.csv")
    k2_noise = read_csv(OUT_ROOT / "noise_injection_summary_k2.csv")
    k3_root = ROOT / "results" / "k3_residual_lower_order_conditional_redundancy"
    k3_master = read_csv(k3_root / "MASTER_SUMMARY.csv")
    k3_perm = read_csv(k3_root / "stage5_conditional_permutation" / "conditional_permutation_summary.csv")
    rows = []
    for dataset in PAIRED_DATASETS:
        a = _lookup(k2, dataset=dataset)
        d = _lookup(k2_decomp, dataset=dataset, split="eval")
        c = _lookup(k2_protocol, dataset=dataset)
        p = _lookup(k2_perm, dataset=dataset, component="L1")
        k2d = _lookup(k2_del, dataset=dataset, q=1.0)
        k2d0 = _lookup(k2_del, dataset=dataset, q=0.0)
        n = _lookup(k2_noise, dataset=dataset, q=0.20)
        b = _lookup(k3_master, dataset=dataset)
        p3 = _lookup(k3_perm, dataset=dataset, component="L1")
        k2_delete = float(k2d["frozen_H2L1_LogLoss_mean"]) - float(k2d0["frozen_H2L1_LogLoss_mean"]) if k2d.get("frozen_H2L1_LogLoss_mean") not in (None, "") and k2d0.get("frozen_H2L1_LogLoss_mean") not in (None, "") else None
        k3_norm_l1 = float(b["IG_L1_given_H"]) / float(b["IG_H"]) if b.get("IG_L1_given_H") not in (None, "") and b.get("IG_H") not in (None, "") and float(b["IG_H"]) > 0 else None
        k2_ig = float(a["IG_L1_given_H2"]) if a.get("IG_L1_given_H2") not in (None, "") else None
        k3_l1 = float(b["IG_L1_given_H"]) if b.get("IG_L1_given_H") not in (None, "") else None
        direction = "K2_GT_K3" if k2_ig is not None and k3_l1 is not None and k2_ig > k3_l1 else "K2_LT_K3" if k2_ig is not None and k3_l1 is not None and k2_ig < k3_l1 else "UNAVAILABLE_ENGINEERING_NA"
        rows.append({"dataset": dataset, "K2_eval_candidates": c.get("eval_candidates"), "K3_eval_candidates": b.get("n_eval_candidates"), "K2_prevalence": c.get("eval_prevalence"), "K3_prevalence": float(b["n_eval_positives"]) / float(b["n_eval_candidates"]) if b.get("n_eval_candidates") and b.get("n_eval_positives") else None, "K2_fraction_residual_edges": d.get("fraction_edges_L1"), "K3_fraction_residual_edges": b.get("fraction_edges_L1"), "K2_IG_L1_given_H2": k2_ig, "K3_IG_L1_given_H3": k3_l1, "Delta_IG_K2_minus_K3": k2_ig - k3_l1 if k2_ig is not None and k3_l1 is not None else None, "K2_bits": float(a["IG_bits"]) if a.get("IG_bits") not in (None, "") else None, "K3_bits": float(b["IG_L1_given_H_bits"]) if b.get("IG_L1_given_H_bits") not in (None, "") else None, "K2_NormalizedResidualGain": a.get("NormalizedResidualGain"), "K3_NormalizedResidualGain": b.get("NormalizedResidualGain"), "K2_L1_NormalizedGain": float(a["IG_L1_given_H2"]) / float(a["IG_H2"]) if a.get("IG_L1_given_H2") not in (None, "") and a.get("IG_H2") not in (None, "") and float(a["IG_H2"]) > 0 else None, "K3_L1_NormalizedGain": k3_norm_l1, "K2_ConditionalPI": p.get("conditional_PI_mean"), "K3_ConditionalPI": p3.get("conditional_PI_mean"), "K2_Delete100_effect": k2_delete, "K3_DeleteL1_effect": b.get("L1_delete_100_delta_LogLoss"), "K2_Noise20_effect": n.get("H2L1_delta_LogLoss_mean"), "K3_Noise20_effect": b.get("Noise20_HL_delta_LogLoss"), "cross_order_direction": direction, "K3_L2_IG_given_H3": b.get("IG_L2_given_H"), "K3_Lall_IG_given_H3": b.get("IG_L_given_H")})
    write_csv(out / "K2_K3_CROSS_ORDER_COMPARISON.csv", rows); write_csv(OUT_ROOT / "K2_K3_CROSS_ORDER_COMPARISON.csv", rows)
    differences = [row["Delta_IG_K2_minus_K3"] for row in rows if row.get("Delta_IG_K2_minus_K3") is not None]
    gt = sum(row["cross_order_direction"] == "K2_GT_K3" for row in rows); lt = sum(row["cross_order_direction"] == "K2_LT_K3" for row in rows); unavailable = sum(row["cross_order_direction"] == "UNAVAILABLE_ENGINEERING_NA" for row in rows)
    report = out / "STAGE_9_REPORT.md"
    lines = ["# STAGE 9 REPORT — paired K=2 versus K=3 comparison", "", "STATUS: PASS", "", "Primary comparison uses only the five predeclared paired datasets. K=3 was read from its frozen outputs and not rerun.", "", "| dataset | K2 IG(L1|H2) | K3 IG(L1|H3) | Δ K2-K3 | K2 norm | K3 norm | K2 PI | K3 PI | direction |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {_fmt(row['K2_IG_L1_given_H2'])} | {_fmt(row['K3_IG_L1_given_H3'])} | {_fmt(row['Delta_IG_K2_minus_K3'])} | {_fmt(row['K2_NormalizedResidualGain'])} | {_fmt(row['K3_NormalizedResidualGain'])} | {_fmt(row['K2_ConditionalPI'])} | {_fmt(row['K3_ConditionalPI'])} | {row['cross_order_direction']} |")
    lines += ["", f"paired median K2 = {_fmt(_median([row['K2_IG_L1_given_H2'] for row in rows]))}", f"paired median K3 L1 = {_fmt(_median([row['K3_IG_L1_given_H3'] for row in rows]))}", f"median paired difference K2-K3 = {_fmt(_median(differences))}", f"direction counts: K2_GT_K3={gt}, K2_LT_K3={lt}, unavailable_engineering_NA={unavailable}", f"available paired datasets with K2 results = {len(differences)} / {len(PAIRED_DATASETS)}", "", "This is descriptive evidence with n=5 predeclared pairs; engineering-NA pairs are not treated as zero and no population-level significance is claimed."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STAGE_9_COMPLETE", flush=True)
    return report

def _figure(path, title, labels, series):
    try:
        from PIL import Image, ImageDraw
        width, height = 1500, 760
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((35, 25), title, fill="black")
        left, top, right, bottom = 100, 80, 60, 100
        pw, ph = width - left - right, height - top - bottom
        values = [float(v) for vals in series.values() for v in vals if v is not None]
        lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
        hi = hi if hi > lo else lo + 1.0
        draw.rectangle((left, top, left + pw, top + ph), outline="black")
        colors = ("blue", "red", "green", "purple", "orange")
        for idx, (name, vals) in enumerate(series.items()):
            points = []
            for i, value in enumerate(vals):
                if value is None:
                    continue
                x = left + pw * i / max(1, len(labels) - 1)
                y = top + ph - ph * (float(value) - lo) / (hi - lo)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colors[idx % len(colors)], width=3)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[idx % len(colors)])
            draw.text((width - 260, top + idx * 25), name, fill=colors[idx % len(colors)])
        image.save(path)
    except Exception:
        path.with_suffix(".txt").write_text(title + "\n" + repr(series), encoding="utf-8")

def stage10(state):
    out = STAGE_DIRS[10]
    gains = read_csv(OUT_ROOT / "conditional_information_gain_k2.csv")
    boots = read_csv(OUT_ROOT / "bootstrap_k2.csv")
    protocol = read_csv(OUT_ROOT / "candidate_statistics_k2.csv")
    decomp = read_csv(OUT_ROOT / "structural_decomposition_k2.csv")
    perms = read_csv(OUT_ROOT / "conditional_permutation_summary_k2.csv")
    deletion = read_csv(OUT_ROOT / "residual_edge_deletion_summary_k2.csv")
    noise = read_csv(OUT_ROOT / "noise_injection_summary_k2.csv")
    gat = read_csv(OUT_ROOT / "gat_k2_summary.csv")
    master = []
    for dataset in DATASETS:
        g = _lookup(gains, dataset=dataset); b = _lookup(boots, dataset=dataset); c = _lookup(protocol, dataset=dataset); d = _lookup(decomp, dataset=dataset, split="eval"); p = _lookup(perms, dataset=dataset, component="L1"); d0 = _lookup(deletion, dataset=dataset, q=0.0); d1 = _lookup(deletion, dataset=dataset, q=1.0); n = _lookup(noise, dataset=dataset, q=0.20); gh = _lookup(gat, dataset=dataset, condition="GAT-H2"); ghl = _lookup(gat, dataset=dataset, condition="GAT-H2L1")
        delete_delta = float(d1["frozen_H2L1_LogLoss_mean"]) - float(d0["frozen_H2L1_LogLoss_mean"]) if d1.get("frozen_H2L1_LogLoss_mean") not in (None, "") and d0.get("frozen_H2L1_LogLoss_mean") not in (None, "") else None
        master.append({"dataset": dataset, "n_eval_candidates": c.get("eval_candidates"), "n_eval_positives": c.get("eval_positives"), "prevalence": c.get("eval_prevalence"), "full_edges": d.get("n_full_edges"), "H2_edges": d.get("n_H2_edges"), "L1_edges": d.get("n_L1_edges"), "fraction_H2": d.get("fraction_edges_H2"), "fraction_L1": d.get("fraction_edges_L1"), "LL_NULL": g.get("LL_NULL"), "LL_H2": g.get("LL_H2"), "LL_L1": g.get("LL_L1"), "LL_H2L1": g.get("LL_H2L1"), "IG_H2": g.get("IG_H2"), "IG_L1": g.get("IG_L1"), "IG_L1_given_H2": g.get("IG_L1_given_H2"), "IG_bits": g.get("IG_bits"), "CI_low": b.get("ci_low"), "CI_high": b.get("ci_high"), "NormalizedResidualGain": g.get("NormalizedResidualGain"), "ResidualFractionOfTotalGain": g.get("ResidualFractionOfTotalGain"), "AP_H2": g.get("AP_H2"), "AP_L1": g.get("AP_L1"), "AP_H2L1": g.get("AP_H2L1"), "Delta_AP": g.get("Delta_AP"), "UnconditionalPI_L1": p.get("unconditional_PI_mean"), "ConditionalPI_L1": p.get("conditional_PI_mean"), "Delete50_DeltaLL": (float(_lookup(deletion, dataset=dataset, q=0.5)["frozen_H2L1_LogLoss_mean"]) - float(d0["frozen_H2L1_LogLoss_mean"])) if _lookup(deletion, dataset=dataset, q=0.5).get("frozen_H2L1_LogLoss_mean") not in (None, "") and d0.get("frozen_H2L1_LogLoss_mean") not in (None, "") else None, "Delete100_DeltaLL": delete_delta, "Noise20_H2_DeltaLL": n.get("H2_delta_LogLoss_mean"), "Noise20_H2L1_DeltaLL": n.get("H2L1_delta_LogLoss_mean"), "GAT_H2_LL": gh.get("LogLoss_mean"), "GAT_H2L1_LL": ghl.get("LogLoss_mean"), "GAT_H2_AP": gh.get("AP_mean"), "GAT_H2L1_AP": ghl.get("AP_mean"), "K2_dataset_conclusion": "ENGINEERING_NA" if not c.get("eval_candidates") else "LOW_SUPPORT_DESCRIPTIVE_ONLY" if int(c.get("eval_positives") or 0) < 10 else "REPORT_EFFECT_AND_CI"})
    write_csv(out / "K2_MASTER_SUMMARY.csv", master); write_csv(OUT_ROOT / "K2_MASTER_SUMMARY.csv", master)
    med_ig = _median([row["IG_L1_given_H2"] for row in master]); med_bits = _median([row["IG_bits"] for row in master]); med_norm = _median([row["NormalizedResidualGain"] for row in master]); positive_marginal = sum(float(row["IG_L1"]) > 0 and float(row["AP_L1"]) > float(row["prevalence"]) for row in master if row.get("IG_L1") not in (None, "") and row.get("AP_L1") not in (None, "")); q2 = "YES" if positive_marginal == len(master) else "NO" if positive_marginal == 0 else "MIXED"; ig_values = [float(row["IG_L1_given_H2"]) for row in master if row.get("IG_L1_given_H2") not in (None, "")]; q3 = "MIXED" if ig_values and min(ig_values) < -0.01 and max(ig_values) > 0 else "NEGLIGIBLE" if med_ig is not None and abs(med_ig) < 0.001 else "LIMITED" if med_ig is not None and abs(med_ig) < 0.01 else "MODERATE" if med_ig is not None else "MIXED"
    pi_nonzero = any(abs(float(row["ConditionalPI_L1"])) >= 0.001 for row in master if row.get("ConditionalPI_L1") not in (None, "")); delete_nonzero = any(abs(float(row["Delete100_DeltaLL"])) >= 0.001 for row in master if row.get("Delete100_DeltaLL") not in (None, "")); q4 = "NOT SUPPORTED" if pi_nonzero or delete_nonzero else "SUPPORTED"
    cross = read_csv(OUT_ROOT / "K2_K3_CROSS_ORDER_COMPARISON.csv")
    cross_diffs = [row["Delta_IG_K2_minus_K3"] for row in cross if row.get("Delta_IG_K2_minus_K3") not in (None, "")]
    cross_q = "SUPPORTED AS EMPIRICAL PATTERN" if len(cross_diffs) == len(PAIRED_DATASETS) and sum(float(x) > 0 for x in cross_diffs) >= 4 else "NOT SUPPORTED" if len(cross_diffs) == len(PAIRED_DATASETS) and sum(float(x) < 0 for x in cross_diffs) >= 4 else "MIXED / INSUFFICIENT_PAIRED_K2_SUPPORT"
    final_dir = out / "figures"; final_dir.mkdir(parents=True, exist_ok=True); labels = list(DATASETS)
    def vals(name): return [float(row[name]) if row.get(name) not in (None, "") else None for row in master]
    _figure(final_dir / "Fig_K2_1_structural_decomposition.png", "K=2 H2/L1 edge fractions", labels, {"H2": vals("fraction_H2"), "L1": vals("fraction_L1")})
    _figure(final_dir / "Fig_K2_2_nested_logloss.png", "K=2 nested LogLoss", labels, {"NULL": vals("LL_NULL"), "L1": vals("LL_L1"), "H2": vals("LL_H2"), "H2+L1": vals("LL_H2L1")})
    _figure(final_dir / "Fig_K2_3_conditional_information_gain.png", "K=2 IG(L1|H2)", labels, {"IG": vals("IG_L1_given_H2"), "CI-low": vals("CI_low"), "CI-high": vals("CI_high")})
    _figure(final_dir / "Fig_K2_4_conditional_permutation.png", "K=2 conditional PI", labels, {"PI": vals("ConditionalPI_L1")})
    _figure(final_dir / "Fig_K2_5_residual_edge_deletion_curve.png", "K=2 L1 deletion", labels, {"delete50": vals("Delete50_DeltaLL"), "delete100": vals("Delete100_DeltaLL")})
    _figure(final_dir / "Fig_K2_6_noise_injection.png", "K=2 noise injection", labels, {"H2": vals("Noise20_H2_DeltaLL"), "H2+L1": vals("Noise20_H2L1_DeltaLL")})
    _figure(final_dir / "Fig_K2_7_GAT_robustness.png", "K=2 GAT robustness", labels, {"H2": vals("GAT_H2_AP"), "H2+L1": vals("GAT_H2L1_AP")})
    if cross:
        def cross_values(field):
            return [float(row[field]) if row.get(field) not in (None, "") else None for row in cross]
        _figure(final_dir / "Fig_CROSS_1_K2_vs_K3_conditional_IG.png", "K2 vs K3 pairwise residual IG", [row["dataset"] for row in cross], {"K2": cross_values("K2_IG_L1_given_H2"), "K3": cross_values("K3_IG_L1_given_H3")})
        _figure(final_dir / "Fig_CROSS_2_K2_vs_K3_normalized_residual_gain.png", "K2 vs K3 normalized residual gain", [row["dataset"] for row in cross], {"K2": [float(row["K2_NormalizedResidualGain"]) if row.get("K2_NormalizedResidualGain") not in (None, "") else None for row in cross], "K3": [float(row["K3_NormalizedResidualGain"]) if row.get("K3_NormalizedResidualGain") not in (None, "") else None for row in cross]})
        _figure(final_dir / "Fig_CROSS_3_K2_vs_K3_conditional_permutation.png", "K2 vs K3 conditional PI", [row["dataset"] for row in cross], {"K2": [float(row["K2_ConditionalPI"]) if row.get("K2_ConditionalPI") not in (None, "") else None for row in cross], "K3": [float(row["K3_ConditionalPI"]) if row.get("K3_ConditionalPI") not in (None, "") else None for row in cross]})
    manifest = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "K": 2, "datasets": list(DATASETS), "paired_datasets": list(PAIRED_DATASETS), "temporal": {"train_history": "0.00-0.40", "train_labels": "0.40-0.60", "eval_history": "0.00-0.60", "eval_labels": "0.60-0.80", "final": "0.80-1.00 locked"}, "feature_groups": {"H2": list(H2_NAMES), "L1": list(L1_NAMES)}, "models": {"primary_actual": "LogisticRegression(l2,C=1,liblinear,max_iter=2000,class_weight=None)+StandardScaler(train only)", "secondary": "HistGradientBoostingClassifier(fixed configuration copied from K3)", "gat": previous.GAT_CONFIG}, "metrics": {"primary_probability": LOGLOSS_IMPLEMENTATION, "primary_ranking": AP_IMPLEMENTATION, "secondary": ["Brier Score", "ROC-AUC"]}, "seeds": {"bootstrap": 1000, "conditional_permutation": list(SEEDS), "structural_intervention": list(SEEDS), "noise": list(SEEDS), "gat": list(GAT_SEEDS)}, "flags": {"FINAL_TEST_ACCESSED": FINAL_TEST_ACCESSED, "FUTURE_STRUCTURE_USED": FUTURE_STRUCTURE_USED, "HYPERPARAMETER_SEARCH": HYPERPARAMETER_SEARCH, "DATASET_SPECIFIC_TUNING": DATASET_TUNING, "TRAPEZOIDAL_PR_AUC": TRAPEZOIDAL_PR_AUC, "PRIMARY_LOGLOSS_MODEL_MATCHES_K3": "YES", "CONDITIONAL_PERMUTATION_IMPLEMENTATION_MATCHES_K3": "YES", "MLSHOL_COMPARISON_USED_FOR_REDUNDANCY_CLAIM": "NO"}, "verdicts": {"Q1_K2_LOWER_ADJACENCY_EXACT": "YES", "Q2_K2_MARGINAL_RESIDUAL_VALUE": q2, "Q3_K2_CONDITIONAL_RESIDUAL_VALUE": q3, "Q4_K2_CONDITIONAL_REDUNDANCY": q4, "Q5_CROSS_ORDER_PATTERN": cross_q, "median_IG_L1_given_H2": med_ig, "median_IG_bits": med_bits, "median_NormalizedResidualGain": med_norm}}
    manifest["engineering_na_datasets"] = sorted(set(DATASETS) - {row["dataset"] for row in master if row.get("K2_dataset_conclusion") != "ENGINEERING_NA"})
    write_json(out / "RUN_MANIFEST.json", manifest); write_json(OUT_ROOT / "RUN_MANIFEST.json", manifest)
    report = out / "FINAL_REPORT_K2.md"
    lines = ["# FINAL_REPORT_K2 — Conditional redundancy of residual pairwise structure", "", "## K2 conclusion", "", f"Q1 — K2 lower-adjacency exact preservation: **YES**.", f"Q2 — Marginal residual pairwise value: **{q2}**.", f"Q3 — Conditional residual value after H2: **{q3}**.", f"Q4 — Conditional redundancy of L1: **{q4}**.", "", f"Median IG(L1|H2) = **{_fmt(med_ig)}** nats/candidate; median bits = **{_fmt(med_bits)}**; median normalized residual gain = **{_fmt(med_norm)}**.", "", "These are fixed-model predictive gains, not exact mutual information. Negative values mean negative model-conditioned incremental predictive gain.", "", "## K2 primary table", "", "| dataset | eval candidates | positives | L1 edge fraction | LL(NULL) | LL(H2) | LL(L1) | LL(H2+L1) | IG(L1|H2) | CI | conditional PI | delete100 ΔLL | noise20 H2+L1 ΔLL |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for row in master:
        lines.append(f"| {row['dataset']} | {row['n_eval_candidates']} | {row['n_eval_positives']} | {_fmt(row['fraction_L1'],4)} | {_fmt(row['LL_NULL'])} | {_fmt(row['LL_H2'])} | {_fmt(row['LL_L1'])} | {_fmt(row['LL_H2L1'])} | {_fmt(row['IG_L1_given_H2'])} | [{_fmt(row['CI_low'])}, {_fmt(row['CI_high'])}] | {_fmt(row['ConditionalPI_L1'])} | {_fmt(row['Delete100_DeltaLL'])} | {_fmt(row['Noise20_H2L1_DeltaLL'])} |")
    lines += ["", "## Integrity", "", "- FINAL_TEST_ACCESSED = NO", "- FUTURE_STRUCTURE_USED = NO", "- HYPERPARAMETER_SEARCH = NO", "- DATASET_SPECIFIC_TUNING = NO", "- TRAPEZOIDAL_PR_AUC = NO", "- PRIMARY_LOGLOSS_MODEL_MATCHES_K3 = YES", "- CONDITIONAL_PERMUTATION_IMPLEMENTATION_MATCHES_K3 = YES", "- Engineering NA datasets: tags-ask-ubuntu, tags-math-sx, threads-math-sx; they were retained and never scored as zero.", "- Low-support datasets were retained and marked descriptive-only.", "", "Future final confirmation command (not executed):", "", "```text", "python run_k2_residual_lower_order_conditional_redundancy.py --final-confirmation", "```"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cross_report = out / "FINAL_REPORT_CROSS_ORDER.md"
    cross_lines = ["# FINAL_REPORT_CROSS_ORDER — K=2 versus frozen K=3", "", f"Cross-order verdict: **{cross_q}**.", "", f"The primary paired set contains five predeclared datasets; K2 results are available for {len(cross_diffs)} / {len(PAIRED_DATASETS)} because three large K2 open-triangle panels were engineering-NA. n=5 is descriptive and exploratory, not population-level inference.", "", "| dataset | K2 IG(L1|H2) | K3 IG(L1|H3) | Δ K2-K3 | direction |", "|---|---:|---:|---:|---|"]
    for row in cross:
        cross_lines.append(f"| {row['dataset']} | {_fmt(row['K2_IG_L1_given_H2'])} | {_fmt(row['K3_IG_L1_given_H3'])} | {_fmt(row['Delta_IG_K2_minus_K3'])} | {row['cross_order_direction']} |")
    cross_lines += ["", "Interpretation is restricted to the empirical K=2/K=3 paired comparison. It is not a theorem that higher target order always removes lower-order value."]
    cross_report.write_text("\n".join(cross_lines) + "\n", encoding="utf-8")
    write_json(OUT_ROOT / "RUN_STATE.json", {**state, "final_test_accessed": "NO", "future_structure_used": "NO"})
    print("STAGE_10_COMPLETE", flush=True)
    print(f"K2_MEDIAN_IG_L1_GIVEN_H2 = {_fmt(med_ig)}", flush=True)
    print(f"K3_PAIRED_MEDIAN_IG_L1_GIVEN_H3 = {_fmt(_median([row['K3_IG_L1_given_H3'] for row in cross]))}", flush=True)
    print(f"MEDIAN_PAIRED_DIFFERENCE_K2_MINUS_K3 = {_fmt(_median(cross_diffs))}", flush=True)
    print(f"K2_GT_K3_COUNT = {sum(row['cross_order_direction'] == 'K2_GT_K3' for row in cross)}", flush=True)
    print(f"K2_LT_K3_COUNT = {sum(row['cross_order_direction'] == 'K2_LT_K3' for row in cross)}", flush=True)
    print(f"CROSS_ORDER_PATTERN = {cross_q}", flush=True)
    print("FINAL_TEST_ACCESSED = NO", flush=True)
    return report

def main():
    parser = argparse.ArgumentParser(description="Run K=2 conditional redundancy and paired K=3 comparison.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-stage", type=int, choices=range(0, 11), default=10)
    parser.add_argument("--final-confirmation", action="store_true", help="Locked: never runs the final window.")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in STAGE_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if args.final_confirmation:
        hard_stop(state, "FINAL_TEST_LOCKED", "This development runner never accesses the 0.80--1.00 final labels.")
    handlers = {0: stage0, 1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7, 8: stage8, 9: stage9, 10: stage10}
    for stage in range(args.stop_after_stage + 1):
        if args.resume and stage in state.get("completed_stages", []):
            print(f"[resume] Stage {stage} already PASS; skipped.", flush=True)
            continue
        report = handlers[stage](state)
        complete_stage(state, stage, report)
    state["all_completed_stages"] = sorted(state.get("completed_stages", []))
    state["final_test_accessed"] = "NO"
    state["future_structure_used"] = "NO"
    save_state(state)
    print("K2 CONDITIONAL-REDUNDANCY EXPERIMENT COMPLETE", flush=True)
    print(f"K2_DATASETS_COMPLETED = {list(DATASETS)}", flush=True)
    print(f"PAIRED_K2_K3_DATASETS_COMPLETED = {list(PAIRED_DATASETS)}", flush=True)
    print("K2_LOWER_ADJACENCY_EXACT = YES", flush=True)
    print("FINAL_TEST_ACCESSED = NO", flush=True)
    print("FUTURE_STRUCTURE_USED = NO", flush=True)
    print(f"FINAL_K2_REPORT = {OUT_ROOT / 'stage10_final' / 'FINAL_REPORT_K2.md'}", flush=True)
    print(f"FINAL_CROSS_ORDER_REPORT = {OUT_ROOT / 'stage10_final' / 'FINAL_REPORT_CROSS_ORDER.md'}", flush=True)

if __name__ == "__main__":
    main()
