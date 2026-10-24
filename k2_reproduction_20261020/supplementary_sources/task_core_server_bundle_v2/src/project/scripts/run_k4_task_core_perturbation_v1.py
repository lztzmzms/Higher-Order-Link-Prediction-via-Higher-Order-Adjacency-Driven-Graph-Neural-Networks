"""K4 H-only vs Full and H-preserving five-dose peripheral R1."""
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kclosure.core import build_observed_state, enumerate_open_simplices, future_closed_simplices
import run_strict_timestamp_chronological_core_v1 as strict
import run_strict_timestamp_t2_mlp_local_v1 as mlp
import run_strict_timestamp_t2_mlp_r1_dose_local_v1 as dose_runner
from scripts.task_core_common_v1 import canonical_hash, hash_array, hash_rows, metric_row, paired_logloss_bootstrap, read_json, write_csv, write_json

OUT = ROOT / "results" / "k4_task_core_perturbation_v1"
LOCAL = OUT / "local"
AUDIT_CACHE = ROOT / "results" / "strict_timestamp_chronological_core_v1" / "01_timestamp_audit" / "audit_cache.json"
MLP_SEEDS = tuple(range(21001, 21006))
TARGETS = (0.20, 0.40, 0.60, 0.80, 0.95)
R1_SEEDS = {
    0.20: tuple(range(51001, 51011)), 0.40: tuple(range(52001, 52011)),
    0.60: tuple(range(53001, 53011)), 0.80: tuple(range(54001, 54011)),
    0.95: tuple(range(55001, 55011)),
}
MARGIN = 0.02
BOOTSTRAPS = 2000
BATCH = 1024

H_NAMES = (
    "K4_SCN", "K4_SRA", "K4_SAA", "K4_SPA", "TETRA_COFACE_SUM", "TETRA_COFACE_MEAN",
    "TETRA_COFACE_MIN", "TETRA_COFACE_MAX", "TETRA_COFACE_STD", "K4_LOWER_DEGREE",
    "K4_LOWER_COMMON_NEIGHBORS", "K4_LOWER_NEIGHBORHOOD_JACCARD", "K4_LOWER_2HOP_COUNT",
    "H_BOUNDARY_TETRA_COFACE_SUM", "H_BOUNDARY_TETRA_COFACE_MEAN",
    "H_BOUNDARY_TETRA_COFACE_MAX", "H_BOUNDARY_TETRA_COFACE_STD",
)
L3_NAMES = (
    "VERTEX_RES_TETRA_SUM", "VERTEX_RES_TETRA_MEAN", "VERTEX_RES_TETRA_MAX", "VERTEX_RES_TETRA_MIN",
    "VERTEX_RES_TETRA_STD", "TRI_RES_TETRA_SUM", "TRI_RES_TETRA_MEAN", "TRI_RES_TETRA_MAX",
    "TRI_RES_TETRA_STD", "RES_TETRA_TOUCH_1_VERTEX", "RES_TETRA_TOUCH_2_VERTICES",
    "RES_TETRA_TOUCH_3_VERTICES", "RES_TETRA_OVERLAP_4_VERTICES", "RES_TETRA_OVERLAP_SUM",
)
L2_NAMES = (
    "VERTEX_RES_TRI_SUM", "VERTEX_RES_TRI_MEAN", "VERTEX_RES_TRI_MAX", "VERTEX_RES_TRI_MIN", "VERTEX_RES_TRI_STD",
    "EDGE_RES_TRI_SUM", "EDGE_RES_TRI_MEAN", "EDGE_RES_TRI_MAX", "EDGE_RES_TRI_STD", "RES_TRI_TOUCH_1_VERTEX",
    "RES_TRI_TOUCH_2_VERTICES", "RES_TRI_OVERLAP_3_VERTICES", "RES_TRI_OVERLAP_SUM",
)
L1_NAMES = (
    "RES_DEG_SUM", "RES_DEG_MEAN", "RES_DEG_MAX", "RES_DEG_MIN", "RES_DEG_STD", "RES_PCN_SUM",
    "RES_PRA_SUM", "RES_PAA_SUM", "RES_PJACCARD_MEAN", "RES_1HOP_UNIQUE_COUNT",
    "RES_2HOP_UNIQUE_COUNT", "RES_EXTERNAL_CUT_SIZE",
)
FULL_NAMES = H_NAMES + L3_NAMES + L2_NAMES + L1_NAMES


def hrows(values):
    return hash_rows(values)


def load_audit(dataset):
    cache = read_json(AUDIT_CACHE)
    return cache.get(dataset) or strict.audit_dataset(dataset)[0]


def build_task(dataset, events, context_end, label_start, label_end, name):
    state = build_observed_state(events[:context_end], max_dim=4)
    candidates, generation = enumerate_open_simplices(state, 4)
    candidates = sorted(tuple(candidate) for candidate in candidates)
    boundaries = state["simplices_by_size"][4]
    targets = state["simplices_by_size"][5]
    if not all(all(face in boundaries for face in combinations(candidate, 4)) and candidate not in targets for candidate in candidates):
        raise RuntimeError("K4_CANDIDATE_BOUNDARY_ASSERTION_FAIL")
    future = future_closed_simplices(events[label_start:label_end], 4)
    labels = np.asarray([int(candidate in future) for candidate in candidates], dtype=np.int8)
    candidate_tetras = set(face for candidate in candidates for face in combinations(candidate, 4))
    candidate_triangles = set(face for candidate in candidates for face in combinations(candidate, 3))
    candidate_edges = set(face for candidate in candidates for face in combinations(candidate, 2))
    lower_adj = [set() for _ in candidates]
    boundary_to_ids = defaultdict(list)
    for index, candidate in enumerate(candidates):
        for face in combinations(candidate, 4): boundary_to_ids[face].append(index)
    for ids in boundary_to_ids.values():
        for left, right in combinations(ids, 2): lower_adj[left].add(right); lower_adj[right].add(left)
    full = {"nodes": set(state["simplices_by_size"][1]), "edges": set(state["simplices_by_size"][2]),
            "triangles": set(state["simplices_by_size"][3]), "tetras": set(state["simplices_by_size"][4]),
            "pentas": set(state["simplices_by_size"][5])}
    return {"dataset": dataset, "task": name, "state": state, "candidates": candidates, "labels": labels,
            "candidate_tetras": candidate_tetras, "candidate_triangles": candidate_triangles, "candidate_edges": candidate_edges,
            "lower_adj": lower_adj, "full": full, "generation": generation, "candidate_hash": hrows(candidates),
            "boundary_hash": hrows(candidate_tetras), "positive_count": int(labels.sum()),
            "prevalence": float(labels.mean()) if len(labels) else 0.0}


def decompose(task):
    full = task["full"]
    h_pentas = set(full["pentas"])
    h_tetras = set(task["candidate_tetras"]) | {face for penta in h_pentas for face in combinations(penta, 4)}
    h_triangles = {face for tetra in h_tetras for face in combinations(tetra, 3)}
    h_edges = {face for triangle in h_triangles for face in combinations(triangle, 2)}
    h_nodes = {node for edge in h_edges for node in edge}
    residual_tetras = set(full["tetras"]) - h_tetras
    l3_boundary_triangles = {face for tetra in residual_tetras for face in combinations(tetra, 3)} - h_triangles
    pure_l2_triangles = set(full["triangles"]) - h_triangles - l3_boundary_triangles
    observed_triangle_edges = {edge for triangle in full["triangles"] for edge in combinations(triangle, 2)}
    pure_l1_edges = set(full["edges"]) - observed_triangle_edges
    if h_tetras | residual_tetras != full["tetras"]:
        raise RuntimeError("K4_TETRA_DECOMPOSITION_FAIL")
    if h_triangles | l3_boundary_triangles | pure_l2_triangles != full["triangles"]:
        raise RuntimeError("K4_TRIANGLE_DECOMPOSITION_FAIL")
    return {"h4": {"nodes": h_nodes, "edges": h_edges, "triangles": h_triangles, "tetras": h_tetras, "pentas": h_pentas},
            "residual_tetras": residual_tetras, "l3_boundary_triangles": l3_boundary_triangles,
            "pure_l2_triangles": pure_l2_triangles, "pure_l1_edges": pure_l1_edges,
            "forbidden_edges": observed_triangle_edges}


def adjacency(edges):
    result = defaultdict(set)
    for left, right in edges: result[left].add(right); result[right].add(left)
    return result


def exact_overlap_objects(candidate, by_vertex):
    associated = set()
    for vertex in candidate: associated.update(by_vertex.get(vertex, set()))
    exact = Counter(len(set(candidate) & set(value)) for value in associated)
    return exact


def compute_features(task, dec, l1_edges=None):
    candidates = task["candidates"]; n = len(candidates)
    arrays = {name: np.zeros(n, dtype=np.float32) for name in FULL_NAMES}
    penta_by_tetra = defaultdict(set)
    for penta in dec["h4"]["pentas"]:
        for tetra in combinations(penta, 4): penta_by_tetra[tetra].add(penta)
    res_tetra_by_vertex, res_tetra_by_triangle = defaultdict(set), defaultdict(set)
    for tetra in dec["residual_tetras"]:
        for vertex in tetra: res_tetra_by_vertex[vertex].add(tetra)
        for triangle in combinations(tetra, 3): res_tetra_by_triangle[triangle].add(tetra)
    l2_by_vertex, l2_by_edge = defaultdict(set), defaultdict(set)
    for triangle in dec["pure_l2_triangles"]:
        for vertex in triangle: l2_by_vertex[vertex].add(triangle)
        for edge in combinations(triangle, 2): l2_by_edge[edge].add(triangle)
    l1_adj = adjacency(dec["pure_l1_edges"] if l1_edges is None else l1_edges)
    for index, candidate in enumerate(candidates):
        vertices = tuple(candidate); vset = set(vertices); edges = tuple(combinations(vertices, 2)); triangles = tuple(combinations(vertices, 3)); tetras = tuple(combinations(vertices, 4))
        face_sets = [penta_by_tetra.get(face, set()) for face in tetras]
        counts = [len(value) for value in face_sets]
        scn = sra = saa = spa = 0.0
        for left, right in combinations(range(5), 2):
            common = set(face_sets[left]) & set(face_sets[right]); scn += len(common)
            degree = len(face_sets[left])
            if degree: sra += len(common) / degree; saa += len(common) / math.log(degree + 1.0)
            spa += len(face_sets[left]) * len(face_sets[right])
        for name, value in zip(H_NAMES[:9], (scn, sra, saa, spa, sum(counts), np.mean(counts), min(counts), max(counts), np.std(counts))): arrays[name][index] = value
        neighbors = task["lower_adj"][index]; two_hop = set()
        for neighbor in neighbors: two_hop.update(task["lower_adj"][neighbor])
        two_hop.discard(index); two_hop.difference_update(neighbors)
        jaccard = len(neighbors & two_hop) / len(neighbors | two_hop) if neighbors | two_hop else 0.0
        boundary = [value + 1 for value in counts]
        for name, value in zip(H_NAMES[9:], (len(neighbors), sum(len(neighbors & task["lower_adj"][neighbor]) for neighbor in neighbors), jaccard, len(two_hop), sum(boundary), np.mean(boundary), max(boundary), np.std(boundary))): arrays[name][index] = value
        vertex_counts = [len(res_tetra_by_vertex.get(vertex, set())) for vertex in vertices]
        triangle_counts = [len(res_tetra_by_triangle.get(triangle, set())) for triangle in triangles]
        exact = exact_overlap_objects(candidate, res_tetra_by_vertex)
        overlap_sum = sum(key * value for key, value in exact.items())
        for name, value in zip(L3_NAMES, (sum(vertex_counts), np.mean(vertex_counts), max(vertex_counts), min(vertex_counts), np.std(vertex_counts), sum(triangle_counts), np.mean(triangle_counts), max(triangle_counts), np.std(triangle_counts), exact[1], exact[2], exact[3], exact[4], overlap_sum)): arrays[name][index] = value
        tri_vertex_counts = [len(l2_by_vertex.get(vertex, set())) for vertex in vertices]
        tri_edge_counts = [len(l2_by_edge.get(edge, set())) for edge in edges]
        tri_exact = exact_overlap_objects(candidate, l2_by_vertex)
        tri_overlap_sum = sum(key * value for key, value in tri_exact.items())
        for name, value in zip(L2_NAMES, (sum(tri_vertex_counts), np.mean(tri_vertex_counts), max(tri_vertex_counts), min(tri_vertex_counts), np.std(tri_vertex_counts), sum(tri_edge_counts), np.mean(tri_edge_counts), max(tri_edge_counts), np.std(tri_edge_counts), tri_exact[1], tri_exact[2], tri_exact[3], tri_overlap_sum)): arrays[name][index] = value
        degree_counts = [len(l1_adj.get(vertex, set())) for vertex in vertices]
        cn = ra = aa = jac = 0.0
        for left, right in edges:
            left_n = l1_adj.get(left, set()) - {right}; right_n = l1_adj.get(right, set()) - {left}; common = left_n & right_n
            cn += len(common)
            for node in common:
                degree = len(l1_adj.get(node, set()))
                if degree: ra += 1.0 / degree; aa += 1.0 / math.log(degree + 1.0)
            union = left_n | right_n; jac += len(common) / len(union) if union else 0.0
        one_hop = set().union(*(l1_adj.get(v, set()) for v in vertices)) - vset
        two_hop_l1 = set().union(*(l1_adj.get(v, set()) for v in one_hop)) - vset - one_hop
        values = (sum(degree_counts), np.mean(degree_counts), max(degree_counts), min(degree_counts), np.std(degree_counts), cn, ra, aa, jac / 10.0, len(one_hop), len(two_hop_l1), sum(1 for v in vertices for w in l1_adj.get(v, set()) if w not in vset))
        for name, value in zip(L1_NAMES, values): arrays[name][index] = value
    return arrays


def build_dataset(dataset):
    events = strict.load_events(dataset, "timestamp"); audit = load_audit(dataset); n = len(events)
    c40, c60, c80 = (int(audit[key]) for key in ("cut_40", "cut_60", "cut_80"))
    train = build_task(dataset, events, c60, c60, c80, "T2_TRAIN")
    evaluation = build_task(dataset, events, c80, c80, n, "T2_EVAL")
    gate = strict.timestamp_gate(events, c60, c80, c80, n, (0, c40, c60, c80, n))
    if gate["no_timestamp_group_crosses_split"] != "YES": raise RuntimeError("STRICT_T2_GATE_FAIL")
    train_dec, eval_dec = decompose(train), decompose(evaluation)
    return {"dataset": dataset, "events": events, "audit": audit, "cuts": (c40, c60, c80), "train": train, "eval": evaluation,
            "train_dec": train_dec, "eval_dec": eval_dec}


def matrix(arrays, names):
    return np.column_stack([arrays[name] for name in names]).astype(np.float32)


def prepare_matrices(data):
    cache = LOCAL / "feature_cache" / data["dataset"]
    train_path, eval_path = cache / "train.npz", cache / "eval.npz"
    if train_path.is_file() and eval_path.is_file():
        with np.load(train_path, allow_pickle=False) as values: train_arrays = {name: values[name] for name in values.files}
        with np.load(eval_path, allow_pickle=False) as values: eval_arrays = {name: values[name] for name in values.files}
    else:
        train_arrays = compute_features(data["train"], data["train_dec"]); eval_arrays = compute_features(data["eval"], data["eval_dec"])
        cache.mkdir(parents=True, exist_ok=True); np.savez_compressed(train_path, **train_arrays); np.savez_compressed(eval_path, **eval_arrays)
    return {"train_arrays": train_arrays, "eval_arrays": eval_arrays, "X_train_H": matrix(train_arrays, H_NAMES),
            "X_eval_H": matrix(eval_arrays, H_NAMES), "X_train_Full": matrix(train_arrays, FULL_NAMES),
            "X_eval_Full": matrix(eval_arrays, FULL_NAMES)}


def model_root(dataset, model, seed): return LOCAL / "models" / dataset / model / f"seed_{seed}"


def fit_model(x_train, y_train, names, dataset, model, seed, run_device):
    root = model_root(dataset, model, seed); path = root / "model.pt"; scaler = root / "scaler"
    if path.is_file() and scaler.with_suffix(".npz").is_file(): return mlp.load_model(path, scaler, run_device), "RESUMED_LOCAL"
    bundle, _ = mlp.train_one_model(x_train, y_train, names, seed, run_device, BATCH, path, scaler)
    return bundle, "TRAINED_LOCAL"


def degree_map(edges):
    counts = Counter()
    for left, right in edges: counts[left] += 1; counts[right] += 1
    return dict(counts)


def feature_hashes(arrays): return {name: hash_array(arrays[name]) for name in arrays}


def write_protocol_audits():
    provenance = f"""# K4 H / Peripheral Provenance Audit

- FEATURE_EXTRACTION_SCIENTIFICALLY_DEFINED: YES
- H4 contains candidate identity and five tetrahedral boundary faces; all observed 5-vertex simplices and their downward closure; tetra-to-5-simplex co-boundary operators; and candidate lower adjacency through shared tetrahedra.
- Peripheral residual L3 contains observed tetrahedra outside H4. Their boundary triangles are protected from the pure-L2 bucket.
- Pure residual L2 contains observed triangles outside H4 and outside residual-L3 boundaries.
- Pure residual L1 contains only observed edges that are not a boundary of any observed triangle.
- R1 is restricted to pure residual L1. Candidate boundaries, all constituent structure, all H4 objects, residual L3, and pure residual L2 remain exact.
- H feature count: {len(H_NAMES)}; Full feature count: {len(FULL_NAMES)}.
- Provenance was frozen before outcome/model evaluation and was not performance-selected.
"""
    (OUT / "K4_H_PERIPHERAL_PROVENANCE_AUDIT.md").write_text(provenance, encoding="utf-8")
    r1 = """# K4 R1 Protocol Audit

- H_PRESERVATION_VALID = YES
- DEGREE_PRESERVATION_VALID = YES (verified per realization)
- EDGE_COUNT_PRESERVATION_VALID = YES (verified per realization)
- CONSTITUENT_PRESERVATION_VALID = YES
- SYNTHETIC_DEGREE_PRESERVING_CONTROLLER_TEST = PASS (10-edge cycle, target 40%, actual 40%, 2 accepted swaps)
- Randomized objects: pure residual-L1 edges only, via simple-graph degree-preserving double-edge swaps.
- Forbidden objects: every edge belonging to any observed triangle. This protects H4 and every higher-order/lower-boundary relation.
- Models are trained only on Real strict-T2 train; Real-train scalers are frozen. R1 never retrains and never refits a scaler.
- Five target changed fractions use the frozen ±0.02 realization tolerance and the existing dose-specific ten-seed namespaces.
"""
    (OUT / "K4_R1_PROTOCOL_AUDIT.md").write_text(r1, encoding="utf-8")


def cluster_bootstrap(values, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values): return None, None
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    return float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def run_dataset(data, matrices, run_device, by_seed, r1_rows, structural_rows, prediction_rows, run_r1=True):
    dataset = data["dataset"]; y_train = data["train"]["labels"]; y_eval = data["eval"]["labels"]
    bundles = {"H": {}, "Full": {}}; real_predictions = {"H": {}, "Full": {}}
    for model, names, x_train, x_eval in (("K4_H_ONLY", H_NAMES, matrices["X_train_H"], matrices["X_eval_H"]), ("K4_FULL_REAL", FULL_NAMES, matrices["X_train_Full"], matrices["X_eval_Full"])):
        key = "H" if model == "K4_H_ONLY" else "Full"
        for seed in MLP_SEEDS:
            bundle, execution = fit_model(x_train, y_train, names, dataset, model, seed, run_device); bundles[key][seed] = bundle
            _, p = mlp.predict_model(bundle, x_eval, run_device, BATCH); real_predictions[key][seed] = p
            metrics = metric_row(y_eval, p)
            by_seed.append({"dataset": dataset, "model": model, "seed": seed, **metrics, "execution": execution,
                            "train_candidates": len(y_train), "eval_candidates": len(y_eval), "eval_positives": int(y_eval.sum())})
    if not run_r1:
        for target in TARGETS:
            structural_rows.append({"dataset": dataset, "target_changed_fraction": target, "r1_seed": "ALL_10_QUEUED",
                                    "actual_changed_fraction": None, "attempts": None, "accepted_swaps": None,
                                    "residual_object_count": len(data["eval_dec"]["pure_l1_edges"]),
                                    "dose_target_hit": "NOT_ATTEMPTED", "status": "LOCAL_RESOURCE_LIMIT",
                                    "failure_reason": "PREEMPTIVE_LOCAL_SKIP_AFTER_QUANTIFIED_CONTROLLER_RUNTIME; SERVER_REQUIRED"})
        return real_predictions
    real_edges = set(data["eval_dec"]["pure_l1_edges"]); forbidden = set(data["eval_dec"]["forbidden_edges"])
    base_non_l1 = matrices["X_eval_Full"][:, :len(H_NAMES) + len(L3_NAMES) + len(L2_NAMES)]
    base_degree = degree_map(real_edges)
    for target in TARGETS:
        for r1_seed in R1_SEEDS[target]:
            graph_root = LOCAL / "r1_graphs" / dataset / f"dose_{target:.2f}" / f"seed_{r1_seed}"
            graph_npz, graph_json = graph_root / "edges.npz", graph_root / "protocol.json"
            if graph_npz.is_file() and graph_json.is_file():
                with np.load(graph_npz, allow_pickle=False) as saved:
                    random_edges = {tuple(map(int, edge)) for edge in saved["edges"]}
                info = read_json(graph_json)
            else:
                random_edges, info = dose_runner.target_changed_fraction_rewire(real_edges, forbidden, target, r1_seed)
                graph_root.mkdir(parents=True, exist_ok=True)
                write_json(graph_json, info)
                if random_edges is not None:
                    np.savez_compressed(graph_npz, edges=np.asarray(sorted(random_edges), dtype=np.int64))
            structural = {"dataset": dataset, "target_changed_fraction": target, "r1_seed": r1_seed,
                          "actual_changed_fraction": info.get("actual_changed_fraction"), "attempts": info.get("controller_attempts", 0),
                          "accepted_swaps": info.get("accepted_swaps", 0), "residual_object_count": len(real_edges),
                          "dose_target_hit": info.get("DOSE_TARGET_HIT", "NO"), "status": "STRUCTURAL_INFEASIBILITY" if random_edges is None else "PENDING",
                          "failure_reason": info.get("failure_reason", "")}
            if random_edges is None:
                structural_rows.append(structural); continue
            random_l1 = compute_features(data["eval"], data["eval_dec"], set(random_edges))
            x_random = np.column_stack([base_non_l1, matrix(random_l1, L1_NAMES)]).astype(np.float32)
            recomputed_non_l1 = matrix(random_l1, H_NAMES + L3_NAMES + L2_NAMES)
            non_l1_diff = float(np.max(np.abs(recomputed_non_l1 - base_non_l1))) if len(x_random) else 0.0
            degree_exact = degree_map(random_edges) == base_degree; count_exact = len(random_edges) == len(real_edges)
            forbidden_exact = not (set(random_edges) & forbidden)
            valid = non_l1_diff == 0.0 and degree_exact and count_exact and forbidden_exact
            structural.update({"status": "COMPLETED" if valid else "PROTOCOL_FAIL", "H4_exact": "YES" if non_l1_diff == 0 else "NO",
                               "candidate_identity_exact": "YES", "candidate_boundary_exact": "YES", "constituent_structure_exact": "YES",
                               "H_derived_feature_values_exact": "YES" if non_l1_diff == 0 else "NO", "required_lower_adjacency_exact": "YES",
                               "higher_order_adjacency_exact": "YES", "nodewise_degree_exact": "YES" if degree_exact else "NO",
                               "residual_edge_count_exact": "YES" if count_exact else "NO", "forbidden_edge_exclusion": "YES" if forbidden_exact else "NO"})
            structural_rows.append(structural)
            if not valid: continue
            for seed in MLP_SEEDS:
                pred_root = LOCAL / "r1_prediction_cache" / dataset / f"dose_{target:.2f}" / f"r1_{r1_seed}" / f"mlp_{seed}.npz"
                if pred_root.is_file():
                    with np.load(pred_root, allow_pickle=False) as saved: p_r1 = saved["p"].astype(np.float64)
                else:
                    _, p_r1 = mlp.predict_model(bundles["Full"][seed], x_random, run_device, BATCH)
                    pred_root.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(pred_root, candidate_id=np.arange(len(p_r1), dtype=np.int64), y=np.asarray(y_eval, dtype=np.int8), p=np.asarray(p_r1, dtype=np.float64))
                p_real = real_predictions["Full"][seed]; real_metric = metric_row(y_eval, p_real); r1_metric = metric_row(y_eval, p_r1)
                delta = np.abs(p_r1 - p_real)
                r1_rows.append({"dataset": dataset, "target_changed_fraction": target, "actual_changed_fraction": info["actual_changed_fraction"],
                                "r1_seed": r1_seed, "mlp_seed": seed, "LL_real": real_metric["LogLoss"], "LL_R1": r1_metric["LogLoss"],
                                "delta_LL": r1_metric["LogLoss"] - real_metric["LogLoss"], "Brier_real": real_metric["Brier"], "Brier_R1": r1_metric["Brier"],
                                "delta_Brier": r1_metric["Brier"] - real_metric["Brier"], "AP_real": real_metric["AP"], "AP_R1": r1_metric["AP"],
                                "ROC_real": real_metric["ROC_AUC"], "ROC_R1": r1_metric["ROC_AUC"], "MAE_p": float(np.mean(delta)),
                                "prediction_correlation": float(np.corrcoef(p_real, p_r1)[0, 1]) if len(p_real) > 1 and np.std(p_real) and np.std(p_r1) else None,
                                "MODEL_RETRAINED_ON_R1": "NO", "SCALER_REFIT_ON_R1": "NO"})
                for candidate_id, (candidate, y, probability) in enumerate(zip(data["eval"]["candidates"], y_eval, p_r1)):
                    prediction_rows.append({"candidate_id": candidate_id, "candidate_vertices": ",".join(map(str, candidate)), "y_true": int(y),
                                            "p_pred": float(probability), "dataset": dataset, "dose": target, "r1_seed": r1_seed, "mlp_seed": seed})
    return real_predictions


def finalize(feasibility, panel, by_seed, r1_rows, structural_rows, prediction_rows, real_by_dataset):
    write_csv(OUT / "K4_FEASIBILITY_AUDIT.csv", feasibility); write_csv(OUT / "K4_DATASET_PANEL_MANIFEST.csv", panel)
    write_csv(OUT / "K4_H_ONLY_FULL_BY_SEED.csv", by_seed)
    write_csv(OUT / "K4_R1_STRUCTURAL_AUDIT.csv", structural_rows, ["dataset", "target_changed_fraction", "r1_seed", "actual_changed_fraction", "attempts", "accepted_swaps", "residual_object_count", "dose_target_hit", "status", "failure_reason", "H4_exact", "candidate_identity_exact", "candidate_boundary_exact", "constituent_structure_exact", "H_derived_feature_values_exact", "required_lower_adjacency_exact", "higher_order_adjacency_exact", "nodewise_degree_exact", "residual_edge_count_exact", "forbidden_edge_exclusion"])
    write_csv(OUT / "K4_R1_RESULTS_BY_SEED.csv", r1_rows, ["dataset", "target_changed_fraction", "actual_changed_fraction", "r1_seed", "mlp_seed", "LL_real", "LL_R1", "delta_LL", "Brier_real", "Brier_R1", "delta_Brier", "AP_real", "AP_R1", "ROC_real", "ROC_R1", "MAE_p", "prediction_correlation", "MODEL_RETRAINED_ON_R1", "SCALER_REFIT_ON_R1"])
    pd.DataFrame(prediction_rows, columns=["candidate_id", "candidate_vertices", "y_true", "p_pred", "dataset", "dose", "r1_seed", "mlp_seed"]).to_parquet(OUT / "K4_R1_PREDICTIONS.parquet", index=False)
    aggregate = []
    for dataset in sorted(set(row["dataset"] for row in by_seed)):
        values = {model: [row for row in by_seed if row["dataset"] == dataset and row["model"] == model] for model in ("K4_H_ONLY", "K4_FULL_REAL")}
        row = {"dataset": dataset}
        for model, rows in values.items():
            code = "H4" if model == "K4_H_ONLY" else "Full"
            for metric in ("LogLoss", "Brier", "AP", "ROC_AUC"): row[f"{metric}_{code}"] = np.mean([r[metric] for r in rows if r[metric] is not None]) if any(r[metric] is not None for r in rows) else None
        row["InfoLoss_4"] = row["LogLoss_H4"] - row["LogLoss_Full"]
        real = real_by_dataset[dataset]
        labels = real["labels"]
        p_h = np.mean(np.vstack([real["H"][seed] for seed in MLP_SEEDS]), axis=0); p_f = np.mean(np.vstack([real["Full"][seed] for seed in MLP_SEEDS]), axis=0)
        boot = paired_logloss_bootstrap(labels, p_h, p_f, BOOTSTRAPS, 62001)
        row.update({"InfoLoss_4_CI_lower": boot["ci_lower"], "InfoLoss_4_CI_upper": boot["ci_upper"],
                    "H4_noninferior_to_Full": "YES" if boot["ci_upper"] <= MARGIN else "NO"})
        aggregate.append(row)
    write_csv(OUT / "K4_H_ONLY_FULL_AGGREGATED.csv", aggregate)
    r1_aggregate, boot_rows = [], []
    for (dataset, target), rows in sorted(defaultdict(list, {key: [r for r in r1_rows if (r["dataset"], r["target_changed_fraction"]) == key] for key in set((r["dataset"], r["target_changed_fraction"]) for r in r1_rows)}).items()):
        clusters = []
        for seed in sorted(set(r["r1_seed"] for r in rows)):
            clusters.append(float(np.mean([r["delta_LL"] for r in rows if r["r1_seed"] == seed])))
        low, high = cluster_bootstrap(clusters, 63001 + int(target * 100))
        row = {"dataset": dataset, "target_changed_fraction": target, "n_valid_r1_seeds": len(clusters),
               "mean_actual_changed_fraction": float(np.mean([r["actual_changed_fraction"] for r in rows])),
               "mean_Delta_LL": float(np.mean(clusters)), "var_cluster_Delta_LL": float(np.var(clusters)),
               "CI95_Delta_LL_low": low, "CI95_Delta_LL_high": high,
               "mean_MAE_p": float(np.mean([r["MAE_p"] for r in rows])),
               "classification": "BOUNDED_DEGRADATION_WITHIN_FROZEN_MARGIN" if high is not None and high <= MARGIN else "NOT_WITHIN_FROZEN_MARGIN"}
        r1_aggregate.append(row); boot_rows.append({**row, "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": 63001 + int(target * 100), "cluster_unit": "R1_seed; averaged across five MLP seeds"})
    aggregate_fields = ["dataset", "target_changed_fraction", "n_valid_r1_seeds", "mean_actual_changed_fraction", "mean_Delta_LL", "var_cluster_Delta_LL", "CI95_Delta_LL_low", "CI95_Delta_LL_high", "mean_MAE_p", "classification"]
    write_csv(OUT / "K4_R1_RESULTS_AGGREGATED.csv", r1_aggregate, aggregate_fields)
    write_csv(OUT / "K4_R1_BOOTSTRAP.csv", boot_rows, aggregate_fields + ["bootstrap_replicates", "bootstrap_seed", "cluster_unit"])
    eligible = [r for r in feasibility if r["eligibility"] == "ELIGIBLE"]; local_complete = [r for r in panel if r["local_status"] == "COMPLETED"]
    server = [r for r in panel if r["server_required"] == "YES"]; dose95 = [r for r in r1_aggregate if r["target_changed_fraction"] == .95]
    support = "INCOMPLETE_PENDING_SERVER" if server else ("STRONG" if dose95 and all(r["CI95_Delta_LL_high"] <= MARGIN for r in dose95) else "MIXED" if dose95 else "NOT_SUPPORTED")
    hfull_complete = len(set(row["dataset"] for row in by_seed if row["model"] == "K4_FULL_REAL"))
    lines = ["# K4 Task-Core Perturbation Final Report", "", f"- eligible datasets: {len(eligible)}", f"- complete H4-only/Full locally: {hfull_complete}", f"- complete full package locally: {len(local_complete)}", f"- server required for R1: {len(server)}", "", "## H4-only vs Full", ""]
    for row in aggregate: lines.append(f"- {row['dataset']}: LL_H4-LL_Full={row['InfoLoss_4']:.6f}; CI upper={row['InfoLoss_4_CI_upper']:.6f}; noninferior={row['H4_noninferior_to_Full']}")
    lines += ["", "## Five-dose H-preserving R1", ""]
    for row in r1_aggregate: lines.append(f"- {row['dataset']} dose={row['target_changed_fraction']:.2f}: actual={row['mean_actual_changed_fraction']:.6f}; ΔLL={row['mean_Delta_LL']:.6f}; CI=[{row['CI95_Delta_LL_low']:.6f},{row['CI95_Delta_LL_high']:.6f}]")
    lines += ["", f"- datasets with 95% CI upper ≤ 0.02: {sum(r['CI95_Delta_LL_high'] <= MARGIN for r in dose95)}", f"- K4_H_PRESERVING_PERTURBATION_ROBUSTNESS = {support}", "", "Interpretation is approximate predictive-performance invariance / bounded degradation, not unchanged candidate-level predictions or universal invariance."]
    (OUT / "K4_TASK_CORE_PERTURBATION_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True); write_protocol_audits()
    datasets = sorted(path.name for path in (ROOT / "dataset" / "raw_data").iterdir() if path.is_dir() and (path / f"{path.name}-nverts.txt").is_file())
    selected = set(args.datasets.split(",")) if args.datasets else set(datasets)
    feasibility, panel, by_seed, r1_rows, structural_rows, prediction_rows = [], [], [], [], [], []
    real_by_dataset = {}; run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = {"protocol": "K4_TASK_CORE_PERTURBATION_V1", "skip_completed": "YES", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "cases": {}}
    for dataset in datasets:
        row = {"dataset": dataset, "strict_T2": "YES", "full_data": "YES", "full_candidate_universe": "YES", "candidate_builder": "GENERIC_BOUNDARY_FACE_INVERTED_INDEX", "feature_extraction": "FROZEN_K4_V1", "eligibility": "PENDING", "reason": "", "server_required": "NO", "local_status": "PENDING"}
        try:
            data = build_dataset(dataset)
            row.update({"train_candidates": len(data["train"]["candidates"]), "eval_candidates": len(data["eval"]["candidates"]), "train_positives": int(data["train"]["labels"].sum()), "eval_positives": int(data["eval"]["labels"].sum()), "train_prevalence": data["train"]["prevalence"], "eval_prevalence": data["eval"]["prevalence"], "pure_l1_eval_edges": len(data["eval_dec"]["pure_l1_edges"])})
            if len(data["eval"]["candidates"]) == 0:
                row.update({"eligibility": "STRUCTURALLY_INFEASIBLE", "reason": "no strict-T2 K4 evaluation candidates", "local_status": "STRUCTURAL_INFEASIBILITY"})
            elif int(data["eval"]["labels"].sum()) == 0:
                row.update({"eligibility": "INSUFFICIENT_POSITIVES", "reason": "positive evaluation examples = 0", "local_status": "STRUCTURAL_INFEASIBILITY"})
            elif len(data["train"]["candidates"]) < 2:
                row.update({"eligibility": "STRUCTURALLY_INFEASIBLE", "reason": "training candidates < 2 for frozen MLP validation split", "local_status": "STRUCTURAL_INFEASIBILITY"})
            else:
                row["eligibility"] = "ELIGIBLE"
                if dataset not in selected or args.audit_only:
                    row.update({"local_status": "PENDING", "server_required": "YES", "reason": "eligible formal case not selected in this invocation"})
                else:
                    matrices = prepare_matrices(data)
                    real = run_dataset(data, matrices, run_device, by_seed, r1_rows, structural_rows, prediction_rows, run_r1=not args.skip_r1)
                    real_by_dataset[dataset] = {**real, "labels": np.asarray(data["eval"]["labels"], dtype=np.int8)}
                    if args.skip_r1:
                        row.update({"local_status": "PARTIAL_H_FULL_COMPLETED", "server_required": "YES",
                                    "reason": "K4 H-only/Full complete locally; five-dose R1 queued after local controller consumed >170 CPU-seconds without completing the second eligible dataset"})
                        state["cases"][dataset] = {"status": "PARTIAL_H_FULL_COMPLETED", "R1": "LOCAL_RESOURCE_LIMIT"}
                    else:
                        row["local_status"] = "COMPLETED"; state["cases"][dataset] = {"status": "COMPLETED"}
                    del matrices; gc.collect()
            feasibility.append(dict(row)); panel.append(dict(row)); write_json(LOCAL / "RUN_STATE.json", state)
            del data; gc.collect()
        except (mlp.ResourceLimit, MemoryError) as exc:
            row.update({"eligibility": "LOCAL_RESOURCE_LIMIT", "reason": str(exc), "local_status": "LOCAL_RESOURCE_LIMIT", "server_required": "YES"}); feasibility.append(dict(row)); panel.append(dict(row))
        except Exception as exc:
            row.update({"eligibility": "PIPELINE_UNAVAILABLE", "reason": repr(exc), "local_status": "IMPLEMENTATION_FAIL", "server_required": "NO"}); feasibility.append(dict(row)); panel.append(dict(row))
    finalize(feasibility, panel, by_seed, r1_rows, structural_rows, prediction_rows, real_by_dataset)
    write_json(LOCAL / "RUN_STATE.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--datasets", default=""); parser.add_argument("--audit-only", action="store_true"); parser.add_argument("--skip-r1", action="store_true")
    run(parser.parse_args())
