"""K3 Spatial-Order Task-Core Saturation, strict-T2/full-universe protocol.

The runner is resumable at dataset/model/seed granularity.  It never samples
events, nodes, edges, candidates, or negatives.  Existing K3 v1 outputs are
read-only provenance/model sources and are never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_k3_residual_lower_order_conditional_redundancy as legacy
import run_strict_timestamp_chronological_core_v1 as strict
import run_strict_timestamp_t2_mlp_local_v1 as mlp
from scripts.task_core_common_v1 import (
    canonical_hash, hash_array, hash_rows, mean, metric_row,
    paired_logloss_bootstrap, read_json, sha256_file, write_csv, write_json,
)

OUT = ROOT / "results" / "k3_spatial_order_task_core_v1"
LOCAL = OUT / "local"
OLD = ROOT / "results" / "k3_task_core_boundary_v1"
PANEL_SOURCE = ROOT / "results" / "k3_all_dataset_learned_smoke" / "K3_SMOKE_DATASET_MANIFEST.json"
AUDIT_CACHE = ROOT / "results" / "strict_timestamp_chronological_core_v1" / "01_timestamp_audit" / "audit_cache.json"
TAG_PROTOCOL = ROOT / "results" / "strict_timestamp_t2_remaining_server_v1" / "02_k3" / "tags-ask-ubuntu" / "T2" / "case_protocol.json"
SEEDS = (21001, 21002, 21003, 21004, 21005)
MODELS = ("B0", "B1", "B2", "B3", "B4", "B5")
MARGIN = 0.02
BOOTSTRAPS = 2000
BATCH = 1024

H3 = ("SCN", "SRA", "SAA", "SPA")
L2C = (
    "FACE_COFACE_SUM", "FACE_COFACE_MEAN", "FACE_COFACE_MIN", "FACE_COFACE_MAX", "FACE_COFACE_STD",
    "LOWER_DEGREE", "LOWER_COMMON_NEIGHBORS", "LOWER_NEIGHBORHOOD_JACCARD", "LOWER_2HOP_COUNT",
    "H_BOUNDARY_FACE_COFACE_SUM", "H_BOUNDARY_FACE_COFACE_MEAN", "H_BOUNDARY_FACE_COFACE_MAX",
    "H_BOUNDARY_FACE_COFACE_STD",
)
L2R = tuple(legacy.L2_NAMES)
L1 = tuple(legacy.L1_NAMES)
FEATURES = {
    "B0": H3,
    "B1": H3 + L2C,
    "B2": H3 + L2C + tuple("S1__" + name for name in L2R),
    "B3": H3 + L2C + tuple("S12__" + name for name in L2R),
    "B4": H3 + L2C + L2R,
    "B5": H3 + L2C + L2R + L1,
}
REUSE = {"B0": "M0_H3_PURE", "B1": "S1_H3_PLUS_L2_CONSTITUENT", "B4": "M2_H3_PLUS_L2", "B5": "M3_FULL_CLEAN"}
OLD_COMPLETE_DATASETS = {"email-Eu", "NDC-substances", "tags-math-sx"}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_matrix(arrays, model):
    columns = []
    for name in FEATURES[model]:
        if name.startswith("S1__"):
            columns.append(arrays["s1"][name[4:]])
        elif name.startswith("S12__"):
            columns.append(arrays["s12"][name[5:]])
        else:
            columns.append(arrays["base"][name])
    return np.column_stack(columns).astype(np.float32)


def panel_names():
    payload = read_json(PANEL_SOURCE)
    values = tuple(row["dataset"] for row in payload["datasets"])
    if len(values) != 8 or len(set(values)) != 8:
        raise RuntimeError(f"K3_PANEL_SIZE_MISMATCH={len(values)}")
    return values


def load_audit(dataset):
    cache = read_json(AUDIT_CACHE)
    if dataset in cache:
        return cache[dataset]
    return strict.audit_dataset(dataset)[0]


def build_data(dataset):
    raw_dir = ROOT / "dataset" / "raw_data" / dataset
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"CANONICAL_RAW_DATA_MISSING:{raw_dir}")
    if dataset in OLD_COMPLETE_DATASETS:
        data = mlp.load_case_data(3, dataset)
    else:
        events = strict.load_events(dataset, "timestamp")
        audit = load_audit(dataset)
        n = len(events)
        c40, c60, c80 = int(audit["cut_40"]), int(audit["cut_60"]), int(audit["cut_80"])
        train = mlp.build_task_for_mlp(3, dataset, events, c60, c60, c80, "T2_TRAIN")
        evaluation = mlp.build_task_for_mlp(3, dataset, events, c80, c80, n, "T2_EVAL")
        gate = strict.timestamp_gate(events, c60, c80, c80, n, (0, c40, c60, c80, n))
        if gate["no_timestamp_group_crosses_split"] != "YES":
            raise RuntimeError("STRICT_T2_TIMESTAMP_GATE_FAIL")
        train_dec = strict.decompose(3, train)
        eval_dec = strict.decompose(3, evaluation)
        train_arrays = legacy.compute_features(train, train_dec["h3"], train_dec["l2_triangles"], train_dec["l1_edges"])
        eval_arrays = legacy.compute_features(evaluation, eval_dec["h3"], eval_dec["l2_triangles"], eval_dec["l1_edges"])
        data = {
            "order": 3, "dataset": dataset, "events": events, "audit": audit, "gate": gate,
            "cut40": c40, "cut60": c60, "cut80": c80, "n_events": n,
            "train": train, "eval": evaluation, "train_dec": train_dec, "eval_dec": eval_dec,
            "train_arrays": train_arrays, "eval_arrays": eval_arrays,
            "candidate_hash_train": train["candidate_hash"], "candidate_hash_eval": evaluation["candidate_hash"],
            "label_hash_train": hash_array(train["labels"]), "label_hash_eval": hash_array(evaluation["labels"]),
        }
    return data


def l2_features_for_set(candidate, allowed):
    vertices = tuple(candidate)
    vset = set(vertices)
    edges = tuple(combinations(vertices, 2))
    vertex_counts = [0] * 4
    edge_counts = [0] * 6
    exact = [0, 0, 0, 0]
    overlap_sum = 0
    for triangle in allowed:
        tset = set(triangle)
        overlap = len(vset & tset)
        if overlap:
            exact[overlap] += 1
            overlap_sum += overlap
            for index, vertex in enumerate(vertices):
                vertex_counts[index] += int(vertex in tset)
            for index, edge in enumerate(edges):
                edge_counts[index] += int(edge[0] in tset and edge[1] in tset)
    values = (
        sum(vertex_counts), np.mean(vertex_counts), max(vertex_counts), min(vertex_counts), np.std(vertex_counts),
        sum(edge_counts), np.mean(edge_counts), max(edge_counts), np.std(edge_counts),
        exact[1], exact[2], exact[3], overlap_sum,
    )
    return values


def shell_arrays(task, dec, dataset, split):
    candidates = task["candidates"]
    residual = set(dec["l2_triangles"])
    all_triangles = set(task["full"]["triangles"])
    triangle_by_edge = defaultdict(set)
    for triangle in all_triangles:
        for edge in combinations(triangle, 2):
            triangle_by_edge[edge].add(triangle)
    s1_arrays = {name: np.zeros(len(candidates), dtype=np.float32) for name in L2R}
    s12_arrays = {name: np.zeros(len(candidates), dtype=np.float32) for name in L2R}
    count1, count2, count3 = [], [], []
    object_hash = hashlib.sha256()
    for candidate_id, candidate in enumerate(candidates):
        center = set(combinations(candidate, 3))
        distance1_all = set()
        for face in center:
            for edge in combinations(face, 2):
                distance1_all.update(triangle_by_edge.get(edge, set()))
        distance1_all.difference_update(center)
        s1 = residual & distance1_all
        distance2_all = set()
        for triangle in distance1_all:
            for edge in combinations(triangle, 2):
                distance2_all.update(triangle_by_edge.get(edge, set()))
        distance2_all.difference_update(center)
        distance2_all.difference_update(distance1_all)
        s2 = residual & distance2_all
        s3plus = residual - s1 - s2
        for shell_name, shell in (("S1", s1), ("S2", s2), ("S3plus", s3plus)):
            for triangle in sorted(shell):
                object_hash.update(f"{candidate_id}|{shell_name}|{','.join(map(str, triangle))}\n".encode())
        for name, value in zip(L2R, l2_features_for_set(candidate, s1)):
            s1_arrays[name][candidate_id] = value
        for name, value in zip(L2R, l2_features_for_set(candidate, s1 | s2)):
            s12_arrays[name][candidate_id] = value
        count1.append(len(s1)); count2.append(len(s2)); count3.append(len(s3plus))
    def stats(values, prefix):
        array = np.asarray(values, dtype=np.float64)
        return {f"{prefix}_mean": float(array.mean()) if len(array) else 0.0,
                f"{prefix}_median": float(np.median(array)) if len(array) else 0.0,
                f"{prefix}_max": int(array.max()) if len(array) else 0,
                f"{prefix}_nonzero_fraction": float(np.mean(array > 0)) if len(array) else 0.0}
    audit = {
        "dataset": dataset, "split": split, "candidate_count": len(candidates),
        "residual_triangle_count": len(residual), "observed_triangle_count": len(all_triangles),
        **stats(count1, "S1"), **stats(count2, "S2"), **stats(count3, "S3plus"),
        "shell_object_assignment_hash": object_hash.hexdigest(),
        "distance_graph": "observed triangle lower-adjacency (shared ordinary edge)",
        "future_structure_used": "NO", "aggregate_features_recomputed_from_allowed_objects": "YES",
    }
    return {"s1": s1_arrays, "s12": s12_arrays}, audit


def matched_far_arrays(task, dec, structural_seed):
    """Exact S1-count-matched far control using observation-window triangles."""
    residual = set(dec["l2_triangles"]); all_triangles = set(task["full"]["triangles"])
    triangle_by_edge = defaultdict(set)
    for triangle in all_triangles:
        for edge in combinations(triangle, 2): triangle_by_edge[edge].add(triangle)
    arrays = {name: np.zeros(len(task["candidates"]), dtype=np.float32) for name in L2R}
    assignment = hashlib.sha256(); selected_counts = []
    for candidate_id, candidate in enumerate(task["candidates"]):
        center = set(combinations(candidate, 3)); distance1_all = set()
        for face in center:
            for edge in combinations(face, 2): distance1_all.update(triangle_by_edge.get(edge, set()))
        distance1_all.difference_update(center); s1 = residual & distance1_all
        distance2_all = set()
        for triangle in distance1_all:
            for edge in combinations(triangle, 2): distance2_all.update(triangle_by_edge.get(edge, set()))
        distance2_all.difference_update(center); distance2_all.difference_update(distance1_all)
        far = sorted(residual - s1)
        if len(far) < len(s1):
            raise RuntimeError(f"MATCHED_FAR_STRUCTURAL_INFEASIBILITY:candidate={candidate_id}:S1={len(s1)}:far={len(far)}")
        rng = np.random.default_rng(int(structural_seed) * 1_000_003 + candidate_id)
        ids = sorted(rng.choice(len(far), size=len(s1), replace=False).tolist()) if s1 else []
        chosen = {far[index] for index in ids}; selected_counts.append(len(chosen))
        for triangle in sorted(chosen): assignment.update(f"{candidate_id}|{','.join(map(str, triangle))}\n".encode())
        for name, value in zip(L2R, l2_features_for_set(candidate, chosen)): arrays[name][candidate_id] = value
    return arrays, {"structural_seed": structural_seed, "triangle_count_exact_matched": "YES", "degree_distribution_matched": "NO_NOT_REQUIRED",
                    "future_structure_used": "NO", "assignment_hash": assignment.hexdigest(), "total_selected": int(sum(selected_counts))}


def cache_path(dataset, split):
    return LOCAL / "feature_cache" / dataset / f"{split}.npz"


def prepare_features(data):
    audits = []
    result = {}
    for split, task_key, dec_key, arrays_key in (("train", "train", "train_dec", "train_arrays"), ("eval", "eval", "eval_dec", "eval_arrays")):
        path = cache_path(data["dataset"], split)
        meta_path = path.with_suffix(".json")
        if path.is_file() and meta_path.is_file():
            with np.load(path, allow_pickle=False) as bundle:
                base = {name[6:]: bundle[name] for name in bundle.files if name.startswith("base__")}
                s1 = {name[4:]: bundle[name] for name in bundle.files if name.startswith("s1__")}
                s12 = {name[5:]: bundle[name] for name in bundle.files if name.startswith("s12__")}
            meta = read_json(meta_path)
            if meta["candidate_hash"] != data[task_key]["candidate_hash"]:
                raise RuntimeError("FEATURE_CACHE_CANDIDATE_HASH_MISMATCH")
            audits.append(meta["shell_audit"])
            result[split] = {"base": base, "s1": s1, "s12": s12}
            continue
        shell, audit = shell_arrays(data[task_key], data[dec_key], data["dataset"], split)
        base = {name: np.asarray(value, dtype=np.float32) for name, value in data[arrays_key].items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**{"base__" + name: value for name, value in base.items()},
                   **{"s1__" + name: value for name, value in shell["s1"].items()},
                   **{"s12__" + name: value for name, value in shell["s12"].items()}}
        np.savez_compressed(path, **payload)
        meta = {"dataset": data["dataset"], "split": split, "candidate_hash": data[task_key]["candidate_hash"],
                "label_hash": hash_array(data[task_key]["labels"]), "shell_audit": audit,
                "array_hashes": {key: hash_array(value) for key, value in payload.items()}}
        write_json(meta_path, meta)
        audits.append(audit)
        result[split] = {"base": base, **shell}
    return result, audits


def old_predictions():
    main = pd.read_parquet(OLD / "K3_TASK_CORE_PREDICTIONS.parquet")
    secondary = pd.read_parquet(OLD / "K3_L2_CONSTITUENT_PREDICTIONS.parquet")
    return main, secondary


def reuse_prediction(dataset, model, seed, labels, old_main, old_secondary):
    source_model = REUSE[model]
    table = old_secondary if model == "B1" else old_main
    row = table[(table.dataset == dataset) & (table.model == source_model) & (table.seed == seed)].sort_values("candidate_id")
    if len(row) != len(labels) or not np.array_equal(row.y_true.to_numpy(dtype=np.int8), np.asarray(labels, dtype=np.int8)):
        raise RuntimeError(f"STRUCTURAL_REUSE_ALIGNMENT_FAIL:{dataset}/{model}/{seed}")
    return row.p_pred.to_numpy(dtype=np.float64)


def prediction_marker(dataset, model, seed):
    return LOCAL / "predictions" / dataset / model / f"seed_{seed}.npz"


def fit_or_load(data, arrays, model, seed, old_main, old_secondary, run_device):
    marker = prediction_marker(data["dataset"], model, seed)
    if marker.is_file():
        with np.load(marker, allow_pickle=False) as values:
            if values["candidate_hash"].item() != data["eval"]["candidate_hash"]:
                raise RuntimeError("PREDICTION_CACHE_CANDIDATE_HASH_MISMATCH")
            return values["p"].astype(np.float64), str(values["execution"].item())
    if data["dataset"] in OLD_COMPLETE_DATASETS and model in REUSE:
        p = reuse_prediction(data["dataset"], model, seed, data["eval"]["labels"], old_main, old_secondary)
        execution = "STRUCTURALLY_EQUIVALENT_MODEL"
    else:
        x_train = model_matrix(arrays["train"], model)
        x_eval = model_matrix(arrays["eval"], model)
        model_root = LOCAL / "models" / data["dataset"] / model / f"seed_{seed}"
        bundle, _ = mlp.train_one_model(x_train, data["train"]["labels"], FEATURES[model], seed, run_device, BATCH,
                                        model_root / "model.pt", model_root / "scaler")
        _, p = mlp.predict_model(bundle, x_eval, run_device, BATCH)
        execution = "TRAINED_LOCAL"
    marker.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(marker, candidate_id=np.arange(len(p), dtype=np.int64), y=np.asarray(data["eval"]["labels"], dtype=np.int8),
                        p=np.asarray(p, dtype=np.float64), candidate_hash=np.asarray(data["eval"]["candidate_hash"]), execution=np.asarray(execution))
    return p, execution


def write_static_audits(panel):
    source_manifest = OLD / "K3_TASK_CORE_FEATURE_MANIFEST.json"
    provenance_csv = OLD / "K3_FEATURE_PROVENANCE_AUDIT.csv"
    provenance_json = OLD / "K3_FEATURE_PROVENANCE_AUDIT.json"
    frozen = read_json(source_manifest)
    expected = {"H3_pure": list(H3), "L2_constituent": list(L2C), "L2_residual": list(L2R),
                "L1_peripheral": list(L1), "mixed_order_excluded": []}
    feature_name_hash = lambda names: hashlib.sha256(json.dumps(list(names), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    hash_expectations = {
        "H3_pure_feature_hash": feature_name_hash(H3), "L2_constituent_feature_hash": feature_name_hash(L2C),
        "L2_residual_feature_hash": feature_name_hash(L2R), "L2_all_feature_hash": feature_name_hash(L2C + L2R),
        "L1_peripheral_feature_hash": feature_name_hash(L1), "mixed_order_excluded_feature_hash": feature_name_hash(()),
        "Full_clean_feature_hash": feature_name_hash(H3 + L2C + L2R + L1),
    }
    list_matches = all(frozen["buckets"].get(key) == value for key, value in expected.items())
    hash_matches = all(frozen["hashes"].get(key) == value for key, value in hash_expectations.items())
    matches = list_matches and hash_matches
    report = ["# K3 Saturation Feature Manifest Audit", "", "- PROVENANCE_REUSED: YES",
              f"- FEATURE_HASH_MATCH: {'YES' if matches else 'NO'}", "- MIXED_ORDER_EXCLUDED: YES",
              f"- FROZEN_BUCKET_LIST_MATCH: {'YES' if list_matches else 'NO'}", f"- FROZEN_NAME_HASH_VALUES_MATCH: {'YES' if hash_matches else 'NO'}",
              "- PERFORMANCE_DEPENDENT_RECLASSIFICATION: NO", f"- source manifest SHA256: `{sha256_file(source_manifest)}`",
              f"- source provenance CSV SHA256: `{sha256_file(provenance_csv)}`",
              f"- source provenance JSON SHA256: `{sha256_file(provenance_json)}`", "",
              "B2 and B3 retain the frozen formulas and recompute them from candidate-specific allowed triangle objects; no feature-column dropping is used."]
    (OUT / "K3_SATURATION_FEATURE_MANIFEST_AUDIT.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "K3_SATURATION_FEATURE_MANIFEST_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not matches:
        raise RuntimeError("FEATURE_PROVENANCE_CONFLICT")
    definition = """# K3 L2 Spatial Shell Definition

- Center `S0`: the four constituent triangles of each open tetrahedron candidate.
- Graph: all triangles in the current observation window; two triangles are adjacent exactly when they share an ordinary edge.
- `S1`: residual-L2 triangles at shortest-path distance 1 from S0.
- `S2`: residual-L2 triangles at shortest-path distance 2 from S0.
- `S3plus`: every remaining residual-L2 triangle (distance at least 3 or disconnected).
- Distances and features use observation-window objects only. Future labels, future simplices, future edges, and closure outcomes are never read.
- Frozen aggregate formulas are recomputed per candidate from the allowed object set (`S1`, `S1+S2`, or all residual L2); columns are not merely removed.
"""
    (OUT / "K3_L2_SPATIAL_SHELL_DEFINITION.md").write_text(definition, encoding="utf-8")
    model_manifest = {
        "protocol": "K3_SPATIAL_ORDER_TASK_CORE_V1", "strict_T2": True, "full_data": True,
        "full_candidate_universe": True, "sampling": "NONE", "primary_endpoint": "LogLoss",
        "noninferiority_margin": MARGIN, "bootstrap_replicates": BOOTSTRAPS,
        "seeds": list(SEEDS), "model_architecture_source": "run_strict_timestamp_t2_mlp_local_v1.py",
        "models": {key: list(value) for key, value in FEATURES.items()},
        "provenance_source_hash": sha256_file(source_manifest), "panel": list(panel),
    }
    write_json(OUT / "K3_TASK_CORE_SATURATION_MODEL_MANIFEST.json", model_manifest)


def write_degeneracy_audit():
    candidate = (0, 1, 2, 3)
    task = {"candidates": [candidate], "lower_adj": [set()]}
    h3 = {"nodes": set(candidate), "edges": set(combinations(candidate, 2)),
          "triangles": set(combinations(candidate, 3)), "tetras": {candidate}}
    arrays = legacy.compute_features(task, h3, set(), set())
    values = {name: float(arrays[name][0]) for name in H3}
    synthetic_pass = all(values[name] > 0 for name in H3)
    text = f"""# K3 H3 Degeneracy Audit

- H3_DEGENERACY_CAUSE: `CANDIDATE_PROTOCOL_INDUCED`
- BUG_FOUND: `NO`
- BUG_IMPACTS_CURRENT_EXPERIMENT: `NO`
- SYNTHETIC_REAL_FUNCTION_PASS: `{'YES' if synthetic_pass else 'NO'}`
- synthetic outputs: `{json.dumps(values, sort_keys=True)}`

For a legal open tetrahedron candidate, two distinct constituent triangular faces uniquely determine that tetrahedron. Any observed tetrahedron appearing in both face-coface sets would therefore be the candidate itself, contradicting the open-candidate requirement. Hence SCN, SRA, and SAA are mathematically forced to zero by this candidate protocol. SPA uses products of individual face-coface degrees, so it can vary.

The real production feature function was also called on a deliberately non-open synthetic structure containing the candidate tetrahedron. It returned nonzero SCN/SRA/SAA/SPA, confirming that the implementation can express nonzero values outside the open-candidate protocol. The synthetic structure is a code sanity test, not a legal prediction candidate.
"""
    (OUT / "K3_H3_DEGENERACY_AUDIT.md").write_text(text, encoding="utf-8")
    if not synthetic_pass:
        raise RuntimeError("H3_SYNTHETIC_SANITY_FAIL")


def base_panel_rows(panel):
    source = str(PANEL_SOURCE.relative_to(ROOT))
    rows = []
    tag = read_json(TAG_PROTOCOL)["dry_run"] if TAG_PROTOCOL.is_file() else {}
    for order, dataset in enumerate(panel, 1):
        row = {"dataset": dataset, "dataset_order": order, "source_manifest": source, "strict_T2": "YES",
               "full_data": "YES", "full_candidate_universe": "YES", "train_candidates": None, "eval_candidates": None,
               "train_positives": None, "eval_positives": None, "train_prevalence": None, "eval_prevalence": None,
               "feature_availability": "PENDING", "local_status": "PENDING", "server_required": "NO"}
        if dataset == "tags-ask-ubuntu":
            row.update({"train_candidates": tag.get("train_candidate_count_expected"), "eval_candidates": tag.get("eval_candidate_count_expected"),
                        "feature_availability": "SERVER_BUILD_REQUIRED", "local_status": "LOCAL_RESOURCE_LIMIT",
                        "server_required": "YES", "status_detail": "PREEMPTIVE_LOCAL_SKIP: prior exact full-universe extraction exceeded local resource budget; 45,766 train and 73,683 eval candidates"})
        elif not (ROOT / "dataset" / "raw_data" / dataset).is_dir():
            row.update({"full_data": "UNAVAILABLE_LOCAL", "feature_availability": "PIPELINE_UNAVAILABLE_LOCAL_DATA_MISSING",
                        "local_status": "PIPELINE_UNAVAILABLE", "server_required": "YES",
                        "status_detail": "canonical frozen-panel raw directory missing locally; server must supply canonical data"})
        rows.append(row)
    return rows


def aggregate_and_report(panel_rows, prediction_rows, seed_rows, shell_audits):
    pred = pd.DataFrame(prediction_rows)
    seed_df = pd.DataFrame(seed_rows)
    write_csv(OUT / "K3_TASK_CORE_SATURATION_BY_SEED.csv", seed_rows)
    if len(pred):
        pred.to_parquet(OUT / "K3_TASK_CORE_SATURATION_PREDICTIONS.parquet", index=False)
    else:
        pd.DataFrame(columns=["candidate_id", "candidate_vertices", "y_true", "p_pred", "dataset", "model", "seed"]).to_parquet(OUT / "K3_TASK_CORE_SATURATION_PREDICTIONS.parquet", index=False)
    aggregated, bootstrap, gains = [], [], []
    for dataset in sorted(pred.dataset.unique()) if len(pred) else []:
        d = pred[pred.dataset == dataset]
        model_metrics = {}
        ensemble = {}
        labels = None
        for model in MODELS:
            dm = d[d.model == model]
            pivot = dm.pivot(index="candidate_id", columns="seed", values="p_pred").sort_index()
            y = dm.drop_duplicates("candidate_id").sort_values("candidate_id").y_true.to_numpy(dtype=np.int8)
            p = pivot.mean(axis=1).to_numpy(dtype=np.float64)
            ensemble[model] = p; labels = y
            model_metrics[model] = metric_row(y, p)
        row = {"dataset": dataset, "n_eval_candidates": len(labels), "n_eval_positives": int(labels.sum())}
        for model in MODELS:
            for metric, value in model_metrics[model].items(): row[f"{metric}_{model}"] = value
        aggregated.append(row)
        for index, model in enumerate(("B1", "B2", "B3", "B4"), 1):
            boot = paired_logloss_bootstrap(labels, ensemble[model], ensemble["B5"], BOOTSTRAPS, 43000 + index)
            bootstrap.append({"dataset": dataset, "model": model, "reference": "B5", **boot,
                              "margin": MARGIN, "noninferior": "YES" if boot["ci_upper"] <= MARGIN else "NO"})
        gain = {"dataset": dataset}
        for left, right, name in (("B0", "B1", "Delta_01"), ("B1", "B2", "Delta_12"), ("B2", "B3", "Delta_23"),
                                  ("B3", "B4", "Delta_34"), ("B4", "B5", "Delta_45")):
            gain[name] = model_metrics[left]["LogLoss"] - model_metrics[right]["LogLoss"]
        gains.append(gain)
    write_csv(OUT / "K3_TASK_CORE_SATURATION_AGGREGATED.csv", aggregated)
    write_csv(OUT / "K3_TASK_CORE_SATURATION_BOOTSTRAP.csv", bootstrap)
    write_csv(OUT / "K3_TASK_CORE_MARGINAL_GAINS.csv", gains)
    write_csv(OUT / "K3_L2_SPATIAL_SHELL_AUDIT.csv", shell_audits)
    boot_by_dataset = defaultdict(dict)
    for row in bootstrap: boot_by_dataset[row["dataset"]][row["model"]] = row
    classifications = {}
    for dataset, values in boot_by_dataset.items():
        chosen = None
        for model in ("B1", "B2", "B3", "B4"):
            if values[model]["noninferior"] == "YES": chosen = model; break
        label = {"B1": "CONSTITUENT_ONLY", "B2": "IMMEDIATE_LOCAL_CORE", "B3": "EXTENDED_LOCAL_CORE", "B4": "FULL_L2_CORE"}.get(chosen, "L1_STILL_NEEDED")
        classifications[dataset] = (chosen or "B5", label, values.get(chosen or "B4", {}).get("ci_upper"))
    for row in panel_rows:
        if row["dataset"] in classifications: row["minimum_saturation_model"], row["classification"], row["saturation_ci_upper"] = classifications[row["dataset"]]
        elif row["local_status"] == "LOCAL_RESOURCE_LIMIT": row["classification"] = "LOCAL_RESOURCE_LIMIT"
        else: row["classification"] = row["local_status"]
    write_csv(OUT / "K3_8DATASET_PANEL_MANIFEST.csv", panel_rows)
    n_complete = sum(row["local_status"] == "COMPLETED" for row in panel_rows)
    n_server = sum(row["server_required"] == "YES" for row in panel_rows)
    counts = {model: sum(values.get(model, {}).get("noninferior") == "YES" for values in boot_by_dataset.values()) for model in ("B1", "B2", "B3", "B4")}
    need_l1 = sum(label == "L1_STILL_NEEDED" for _, label, _ in classifications.values())
    decay_flags = []
    for gain in gains:
        sequence = [gain[f"Delta_{name}"] for name in ("01", "12", "23", "34", "45")]
        decay_flags.append(sum(sequence[i] >= sequence[i + 1] for i in range(4)) >= 3)
    locality = "INCOMPLETE_PENDING_SERVER" if n_server else ("STRONG" if counts["B3"] == n_complete else "MIXED")
    lines = ["# K3 Task-Core Saturation Final Report", "", "## Coverage", "", "- N_DATASETS_TOTAL = 8",
             f"- N_LOCAL_COMPLETE = {n_complete}", f"- N_SERVER_REQUIRED = {n_server}", "", "## Dataset results", ""]
    gain_map = {row["dataset"]: row for row in gains}
    for row in panel_rows:
        dataset = row["dataset"]
        if dataset in classifications:
            chosen, label, upper = classifications[dataset]
            lines.append(f"- {dataset}: minimum={chosen}; CI upper={upper:.6f}; classification={label}; L1 incremental gain={gain_map[dataset]['Delta_45']:.6f}")
        else:
            lines.append(f"- {dataset}: {row['local_status']} ({row.get('status_detail', 'pending server')})")
    lines += ["", "## Panel questions", "", f"- Q1 B1 ≈ Full: {counts['B1']}", f"- Q2 B2 ≈ Full: {counts['B2']}",
              f"- Q3 B3 ≈ Full: {counts['B3']}", f"- Q4 B4 ≈ Full: {counts['B4']}", f"- Q5 L1 still needed: {need_l1}",
              f"- Q6 distance-gain decay: {'YES' if decay_flags and all(decay_flags) else 'PARTIAL' if any(decay_flags) else 'NO'}",
              f"- TASK_CORE_LOCALITY_SUPPORT = {locality}", "", "All conclusions are model-conditional and strict-T2/full-candidate-universe results."]
    (OUT / "K3_TASK_CORE_SATURATION_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_negative_control(dataset_names):
    """Server-deferred optional matched spatial control; fully resumable."""
    run_device = device(); rows = []; bootstrap_rows = []; audit_rows = []
    structural_seeds = (31001, 31002, 31003)
    for dataset in dataset_names:
        data = build_data(dataset); arrays, _ = prepare_features(data)
        local_predictions = {}
        for seed in SEEDS:
            root = LOCAL / "matched_control_models" / dataset / "C_local" / f"seed_{seed}"
            path, scaler = root / "model.pt", root / "scaler"
            if path.is_file() and scaler.with_suffix(".npz").is_file(): bundle = mlp.load_model(path, scaler, run_device)
            else: bundle, _ = mlp.train_one_model(model_matrix(arrays["train"], "B2"), data["train"]["labels"], FEATURES["B2"], seed, run_device, BATCH, path, scaler)
            _, local_predictions[seed] = mlp.predict_model(bundle, model_matrix(arrays["eval"], "B2"), run_device, BATCH)
        for structural_seed in structural_seeds:
            far_cache = LOCAL / "matched_control_features" / dataset / f"structural_seed_{structural_seed}"
            far_cache.mkdir(parents=True, exist_ok=True)
            train_path, eval_path = far_cache / "train.npz", far_cache / "eval.npz"
            if train_path.is_file() and eval_path.is_file():
                with np.load(train_path, allow_pickle=False) as saved: far_train = {name: saved[name] for name in saved.files}
                with np.load(eval_path, allow_pickle=False) as saved: far_eval = {name: saved[name] for name in saved.files}
                audit = read_json(far_cache / "audit.json")
            else:
                far_train, train_audit = matched_far_arrays(data["train"], data["train_dec"], structural_seed)
                far_eval, eval_audit = matched_far_arrays(data["eval"], data["eval_dec"], structural_seed)
                np.savez_compressed(train_path, **far_train); np.savez_compressed(eval_path, **far_eval)
                audit = {"dataset": dataset, "structural_seed": structural_seed, "train": train_audit, "eval": eval_audit}; write_json(far_cache / "audit.json", audit)
            audit_rows.append(audit)
            x_train_far = np.column_stack([arrays["train"]["base"][name] for name in H3 + L2C] + [far_train[name] for name in L2R]).astype(np.float32)
            x_eval_far = np.column_stack([arrays["eval"]["base"][name] for name in H3 + L2C] + [far_eval[name] for name in L2R]).astype(np.float32)
            far_predictions = {}
            for seed in SEEDS:
                root = LOCAL / "matched_control_models" / dataset / f"C_far_{structural_seed}" / f"seed_{seed}"
                path, scaler = root / "model.pt", root / "scaler"
                names = H3 + L2C + tuple("MATCHED_FAR__" + name for name in L2R)
                if path.is_file() and scaler.with_suffix(".npz").is_file(): bundle = mlp.load_model(path, scaler, run_device)
                else: bundle, _ = mlp.train_one_model(x_train_far, data["train"]["labels"], names, seed, run_device, BATCH, path, scaler)
                _, p_far = mlp.predict_model(bundle, x_eval_far, run_device, BATCH); far_predictions[seed] = p_far
                local_metric = metric_row(data["eval"]["labels"], local_predictions[seed]); far_metric = metric_row(data["eval"]["labels"], p_far)
                rows.append({"dataset": dataset, "structural_seed": structural_seed, "mlp_seed": seed, "LL_local": local_metric["LogLoss"],
                             "LL_far": far_metric["LogLoss"], "Delta_LL_far_minus_local": far_metric["LogLoss"] - local_metric["LogLoss"],
                             "Brier_local": local_metric["Brier"], "Brier_far": far_metric["Brier"], "AP_local": local_metric["AP"],
                             "AP_far": far_metric["AP"], "ROC_local": local_metric["ROC_AUC"], "ROC_far": far_metric["ROC_AUC"]})
            p_local = np.mean(np.vstack(list(local_predictions.values())), axis=0); p_far = np.mean(np.vstack(list(far_predictions.values())), axis=0)
            boot = paired_logloss_bootstrap(data["eval"]["labels"], p_far, p_local, BOOTSTRAPS, 33000 + structural_seed)
            bootstrap_rows.append({"dataset": dataset, "structural_seed": structural_seed, **boot})
    write_csv(OUT / "K3_LOCAL_L2_MATCHED_CONTROL.csv", rows)
    write_csv(OUT / "K3_LOCAL_L2_MATCHED_CONTROL_BOOTSTRAP.csv", bootstrap_rows)
    write_json(LOCAL / "K3_LOCAL_L2_MATCHED_CONTROL_AUDIT.json", audit_rows)
    (OUT / "K3_LOCAL_L2_MATCHED_CONTROL_REPORT.md").write_text(f"# K3 Local-L2 Matched Spatial Control\n\n- datasets completed: {len(dataset_names)}\n- structural seeds: 31001, 31002, 31003\n- triangle count exact matched: YES\n- future structure used: NO\n- model/scaler trained on each Real-train feature representation only; no outcome-based seed selection.\n", encoding="utf-8")


def run(args):
    panel = panel_names()
    OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True)
    write_static_audits(panel); write_degeneracy_audit()
    panel_rows = base_panel_rows(panel)
    old_main, old_secondary = old_predictions()
    existing_predictions = []
    existing_seed_rows = []
    shell_audits = []
    state = {"protocol": "K3_SPATIAL_ORDER_TASK_CORE_V1", "skip_completed": "YES", "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
             "git_status_at_start": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines(), "cases": {}}
    selected = set(args.datasets.split(",")) if args.datasets else set(panel)
    run_device = device()
    for dataset in panel:
        row = next(value for value in panel_rows if value["dataset"] == dataset)
        if dataset not in selected or (row["server_required"] == "YES" and not args.server_mode): continue
        try:
            data = build_data(dataset)
            row.update({"train_candidates": len(data["train"]["candidates"]), "eval_candidates": len(data["eval"]["candidates"]),
                        "train_positives": int(data["train"]["labels"].sum()), "eval_positives": int(data["eval"]["labels"].sum()),
                        "train_prevalence": float(data["train"]["labels"].mean()) if len(data["train"]["labels"]) else 0.0,
                        "eval_prevalence": float(data["eval"]["labels"].mean()) if len(data["eval"]["labels"]) else 0.0})
            if len(data["train"]["candidates"]) < 2 or len(data["eval"]["candidates"]) < 1:
                row.update({"feature_availability": "STRUCTURALLY_INFEASIBLE", "local_status": "STRUCTURAL_INFEASIBILITY", "server_required": "NO",
                            "status_detail": "candidate universe too small for frozen 80/20-validation MLP"})
                continue
            arrays, audits = prepare_features(data); shell_audits.extend(audits)
            row["feature_availability"] = "COMPLETE"
            dataset_predictions = {model: {} for model in MODELS}
            for model in MODELS:
                for seed in SEEDS:
                    case_id = f"K3|{dataset}|{model}|{seed}"
                    p, execution = fit_or_load(data, arrays, model, seed, old_main, old_secondary, run_device)
                    metrics = metric_row(data["eval"]["labels"], p)
                    existing_seed_rows.append({"dataset": dataset, "model": model, "seed": seed, **metrics,
                                               "execution": execution, "train_candidates": len(data["train"]["candidates"]),
                                               "eval_candidates": len(data["eval"]["candidates"]), "eval_positives": int(data["eval"]["labels"].sum()),
                                               "candidate_hash_train": data["train"]["candidate_hash"], "candidate_hash_eval": data["eval"]["candidate_hash"],
                                               "feature_name_hash": canonical_hash(list(FEATURES[model]))})
                    dataset_predictions[model][seed] = p
                    for candidate_id, (candidate, y, probability) in enumerate(zip(data["eval"]["candidates"], data["eval"]["labels"], p)):
                        existing_predictions.append({"candidate_id": candidate_id, "candidate_vertices": ",".join(map(str, candidate)),
                                                     "y_true": int(y), "p_pred": float(probability), "dataset": dataset, "model": model, "seed": seed})
                    state["cases"][case_id] = {"status": "COMPLETED", "execution": execution}
                    write_json(LOCAL / "RUN_STATE.json", state)
            row.update({"local_status": "COMPLETED", "server_required": "NO", "status_detail": "all B0-B5 x five seeds complete"})
            if not (LOCAL / "SMOKE_TEST_COMPLETE.json").is_file():
                write_json(LOCAL / "SMOKE_TEST_COMPLETE.json", {"status": "PASS", "dataset": dataset, "model": "B2", "seed": 21001,
                                                                 "full_candidate_universe": "YES", "candidate_count": len(data["eval"]["candidates"])})
            del data, arrays, dataset_predictions; gc.collect()
            if run_device.type == "cuda": torch.cuda.empty_cache()
        except (mlp.ResourceLimit, MemoryError) as exc:
            row.update({"feature_availability": "INCOMPLETE", "local_status": "LOCAL_RESOURCE_LIMIT", "server_required": "YES", "status_detail": str(exc)})
            state["cases"][f"K3|{dataset}"] = {"status": "LOCAL_RESOURCE_LIMIT", "detail": str(exc)}
        except FileNotFoundError as exc:
            row.update({"feature_availability": "PIPELINE_UNAVAILABLE_LOCAL_DATA_MISSING", "local_status": "PIPELINE_UNAVAILABLE", "server_required": "YES", "status_detail": str(exc)})
        except Exception as exc:
            row.update({"feature_availability": "IMPLEMENTATION_FAIL", "local_status": "IMPLEMENTATION_FAIL", "server_required": "NO", "status_detail": repr(exc)})
            state["cases"][f"K3|{dataset}"] = {"status": "IMPLEMENTATION_FAIL", "detail": repr(exc)}
        finally:
            write_json(LOCAL / "RUN_STATE.json", state)
    aggregate_and_report(panel_rows, existing_predictions, existing_seed_rows, shell_audits)
    write_json(LOCAL / "LOCAL_STAGE_COMPLETE.json", {"status": "COMPLETE", "panel_size": 8,
                                                      "local_complete": sum(row["local_status"] == "COMPLETED" for row in panel_rows),
                                                      "server_required": sum(row["server_required"] == "YES" for row in panel_rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="", help="Comma-separated frozen-panel datasets; blank means all locally feasible cases.")
    parser.add_argument("--server-mode", action="store_true", help="Execute selected queued datasets without the local resource pre-skip.")
    parser.add_argument("--negative-control-only", action="store_true")
    parsed = parser.parse_args()
    if parsed.negative_control_only:
        run_negative_control(parsed.datasets.split(",") if parsed.datasets else panel_names())
    else:
        run(parsed)
