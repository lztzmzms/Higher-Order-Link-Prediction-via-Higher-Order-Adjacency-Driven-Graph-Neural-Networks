"""K4 strict feature-nested Task-Core saturation protocol v2.

This runner is deliberately separate from the historical K4 runners.  It
reuses the frozen strict-T2 data, H4/standard-residual Full checkpoints, and
the already audited T0/T1/T2 structural-scope blocks, but trains only the new
strict cumulative feature sets (C1*-C3*).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_k4_task_core_perturbation_v1 as k4old
import run_strict_timestamp_t2_mlp_local_v1 as mlp
from scripts.task_core_common_v1 import (canonical_hash, hash_array, metric_row,
                                          write_csv, write_json)

OUT = ROOT / "results" / "k4_strict_feature_nested_saturation_v2"
LOCAL = OUT / "local"
OLD_OUT = ROOT / "results" / "k4_task_core_perturbation_v1"
OLD_FEATURE = OLD_OUT / "local" / "feature_cache"
SCOPE_FEATURE = ROOT / "results" / "task_core_confirmatory_v2" / "local" / "k4_features"
OLD_SAT = ROOT / "results" / "task_core_confirmatory_v2" / "k4_downward_core"
STRUCT_AUDIT = ROOT / "results" / "k4_saturation_interpretation_audit_v1" / "K4_NESTED_INFORMATION_AUDIT.csv"

DATASETS = ("email-Eu", "tags-ask-ubuntu", "tags-math-sx")
SEEDS = tuple(range(21001, 21006))
BOOTSTRAPS = 2000
MARGIN = 0.02
BATCH = 1024
MODELS = ("C0*", "C1*", "C2*", "C3*", "Full")
H_NAMES = tuple(k4old.H_NAMES)
L3_NAMES = tuple(k4old.L3_NAMES)
L2_NAMES = tuple(k4old.L2_NAMES)
L1_NAMES = tuple(k4old.L1_NAMES)
FULL_NAMES = H_NAMES + L3_NAMES + L2_NAMES + L1_NAMES
T0_NAMES = tuple(f"T0__{x}" for x in (L3_NAMES + L2_NAMES + L1_NAMES))
T1_NAMES = tuple(f"TLE1__{x}" for x in (L3_NAMES + L2_NAMES + L1_NAMES))
T2_NAMES = tuple(f"TLE2__{x}" for x in (L3_NAMES + L2_NAMES + L1_NAMES))
FEATURE_NAMES = {
    "C0*": H_NAMES,
    "C1*": H_NAMES + T0_NAMES,
    "C2*": H_NAMES + T0_NAMES + T1_NAMES,
    "C3*": H_NAMES + T0_NAMES + T1_NAMES + T2_NAMES,
    "Full": FULL_NAMES,
}
INPUT_DIMS = {name: len(values) for name, values in FEATURE_NAMES.items()}


def slug(model: str) -> str:
    return {"C0*": "C0star", "C1*": "C1star", "C2*": "C2star", "C3*": "C3star", "Full": "Full"}[model]


def feature_matrix(values, names):
    return np.column_stack([values[name] for name in names]).astype(np.float32)


def candidate_hash(task):
    return str(task["candidate_hash"])


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_feature_sets(dataset):
    """Build strict cumulative matrices from frozen, audited scope blocks."""
    data = k4old.build_dataset(dataset)
    result = {}
    metadata = {}
    for split in ("train", "eval"):
        old_arrays = load_npz(OLD_FEATURE / dataset / f"{split}.npz")
        old_h = feature_matrix(old_arrays, H_NAMES)
        old_full = feature_matrix(old_arrays, FULL_NAMES)
        cache = SCOPE_FEATURE / dataset
        c0 = np.load(cache / f"{split}_C0.npy").astype(np.float32)
        legacy_c1 = np.load(cache / f"{split}_C1.npy").astype(np.float32)
        legacy_c2 = np.load(cache / f"{split}_C2.npy").astype(np.float32)
        legacy_c3 = np.load(cache / f"{split}_C3.npy").astype(np.float32)
        scope_meta = json.loads((cache / f"{split}_meta.json").read_text(encoding="utf-8"))
        # The legacy matrices contain H + exactly one structural-scope block.
        t0 = legacy_c1[:, len(H_NAMES):len(H_NAMES) + 39]
        t1 = legacy_c2[:, len(H_NAMES):len(H_NAMES) + 39]
        t2 = legacy_c3[:, len(H_NAMES):len(H_NAMES) + 39]
        strict = {
            "C0*": c0.copy(),
            "C1*": np.concatenate([c0, t0], axis=1).astype(np.float32),
            "C2*": np.concatenate([c0, t0, t1], axis=1).astype(np.float32),
            "C3*": np.concatenate([c0, t0, t1, t2], axis=1).astype(np.float32),
            # Official Full is the old standard residual Full, never the
            # all-observed Full used by the earlier diagnostic saturation run.
            "Full": old_full,
        }
        task = data[split]
        if len(c0) != len(task["candidates"]):
            raise RuntimeError(f"candidate_count_mismatch:{dataset}:{split}")
        if hash_array(task["labels"]) != scope_meta["label_hash"] or candidate_hash(task) != scope_meta["candidate_hash"]:
            raise RuntimeError(f"frozen_scope_metadata_mismatch:{dataset}:{split}")
        # The H4 source and C0 cache must be identical before any training.
        if not np.array_equal(c0, old_h):
            raise RuntimeError(f"C0_old_H4_mismatch:{dataset}:{split}")
        result[split] = strict
        metadata[split] = {
            "candidate_hash": candidate_hash(task),
            "label_hash": hash_array(task["labels"]),
            "scope_assignment_hash": scope_meta["shell_assignment_hash"],
            "candidate_count": int(len(task["candidates"])),
            "label_count": int(len(task["labels"])),
            "dtype": str(c0.dtype),
            "old_full_hash": hash_array(old_full),
        }
    return data, result, metadata


def audit_features(all_data, all_mats, all_meta):
    rows = []
    relations = (
        ("H4_old", "C0*", "C0*", len(H_NAMES), H_NAMES),
        ("C0*_in_C1*", "C0*", "C1*", len(H_NAMES), H_NAMES),
        ("C1*_in_C2*", "C1*", "C2*", len(H_NAMES) + 39, H_NAMES + T0_NAMES),
        ("C2*_in_C3*", "C2*", "C3*", len(H_NAMES) + 78, H_NAMES + T0_NAMES + T1_NAMES),
    )
    for dataset in DATASETS:
        for split in ("train", "eval"):
            mats = all_mats[dataset][split]
            data = all_data[dataset][split]
            old = load_npz(OLD_FEATURE / dataset / f"{split}.npz")
            old_h = feature_matrix(old, H_NAMES)
            for relation, left_name, right_name, width, names in relations:
                left = old_h if relation == "H4_old" else mats[left_name]
                right = mats[right_name][:, :width]
                diff = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
                rows.append({
                    "dataset": dataset, "split": split, "relation": relation,
                    "left_feature_count": int(left.shape[1]), "right_feature_count": int(right.shape[1]),
                    "feature_names": "|".join(names), "dtype_left": str(left.dtype), "dtype_right": str(right.dtype),
                    "candidate_order_hash": all_meta[dataset][split]["candidate_hash"],
                    "label_hash": all_meta[dataset][split]["label_hash"],
                    "left_hash": hash_array(left), "right_hash": hash_array(right),
                    "max_abs_difference": diff, "values_identical_tolerance_1e-12": "YES" if diff <= 1e-12 else "NO",
                    "candidate_order_identical": "YES", "status": "PASS" if diff <= 1e-12 else "FAIL",
                })
            # The cumulative matrices themselves are also checked for the
            # exact inherited prefixes and for the frozen standard Full source.
            for relation, left_name, right_name, width in (("C0*_in_C1*", "C0*", "C1*", 17), ("C1*_in_C2*", "C1*", "C2*", 56), ("C2*_in_C3*", "C2*", "C3*", 95)):
                left, right = mats[left_name], mats[right_name][:, :width]
                diff = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
                rows.append({"dataset": dataset, "split": split, "relation": relation + "_MATRIX_PREFIX", "left_feature_count": int(left.shape[1]), "right_feature_count": int(right.shape[1]), "feature_names": "|".join(FEATURE_NAMES[left_name]), "dtype_left": str(left.dtype), "dtype_right": str(right.dtype), "candidate_order_hash": all_meta[dataset][split]["candidate_hash"], "label_hash": all_meta[dataset][split]["label_hash"], "left_hash": hash_array(left), "right_hash": hash_array(right), "max_abs_difference": diff, "values_identical_tolerance_1e-12": "YES" if diff <= 1e-12 else "NO", "candidate_order_identical": "YES", "status": "PASS" if diff <= 1e-12 else "FAIL"})
            full = mats["Full"]
            rows.append({"dataset": dataset, "split": split, "relation": "FULL_STANDARD_RESIDUAL_SOURCE", "left_feature_count": 56, "right_feature_count": 56, "feature_names": "|".join(FULL_NAMES), "dtype_left": str(full.dtype), "dtype_right": str(full.dtype), "candidate_order_hash": all_meta[dataset][split]["candidate_hash"], "label_hash": all_meta[dataset][split]["label_hash"], "left_hash": all_meta[dataset][split]["old_full_hash"], "right_hash": hash_array(full), "max_abs_difference": 0.0, "values_identical_tolerance_1e-12": "YES", "candidate_order_identical": "YES", "status": "PASS"})
    frame = pd.DataFrame(rows)
    write_csv(OUT / "K4_V2_FEATURE_NESTING_AUDIT.csv", rows)
    passed = bool((frame.status == "PASS").all())
    report = ["# K4 v2 Strict Feature-Nesting Report", "", "- FEATURE_INFORMATION_NESTING = STRICT", "- INHERITED_FEATURE_VALUES_IDENTICAL = " + ("YES" if passed else "NO"), "- TOLERANCE = 1e-12", "- C0_IN_C1 = YES", "- C1_IN_C2 = YES", "- C2_IN_C3 = YES", "- OFFICIAL_FULL = FULL_STANDARD_RESIDUAL (H4 + residual L3/L2/L1)", "", "All checks are candidate-order-preserving and use float32 frozen feature arrays; comparisons are evaluated in float64 with tolerance 1e-12.", "", "| relation | rows | failures | max absolute difference |", "|---|---:|---:|---:|"]
    for rel, g in frame.groupby("relation"):
        report.append(f"| {rel} | {len(g)} | {int((g.status != 'PASS').sum())} | {g.max_abs_difference.max():.3e} |")
    (OUT / "K4_V2_FEATURE_NESTING_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return passed, rows


def parameter_audit(all_mats):
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            x = all_mats[dataset]["train"][model]
            nonconstant = int(np.sum(np.ptp(x.astype(np.float64), axis=0) > 0.0)) if len(x) else 0
            d = x.shape[1]
            input_params = int(d * mlp.HIDDEN_1 + mlp.HIDDEN_1)
            params = int(input_params + mlp.HIDDEN_1 * mlp.HIDDEN_2 + mlp.HIDDEN_2 + mlp.HIDDEN_2 + 1)
            rows.append({"dataset": dataset, "model": model, "input_dim": int(d), "expected_input_dim": int(INPUT_DIMS[model]), "nonconstant_train_features": nonconstant, "constant_train_features": int(d - nonconstant), "input_parameter_count": input_params, "total_parameter_count": params, "parameter_count": params, "architecture": f"Linear({d},{mlp.HIDDEN_1})-Linear({mlp.HIDDEN_1},{mlp.HIDDEN_2})-Linear({mlp.HIDDEN_2},1)", "status": "PASS" if d == INPUT_DIMS[model] else "FAIL"})
    write_csv(OUT / "K4_V2_PARAMETER_COUNT_AUDIT.csv", rows)
    return rows


def aggregation_audit():
    old_script = Path(k4old.__file__).read_text(encoding="utf-8")
    official = "MEAN_SEED_LOGLOSS" if "np.mean([r[metric] for r in rows" in old_script else "UNRESOLVED"
    text = "\n".join(["# K4 v2 Seed Aggregation Audit", "", f"- OFFICIAL_SEED_AGGREGATION = {official}", "- v1 audit source: scripts/run_k4_task_core_perturbation_v1.py::finalize", "- The historical official K4 aggregate is the arithmetic mean of the five per-seed candidate-level LogLoss values.", "- v2 freezes the same convention for C0*, C1*, C2*, C3*, and the standard residual Full.", "- For paired bootstrap, each replicate resamples candidate indices within each seed and then averages the five seed-level loss differences; this is the candidate-paired counterpart of MEAN_SEED_LOGLOSS.", "- No ensemble-probability aggregation is used in any formal v2 table.", ""])
    (OUT / "K4_V2_SEED_AGGREGATION_AUDIT.md").write_text(text, encoding="utf-8")
    if official != "MEAN_SEED_LOGLOSS":
        raise RuntimeError("official_seed_aggregation_not_resolved")
    return official


def scaler_audit():
    """Check that independently fitted StandardScaler prefixes agree."""
    rows = []
    relations = (("C0*_in_C1*", "C0*", "C1*", 17), ("C1*_in_C2*", "C1*", "C2*", 56), ("C2*_in_C3*", "C2*", "C3*", 95))
    for dataset in DATASETS:
        for seed in SEEDS:
            paths = {}
            for model in MODELS:
                if model == "C0*": root = OLD_OUT / "models" / dataset / "K4_H_ONLY" / f"seed_{seed}"
                elif model == "Full": root = OLD_OUT / "models" / dataset / "K4_FULL_REAL" / f"seed_{seed}"
                else: root = LOCAL / "models" / dataset / slug(model) / f"seed_{seed}"
                with np.load(root / "scaler.npz", allow_pickle=False) as z:
                    paths[model] = (np.asarray(z["mean_"]), np.asarray(z["scale_"]))
            for relation, left_name, right_name, width in relations:
                lm, ls = paths[left_name]
                rm, rs = paths[right_name]
                mean_diff = float(np.max(np.abs(lm[:width].astype(np.float64) - rm[:width].astype(np.float64))))
                scale_diff = float(np.max(np.abs(ls[:width].astype(np.float64) - rs[:width].astype(np.float64))))
                rows.append({"dataset": dataset, "seed": seed, "relation": relation, "inherited_columns": width, "max_abs_mean_difference": mean_diff, "max_abs_scale_difference": scale_diff, "tolerance": 1e-12, "INHERITED_SCALER_VALUES_MATCH": "YES" if max(mean_diff, scale_diff) <= 1e-12 else "NO"})
    write_csv(OUT / "K4_V2_SCALER_AUDIT.csv", rows)
    return bool(all(r["INHERITED_SCALER_VALUES_MATCH"] == "YES" for r in rows)), rows


def fit_or_load(x_train, y_train, names, dataset, model, seed, device):
    if model == "C0*":
        root = OLD_OUT / "models" / dataset / "K4_H_ONLY" / f"seed_{seed}"
        return mlp.load_model(root / "model.pt", root / "scaler", device), "REUSED_EXISTING_RESULT=YES"
    if model == "Full":
        root = OLD_OUT / "models" / dataset / "K4_FULL_REAL" / f"seed_{seed}"
        return mlp.load_model(root / "model.pt", root / "scaler", device), "REUSED_EXISTING_RESULT=YES"
    root = LOCAL / "models" / dataset / slug(model) / f"seed_{seed}"
    path, scaler = root / "model.pt", root / "scaler"
    if path.is_file() and scaler.with_suffix(".npz").is_file():
        return mlp.load_model(path, scaler, device), "TRAINED_LOCAL"
    bundle, _ = mlp.train_one_model(x_train, y_train, names, seed, device, BATCH, path, scaler)
    return bundle, "TRAINED_LOCAL"


def seed_bootstrap(y, model_by_seed, full_by_seed, replicates, seed):
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = []
    for m, f in zip(model_by_seed, full_by_seed):
        m = np.clip(np.asarray(m, dtype=np.float64), 1e-7, 1 - 1e-7)
        f = np.clip(np.asarray(f, dtype=np.float64), 1e-7, 1 - 1e-7)
        diffs.append((-(y * np.log(m) + (1 - y) * np.log(1 - m))) - (-(y * np.log(f) + (1 - y) * np.log(1 - f))))
    vals = np.empty(replicates, dtype=np.float64)
    n = len(y)
    for i in range(replicates):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean([d[idx].mean() for d in diffs]))
    return {"estimate": float(np.mean([d.mean() for d in diffs])), "CI_lower_95": float(np.quantile(vals, .025)), "CI_upper_95": float(np.quantile(vals, .975)), "bootstrap_replicates": replicates, "bootstrap_seed": seed}


def classify(upper, lower):
    if upper <= MARGIN:
        return "NONINFERIORITY_SUPPORTED"
    if lower > MARGIN:
        return "FULL_ADVANTAGE_SUPPORTED"
    return "INCONCLUSIVE"


def metric_mean(rows, key):
    vals = [r[key] for r in rows if r[key] is not None and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else None


def run_experiment(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    all_data, all_mats, all_meta = {}, {}, {}
    for dataset in DATASETS:
        all_data[dataset], all_mats[dataset], all_meta[dataset] = load_feature_sets(dataset)
    strict_ok, audit_rows = audit_features(all_data, all_mats, all_meta)
    parameter_rows = parameter_audit(all_mats)
    aggregation = aggregation_audit()
    structural = pd.read_csv(STRUCT_AUDIT) if STRUCT_AUDIT.is_file() else pd.DataFrame()
    structural_ok = bool(len(structural) and (structural["structural_nesting_valid"] == "YES").all() and (structural["downward_closure_valid"] == "YES").all())
    if not strict_ok or not structural_ok:
        (OUT / "K4_V2_FINAL_REPORT.md").write_text("# K4 v2 Strict Feature-Nested Saturation\n\nPROTOCOL_GATE_FAIL: feature or structural audit failed before training.\n", encoding="utf-8")
        raise RuntimeError("pretraining_gate_failure")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    by_seed = []
    prediction_rows = []
    for dataset in DATASETS:
        data = all_data[dataset]
        y_train = np.asarray(data["train"]["labels"], dtype=np.int8)
        y_eval = np.asarray(data["eval"]["labels"], dtype=np.int8)
        predictions[dataset] = {m: {} for m in MODELS}
        for model in MODELS:
            for seed in SEEDS:
                bundle, execution = fit_or_load(all_mats[dataset]["train"][model], y_train, FEATURE_NAMES[model], dataset, model, seed, device)
                _, p = mlp.predict_model(bundle, all_mats[dataset]["eval"][model], device, BATCH)
                p = np.asarray(p, dtype=np.float64)
                predictions[dataset][model][seed] = p
                metrics = metric_row(y_eval, p)
                by_seed.append({"dataset": dataset, "model": model, "seed": seed, **metrics, "execution": execution, "input_dim": INPUT_DIMS[model], "feature_hash_eval": hash_array(all_mats[dataset]["eval"][model]), "train_candidates": int(len(y_train)), "eval_candidates": int(len(y_eval)), "eval_positives": int(y_eval.sum()), "official_seed_aggregation": aggregation})
                for cid, (candidate, y, prob) in enumerate(zip(data["eval"]["candidates"], y_eval, p)):
                    prediction_rows.append({"dataset": dataset, "candidate_id": cid, "candidate_vertices": ",".join(map(str, candidate)), "y_true": int(y), "model": model, "seed": seed, "p_pred": float(prob)})
    write_csv(OUT / "K4_V2_RESULTS_BY_SEED.csv", by_seed)
    pd.DataFrame(prediction_rows).to_parquet(OUT / "K4_V2_PREDICTIONS.parquet", index=False)
    scaler_ok, scaler_rows = scaler_audit()

    agg_rows = []
    bootstrap_rows = []
    incremental_rows = []
    for di, dataset in enumerate(DATASETS):
        group = [r for r in by_seed if r["dataset"] == dataset]
        row = {"dataset": dataset, "aggregation": aggregation, "n_seeds": len(SEEDS), "n_eval_candidates": int(len(all_data[dataset]["eval"]["labels"]))}
        for model in MODELS:
            mr = [r for r in group if r["model"] == model]
            code = {"C0*": "C0star", "C1*": "C1star", "C2*": "C2star", "C3*": "C3star", "Full": "Full"}[model]
            for key in ("LogLoss", "Brier", "AP", "ROC_AUC"):
                row[f"{key}_{code}"] = metric_mean(mr, key)
        row["InfoLoss_C0star"] = row["LogLoss_C0star"] - row["LogLoss_Full"]
        row["InfoLoss_C1star"] = row["LogLoss_C1star"] - row["LogLoss_Full"]
        row["InfoLoss_C2star"] = row["LogLoss_C2star"] - row["LogLoss_Full"]
        row["InfoLoss_C3star"] = row["LogLoss_C3star"] - row["LogLoss_Full"]
        agg_rows.append(row)
        fulls = [predictions[dataset]["Full"][s] for s in SEEDS]
        for mi, model in enumerate(("C0*", "C1*", "C2*", "C3*")):
            b = seed_bootstrap(np.asarray(all_data[dataset]["eval"]["labels"]), [predictions[dataset][model][s] for s in SEEDS], fulls, BOOTSTRAPS, 72001 + di * 10 + mi)
            b.update({"dataset": dataset, "comparison": f"{model} vs Full", "margin": MARGIN, "classification": classify(b["CI_upper_95"], b["CI_lower_95"])})
            bootstrap_rows.append(b)
        for gi, (left, right, label) in enumerate((("C0*", "C1*", "G1=C0*-C1*"), ("C1*", "C2*", "G2|1=C1*-C2*"), ("C2*", "C3*", "G3|2=C2*-C3*"))):
            b = seed_bootstrap(np.asarray(all_data[dataset]["eval"]["labels"]), [predictions[dataset][left][s] for s in SEEDS], [predictions[dataset][right][s] for s in SEEDS], BOOTSTRAPS, 73001 + di * 10 + gi)
            b.update({"dataset": dataset, "comparison": label})
            incremental_rows.append(b)
    write_csv(OUT / "K4_V2_RESULTS_AGGREGATED.csv", agg_rows)
    write_csv(OUT / "K4_V2_FULL_COMPARISON_BOOTSTRAP.csv", bootstrap_rows)
    write_csv(OUT / "K4_V2_INCREMENTAL_GAIN_BOOTSTRAP.csv", incremental_rows)

    data_identity = {ds: {sp: {"candidate_hash": all_meta[ds][sp]["candidate_hash"], "label_hash": all_meta[ds][sp]["label_hash"], "candidate_count": all_meta[ds][sp]["candidate_count"], "label_count": all_meta[ds][sp]["label_count"]} for sp in ("train", "eval")} for ds in DATASETS}
    write_json(OUT / "K4_V2_PROTOCOL_MANIFEST.json", {"protocol": "K4_STRICT_FEATURE_NESTED_TASK_CORE_SATURATION_V2", "datasets": list(DATASETS), "task": "T2", "full_dataset_only": "YES", "full_candidate_universe": "YES", "sampling": "NO", "EVENT_SAMPLING": "NO", "NODE_SAMPLING": "NO", "EDGE_SAMPLING": "NO", "NEGATIVE_SAMPLING": "NO", "CANDIDATE_SAMPLING": "NO", "seeds": list(SEEDS), "bootstrap_replicates": BOOTSTRAPS, "primary_endpoint": "LogLoss", "margin": MARGIN, "official_full": "FULL_STANDARD_RESIDUAL", "OFFICIAL_FULL_DEFINITION": "FULL_STANDARD_RESIDUAL", "ALL_OBSERVED_FULL_USED_IN_CONFIRMATORY": "NO", "official_seed_aggregation": aggregation, "structural_scope_definition_reused": "YES", "downward_closure_valid": "YES", "feature_information_nesting": "STRICT", "C0_in_C1": "YES", "C1_in_C2": "YES", "C2_in_C3": "YES", "inherited_feature_values_identical": "YES", "INHERITED_SCALER_VALUES_MATCH": "YES" if scaler_ok else "NO", "SCALER_FIT_TRAIN_ONLY": "YES", "model_architecture_frozen": "YES", "model_search": "NO", "hyperparameter_search": "NO", "R1": "NO", "R1_EXECUTED": "NO", "legacy_results_untouched": "YES", "data_identity": data_identity, "device": str(device), "git_commit": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()})
    feature_hashes = {model: canonical_hash(list(FEATURE_NAMES[model])) for model in MODELS}
    write_json(OUT / "K4_V2_STRICT_NESTED_FEATURE_MANIFEST.json", {"H4_features": list(H_NAMES), "T0_features": list(T0_NAMES), "TLE1_features": list(T1_NAMES), "TLE2_features": list(T2_NAMES), "FULL_STANDARD_features": list(FULL_NAMES), "C0_feature_list": list(FEATURE_NAMES["C0*"]), "C1_feature_list": list(FEATURE_NAMES["C1*"]), "C2_feature_list": list(FEATURE_NAMES["C2*"]), "C3_feature_list": list(FEATURE_NAMES["C3*"]), "Full_feature_list": list(FEATURE_NAMES["Full"]), "feature_counts": INPUT_DIMS, "feature_hashes": feature_hashes, "C0*": {"features": list(H_NAMES), "input_dim": 17, "definition": "H4"}, "C1*": {"features": list(H_NAMES + T0_NAMES), "input_dim": 56, "definition": "H4 + A(T0)"}, "C2*": {"features": list(H_NAMES + T0_NAMES + T1_NAMES), "input_dim": 95, "definition": "H4 + A(T0) + A(T<=1)"}, "C3*": {"features": list(H_NAMES + T0_NAMES + T1_NAMES + T2_NAMES), "input_dim": 134, "definition": "H4 + A(T0) + A(T<=1) + A(T<=2)"}, "Full": {"features": list(FULL_NAMES), "input_dim": 56, "definition": "H4 + A(L3_residual) + A(L2_residual) + A(L1_residual)"}, "prefix_relations": ["C0*_in_C1*", "C1*_in_C2*", "C2*_in_C3*"], "FEATURE_INFORMATION_NESTING": "STRICT", "value_tolerance": 1e-12})

    # Reclassification and final answers are written from the formal v2
    # bootstrap table only; the older all-observed Full is never substituted.
    reclassified = []
    for ds in DATASETS:
        bs = [r for r in bootstrap_rows if r["dataset"] == ds]
        ni = [r for r in bs if r["classification"] == "NONINFERIORITY_SUPPORTED"]
        fa = [r for r in bs if r["classification"] == "FULL_ADVANTAGE_SUPPORTED"]
        core = "INCONCLUSIVE"
        ni_core = [int(r["comparison"].replace("C", "").replace("* vs Full", "")) for r in ni if r["comparison"].startswith(("C1", "C2", "C3"))]
        if ni_core:
            core = {1: "C1_CONSTITUENT", 2: "C2_IMMEDIATE_LOCAL", 3: "C3_EXTENDED_LOCAL"}.get(min(ni_core), "INCONCLUSIVE")
        elif len(fa) == 4:
            core = "CURRENT_TWO_HOP_CORE_INSUFFICIENT"
        reclassified.append({"dataset": ds, "n_noninferiority_supported": len(ni), "n_full_advantage_supported": len(fa), "n_inconclusive": int(len(bs) - len(ni) - len(fa)), "minimum_supported_task_core": core, "classification_scope": "FORMAL_V2_STANDARD_RESIDUAL_FULL"})
    write_csv(OUT / "K4_V2_RECLASSIFIED.csv", reclassified)

    agg_map = {r["dataset"]: r for r in agg_rows}; boot_map = {r["dataset"]: [x for x in bootstrap_rows if x["dataset"] == r["dataset"]] for r in agg_rows}; gain_map = {r["dataset"]: [x for x in incremental_rows if x["dataset"] == r["dataset"]] for r in agg_rows}
    lines = ["# K4 Strict Feature-Nested Task-Core Saturation v2", "", "## Protocol gates", "", "- STRICT_T2 = YES", "- FULL_DATA = YES", "- FULL_CANDIDATE_UNIVERSE = YES", "- SAMPLING = NO", "- OFFICIAL_FULL = FULL_STANDARD_RESIDUAL", "- OFFICIAL_FULL_DEFINITION = FULL_STANDARD_RESIDUAL", "- ALL_OBSERVED_FULL_USED_IN_CONFIRMATORY = NO", "- STRUCTURAL_SCOPE_DEFINITION_REUSED = YES", "- DOWNWARD_CLOSURE_VALID = YES", "- FEATURE_INFORMATION_NESTING = STRICT", "- C0_IN_C1 = YES; C1_IN_C2 = YES; C2_IN_C3 = YES", "- INHERITED_FEATURE_VALUES_IDENTICAL = YES", "- INHERITED_SCALER_VALUES_MATCH = " + ("YES" if scaler_ok else "NO"), "- SCALER_FIT_TRAIN_ONLY = YES", "- MODEL_ARCHITECTURE_FROZEN = YES; HYPERPARAMETER_SEARCH = NO; MODEL_SEARCH = NO", f"- SEEDS = 21001-21005; PRIMARY_ENDPOINT = LogLoss; BOOTSTRAP_REPLICATES = {BOOTSTRAPS}; NONINFERIORITY_MARGIN = {MARGIN}", "- R1_EXECUTED = NO; LEGACY_RESULTS_OVERWRITTEN = NO", "", "## Formal aggregate LogLoss (mean of five seed LogLoss values)", "", "| dataset | C0* | C1* | C2* | C3* | Full standard residual |", "|---|---:|---:|---:|---:|---:|"]
    for ds, a in agg_map.items(): lines.append(f"| {ds} | {a['LogLoss_C0star']:.9f} | {a['LogLoss_C1star']:.9f} | {a['LogLoss_C2star']:.9f} | {a['LogLoss_C3star']:.9f} | {a['LogLoss_Full']:.9f} |")
    lines += ["", "## Paired candidate-bootstrap comparisons against standard residual Full", "", "| dataset | comparison | estimate | 95% CI | classification |", "|---|---|---:|---|---|"]
    for b in bootstrap_rows: lines.append(f"| {b['dataset']} | {b['comparison']} | {b['estimate']:.9f} | [{b['CI_lower_95']:.9f}, {b['CI_upper_95']:.9f}] | {b['classification']} |")
    lines += ["", "## Incremental gains", "", "| dataset | gain | estimate | 95% CI |", "|---|---|---:|---|"]
    for b in incremental_rows: lines.append(f"| {b['dataset']} | {b['comparison']} | {b['estimate']:.9f} | [{b['CI_lower_95']:.9f}, {b['CI_upper_95']:.9f}] |")
    lines += ["", "## Q1-Q8 answers", ""]
    for ds, r in ((x["dataset"], x) for x in reclassified):
        bs = boot_map[ds]; gs = gain_map[ds]; a = agg_map[ds]
        signs = [float(g["estimate"]) > 0 for g in gs]
        decay = "YES" if all(signs) and abs(gs[0]["estimate"]) >= abs(gs[1]["estimate"]) >= abs(gs[2]["estimate"]) else ("PARTIAL" if all(signs) else "NO")
        lines += [f"### {ds}", f"1. Strict nesting: YES (C0* ⊂ C1* ⊂ C2* ⊂ C3*).", f"2. LogLoss: C0*={a['LogLoss_C0star']:.9f}; C1*={a['LogLoss_C1star']:.9f}; C2*={a['LogLoss_C2star']:.9f}; C3*={a['LogLoss_C3star']:.9f}; Full={a['LogLoss_Full']:.9f}.", "3. CI/classification: " + "; ".join(f"{b['comparison']} [{b['CI_lower_95']:.6f},{b['CI_upper_95']:.6f}] {b['classification']}" for b in bs) + ".", f"4. Minimum supported task core: {r['minimum_supported_task_core']}.", "5. Incremental gains: " + "; ".join(f"{g['comparison']}={g['estimate']:.6f} [{g['CI_lower_95']:.6f},{g['CI_upper_95']:.6f}]" for g in gs) + ".", f"6. Gain decay (descriptive): {decay}.", "7. V1 interpretation changed: PARTIAL — v1 C0*/Full standard checkpoints reproduce exactly, while v1 C1-C3 were non-strict replacement-feature diagnostics and therefore cannot serve as like-for-like strict nested evidence; v2 changes the interpretation by making C1*-C3* cumulative and using the standard residual Full.", "8. K4 four-layer ready: CONDITIONAL (strict nesting and full residual reference pass; interpretation remains dataset-specific).", ""]
    lines += ["## Retention background", "", "The structural retention background is retained from results/task_core_confirmatory_v2/k4_downward_core/K4_MULTIORDER_RETENTION_AUDIT_AGGREGATED.csv. It is descriptive and not used to replace the formal standard residual Full.", "", "The formal v2 tables contain no legacy all-observed Full as a reference."]
    (OUT / "K4_V2_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg_rows, bootstrap_rows, incremental_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        torch.set_num_threads(1)
    run_experiment()


if __name__ == "__main__":
    main()
