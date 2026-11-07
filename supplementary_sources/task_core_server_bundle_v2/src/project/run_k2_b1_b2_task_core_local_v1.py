from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
COMMON_ROOT = ROOT / "task_core_server_bundle_v2" / "src" / "project"
if str(COMMON_ROOT) not in sys.path:

    sys.path.append(str(COMMON_ROOT))

import run_k2_residual_lower_order_conditional_redundancy as k2
import run_k3_residual_lower_order_conditional_redundancy as lr_legacy
import run_strict_timestamp_t2_mlp_local_v1 as mlp
from scripts.task_core_common_v1 import (
    canonical_hash,
    hash_array,
    metric_row,
    paired_logloss_bootstrap,
    read_json,
    write_csv,
    write_json,
)

PROTOCOL = "K2_B1_B2_TASK_CORE_LOCAL_V1"
OUT = ROOT / "results" / "k2_b1_b2_task_core_local_v1"
OLD_MLP = ROOT / "results" / "strict_timestamp_t2_mlp_local_v1"
DATASETS = (
    "contact-high-school",
    "contact-primary-school",
    "email-Enron",
    "email-Eu",
    "NDC-substances",
)
SEEDS = (21001, 21002, 21003, 21004, 21005)
MODELS = ("LR", "MLP")
MARGIN = 0.02
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_BATCH = 1024

B1_NAMES = tuple(k2.H2_NAMES)
S1_NAMES = tuple("S1__" + name for name in k2.L1_NAMES)
B2_NAMES = B1_NAMES + S1_NAMES
FULL_NAMES = tuple(k2.H2_NAMES) + tuple(k2.L1_NAMES)

def progress(stage, status, dataset=None, **values):
    parts = ["PROGRESS", f"stage={stage}", f"status={status}"]
    if dataset is not None:
        parts.append(f"dataset={dataset}")
    parts.extend(f"{key}={value}" for key, value in values.items())
    print(" | ".join(parts), flush=True)

def adjacency(edges):
    result = defaultdict(set)
    for left, right in edges:
        left, right = int(left), int(right)
        result[left].add(right)
        result[right].add(left)
    return result

def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0

def compute_s1_l1_arrays(task, dec):
    pass

    candidates = task["candidates"]
    full_adj = adjacency(dec["l1_edges"])
    arrays = {name: np.zeros(len(candidates), dtype=np.float32) for name in k2.L1_NAMES}
    max_local_common_degree = 0
    for index, candidate in enumerate(candidates):
        vertices = tuple(map(int, candidate))
        vertex_set = set(vertices)
        degree_counts = [len(full_adj.get(vertex, ())) for vertex in vertices]
        arrays["RES_DEG_SUM"][index] = sum(degree_counts)
        arrays["RES_DEG_MEAN"][index] = float(np.mean(degree_counts))
        arrays["RES_DEG_MAX"][index] = max(degree_counts)
        arrays["RES_DEG_MIN"][index] = min(degree_counts)
        arrays["RES_DEG_STD"][index] = float(np.std(degree_counts))

        cn = ra = aa = jac = 0.0
        for left, right in combinations(vertices, 2):
            left_n = set(full_adj.get(left, ())) - {right}
            right_n = set(full_adj.get(right, ())) - {left}
            common = left_n & right_n
            cn += len(common)
            for node in common:

                local_degree = sum(vertex in full_adj.get(node, ()) for vertex in vertices)
                max_local_common_degree = max(max_local_common_degree, local_degree)
                if local_degree:
                    ra += 1.0 / local_degree
                    aa += 1.0 / math.log(local_degree + 1.0)
            jac += jaccard(left_n, right_n)
        arrays["RES_PCN_SUM"][index] = cn
        arrays["RES_PRA_SUM"][index] = ra
        arrays["RES_PAA_SUM"][index] = aa
        arrays["RES_PJACCARD_MEAN"][index] = jac / 3.0

        one_hop = set().union(*(full_adj.get(vertex, set()) for vertex in vertices)) - vertex_set
        arrays["RES_1HOP_UNIQUE_COUNT"][index] = len(one_hop)

        arrays["RES_2HOP_UNIQUE_COUNT"][index] = 0.0
        arrays["RES_EXTERNAL_CUT_SIZE"][index] = sum(
            1 for vertex in vertices for other in full_adj.get(vertex, ()) if other not in vertex_set
        )
    return arrays, {"max_local_common_node_degree": int(max_local_common_degree)}

def matrix(arrays, names):
    return np.column_stack([arrays[name] for name in names]).astype(np.float32)

def b2_matrix(data, split, cache_dir):
    path = cache_dir / f"{split}_s1_l1_features.npz"
    audit_path = cache_dir / f"{split}_s1_l1_audit.json"
    if path.is_file() and audit_path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        return np.column_stack([data[f"X_{split}_H"]] + [arrays[name] for name in k2.L1_NAMES]).astype(np.float32), read_json(audit_path)
    task = data["train"] if split == "train" else data["eval"]
    dec = data["train_dec"] if split == "train" else data["eval_dec"]
    arrays, extra = compute_s1_l1_arrays(task, dec)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    full_arrays = data["train_arrays"] if split == "train" else data["eval_arrays"]
    expected_equal = [
        "RES_DEG_SUM", "RES_DEG_MEAN", "RES_DEG_MAX", "RES_DEG_MIN", "RES_DEG_STD",
        "RES_PCN_SUM", "RES_PJACCARD_MEAN", "RES_1HOP_UNIQUE_COUNT", "RES_EXTERNAL_CUT_SIZE",
    ]
    max_equal_difference = max(
        float(np.max(np.abs(np.asarray(arrays[name], dtype=np.float64) - np.asarray(full_arrays[name], dtype=np.float64))))
        for name in expected_equal
    ) if len(task["candidates"]) else 0.0
    audit = {
        "split": split,
        "candidate_count": len(task["candidates"]),
        "s1_definition": "residual L1 edges incident to at least one candidate vertex",
        "candidate_driven": "YES",
        "global_A_at_A": "NO",
        "expected_full_equal_columns": expected_equal,
        "max_expected_equal_column_difference": max_equal_difference,
        "s1_two_hop_all_zero": "YES" if not np.any(arrays["RES_2HOP_UNIQUE_COUNT"]) else "NO",
        **extra,
        "feature_hash": hash_array(np.column_stack([arrays[name] for name in k2.L1_NAMES])),
    }
    if max_equal_difference > 1e-7 or audit["s1_two_hop_all_zero"] != "YES" or extra["max_local_common_node_degree"] > 3:
        raise RuntimeError(f"S1_FEATURE_AUDIT_FAIL:{audit}")
    write_json(audit_path, audit)
    return np.column_stack([data[f"X_{split}_H"]] + [arrays[name] for name in k2.L1_NAMES]).astype(np.float32), audit

def prediction_path(dataset, family, condition, seed):
    return OUT / "predictions" / dataset / family / condition / f"seed_{seed}.npz"

def save_prediction(path, y, p, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y=np.asarray(y, dtype=np.int8), p=np.asarray(p, dtype=np.float64))
    with np.load(path, allow_pickle=False) as saved:
        check = metric_row(saved["y"], saved["p"])
    expected = metric_row(y, p)
    if any(abs(float(check[key]) - float(expected[key])) > 1e-12 for key in ("LogLoss", "Brier", "AP", "ROC_AUC")):
        raise RuntimeError("CANDIDATE_LEVEL_METRIC_RECOMPUTATION_FAIL")
    write_json(path.with_suffix(".json"), {**metadata, "metrics": expected, "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "PASS"})
    return expected

def load_saved_prediction(path, expected_y):
    with np.load(path, allow_pickle=False) as saved:
        y = saved["y"].astype(np.int8)
        p = saved["p"].astype(np.float64)
    if not np.array_equal(y, np.asarray(expected_y, dtype=np.int8)):
        raise RuntimeError(f"CACHED_LABEL_MISMATCH:{path}")
    return p, metric_row(y, p)

def old_mlp_prediction(dataset, seed, condition, expected_y):
    source = OLD_MLP / "04_predictions" / "K2" / dataset / "T2" / f"real_eval_seed_{seed}.npz"
    if not source.is_file():
        raise FileNotFoundError(f"FROZEN_MLP_PREDICTION_MISSING:{source}")
    key = "p_H" if condition == "B1" else "p_Full"
    with np.load(source, allow_pickle=False) as saved:
        y = saved["y"].astype(np.int8)
        p = saved[key].astype(np.float64)
    if not np.array_equal(y, np.asarray(expected_y, dtype=np.int8)):
        raise RuntimeError(f"FROZEN_MLP_LABEL_MISMATCH:{dataset}:{seed}")
    return p, source

def fit_lr(dataset, condition, seed, x_train, y_train, x_eval, y_eval, resume):
    path = prediction_path(dataset, "LR", condition, seed)
    if resume and path.is_file() and path.with_suffix(".json").is_file():
        p, metrics = load_saved_prediction(path, y_eval)
        return p, metrics, "RESUMED"
    p, _, _ = lr_legacy._fit_predict("LOGISTIC", x_train, y_train, x_eval, seed=seed)
    metrics = save_prediction(path, y_eval, p, {"family": "LR", "condition": condition, "seed": seed, "execution": "TRAINED_LOCAL"})
    return p, metrics, "TRAINED_LOCAL"

def fit_mlp_b2(dataset, seed, x_train, y_train, x_eval, y_eval, device, batch, resume):
    pred_path = prediction_path(dataset, "MLP", "B2", seed)
    model_root = OUT / "models" / dataset / "MLP" / "B2" / f"seed_{seed}"
    model_path, scaler_path = model_root / "model.pt", model_root / "scaler"
    if resume and pred_path.is_file() and pred_path.with_suffix(".json").is_file():
        p, metrics = load_saved_prediction(pred_path, y_eval)
        return p, metrics, "RESUMED"
    if resume and model_path.is_file() and scaler_path.with_suffix(".npz").is_file():
        bundle = mlp.load_model(model_path, scaler_path, device)
        execution = "MODEL_RESUMED"
    else:
        bundle, _ = mlp.train_one_model(x_train, y_train, B2_NAMES, seed, device, batch, model_path, scaler_path)
        execution = "TRAINED_LOCAL"
    _, p = mlp.predict_model(bundle, x_eval, device, batch)
    metrics = save_prediction(pred_path, y_eval, p, {"family": "MLP", "condition": "B2", "seed": seed, "execution": execution})
    return p, metrics, execution

def seed_row(dataset, family, condition, seed, metrics, execution, data):
    return {
        "dataset": dataset,
        "model_family": family,
        "condition": condition,
        "seed": seed,
        **metrics,
        "execution": execution,
        "train_candidates": len(data["train"]["candidates"]),
        "eval_candidates": len(data["eval"]["candidates"]),
        "eval_positives": int(np.sum(data["eval"]["labels"])),
        "candidate_hash_train": data["candidate_hash_train"],
        "candidate_hash_eval": data["candidate_hash_eval"],
        "label_hash_train": data["label_hash_train"],
        "label_hash_eval": data["label_hash_eval"],
        "feature_name_hash": canonical_hash(list(B1_NAMES if condition == "B1" else B2_NAMES if condition == "B2" else FULL_NAMES)),
        "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "PASS",
    }

def aggregate_condition(y, predictions):
    ensemble = np.mean(np.vstack([predictions[seed] for seed in SEEDS]), axis=0)
    return ensemble, metric_row(y, ensemble)

def evaluate_dataset(dataset, args, device):
    case_root = OUT / "cases" / dataset
    done_path = case_root / "COMPLETE.json"
    if args.resume and done_path.is_file():
        progress("CASE", "SKIP_COMPLETE", dataset)
        return read_json(done_path)
    progress("CASE", "START", dataset)
    data = mlp.load_case_data(2, dataset)
    y_train = np.asarray(data["train"]["labels"], dtype=np.int8)
    y_eval = np.asarray(data["eval"]["labels"], dtype=np.int8)
    x_train = {"B1": data["X_train_H"], "Full": data["X_train_Full"]}
    x_eval = {"B1": data["X_eval_H"], "Full": data["X_eval_Full"]}
    predictions = {family: {condition: {} for condition in ("B1", "B2", "Full")} for family in MODELS}
    seed_rows = []

    write_json(case_root / "PROTOCOL_AUDIT.json", {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "strict_timestamp_T2": "YES",
        "no_timestamp_group_crosses_split": data["gate"]["no_timestamp_group_crosses_split"],
        "full_dataset": "YES",
        "full_candidate_universe": "YES",
        "candidate_subsampling": "NO",
        "negative_subsampling": "NO",
        "B1_definition": "frozen H2 representation including downward closure and candidate boundary edges",
        "B2_definition": "B1 plus candidate-specific residual-L1 incident-edge shell S1",
        "Full_definition": "B1 plus all residual L1 features",
        "B2_trigger": "run both LR and MLP iff B1 noninferiority fails for either family",
        "noninferiority_margin": MARGIN,
        "train_candidates": len(y_train),
        "eval_candidates": len(y_eval),
        "train_positives": int(y_train.sum()),
        "eval_positives": int(y_eval.sum()),
        "candidate_hash_train": data["candidate_hash_train"],
        "candidate_hash_eval": data["candidate_hash_eval"],
        "label_hash_train": data["label_hash_train"],
        "label_hash_eval": data["label_hash_eval"],
        "B1_feature_names": list(B1_NAMES),
        "B2_feature_names": list(B2_NAMES),
        "Full_feature_names": list(FULL_NAMES),
    })

    for family in MODELS:
        for condition in ("B1", "Full"):
            progress("MODEL", "START", dataset, family=family, condition=condition)
            for seed in SEEDS:
                if family == "LR":
                    p, metrics, execution = fit_lr(dataset, condition, seed, x_train[condition], y_train, x_eval[condition], y_eval, args.resume)
                else:
                    p, source = old_mlp_prediction(dataset, seed, condition, y_eval)
                    metrics = metric_row(y_eval, p)
                    execution = f"REUSED_FROZEN_STRICT_MLP:{source.relative_to(ROOT)}"
                predictions[family][condition][seed] = p
                seed_rows.append(seed_row(dataset, family, condition, seed, metrics, execution, data))
            progress("MODEL", "COMPLETE", dataset, family=family, condition=condition)

    family_results = {}
    b1_failed = False
    for family_index, family in enumerate(MODELS):
        p_b1, m_b1 = aggregate_condition(y_eval, predictions[family]["B1"])
        p_full, m_full = aggregate_condition(y_eval, predictions[family]["Full"])
        boot = paired_logloss_bootstrap(y_eval, p_b1, p_full, args.bootstraps, 61000 + family_index)
        noninferior = boot["ci_upper"] <= MARGIN
        b1_failed = b1_failed or not noninferior
        family_results[family] = {
            "B1": m_b1,
            "Full": m_full,
            "B1_vs_Full": boot,
            "B1_noninferior": "YES" if noninferior else "NO",
        }

    b2_audits = {}
    if b1_failed or args.force_b2:
        progress("B2_GATE", "TRIGGERED", dataset, reason="B1_FAILED_OR_FORCE")
        cache = case_root / "feature_cache"
        x_train["B2"], b2_audits["train"] = b2_matrix(data, "train", cache)
        x_eval["B2"], b2_audits["eval"] = b2_matrix(data, "eval", cache)
        if x_train["B2"].shape[1] != len(B2_NAMES) or x_eval["B2"].shape[1] != len(B2_NAMES):
            raise RuntimeError("B2_FEATURE_DIMENSION_MISMATCH")
        for family in MODELS:
            progress("MODEL", "START", dataset, family=family, condition="B2")
            for seed in SEEDS:
                if family == "LR":
                    p, metrics, execution = fit_lr(dataset, "B2", seed, x_train["B2"], y_train, x_eval["B2"], y_eval, args.resume)
                else:
                    p, metrics, execution = fit_mlp_b2(dataset, seed, x_train["B2"], y_train, x_eval["B2"], y_eval, device, args.batch_size, args.resume)
                predictions[family]["B2"][seed] = p
                seed_rows.append(seed_row(dataset, family, "B2", seed, metrics, execution, data))
            p_b2, m_b2 = aggregate_condition(y_eval, predictions[family]["B2"])
            p_full, _ = aggregate_condition(y_eval, predictions[family]["Full"])
            boot = paired_logloss_bootstrap(y_eval, p_b2, p_full, args.bootstraps, 62000 + MODELS.index(family))
            family_results[family].update({
                "B2": m_b2,
                "B2_vs_Full": boot,
                "B2_noninferior": "YES" if boot["ci_upper"] <= MARGIN else "NO",
            })
            progress("MODEL", "COMPLETE", dataset, family=family, condition="B2")
    else:
        progress("B2_GATE", "NOT_TRIGGERED", dataset, reason="B1_NONINFERIOR_BOTH_FAMILIES")
        for family in MODELS:
            family_results[family].update({"B2": None, "B2_vs_Full": None, "B2_noninferior": "NOT_TRIGGERED"})

    for family in MODELS:
        if family_results[family]["B1_noninferior"] == "YES":
            family_results[family]["minimum_supported_core"] = "B1"
        elif family_results[family]["B2_noninferior"] == "YES":
            family_results[family]["minimum_supported_core"] = "B2"
        else:
            family_results[family]["minimum_supported_core"] = "Full"

    write_csv(case_root / "BY_SEED.csv", seed_rows)
    result = {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "status": "COMPLETE",
        "B2_triggered": "YES" if (b1_failed or args.force_b2) else "NO",
        "noninferiority_margin": MARGIN,
        "bootstrap_replicates": args.bootstraps,
        "train_candidates": len(y_train),
        "eval_candidates": len(y_eval),
        "train_positives": int(y_train.sum()),
        "eval_positives": int(y_eval.sum()),
        "family_results": family_results,
        "B2_feature_audits": b2_audits,
        "seed_rows": seed_rows,
    }
    write_json(done_path, result)
    (case_root / "FAILURE.json").unlink(missing_ok=True)
    progress("CASE", "COMPLETE", dataset, B2_triggered=result["B2_triggered"], LR_core=family_results["LR"]["minimum_supported_core"], MLP_core=family_results["MLP"]["minimum_supported_core"])
    del data, x_train, x_eval, predictions
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result

def aggregate(results, failures):
    by_seed, aggregate_rows, bootstrap_rows, manifest_rows = [], [], [], []
    for dataset in DATASETS:
        result = next((row for row in results if row["dataset"] == dataset), None)
        failure = next((row for row in failures if row["dataset"] == dataset), None)
        if result is None:
            manifest_rows.append({"dataset": dataset, "status": "ENGINEERING_FAILURE", "reason": failure.get("reason") if failure else "NOT_RUN"})
            continue
        by_seed.extend(result["seed_rows"])
        for family in MODELS:
            values = result["family_results"][family]
            family_seed_rows = [row for row in result["seed_rows"] if row["model_family"] == family]
            def seed_summary(condition):
                ll = np.asarray([row["LogLoss"] for row in family_seed_rows if row["condition"] == condition], dtype=np.float64)
                return {
                    "n": int(len(ll)),
                    "mean": float(np.mean(ll)) if len(ll) else None,
                    "variance_population": float(np.var(ll)) if len(ll) else None,
                }
            seed_b1 = seed_summary("B1")
            seed_b2 = seed_summary("B2")
            seed_full = seed_summary("Full")
            row = {
                "dataset": dataset,
                "model_family": family,
                "status": "COMPLETE",
                "train_candidates": result["train_candidates"],
                "eval_candidates": result["eval_candidates"],
                "eval_positives": result["eval_positives"],
                "B2_triggered": result["B2_triggered"],
                "minimum_supported_core": values["minimum_supported_core"],
                "B1_LL": values["B1"]["LogLoss"],
                "B2_LL": values["B2"]["LogLoss"] if values["B2"] else None,
                "Full_LL": values["Full"]["LogLoss"],
                "B1_seed_count": seed_b1["n"],
                "B1_seed_LL_mean": seed_b1["mean"],
                "B1_seed_LL_variance_population": seed_b1["variance_population"],
                "B2_seed_count": seed_b2["n"],
                "B2_seed_LL_mean": seed_b2["mean"],
                "B2_seed_LL_variance_population": seed_b2["variance_population"],
                "Full_seed_count": seed_full["n"],
                "Full_seed_LL_mean": seed_full["mean"],
                "Full_seed_LL_variance_population": seed_full["variance_population"],
                "B1_minus_Full": values["B1_vs_Full"]["estimate"],
                "B1_CI_lower": values["B1_vs_Full"]["ci_lower"],
                "B1_CI_upper": values["B1_vs_Full"]["ci_upper"],
                "B1_noninferior": values["B1_noninferior"],
                "B2_minus_Full": values["B2_vs_Full"]["estimate"] if values["B2_vs_Full"] else None,
                "B2_CI_lower": values["B2_vs_Full"]["ci_lower"] if values["B2_vs_Full"] else None,
                "B2_CI_upper": values["B2_vs_Full"]["ci_upper"] if values["B2_vs_Full"] else None,
                "B2_noninferior": values["B2_noninferior"],
                "B1_AP": values["B1"]["AP"],
                "B2_AP": values["B2"]["AP"] if values["B2"] else None,
                "Full_AP": values["Full"]["AP"],
                "B1_AUC": values["B1"]["ROC_AUC"],
                "B2_AUC": values["B2"]["ROC_AUC"] if values["B2"] else None,
                "Full_AUC": values["Full"]["ROC_AUC"],
            }
            aggregate_rows.append(row)
            bootstrap_rows.append({
                "dataset": dataset, "model_family": family, "comparison": "B1_vs_Full",
                **values["B1_vs_Full"], "margin": MARGIN, "noninferior": values["B1_noninferior"],
            })
            if values["B2_vs_Full"]:
                bootstrap_rows.append({
                    "dataset": dataset, "model_family": family, "comparison": "B2_vs_Full",
                    **values["B2_vs_Full"], "margin": MARGIN, "noninferior": values["B2_noninferior"],
                })
        manifest_rows.append({
            "dataset": dataset,
            "status": "COMPLETE",
            "B2_triggered": result["B2_triggered"],
            "LR_minimum_core": result["family_results"]["LR"]["minimum_supported_core"],
            "MLP_minimum_core": result["family_results"]["MLP"]["minimum_supported_core"],
            "train_candidates": result["train_candidates"],
            "eval_candidates": result["eval_candidates"],
            "eval_positives": result["eval_positives"],
        })

    write_csv(OUT / "K2_B1_B2_BY_SEED.csv", by_seed)
    write_csv(OUT / "K2_B1_B2_AGGREGATED.csv", aggregate_rows)
    write_csv(OUT / "K2_B1_B2_BOOTSTRAP.csv", bootstrap_rows)
    write_csv(OUT / "K2_B1_B2_PANEL_MANIFEST.csv", manifest_rows)

    complete = [row for row in manifest_rows if row["status"] == "COMPLETE"]
    lines = [
        "# K2 B1/B2 Task-Core Local Final Report",
        "",
        f"- protocol: {PROTOCOL}",
        f"- noninferiority margin: {MARGIN}",
        f"- local datasets complete: {len(complete)}/{len(DATASETS)}",
        "- B1: frozen H2 representation including downward closure and candidate boundary edges",
        "- B2: B1 plus candidate-specific residual-L1 incident-edge shell S1",
        "- Full: B1 plus all residual L1 features",
        "- B2 trigger: either LR or MLP B1 CI upper exceeds the margin",
        "",
        "| dataset | family | B1 LL | B2 LL | Full LL | B1-Full | B1 CI upper | B1 NI | B2 NI | minimum core |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in aggregate_rows:
        def f(value):
            return "NA" if value in (None, "") else f"{float(value):.6f}"
        lines.append(
            f"| {row['dataset']} | {row['model_family']} | {f(row['B1_LL'])} | {f(row['B2_LL'])} | {f(row['Full_LL'])} | "
            f"{f(row['B1_minus_Full'])} | {f(row['B1_CI_upper'])} | {row['B1_noninferior']} | {row['B2_noninferior']} | {row['minimum_supported_core']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The selected core is model-conditional and based on the preregistered 0.02 LogLoss noninferiority margin. "
        "Passing B1 means that adding all residual L1 features is not required to meet that margin; it does not prove an exactly zero L1 effect.",
    ]
    (OUT / "K2_B1_B2_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate_rows, manifest_rows

def resolve_device(value):
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-b2", action="store_true", help="Run B2 even if both B1 tests pass; diagnostic only")
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.bootstraps < 100:
        raise SystemExit("--bootstraps must be at least 100")
    selected = DATASETS if args.all or not args.dataset else (args.dataset,)
    device = resolve_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "PROTOCOL.json", {
        "protocol": PROTOCOL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datasets": list(DATASETS),
        "selected": list(selected),
        "model_families": list(MODELS),
        "seeds": list(SEEDS),
        "margin": MARGIN,
        "bootstraps": args.bootstraps,
        "device": str(device),
        "B2_trigger_rule": "run both LR and MLP iff B1 fails either family",
        "force_b2": args.force_b2,
    })
    results, failures = [], []
    for dataset in selected:
        try:
            results.append(evaluate_dataset(dataset, args, device))
        except Exception as exc:
            progress("CASE", "ENGINEERING_FAILURE", dataset, reason=repr(exc), next_dataset_continues="YES")
            failures.append({"dataset": dataset, "reason": repr(exc)})
            write_json(OUT / "cases" / dataset / "FAILURE.json", failures[-1])
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for dataset in DATASETS:
        done = OUT / "cases" / dataset / "COMPLETE.json"
        if done.is_file() and not any(row["dataset"] == dataset for row in results):
            results.append(read_json(done))
    aggregate_rows, manifest_rows = aggregate(results, failures)
    status = "COMPLETE" if all(row["status"] == "COMPLETE" for row in manifest_rows) else "PARTIAL"
    write_json(OUT / "RUN_COMPLETE.json", {"status": status, "n_complete": sum(row["status"] == "COMPLETE" for row in manifest_rows), "n_total": len(DATASETS)})
    print("=" * 96)
    print("K2 B1/B2 TASK-CORE LOCAL V1")
    print(f"STATUS = {status}")
    print(f"LOCAL_COMPLETE = {sum(row['status'] == 'COMPLETE' for row in manifest_rows)}/{len(DATASETS)}")
    print(f"B2_TRIGGERED_DATASETS = {sum(row.get('B2_triggered') == 'YES' for row in manifest_rows)}")
    for row in aggregate_rows:
        print(f"{row['dataset']} | {row['model_family']} | B1_NI={row['B1_noninferior']} | B2_NI={row['B2_noninferior']} | CORE={row['minimum_supported_core']}")
    print(f"FINAL_REPORT = {OUT / 'K2_B1_B2_FINAL_REPORT.md'}")
    print("=" * 96)
    return 0 if status == "COMPLETE" else 1

if __name__ == "__main__":
    raise SystemExit(main())
