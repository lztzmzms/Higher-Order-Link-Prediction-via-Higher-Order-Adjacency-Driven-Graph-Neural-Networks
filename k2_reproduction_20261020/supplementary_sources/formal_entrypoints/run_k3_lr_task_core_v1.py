"""K3 B0--B5 logistic-regression task-core saturation.

This is the LR counterpart of the frozen K3 MLP saturation protocol.  It
reuses the exact strict-T2 tasks and feature caches produced by
``run_k3_spatial_order_task_core_v1.py``.  No event, candidate, negative, or
feature sampling is performed.  The runner is resumable at dataset/model/seed
granularity and writes candidate-level predictions for auditability.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PROTOCOL = "K3_LR_TASK_CORE_SATURATION_V1"
MODELS = ("B0", "B1", "B2", "B3", "B4", "B5")
SEEDS = (21001, 21002, 21003, 21004, 21005)
MARGIN = 0.02
DEFAULT_LOCAL = ("email-Eu", "NDC-substances", "tags-math-sx")
DEFAULT_SERVER = ("tags-ask-ubuntu", "threads-math-sx")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows) -> None:
    import csv
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def configure_protocol(project_root: Path, logical_root: Path, feature_cache_root: Path,
                       frozen_root: Path | None):
    project_root = project_root.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))
    from scripts import run_k3_spatial_order_task_core_v1 as k3
    from scripts.task_core_common_v1 import metric_row

    k3.ROOT = logical_root.resolve()
    k3.strict.ROOT = k3.ROOT
    k3.mlp.ROOT = k3.ROOT
    k3.LOCAL = feature_cache_root.resolve().parent
    if frozen_root is not None:
        frozen_root = frozen_root.resolve()
        k3.OLD = frozen_root / "k3_old"
        k3.PANEL_SOURCE = frozen_root / "k3_panel_source.json"
        k3.AUDIT_CACHE = frozen_root / "timestamp_audit_cache.json"
        k3.TAG_PROTOCOL = frozen_root / "tags_ask_protocol.json"
    return k3, metric_row


def fast_paired_bootstrap(labels, p_left, p_right, replicates, seed, chunk_size):
    """Exact ordinary iid paired candidate bootstrap in bounded blocks."""
    y = np.asarray(labels, dtype=np.float64)
    left = np.clip(np.asarray(p_left, dtype=np.float64), 1e-7, 1 - 1e-7)
    right = np.clip(np.asarray(p_right, dtype=np.float64), 1e-7, 1 - 1e-7)
    diff = -(y * np.log(left) + (1 - y) * np.log1p(-left))
    diff += y * np.log(right) + (1 - y) * np.log1p(-right)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    n = len(diff)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = rng.integers(0, n, size=(stop - start, n))
        values[start:stop] = np.mean(diff[indices], axis=1)
    return {
        "estimate": float(np.mean(diff)),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "implementation": "exact_iid_paired_chunked_vectorized",
        "chunk_size": int(chunk_size),
    }


def prediction_path(output_root: Path, dataset: str, model: str, seed: int) -> Path:
    return output_root / "predictions" / dataset / model / f"seed_{seed}.npz"


def fit_one(output_root: Path, dataset: str, model_name: str, seed: int,
            x_train, y_train, x_eval, y_eval, feature_names, resume: bool, metric_row):
    pred_path = prediction_path(output_root, dataset, model_name, seed)
    model_root = output_root / "models" / dataset / model_name / f"seed_{seed}"
    if resume and pred_path.is_file() and (model_root / "model.npz").is_file():
        with np.load(pred_path, allow_pickle=False) as bundle:
            p = bundle["p"].astype(np.float64)
            saved_y = bundle["y"].astype(np.int8)
        if len(p) != len(y_eval) or not np.array_equal(saved_y, np.asarray(y_eval, dtype=np.int8)):
            raise RuntimeError(f"PREDICTION_ALIGNMENT_FAIL:{dataset}/{model_name}/{seed}")
        return seed, p, metric_row(y_eval, p), "RESUMED"

    scaler = StandardScaler(copy=True)
    train_scaled = scaler.fit_transform(np.asarray(x_train, dtype=np.float64))
    eval_scaled = scaler.transform(np.asarray(x_eval, dtype=np.float64))
    classifier = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", max_iter=2000,
        class_weight=None, random_state=int(seed),
    )
    classifier.fit(train_scaled, np.asarray(y_train, dtype=np.int8))
    p = classifier.predict_proba(eval_scaled)[:, 1].astype(np.float64)
    model_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_root / "scaler.npz", mean_=scaler.mean_, scale_=scaler.scale_, var_=scaler.var_)
    np.savez_compressed(model_root / "model.npz", coef_=classifier.coef_, intercept_=classifier.intercept_, classes_=classifier.classes_)
    write_json(model_root / "metadata.json", {
        "protocol": PROTOCOL, "dataset": dataset, "representation": model_name,
        "family": "LR", "seed": int(seed), "penalty": "l2", "C": 1.0,
        "solver": "liblinear", "max_iter": 2000, "class_weight": None,
        "scaler_fit": "train_only", "feature_names": list(feature_names),
        "n_iter": classifier.n_iter_.tolist(),
    })
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, candidate_id=np.arange(len(p), dtype=np.int64),
                        y=np.asarray(y_eval, dtype=np.int8), p=p)
    return seed, p, metric_row(y_eval, p), "TRAINED"


def write_prediction_parquet(output_root: Path, datasets) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return
    target = output_root / "K3_LR_TASK_CORE_PREDICTIONS.parquet"
    writer = None
    try:
        for dataset in datasets:
            for model in MODELS:
                for seed in SEEDS:
                    path = prediction_path(output_root, dataset, model, seed)
                    if not path.is_file():
                        continue
                    with np.load(path, allow_pickle=False) as bundle:
                        p = bundle["p"].astype(np.float64)
                        y = bundle["y"].astype(np.int8)
                    table = pa.table({
                        "dataset": np.repeat(dataset, len(p)),
                        "model_family": np.repeat("LR", len(p)),
                        "model": np.repeat(model, len(p)),
                        "seed": np.repeat(seed, len(p)).astype(np.int32),
                        "candidate_id": np.arange(len(p), dtype=np.int64),
                        "y_true": y,
                        "p_pred": p,
                    })
                    if writer is None:
                        writer = pq.ParquetWriter(target, table.schema, compression="zstd")
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def evaluate_dataset(k3, metric_row, dataset: str, output_root: Path, args):
    complete_path = output_root / "cases" / dataset / "COMPLETE.json"
    if args.resume and complete_path.is_file():
        print(f"CASE_RESUMED dataset={dataset}", flush=True)
        return read_json(complete_path)

    print(f"CASE_START dataset={dataset}", flush=True)
    data = k3.build_data(dataset)
    arrays, shell_audits = k3.prepare_features(data)
    y_train = np.asarray(data["train"]["labels"], dtype=np.int8)
    y_eval = np.asarray(data["eval"]["labels"], dtype=np.int8)
    if len(np.unique(y_train)) != 2:
        raise RuntimeError(f"LR_TRAIN_REQUIRES_TWO_CLASSES:{dataset}")

    predictions = {model: {} for model in MODELS}
    seed_rows = []
    for model in MODELS:
        x_train = k3.model_matrix(arrays["train"], model)
        x_eval = k3.model_matrix(arrays["eval"], model)
        workers = min(args.lr_workers, len(SEEDS))
        print(f"MODEL_GROUP_START dataset={dataset} model={model} workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            jobs = {
                executor.submit(fit_one, output_root, dataset, model, seed,
                                x_train, y_train, x_eval, y_eval, k3.FEATURES[model],
                                args.resume, metric_row): seed
                for seed in SEEDS
            }
            for future in as_completed(jobs):
                seed, p, metrics, execution = future.result()
                predictions[model][seed] = p
                seed_rows.append({
                    "dataset": dataset, "model_family": "LR", "model": model,
                    "seed": seed, **metrics, "execution": execution,
                    "train_candidates": len(y_train), "train_positives": int(y_train.sum()),
                    "eval_candidates": len(y_eval), "eval_positives": int(y_eval.sum()),
                    "candidate_hash_train": data["train"]["candidate_hash"],
                    "candidate_hash_eval": data["eval"]["candidate_hash"],
                })
        del x_train, x_eval
        gc.collect()

    ensembles = {
        model: np.mean(np.vstack([predictions[model][seed] for seed in SEEDS]), axis=0)
        for model in MODELS
    }
    metrics = {model: metric_row(y_eval, ensembles[model]) for model in MODELS}
    bootstraps = []
    for index, model in enumerate(("B1", "B2", "B3", "B4"), 1):
        boot = fast_paired_bootstrap(y_eval, ensembles[model], ensembles["B5"],
                                     args.bootstraps, 83000 + index, args.bootstrap_chunk)
        bootstraps.append({
            "dataset": dataset, "model_family": "LR", "model": model,
            "reference": "B5", **boot, "margin": MARGIN,
            "noninferior": "YES" if boot["ci_upper"] <= MARGIN else "NO",
        })
    minimum = next((row["model"] for row in bootstraps if row["noninferior"] == "YES"), "B5")
    classification = {
        "B1": "CONSTITUENT_ONLY", "B2": "IMMEDIATE_LOCAL_CORE",
        "B3": "EXTENDED_LOCAL_CORE", "B4": "FULL_L2_CORE", "B5": "L1_STILL_NEEDED",
    }[minimum]
    aggregate = {
        "dataset": dataset, "model_family": "LR", "n_eval_candidates": len(y_eval),
        "n_eval_positives": int(y_eval.sum()), "minimum_saturation_model": minimum,
        "classification": classification,
    }
    for model in MODELS:
        for metric, value in metrics[model].items():
            aggregate[f"{metric}_{model}"] = value
    chosen = next((row for row in bootstraps if row["model"] == minimum), None)
    aggregate["saturation_ci_upper"] = chosen["ci_upper"] if chosen else None
    result = {
        "protocol": PROTOCOL, "status": "COMPLETE", "dataset": dataset,
        "aggregate": aggregate, "bootstrap": bootstraps,
        "seed_rows": sorted(seed_rows, key=lambda row: (row["model"], row["seed"])),
        "shell_audits": shell_audits,
    }
    write_csv(output_root / "cases" / dataset / "BY_SEED.csv", result["seed_rows"])
    write_csv(output_root / "cases" / dataset / "BOOTSTRAP.csv", bootstraps)
    write_json(complete_path, result)
    print(f"CASE_COMPLETE dataset={dataset} minimum={minimum}", flush=True)
    return result


def finalize(output_root: Path, datasets, results, failures, args) -> int:
    seed_rows, aggregated, bootstraps, shell_audits, panel = [], [], [], [], []
    by_dataset = {row["dataset"]: row for row in results}
    for dataset in datasets:
        result = by_dataset.get(dataset)
        if result is None:
            failure = next((row for row in failures if row["dataset"] == dataset), {})
            panel.append({"dataset": dataset, "model_family": "LR", "status": "FAILED", "reason": failure.get("reason", "NOT_RUN")})
            continue
        seed_rows.extend(result["seed_rows"])
        aggregated.append(result["aggregate"])
        bootstraps.extend(result["bootstrap"])
        shell_audits.extend(result["shell_audits"])
        panel.append({"dataset": dataset, "model_family": "LR", "status": "COMPLETE",
                      "minimum_saturation_model": result["aggregate"]["minimum_saturation_model"],
                      "classification": result["aggregate"]["classification"]})
    write_csv(output_root / "K3_LR_TASK_CORE_BY_SEED.csv", seed_rows)
    write_csv(output_root / "K3_LR_TASK_CORE_AGGREGATED.csv", aggregated)
    write_csv(output_root / "K3_LR_TASK_CORE_BOOTSTRAP.csv", bootstraps)
    write_csv(output_root / "K3_LR_TASK_CORE_PANEL_MANIFEST.csv", panel)
    write_csv(output_root / "K3_LR_L2_SPATIAL_SHELL_AUDIT.csv", shell_audits)
    write_prediction_parquet(output_root, datasets)
    lines = [
        "# K3 LR Task-Core Saturation v1", "",
        f"- protocol: {PROTOCOL}",
        "- strict T2; full event streams; full candidate universes; sampling=NO",
        "- LR: train-only StandardScaler + LogisticRegression(L2,C=1,liblinear,max_iter=2000)",
        f"- seeds: {', '.join(map(str, SEEDS))}; paired bootstrap={args.bootstraps}; margin={MARGIN}", "",
        "| dataset | B0 LL | B1 LL | B2 LL | B3 LL | B4 LL | B5/Full LL | minimum | CI upper | classification |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in aggregated:
        ci = row.get("saturation_ci_upper")
        lines.append(
            f"| {row['dataset']} | {row['LogLoss_B0']:.6f} | {row['LogLoss_B1']:.6f} | "
            f"{row['LogLoss_B2']:.6f} | {row['LogLoss_B3']:.6f} | {row['LogLoss_B4']:.6f} | "
            f"{row['LogLoss_B5']:.6f} | {row['minimum_saturation_model']} | "
            f"{ci:.6f} | {row['classification']} |" if ci is not None else
            f"| {row['dataset']} | {row['LogLoss_B0']:.6f} | {row['LogLoss_B1']:.6f} | "
            f"{row['LogLoss_B2']:.6f} | {row['LogLoss_B3']:.6f} | {row['LogLoss_B4']:.6f} | "
            f"{row['LogLoss_B5']:.6f} | B5 | — | L1_STILL_NEEDED |"
        )
    (output_root / "K3_LR_TASK_CORE_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    complete = sum(row["status"] == "COMPLETE" for row in panel)
    write_json(output_root / "RUN_COMPLETE.json", {
        "status": "COMPLETE" if complete == len(datasets) else "PARTIAL",
        "complete": complete, "total": len(datasets), "datasets": list(datasets),
    })
    return 0 if complete == len(datasets) else 1


def main(argv=None) -> int:
    global SEEDS
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=workspace)
    parser.add_argument("--logical-root", type=Path)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=workspace / "results" / "k3_lr_task_core_v1")
    parser.add_argument("--datasets", default=",".join(DEFAULT_LOCAL))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                        help="comma-separated LR seeds (default: frozen formal seeds)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr-workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--bootstrap-chunk", type=int, default=4)
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.logical_root = (args.logical_root or args.project_root).resolve()
    args.feature_cache_root = (args.feature_cache_root or
        (args.logical_root / "results" / "k3_spatial_order_task_core_v1" / "local" / "feature_cache")).resolve()
    args.output_root = args.output_root.resolve()
    try:
        SEEDS = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    except ValueError as exc:
        raise SystemExit("seeds must be comma-separated integers") from exc
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    allowed = set(DEFAULT_LOCAL + DEFAULT_SERVER)
    if not datasets or any(value not in allowed for value in datasets):
        raise SystemExit(f"datasets must be drawn from {sorted(allowed)}")
    if not SEEDS or len(set(SEEDS)) != len(SEEDS):
        raise SystemExit("seeds must be non-empty and unique")
    if args.lr_workers < 1 or args.bootstraps < 100 or args.bootstrap_chunk < 1:
        raise SystemExit("invalid worker/bootstrap setting")
    args.output_root.mkdir(parents=True, exist_ok=True)
    k3, metric_row = configure_protocol(args.project_root, args.logical_root,
                                        args.feature_cache_root, args.frozen_root)
    write_json(args.output_root / "PROTOCOL.json", {
        "protocol": PROTOCOL, "datasets": list(datasets), "model_family": "LR",
        "representations": list(MODELS), "seeds": list(SEEDS), "margin": MARGIN,
        "bootstrap_replicates": args.bootstraps, "strict_T2": True,
        "full_data": True, "full_candidate_universe": True, "sampling": "NONE",
        "project_root": str(args.project_root), "logical_root": str(args.logical_root),
        "feature_cache_root": str(args.feature_cache_root), "python": sys.version,
        "platform": platform.platform(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    results, failures = [], []
    for dataset in datasets:
        try:
            results.append(evaluate_dataset(k3, metric_row, dataset, args.output_root, args))
        except Exception as exc:
            failure = {"dataset": dataset, "reason": repr(exc)}
            failures.append(failure)
            write_json(args.output_root / "cases" / dataset / "FAILURE.json", failure)
            print(f"CASE_FAILED dataset={dataset} reason={exc!r}", flush=True)
        finally:
            gc.collect()
    return finalize(args.output_root, datasets, results, failures, args)


if __name__ == "__main__":
    raise SystemExit(main())
