#!/usr/bin/env python3
"""Prepare, run, and aggregate the selected K=2 Dual/MLSHoL three-seed jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATASETS = [
    "email-Enron", "contact-high-school", "contact-primary-school", "email-Eu",
    "NDC-substances", "tags-ask-ubuntu", "tags-math-sx", "threads-ask-ubuntu",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_logged(command, log_path, env):
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    return process.returncode, time.time() - started


def load_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def metric_rows(model, out_dir):
    if model == "Dual-Mapping":
        return load_csv_rows(Path(out_dir) / "seed_results.csv")
    return load_csv_rows(Path(out_dir) / "results.csv")


def write_aggregate(results_root: Path, protocol_root: Path, specs, datasets):
    records = []
    job_wall = {}
    run_log = results_root / "run_manifest.jsonl"
    if run_log.is_file():
        for line in run_log.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("phase") == "model_run":
                job_wall[(row.get("dataset"), row.get("model"))] = safe_float(row.get("wall_seconds")) or 0.0
    for model, table in (("Dual-Mapping", specs["dual"]), ("MLSHoL", specs["mlshol"])):
        for dataset in datasets:
            spec = table[dataset]
            model_dir = "dual" if model == "Dual-Mapping" else "mlshol"
            out_dir = results_root / model_dir / dataset / spec["name"]
            rows = metric_rows(model, out_dir)
            manifest = read_json(protocol_root / "shared_splits" / dataset / "protocol_manifest.json")
            prevalence = float(manifest["test_prevalence"])
            successes, failures, runtime = [], 0, 0.0
            for row in rows:
                ap = safe_float(row.get("auc_pr") or row.get("ap") or row.get("average_precision"))
                if ap is None:
                    failures += 1
                    continue
                successes.append(ap)
                runtime += safe_float(row.get("training_seconds") or row.get("train_seconds")) or 0.0
            lifts = [value / prevalence for value in successes] if prevalence > 0 else []
            failures += max(0, len(specs["model_seeds"]) - len(rows))
            records.append({
                "dataset": dataset,
                "model": model,
                "config": spec["name"],
                "n_success": len(successes),
                "n_failed": failures,
                "test_positives": manifest["test_positive_count"],
                "test_candidates": manifest["test_candidate_count"],
                "test_prevalence": prevalence,
                "mean_AP": float(np.mean(successes)) if successes else None,
                "std_AP_population": float(np.std(successes, ddof=0)) if successes else None,
                "mean_Lift": float(np.mean(lifts)) if lifts else None,
                "std_Lift_population": float(np.std(lifts, ddof=0)) if lifts else None,
                "total_model_run_wall_seconds": job_wall.get((dataset, model), runtime),
                "reference_match": manifest["reference_match"],
            })
    output = results_root / "summary_all.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        writer.writeheader()
        writer.writerows(records)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="protocol_data")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()
    config = read_json(root / "configs" / "model_selection.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_root = Path(args.results_root).resolve() if args.results_root else root / "results" / f"k2_dual_mlshol_3seed_{stamp}"
    if results_root.exists() and any(results_root.iterdir()):
        raise RuntimeError(f"Results directory already contains files: {results_root}")
    results_root.mkdir(parents=True, exist_ok=True)
    lock_path = results_root / ".runner.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Another runner already owns {results_root}") from exc
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_handle:
        lock_handle.write(f"pid={os.getpid()} started_utc={stamp}\n")
    run_log = results_root / "run_manifest.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    failures = 0
    try:
        manifest_file = data_root / "run_manifest.json"
        if args.force_prepare or not manifest_file.exists():
            prepare_command = [
                sys.executable, str(root / "prepare_shared_protocol.py"),
                "--source-root", str(root), "--data-root", str(data_root),
                "--datasets", *args.datasets,
            ]
            if not args.allow_reference_mismatch:
                prepare_command.append("--require-reference-match")
            rc, seconds = run_logged(prepare_command, results_root / "protocol_prepare.log", env)
            append_jsonl(run_log, {
                "phase": "protocol_prepare",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "return_code": rc,
                "wall_seconds": seconds,
                "command": prepare_command,
            })
            if rc:
                raise RuntimeError(f"Protocol preparation failed (exit={rc}); inspect {results_root / 'protocol_prepare.log'}")
        protocol_rows = read_json(manifest_file)
        manifest_datasets = {row.get("dataset") for row in protocol_rows}
        missing_datasets = [dataset for dataset in args.datasets if dataset not in manifest_datasets]
        if missing_datasets:
            raise RuntimeError(
                "Protocol run manifest is missing requested datasets: "
                + ", ".join(missing_datasets)
                + ". Re-run with --force-prepare."
            )
        mismatches = [row["dataset"] for row in protocol_rows if row["dataset"] in args.datasets and not row["reference_match"]]
        if mismatches and not args.allow_reference_mismatch:
            raise RuntimeError("Reference sample/count mismatch: " + ", ".join(mismatches))

        for dataset in args.datasets:
            for model, wrapper, table in (
                ("Dual-Mapping", "run_dual_seed.py", config["dual"]),
                ("MLSHoL", "run_mlshol_seed.py", config["mlshol"]),
            ):
                spec = table[dataset]
                job_name = f"{model.lower().replace('-', '')}_{dataset}_{spec['name']}"
                log_path = results_root / "logs" / f"{job_name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, str(root / wrapper),
                    "--dataset", dataset,
                    "--data-root", str(data_root),
                    "--results-root", str(results_root),
                    "--device", args.device,
                ]
                started_utc = datetime.now(timezone.utc).isoformat()
                rc, seconds = run_logged(command, log_path, env)
                append_jsonl(run_log, {
                    "phase": "model_run", "dataset": dataset, "model": model,
                    "config": spec["name"], "seeds": config["model_seeds"],
                    "return_code": rc, "wall_seconds": seconds,
                    "started_utc": started_utc,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "command": command, "log": str(log_path),
                })
                if rc:
                    failures += 1
                    print(f"FAILED {model} {dataset} ({spec['name']}); see {log_path}", flush=True)
                else:
                    print(f"DONE {model} {dataset} ({spec['name']}) seeds={config['model_seeds']}", flush=True)
        summary = write_aggregate(results_root, data_root, config, args.datasets)
        print(f"Wrote {summary}", flush=True)
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass
    if failures:
        raise SystemExit(f"{failures} model/dataset jobs failed; see {run_log}")


if __name__ == "__main__":
    main()
