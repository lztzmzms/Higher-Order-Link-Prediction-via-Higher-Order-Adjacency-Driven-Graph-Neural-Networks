#!/usr/bin/env python3
"""Summarize the email-Enron server run with AP, Lift, and runtime."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, "", "nan", "NaN"):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def load_wall_times(results_root: Path) -> dict[str, float]:
    output: dict[str, float] = {}
    path = results_root / "run_manifest.jsonl"
    if not path.is_file():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("phase") != "model_run" or row.get("dataset") != "email-Enron":
            continue
        seconds = number(row, "wall_seconds")
        if seconds is not None:
            output[str(row.get("model"))] = seconds
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--protocol-root", type=Path, default=Path("protocol_data"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results_root = args.results_root.resolve()
    protocol_root = args.protocol_root.resolve()
    manifest = json.loads((protocol_root / "shared_splits" / "email-Enron" / "protocol_manifest.json").read_text(encoding="utf-8"))
    prevalence = float(manifest["test_prevalence"])
    wall = load_wall_times(results_root)
    config = json.loads((root / "configs" / "model_selection.json").read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    for model, directory, metric_file, table_name in (
        ("Dual-Mapping", "dual", "seed_results.csv", "dual"),
        ("MLSHoL", "mlshol", "results.csv", "mlshol"),
    ):
        config_name = config[table_name]["email-Enron"]["name"]
        path = results_root / directory / "email-Enron" / config_name / metric_file
        metric_rows = read_csv(path)
        aps: list[float] = []
        for metric_row in metric_rows:
            ap = number(metric_row, "auc_pr", "ap", "average_precision")
            if ap is None:
                continue
            aps.append(ap)
            rows.append({
                "dataset": "email-Enron",
                "model": model,
                "seed": metric_row.get("seed", ""),
                "AP": ap,
                "Lift": ap / prevalence,
                "train_seconds": number(metric_row, "training_seconds", "train_seconds"),
                "job_wall_seconds": wall.get(model),
                "test_candidates": int(manifest["test_candidate_count"]),
                "test_positives": int(manifest["test_positive_count"]),
                "test_prevalence": prevalence,
            })
        if aps:
            rows.append({
                "dataset": "email-Enron",
                "model": model,
                "seed": "mean",
                "AP": float(np.mean(aps)),
                "Lift": float(np.mean(aps) / prevalence),
                "train_seconds": None,
                "job_wall_seconds": wall.get(model),
                "test_candidates": int(manifest["test_candidate_count"]),
                "test_positives": int(manifest["test_positive_count"]),
                "test_prevalence": prevalence,
            })

    if not rows:
        raise SystemExit("No completed email-Enron metric rows found; inspect run_manifest.jsonl and logs.")
    output = results_root / "email_enron_metrics.csv"
    fields = ["dataset", "model", "seed", "AP", "Lift", "train_seconds", "job_wall_seconds", "test_candidates", "test_positives", "test_prevalence"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"email-Enron candidates={manifest['test_candidate_count']} positives={manifest['test_positive_count']} prevalence={prevalence:.12f}")
    for row in rows:
        print(f"{row['model']:12s} seed={str(row['seed']):>4s} AP={float(row['AP']):.8f} Lift={float(row['Lift']):.6f} train_seconds={row['train_seconds']} job_wall_seconds={row['job_wall_seconds']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
