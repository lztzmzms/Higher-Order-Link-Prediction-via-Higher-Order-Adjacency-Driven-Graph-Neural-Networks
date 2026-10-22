"""Shared, frozen utilities for the K3/K4 task-core experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows, fieldnames=None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=json_default))


def hash_rows(rows) -> str:
    return sha256_text("\n".join(",".join(map(str, row)) for row in sorted(tuple(map(int, row)) for row in rows)))


def hash_array(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return sha256_bytes(header + array.tobytes())


def metric_row(labels, probabilities):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(labels, dtype=np.int8)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    result = {
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Brier": float(brier_score_loss(y, p)),
        "AP": float(average_precision_score(y, p)) if int(y.sum()) else None,
        "ROC_AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }
    return result


def paired_logloss_bootstrap(labels, p_left, p_right, replicates=2000, seed=41001):
    y = np.asarray(labels, dtype=np.float64)
    left = np.clip(np.asarray(p_left, dtype=np.float64), 1e-7, 1 - 1e-7)
    right = np.clip(np.asarray(p_right, dtype=np.float64), 1e-7, 1 - 1e-7)
    per_left = -(y * np.log(left) + (1 - y) * np.log(1 - left))
    per_right = -(y * np.log(right) + (1 - y) * np.log(1 - right))
    diff = per_left - per_right
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    n = len(diff)
    for index in range(replicates):
        values[index] = float(np.mean(diff[rng.integers(0, n, size=n)]))
    return {
        "estimate": float(np.mean(diff)),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def mean(values):
    clean = [float(v) for v in values if v not in (None, "") and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def safe_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None
