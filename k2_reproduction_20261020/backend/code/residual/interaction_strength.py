"""Leakage-safe historical interaction-strength features for temporal HOP.

Every statistic in this module is computed from the supplied context events
only.  Callers are responsible for passing events with ``timestamp < T``.
No closure labels or future events are accepted by the public builders.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]


def canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class EdgeStrength:
    count: float
    log_count: float
    decay: float
    recency: float
    last_time: float


def build_boundary_strength(
    context_events: Iterable[Mapping],
    cutoff: float,
    decay_lambda: float = 0.0,
) -> Dict[Edge, EdgeStrength]:
    """Build edge strength using only events whose timestamp is before cutoff."""
    context_events = [
        event for event in context_events if float(event["time"]) < float(cutoff)
    ]
    if float(decay_lambda) <= 0.0:
        earliest = min((float(event["time"]) for event in context_events), default=float(cutoff))
        # Automatic, unit-safe half-life: 15% of the observed context span.
        half_life = max(0.15 * (float(cutoff) - earliest), 1.0)
        decay_lambda = math.log(2.0) / half_life
    counts = defaultdict(float)
    decays = defaultdict(float)
    last = {}
    for event in context_events:
        timestamp = float(event["time"])
        vertices = tuple(sorted(set(map(int, event["vertices"]))))
        age = max(0.0, float(cutoff) - timestamp)
        contribution = math.exp(-float(decay_lambda) * age)
        for u, v in combinations(vertices, 2):
            edge = canonical_edge(u, v)
            counts[edge] += 1.0
            decays[edge] += contribution
            last[edge] = max(timestamp, last.get(edge, -math.inf))
    return {
        edge: EdgeStrength(
            count=value,
            log_count=math.log1p(value),
            decay=float(decays[edge]),
            recency=max(0.0, float(cutoff) - float(last[edge])),
            last_time=float(last[edge]),
        )
        for edge, value in counts.items()
    }


def _aggregate(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    positive = np.maximum(arr, 1e-12)
    harmonic = float(len(arr) / np.sum(1.0 / positive)) if len(arr) else 0.0
    geometric = float(np.exp(np.mean(np.log(positive)))) if len(arr) else 0.0
    return {
        f"{prefix}_sum": float(arr.sum()),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_harmonic": harmonic,
        f"{prefix}_geometric": geometric,
    }


def candidate_strength_row(
    triangle: Triangle,
    strength: Mapping[Edge, EdgeStrength],
) -> Dict[str, float]:
    a, b, c = triangle
    records = [
        strength.get(canonical_edge(a, b), EdgeStrength(0, 0, 0, math.inf, -math.inf)),
        strength.get(canonical_edge(a, c), EdgeStrength(0, 0, 0, math.inf, -math.inf)),
        strength.get(canonical_edge(b, c), EdgeStrength(0, 0, 0, math.inf, -math.inf)),
    ]
    row = {"u": a, "v": b, "w": c}
    row.update(_aggregate([r.count for r in records], "strength"))
    row.update(_aggregate([r.log_count for r in records], "log_strength"))
    row.update(_aggregate([r.decay for r in records], "decay_strength"))
    finite_recency = [r.recency if math.isfinite(r.recency) else 0.0 for r in records]
    row.update(_aggregate(finite_recency, "recency"))
    return row


def candidate_strength_matrix(candidates, strength):
    rows = [candidate_strength_row(tuple(map(int, tri)), strength) for tri in candidates]
    names = [
        f"{prefix}_{stat}"
        for prefix in ("strength", "log_strength", "decay_strength", "recency")
        for stat in ("sum", "mean", "min", "max", "std", "harmonic", "geometric")
    ]
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    return matrix, names, rows


def strength_value(record: EdgeStrength | None, mode: str) -> float:
    if mode == "binary":
        return 1.0
    if record is None:
        return 0.0
    if mode in {"weighted-count", "boundary-count"}:
        return float(record.count)
    if mode in {"weighted-log", "boundary-log"}:
        return float(record.log_count)
    if mode in {"weighted-decay", "boundary-decay"}:
        return float(record.decay)
    raise ValueError(f"Unknown strength mode: {mode}")


def dataset_strength_statistics(events, cutoff, decay_lambda=0.0):
    strength = build_boundary_strength(events, cutoff, decay_lambda)
    multiplicities = np.asarray([x.count for x in strength.values()], dtype=np.float64)
    decays = np.asarray([x.decay for x in strength.values()], dtype=np.float64)
    memberships = defaultdict(int)
    for event in events:
        if float(event["time"]) >= float(cutoff):
            continue
        vertices = set(map(int, event["vertices"]))
        for node in vertices:
            memberships[node] += 1
    total_decay = float(decays.sum())
    concentration = float(np.square(decays / total_decay).sum()) if total_decay > 0 else 0.0
    return {
        "mean_edge_multiplicity": float(multiplicities.mean()) if len(multiplicities) else 0.0,
        "median_edge_multiplicity": float(median(multiplicities)) if len(multiplicities) else 0.0,
        "fraction_repeated_edges": float(np.mean(multiplicities > 1)) if len(multiplicities) else 0.0,
        "CV_edge_multiplicity": float(multiplicities.std() / multiplicities.mean()) if len(multiplicities) and multiplicities.mean() else 0.0,
        "recent_activity_concentration": concentration,
        "max_simplex_membership_mean": float(np.mean(list(memberships.values()))) if memberships else 0.0,
    }


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
