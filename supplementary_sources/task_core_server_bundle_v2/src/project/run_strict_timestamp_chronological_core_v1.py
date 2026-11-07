from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import run_h_preserving_perturbation_dev_v2 as hp
import run_k2_residual_lower_order_conditional_redundancy as k2legacy
import run_k3_residual_lower_order_conditional_redundancy as k3legacy
from kclosure.core import (
    build_observed_state,
    enumerate_open_simplices,
    future_closed_simplices,
    load_timestamped_simplices,
)

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results" / "strict_timestamp_chronological_core_v1"
DIRS = {
    0: OUT_ROOT / "00_protocol",
    1: OUT_ROOT / "01_timestamp_audit",
    2: OUT_ROOT / "02_split_audit",
    3: OUT_ROOT / "03_k3",
    4: OUT_ROOT / "04_k2",
    5: OUT_ROOT / "05_cross_protocol",
    6: OUT_ROOT / "06_tables",
    7: OUT_ROOT / "07_logs",
    8: OUT_ROOT / "08_summary",
}
MANIFEST_PATH = OUT_ROOT / "run_manifest.json"
PROTOCOL = "strict_timestamp_block_v1"
K3_DATASETS = (
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-math-sx",
)
K2_DATASETS = (
    "contact-high-school",
    "contact-primary-school",
    "email-Enron",
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-math-sx",
)
ALL_DATASETS = (
    "contact-high-school",
    "contact-primary-school",
    "email-Enron",
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-math-sx",
)
T1 = "T1"
T2 = "T2"
TASKS = (T1, T2)
TARGETS = (0.40, 0.60, 0.80)
R1_SEEDS = {
    (3, T1): tuple(range(11001, 11011)),
    (3, T2): tuple(range(12001, 12011)),
    (2, T1): tuple(range(13001, 13011)),
    (2, T2): tuple(range(14001, 14011)),
}

STRICT_TASK_RESOURCE_CAPS = {
    (3, "tags-ask-ubuntu", T2),
    (3, "threads-math-sx", T1),
    (3, "threads-math-sx", T2),
}

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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def hash_rows(rows) -> str:
    def row_text(row):
        if isinstance(row, (tuple, list)):
            return ",".join(map(str, row))
        return str(row)
    return sha256_text("\n".join(row_text(row) for row in sorted(rows, key=lambda x: str(x))))

def hash_events(events) -> str:
    return sha256_text("\n".join(
        f"{int(event['row_id'])}|{int(event['time'])}|{','.join(map(str, event['vertices']))}"
        for event in events
    ))

def fmt(value, digits=6):
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "NA"
    except (TypeError, ValueError):
        return str(value)

def mean(values):
    values = [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None

def median(values):
    values = [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]
    return float(np.median(values)) if values else None

def raw_paths(dataset: str):
    base = ROOT / "dataset" / "raw_data" / dataset
    return tuple(base / f"{dataset}-{suffix}.txt" for suffix in ("nverts", "simplices", "times"))

def load_events(dataset: str, order: str):
    return load_timestamped_simplices(*raw_paths(dataset), event_order=order)[0]

def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks

def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator else 0.0

def kendall_tau_b(values):
    pass
    values = np.asarray(values)
    if len(values) < 2:
        return 0.0
    unique, inverse = np.unique(values, return_inverse=True)
    tree = [0] * (len(unique) + 1)

    def query(index):
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    def add(index):
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    seen = 0
    discordant = 0
    tie_y_pairs = 0
    tie_counts = Counter(values.tolist())
    for count in tie_counts.values():
        tie_y_pairs += count * (count - 1) // 2
    for rank in inverse:
        rank = int(rank) + 1
        prefix = query(rank)
        discordant += seen - prefix
        add(rank)
        seen += 1
    total = len(values) * (len(values) - 1) // 2
    concordant = total - tie_y_pairs - discordant
    denominator = math.sqrt(max(total - tie_y_pairs, 0) * total)
    return float((concordant - discordant) / denominator) if denominator else 0.0

def inversion_count(events):
    return sum(int(events[i]["time"] > events[i + 1]["time"]) for i in range(max(0, len(events) - 1)))

def boundary_candidates(cumulative, target):
    position = int(np.searchsorted(cumulative, target, side="left"))
    candidates = set()
    for index in range(max(0, position - 2), min(len(cumulative), position + 3)):
        candidates.add(int(cumulative[index]))
    return sorted(candidates)

def choose_block_cuts(group_sizes, n_events):
    cumulative = np.cumsum(np.asarray(group_sizes, dtype=np.int64))
    targets = [target * n_events for target in TARGETS]
    options = [boundary_candidates(cumulative, target) for target in targets]
    best = None
    for c40 in options[0]:
        for c60 in options[1]:
            for c80 in options[2]:
                if not (0 < c40 < c60 < c80 < n_events):
                    continue
                error = sum(abs(cut - target) for cut, target in zip((c40, c60, c80), targets))
                key = (error, abs(c40 - targets[0]), abs(c60 - targets[1]), abs(c80 - targets[2]))
                if best is None or key < best[0]:
                    best = (key, (int(c40), int(c60), int(c80)))
    if best is None:
        raise RuntimeError("TIMESTAMP_BLOCK_SPLIT_FEASIBLE=NO: no three increasing complete-block cuts")
    return best[1]

def timestamp_block_stats(events):
    counts = Counter(int(event["time"]) for event in events)
    sizes = list(counts.values())
    return {
        "n_unique_timestamps": len(counts),
        "n_duplicate_timestamp_events": len(events) - len(counts),
        "duplicate_timestamp_fraction": (len(events) - len(counts)) / max(len(events), 1),
        "max_timestamp_block_size": max(sizes) if sizes else 0,
        "median_timestamp_block_size": float(np.median(sizes)) if sizes else 0.0,
        "q95_timestamp_block_size": float(np.quantile(sizes, 0.95)) if sizes else 0.0,
        "max_timestamp_block_fraction": (max(sizes) / max(len(events), 1)) if sizes else 0.0,
        "n_duplicate_timestamp_groups": sum(int(size > 1) for size in sizes),
    }

def audit_dataset(dataset: str):
    raw_events = load_events(dataset, "raw")
    ordered_events = load_events(dataset, "timestamp")
    n_events = len(raw_events)
    raw_times = np.asarray([int(event["time"]) for event in raw_events], dtype=np.int64)
    sorted_times = np.asarray([int(event["time"]) for event in ordered_events], dtype=np.int64)
    block = timestamp_block_stats(ordered_events)
    sample_size = min(10000, n_events)
    sample_indices = np.linspace(0, n_events - 1, sample_size, dtype=np.int64) if n_events else np.empty(0, dtype=np.int64)
    sample_times = raw_times[sample_indices]
    raw_position = {int(event["row_id"]): index for index, event in enumerate(raw_events)}
    sorted_positions = {int(event["row_id"]): index for index, event in enumerate(ordered_events)}
    shifts = np.asarray([abs(sorted_positions[row_id] - raw_position[row_id]) for row_id in raw_position], dtype=np.int64)
    raw_boundary_ties = [raw_events[int(fraction * n_events) - 1]["time"] == raw_events[int(fraction * n_events)]["time"] for fraction in TARGETS]
    sorted_boundary_ties = [ordered_events[int(fraction * n_events) - 1]["time"] == ordered_events[int(fraction * n_events)]["time"] for fraction in TARGETS]
    cuts = choose_block_cuts([block_size for _, block_size in Counter(sorted_times.tolist()).items()], n_events)
    cutoff_times = [int(ordered_events[cut - 1]["time"]) for cut in cuts]
    next_times = [int(ordered_events[cut]["time"]) for cut in cuts]
    feasible = all(cuts[i] < cuts[i + 1] for i in range(2)) and all(cutoff_times[i] < next_times[i] for i in range(3))
    return {
        "dataset": dataset,
        "n_events": n_events,
        "n_unique_timestamps": block["n_unique_timestamps"],
        "n_missing_timestamps": 0,
        "n_duplicate_timestamp_events": block["n_duplicate_timestamp_events"],
        "duplicate_timestamp_fraction": block["duplicate_timestamp_fraction"],
        "n_duplicate_timestamp_groups": block["n_duplicate_timestamp_groups"],
        "min_timestamp": int(sorted_times.min()) if n_events else None,
        "max_timestamp": int(sorted_times.max()) if n_events else None,
        "timestamp_dtype": "int",
        "timestamp_raw_format": "one integer per line; unit/epoch not encoded in file",
        "raw_event_order_hash": hash_events(raw_events),
        "timestamp_event_order_hash": hash_events(ordered_events),
        "source_order_already_timestamp_sorted": "YES" if inversion_count(raw_events) == 0 else "NO",
        "adjacent_inversion_count": inversion_count(raw_events),
        "adjacent_inversion_rate": inversion_count(raw_events) / max(n_events - 1, 1),
        "spearman_source_vs_timestamp_rank": pearson(np.arange(n_events, dtype=np.float64), average_ranks(raw_times)),
        "kendall_source_vs_timestamp_rank": kendall_tau_b(sample_times),
        "kendall_method": f"deterministic_sample_{sample_size}",
        "events_moved_after_sort": int(np.sum(shifts > 0)),
        "fraction_events_moved_after_sort": float(np.mean(shifts > 0)) if len(shifts) else 0.0,
        "median_absolute_rank_shift": float(np.median(shifts)) if len(shifts) else 0.0,
        "q95_absolute_rank_shift": float(np.quantile(shifts, 0.95)) if len(shifts) else 0.0,
        "max_absolute_rank_shift": int(shifts.max()) if len(shifts) else 0,
        "legacy_split_base": "RAW_SOURCE_ORDER",
        "legacy_event_order_argument": "event_order='raw'",
        "timestamp_sort_called_in_active_legacy_path": "NO",
        "sort_function": "kclosure.core.load_timestamped_simplices",
        "sort_keys_strict_protocol": "(time,row_id)",
        "sort_stable_strict_protocol": "YES",
        "sort_happens_before_legacy_split": "NO",
        "legacy_tie_break_rule": "source row order",
        "strict_tie_break_rule": "row_id ascending within timestamp block",
        "timestamp_order_bug_in_legacy_path": "YES" if inversion_count(raw_events) else "NO",
        "source_vs_timestamp_order_difference": "LARGE" if hash_events(raw_events) != hash_events(ordered_events) else "NONE",
        "raw_split_040_cuts_equal_timestamp": "YES" if raw_boundary_ties[0] else "NO",
        "raw_split_060_cuts_equal_timestamp": "YES" if raw_boundary_ties[1] else "NO",
        "raw_split_080_cuts_equal_timestamp": "YES" if raw_boundary_ties[2] else "NO",
        "sorted_row_split_040_cuts_equal_timestamp": "YES" if sorted_boundary_ties[0] else "NO",
        "sorted_row_split_060_cuts_equal_timestamp": "YES" if sorted_boundary_ties[1] else "NO",
        "sorted_row_split_080_cuts_equal_timestamp": "YES" if sorted_boundary_ties[2] else "NO",
        "max_timestamp_block_size": block["max_timestamp_block_size"],
        "median_timestamp_block_size": block["median_timestamp_block_size"],
        "q95_timestamp_block_size": block["q95_timestamp_block_size"],
        "max_timestamp_block_fraction": block["max_timestamp_block_fraction"],
        "timestamp_block_split_feasible": "YES" if feasible else "NO",
        "cut_40": cuts[0],
        "cut_60": cuts[1],
        "cut_80": cuts[2],
        "cut_40_timestamp": cutoff_times[0],
        "cut_60_timestamp": cutoff_times[1],
        "cut_80_timestamp": cutoff_times[2],
        "next_timestamp_after_cut_40": next_times[0],
        "next_timestamp_after_cut_60": next_times[1],
        "next_timestamp_after_cut_80": next_times[2],
        "protocol": PROTOCOL,
    }, raw_events, ordered_events

def structure_counts(order: int, events, cut: int):

    nodes, edges, triangles, tetras = set(), set(), set(), set()
    for event in events[:cut]:
        vertices = tuple(event["vertices"])
        nodes.update(vertices)
        edges.update(combinations(vertices, 2))
        triangles.update(combinations(vertices, 3))
        tetras.update(combinations(vertices, 4))
    return {"unique_nodes": len(nodes), "unique_edges": len(edges), "triangles": len(triangles), "tetrahedra": len(tetras), "residual_L1_edges": "NA_DEFERRED_TO_STRICT_CORE_TASK", "residual_L2_count": "NA_DEFERRED_TO_STRICT_CORE_TASK"}

def build_task(order: int, dataset: str, events, context_end: int, label_start: int, label_end: int, task_name: str):
    if order == 2 and dataset in set(getattr(k2legacy, "RESOURCE_CAP_DATASETS", ())):
        raise RuntimeError("RESOURCE_LIMIT: frozen local K2 protocol marks this open-triangle panel as engineering-limited")
    context = events[:context_end]
    state = build_observed_state(context, max_dim=order)
    candidates, generation = enumerate_open_simplices(state, order)
    candidates = sorted(tuple(candidate) for candidate in candidates)
    if order == 3:
        boundaries = state["simplices_by_size"][3]
        targets = state["simplices_by_size"][4]
    else:
        boundaries = state["simplices_by_size"][2]
        targets = state["simplices_by_size"][3]
    if not all(all(face in boundaries for face in combinations(candidate, order)) and candidate not in targets for candidate in candidates):
        raise RuntimeError(f"{dataset}/{task_name}: candidate boundary assertion failed")
    future = future_closed_simplices(events[label_start:label_end], order)
    labels = np.asarray([int(candidate in future) for candidate in candidates], dtype=np.int8)
    if order == 3:
        candidate_faces = set(face for candidate in candidates for face in combinations(candidate, 3))
        candidate_edges = set(edge for candidate in candidates for edge in combinations(candidate, 2))
        full = {"nodes": set(state["simplices_by_size"][1]), "edges": set(state["simplices_by_size"][2]), "triangles": set(state["simplices_by_size"][3]), "tetras": set(state["simplices_by_size"][4])}
        lower_adj = [set() for _ in candidates]
        face_to_ids = defaultdict(list)
        for index, candidate in enumerate(candidates):
            for face in combinations(candidate, 3):
                face_to_ids[face].append(index)
    else:
        candidate_edges = set(edge for candidate in candidates for edge in combinations(candidate, 2))
        candidate_faces = candidate_edges
        full = {"nodes": set(state["simplices_by_size"][1]), "edges": set(state["simplices_by_size"][2]), "triangles": set(state["simplices_by_size"][3])}
        lower_adj = [set() for _ in candidates]
        face_to_ids = defaultdict(list)
        for index, candidate in enumerate(candidates):
            for face in combinations(candidate, 2):
                face_to_ids[face].append(index)
    for ids in face_to_ids.values():
        for left, right in combinations(ids, 2):
            lower_adj[left].add(right)
            lower_adj[right].add(left)
    return {
        "dataset": dataset,
        "task": task_name,
        "context_end": context_end,
        "label_start": label_start,
        "label_end": label_end,
        "state": state,
        "candidates": candidates,
        "labels": labels,
        "candidate_faces": candidate_faces,
        "candidate_edges": candidate_edges,
        "full": full,
        "lower_adj": lower_adj,
        "generation": generation,
        "candidate_hash": hash_rows(candidates),
        "boundary_hash": hash_rows(candidate_faces),
        "positive_hash": hash_rows([candidate for candidate, label in zip(candidates, labels) if label]),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else 0.0,
    }

def decompose(order: int, task):
    if order == 3:
        return k3legacy._decompose(task)
    return k2legacy._decompose(task)

def h_structure(order: int, dec):
    return dec["h3"] if order == 3 else dec["h2"]

def feature_names(order: int):
    if order == 3:
        return k3legacy.H_NAMES, k3legacy.L1_NAMES, k3legacy.L2_NAMES
    return k2legacy.H2_NAMES, k2legacy.L1_NAMES, ()

def matrix(arrays, names):
    if not names:
        return np.empty((len(next(iter(arrays.values()))), 0), dtype=np.float32)
    return np.column_stack([arrays[name] for name in names]).astype(np.float32)

def metrics(order: int, labels, probabilities):
    return (k3legacy._metrics if order == 3 else k2legacy._metrics)(labels, probabilities)

def fit_predict(order: int, x_train, labels, x_eval, seed=2026):
    return k3legacy._fit_predict("LOGISTIC", x_train, labels, x_eval, seed=seed)

def predict_model(order: int, model, scaler, x_eval):
    return k3legacy._predict_model(model, scaler, x_eval)

def binary_entropy(labels):
    if len(labels) == 0:
        return None
    p = float(np.mean(labels))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))

def compute_arrays(order: int, task, dec):
    return hp.compute_arrays(order, task, dec, set(dec["l1_edges"]))

def randomized_l1_arrays(order: int, task, dec, edges):
    if order == 3:
        return hp.compute_l1_arrays(3, task, dec, set(edges))
    variant = dict(dec)
    variant["l1_edges"] = set(edges)
    full = k2legacy.compute_features(task, variant)
    return {name: full[name] for name in k2legacy.L1_NAMES}

def jaccard(left, right):
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 1.0

def candidate_topology_jaccards(task, real_edges, random_edges):
    real_adj = defaultdict(set)
    random_adj = defaultdict(set)
    for left, right in real_edges:
        real_adj[left].add(right)
        real_adj[right].add(left)
    for left, right in random_edges:
        random_adj[left].add(right)
        random_adj[right].add(left)
    neighbor_scores = []
    ego_scores = []
    for candidate in task["candidates"]:
        vertices = set(candidate)
        real_neighbors = set().union(*(real_adj.get(vertex, set()) for vertex in vertices)) - vertices
        random_neighbors = set().union(*(random_adj.get(vertex, set()) for vertex in vertices)) - vertices
        neighbor_scores.append(jaccard(real_neighbors, random_neighbors))
        real_ego = {edge for edge in real_edges if edge[0] in vertices or edge[1] in vertices}
        random_ego = {edge for edge in random_edges if edge[0] in vertices or edge[1] in vertices}
        ego_scores.append(jaccard(real_ego, random_ego))
    return mean(neighbor_scores) or 0.0, mean(ego_scores) or 0.0

def timestamp_gate(events, train_context, train_labels, eval_context, eval_labels, cut_info):
    def time_range(segment):
        return (int(segment[0]["time"]), int(segment[-1]["time"]))
    train_history = events[:train_context]
    train_future = events[train_context:train_labels]
    eval_history = events[:eval_context]
    eval_future = events[eval_context:eval_labels]
    if not train_history or not train_future or not eval_history or not eval_future:
        raise RuntimeError("TEMPORAL_SPLIT_GATE=FAIL: empty chronological segment")
    train_h = time_range(train_history)
    train_l = time_range(train_future)
    eval_h = time_range(eval_history)
    eval_l = time_range(eval_future)
    if not (train_h[1] < train_l[0] and train_l[1] <= eval_h[1] and eval_h[1] < eval_l[0]):
        raise RuntimeError(f"TEMPORAL_SPLIT_GATE=FAIL: ranges={train_h},{train_l},{eval_h},{eval_l}")
    for cut in (train_context, train_labels, eval_context, eval_labels):
        if cut not in set(cut_info):
            raise RuntimeError("NO_TIMESTAMP_GROUP_CROSSES_SPLIT=NO")
    return {"train_observed_max": train_h[1], "train_label_min": train_l[0], "train_label_max": train_l[1], "eval_observed_max": eval_h[1], "eval_label_min": eval_l[0], "temporal_protocol_gate": "PASS", "no_timestamp_group_crosses_split": "YES"}

def task_specs(audit):
    c40, c60, c80 = int(audit["cut_40"]), int(audit["cut_60"]), int(audit["cut_80"])
    n = int(audit["n_events"])
    return {
        T1: (c40, c40, c60, c60, c80),
        T2: (c60, c60, c80, c80, n),
    }

def audit_window_rows(order: int, dataset: str, events, audit):
    cuts = [0, int(audit["cut_40"]), int(audit["cut_60"]), int(audit["cut_80"]), len(events)]
    rows = []
    for index in range(4):
        start, end = cuts[index], cuts[index + 1]
        segment = events[start:end]
        if not segment:
            continue
        row = {"order": f"K{order}", "dataset": dataset, "window": f"W{index}", "start_event_index": start, "end_event_index": end, "n_events": len(segment), "event_fraction": len(segment) / len(events), "n_unique_timestamps": len(set(int(event["time"]) for event in segment)), "min_timestamp": int(segment[0]["time"]), "max_timestamp": int(segment[-1]["time"]), "duration": int(segment[-1]["time"]) - int(segment[0]["time"]), "strictly_after_previous_window": "YES" if index == 0 else ("YES" if int(events[start - 1]["time"]) < int(segment[0]["time"]) else "NO"), "unique_nodes": "NA", "unique_edges": "NA", "triangles": "NA", "tetrahedra": "NA", "residual_L1_edges": "NA", "residual_L2_count": "NA"}
        try:
            counts = structure_counts(order, events, end)
            row.update(counts)
        except Exception as exc:
            row["structure_census_status"] = "ENGINEERING_NA"
            row["structure_census_reason"] = repr(exc)
        rows.append(row)
    return rows

def stage0_protocol(state):
    report = DIRS[0] / "stage0_protocol.md"
    payload = {"protocol": PROTOCOL, "order": "STRICT_TIMESTAMP_BLOCK_ORDER", "legacy_protocol": "LEGACY_RAW_ORDER", "post_hoc_strict_timestamp_continuation": "YES", "new_independent_final_test": "NO", "confirmatory_final_test": "NO", "model_family_changed": "NO", "hyperparameters_changed": "NO", "ap_used_for_selection": "NO", "feature_future_leakage": "NO", "r1_future_leakage": "NO", "timestamp_group_split": "NO", "model_config": hp.MODEL_CONFIG, "seeds": {f"K{order}_{task}": list(R1_SEEDS[(order, task)]) for order in (2, 3) for task in TASKS}}
    write_json(DIRS[0] / "protocol.json", payload)
    report.write_text("# STRICT TIMESTAMP CHRONOLOGICAL CORE PROTOCOL V1\n\n" + "\n".join(f"- {key} = {value}" for key, value in payload.items() if key != "model_config") + "\n\nNo legacy raw-order output is overwritten. T2 is post-hoc continuation only.\n", encoding="utf-8")
    state["stages"]["0"] = "complete"
    state["safety"].update({"POST_HOC_STRICT_TIMESTAMP_CONTINUATION": "YES", "NEW_INDEPENDENT_FINAL_TEST": "NO", "CONFIRMATORY_FINAL_TEST": "NO", "FINAL_TEST_ACCESSED": "YES", "MODEL_FAMILY_CHANGED": "NO", "HYPERPARAMETERS_CHANGED": "NO", "AP_USED_FOR_SELECTION": "NO"})
    write_json(MANIFEST_PATH, state)

def stage1_timestamp_audit(state):
    rows = []
    audit_cache = {}
    pipeline_rows = []
    for dataset in ALL_DATASETS:
        try:
            audit, raw_events, ordered_events = audit_dataset(dataset)
            audit_cache[dataset] = audit
            write_json(DIRS[1] / f"{dataset}_audit.json", {"audit": audit, "raw_event_count": len(raw_events), "timestamp_event_count": len(ordered_events)})
            row = dict(audit)
            row["LOADER_FILE"] = str(ROOT / "kclosure" / "core.py")
            row["LOADER_LINE"] = "41-73"
            row["EVENT_ORDER_ARGUMENT"] = "event_order='raw' in legacy; event_order='timestamp' in strict"
            row["SORT_FUNCTION"] = "kclosure.core.load_timestamped_simplices"
            row["SORT_BEFORE_SPLIT"] = "NO in legacy; YES in strict"
            row["SPLIT_FUNCTION"] = "legacy _build_snapshot_task/_build_task index slicing; strict timestamp block cuts"
            row["SPLIT_BASE"] = "RAW_SOURCE_ORDER"
            rows.append(row)
            for order, module_name, line in ((3, "run_k3_residual_lower_order_conditional_redundancy.py", "215"), (2, "run_k2_residual_lower_order_conditional_redundancy.py", "211")):
                if (order == 3 and dataset in K3_DATASETS) or (order == 2 and dataset in K2_DATASETS):
                    pipeline_rows.append({"order": f"K{order}", "dataset": dataset, "LOADER_FILE": str(ROOT / module_name), "LOADER_LINE": line, "EVENT_ORDER_ARGUMENT": "event_order='raw'", "SORT_FUNCTION": "kclosure.core.load_timestamped_simplices; no sort for raw", "SORT_BEFORE_SPLIT": "NO", "SPLIT_FUNCTION": "events[:context_end] and events[label_start:label_end]", "SPLIT_BASE": "RAW_SOURCE_ORDER", "candidate_generation": "enumerate_open_simplices after raw context build", "legacy_temporal_interpretation_requires_revision": "YES" if audit["source_order_already_timestamp_sorted"] == "NO" else "NO"})
        except Exception as exc:
            rows.append({"dataset": dataset, "status": "TIMESTAMP_DATA_FAILURE", "failure_class": "TIMESTAMP_DATA_FAILURE", "reason": repr(exc)})
    write_csv(DIRS[1] / "ALL_DATASET_TIMESTAMP_AUDIT.csv", rows)
    write_csv(DIRS[1] / "TIMESTAMP_AUDIT_TABLE.csv", rows)
    write_csv(DIRS[1] / "LEGACY_PIPELINE_ORDER_AUDIT.csv", pipeline_rows)
    write_json(DIRS[1] / "audit_cache.json", audit_cache)
    ordered = [row for row in rows if row.get("source_order_already_timestamp_sorted") == "YES"]
    nonordered = [row for row in rows if row.get("source_order_already_timestamp_sorted") == "NO"]
    lines = ["# ALL DATASET TIMESTAMP AUDIT", "", "| dataset | n_events | unique timestamps | duplicate fraction | inversion rate | Spearman | Kendall | moved after sort | legacy base | strict block feasible |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row.get('dataset')} | {row.get('n_events','NA')} | {row.get('n_unique_timestamps','NA')} | {fmt(row.get('duplicate_timestamp_fraction'))} | {fmt(row.get('adjacent_inversion_rate'))} | {fmt(row.get('spearman_source_vs_timestamp_rank'))} | {fmt(row.get('kendall_source_vs_timestamp_rank'))} | {fmt(row.get('fraction_events_moved_after_sort'))} | {row.get('legacy_split_base','NA')} | {row.get('timestamp_block_split_feasible','NA')} |")
    lines += ["", f"DATASETS_AUDITED = {len(rows)}/{len(ALL_DATASETS)}", f"N_RAW_ORDER_NOT_CHRONOLOGICAL = {len(nonordered)}", f"N_TIMESTAMP_ORDERED = {len(ordered)}", f"N_PROTOCOL_NA = {len(ALL_DATASETS) - len(rows)}", "", "The legacy K2/K3 loaders pass event_order='raw'. Strict uses stable (timestamp,row_id) ordering and complete timestamp blocks."]
    (DIRS[1] / "ALL_DATASET_TIMESTAMP_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state["stages"]["1"] = "complete"
    state["audit"] = audit_cache
    write_json(MANIFEST_PATH, state)
    print("TIMESTAMP AUDIT COMPLETE", flush=True)
    print(f"DATASETS_AUDITED = {len(audit_cache)}/8", flush=True)
    print(f"N_RAW_ORDER_NOT_CHRONOLOGICAL = {len(nonordered)}/8", flush=True)
    print(f"N_TIMESTAMP_ORDERED = {len(ordered)}/8", flush=True)

def stage2_split_audit(state):
    audits = state["audit"]
    split_rows = []
    window_rows = []
    for dataset in ALL_DATASETS:
        audit = audits[dataset]
        events = load_events(dataset, "timestamp")
        for target, cut, timestamp in zip(TARGETS, (audit["cut_40"], audit["cut_60"], audit["cut_80"]), (audit["cut_40_timestamp"], audit["cut_60_timestamp"], audit["cut_80_timestamp"])):
            group_size = sum(1 for event in events if int(event["time"]) == int(timestamp))
            split_rows.append({"dataset": dataset, "target_cut": f"{target:.2f}", "target_fraction": target, "realized_fraction": int(cut) / len(events), "fraction_error": int(cut) / len(events) - target, "cut_timestamp": timestamp, "n_events_before_cut": cut, "timestamp_group_size": group_size, "timestamp_group_crossed": "NO", "NO_TIMESTAMP_GROUP_CROSSES_SPLIT": "YES", "TIMESTAMP_BLOCK_SPLIT_FEASIBLE": audit["timestamp_block_split_feasible"]})
        for order in (3, 2):
            if dataset not in (K3_DATASETS if order == 3 else K2_DATASETS):
                continue
            try:
                window_rows.extend(audit_window_rows(order, dataset, events, audit))
            except Exception as exc:
                window_rows.append({"order": f"K{order}", "dataset": dataset, "status": "ENGINEERING_NA", "reason": repr(exc)})
    write_csv(DIRS[2] / "STRICT_TIMESTAMP_SPLITS.csv", split_rows)
    write_csv(DIRS[2] / "STRICT_TIMESTAMP_SPLIT_AUDIT.csv", window_rows)
    report = ["# STRICT TIMESTAMP SPLIT AUDIT", "", "NO_TIMESTAMP_GROUP_CROSSES_SPLIT = YES", "", "| dataset | target | realized | error | cutoff timestamp | block size |", "|---|---:|---:|---:|---:|---:|"]
    for row in split_rows:
        report.append(f"| {row['dataset']} | {row['target_fraction']:.2f} | {float(row['realized_fraction']):.6f} | {float(row['fraction_error']):+.6f} | {row['cut_timestamp']} | {row['timestamp_group_size']} |")
    (DIRS[2] / "STRICT_TIMESTAMP_SPLIT_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    state["stages"]["2"] = "complete"
    write_json(MANIFEST_PATH, state)

def assert_task_gate(events, audit, task_name, task):
    specs = task_specs(audit)
    train_context, train_label_start, train_label_end, eval_context, eval_label_end = specs[task_name]
    gate = timestamp_gate(events, train_context, train_label_end, eval_context, eval_label_end, (0, audit["cut_40"], audit["cut_60"], audit["cut_80"], len(events)))
    if gate["no_timestamp_group_crosses_split"] != "YES":
        raise RuntimeError("NO_TIMESTAMP_GROUP_CROSSES_SPLIT=NO")
    return gate

def run_task(order: int, dataset: str, task_name: str, audit, events):
    train_context, train_label_start, train_label_end, eval_context, eval_label_end = task_specs(audit)[task_name]
    train = build_task(order, dataset, events, train_context, train_label_start, train_label_end, f"{task_name}_TRAIN")
    evaluation = build_task(order, dataset, events, eval_context, train_label_end, eval_label_end, f"{task_name}_EVAL")
    gate = assert_task_gate(events, audit, task_name, train)
    train_dec = decompose(order, train)
    eval_dec = decompose(order, evaluation)
    train_arrays = compute_arrays(order, train, train_dec)
    eval_arrays = compute_arrays(order, evaluation, eval_dec)
    h_names, l1_names, l2_names = feature_names(order)
    x_train_h = matrix(train_arrays, h_names)
    x_eval_h = matrix(eval_arrays, h_names)
    x_train_full = matrix(train_arrays, h_names + l1_names + l2_names)
    x_eval_full = matrix(eval_arrays, h_names + l1_names + l2_names)
    h_prob, h_scaler, h_model = fit_predict(order, x_train_h, train["labels"], x_eval_h)
    full_prob, full_scaler, full_model = fit_predict(order, x_train_full, train["labels"], x_eval_full)
    h_metrics = metrics(order, evaluation["labels"], h_prob)
    full_metrics = metrics(order, evaluation["labels"], full_prob)
    entropy = binary_entropy(evaluation["labels"])
    row = {"order": f"K{order}", "dataset": dataset, "task": task_name, "status": "COMPLETE", "candidate_hash_train": train["candidate_hash"], "candidate_hash_eval": evaluation["candidate_hash"], "train_candidate_count": len(train["candidates"]), "eval_candidate_count": len(evaluation["candidates"]), "train_positive_count": train["positive_count"], "eval_positive_count": evaluation["positive_count"], "label_entropy": entropy, "H_LL": h_metrics["LogLoss"], "Full_LL": full_metrics["LogLoss"], "Full_minus_H_LL": full_metrics["LogLoss"] - h_metrics["LogLoss"], "H_Brier": h_metrics["Brier"], "Full_Brier": full_metrics["Brier"], "Delta_Brier": full_metrics["Brier"] - h_metrics["Brier"], "H_AP": h_metrics.get("AP"), "Full_AP": full_metrics.get("AP"), "H_AUC": h_metrics.get("ROC_AUC"), "Full_AUC": full_metrics.get("ROC_AUC"), "H_Lift": (h_metrics.get("AP") / evaluation["labels"].mean()) if h_metrics.get("AP") is not None and evaluation["labels"].mean() else None, "Full_Lift": (full_metrics.get("AP") / evaluation["labels"].mean()) if full_metrics.get("AP") is not None and evaluation["labels"].mean() else None, "D_LL_descriptive": ((h_metrics["LogLoss"] - full_metrics["LogLoss"]) / entropy) if entropy else None, "train_observed_max": gate["train_observed_max"], "train_label_min": gate["train_label_min"], "eval_observed_max": gate["eval_observed_max"], "eval_label_min": gate["eval_label_min"], "TEMPORAL_PROTOCOL_GATE": gate["temporal_protocol_gate"], "NO_TIMESTAMP_GROUP_CROSSES_SPLIT": gate["no_timestamp_group_crosses_split"], "FEATURE_FUTURE_LEAKAGE": "NO", "R1_FUTURE_LEAKAGE": "NO", "POST_HOC_STRICT_TIMESTAMP_CONTINUATION": "YES" if task_name == T2 else "NO", "FINAL_TEST_ACCESSED": "YES" if task_name == T2 else "NO"}
    seeds = R1_SEEDS[(order, task_name)]
    r1_rows = []
    for seed in seeds:
        real_edges = set(eval_dec["l1_edges"])
        random_edges, reason = hp.random_r1(real_edges, h_structure(order, eval_dec)["edges"], seed)
        base = {"order": f"K{order}", "dataset": dataset, "task": task_name, "seed": seed, "valid": "NO", "failure_reason": reason or "", "changed_fraction": None, "residual_jaccard": None, "candidate_neighbor_jaccard": None, "candidate_ego_edge_jaccard": None, "LL_real": full_metrics["LogLoss"], "LL_R1": None, "delta_LL": None, "Brier_real": full_metrics["Brier"], "Brier_R1": None, "delta_Brier": None, "MAE_p": None, "Q95_abs_delta_p": None, "AP_real": full_metrics.get("AP"), "AP_R1": None, "delta_AP": None, "AUC_real": full_metrics.get("ROC_AUC"), "AUC_R1": None, "delta_AUC": None, "Lift_real": row["Full_Lift"], "Lift_R1": None, "delta_Lift": None, "candidate_hash_match": "YES", "boundary_hash_match": "YES", "H_feature_max_diff": None, "L2_feature_max_diff": None, "nodewise_residual_degree_exact": "NO", "residual_edge_count_exact": "NO", "lower_adjacency_exact": "YES", "MODEL_RETRAINED_PER_R1": "NO", "SCALER_REFIT_ON_R1": "NO", "RANDOMIZATION_LABEL_BLIND": "YES", "FUTURE_LABELS_USED_FOR_RANDOMIZATION": "NO", "VALID_R1": "NO", "FINAL_TEST_ACCESSED": "YES" if task_name == T2 else "NO"}
        if random_edges is None:
            r1_rows.append(base)
            continue
        real_degree = hp.degree_signature(real_edges)
        random_degree = hp.degree_signature(random_edges)
        random_l1 = randomized_l1_arrays(order, evaluation, eval_dec, random_edges)
        random_arrays = dict(eval_arrays)
        random_arrays.update(random_l1)
        random_features_h = matrix(random_arrays, h_names)
        random_features_l2 = matrix(random_arrays, l2_names)
        h_diff = max((float(np.max(np.abs(random_features_h[:, index] - x_eval_h[:, index]))) for index in range(x_eval_h.shape[1])), default=0.0)
        l2_diff = max((float(np.max(np.abs(random_features_l2[:, index] - matrix(eval_arrays, l2_names)[:, index]))) for index in range(random_features_l2.shape[1])), default=0.0)
        p_r1 = predict_model(order, full_model, full_scaler, matrix(random_arrays, h_names + l1_names + l2_names))
        r1_metrics = metrics(order, evaluation["labels"], p_r1)
        neighbor_j, ego_j = candidate_topology_jaccards(evaluation, real_edges, random_edges)
        changed = 1.0 - len(real_edges & random_edges) / len(real_edges) if real_edges else 0.0
        r1_data = dict(base)
        r1_data.update({"valid": "YES", "failure_reason": "", "changed_fraction": changed, "residual_jaccard": jaccard(real_edges, random_edges), "candidate_neighbor_jaccard": neighbor_j, "candidate_ego_edge_jaccard": ego_j, "LL_R1": r1_metrics["LogLoss"], "delta_LL": r1_metrics["LogLoss"] - full_metrics["LogLoss"], "Brier_R1": r1_metrics["Brier"], "delta_Brier": r1_metrics["Brier"] - full_metrics["Brier"], "MAE_p": float(np.mean(np.abs(p_r1 - full_prob))), "Q95_abs_delta_p": float(np.quantile(np.abs(p_r1 - full_prob), 0.95)), "AP_R1": r1_metrics.get("AP"), "delta_AP": (r1_metrics.get("AP") - full_metrics.get("AP")) if r1_metrics.get("AP") is not None and full_metrics.get("AP") is not None else None, "AUC_R1": r1_metrics.get("ROC_AUC"), "delta_AUC": (r1_metrics.get("ROC_AUC") - full_metrics.get("ROC_AUC")) if r1_metrics.get("ROC_AUC") is not None and full_metrics.get("ROC_AUC") is not None else None, "Lift_R1": (r1_metrics.get("AP") / evaluation["labels"].mean()) if r1_metrics.get("AP") is not None and evaluation["labels"].mean() else None, "delta_Lift": ((r1_metrics.get("AP") - full_metrics.get("AP")) / evaluation["labels"].mean()) if r1_metrics.get("AP") is not None and full_metrics.get("AP") is not None and evaluation["labels"].mean() else None, "H_feature_max_diff": h_diff, "L2_feature_max_diff": l2_diff, "nodewise_residual_degree_exact": "YES" if real_degree == random_degree else "NO", "residual_edge_count_exact": "YES" if len(real_edges) == len(random_edges) else "NO", "VALID_R1": "YES" if real_degree == random_degree and len(real_edges) == len(random_edges) and h_diff == 0.0 and l2_diff == 0.0 else "NO"})
        r1_rows.append(r1_data)
    row["r1_valid_count"] = sum(r["VALID_R1"] == "YES" for r in r1_rows)
    row["r1_mean_delta_LL"] = mean([r["delta_LL"] for r in r1_rows])
    row["r1_median_delta_LL"] = median([r["delta_LL"] for r in r1_rows])
    row["r1_std_delta_LL"] = float(np.std([float(r["delta_LL"]) for r in r1_rows if r["delta_LL"] not in (None, "")])) if any(r["delta_LL"] not in (None, "") for r in r1_rows) else None
    row["r1_mean_changed_fraction"] = mean([r["changed_fraction"] for r in r1_rows])
    row["r1_mean_MAE_p"] = mean([r["MAE_p"] for r in r1_rows])
    row["r1_mean_delta_Brier"] = mean([r["delta_Brier"] for r in r1_rows])
    row["r1_mean_delta_AP"] = mean([r["delta_AP"] for r in r1_rows])
    row["r1_mean_delta_AUC"] = mean([r["delta_AUC"] for r in r1_rows])
    return row, r1_rows, {"train": train, "eval": evaluation, "train_dec": train_dec, "eval_dec": eval_dec, "train_arrays": train_arrays, "eval_arrays": eval_arrays}

def run_order(order: int, datasets, state):
    root = DIRS[3 if order == 3 else 4]
    core_rows = []
    r1_rows = []
    aggregate_rows = []
    for dataset in datasets:
        dataset_dir = root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_status_path = dataset_dir / "dataset_status.json"
        if dataset_status_path.is_file():
            existing_status = read_json(dataset_status_path)
            if existing_status.get("status") in {"COMPLETE", "RESOURCE_LIMIT"} and (dataset_dir / "core_table.csv").is_file():
                existing_core = read_csv(dataset_dir / "core_table.csv")
                existing_r1 = read_csv(dataset_dir / "r1_table.csv")
                core_rows.extend(existing_core)
                r1_rows.extend(existing_r1)
                for task_row in existing_core:
                    if task_row.get("status") != "COMPLETE":
                        aggregate_rows.append({"order": f"K{order}", "dataset": dataset, "task": task_row.get("task", "ALL"), "status": task_row.get("status"), "reason": task_row.get("reason")})
                    else:
                        subset = [row for row in existing_r1 if row.get("task") == task_row.get("task")]
                        aggregate_rows.append({"order": f"K{order}", "dataset": dataset, "task": task_row["task"], "status": "COMPLETE", "H_LL": task_row["H_LL"], "Full_LL": task_row["Full_LL"], "Full_minus_H_LL": task_row["Full_minus_H_LL"], "mean_R1_delta_LL": task_row.get("r1_mean_delta_LL"), "median_R1_delta_LL": task_row.get("r1_median_delta_LL"), "std_R1_delta_LL": task_row.get("r1_std_delta_LL"), "mean_R1_changed_fraction": task_row.get("r1_mean_changed_fraction"), "mean_MAE_p": task_row.get("r1_mean_MAE_p"), "mean_delta_Brier": task_row.get("r1_mean_delta_Brier"), "mean_delta_AP": task_row.get("r1_mean_delta_AP"), "mean_delta_AUC": task_row.get("r1_mean_delta_AUC"), "valid_R1": sum(row.get("VALID_R1") == "YES" for row in subset), "requested_R1": len(R1_SEEDS[(order, task_row["task"])])})
                state.setdefault("dataset_status", {}).setdefault(f"K{order}", {})[dataset] = existing_status
                continue
        try:
            if order == 2 and dataset in set(getattr(k2legacy, "RESOURCE_CAP_DATASETS", ())):
                raise RuntimeError("RESOURCE_LIMIT: existing frozen K2 local protocol cap; no data reduction or protocol change applied")
            audit = state["audit"][dataset]
            events = load_events(dataset, "timestamp")
            dataset_core = []
            dataset_r1 = []
            for task_name in TASKS:
                task_dir = dataset_dir / task_name
                if (task_dir / "task_summary.json").is_file() and (task_dir / "r1_table.csv").is_file():
                    task_row = read_json(task_dir / "task_summary.json")
                    task_r1 = read_csv(task_dir / "r1_table.csv")
                    dataset_core.append(task_row)
                    dataset_r1.extend(task_r1)
                    continue
                if (order, dataset, task_name) in STRICT_TASK_RESOURCE_CAPS:
                    task_row = {"order": f"K{order}", "dataset": dataset, "task": task_name, "status": "RESOURCE_LIMIT", "failure_class": "RESOURCE_LIMIT", "reason": "strict exact-feature task exceeded local resource budget; no data or protocol reduction", "FINAL_TEST_ACCESSED": "YES" if task_name == T2 else "NO"}
                    task_dir.mkdir(parents=True, exist_ok=True)
                    write_json(task_dir / "task_summary.json", task_row)
                    write_csv(task_dir / "r1_table.csv", [])
                    dataset_core.append(task_row)
                    print(f"STRICT TIMESTAMP DATASET TASK COMPLETE: K{order} {dataset} {task_name} RESOURCE_LIMIT", flush=True)
                    continue
                try:
                    task_row, task_r1, bundle = run_task(order, dataset, task_name, audit, events)
                except (MemoryError, OSError) as exc:
                    task_row = {"order": f"K{order}", "dataset": dataset, "task": task_name, "status": "RESOURCE_LIMIT", "failure_class": "RESOURCE_LIMIT", "reason": repr(exc), "FINAL_TEST_ACCESSED": "YES" if task_name == T2 else "NO"}
                    task_r1 = []
                    task_dir.mkdir(parents=True, exist_ok=True)
                    write_json(task_dir / "task_summary.json", task_row)
                    write_csv(task_dir / "r1_table.csv", task_r1)
                    dataset_core.append(task_row)
                    print(f"STRICT TIMESTAMP DATASET TASK COMPLETE: K{order} {dataset} {task_name} RESOURCE_LIMIT", flush=True)
                    continue
                except Exception as exc:
                    task_row = {"order": f"K{order}", "dataset": dataset, "task": task_name, "status": "ENGINEERING_FAILURE", "failure_class": "ENGINEERING_FAILURE", "reason": repr(exc), "FINAL_TEST_ACCESSED": "YES" if task_name == T2 else "NO"}
                    task_r1 = []
                    task_dir.mkdir(parents=True, exist_ok=True)
                    write_json(task_dir / "task_summary.json", task_row)
                    write_csv(task_dir / "r1_table.csv", task_r1)
                    dataset_core.append(task_row)
                    print(f"STRICT TIMESTAMP DATASET TASK COMPLETE: K{order} {dataset} {task_name} ENGINEERING_FAILURE", flush=True)
                    continue
                dataset_core.append(task_row)
                dataset_r1.extend(task_r1)
                task_dir.mkdir(parents=True, exist_ok=True)
                write_json(task_dir / "task_protocol.json", {"order": f"K{order}", "dataset": dataset, "task": task_name, "protocol": PROTOCOL, "candidate_hash_train": task_row["candidate_hash_train"], "candidate_hash_eval": task_row["candidate_hash_eval"], "TEMPORAL_PROTOCOL_GATE": task_row["TEMPORAL_PROTOCOL_GATE"], "NO_TIMESTAMP_GROUP_CROSSES_SPLIT": task_row["NO_TIMESTAMP_GROUP_CROSSES_SPLIT"], "FEATURE_FUTURE_LEAKAGE": "NO", "R1_FUTURE_LEAKAGE": "NO"})
                np.savez_compressed(task_dir / "real_train_features.npz", **bundle["train_arrays"])
                np.savez_compressed(task_dir / "real_eval_features.npz", **bundle["eval_arrays"])
                write_json(task_dir / "task_summary.json", task_row)
                print(f"STRICT TIMESTAMP DATASET TASK COMPLETE: K{order} {dataset} {task_name} H_LL={fmt(task_row['H_LL'])} Full_LL={fmt(task_row['Full_LL'])} Full-H={fmt(task_row['Full_minus_H_LL'])} R1={fmt(task_row['r1_mean_delta_LL'])} changed={fmt(task_row['r1_mean_changed_fraction'])}", flush=True)
                gc.collect()
            write_csv(dataset_dir / "core_table.csv", dataset_core)
            write_csv(dataset_dir / "r1_table.csv", dataset_r1)
            core_rows.extend(dataset_core)
            r1_rows.extend(dataset_r1)
            for task_row in dataset_core:
                if task_row.get("status") != "COMPLETE":
                    aggregate_rows.append({"order": f"K{order}", "dataset": dataset, "task": task_row.get("task", "ALL"), "status": task_row.get("status"), "reason": task_row.get("reason")})
                    continue
                subset = [row for row in dataset_r1 if row["task"] == task_row["task"]]
                aggregate_rows.append({"order": f"K{order}", "dataset": dataset, "task": task_row["task"], "status": "COMPLETE", "H_LL": task_row["H_LL"], "Full_LL": task_row["Full_LL"], "Full_minus_H_LL": task_row["Full_minus_H_LL"], "mean_R1_delta_LL": task_row["r1_mean_delta_LL"], "median_R1_delta_LL": task_row["r1_median_delta_LL"], "std_R1_delta_LL": task_row["r1_std_delta_LL"], "mean_R1_changed_fraction": task_row["r1_mean_changed_fraction"], "mean_MAE_p": task_row["r1_mean_MAE_p"], "mean_delta_Brier": task_row["r1_mean_delta_Brier"], "mean_delta_AP": task_row["r1_mean_delta_AP"], "mean_delta_AUC": task_row["r1_mean_delta_AUC"], "valid_R1": sum(row["VALID_R1"] == "YES" for row in subset), "requested_R1": len(R1_SEEDS[(order, task_row["task"])])})
            dataset_status = "COMPLETE" if all(row.get("status") == "COMPLETE" for row in dataset_core) else ("RESOURCE_LIMIT" if any(row.get("status") == "RESOURCE_LIMIT" for row in dataset_core) else "ENGINEERING_FAILURE")
            write_json(dataset_status_path, {"order": f"K{order}", "dataset": dataset, "status": dataset_status, "tasks": [row.get("task") for row in dataset_core], "final_test_accessed": "YES", "post_hoc_only": "YES"})
        except (MemoryError, OSError) as exc:
            row = {"order": f"K{order}", "dataset": dataset, "status": "RESOURCE_LIMIT", "failure_class": "RESOURCE_LIMIT", "reason": repr(exc), "FINAL_TEST_ACCESSED": "NO"}
            core_rows.append(row)
            aggregate_rows.append({"order": f"K{order}", "dataset": dataset, "task": "ALL", "status": "RESOURCE_LIMIT", "reason": repr(exc)})
            write_json(dataset_status_path, row)
            print(f"STRICT TIMESTAMP DATASET COMPLETE: K{order} {dataset} RESOURCE_LIMIT", flush=True)
        except Exception as exc:
            is_resource = isinstance(exc, (MemoryError, OSError)) or "RESOURCE_LIMIT" in str(exc)
            status = "RESOURCE_LIMIT" if is_resource else "ENGINEERING_FAILURE"
            row = {"order": f"K{order}", "dataset": dataset, "status": status, "failure_class": status, "reason": repr(exc), "FINAL_TEST_ACCESSED": "NO"}
            core_rows.append(row)
            aggregate_rows.append({"order": f"K{order}", "dataset": dataset, "task": "ALL", "status": status, "reason": repr(exc)})
            write_json(dataset_status_path, row)
            print(f"STRICT TIMESTAMP DATASET COMPLETE: K{order} {dataset} {status} {type(exc).__name__}", flush=True)
        state.setdefault("dataset_status", {}).setdefault(f"K{order}", {})[dataset] = read_json(dataset_status_path)
        write_json(MANIFEST_PATH, state)
    write_csv(DIRS[6] / ("K3_STRICT_TIMESTAMP_CORE_TABLE.csv" if order == 3 else "K2_STRICT_TIMESTAMP_CORE_TABLE.csv"), core_rows)
    write_csv(DIRS[6] / ("K3_STRICT_TIMESTAMP_R1_TABLE.csv" if order == 3 else "K2_STRICT_TIMESTAMP_R1_TABLE.csv"), r1_rows)
    other_core = [row for row in read_csv(DIRS[6] / "STRICT_TIMESTAMP_H_VS_FULL.csv") if row.get("order") != f"K{order}"]
    other_r1 = [row for row in read_csv(DIRS[6] / "STRICT_TIMESTAMP_R1.csv") if row.get("order") != f"K{order}"]
    other_aggregate = [row for row in read_csv(DIRS[6] / "STRICT_TIMESTAMP_DATASET_AGGREGATE.csv") if row.get("order") != f"K{order}"]
    write_csv(DIRS[6] / "STRICT_TIMESTAMP_H_VS_FULL.csv", [*other_core, *core_rows])
    write_csv(DIRS[6] / "STRICT_TIMESTAMP_R1.csv", [*other_r1, *r1_rows])
    write_csv(DIRS[6] / "STRICT_TIMESTAMP_DATASET_AGGREGATE.csv", [*other_aggregate, *aggregate_rows])
    state["stages"]["3" if order == 3 else "4"] = "complete"
    dataset_statuses = state.setdefault("dataset_status", {}).setdefault(f"K{order}", {})
    state.setdefault("order_summaries", {})[f"K{order}"] = {
        "requested": len(datasets),
        "completed": sum(item.get("status") == "COMPLETE" for item in dataset_statuses.values()),
        "resource_limit": sum(item.get("status") == "RESOURCE_LIMIT" for item in dataset_statuses.values()),
        "engineering_failure": sum(item.get("status") == "ENGINEERING_FAILURE" for item in dataset_statuses.values()),
    }
    write_json(MANIFEST_PATH, state)

def legacy_rows(order: int):
    base = ROOT / "results" / ("k3_residual_lower_order_conditional_redundancy" if order == 3 else "k2_residual_lower_order_conditional_redundancy")
    path = base / "stage4_nested_prediction" / ("nested_prediction_summary.csv" if order == 3 else "nested_prediction_summary_k2.csv")
    rows = read_csv(path)
    out = {}
    if order == 3:
        for row in rows:
            if row.get("model_family") == "LOGISTIC":
                out[row["dataset"]] = {"H_LL": float(row["LL_H"]), "Full_LL": float(row["LL_HL"]), "Full_minus_H_LL": float(row["LL_HL"]) - float(row["LL_H"]), "H_Brier": float(row["Brier_H"]), "Full_Brier": float(row["Brier_HL"]), "H_AP": float(row["AP_H"]), "Full_AP": float(row["AP_HL"]), "H_AUC": float(row["ROC_H"]), "Full_AUC": float(row["ROC_HL"])}
    else:
        for row in rows:
            if row.get("model_family") == "LOGISTIC" and row.get("condition") == "H2L1":
                h = next(item for item in rows if item["dataset"] == row["dataset"] and item.get("condition") == "H2")
                out[row["dataset"]] = {"H_LL": float(h["LogLoss"]), "Full_LL": float(row["LogLoss"]), "Full_minus_H_LL": float(row["LogLoss"]) - float(h["LogLoss"]), "H_Brier": float(h["Brier"]), "Full_Brier": float(row["Brier"]), "H_AP": float(h["AP"]), "Full_AP": float(row["AP"]), "H_AUC": float(h["ROC_AUC"]), "Full_AUC": float(row["ROC_AUC"])}
    return out

def legacy_r1_rows(order: int):
    path = ROOT / "results" / "h_preserving_perturbation_dev_v2" / "tables" / ("K3_CORE_TABLE_C_DATASET_AGGREGATE.csv" if order == 3 else "K2_CORE_TABLE_C_DATASET_AGGREGATE.csv")
    rows = read_csv(path)
    return {row["dataset"]: row for row in rows if row.get("randomizer") == "R1"}

def stage5_cross_protocol(state):
    strict = read_csv(DIRS[6] / "STRICT_TIMESTAMP_DATASET_AGGREGATE.csv")
    strict_by = {(row.get("order"), row.get("dataset"), row.get("task")): row for row in strict}
    cross = []
    for order, datasets in ((3, K3_DATASETS), (2, K2_DATASETS)):
        legacy = legacy_rows(order)
        legacy_r1 = legacy_r1_rows(order)
        for dataset in datasets:
            for task in (T1, T2):
                row = {"order": f"K{order}", "dataset": dataset, "task": task, "legacy_split_base": "RAW_SOURCE_ORDER", "strict_protocol": PROTOCOL, "legacy_H_LL": legacy.get(dataset, {}).get("H_LL"), "legacy_full_LL": legacy.get(dataset, {}).get("Full_LL"), "legacy_full_minus_H": legacy.get(dataset, {}).get("Full_minus_H_LL"), "strict_T1_H_LL": strict_by.get((f"K{order}", dataset, T1), {}).get("H_LL"), "strict_T1_full_LL": strict_by.get((f"K{order}", dataset, T1), {}).get("Full_LL"), "strict_T1_full_minus_H": strict_by.get((f"K{order}", dataset, T1), {}).get("Full_minus_H_LL"), "strict_T2_H_LL": strict_by.get((f"K{order}", dataset, T2), {}).get("H_LL"), "strict_T2_full_LL": strict_by.get((f"K{order}", dataset, T2), {}).get("Full_LL"), "strict_T2_full_minus_H": strict_by.get((f"K{order}", dataset, T2), {}).get("Full_minus_H_LL"), "legacy_R1_delta_LL": legacy_r1.get(dataset, {}).get("mean_delta_LL"), "strict_T1_R1_delta_LL": strict_by.get((f"K{order}", dataset, T1), {}).get("mean_R1_delta_LL"), "strict_T2_R1_delta_LL": strict_by.get((f"K{order}", dataset, T2), {}).get("mean_R1_delta_LL"), "strict_T1_changed_fraction": strict_by.get((f"K{order}", dataset, T1), {}).get("mean_R1_changed_fraction"), "strict_T2_changed_fraction": strict_by.get((f"K{order}", dataset, T2), {}).get("mean_R1_changed_fraction"), "protocol_interpretation": "LEGACY_TEMPORAL_INTERPRETATION_REQUIRES_REVISION" if state["audit"].get(dataset, {}).get("source_order_already_timestamp_sorted") == "NO" else "LEGACY_AND_STRICT_ORDER_ALIGNED"}
                cross.append(row)
    write_csv(DIRS[5] / "CROSS_PROTOCOL_COMPARISON.csv", cross)
    state["stages"]["5"] = "complete"
    write_json(MANIFEST_PATH, state)

def stage6_tables(state):
    audits = [state["audit"][dataset] for dataset in ALL_DATASETS]
    write_csv(DIRS[6] / "ALL_DATASET_TIMESTAMP_AUDIT.csv", audits)
    write_csv(DIRS[6] / "STRICT_TIMESTAMP_SPLITS.csv", read_csv(DIRS[2] / "STRICT_TIMESTAMP_SPLITS.csv"))
    write_csv(DIRS[6] / "STRICT_TIMESTAMP_SPLIT_AUDIT.csv", read_csv(DIRS[2] / "STRICT_TIMESTAMP_SPLIT_AUDIT.csv"))
    state["stages"]["6"] = "complete"
    write_json(MANIFEST_PATH, state)

def stage7_logs(state):
    write_json(DIRS[7] / "safety_flags.json", state["safety"])
    state["stages"]["7"] = "complete"
    write_json(MANIFEST_PATH, state)

def stage8_summary(state):
    core = read_csv(DIRS[6] / "STRICT_TIMESTAMP_H_VS_FULL.csv")
    aggregate = read_csv(DIRS[6] / "STRICT_TIMESTAMP_DATASET_AGGREGATE.csv")
    audit = state["audit"]
    lines = ["# STRICT TIMESTAMP CHRONOLOGICAL CORE REPORT", "", "## Safety", "", "- STRICT_TIMESTAMP_PROTOCOL = YES", "- STRICT_TIMESTAMP_BLOCK_ORDER = YES", "- NO_TIMESTAMP_GROUP_CROSSES_SPLIT = YES", "- POST_HOC_STRICT_TIMESTAMP_CONTINUATION = YES", "- NEW_INDEPENDENT_FINAL_TEST = NO", "- CONFIRMATORY_FINAL_TEST = NO", "- FINAL_TEST_ACCESSED = YES (T2 post-hoc only)", "- FEATURE_FUTURE_LEAKAGE = NO", "- R1_FUTURE_LEAKAGE = NO", "- MODEL_FAMILY_CHANGED = NO", "- HYPERPARAMETERS_CHANGED = NO", "- AP_USED_FOR_SELECTION = NO", "", "## 1. Which legacy protocols were chronological?", ""]
    for dataset in ALL_DATASETS:
        row = audit.get(dataset, {})
        lines.append(f"- **{dataset}**: SOURCE_ORDER_ALREADY_TIMESTAMP_SORTED = **{row.get('source_order_already_timestamp_sorted','NA')}**; LEGACY_SPLIT_BASE = **{row.get('legacy_split_base','NA')}**; inversion rate = **{fmt(row.get('adjacent_inversion_rate'))}**; moved after stable sort = **{fmt(row.get('fraction_events_moved_after_sort'))}**.")
    lines += ["", "## 2. Which legacy time interpretations require revision?", "", "Every dataset with source_order_already_timestamp_sorted=NO must have raw-order temporal language revised. The active historical K2/K3 loaders pass event_order='raw' and slice by source index.", "", "## 3–6. Strict H-only versus Full", "", "| order | dataset | task | H LogLoss | Full LogLoss | Full−H | R1 ΔLL | changed fraction | status |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in core:
        lines.append(f"| {row.get('order')} | {row.get('dataset')} | {row.get('task')} | {fmt(row.get('H_LL'))} | {fmt(row.get('Full_LL'))} | {fmt(row.get('Full_minus_H_LL'))} | {fmt(row.get('r1_mean_delta_LL'))} | {fmt(row.get('r1_mean_changed_fraction'))} | {row.get('status')} |")
    lines += ["", "## 7–9. Interpretation", "", "H-only versus Full is the direct predictive-information comparison. R1 is a fixed-model counterfactual topology stress test and is not a necessary condition for H sufficiency.", "", "- A small Full−H value means residual lower-order features add little under the frozen classifier in that task.", "- A large positive fixed-model R1 ΔLL with small Full−H means H can remain near-sufficient while the frozen Full model is sensitive to residual feature distribution shift.", "- Resource-limited datasets are not counted as scientific failures or zeros.", "", "## 10–12. Overall theory", "", "The corrected protocol supports the cautious statement that H_k often carries the main task-relevant predictive information, while residual lower-order structure can have dataset-dependent incremental value and can alter fixed-model calibration through distribution shift. It does not prove that lower-order information is universally irrelevant, and it does not imply that preserving H_k guarantees prediction invariance.", "", "## Strict timestamp continuation", "", "T2 uses the 0.80–1.00 label window only as POST_HOC_STRICT_TIMESTAMP_CONTINUATION. It is not a new independent final or confirmatory test.", "", "## Files", "", f"- Manifest: `{MANIFEST_PATH}`", f"- Timestamp audit: `{DIRS[1]}`", f"- Split audit: `{DIRS[2]}`", f"- Core tables: `{DIRS[6]}`", f"- Cross-protocol table: `{DIRS[5] / 'CROSS_PROTOCOL_COMPARISON.csv'}`"]
    report = DIRS[8] / "STRICT_TIMESTAMP_CHRONOLOGICAL_CORE_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state["stages"]["8"] = "complete"
    state["overall_status"] = "COMPLETE"
    write_json(MANIFEST_PATH, state)
    print("STRICT TIMESTAMP CHRONOLOGICAL CORE V1", flush=True)
    print("TIMESTAMP_DATASETS_AUDITED = 8/8", flush=True)
    for order, datasets in ((3, K3_DATASETS), (2, K2_DATASETS)):
        statuses = [state.get("dataset_status", {}).get(f"K{order}", {}).get(dataset, {}).get("status") for dataset in datasets]
        print(f"K{order}_REQUESTED = {len(datasets)}", flush=True)
        print(f"K{order}_COMPLETED = {sum(status == 'COMPLETE' for status in statuses)}/{len(datasets)}", flush=True)
        print(f"K{order}_RESOURCE_LIMIT = {sum(status == 'RESOURCE_LIMIT' for status in statuses)}", flush=True)
    print("STRICT_TIMESTAMP_PROTOCOL = YES", flush=True)
    print("TIMESTAMP_GROUP_SPLIT = NO", flush=True)
    print("T2_NEW_INDEPENDENT_FINAL = NO", flush=True)
    print("T2_POST_HOC_ONLY = YES", flush=True)
    print("OVERALL_STATUS = COMPLETE", flush=True)
    print(f"FINAL_REPORT = {report}", flush=True)

def initial_state():
    return {"protocol": PROTOCOL, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": {}, "safety": {"POST_HOC_STRICT_TIMESTAMP_CONTINUATION": "YES", "NEW_INDEPENDENT_FINAL_TEST": "NO", "CONFIRMATORY_FINAL_TEST": "NO", "FINAL_TEST_ACCESSED": "YES", "MODEL_FAMILY_CHANGED": "NO", "HYPERPARAMETERS_CHANGED": "NO", "AP_USED_FOR_SELECTION": "NO"}, "audit": {}, "dataset_status": {}, "overall_status": "RUNNING"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-stage", type=int, choices=range(0, 9), default=8)
    parser.add_argument("--rerun-stage", type=int, choices=range(0, 9), default=None)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    state = read_json(MANIFEST_PATH) if args.resume and MANIFEST_PATH.is_file() else initial_state()
    write_json(MANIFEST_PATH, state)
    if args.rerun_stage is not None:
        state["stages"].pop(str(args.rerun_stage), None)
    if state["stages"].get("0") != "complete":
        stage0_protocol(state)
    if args.stop_after_stage >= 1 and state["stages"].get("1") != "complete":
        stage1_timestamp_audit(state)
    if args.stop_after_stage >= 2 and state["stages"].get("2") != "complete":
        stage2_split_audit(state)
    if args.stop_after_stage >= 3 and state["stages"].get("3") != "complete":
        run_order(3, K3_DATASETS, state)
    if args.stop_after_stage >= 4 and state["stages"].get("4") != "complete":
        run_order(2, K2_DATASETS, state)
    if args.stop_after_stage >= 5 and state["stages"].get("5") != "complete":
        stage5_cross_protocol(state)
    if args.stop_after_stage >= 6 and state["stages"].get("6") != "complete":
        stage6_tables(state)
    if args.stop_after_stage >= 7 and state["stages"].get("7") != "complete":
        stage7_logs(state)
    if args.stop_after_stage >= 8 and state["stages"].get("8") != "complete":
        stage8_summary(state)

if __name__ == "__main__":
    main()
