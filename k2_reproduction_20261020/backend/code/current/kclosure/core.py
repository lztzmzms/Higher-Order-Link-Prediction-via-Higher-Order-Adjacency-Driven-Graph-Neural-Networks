"""Dimension-generic temporal closure construction.

For target dimension k, a candidate is a (k+1)-vertex set whose complete
(k-1)-dimensional boundary is present at the cutoff while the k-simplex itself
is absent.  Enumeration is driven by observed boundary faces and never scans
``combinations(nodes, k + 1)``.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None


Simplex = Tuple[int, ...]


def _rss_bytes() -> int:
    if psutil is None:
        return 0
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _read_int_lines(path: Path) -> List[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def load_timestamped_simplices(
    nverts_path: Path,
    simplices_path: Path,
    times_path: Path,
    event_order: str = "timestamp",
):
    """Load raw simplex events and apply a stable chronological ordering."""
    nverts = _read_int_lines(nverts_path)
    timestamps = _read_int_lines(times_path)
    flat = _read_int_lines(simplices_path)
    if len(nverts) != len(timestamps):
        raise ValueError("nverts/times row count mismatch")
    if sum(nverts) != len(flat):
        raise ValueError("sum(nverts) does not match flattened simplex data")

    events = []
    offset = 0
    for row_id, (size, timestamp) in enumerate(zip(nverts, timestamps)):
        vertices = tuple(sorted(set(flat[offset : offset + size])))
        offset += size
        if not vertices:
            continue
        events.append({"vertices": vertices, "time": int(timestamp), "row_id": row_id})

    if event_order == "timestamp":
        events.sort(key=lambda event: (event["time"], event["row_id"]))
    elif event_order != "raw":
        raise ValueError(f"Unsupported event order: {event_order}")

    return events, {
        "raw_rows": len(nverts),
        "kept_simplices": len(events),
        "event_order": event_order,
        "timestamps_nondecreasing": all(
            events[i]["time"] <= events[i + 1]["time"] for i in range(len(events) - 1)
        ),
    }


def build_first_seen(events: Sequence[Mapping], max_size: int = 4):
    """First event/time for every face under the legacy downward closure rule."""
    first_event: Dict[int, Dict[Simplex, int]] = {
        size: {} for size in range(1, max_size + 1)
    }
    first_time: Dict[int, Dict[Simplex, int]] = {
        size: {} for size in range(1, max_size + 1)
    }
    for event_id, event in enumerate(events):
        vertices = event["vertices"]
        for size in range(1, min(max_size, len(vertices)) + 1):
            for face in combinations(vertices, size):
                if face not in first_event[size]:
                    first_event[size][face] = event_id
                    first_time[size][face] = int(event["time"])
    return first_event, first_time


def build_observed_state(events: Sequence[Mapping], max_dim: int = 3):
    """Construct a downward-closed simplicial state from context events."""
    by_size: Dict[int, Set[Simplex]] = {
        size: set() for size in range(1, max_dim + 2)
    }
    for event in events:
        vertices = event["vertices"]
        for size in range(1, min(max_dim + 1, len(vertices)) + 1):
            by_size[size].update(combinations(vertices, size))
    return {
        "simplices_by_size": by_size,
        "nodes": by_size[1],
        "edges": by_size.get(2, set()),
        "triangles": by_size.get(3, set()),
        "tetrahedra": by_size.get(4, set()),
    }


def simplex_boundaries(simplex: Sequence[int]) -> Tuple[Simplex, ...]:
    simplex = tuple(sorted(simplex))
    return tuple(combinations(simplex, len(simplex) - 1))


def enumerate_open_simplices(observed_state, simplex_dim: int):
    """Enumerate open k-simplices from their observed (k-1)-faces.

    The inverted index maps every (k-2)-face to the vertices completing it to
    an observed (k-1)-face.  Intersecting these completion sets for one boundary
    face yields exactly the possible final vertices.  This is deterministic and
    avoids scanning all node (k+1)-tuples.
    """
    if simplex_dim < 2:
        raise ValueError("simplex_dim must be at least 2")
    started = time.perf_counter()
    rss_before = _rss_bytes()
    boundary_size = simplex_dim
    target_size = simplex_dim + 1
    boundaries: Set[Simplex] = observed_state["simplices_by_size"][boundary_size]
    observed_targets: Set[Simplex] = observed_state["simplices_by_size"][target_size]

    ridge_to_completions: Dict[Simplex, Set[int]] = defaultdict(set)
    for boundary in boundaries:
        for ridge in combinations(boundary, boundary_size - 1):
            completion = next(iter(set(boundary) - set(ridge)))
            ridge_to_completions[ridge].add(completion)

    raw_candidates = 0
    candidate_set: Set[Simplex] = set()
    for boundary in sorted(boundaries):
        completion_sets = [
            ridge_to_completions[ridge]
            for ridge in combinations(boundary, boundary_size - 1)
        ]
        possible = set.intersection(*(set(values) for values in completion_sets))
        for vertex in sorted(possible - set(boundary)):
            candidate = tuple(sorted((*boundary, vertex)))
            # The intersection proves every boundary face exists; retain the
            # explicit assertion as a guard against future refactors.
            if not all(face in boundaries for face in simplex_boundaries(candidate)):
                raise AssertionError("incomplete boundary admitted as a candidate")
            if candidate not in observed_targets:
                raw_candidates += 1
                candidate_set.add(candidate)

    candidates = sorted(candidate_set)
    elapsed = time.perf_counter() - started
    rss_after = _rss_bytes()
    return candidates, {
        "candidate_generation_seconds": elapsed,
        "peak_CPU_RAM_bytes": max(rss_before, rss_after),
        "num_raw_candidates_before_dedup": raw_candidates,
        "num_unique_candidates": len(candidates),
        "enumeration_method": "boundary_face_inverted_index",
    }


def future_closed_simplices(future_events: Sequence[Mapping], simplex_dim: int) -> Set[Simplex]:
    target_size = simplex_dim + 1
    closed: Set[Simplex] = set()
    for event in future_events:
        if len(event["vertices"]) >= target_size:
            closed.update(combinations(event["vertices"], target_size))
    return closed


def build_derived_graph(candidates: Sequence[Simplex], simplex_dim: int):
    """Binary lower adjacency via shared complete boundary faces."""
    face_to_candidates: Dict[Simplex, List[int]] = defaultdict(list)
    for candidate_id, candidate in enumerate(candidates):
        for face in simplex_boundaries(candidate):
            face_to_candidates[face].append(candidate_id)

    edges: Set[Tuple[int, int]] = set()
    for ids in face_to_candidates.values():
        for left, right in combinations(ids, 2):
            edges.add((left, right) if left < right else (right, left))
    edge_pairs = np.asarray(sorted(edges), dtype=np.int64)
    if edge_pairs.size == 0:
        edge_pairs = np.empty((0, 2), dtype=np.int64)
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.vstack(
            (np.concatenate((edge_pairs[:, 0], edge_pairs[:, 1])),
             np.concatenate((edge_pairs[:, 1], edge_pairs[:, 0])))
        )
    return {
        "num_nodes": len(candidates),
        "edge_pairs": edge_pairs,
        "edge_index": edge_index,
        "face_to_candidates": face_to_candidates,
        "weighted_adjacency": False,
    }


def derived_graph_statistics(derived_graph) -> Dict[str, float]:
    n = int(derived_graph["num_nodes"])
    edges = derived_graph["edge_pairs"]
    adjacency = [set() for _ in range(n)]
    for left, right in edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    isolated = sum(not neighbors for neighbors in adjacency)
    seen = set()
    components = 0
    for source in range(n):
        if source in seen:
            continue
        components += 1
        queue = deque([source])
        seen.add(source)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return {
        "derived_nodes": n,
        "derived_edges": int(len(edges)),
        "connected_components": components,
        "isolated_nodes": isolated,
        "isolated_node_ratio": isolated / max(n, 1),
    }


def build_candidate_task(
    events: Sequence[Mapping],
    context_end: int,
    future_end: int,
    simplex_dim: int,
    name: str,
):
    context = events[:context_end]
    future = events[context_end:future_end]
    state = build_observed_state(context, max_dim=max(3, simplex_dim))
    candidates, generation = enumerate_open_simplices(state, simplex_dim)
    future_closed = future_closed_simplices(future, simplex_dim)
    labels = np.asarray([int(candidate in future_closed) for candidate in candidates], dtype=np.int64)
    derived = build_derived_graph(candidates, simplex_dim)
    face_counts = [len(ids) for ids in derived["face_to_candidates"].values()]
    cutoff = int(context[-1]["time"]) if context else None
    window_end = int(future[-1]["time"]) if future else cutoff
    return {
        "name": name,
        "simplex_dim": simplex_dim,
        "context_end": context_end,
        "future_end": future_end,
        "cutoff": cutoff,
        "prediction_window_end": window_end,
        "state": state,
        "candidates": candidates,
        "labels": labels,
        "derived_graph": derived,
        "generation": generation,
        "boundary_incidence": {
            "number_of_unique_boundaries": len(face_counts),
            "mean_simplices_per_boundary": float(np.mean(face_counts)) if face_counts else 0.0,
            "max_simplices_per_boundary": int(max(face_counts)) if face_counts else 0,
        },
    }


def census_row(task) -> Dict[str, float]:
    state = task["state"]
    labels = task["labels"]
    positives = int(labels.sum())
    candidates = len(labels)
    row = {
        "cutoff_fraction": task["context_end"],
        "cutoff_timestamp": task["cutoff"],
        "prediction_window_end": task["prediction_window_end"],
        "num_nodes": len(state["simplices_by_size"][1]),
        "num_edges": len(state["simplices_by_size"][2]),
        "num_2simplices": len(state["simplices_by_size"][3]),
        "num_3simplices": len(state["simplices_by_size"][4]),
        "num_open_3simplices": candidates,
        "num_future_positive_3simplices": positives,
        "num_future_negative_3simplices": candidates - positives,
        "random_baseline": positives / max(candidates, 1),
        **task["boundary_incidence"],
        **task["generation"],
    }
    row["number_of_unique_triangular_boundaries"] = row["number_of_unique_boundaries"]
    row["mean_tetrahedra_per_triangular_boundary"] = row["mean_simplices_per_boundary"]
    row["max_tetrahedra_per_triangular_boundary"] = row["max_simplices_per_boundary"]
    return row


def structural_features(task) -> Tuple[np.ndarray, List[str]]:
    """Natural k=2 -> k=3 feature generalization without priors or weighting."""
    candidates = task["candidates"]
    state = task["state"]
    derived = task["derived_graph"]
    boundary_size = task["simplex_dim"]
    lower_size = boundary_size - 1
    boundary_set = state["simplices_by_size"][boundary_size]

    lower_support = defaultdict(int)
    for boundary in boundary_set:
        for lower in combinations(boundary, lower_size):
            lower_support[lower] += 1
    node_degree = defaultdict(int)
    for u, v in state["simplices_by_size"][2]:
        node_degree[u] += 1
        node_degree[v] += 1
    dual_degree = np.zeros(len(candidates), dtype=np.float32)
    for left, right in derived["edge_pairs"]:
        dual_degree[int(left)] += 1
        dual_degree[int(right)] += 1

    rows = []
    for index, candidate in enumerate(candidates):
        boundaries = simplex_boundaries(candidate)
        boundary_degrees = []
        boundary_supports = []
        for boundary in boundaries:
            lowers = list(combinations(boundary, lower_size))
            boundary_degrees.append(sum(lower_support[face] for face in lowers))
            # Number of observed target simplices incident to this boundary.
            boundary_supports.append(
                sum(1 for target in state["simplices_by_size"][boundary_size + 1] if set(boundary) < set(target))
            )
        vertex_degrees = [node_degree[node] for node in candidate]
        values = []
        for group in (boundary_degrees, boundary_supports, vertex_degrees):
            values.extend((min(group), float(np.mean(group)), max(group), sum(group)))
        values.extend((dual_degree[index], math.log1p(float(dual_degree[index]))))
        rows.append([math.log1p(max(0.0, float(value))) for value in values])
    names = [
        "boundary_degree_min", "boundary_degree_mean", "boundary_degree_max", "boundary_degree_sum",
        "boundary_support_min", "boundary_support_mean", "boundary_support_max", "boundary_support_sum",
        "vertex_degree_min", "vertex_degree_mean", "vertex_degree_max", "vertex_degree_sum",
        "dual_degree", "log_dual_degree",
    ]
    return np.asarray(rows, dtype=np.float32), names
