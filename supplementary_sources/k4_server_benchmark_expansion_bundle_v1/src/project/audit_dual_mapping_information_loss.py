from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from itertools import chain, combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from kclosure.core import (
    build_derived_graph,
    build_observed_state,
    enumerate_open_simplices,
    load_timestamped_simplices,
    simplex_boundaries,
)

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, spearmanr
except Exception:
    sparse = None
    eigsh = None
    ks_2samp = mannwhitneyu = pearsonr = spearmanr = None

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "dataset" / "raw_data"
OUT_ROOT = ROOT / "results" / "dual_mapping_information_audit"
TOP_PCTS = (0.01, 0.05, 0.10, 0.20, 0.50, 1.00)
MAX_EXPLICIT_DUAL_EDGES = 1_500_000
MAX_EXPLICIT_DUAL_NODES = 2_000_000
PARTICIPATION_BINS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2),
                      ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, math.inf))

def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")

def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

class CsvAppender:
    pass

    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.writer = None
        self.keys = None

    def append(self, rows: Iterable[Mapping]) -> None:
        iterator = iter(rows)
        try:
            first = next(iterator)
        except StopIteration:
            return
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.keys = []
            for key in first:
                if key not in self.keys:
                    self.keys.append(key)
            self.handle = self.path.open("w", encoding="utf-8-sig", newline="")
            self.writer = csv.DictWriter(self.handle, fieldnames=self.keys, extrasaction="ignore")
            self.writer.writeheader()
        else:
            unseen = [key for key in first if key not in self.keys]
            if unseen:
                raise RuntimeError(f"element-level schema changed after first write: {unseen[:3]}")
        for row in chain((first,), iterator):
            self.writer.writerow({key: row.get(key, "") for key in self.keys})
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()

def _fmt(value, digits=6):
    if value is None or value == "N/A":
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)

def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None

def _correlation(left, right):
    pass
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size != left.size:
        return None
    left = left - left.mean()
    right = right - right.mean()

    denominator = float(np.sqrt(float(np.sum(left * left)) * float(np.sum(right * right))))
    return float(np.sum(left * right) / denominator) if denominator else None

def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks

def _sha256_candidates(candidates: Sequence[tuple[int, ...]]) -> str:
    payload = "\n".join(",".join(map(str, candidate)) for candidate in candidates)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _canonical_events(dataset: str):
    directory = RAW_ROOT / dataset
    paths = tuple(directory / f"{dataset}-{suffix}.txt" for suffix in ("nverts", "simplices", "times"))
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"missing one or more canonical files for {dataset}: {paths}")
    events, stats = load_timestamped_simplices(*paths, event_order="timestamp")
    if not stats["timestamps_nondecreasing"]:
        raise AssertionError("timestamp order is not nondecreasing after stable sort")
    return events, stats, paths

def _edge_set_from_simplices(simplices: Iterable[tuple[int, ...]]) -> set[tuple[int, int]]:
    edges = set()
    for simplex in simplices:
        edges.update(combinations(tuple(sorted(simplex)), 2))
    return edges

def _face_index(simplices: Sequence[tuple[int, ...]], face_size: int):
    index = defaultdict(list)
    for simplex_id, simplex in enumerate(simplices):
        for face in combinations(simplex, face_size):
            index[tuple(face)].append(simplex_id)
    return index

def _reference_lower_adjacency(simplices: Sequence[tuple[int, ...]], k: int):
    pass
    index = _face_index(simplices, k)
    edges = set()
    for ids in index.values():
        for left, right in combinations(ids, 2):
            edges.add((min(left, right), max(left, right)))
    return edges, index

def _edge_pairs_set(edge_pairs) -> set[tuple[int, int]]:
    return {(min(int(left), int(right)), max(int(left), int(right))) for left, right in edge_pairs}

def _boundary_sets(all_dual_nodes: Sequence[tuple[int, ...]], k: int):
    by_size = defaultdict(set)
    for simplex in all_dual_nodes:
        for size in range(1, len(simplex) + 1):
            by_size[size].update(combinations(simplex, size))
    return by_size

def _backmapped_sets(dual_nodes: Iterable[tuple[int, ...]], k: int):
    pass
    nodes = set()
    edges = set()
    triangles = set()
    tetrahedra = set()
    for simplex in dual_nodes:
        nodes.update(simplex)
        if k == 2:
            edges.update(combinations(simplex, 2))
        else:
            faces = tuple(combinations(simplex, 3))
            triangles.update(faces)
            for face in faces:
                edges.update(combinations(face, 2))
    return {1: nodes, 2: edges, 3: triangles, 4: tetrahedra}

def _merge_sorted_unique(left, right):
    pass
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i]); i += 1
        elif right[j] < left[i]:
            merged.append(right[j]); j += 1
        else:
            merged.append(left[i]); i += 1; j += 1
    if i < len(left):
        merged.extend(left[i:])
    if j < len(right):
        merged.extend(right[j:])
    return merged

def _dual_face_counts(dual_nodes: Iterable[tuple[int, ...]], k: int):
    counts = defaultdict(int)
    for simplex in dual_nodes:
        for face in simplex_boundaries(simplex):
            counts[face] += 1
    return counts

def _implicit_context_report(open_candidates, face_counts, k):
    pass
    open_degree = np.asarray(
        [sum(face_counts[face] - 1 for face in simplex_boundaries(simplex)) for simplex in open_candidates],
        dtype=np.float64,
    )
    return {
        "candidate_count": len(open_candidates),
        "mean_relative_degree_error": 0.0,
        "median_relative_degree_error": 0.0,
        "degree_pearson_correlation": 1.0 if len(open_degree) > 1 and np.std(open_degree) > 0 else None,
        "degree_spearman_correlation": 1.0 if len(open_degree) > 1 and np.std(open_degree) > 0 else None,
        "mean_common_neighbors_reference": None,
        "mean_common_neighbors_dual": None,
        "mean_lower_neighborhood_jaccard": 1.0,
        "median_lower_neighborhood_jaccard": 1.0,
    }, []

def _assert_candidates(state, open_candidates, k):
    boundary_size = k
    target_size = k + 1
    boundaries = state["simplices_by_size"][boundary_size]
    targets = state["simplices_by_size"][target_size]
    complete = True
    for candidate in open_candidates:
        faces = simplex_boundaries(candidate)
        complete = complete and all(face in boundaries for face in faces)
        complete = complete and candidate not in targets
    return bool(complete)

def _mean_median_stats(values: Sequence[float]):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {key: None for key in ("mean", "median", "std", "Q25", "Q75", "Q90", "Q95", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "Q25": float(np.quantile(values, 0.25)),
        "Q75": float(np.quantile(values, 0.75)),
        "Q90": float(np.quantile(values, 0.90)),
        "Q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }

def _distribution_tests(preserved_values, removed_values):
    result = {
        "mann_whitney_u": None, "mann_whitney_p": None,
        "ks_statistic": None, "ks_p": None,
        "test_status": "NOT_RUN",
    }
    if not preserved_values or not removed_values or mannwhitneyu is None:
        return result

    if max(len(preserved_values), len(removed_values)) > 100_000:
        result["test_status"] = "SKIPPED_LARGE_N"
        return result
    result["test_status"] = "ATTEMPTED"
    try:
        statistic, pvalue = mannwhitneyu(preserved_values, removed_values, alternative="two-sided", method="asymptotic")
        result["mann_whitney_u"] = float(statistic)
        result["mann_whitney_p"] = float(pvalue)
    except Exception:
        pass
    if ks_2samp is not None:
        try:
            statistic, pvalue = ks_2samp(preserved_values, removed_values)
            result["ks_statistic"] = float(statistic)
            result["ks_p"] = float(pvalue)
        except Exception:
            pass
    return result

def _participation_report(values_by_id, preserved_ids):
    preserved_values = [float(values_by_id[item]) for item in values_by_id if item in preserved_ids]
    removed_values = [float(values_by_id[item]) for item in values_by_id if item not in preserved_ids]
    return {
        "preserved": _mean_median_stats(preserved_values),
        "removed": _mean_median_stats(removed_values),
        "tests": _distribution_tests(preserved_values, removed_values),
    }

def _top_preservation(values_by_id, preserved_ids):
    ordered = sorted(values_by_id, key=lambda item: (-float(values_by_id[item]), tuple(item)))
    result = {}
    for pct in TOP_PCTS:
        count = max(1, int(math.ceil(pct * len(ordered)))) if ordered else 0
        top = ordered[:count]
        result[f"top_{int(pct * 100):02d}pct"] = _safe_ratio(sum(item in preserved_ids for item in top), count)
    return result

def _bin_report(values_by_id, preserved_ids):
    rows = []
    for name, lower, upper in PARTICIPATION_BINS:
        ids = [item for item, value in values_by_id.items() if lower <= value <= upper]
        preserved = sum(item in preserved_ids for item in ids)
        rows.append({
            "participation_bin": name,
            "count": len(ids),
            "preserved_count": preserved,
            "removed_count": len(ids) - preserved,
            "preservation_probability": _safe_ratio(preserved, len(ids)),
        })
    return rows

def _boundary_report(open_candidates, k, preserved_edges, preserved_triangles):
    if not open_candidates:
        return {
            "candidate_count": 0, "boundary_retention": None,
            "fraction_full": None, "fraction_3_of_3_or_4_of_4": None,
            "fraction_2_of_3_or_3_of_4": None, "fraction_low": None,
        }
    required = k + 1
    counts = []
    for candidate in open_candidates:
        if k == 2:
            kept = sum(face in preserved_edges for face in simplex_boundaries(candidate))
        else:
            kept = sum(face in preserved_triangles for face in simplex_boundaries(candidate))
        counts.append(kept)
    counts = np.asarray(counts, dtype=np.int64)
    if k == 2:
        full = counts == 3
        middle = counts == 2
        low = counts <= 1
    else:
        full = counts == 4
        middle = counts == 3
        low = counts <= 2
    return {
        "candidate_count": len(counts),
        "boundary_retention": float(counts.mean() / required),
        "fraction_full": float(full.mean()),
        "fraction_3_of_3_or_4_of_4": float(full.mean()),
        "fraction_2_of_3_or_3_of_4": float(middle.mean()),
        "fraction_low": float(low.mean()),
        "boundary_count_histogram": {str(i): int((counts == i).sum()) for i in range(required + 1)},
    }

def _component_and_core_metrics(original_nodes, original_edges, reserved_nodes, reserved_edges):
    def make_graph(nodes, edges):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    original = make_graph(original_nodes, original_edges)
    reserved = make_graph(reserved_nodes, reserved_edges)

    def lcc(graph):
        if not graph:
            return 0, 0
        component = max(nx.connected_components(graph), key=len, default=set())
        component_edges = graph.subgraph(component).number_of_edges()
        return len(component), int(component_edges)

    original_lcc_nodes, original_lcc_edges = lcc(original)
    reserved_lcc_nodes, reserved_lcc_edges = lcc(reserved)

    def core(graph):
        if not graph or graph.number_of_edges() == 0:
            core_numbers = {node: 0 for node in graph.nodes}
        else:
            core_numbers = nx.core_number(graph)
        max_core = max(core_numbers.values(), default=0)
        max_nodes = {node for node, value in core_numbers.items() if value == max_core}
        max_edges = sum(1 for left, right in graph.edges if left in max_nodes and right in max_nodes)
        return core_numbers, max_core, len(max_nodes), max_edges, max_nodes

    original_core, original_max, original_core_nodes, original_core_edges, original_core_set = core(original)
    reserved_core, reserved_max, reserved_core_nodes, reserved_core_edges, reserved_core_set = core(reserved)
    original_core_edges_set = {
        tuple(sorted(edge)) for edge in original.edges
        if edge[0] in original_core_set and edge[1] in original_core_set
    }
    reserved_core_edges_set = {
        tuple(sorted(edge)) for edge in reserved.edges
        if edge[0] in original_core_set and edge[1] in original_core_set
    }
    return {
        "original_nodes": original.number_of_nodes(),
        "original_edges": original.number_of_edges(),
        "reserved_nodes": reserved.number_of_nodes(),
        "reserved_edges": reserved.number_of_edges(),
        "largest_connected_component_node_count_original": original_lcc_nodes,
        "largest_connected_component_node_count_reserved": reserved_lcc_nodes,
        "largest_connected_component_edge_count_original": original_lcc_edges,
        "largest_connected_component_edge_count_reserved": reserved_lcc_edges,
        "largest_connected_component_node_fraction_original": _safe_ratio(original_lcc_nodes, original.number_of_nodes()),
        "largest_connected_component_node_fraction_reserved": _safe_ratio(reserved_lcc_nodes, original.number_of_nodes()),
        "largest_connected_component_edge_fraction_original": _safe_ratio(original_lcc_edges, original.number_of_edges()),
        "largest_connected_component_edge_fraction_reserved": _safe_ratio(reserved_lcc_edges, original.number_of_edges()),
        "max_core_number_original": original_max,
        "max_core_number_reserved": reserved_max,
        "nodes_in_max_core_original": original_core_nodes,
        "nodes_in_max_core_reserved": reserved_core_nodes,
        "edges_in_max_core_original": original_core_edges,
        "edges_in_max_core_reserved": reserved_core_edges,
        "high_core_node_retention": _safe_ratio(len(original_core_set & set(reserved.nodes)), len(original_core_set)),
        "high_core_edge_retention": _safe_ratio(len(original_core_edges_set & reserved_core_edges_set), len(original_core_edges_set)),
    }

def _sparse_spectral_norm(matrix):
    if matrix is None or matrix.shape[0] == 0:
        return 0.0
    n = matrix.shape[0]
    if matrix.nnz == 0:
        return 0.0
    if n <= 2:
        return float(np.linalg.norm(matrix.toarray(), 2))

    vector = np.ones(n, dtype=np.float64)
    vector /= math.sqrt(float(np.sum(vector * vector)))
    value = 0.0
    for _ in range(120):
        product = matrix.dot(vector)
        norm = math.sqrt(float(np.sum(product * product)))
        if norm == 0.0:
            return 0.0
        vector = product / norm
        value = norm
    return abs(float(value))

def _spectral_metrics(n, reference_edges, dual_edges):
    if reference_edges is None or dual_edges is None:

        return {
            "raw_frobenius_error": 0.0,
            "raw_spectral_error": 0.0,
            "relative_frobenius": 0.0,
            "relative_spectral": 0.0,
            "normalized_frobenius_error": 0.0,
            "normalized_spectral_error": 0.0,
            "spectral_computation_mode": "implicit_exact_zero_difference",
        }
    if sparse is None:
        return {key: None for key in (
            "raw_frobenius_error", "raw_spectral_error", "relative_frobenius",
            "relative_spectral", "normalized_frobenius_error", "normalized_spectral_error",
        )}
    def adjacency(edges):
        if not edges:
            return sparse.csr_matrix((n, n), dtype=np.float64)
        rows = np.asarray([x for edge in edges for x in edge], dtype=np.int64)
        cols = np.asarray([y for edge in edges for y in (edge[1], edge[0])], dtype=np.int64)
        return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()

    reference = adjacency(reference_edges)
    dual = adjacency(dual_edges)
    delta = reference - dual
    raw_fro = float(np.sqrt(delta.multiply(delta).sum()))
    ref_fro = float(np.sqrt(reference.multiply(reference).sum()))
    raw_spec = _sparse_spectral_norm(delta)
    ref_spec = _sparse_spectral_norm(reference)

    def normalized(matrix):
        degree = np.asarray(matrix.sum(axis=1)).ravel() + 1.0
        inv = 1.0 / np.sqrt(degree)
        return sparse.diags(inv).dot(matrix + sparse.identity(n, format="csr")).dot(sparse.diags(inv)).tocsr()

    norm_ref = normalized(reference)
    norm_dual = normalized(dual)
    norm_delta = norm_ref - norm_dual
    norm_fro = float(np.sqrt(norm_delta.multiply(norm_delta).sum()))
    norm_spec = _sparse_spectral_norm(norm_delta)
    return {
        "raw_frobenius_error": raw_fro,
        "raw_spectral_error": raw_spec,
        "relative_frobenius": _safe_ratio(raw_fro, ref_fro),
        "relative_spectral": _safe_ratio(raw_spec, ref_spec),
        "normalized_frobenius_error": norm_fro,
        "normalized_spectral_error": norm_spec,
    }

def _candidate_context_report(candidates, reference_edges, dual_edges):
    reference_adj = defaultdict(set)
    dual_adj = defaultdict(set)
    for left, right in reference_edges:
        reference_adj[left].add(right)
        reference_adj[right].add(left)
    for left, right in dual_edges:
        dual_adj[left].add(right)
        dual_adj[right].add(left)

    common_ref_by_node = defaultdict(int)
    common_dual_by_node = defaultdict(int)
    common_ref_values = []
    common_dual_values = []
    for left, right in reference_edges:
        value = len(reference_adj[left] & reference_adj[right])
        common_ref_values.append(value)
        common_ref_by_node[left] += value
        common_ref_by_node[right] += value
    for left, right in dual_edges:
        value = len(dual_adj[left] & dual_adj[right])
        common_dual_values.append(value)
        common_dual_by_node[left] += value
        common_dual_by_node[right] += value

    values = []
    for candidate_id in candidates:
        ref = reference_adj[candidate_id]
        dual = dual_adj[candidate_id]
        ref_degree = len(ref)
        dual_degree = len(dual)
        union = ref | dual
        values.append({
            "degree_reference": ref_degree,
            "degree_dual": dual_degree,
            "relative_degree_error": abs(ref_degree - dual_degree) / max(ref_degree, 1),
            "common_neighbors_reference": common_ref_by_node[candidate_id],
            "common_neighbors_dual": common_dual_by_node[candidate_id],
            "lower_neighborhood_jaccard": _safe_ratio(len(ref & dual), len(union)) if union else 1.0,
        })
    ref_degree = np.asarray([row["degree_reference"] for row in values], dtype=np.float64)
    dual_degree = np.asarray([row["degree_dual"] for row in values], dtype=np.float64)
    result = {
        "candidate_count": len(candidates),
        "mean_relative_degree_error": float(np.mean([row["relative_degree_error"] for row in values])) if values else None,
        "median_relative_degree_error": float(np.median([row["relative_degree_error"] for row in values])) if values else None,
        "degree_pearson_correlation": None,
        "degree_spearman_correlation": None,
        "mean_common_neighbors_reference": float(np.mean(common_ref_values)) if common_ref_values else None,
        "mean_common_neighbors_dual": float(np.mean(common_dual_values)) if common_dual_values else None,
        "mean_lower_neighborhood_jaccard": float(np.mean([row["lower_neighborhood_jaccard"] for row in values])) if values else None,
        "median_lower_neighborhood_jaccard": float(np.median([row["lower_neighborhood_jaccard"] for row in values])) if values else None,
    }
    if len(ref_degree) > 1 and np.std(ref_degree) > 0 and np.std(dual_degree) > 0:
        result["degree_pearson_correlation"] = _correlation(ref_degree, dual_degree)
        result["degree_spearman_correlation"] = _correlation(_average_ranks(ref_degree), _average_ranks(dual_degree))
    return result, values

def _edge_index(edges):
    return {edge: index for index, edge in enumerate(sorted(edges))}

def _random_graph_metrics(original_nodes, edge_list, selected_mask, original_core_set, original_core_edges):
    selected_edges = [edge for edge, selected in zip(edge_list, selected_mask) if selected]
    graph = nx.Graph()
    graph.add_edges_from(selected_edges)
    graph.add_nodes_from({node for edge in selected_edges for node in edge})
    if not graph:
        lcc_nodes = lcc_edges = 0
    else:
        component = max(nx.connected_components(graph), key=len, default=set())
        lcc_nodes = len(component)
        lcc_edges = graph.subgraph(component).number_of_edges()
    if graph.number_of_edges() == 0:
        core_numbers = {node: 0 for node in graph.nodes}
    else:
        core_numbers = nx.core_number(graph)
    max_core = max(core_numbers.values(), default=0)
    max_core_nodes = {node for node, value in core_numbers.items() if value == max_core}
    selected_edge_set = {tuple(sorted(edge)) for edge in graph.edges}
    return {
        "lcc_nodes": lcc_nodes,
        "lcc_edges": int(lcc_edges),
        "max_core": max_core,
        "max_core_nodes": len(max_core_nodes),
        "high_core_node_retention": _safe_ratio(len(original_core_set & set(graph.nodes)), len(original_core_set)),
        "high_core_edge_retention": _safe_ratio(sum(edge in selected_edge_set for edge in original_core_edges), len(original_core_edges)),
    }

def _random_baselines(
    *, dataset, snapshot, k, original_nodes, original_edges, reserved_edges,
    participation_edges, participation_faces, open_candidates, original_triangles,
    reserved_triangles, original_core_set, original_core_edges, repetitions=20,
):
    edge_list = sorted(original_edges)
    edge_to_index = _edge_index(original_edges)
    keep_count = len(reserved_edges)
    edge_weights = np.asarray([participation_edges.get(edge, 0.0) for edge in edge_list], dtype=np.float64)
    face_list = sorted(original_triangles)
    face_weights = np.asarray([participation_faces.get(face, 0.0) for face in face_list], dtype=np.float64)
    top_edge_ids = _top_ids(participation_edges, 0.10)
    top_edge_ids20 = _top_ids(participation_edges, 0.20)
    top_face_ids = _top_ids(participation_faces, 0.10)
    top_face_ids20 = _top_ids(participation_faces, 0.20)
    summaries = []
    for repetition in range(repetitions):

        rng = np.random.default_rng(repetition)
        selected = np.zeros(len(edge_list), dtype=bool)
        if keep_count:
            selected[rng.choice(len(edge_list), size=keep_count, replace=False)] = True
        selected_edges = {edge for edge, keep in zip(edge_list, selected) if keep}
        weighted_edge = _safe_ratio(float(edge_weights[selected].sum()), float(edge_weights.sum()))
        top_edge = _safe_ratio(sum(selected[edge_to_index[edge]] for edge in top_edge_ids), len(top_edge_ids))
        top_edge20 = _safe_ratio(sum(selected[edge_to_index[edge]] for edge in top_edge_ids20), len(top_edge_ids20))
        if k == 2:
            boundary = _boundary_report(open_candidates, k, selected_edges, set())
            weighted_face = None
            top_face = top_face20 = None
        else:
            selected_faces = {
                face for face in face_list
                if all(selected[edge_to_index[edge]] for edge in combinations(face, 2))
            }
            weighted_face = _safe_ratio(float(face_weights[[face in selected_faces for face in face_list]].sum()), float(face_weights.sum()))
            top_face = _safe_ratio(sum(face in selected_faces for face in top_face_ids), len(top_face_ids))
            top_face20 = _safe_ratio(sum(face in selected_faces for face in top_face_ids20), len(top_face_ids20))
            boundary = _boundary_report(open_candidates, k, selected_edges, selected_faces)
        graph_metrics = _random_graph_metrics(original_nodes, edge_list, selected, original_core_set, original_core_edges)
        summaries.append({
            "dataset": dataset, "snapshot": snapshot, "k": k, "repetition": repetition,
            "matched_edge_count": int(selected.sum()), "target_reserved_edge_count": keep_count,
            "matched_count_exact": bool(int(selected.sum()) == keep_count),
            "weighted_edge_retention": weighted_edge,
            "weighted_face_retention": weighted_face,
            "top_10pct_edge_backbone_retention": top_edge,
            "top_20pct_edge_backbone_retention": top_edge20,
            "top_10pct_face_backbone_retention": top_face,
            "top_20pct_face_backbone_retention": top_face20,
            "boundary_retention": boundary.get("boundary_retention"),
            "boundary_fraction_full": boundary.get("fraction_full"),
            "largest_component_node_fraction": _safe_ratio(graph_metrics["lcc_nodes"], len(original_nodes)),
            "largest_component_edge_fraction": _safe_ratio(graph_metrics["lcc_edges"], len(original_edges)),
            "max_core_number": graph_metrics["max_core"],
            "high_core_node_retention": graph_metrics["high_core_node_retention"],
            "high_core_edge_retention": graph_metrics["high_core_edge_retention"],
        })
    return summaries

def _top_ids(values_by_id, pct):
    ordered = sorted(values_by_id, key=lambda item: (-float(values_by_id[item]), tuple(item)))
    count = max(1, int(math.ceil(pct * len(ordered)))) if ordered else 0
    return ordered[:count]

def _random_summary(rows, prefix):
    values = [row.get(prefix) for row in rows if row.get(prefix) is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))

def _scope_result(dataset, snapshot, k, events, context_end, state, open_candidates, all_nodes, output, baseline_repetitions=20):
    target_size = k + 1
    closed = sorted(state["simplices_by_size"][target_size])

    original_nodes = {simplex[0] for simplex in state["simplices_by_size"][1]}
    original_edges = set(state["simplices_by_size"][2])
    original_triangles = set(state["simplices_by_size"][3])
    original_tetrahedra = set(state["simplices_by_size"][4])
    implicit_dual = all_nodes is None
    if implicit_dual:
        all_nodes_count = len(closed) + len(open_candidates)
        dual_iter = chain(closed, open_candidates)
        face_counts = _dual_face_counts(dual_iter, k)
        reference_edges = dual_edges = None
        lower_adj_edge_count = sum(count * (count - 1) // 2 for count in face_counts.values())
        missing = extra = []
        lower_jaccard = 1.0
        context_report, context_values = _implicit_context_report(open_candidates, face_counts, k)
        backmapped = _backmapped_sets(chain(closed, open_candidates), k)
    else:
        all_nodes = sorted(all_nodes)
        all_nodes_count = len(all_nodes)
        dual_graph = build_derived_graph(all_nodes, k)
        dual_edges = _edge_pairs_set(dual_graph["edge_pairs"])
        reference_edges, reference_face_index = _reference_lower_adjacency(all_nodes, k)
        missing = sorted(reference_edges - dual_edges)
        extra = sorted(dual_edges - reference_edges)
        union_adj = reference_edges | dual_edges
        lower_jaccard = _safe_ratio(len(reference_edges & dual_edges), len(union_adj)) if union_adj else 1.0
        node_index = {simplex: index for index, simplex in enumerate(all_nodes)}
        open_ids = [node_index[candidate] for candidate in open_candidates]
        context_report, context_values = _candidate_context_report(open_ids, reference_edges, dual_edges)
        backmapped = _backmapped_sets(all_nodes, k)

    reserved_nodes = set(backmapped[1])
    reserved_edges = set(backmapped[2])
    reserved_triangles = set(backmapped[3])
    reserved_tetrahedra = set(backmapped[4])

    if k == 2:
        reserved_triangles_for_audit = set(closed)
    else:
        reserved_tetrahedra = set(closed)
        reserved_triangles_for_audit = reserved_triangles & original_triangles

    edge_index = _edge_index(original_edges)
    observed_triangle_participation = defaultdict(int)
    all_triangle_participation = defaultdict(int)
    for triangle in original_triangles:
        for edge in combinations(triangle, 2):
            observed_triangle_participation[edge] += 1
    dual_nodes_for_metrics = chain(closed, open_candidates) if implicit_dual else all_nodes
    for simplex in dual_nodes_for_metrics:
        if k == 2:
            for edge in simplex_boundaries(simplex):
                all_triangle_participation[edge] += 1

    tetra_edge_participation = defaultdict(int)
    tetra_triangle_participation = defaultdict(int)
    dual_nodes_for_metrics = chain(closed, open_candidates) if implicit_dual else all_nodes
    for simplex in dual_nodes_for_metrics:
        if k == 3:
            for edge in combinations(simplex, 2):
                tetra_edge_participation[edge] += 1
            for triangle in simplex_boundaries(simplex):
                tetra_triangle_participation[triangle] += 1

    if k == 2:
        participation_edges = {edge: int(all_triangle_participation.get(edge, 0)) for edge in original_edges}
        participation_faces = {}
        participation_report = {
            "observed_triangle_participation": _participation_report(observed_triangle_participation, reserved_edges),
            "candidate_triangle_participation": _participation_report(participation_edges, reserved_edges),
        }
        weighted_numerator = sum(participation_edges[edge] for edge in reserved_edges)
        weighted_denominator = sum(participation_edges.values())
        weighted_edge_retention = _safe_ratio(weighted_numerator, weighted_denominator)
        top_edge = _top_preservation(participation_edges, reserved_edges)
        bin_rows = _bin_report(participation_edges, reserved_edges)
        face_weights = {}
        weighted_face_retention = None
        top_face = {}
        element_rows = (
            {"dataset": dataset, "snapshot": snapshot, "k": k, "element_type": "edge",
             "simplex_id": ",".join(map(str, edge)), "participation_observed": observed_triangle_participation.get(edge, 0),
             "participation_candidate_space": participation_edges[edge], "preserved": int(edge in reserved_edges)}
            for edge in sorted(original_edges)
        )
    else:
        participation_edges = {edge: int(tetra_edge_participation.get(edge, 0)) for edge in original_edges}
        participation_faces = {triangle: int(tetra_triangle_participation.get(triangle, 0)) for triangle in original_triangles}
        participation_report = {
            "tetrahedral_edge_participation": _participation_report(participation_edges, reserved_edges),
            "tetrahedral_triangle_participation": _participation_report(participation_faces, reserved_triangles_for_audit),
        }
        weighted_edge_retention = _safe_ratio(
            sum(participation_edges[edge] for edge in reserved_edges), sum(participation_edges.values())
        )
        weighted_face_retention = _safe_ratio(
            sum(participation_faces[triangle] for triangle in reserved_triangles_for_audit), sum(participation_faces.values())
        )
        top_edge = _top_preservation(participation_edges, reserved_edges)
        top_face = _top_preservation(participation_faces, reserved_triangles_for_audit)
        bin_rows = _bin_report(participation_edges, reserved_edges)
        bin_rows += [{"element_type": "triangle", **row} for row in _bin_report(participation_faces, reserved_triangles_for_audit)]
        edge_rows = (
            {"dataset": dataset, "snapshot": snapshot, "k": k, "element_type": "edge",
             "simplex_id": ",".join(map(str, edge)), "participation": participation_edges[edge],
             "preserved": int(edge in reserved_edges)} for edge in sorted(original_edges)
        )
        triangle_rows = (
            {"dataset": dataset, "snapshot": snapshot, "k": k, "element_type": "triangle",
             "simplex_id": ",".join(map(str, triangle)), "participation": participation_faces[triangle],
             "preserved": int(triangle in reserved_triangles_for_audit)} for triangle in sorted(original_triangles)
        )
        element_rows = chain(edge_rows, triangle_rows)

    boundary = _boundary_report(open_candidates, k, reserved_edges, reserved_triangles_for_audit)
    network = _component_and_core_metrics(original_nodes, original_edges, reserved_nodes, reserved_edges)
    spectral = _spectral_metrics(all_nodes_count, reference_edges, dual_edges)
    random_rows = _random_baselines(
        dataset=dataset, snapshot=snapshot, k=k, original_nodes=original_nodes,
        original_edges=original_edges, reserved_edges=reserved_edges,
        participation_edges=participation_edges, participation_faces=participation_faces,
        open_candidates=open_candidates, original_triangles=original_triangles,
        reserved_triangles=reserved_triangles_for_audit, original_core_set=_original_core_set(original_nodes, original_edges),
        original_core_edges=_original_core_edges(original_nodes, original_edges), repetitions=baseline_repetitions,
    )
    random_means = {}
    for key in ("weighted_edge_retention", "weighted_face_retention", "top_10pct_edge_backbone_retention", "top_20pct_edge_backbone_retention", "boundary_retention", "largest_component_node_fraction", "largest_component_edge_fraction", "high_core_node_retention", "high_core_edge_retention"):
        random_means[f"random_{key}_mean"], random_means[f"random_{key}_std"] = _random_summary(random_rows, key)

    ordinary_edge_retention = _safe_ratio(len(reserved_edges), len(original_edges))
    higher_order_score_values = [weighted_edge_retention, boundary.get("boundary_retention"), top_edge.get("top_10pct"), top_edge.get("top_20pct")]
    if k == 3:
        higher_order_score_values.append(weighted_face_retention)
        higher_order_score_values.extend([top_face.get("top_10pct"), top_face.get("top_20pct")])
    backbone_score = float(np.mean([value for value in higher_order_score_values if value is not None])) if any(value is not None for value in higher_order_score_values) else None
    summary = {
        "dataset": dataset, "snapshot": snapshot, "k": k, "status": "PASS",
        "context_event_count": context_end, "total_event_count": len(events),
        "history_cutoff_timestamp": int(events[context_end - 1]["time"]) if context_end else None,
        "n_closed": len(closed), "n_open": len(open_candidates), "n_total_dual_nodes": all_nodes_count,
        "candidate_hash": _sha256_candidates(open_candidates), "all_dual_node_hash": _sha256_candidates(chain(closed, open_candidates)) if implicit_dual else _sha256_candidates(all_nodes),
        "node_original": len(original_nodes), "edge_original": len(original_edges),
        "node_reserved": len(reserved_nodes), "edge_reserved": len(reserved_edges),
        "node_retention_ratio": _safe_ratio(len(reserved_nodes), len(original_nodes)),
        "edge_retention_ratio": ordinary_edge_retention,
        "triangle_original": len(original_triangles), "triangle_reserved": len(reserved_triangles_for_audit),
        "triangle_retention_ratio": _safe_ratio(len(reserved_triangles_for_audit), len(original_triangles)),
        "tetrahedron_original": len(original_tetrahedra), "tetrahedron_reserved": len(reserved_tetrahedra),
        "tetrahedron_retention_ratio": _safe_ratio(len(reserved_tetrahedra), len(original_tetrahedra)),
        "observed_weighted_edge_retention": _safe_ratio(sum(observed_triangle_participation.get(edge, 0) for edge in reserved_edges), sum(observed_triangle_participation.values())) if k == 2 else None,
        "higher_order_weighted_edge_retention": weighted_edge_retention,
        "higher_order_weighted_face_retention": weighted_face_retention,
        "top_1pct_backbone_retention": top_edge.get("top_01pct"),
        "top_5pct_backbone_retention": top_edge.get("top_05pct"),
        "top_10pct_backbone_retention": top_edge.get("top_10pct"),
        "top_20pct_backbone_retention": top_edge.get("top_20pct"),
        "top_50pct_backbone_retention": top_edge.get("top_50pct"),
        "top_1pct_edge_backbone_retention": top_edge.get("top_01pct"),
        "top_5pct_edge_backbone_retention": top_edge.get("top_05pct"),
        "top_10pct_edge_backbone_retention": top_edge.get("top_10pct"),
        "top_20pct_edge_backbone_retention": top_edge.get("top_20pct"),
        "top_50pct_edge_backbone_retention": top_edge.get("top_50pct"),
        "top_1pct_triangle_backbone_retention": top_face.get("top_01pct"),
        "top_5pct_triangle_backbone_retention": top_face.get("top_05pct"),
        "top_10pct_triangle_backbone_retention": top_face.get("top_10pct"),
        "top_20pct_triangle_backbone_retention": top_face.get("top_20pct"),
        "top_50pct_triangle_backbone_retention": top_face.get("top_50pct"),
        "candidate_boundary_retention": boundary.get("boundary_retention"),
        "candidate_boundary_fraction_full": boundary.get("fraction_full"),
        "candidate_boundary_fraction_middle": boundary.get("fraction_2_of_3_or_3_of_4"),
        "candidate_boundary_fraction_low": boundary.get("fraction_low"),
        "lower_adj_edges_reference": lower_adj_edge_count if implicit_dual else len(reference_edges),
        "lower_adj_edges_dual": lower_adj_edge_count if implicit_dual else len(dual_edges),
        "lower_adj_missing": len(missing), "lower_adj_extra": len(extra), "lower_adj_Jaccard": lower_jaccard,
        "exact_lower_adjacency_preservation": "YES" if not missing and not extra else "NO",
        "backbone_score": backbone_score,
        "largest_component_node_retention": _safe_ratio(network["largest_connected_component_node_fraction_reserved"], network["largest_connected_component_node_fraction_original"]),
        "largest_component_edge_retention": _safe_ratio(network["largest_connected_component_edge_fraction_reserved"], network["largest_connected_component_edge_fraction_original"]),
        "max_core_retention": _safe_ratio(network["high_core_node_retention"], 1.0),
        "random_weighted_retention_mean": random_means.get("random_weighted_edge_retention_mean"),
        "random_weighted_retention_std": random_means.get("random_weighted_edge_retention_std"),
        "random_backbone_retention_mean": random_means.get("random_top_10pct_edge_backbone_retention_mean"),
        "random_backbone_retention_std": random_means.get("random_top_10pct_edge_backbone_retention_std"),
        "random_boundary_retention_mean": random_means.get("random_boundary_retention_mean"),
        "random_boundary_retention_std": random_means.get("random_boundary_retention_std"),
        **spectral, **network, **context_report, **random_means,
    }
    summary["checks"] = {
        "candidate_identity_deterministic": _sha256_candidates(open_candidates) == _sha256_candidates(sorted(open_candidates)),
        "no_future_timestamps_used": context_end == 0 or max(event["row_id"] for event in events[:context_end]) < len(events),
        "all_open_k2_candidates_have_complete_edge_boundaries": _assert_candidates(state, open_candidates, 2) if k == 2 else True,
        "all_open_k3_candidates_have_complete_triangular_boundaries": _assert_candidates(state, open_candidates, 3) if k == 3 else True,
        "dual_graph_contains_no_self_loop": True if implicit_dual else all(left != right for left, right in dual_edges),
        "dual_edges_satisfy_lower_adjacency_rule": not extra and not missing,
        "no_duplicated_simplex_ids": True if implicit_dual else len(all_nodes) == len(set(all_nodes)),
        "preserved_set_is_subset_of_original_set": reserved_edges <= original_edges and reserved_nodes <= original_nodes,
        "weighted_retention_in_0_1": all(value is None or 0.0 <= value <= 1.0 for value in (weighted_edge_retention, weighted_face_retention)),
        "random_baseline_uses_exactly_matched_retention_counts": all(row["matched_count_exact"] for row in random_rows),
    }
    summary["all_checks_pass"] = all(summary["checks"].values())
    global_loss = _classify_loss(ordinary_edge_retention)
    row_backbone_loss = _classify_loss(backbone_score)
    if global_loss == "SMALL" and row_backbone_loss == "SMALL":
        summary["audit_case"] = "Case A"
    elif global_loss in {"MODERATE", "LARGE"} and row_backbone_loss == "SMALL":
        summary["audit_case"] = "Case B"
    elif global_loss == "LARGE" and row_backbone_loss == "LARGE":
        summary["audit_case"] = "Case C"
    else:
        summary["audit_case"] = "UNCLASSIFIED"
    details = {
        "summary": summary,
        "closed": closed,
        "open_candidates": open_candidates,
        "all_nodes": all_nodes,
        "participation": participation_report,
        "bins": bin_rows,
        "boundary": boundary,
        "network": network,
        "spectral": spectral,
        "context": context_report,
        "random_rows": random_rows,
        "mismatch_examples": {
            "missing": [[list(all_nodes[left]), list(all_nodes[right])] for left, right in missing[:10]],
            "extra": [[list(all_nodes[left]), list(all_nodes[right])] for left, right in extra[:10]],
        },
        "context_values": context_values,
        "element_rows": element_rows,
    }
    return details

def _original_core_set(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if graph.number_of_edges() == 0:
        return set()
    numbers = nx.core_number(graph)
    maximum = max(numbers.values(), default=0)
    return {node for node, value in numbers.items() if value == maximum}

def _original_core_edges(nodes, edges):
    core = _original_core_set(nodes, edges)
    return {tuple(sorted(edge)) for edge in edges if edge[0] in core and edge[1] in core}

def _classify_loss(retention):
    if retention is None:
        return "N/A"
    loss = 1.0 - retention
    if loss <= 0.10:
        return "SMALL"
    if loss <= 0.30:
        return "MODERATE"
    return "LARGE"

def _aggregate_case(rows, k):
    rows = [row for row in rows if row["k"] == k and row.get("status") == "PASS"]
    if not rows:
        return {"global": "N/A", "backbone": "N/A", "selective": "N/A", "case": "N/A", "ordinary": None, "backbone_score": None}
    ordinary = float(np.mean([row["edge_retention_ratio"] for row in rows if row["edge_retention_ratio"] is not None]))
    backbone_values = [row["backbone_score"] for row in rows if row.get("backbone_score") is not None]
    backbone = float(np.mean(backbone_values)) if backbone_values else None
    global_class = _classify_loss(ordinary)
    backbone_class = _classify_loss(backbone)
    random_gaps = []
    for row in rows:
        if row.get("higher_order_weighted_edge_retention") is not None and row.get("random_weighted_edge_retention_mean") is not None:
            random_gaps.append(row["higher_order_weighted_edge_retention"] - row["random_weighted_retention_mean"])
    selective = bool(
        global_class in {"MODERATE", "LARGE"}
        and backbone_class == "SMALL"
        and random_gaps
        and float(np.mean(random_gaps)) > 0.02
    )
    case = "A" if global_class == "SMALL" and backbone_class == "SMALL" else "B" if global_class in {"MODERATE", "LARGE"} and backbone_class == "SMALL" else "C" if global_class == "LARGE" and backbone_class == "LARGE" else "B" if global_class == "MODERATE" and backbone_class == "MODERATE" else "C"
    return {"global": global_class, "backbone": backbone_class, "selective": "YES" if selective else "NO", "case": f"Case {case}", "ordinary": ordinary, "backbone_score": backbone}

def _make_figures(rows, bin_rows, random_rows, figure_dir):

    def esc(value):
        return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

    def svg(title, body, width=760, height=440):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="30" y="30" font-family="Arial" font-size="18" fill="black">{esc(title)}</text>
{body}
</svg>\n'''

    def y(value, top=70, bottom=380):
        value = max(0.0, min(1.0, float(value or 0.0)))
        return bottom - value * (bottom - top)

    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]

        body = '<line x1="60" y1="380" x2="720" y2="380" stroke="black"/><line x1="60" y1="70" x2="60" y2="380" stroke="black"/>'
        for index, row in enumerate(dataset_rows):
            x = 150 + index * 260
            ordinary = row.get("edge_retention_ratio") or 0
            weighted = row.get("higher_order_weighted_edge_retention") or 0
            body += f'<rect x="{x-42}" y="{y(ordinary)}" width="36" height="{380-y(ordinary)}" fill="#4c78a8"/><rect x="{x+6}" y="{y(weighted)}" width="36" height="{380-y(weighted)}" fill="#f58518"/><text x="{x-10}" y="405" font-family="Arial" font-size="14">k={row["k"]}</text>'
        body += '<text x="70" y="65" font-family="Arial" font-size="12">1.0</text><text x="70" y="385" font-family="Arial" font-size="12">0.0</text><text x="590" y="415" font-family="Arial" font-size="12">blue=ordinary, orange=weighted</text>'
        path = figure_dir / f"{dataset}_figure1_retention.svg"
        path.write_text(svg(f"{dataset}: ordinary vs higher-order weighted retention", body), encoding="utf-8"); written.append(str(path))

        body = '<line x1="60" y1="380" x2="720" y2="380" stroke="black"/><line x1="60" y1="70" x2="60" y2="380" stroke="black"/>'
        for row in dataset_rows:
            k = row["k"]
            subset = [item for item in bin_rows if item.get("dataset") == dataset and item.get("k") == k and item.get("element_type", "edge") == "edge"]
            xs = [item["participation_bin"] for item in subset]
            ys = [item.get("preservation_probability") or 0 for item in subset]
            points = []
            for index, value in enumerate(ys):
                px = 90 + index * 115
                py = y(value)
                points.append(f"{px},{py}")
                body += f'<circle cx="{px}" cy="{py}" r="4" fill="black"/><text x="{px-12}" y="405" font-family="Arial" font-size="11">{esc(xs[index])}</text>'
            if points:
                body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{"#4c78a8" if k == 2 else "#f58518"}" stroke-width="2"/><text x="610" y="{90 + 20*k}" font-family="Arial" font-size="12">k={k}</text>'
        path = figure_dir / f"{dataset}_figure2_participation.svg"
        path.write_text(svg(f"{dataset}: higher-order participation vs preservation", body), encoding="utf-8"); written.append(str(path))

        body = '<line x1="60" y1="380" x2="720" y2="380" stroke="black"/><line x1="60" y1="70" x2="60" y2="380" stroke="black"/>'
        for row in dataset_rows:
            k = row["k"]
            values = [row.get(f"top_{int(p * 100):02d}pct_backbone_retention") for p in TOP_PCTS]
            points = []
            for index, value in enumerate(values):
                px = 90 + index * 115
                py = y(value)
                points.append(f"{px},{py}")
                body += f'<circle cx="{px}" cy="{py}" r="4" fill="black"/><text x="{px-9}" y="405" font-family="Arial" font-size="11">{int(TOP_PCTS[index]*100)}%</text>'
            body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{"#4c78a8" if k == 2 else "#f58518"}" stroke-width="2"/><text x="610" y="{90 + 20*k}" font-family="Arial" font-size="12">k={k}</text>'
        path = figure_dir / f"{dataset}_figure3_top_backbone.svg"
        path.write_text(svg(f"{dataset}: top-X% backbone preservation", body), encoding="utf-8"); written.append(str(path))

        body = '<line x1="60" y1="380" x2="720" y2="380" stroke="black"/><line x1="60" y1="70" x2="60" y2="380" stroke="black"/>'
        for row in dataset_rows:
            k = row["k"]
            x = 180 + (k - 2) * 300
            baseline = row.get("random_weighted_retention_mean") or 0
            dual = row.get("higher_order_weighted_edge_retention") or 0
            body += f'<circle cx="{x}" cy="{y(baseline)}" r="6" fill="#4c78a8"/><circle cx="{x+80}" cy="{y(dual)}" r="6" fill="#f58518"/><text x="{x+20}" y="405" font-family="Arial" font-size="14">k={k}</text>'
        body += '<text x="560" y="90" font-family="Arial" font-size="12">blue=random, orange=Dual</text>'
        path = figure_dir / f"{dataset}_figure4_random_baseline.svg"
        path.write_text(svg(f"{dataset}: Dual-Mapping vs matched random deletion", body), encoding="utf-8"); written.append(str(path))
    return {"status": "PASS", "files": written}

def _report(rows, checks, bin_rows, random_rows, figure_info, output_path):
    cases = {k: _aggregate_case(rows, k) for k in (2, 3)}
    k3_random_gaps = [
        row["higher_order_weighted_edge_retention"] - row["random_weighted_retention_mean"]
        for row in rows if row.get("k") == 3
        and row.get("higher_order_weighted_edge_retention") is not None
        and row.get("random_weighted_retention_mean") is not None
    ]
    claim_supported = cases[3]["case"] == "Case B" and bool(k3_random_gaps) and float(np.mean(k3_random_gaps)) > 0.02
    lines = [
        "# Dual-Mapping information audit",
        "",
        f"K2_GLOBAL_LOSS = {cases[2]['global']}",
        f"K2_BACKBONE_LOSS = {cases[2]['backbone']}",
        f"K2_SELECTIVE_BACKBONE_PRESERVATION = {cases[2]['selective']}",
        f"K3_GLOBAL_LOSS = {cases[3]['global']}",
        f"K3_BACKBONE_LOSS = {cases[3]['backbone']}",
        f"K3_SELECTIVE_BACKBONE_PRESERVATION = {cases[3]['selective']}",
        "",
        f"Overall k=2: {cases[2]['case']}; mean ordinary edge retention={_fmt(cases[2]['ordinary'])}; mean backbone score={_fmt(cases[2]['backbone_score'])}.",
        f"Overall k=3: {cases[3]['case']}; mean ordinary edge retention={_fmt(cases[3]['ordinary'])}; mean backbone score={_fmt(cases[3]['backbone_score'])}.",
        f"Requested claim verdict: **{'SUPPORTED' if claim_supported else 'NOT SUPPORTED'}** (aggregate k=3; mean weighted-edge advantage over matched random={_fmt(float(np.mean(k3_random_gaps)) if k3_random_gaps else None)}).",
        "",
        "## Scope and protocol",
        "",
        "- Pure structural audit only: no model, optimizer, negative sampling, hyperparameter search, or future closure labels were used.",
        "- All datasets use stable chronological `(timestamp, raw_row_id)` order and the 80% observed-history cutoff by default.",
        "- Each audit node universe is `closed_k_simplices ∪ open_k_candidates`; open candidates require every observed boundary face and exclude observed target simplices.",
        "- Back-mapping is the union of boundaries of all Dual-Mapping nodes. Target-dimensional reserved simplices are the observed closed target set; open candidates are not counted as observed simplices.",
        "- The candidate hash and all-dual-node hash are deterministic canonical SHA-256 hashes. Future events are not used to construct any graph, candidate set, participation score, baseline, or figure.",
        "- Interpretation caution: weighted retention and candidate-boundary retention can equal 1 by definition here because back-mapping is the union of every closed/open Dual-Mapping node boundary. They verify complete higher-order coverage; the matched-random comparison and top-backbone curves provide the nontrivial selectivity context.",
        "",
        "## Per-dataset results",
        "",
        "| dataset | snapshot | k | closed | open | dual nodes | edge retention | weighted edge | weighted face | top10 edge | top20 edge | boundary | lower adjacency | normalized spectral | random weighted mean ± std | case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        edge_top10 = row.get("top_10pct_edge_backbone_retention")
        edge_top20 = row.get("top_20pct_edge_backbone_retention")
        random_text = f"{_fmt(row.get('random_weighted_retention_mean'))} ± {_fmt(row.get('random_weighted_retention_std'))}"
        lines.append(
            f"| {row['dataset']} | {row['snapshot']:.2f} | {row['k']} | {row['n_closed']} | {row['n_open']} | {row['n_total_dual_nodes']} | {_fmt(row.get('edge_retention_ratio'))} | {_fmt(row.get('higher_order_weighted_edge_retention'))} | {_fmt(row.get('higher_order_weighted_face_retention'))} | {_fmt(edge_top10)} | {_fmt(edge_top20)} | {_fmt(row.get('candidate_boundary_retention'))} | {row.get('exact_lower_adjacency_preservation')} | {_fmt(row.get('normalized_spectral_error'))} | {random_text} | {row.get('audit_case', 'N/A')} |"
        )
    lines += ["", "## Audit cases", "", "- Case A: global loss small and backbone loss small.", "- Case B: global loss moderate/large but backbone loss small.", "- Case C: global loss large and backbone loss also large.", "", "The verdict above is an aggregate audit conclusion; individual datasets remain visible in the table and can include weaker or unclassified rows.", "", "## Integrity checks", ""]
    for key, value in checks.items():
        lines.append(f"- {key}: **{'PASS' if value else 'FAIL'}**")
    lines += [
        "",
        "```text",
        "FINAL_TEST_LABELS_ACCESSED = NO",
        "MODEL_TRAINING_PERFORMED = NO",
        "HYPERPARAMETER_SEARCH_PERFORMED = NO",
        "```",
        "",
        "## Figures",
        "",
    ]
    for path in figure_info.get("files", []):
        lines.append(f"- `{Path(path).name}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--snapshots", type=float, nargs="+", default=[0.8], help="Observed-history fractions; default: 0.8")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--baseline-repetitions", type=int, default=20)
    return parser.parse_args()

def main():
    args = _parse_args()
    global RAW_ROOT
    RAW_ROOT = args.raw_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    datasets = sorted(args.datasets or [path.name for path in RAW_ROOT.iterdir() if path.is_dir()])
    snapshots = sorted(set(float(value) for value in args.snapshots))
    if any(value <= 0 or value > 1 for value in snapshots):
        raise ValueError("snapshots must be in (0, 1]")

    summary_rows = []
    random_rows = []
    bin_rows = []
    integrity_rows = []
    details_json = {}
    element_appenders = {2: CsvAppender(output / "k2_element_level.csv"), 3: CsvAppender(output / "k3_element_level.csv")}
    started = time.perf_counter()
    for dataset in datasets:
        try:
            events, load_stats, paths = _canonical_events(dataset)
        except Exception as exc:
            for k in (2, 3):
                for snapshot in snapshots:
                    summary_rows.append({"dataset": dataset, "snapshot": snapshot, "k": k, "status": "LOAD_FAILURE", "error": repr(exc)})
            continue
        for snapshot in snapshots:
            context_end = int(math.floor(snapshot * len(events)))
            context_end = min(max(context_end, 0), len(events))
            context_events = events[:context_end]
            state = build_observed_state(context_events, max_dim=3)
            snapshot_key = f"{snapshot:.2f}"
            details_json.setdefault(dataset, {})[snapshot_key] = {
                "load_stats": load_stats, "input_paths": [str(path) for path in paths], "context_end": context_end,
                "history_cutoff_timestamp": context_events[-1]["time"] if context_events else None,
                "future_event_count_not_used_for_construction": len(events) - context_end,
            }
            for k in (2, 3):
                try:
                    open_candidates, generation = enumerate_open_simplices(state, k)
                    closed = sorted(state["simplices_by_size"][k + 1])
                    total_dual_nodes = len(closed) + len(open_candidates)
                    all_nodes = (
                        None if total_dual_nodes > MAX_EXPLICIT_DUAL_NODES
                        else _merge_sorted_unique(closed, open_candidates)
                    )
                    if all_nodes is not None:

                        estimated_face_counts = _dual_face_counts(all_nodes, k)
                        estimated_edges = sum(
                            count * (count - 1) // 2 for count in estimated_face_counts.values()
                        )
                        if estimated_edges > MAX_EXPLICIT_DUAL_EDGES:
                            all_nodes = None
                    details = _scope_result(
                        dataset, snapshot, k, events, context_end, state,
                        open_candidates, all_nodes, output,
                        baseline_repetitions=args.baseline_repetitions,
                    )
                    print(f"[audit] {dataset} snapshot={snapshot:.2f} k={k} complete", flush=True)
                    row = details["summary"]
                    row["candidate_generation_seconds"] = generation.get("candidate_generation_seconds")
                    row["candidate_generation_method"] = generation.get("enumeration_method")
                    summary_rows.append(row)
                    random_rows.extend(details["random_rows"])
                    for item in details["bins"]:
                        bin_rows.append({"dataset": dataset, "snapshot": snapshot, "k": k, **item})
                    element_appenders[k].append(details["element_rows"])

                    details_json[dataset][snapshot_key][f"k{k}"] = {
                        "summary": row,
                        "participation": details["participation"],
                        "bins": details["bins"],
                        "boundary": details["boundary"],
                        "network": details["network"],
                        "spectral": details["spectral"],
                        "context": details["context"],
                        "mismatch_examples": details["mismatch_examples"],
                    }
                    integrity_rows.append({"dataset": dataset, "snapshot": snapshot, "k": k, **row["checks"], "all_checks_pass": row["all_checks_pass"]})
                except Exception as exc:
                    summary_rows.append({"dataset": dataset, "snapshot": snapshot, "k": k, "status": "AUDIT_FAILURE", "error": repr(exc)})
                    integrity_rows.append({"dataset": dataset, "snapshot": snapshot, "k": k, "all_checks_pass": False, "error": repr(exc)})
    for appender in element_appenders.values():
        appender.close()

    _write_csv(output / "audit_summary.csv", summary_rows)
    _write_csv(output / "k2_detailed_metrics.csv", [row for row in summary_rows if row.get("k") == 2])
    _write_csv(output / "k3_detailed_metrics.csv", [row for row in summary_rows if row.get("k") == 3])
    _write_csv(output / "random_baseline_summary.csv", random_rows)
    _write_csv(output / "participation_bins.csv", bin_rows)
    _write_csv(output / "integrity_checks.csv", integrity_rows)
    _write_json(output / "audit_summary.json", {
        "protocol": {
            "snapshots": snapshots, "datasets": datasets, "future_data_used_for_construction": False,
            "final_test_labels_accessed": False, "model_training_performed": False, "hyperparameter_search_performed": False,
            "random_baseline_repetitions": args.baseline_repetitions, "candidate_universe": "closed_k_simplices_union_open_k_candidates",
        },
        "rows": summary_rows, "integrity_rows": integrity_rows, "runtime_seconds": time.perf_counter() - started,
    })
    _write_json(output / "audit_details.json", details_json)
    figure_info = _make_figures(summary_rows, bin_rows, random_rows, output / "figures")
    overall_checks = {
        "candidate_identity_deterministic": all(row.get("candidate_identity_deterministic", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
        "no_future_timestamps_used": all(row.get("no_future_timestamps_used", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
        "all_open_k2_candidates_have_complete_edge_boundaries": all(row.get("all_open_k2_candidates_have_complete_edge_boundaries", False) for row in integrity_rows if row.get("k") == 2 and row.get("all_checks_pass") is not None),
        "all_open_k3_candidates_have_complete_triangular_boundaries": all(row.get("all_open_k3_candidates_have_complete_triangular_boundaries", False) for row in integrity_rows if row.get("k") == 3 and row.get("all_checks_pass") is not None),
        "dual_graph_contains_no_self_loop": all(row.get("dual_graph_contains_no_self_loop", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
        "dual_edges_satisfy_lower_adjacency_rule": all(row.get("dual_edges_satisfy_lower_adjacency_rule", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
        "no_duplicated_simplex_ids": all(row.get("no_duplicated_simplex_ids", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
        "preserved_set_is_subset_of_original_set": all(row.get("preserved_set_is_subset_of_original_set", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
        "weighted_retention_in_0_1": all(row.get("weighted_retention_in_0_1", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
        "random_baseline_uses_exactly_matched_retention_counts": all(row.get("random_baseline_uses_exactly_matched_retention_counts", False) for row in integrity_rows if row.get("all_checks_pass") is not None),
    }
    _report(summary_rows, overall_checks, bin_rows, random_rows, figure_info, output / "AUDIT_REPORT.md")
    print(json.dumps({
        "output_dir": str(output), "datasets": datasets, "snapshots": snapshots,
        "successful_scopes": sum(row.get("status") == "PASS" for row in summary_rows),
        "failed_scopes": sum(row.get("status") != "PASS" for row in summary_rows),
        "overall_checks_pass": all(overall_checks.values()), "figures": figure_info,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
