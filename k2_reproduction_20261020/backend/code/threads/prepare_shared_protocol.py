#!/usr/bin/env python3
"""Prepare one shared, source-row-order K=2 protocol for Dual and MLSHoL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np


DATASETS = [
    "email-Enron",
    "contact-high-school",
    "contact-primary-school",
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-ask-ubuntu",
    "threads-math-sx",
]
LARGE_DATASETS = {
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-ask-ubuntu",
    "threads-math-sx",
}
NODE_BUDGETS = [800, 600, 400, 300, 250, 200, 160, 120, 100, 80, 60, 40]
EXPECTED_NODE_COUNTS = {
    "email-Eu": 200,
    "NDC-substances": 200,
    "tags-ask-ubuntu": 40,
    "tags-math-sx": 40,
    "threads-ask-ubuntu": 600,
    "threads-math-sx": 600,
}
NODE_RESOURCE_LIMITS = {"threads-math-sx": (10000, 5000)}
EXPECTED_TEST_COUNTS = {
    "NDC-substances": (13, 774),
    "contact-high-school": (201, 25056),
    "contact-primary-school": (33, 4650),
    "email-Enron": (94, 2737),
    "email-Eu": (141, 10893),
    "tags-ask-ubuntu": (61, 6145),
    "tags-math-sx": (179, 4791),
    "threads-ask-ubuntu": (19, 3851),
    "threads-math-sx": (61, 29175),
}


def read_ints(path: Path) -> list[int]:
    return [int(value) for value in path.read_text(encoding="utf-8").splitlines() if value.strip()]


def digest_rows(rows) -> str:
    payload = "".join(" ".join(map(str, row)) + "\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_ints(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(value)}\n" for value in values), encoding="utf-8")


def write_triangles(path: Path, triangles) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{a} {b} {c}\n" for a, b, c in triangles), encoding="utf-8"
    )


def load_raw(raw_dir: Path, dataset: str):
    counts = read_ints(raw_dir / f"{dataset}-nverts.txt")
    flat = read_ints(raw_dir / f"{dataset}-simplices.txt")
    times = read_ints(raw_dir / f"{dataset}-times.txt")
    if len(counts) != len(times) or sum(counts) != len(flat):
        raise ValueError(f"Malformed raw files for {dataset}")
    events, offset = [], 0
    for event_id, (count, timestamp) in enumerate(zip(counts, times)):
        vertices = tuple(sorted(set(flat[offset : offset + count])))
        offset += count
        if not vertices:
            raise ValueError(f"Empty simplex at raw row {event_id} in {dataset}")
        events.append((int(timestamp), vertices))
    return counts, times, events


def node_expansion_order(events, seed: int, max_nodes: int = 800):
    """History-only deterministic co-occurrence expansion seeded by a >=3 event."""
    rng = random.Random(seed)
    cooc = Counter()
    adjacency = defaultdict(Counter)
    nodes = set()
    history = events[: int(0.60 * len(events))]
    for _, vertices in history:
        nodes.update(vertices)
        for left, right in combinations(vertices, 2):
            cooc[left] += 1
            cooc[right] += 1
            adjacency[left][right] += 1
            adjacency[right][left] += 1
    tie_key = {node: rng.random() for node in nodes}
    global_order = sorted(nodes, key=lambda node: (-cooc[node], tie_key[node], node))

    anchor_events = [vertices for _, vertices in history if len(vertices) >= 3][:100]
    rng.shuffle(anchor_events)
    selected = []
    if anchor_events:
        anchor = anchor_events[0]
        selected = sorted(anchor, key=lambda node: (-cooc[node], tie_key.get(node, 0.0), node))[:max_nodes]
    elif global_order:
        selected = [global_order[0]]

    selected_set = set(selected)
    order = list(selected)
    while len(order) < min(max_nodes, len(nodes)):
        frontier = set()
        for node in selected_set:
            frontier.update(adjacency[node])
        frontier.difference_update(selected_set)
        if frontier:
            chosen = min(
                frontier,
                key=lambda node: (
                    -sum(adjacency[node].get(member, 0) for member in selected_set),
                    -cooc[node],
                    tie_key[node],
                    node,
                ),
            )
        else:
            chosen = next((node for node in global_order if node not in selected_set), None)
            if chosen is None:
                break
        selected_set.add(chosen)
        order.append(chosen)
    return order


def resource_counts_by_budget(events, expansion_order, end_index, budgets):
    rank = {node: index for index, node in enumerate(expansion_order)}
    edge_threshold, closed_threshold = {}, {}
    for _, vertices in events[:end_index]:
        positions = [rank[node] for node in vertices if node in rank]
        for edge in combinations(positions, 2):
            key = tuple(sorted(edge))
            threshold = max(key)
            edge_threshold[key] = min(threshold, edge_threshold.get(key, threshold))
        for triangle in combinations(positions, 3):
            key = tuple(sorted(triangle))
            threshold = max(key)
            closed_threshold[key] = min(threshold, closed_threshold.get(key, threshold))
    results = {}
    for budget in budgets:
        results[budget] = (
            sum(threshold < budget for threshold in edge_threshold.values()),
            sum(threshold < budget for threshold in closed_threshold.values()),
        )
    return results


def choose_nodes(dataset: str, events, seed: int, node_lists_dir: Path | None = None):
    if dataset not in LARGE_DATASETS:
        return None, {"sampled": False, "selected_nodes": None}
    explicit_path = node_lists_dir / f"{dataset}.txt" if node_lists_dir else None
    if explicit_path is not None and explicit_path.is_file():
        selected_nodes = sorted(set(read_ints(explicit_path)))
        if len(selected_nodes) != EXPECTED_NODE_COUNTS[dataset]:
            raise ValueError(
                f"{explicit_path} has {len(selected_nodes)} unique nodes; "
                f"expected {EXPECTED_NODE_COUNTS[dataset]}"
            )
        available = {node for _, vertices in events[: int(0.60 * len(events))] for node in vertices}
        if not set(selected_nodes).issubset(available):
            raise ValueError(f"{explicit_path} contains nodes absent from the first 60% history")
        edges, triangles = context_resource_counts_for_set(events, selected_nodes, int(0.80 * len(events)))
        max_edges, max_triangles = NODE_RESOURCE_LIMITS.get(dataset, (4000, 3000))
        if edges > max_edges or triangles > max_triangles:
            raise RuntimeError(
                f"{explicit_path} violates resource limits: edges={edges}>{max_edges}, closed_triangles={triangles}>{max_triangles}"
            )
        return selected_nodes, {
            "sampled": True,
            "node_selection_source": explicit_path.name,
            "selected_node_count": len(selected_nodes),
            "expected_reference_node_count": EXPECTED_NODE_COUNTS[dataset],
            "reference_node_count_match": True,
            "context_edges_first80": edges,
            "context_closed_triangles_first80": triangles,
            "node_selection_seed": seed,
            "selected_node_sha256": hashlib.sha256(
                ",".join(map(str, selected_nodes)).encode("utf-8")
            ).hexdigest(),
            "selected_nodes": selected_nodes,
        }
    order = node_expansion_order(events, seed)
    end80 = int(0.80 * len(events))
    resource_counts = resource_counts_by_budget(events, order, end80, NODE_BUDGETS)
    max_edges, max_triangles = NODE_RESOURCE_LIMITS.get(dataset, (4000, 3000))
    chosen_budget = None
    edges = triangles = None
    for budget in NODE_BUDGETS:
        if len(order) < budget:
            continue
        edge_count, closed_count = resource_counts[budget]
        if edge_count <= max_edges and closed_count <= max_triangles:
            chosen_budget = budget
            edges, triangles = edge_count, closed_count
            break
    if chosen_budget is None:
        raise RuntimeError(f"{dataset}: no configured node budget satisfies edge/triangle limits")
    selected_nodes = sorted(order[:chosen_budget])
    expected = EXPECTED_NODE_COUNTS[dataset]
    return selected_nodes, {
        "sampled": True,
        "node_selection_source": "history_only_cooccurrence_expansion",
        "selected_node_count": len(selected_nodes),
        "expected_reference_node_count": expected,
        "reference_node_count_match": len(selected_nodes) == expected,
        "context_edges_first80": edges,
        "context_closed_triangles_first80": triangles,
        "node_selection_seed": seed,
        "selected_node_sha256": hashlib.sha256(
            ",".join(map(str, selected_nodes)).encode("utf-8")
        ).hexdigest(),
        "selected_nodes": selected_nodes,
    }


def context_resource_counts_for_set(events, selected_nodes, end_index):
    selected = set(selected_nodes)
    edges, closed = set(), set()
    for _, vertices in events[:end_index]:
        # Match the formal sampler and the filtering step below: retain an
        # event only when every one of its vertices belongs to the sample.
        if not set(vertices).issubset(selected):
            continue
        kept = vertices
        edges.update(tuple(sorted(edge)) for edge in combinations(kept, 2))
        closed.update(tuple(sorted(triangle)) for triangle in combinations(kept, 3))
    return len(edges), len(closed)


def filter_events_by_nodes(events, selected_nodes):
    if selected_nodes is None:
        return list(events), len(events)
    keep = set(selected_nodes)
    filler = min(keep)
    filtered, retained = [], 0
    for timestamp, vertices in events:
        if all(node in keep for node in vertices):
            filtered.append((timestamp, vertices))
            retained += 1
        else:
            # Preserve every source-row index and timestamp while removing this
            # event's edges/triangles from all model-side history readers.
            filtered.append((timestamp, (filler,)))
    return filtered, retained


def graph_task(events, context_end: int, future_start: int, future_end: int):
    context, future = events[:context_end], events[future_start:future_end]
    nodes, edges, closed = set(), set(), set()
    for _, vertices in context:
        nodes.update(vertices)
        edges.update(tuple(sorted(edge)) for edge in combinations(vertices, 2))
        closed.update(tuple(sorted(triangle)) for triangle in combinations(vertices, 3))
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(sorted(edges))
    adjacency = {node: set(graph.neighbors(node)) for node in graph.nodes}
    candidates = []
    for left in sorted(adjacency):
        for middle in sorted(node for node in adjacency[left] if node > left):
            for right in sorted(node for node in adjacency[left] & adjacency[middle] if node > middle):
                triangle = (left, middle, right)
                if triangle not in closed:
                    candidates.append(triangle)
    future_closed = set()
    for _, vertices in future:
        future_closed.update(tuple(sorted(triangle)) for triangle in combinations(vertices, 3))
    positives = sorted(set(candidates) & future_closed)
    return graph, sorted(candidates), positives, sorted(closed)


def classwise_train_val_split(candidates, positives, seed: int):
    positive_set = set(positives)
    by_label = {0: [], 1: []}
    for triangle in candidates:
        by_label[int(triangle in positive_set)].append(triangle)
    if min(map(len, by_label.values())) < 2:
        raise RuntimeError("60–80% candidate pool must contain at least two examples per class")
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for label in (1, 0):
        rows = np.asarray(by_label[label], dtype=np.int64)
        order = rng.permutation(len(rows))
        val_count = min(max(1, int(np.floor(0.20 * len(rows)))), len(rows) - 1)
        val.extend(tuple(map(int, rows[index])) for index in order[:val_count])
        fit.extend(tuple(map(int, rows[index])) for index in order[val_count:])
    return fit, val


def cap_fit_validation(rows, positive_set, cap, negatives_per_positive, seed):
    if len(rows) <= cap:
        return list(rows)
    positive = [row for row in rows if row in positive_set]
    negative = [row for row in rows if row not in positive_set]
    if not positive or not negative:
        raise RuntimeError("train/validation split has only one class")
    if len(positive) > cap:
        raise RuntimeError(f"positive count {len(positive)} exceeds candidate cap {cap}")
    negative_cap = min(len(negative), negatives_per_positive * len(positive), cap - len(positive))
    rng = np.random.default_rng(seed)
    if negative_cap < len(negative):
        negative = [negative[int(index)] for index in rng.choice(len(negative), negative_cap, replace=False)]
    selected = positive + negative
    order = rng.permutation(len(selected))
    return [selected[int(index)] for index in order]


def cap_test_unlabeled(candidates, cap, seed):
    if len(candidates) <= cap:
        return list(candidates)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(candidates), size=cap, replace=False))
    return [candidates[int(index)] for index in chosen]


def write_task(root: Path, dataset: str, split: str, graph, open_rows, positives, closed):
    directory = root / "dataset" / split
    directory.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, directory / f"{dataset}_G_{split}.graphml")
    write_triangles(directory / f"{dataset}_openTriangles_{split}.txt", open_rows)
    write_triangles(directory / f"{dataset}_new_closures_{split}.txt", positives)
    write_triangles(directory / f"{dataset}_closeTriangles_{split}.txt", closed)


def prepare_one(source_root: Path, data_root: Path, dataset: str, seed: int, node_lists_dir: Path | None):
    source_dir = source_root / "dataset" / "raw_data" / dataset
    counts, times, events = load_raw(source_dir, dataset)
    n_events = len(events)
    selected_nodes, sampling = choose_nodes(dataset, events, seed, node_lists_dir)
    filtered, retained = filter_events_by_nodes(events, selected_nodes)

    raw_out = data_root / "dataset" / "raw_data" / dataset
    raw_out.mkdir(parents=True, exist_ok=True)
    write_ints(raw_out / f"{dataset}-nverts.txt", [len(vertices) for _, vertices in filtered])
    write_ints(raw_out / f"{dataset}-times.txt", times)
    write_ints(raw_out / f"{dataset}-simplices.txt", [node for _, vertices in filtered for node in vertices])

    train_context_end = int(0.60 * n_events)
    train_future_end = int(0.80 * n_events)
    train_graph, full_train_candidates, full_train_positive, train_closed = graph_task(
        filtered, train_context_end, train_context_end, train_future_end
    )
    fit_pool, val_pool = classwise_train_val_split(full_train_candidates, full_train_positive, 2026)
    positive_set = set(full_train_positive)
    fit_rows = cap_fit_validation(fit_pool, positive_set, 20000, 20, 2037)
    val_rows = cap_fit_validation(val_pool, positive_set, 5000, 50, 2038)
    # Candidate order is explicit and shared by both adapters. This also fixes
    # the positional fit/validation arrays consumed by the model wrappers.
    train_rows = fit_rows + val_rows
    train_labels = [int(row in positive_set) for row in train_rows]
    fit_indices = list(range(len(fit_rows)))
    val_indices = list(range(len(fit_rows), len(train_rows)))
    fit_positive_rows = [row for row in fit_rows if row in positive_set]

    test_context_end = int(0.80 * n_events)
    test_graph, full_test_candidates, full_test_positive, test_closed = graph_task(
        filtered, test_context_end, test_context_end, n_events
    )
    test_rows = cap_test_unlabeled(full_test_candidates, 50000, 2039)
    test_positive_set = set(full_test_positive)
    test_positive_rows = [row for row in test_rows if row in test_positive_set]

    write_task(data_root, dataset, "train", train_graph, train_rows, fit_positive_rows + [row for row in val_rows if row in positive_set], train_closed)
    write_task(data_root, dataset, "test", test_graph, test_rows, test_positive_rows, test_closed)
    write_triangles(
        data_root / "dataset" / "train" / f"{dataset}_fit_positive_train.txt",
        fit_positive_rows,
    )
    split_dir = data_root / "shared_splits" / dataset
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_dir / "fit_val_indices.npz",
        fit_indices=np.asarray(fit_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        train_labels=np.asarray(train_labels, dtype=np.int64),
    )

    expected = EXPECTED_TEST_COUNTS.get(dataset)
    observed = (len(test_positive_rows), len(test_rows))
    expected_node_match = sampling.get("reference_node_count_match", True)
    reference_match = (expected is None or observed == expected) and expected_node_match
    manifest = {
        "dataset": dataset,
        "protocol": "K2 raw-source-row order; train context 0-60/future 60-80; test context 0-80/future 80-100",
        "event_order": "raw",
        "source_event_count": n_events,
        "source_event_sha256": hashlib.sha256(
            (source_dir / f"{dataset}-simplices.txt").read_bytes()
        ).hexdigest(),
        "retained_events": retained,
        "filtered_rows_preserve_original_indices": True,
        "filtered_rows_replaced_by_singleton_sentinel": selected_nodes is not None,
        "sampling": sampling,
        "train_full_candidate_count_before_caps": len(full_train_candidates),
        "train_full_positive_count_before_caps": len(full_train_positive),
        "fit_candidate_count": len(fit_rows),
        "fit_positive_count": sum(train_labels[: len(fit_rows)]),
        "validation_candidate_count": len(val_rows),
        "validation_positive_count": sum(train_labels[len(fit_rows) :]),
        "train_union_count": len(train_rows),
        "train_union_positive_count": sum(train_labels),
        "train_candidate_sha256": digest_rows(train_rows),
        "train_label_sha256": hashlib.sha256(bytes(train_labels)).hexdigest(),
        "fit_indices": fit_indices,
        "validation_indices": val_indices,
        "test_candidate_count_before_cap": len(full_test_candidates),
        "test_candidate_count": len(test_rows),
        "test_positive_count": len(test_positive_rows),
        "test_prevalence": len(test_positive_rows) / max(1, len(test_rows)),
        "test_candidate_sha256": digest_rows(test_rows),
        "expected_reference_test_positive_candidate_count": None if expected is None else expected[0],
        "expected_reference_test_candidate_count": None if expected is None else expected[1],
        "test_reference_count_match": None if expected is None else observed == expected,
        "reference_match": reference_match,
        "random_seeds": {
            "node_sampling": seed,
            "train_val_split": 2026,
            "fit_negative_sampling": 2037,
            "validation_negative_sampling": 2038,
            "test_unlabeled_sampling": 2039,
        },
    }
    manifest_path = split_dir / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "protocol_data"))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--node-seed", type=int, default=2026)
    parser.add_argument("--node-lists-dir", default=None)
    parser.add_argument("--require-reference-match", action="store_true")
    args = parser.parse_args()
    source_root, data_root = Path(args.source_root).resolve(), Path(args.data_root).resolve()
    node_lists_dir = Path(args.node_lists_dir).resolve() if args.node_lists_dir else (source_root / "node_samples")
    rows = []
    for dataset in args.datasets:
        row = prepare_one(source_root, data_root, dataset, args.node_seed, node_lists_dir)
        rows.append(row)
        print(
            f"{dataset}: fit={row['fit_candidate_count']} val={row['validation_candidate_count']} "
            f"test={row['test_positive_count']}/{row['test_candidate_count']} "
            f"reference_match={row['reference_match']}",
            flush=True,
        )
        if args.require_reference_match and not row["reference_match"]:
            raise RuntimeError(
                f"{dataset}: generated sample does not match the supplied fixed-protocol counts; "
                "do not launch model jobs with an unaligned prevalence/sample."
            )
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "run_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
