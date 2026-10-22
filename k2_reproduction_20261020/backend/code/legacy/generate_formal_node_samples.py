#!/usr/bin/env python3
"""Recreate the five formal K2 node lists using the original server sampler.

Run from the K2 package root (the directory containing dataset/raw_data):

    python3 generate_formal_node_samples.py

This uses only the Python standard library. The selected nodes and sampling
rules match server_package/prepare_server_data.py, not the K2 package's
different fallback reconstruction sampler.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


DATASETS = (
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-ask-ubuntu",
)
BUDGETS = (800, 600, 400, 300, 250, 200, 160, 120, 100, 80, 60, 40)
EXPECTED_COUNTS = {
    "email-Eu": 200,
    "NDC-substances": 200,
    "tags-ask-ubuntu": 40,
    "tags-math-sx": 40,
    "threads-ask-ubuntu": 600,
}


def read_ints(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows(raw_root: Path, dataset: str) -> list[dict]:
    folder = raw_root / dataset
    nverts = read_ints(folder / f"{dataset}-nverts.txt")
    flat_vertices = read_ints(folder / f"{dataset}-simplices.txt")
    times = (folder / f"{dataset}-times.txt").read_text(encoding="utf-8").splitlines()
    if len(nverts) != len(times) or sum(nverts) != len(flat_vertices):
        raise ValueError(f"{dataset}: raw nverts/simplices/times lengths do not match")
    rows = []
    offset = 0
    for index, count in enumerate(nverts):
        vertices = flat_vertices[offset : offset + count]
        offset += count
        rows.append({"idx": index, "vertices": vertices, "time": times[index]})
    return rows


def original_node_order(rows: list[dict], seed: int) -> list[int]:
    """The history-only BFS/co-occurrence ordering in the formal data sampler."""
    history = rows[: int(0.6 * len(rows))]
    adjacency: dict[int, set[int]] = defaultdict(set)
    for row in history:
        nodes = sorted(set(row["vertices"]))
        for node in nodes:
            adjacency[node].update(other for other in nodes if other != node)

    nodes = sorted(adjacency)
    rng = random.Random(seed)
    tie = {node: rng.random() for node in nodes}
    key = lambda node: (-len(adjacency[node]), tie[node], node)

    triangle_events = [row for row in history if len(set(row["vertices"])) >= 3]
    rng.shuffle(triangle_events)
    seeds = sorted(
        {node for row in triangle_events[:100] for node in row["vertices"]},
        key=key,
    )
    if not seeds:
        seeds = sorted(nodes, key=key)[:25]

    result: list[int] = []
    seen: set[int] = set()
    frontier = seeds
    while frontier:
        for node in frontier:
            if node not in seen:
                result.append(node)
                seen.add(node)
        neighbors = set().union(*(adjacency[node] for node in frontier))
        frontier = sorted(neighbors - seen, key=key)
    result.extend(node for node in sorted(nodes, key=key) if node not in seen)
    return result


def history_resources(rows: list[dict], selected: set[int]) -> tuple[int, int]:
    """Count simplices only in events fully induced by the selected nodes."""
    edges: set[tuple[int, int]] = set()
    triangles: set[tuple[int, int, int]] = set()
    for row in rows[: int(0.8 * len(rows))]:
        vertices = sorted(set(row["vertices"]))
        if not vertices or not set(vertices).issubset(selected):
            continue
        edges.update(combinations(vertices, 2))
        triangles.update(combinations(vertices, 3))
    return len(edges), len(triangles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", default="dataset/raw_data")
    parser.add_argument("--output-dir", default="node_samples")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-edges", type=int, default=4000)
    parser.add_argument("--max-context-triangles", type=int, default=3000)
    args = parser.parse_args()

    raw_root = Path(args.raw_data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "sampler": "original server_package.prepare_server_data.py node_order/history_resources",
        "rule": "rank from first 60% history; retain fully induced raw events; resource counts on first 80%",
        "datasets": [],
    }

    for dataset in DATASETS:
        rows = load_rows(raw_root, dataset)
        order = original_node_order(rows, args.seed)
        selected = None
        selected_budget = None
        selected_edges = selected_triangles = None
        for budget in sorted({min(len(order), size) for size in BUDGETS}, reverse=True):
            candidate = set(order[:budget])
            edges, triangles = history_resources(rows, candidate)
            if edges <= args.max_edges and triangles <= args.max_context_triangles:
                selected = candidate
                selected_budget = budget
                selected_edges, selected_triangles = edges, triangles
                break

        if selected is None:
            raise RuntimeError(f"{dataset}: no configured budget satisfies the resource limits")
        if len(selected) != EXPECTED_COUNTS[dataset]:
            raise RuntimeError(
                f"{dataset}: sampler selected {len(selected)} nodes, "
                f"but the formal reference requires {EXPECTED_COUNTS[dataset]}"
            )

        selected_ids = sorted(selected)
        (output_root / f"{dataset}.txt").write_text(
            "".join(f"{node}\n" for node in selected_ids),
            encoding="utf-8",
            newline="\n",
        )
        retained_events = sum(
            1
            for row in rows
            if set(row["vertices"]).issubset(selected)
        )
        manifest["datasets"].append(
            {
                "dataset": dataset,
                "budget": selected_budget,
                "selected_node_count": len(selected_ids),
                "selected_node_ids": selected_ids,
                "source_event_count": len(rows),
                "retained_event_count": retained_events,
                "context_edges_first80": selected_edges,
                "context_closed_triangles_first80": selected_triangles,
            }
        )
        print(
            f"{dataset}: nodes={len(selected_ids)}, events={retained_events}/{len(rows)}, "
            f"first80_edges={selected_edges}, first80_closed_triangles={selected_triangles}",
            flush=True,
        )

    (output_root / "sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Node lists and manifest written to {output_root}", flush=True)


if __name__ == "__main__":
    main()
