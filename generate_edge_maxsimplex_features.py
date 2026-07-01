#!/usr/bin/env python
"""Generate edge maximal-simplex features for derived-graph triplets.

This script reads raw timestamped simplices, builds the historical simplicial
complex up to a context time ratio, extracts maximal simplices, and writes
fixed-width TSV features aligned by triplet value rather than gpickle node id.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Simplex = Tuple[int, ...]
Triplet = Tuple[int, int, int]
Edge = Tuple[int, int]


EDGE_FEATURE_NAMES = (
    "count",
    "max_size",
    "mean_size",
    "min_size",
    "size_hist_2",
    "size_hist_3",
    "size_hist_4",
    "size_hist_5",
    "size_hist_ge6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate maximal-simplex edge features for gpickle triplets."
    )
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--gpickle_path", required=True, type=Path)
    parser.add_argument("--output_txt", required=True, type=Path)
    parser.add_argument("--context_ratio", required=True, type=float)
    return parser.parse_args()


def find_one(raw_dir: Path, patterns: Sequence[str], label: str) -> Path:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(raw_dir.glob(pattern))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {label} file under {raw_dir}; "
            f"patterns={patterns}, matches={[str(p) for p in unique]}"
        )
    return unique[0]


def discover_raw_files(raw_dir: Path) -> Tuple[Path, Path, Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    nverts = find_one(raw_dir, ("*-nverts.txt", "nverts.txt", "nverts"), "nverts")
    simplices = find_one(raw_dir, ("*-simplices.txt", "simplices.txt", "simplices"), "simplices")
    times = find_one(raw_dir, ("*-times.txt", "*-time.txt", "times.txt", "time.txt", "times", "time"), "time")
    return nverts, simplices, times


def read_int_lines(path: Path) -> List[int]:
    values: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                values.append(int(text))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} is not an integer: {text!r}") from exc
    return values


def load_timestamped_simplices(raw_dir: Path) -> List[Tuple[int, int, Simplex]]:
    nverts_path, simplices_path, times_path = discover_raw_files(raw_dir)
    nverts = read_int_lines(nverts_path)
    flat_vertices = read_int_lines(simplices_path)
    times = read_int_lines(times_path)

    if len(nverts) != len(times):
        raise ValueError(
            f"nverts and times length mismatch: {len(nverts)} != {len(times)}"
        )
    total_vertices = sum(nverts)
    if total_vertices != len(flat_vertices):
        raise ValueError(
            f"simplices length mismatch: sum(nverts)={total_vertices}, "
            f"len(simplices)={len(flat_vertices)}"
        )

    records: List[Tuple[int, int, Simplex]] = []
    offset = 0
    for idx, nv in enumerate(nverts):
        if nv < 0:
            raise ValueError(f"{nverts_path}:{idx + 1} has negative nverts={nv}")
        raw_vertices = flat_vertices[offset : offset + nv]
        offset += nv
        vertices = tuple(sorted(set(int(v) for v in raw_vertices)))
        if vertices:
            records.append((int(times[idx]), idx, vertices))
    records.sort(key=lambda item: (item[0], item[1]))
    return records


def select_context(records: Sequence[Tuple[int, int, Simplex]], context_ratio: float):
    if not 0.0 <= context_ratio <= 1.0:
        raise ValueError("--context_ratio must be in [0, 1]")
    if not records:
        return []
    min_time = records[0][0]
    max_time = records[-1][0]
    cutoff = min_time + context_ratio * (max_time - min_time)
    return [record for record in records if record[0] <= cutoff]


def is_covered(simplex_set: Set[int], vertex_to_max_ids: Dict[int, Set[int]]) -> bool:
    candidate_ids = None
    for vertex in simplex_set:
        ids = vertex_to_max_ids.get(vertex, set())
        if not ids:
            return False
        if candidate_ids is None or len(ids) < len(candidate_ids):
            candidate_ids = set(ids)
        else:
            candidate_ids &= ids
        if not candidate_ids:
            return False
    return bool(candidate_ids)


def extract_maximal_simplices(simplices: Iterable[Simplex]) -> List[Simplex]:
    unique = sorted(set(simplices), key=lambda s: (-len(s), s))
    maximal: List[Simplex] = []
    vertex_to_max_ids: Dict[int, Set[int]] = defaultdict(set)

    for simplex in unique:
        simplex_set = set(simplex)
        if is_covered(simplex_set, vertex_to_max_ids):
            continue
        max_id = len(maximal)
        maximal.append(simplex)
        for vertex in simplex:
            vertex_to_max_ids[vertex].add(max_id)
    return maximal


def count_closure_simplices(maximal_simplices: Sequence[Simplex]) -> int:
    closure: Set[Simplex] = set()
    for simplex in maximal_simplices:
        for size in range(1, len(simplex) + 1):
            closure.update(itertools.combinations(simplex, size))
    return len(closure)


def build_edge_index(maximal_simplices: Sequence[Simplex]) -> Dict[Edge, List[int]]:
    edge_to_max_ids: Dict[Edge, List[int]] = defaultdict(list)
    for max_id, simplex in enumerate(maximal_simplices):
        if len(simplex) < 2:
            continue
        for edge in itertools.combinations(simplex, 2):
            edge_to_max_ids[(int(edge[0]), int(edge[1]))].append(max_id)
    return edge_to_max_ids


def read_gpickle_triplets(gpickle_path: Path) -> List[Triplet]:
    with gpickle_path.open("rb") as handle:
        graph = pickle.load(handle)

    triplets: List[Triplet] = []
    for _, data in graph.nodes(data=True):
        if "triplet" not in data:
            raise ValueError(f"{gpickle_path} contains a node without 'triplet'")
        tri = tuple(sorted(int(x) for x in data["triplet"]))
        if len(tri) != 3 or len(set(tri)) != 3:
            raise ValueError(f"Invalid triplet in {gpickle_path}: {data['triplet']!r}")
        triplets.append(tri)  # type: ignore[arg-type]
    return triplets


def edge_stats(max_ids: Sequence[int], maximal_simplices: Sequence[Simplex]) -> List[float]:
    if not max_ids:
        return [0.0] * len(EDGE_FEATURE_NAMES)

    sizes = [len(maximal_simplices[max_id]) for max_id in max_ids]
    return [
        float(len(sizes)),
        float(max(sizes)),
        float(sum(sizes) / len(sizes)),
        float(min(sizes)),
        float(sum(1 for size in sizes if size == 2)),
        float(sum(1 for size in sizes if size == 3)),
        float(sum(1 for size in sizes if size == 4)),
        float(sum(1 for size in sizes if size == 5)),
        float(sum(1 for size in sizes if size >= 6)),
    ]


def triplet_features(
    triplet: Triplet,
    edge_to_max_ids: Dict[Edge, List[int]],
    maximal_simplices: Sequence[Simplex],
) -> Tuple[List[float], bool]:
    a, b, c = triplet
    edge_ids = [
        edge_to_max_ids.get((a, b), []),
        edge_to_max_ids.get((a, c), []),
        edge_to_max_ids.get((b, c), []),
    ]

    features: List[float] = []
    for ids in edge_ids:
        features.extend(edge_stats(ids, maximal_simplices))

    sets = [set(ids) for ids in edge_ids]
    union_ids = set().union(*sets)
    if union_ids:
        union_sizes = [len(maximal_simplices[max_id]) for max_id in union_ids]
        union_count = float(len(union_ids))
        union_max_size = float(max(union_sizes))
        union_mean_size = float(sum(union_sizes) / len(union_sizes))
    else:
        union_count = 0.0
        union_max_size = 0.0
        union_mean_size = 0.0

    shared_ids = sets[0] & sets[1] & sets[2]
    features.extend(
        [
            union_count,
            union_max_size,
            union_mean_size,
            float(len(shared_ids)),
        ]
    )
    return features, bool(union_ids)


def header() -> List[str]:
    names = ["triplet"]
    for edge_name in ("edge_ab", "edge_ac", "edge_bc"):
        names.extend(f"{edge_name}_{name}" for name in EDGE_FEATURE_NAMES)
    names.extend(
        [
            "tri_union_count",
            "tri_union_max_size",
            "tri_union_mean_size",
            "tri_all_edges_shared_count",
        ]
    )
    return names


def main() -> None:
    args = parse_args()
    records = load_timestamped_simplices(args.raw_dir)
    context_records = select_context(records, args.context_ratio)
    context_simplices = [record[2] for record in context_records]
    maximal_simplices = extract_maximal_simplices(context_simplices)
    closure_count = count_closure_simplices(maximal_simplices)
    edge_to_max_ids = build_edge_index(maximal_simplices)
    triplets = read_gpickle_triplets(args.gpickle_path)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    missing = 0
    with args.output_txt.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header())
        for triplet in triplets:
            features, has_match = triplet_features(triplet, edge_to_max_ids, maximal_simplices)
            if has_match:
                matched += 1
            else:
                missing += 1
            writer.writerow(["|".join(str(v) for v in triplet)] + [f"{v:.10g}" for v in features])

    print(f"raw simplices: {len(records)}")
    print(f"context_ratio: {args.context_ratio}")
    print(f"historical simplices: {len(context_records)}")
    print(f"closure simplices: {closure_count}")
    print(f"maximal simplices: {len(maximal_simplices)}")
    print(f"edge index edges: {len(edge_to_max_ids)}")
    print(f"gpickle triplets: {len(triplets)}")
    print(f"matched triplets: {matched}")
    print(f"missing triplets: {missing}")
    print(f"wrote: {args.output_txt}")


if __name__ == "__main__":
    main()
