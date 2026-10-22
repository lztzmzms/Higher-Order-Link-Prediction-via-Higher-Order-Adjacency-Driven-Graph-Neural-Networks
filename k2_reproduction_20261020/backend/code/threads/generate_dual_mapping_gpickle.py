# -*- coding: utf-8 -*-
"""Build Dual-Mapping triangle-node graphs for node classification.

The Dual-Mapping process follows the paper's Triangle-to-Node idea:
each 2-simplex candidate (triangle, open or closed) is mapped to one
node in a derived graph, and two derived nodes are connected when their
source triangles share a common 1-face (an original graph edge).

The generated NetworkX gpickle files are compatible with
node_classfication-emmbedding.py: every node has "triplet", "label",
and "features" attributes.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Set, Tuple

import networkx as nx
import numpy as np


Triangle = Tuple[int, int, int]
Face = Tuple[int, int]


def parse_triangle_line(line: str) -> Triangle | None:
    parts = line.strip().replace(",", " ").split()
    if len(parts) < 3:
        return None
    try:
        tri = tuple(sorted(int(x) for x in parts[:3]))
    except ValueError:
        return None
    if len(set(tri)) != 3:
        return None
    return tri  # type: ignore[return-value]


def read_triangles(path: Path) -> List[Triangle]:
    triangles: List[Triangle] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            tri = parse_triangle_line(line)
            if tri is not None:
                triangles.append(tri)
    return triangles


def triangle_faces(tri: Triangle) -> Tuple[Face, Face, Face]:
    a, b, c = tri
    return ((a, b), (a, c), (b, c))


def add_source(
    records: MutableMapping[Triangle, Set[str]],
    triangles: Iterable[Triangle],
    source: str,
) -> None:
    for tri in triangles:
        records.setdefault(tri, set()).add(source)


def build_records(
    close_path: Path,
    open_path: Path,
    new_closures_path: Path,
    include_new_closures_as_nodes: bool,
) -> Tuple[OrderedDict[Triangle, Set[str]], Dict[str, int]]:
    close_triangles = read_triangles(close_path)
    open_triangles = read_triangles(open_path)
    new_closures = read_triangles(new_closures_path)

    records: OrderedDict[Triangle, Set[str]] = OrderedDict()
    add_source(records, close_triangles, "closed")
    add_source(records, open_triangles, "open")
    if include_new_closures_as_nodes:
        add_source(records, new_closures, "new_closure")
    else:
        new_set = set(new_closures)
        for tri in records:
            if tri in new_set:
                records[tri].add("new_closure")

    stats = {
        "close_lines": len(close_triangles),
        "close_unique": len(set(close_triangles)),
        "open_lines": len(open_triangles),
        "open_unique": len(set(open_triangles)),
        "new_closure_lines": len(new_closures),
        "new_closure_unique": len(set(new_closures)),
        "nodes_after_union": len(records),
    }
    return records, stats


def build_face_index(triangles: Sequence[Triangle]) -> Dict[Face, List[int]]:
    face_to_nodes: Dict[Face, List[int]] = defaultdict(list)
    for node_id, tri in enumerate(triangles):
        for face in triangle_faces(tri):
            face_to_nodes[face].append(node_id)
    return face_to_nodes


def build_lower_adjacency_edges(face_to_nodes: Dict[Face, List[int]]) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for node_ids in face_to_nodes.values():
        if len(node_ids) < 2:
            continue
        for u, v in itertools.combinations(node_ids, 2):
            edges.add((u, v) if u < v else (v, u))
    return edges


def jaccard(a: Set[int], b: Set[int]) -> float:
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b)) / float(len(union))


def make_features(
    tri: Triangle,
    node_id: int,
    face_to_nodes: Dict[Face, List[int]],
    feature_mode: str,
) -> np.ndarray:
    if feature_mode == "ids":
        return np.asarray(tri, dtype=np.float32)

    faces = triangle_faces(tri)
    face_sets = [set(face_to_nodes[face]) for face in faces]
    neighbor_sets = [nodes - {node_id} for nodes in face_sets]
    lower_neighbors = set().union(*neighbor_sets)

    face_support = [len(nodes) for nodes in face_sets]
    face_neighbor_counts = [len(nodes) for nodes in neighbor_sets]
    face_jaccards = [
        jaccard(neighbor_sets[0], neighbor_sets[1]),
        jaccard(neighbor_sets[0], neighbor_sets[2]),
        jaccard(neighbor_sets[1], neighbor_sets[2]),
    ]
    common_lower_pair_sum = (
        len(neighbor_sets[0] & neighbor_sets[1])
        + len(neighbor_sets[0] & neighbor_sets[2])
        + len(neighbor_sets[1] & neighbor_sets[2])
    )

    values = [
        float(tri[0]),
        float(tri[1]),
        float(tri[2]),
        float(face_support[0]),
        float(face_support[1]),
        float(face_support[2]),
        float(face_neighbor_counts[0]),
        float(face_neighbor_counts[1]),
        float(face_neighbor_counts[2]),
        float(len(lower_neighbors)),
        float(face_jaccards[0]),
        float(face_jaccards[1]),
        float(face_jaccards[2]),
        float(common_lower_pair_sum),
    ]
    return np.asarray(values, dtype=np.float32)


def label_for_sources(sources: Set[str], closed_label: int) -> int:
    if "closed" in sources:
        return closed_label
    if "new_closure" in sources:
        return 1
    return 0


def build_dual_mapping_graph(
    close_path: Path,
    open_path: Path,
    new_closures_path: Path,
    split_name: str,
    feature_mode: str = "local",
    closed_label: int = 1,
    include_new_closures_as_nodes: bool = True,
) -> Tuple[nx.Graph, Dict[str, int]]:
    records, stats = build_records(
        close_path=close_path,
        open_path=open_path,
        new_closures_path=new_closures_path,
        include_new_closures_as_nodes=include_new_closures_as_nodes,
    )
    triangles = list(records.keys())
    face_to_nodes = build_face_index(triangles)
    derived_edges = build_lower_adjacency_edges(face_to_nodes)

    graph = nx.Graph()
    for node_id, tri in enumerate(triangles):
        sources = records[tri]
        label = label_for_sources(sources, closed_label=closed_label)
        graph.add_node(
            node_id,
            triplet=tri,
            label=int(label),
            features=make_features(tri, node_id, face_to_nodes, feature_mode),
            sources=tuple(sorted(sources)),
        )
    graph.add_edges_from(derived_edges)

    labels = [data["label"] for _, data in graph.nodes(data=True)]
    stats.update(
        {
            "split": split_name,
            "derived_nodes": graph.number_of_nodes(),
            "derived_edges": graph.number_of_edges(),
            "label_0": labels.count(0),
            "label_1": labels.count(1),
            "feature_dim": int(next(iter(graph.nodes(data=True)))[1]["features"].shape[0])
            if graph.number_of_nodes()
            else 0,
            "lower_faces": len(face_to_nodes),
            "max_triangles_per_face": max((len(v) for v in face_to_nodes.values()), default=0),
        }
    )
    graph.graph.update(
        {
            "dual_mapping": True,
            "split": split_name,
            "edge_rule": "lower_adjacency_shared_1_face",
            "label_rule": "1 for closed triangles and future closures; 0 for remaining open triangles",
            "feature_mode": feature_mode,
            "feature_names": (
                ["tri_a", "tri_b", "tri_c"]
                if feature_mode == "ids"
                else [
                    "tri_a",
                    "tri_b",
                    "tri_c",
                    "support_ab",
                    "support_ac",
                    "support_bc",
                    "neighbors_ab",
                    "neighbors_ac",
                    "neighbors_bc",
                    "lower_degree",
                    "jaccard_ab_ac",
                    "jaccard_ab_bc",
                    "jaccard_ac_bc",
                    "common_lower_pair_sum",
                ]
            ),
            "stats": dict(stats),
        }
    )
    return graph, stats


def write_gpickle(graph: nx.Graph, path: Path, overwrite: bool, backup_existing: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        if backup_existing:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup_path = path.with_suffix(path.suffix + f".bak_{stamp}")
            shutil.copy2(path, backup_path)
            print(f"Backed up existing file: {backup_path}")
    with path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def default_path(root: Path, split: str, dataset: str, kind: str) -> Path:
    return root / "dataset" / split / f"{dataset}_{kind}_{split}.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dual-Mapping gpickle graphs for contact-high-school triangle closure prediction."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", default="contact-high-school")
    parser.add_argument("--feature-mode", choices=["local", "ids"], default="local")
    parser.add_argument("--closed-label", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--no-include-new-closures-as-nodes",
        action="store_true",
        help="Only mark new_closures that are already present in open/close files.",
    )
    parser.add_argument("--train-output", type=Path, default=Path("classified_network_contact-high-school_train_1.gpickle"))
    parser.add_argument("--test-output", type=Path, default=Path("classified_network_contact-high-school_all_1.gpickle"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-backup-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    include_new = not args.no_include_new_closures_as_nodes

    jobs = [
        (
            "train",
            default_path(root, "train", args.dataset, "closeTriangles"),
            default_path(root, "train", args.dataset, "openTriangles"),
            default_path(root, "train", args.dataset, "new_closures"),
            (root / args.train_output).resolve() if not args.train_output.is_absolute() else args.train_output,
        ),
        (
            "test",
            default_path(root, "test", args.dataset, "closeTriangles"),
            default_path(root, "test", args.dataset, "openTriangles"),
            default_path(root, "test", args.dataset, "new_closures"),
            (root / args.test_output).resolve() if not args.test_output.is_absolute() else args.test_output,
        ),
    ]

    for split_name, close_path, open_path, new_path, output_path in jobs:
        for path in (close_path, open_path, new_path):
            if not path.exists():
                raise FileNotFoundError(path)
        graph, stats = build_dual_mapping_graph(
            close_path=close_path,
            open_path=open_path,
            new_closures_path=new_path,
            split_name=split_name,
            feature_mode=args.feature_mode,
            closed_label=args.closed_label,
            include_new_closures_as_nodes=include_new,
        )
        write_gpickle(
            graph,
            output_path,
            overwrite=args.overwrite,
            backup_existing=not args.no_backup_existing,
        )
        print(
            f"{split_name}: wrote {output_path} | "
            f"nodes={stats['derived_nodes']} edges={stats['derived_edges']} "
            f"label0={stats['label_0']} label1={stats['label_1']} "
            f"feature_dim={stats['feature_dim']}"
        )


if __name__ == "__main__":
    main()
