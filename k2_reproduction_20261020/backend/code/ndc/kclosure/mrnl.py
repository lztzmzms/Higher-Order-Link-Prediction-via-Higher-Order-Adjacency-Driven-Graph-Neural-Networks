"""Generalized multiple-role node labeling (MRNL/TRNL-compatible)."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx


def _positive_partitions(total: int, length: int, minimum: int = 1):
    if length == 0:
        if total == 0:
            yield ()
        return
    for value in range(minimum, total + 1):
        for suffix in _positive_partitions(total - value, length - 1, value):
            yield (value, *suffix)


def distance_vector_label(vector: Sequence[int]) -> int:
    """Match the legacy k=2 ordering and extend it to any target arity."""
    vector = tuple(sorted(int(value) for value in vector))
    length = len(vector)
    radius = sum(vector)
    offset = 0
    for previous in range(length, radius):
        offset += sum(1 for _ in _positive_partitions(previous, length))
    partitions = list(_positive_partitions(radius, length))
    return 2 + offset + partitions.index(vector) if vector in partitions else 0


def remove_target_edges(graph: nx.Graph, target: Sequence[int]) -> nx.Graph:
    result = nx.Graph(graph)
    result.remove_edges_from(combinations(tuple(target), 2))
    return result


def generalized_mrnl_node_labeling(graph: nx.Graph, target: Sequence[int]):
    """Label all target nodes as 1 and other nodes by sorted target distances."""
    target = tuple(sorted(target))
    if len(target) not in (3, 4):
        raise ValueError("MRNL smoke implementation supports 3 or 4 target nodes")
    work = remove_target_edges(graph, target)
    nodes = list(work.nodes)
    distance_maps: Dict[int, Dict[int, int]] = {
        source: nx.single_source_shortest_path_length(work, source) for source in target
    }
    labels: List[List[int]] = []
    vectors: List[Tuple[int, ...]] = []
    for node in nodes:
        if node in target:
            labels.append([1])
            vectors.append(tuple(0 if other != node else 0 for other in target))
            continue
        if any(node not in distance_maps[source] for source in target):
            raise nx.NetworkXNoPath("local enclosing graph is disconnected after target-edge removal")
        vector = tuple(sorted(distance_maps[source][node] for source in target))
        if len(vector) != len(target):
            raise AssertionError("MRNL distance vector arity mismatch")
        vectors.append(vector)
        labels.append([distance_vector_label(vector)])
    return nodes, labels, vectors
