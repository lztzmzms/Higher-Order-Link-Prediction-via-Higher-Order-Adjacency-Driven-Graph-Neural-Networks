"""Original one-hop extraction, TRNL labels and observed simplicial skeleton."""
import random
from itertools import combinations
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

def combinations_to_labels(distance_sum):
    offset = 0
    triples = []
    labels = []

    def generate_partitions(total, result, count, output):
        if total < 0:
            return
        if total == 0:
            part = result[:count]
            if len(part) == 3:
                output.append(part)
            return
        value = 1 if count == 0 else result[count - 1]
        while value <= total:
            result[count] = value
            generate_partitions(total - value, result, count + 1, output)
            value += 1

    for radius in range(3, distance_sum + 1):
        partitions = []
        generate_partitions(radius, [None] * radius, 0, partitions)
        for index, part in enumerate(partitions):
            triples.append(part)
            labels.append(1 + offset + index + 1)
        offset += len(partitions)
    return triples, labels

def remove_target_link(k_hop_subgraph, tri):
    graph = nx.Graph(k_hop_subgraph)
    graph.remove_edges_from(list(combinations(tri, 2)))
    return graph

def trnl_node_labeling(k_hop_subgraph, tri, disconnected_policy="skip"):
    node_1, node_2, node_3 = tuple(tri)
    graph = remove_target_link(k_hop_subgraph, tri)
    node_list = list(graph.nodes)
    index_1 = node_list.index(node_1)
    index_2 = node_list.index(node_2)
    index_3 = node_list.index(node_3)

    distance_maps = [
        nx.single_source_shortest_path_length(graph, source=source)
        for source in (node_1, node_2, node_3)
    ]
    if disconnected_policy == "skip" and any(
        len(distance_map) != len(node_list) for distance_map in distance_maps
    ):
        raise nx.NetworkXNoPath("target-removed local graph is disconnected")
    dist2node_1 = [distance_maps[0].get(node) for node in node_list]
    dist2node_2 = [distance_maps[1].get(node) for node in node_list]
    dist2node_3 = [distance_maps[2].get(node) for node in node_list]
    dist2node_1[index_2] = 0
    dist2node_1[index_3] = 0
    dist2node_2[index_1] = 0
    dist2node_2[index_3] = 0
    dist2node_3[index_1] = 0
    dist2node_3[index_2] = 0

    labels = [None] * len(node_list)
    for index in range(len(node_list)):
        if any(distance is None for distance in (
            dist2node_1[index], dist2node_2[index], dist2node_3[index]
        )):
            labels[index] = [0]
            continue
        distances = sorted([dist2node_1[index], dist2node_2[index], dist2node_3[index]])
        triples, trnl_labels = combinations_to_labels(sum(distances))
        labels[index] = [trnl_labels[triples.index(distances)]] if distances in triples else [0]
    labels[index_1] = [1]
    labels[index_2] = [1]
    labels[index_3] = [1]
    return labels

def extract_k_hop_subgraph(
    graph,
    num_hops,
    tri,
    sample_ratio=1.0,
    max_nodes_per_step=None,
    enforce_node_cap=False,
):
    visited = set(tri)
    current_level = set(tri)
    for _ in range(1, num_hops + 1):
        next_level = set()
        for node in current_level:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited and neighbor not in next_level:
                    next_level.add(neighbor)
        if sample_ratio < 1.0:
            next_level = set(random.sample(list(next_level), int(sample_ratio * len(next_level))))
        if max_nodes_per_step is not None and len(next_level) > max_nodes_per_step:
            next_level = set(random.sample(list(next_level), max_nodes_per_step))
        if not next_level:
            break
        if enforce_node_cap:
            visited.update(next_level)
        else:
            for node in current_level:
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
        current_level = next_level
    return nx.subgraph(graph, visited)

def construct_labeled_networkx_graph(
    k_hop_subgraph,
    old_simplices,
    tri,
    positives,
    labelling="TRNL",
    disconnected_policy="skip",
    label_graph=None,
    observed_2_skeleton=False,
):
    if labelling != "TRNL":
        raise ValueError(f"Only TRNL is supported by this runner, got {labelling}")
    node_list = list(k_hop_subgraph.nodes)
    edge_list = list(k_hop_subgraph.edges)


    embeddings = trnl_node_labeling(
        k_hop_subgraph if label_graph is None else label_graph,
        tri,
        disconnected_policy=disconnected_policy,
    )
    nx.set_node_attributes(
        k_hop_subgraph,
        {node_list[index]: {"feature": embeddings[index]} for index in range(len(embeddings))},
    )

    observed_simplices = set()
    node_set = set(node_list)
    for old_simplex in old_simplices:
        if len(old_simplex) >= 3:
            simplex = sorted(set(old_simplex) & node_set)
            if len(simplex) >= 3:
                observed_simplices.add(tuple(simplex))


    simplex_keys = {tuple(sorted(simplex)) for simplex in observed_simplices}
    simplex_keys.update(tuple(sorted(edge)) for edge in edge_list)
    if observed_2_skeleton:


        skeleton = set()
        for key in simplex_keys:
            vertices = tuple(key)
            for size in (2, 3):
                if len(vertices) >= size:
                    skeleton.update(combinations(vertices, size))
        simplices = [list(simplex) for simplex in sorted(skeleton, key=lambda value: (len(value), value))]
    else:
        maximal = []
        for key in sorted(simplex_keys, key=lambda value: (-len(value), value)):
            key_set = set(key)
            if not any(key_set < set(parent) for parent in maximal):
                maximal.append(key)
        simplices = [list(simplex) for simplex in sorted(maximal, key=lambda value: (len(value), value))]
    return nx.Graph(k_hop_subgraph, y=1 if tri in positives else 0, simplices=simplices)

def to_pyg_data(k_hop_subgraph):
    node_list = list(k_hop_subgraph.nodes)
    node_index = {node: index for index, node in enumerate(node_list)}
    node_feature = [k_hop_subgraph.nodes[node]["feature"] for node in node_list]
    directed = k_hop_subgraph.to_directed() if not nx.is_directed(k_hop_subgraph) else k_hop_subgraph
    graph = nx.convert_node_labels_to_integers(directed)
    nodes_new = list(graph.nodes)
    edges_new = list(graph.edges)
    edge_index = torch.tensor(edges_new, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_feature, dtype=torch.float)
    simplices = [list(simplex) for simplex in k_hop_subgraph.graph["simplices"]]
    for simplex in simplices:
        for i in range(len(simplex)):
            simplex[i] = nodes_new[node_index[simplex[i]]]
        simplex.sort()
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([k_hop_subgraph.graph["y"]], dtype=torch.long),
        node_ids=torch.tensor(nodes_new),
        original_node_ids=torch.tensor([int(node) for node in node_list], dtype=torch.long),
        num_nodes=len(node_list),
        simplices=simplices,
    )
