import numpy as np
import torch
import gudhi as gd
import itertools
import networkx as nx

from tqdm import tqdm
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from joblib import delayed
from simplicial_complex.complex import Cochain, Complex

def pyg_to_simplex_tree(simplices):
    pass

    st = gd.SimplexTree()
    for simplex in simplices:
        st.insert(simplex)

    return st

def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]

def build_tables(simplex_tree, size: int):
    complex_dim = simplex_tree.dimension()
    id_maps = [{} for _ in range(complex_dim+1)]
    simplex_tables = [[] for _ in range(complex_dim+1)]
    boundaries_tables = [[] for _ in range(complex_dim + 1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps

def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int):
    pass
    boundaries = [{} for _ in range(complex_dim+2)]
    coboundaries = [{} for _ in range(complex_dim+2)]
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    for simplex, _ in simplex_tree.get_simplices():
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim-1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        simplex_coboundaries = simplex_tree.get_cofaces(simplex, 1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries

def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    pass
    def initialise_structure():
        return [[] for _ in range(complex_dim+1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    for dim in range(complex_dim+1):
        for simplex, id in id_maps[dim].items():

            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):

                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]

                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])

                    all_shared_coboundaries[dim - 1].extend([id, id])

            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):

                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]

                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])

                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes

def construct_features(vx: Tensor, simplex_tables, init_method: str) -> List:
    pass

    features = [vx]
    for dim in range(1, len(simplex_tables)):
        aux_1 = []
        aux_0 = []
        for c, simplex in enumerate(simplex_tables[dim]):
            aux_1 += [c for _ in range(len(simplex))]
            aux_0 += simplex
        node_simplex_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_simplex_index[0])
        features.append(scatter(in_features, node_simplex_index[1], dim=0,
                                dim_size=len(simplex_tables[dim]), reduce=init_method))
    return features

def extract_labels(y):
    complex_y = y
    return complex_y

def generate_cochain(dim, x, all_upper_index, all_lower_index, all_shared_boundaries,
                     all_shared_coboundaries, simplex_tables, boundaries_tables,
                     complex_dim, y=None):
    pass
    if dim == 0:
        assert len(all_lower_index[dim]) == 0
        assert len(all_shared_boundaries[dim]) == 0

    num_simplices_down = len(simplex_tables[dim-1]) if dim > 0 else None
    num_simplices_up = len(simplex_tables[dim+1]) if dim < complex_dim else None

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t()
                if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t()
                 if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                           if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                         if len(all_shared_boundaries[dim]) > 0 else None)

    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, simplex in enumerate(boundaries_tables[dim]):
            for boundary in simplex:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)

    if num_simplices_down is None:
        assert shared_boundaries is None
    if num_simplices_up == 0:
        assert shared_coboundaries is None

    if up_index is not None:
        assert up_index.size(1) == shared_coboundaries.size(0)
        assert num_simplices_up == shared_coboundaries.max() + 1
    if down_index is not None:
        assert down_index.size(1) == shared_boundaries.size(0)
        assert num_simplices_down >= shared_boundaries.max() + 1

    return Cochain(dim=dim, x=x, upper_index=up_index, lower_index=down_index,
                   shared_boundaries=shared_boundaries, shared_coboundaries=shared_coboundaries,
                   y=y, num_simplices_down=num_simplices_down, num_simplices_up=num_simplices_up,
                   boundary_index=boundary_index)

def compute_clique_complex_with_gudhi(x:Tensor, edge_index: Adj, size: int, simplices: list,
                                      expansion_dim: int = 2, y: Tensor = None,
                                      include_down_adj=True, init_method: str = 'sum',
                                      expand_cliques: bool = True):
    pass

    assert x is not None
    assert isinstance(edge_index, Tensor)

    simplex_tree = pyg_to_simplex_tree(simplices=simplices)

    if expand_cliques:
        simplex_tree.expansion(expansion_dim)
    complex_dim = simplex_tree.dimension()

    simplex_tables, id_maps = build_tables(simplex_tree, size)

    boundaries_tables, boundaries, coboundaries = (
        extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim))

    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, coboundaries, id_maps,
                                                                             complex_dim, include_down_adj)
    xs = construct_features(x, simplex_tables, init_method)
    complex_y = extract_labels(y)

    cochains = []
    for i in range(complex_dim+1):
        cochain = generate_cochain(dim=i, x=xs[i], all_upper_index=upper_idx, all_lower_index=lower_idx,
                                    all_shared_boundaries=shared_boundaries, all_shared_coboundaries=shared_coboundaries,
                                    simplex_tables=simplex_tables, boundaries_tables=boundaries_tables,
                                    complex_dim=complex_dim, y=None)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim)

def convert_graph_dataset_with_gudhi(dataset, expansion_dim: int, include_down_adj=True,
                                     init_method: str = 'sum', expand_cliques: bool = True):
    dimension = -1
    complexes = []
    num_features = [None for _ in range(expansion_dim+1)]

    for data in tqdm(dataset):
        complex = compute_clique_complex_with_gudhi(data.x, data.edge_index, data.num_nodes,
                                                    expansion_dim=expansion_dim, y=data.y, simplices=data.simplices,
                                                    include_down_adj=include_down_adj, init_method=init_method,
                                                    expand_cliques=expand_cliques)
        if hasattr(data, "original_node_ids"):
            complex.cochains[0].original_node_ids = data.original_node_ids.clone().long()

        complexes.append(complex)

    return complexes

