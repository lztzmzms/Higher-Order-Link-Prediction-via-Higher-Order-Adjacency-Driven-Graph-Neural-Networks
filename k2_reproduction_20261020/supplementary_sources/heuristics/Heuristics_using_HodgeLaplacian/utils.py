import itertools
import torch
import numpy as np
import torch_geometric
from scipy.sparse import coo_matrix, csr_matrix
from sklearn import metrics as met
import gudhi as gd


def pyg_to_simplex_tree(simplices):
    """construct a simplex tree from a PyG Graph,
    simplices:simplex saved in data."""
    st = gd.SimplexTree()
    for simplex in simplices:
        st.insert(simplex)

    return st

def get_nodes(st):
    node_list = []
    for vertices, _ in st.get_skeleton(0):
        node_list.append(vertices)

    return node_list

def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]

def build_tables_ids(st, node_list):
    complex_dim = st.dimension()
    id_maps = [{} for _ in range(complex_dim+1)] # 创建一个空的列表，里面有维度加一个{}
    simplex_tables = [[] for _ in range(complex_dim+1)]

    simplex_tables[0] = [node for node in node_list]  # 0-simplex
    id_maps[0] = {tuple(node): node_list.index(node) for node in node_list}  # give an id to every 0-simplex
    for simplex, _ in st.get_simplices():  # 1维以上的数目统计
        dim = len(simplex) - 1
        if dim == 0:
            continue
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps
def build_tables_ids_version2(st, node_list):
    max_dim = 2
    id_maps = [{} for _ in range(max_dim+1)] # 创建一个空的列表，里面有维度加一个{}
    simplex_tables = [[] for _ in range(max_dim+1)]

    simplex_tables[0] = [node for node in node_list]  # 0-simplex
    id_maps[0] = {tuple(node): node_list.index(node) for node in node_list}  # give an id to every 0-simplex
    for simplex, _ in st.get_skeleton(max_dim):
        dim = len(simplex) - 1
        if dim == 0:
            continue
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)
    return simplex_tables, id_maps

def extract_boundaries_and_coboundaries(st, id_maps, complex_dim: int):
    boundaries = [{} for _ in range(complex_dim + 2)]
    coboundaries = [{} for _ in range(complex_dim + 2)]
    boundaries_tables = [[] for _ in range(complex_dim + 1)]
    for simplex, _ in st.get_simplices():
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        if simplex_dim > 0:  # 1-simplex, 2-simplex, ...
            boundaries_ids = [id_maps[simplex_dim - 1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        simplex_coboundaries = st.get_cofaces(simplex, codimension=1)  # codimension: 比simplex高codimension维的意思,不是coface的维度
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1
            if tuple(simplex) not in level_coboundaries:  # 判断，然后创建空列表/直接append tuple
                level_coboundaries[tuple(simplex)] = list()  # Dict: List
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))  # dict:list[tuple]

            if tuple(coboundary) not in level_boundaries:  # 得到coboundary的boundary节点
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries

def extract_coboundaries_version2(st, complex_dim =2):

    coboundaries = [{} for _ in range(complex_dim + 2)]

    for simplex, _ in st.get_skeleton(complex_dim):
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        if simplex_dim == 1:
            simplex_coboundaries = st.get_cofaces(simplex, codimension=1)  # codimension: 比simplex高codimension维的意思,不是coface的维度
            for coboundary, _ in simplex_coboundaries:
                assert len(coboundary) == len(simplex) + 1
                if tuple(simplex) not in level_coboundaries:  # 判断，然后创建空列表/直接append tuple
                    level_coboundaries[tuple(simplex)] = list()  # Dict: List
                level_coboundaries[tuple(simplex)].append(tuple(coboundary))  # dict:list[tuple]

    return coboundaries


def build_adj(boundaries, coboundaries, id_maps, complex_dim: int):
    def initialise_structure():
        return [[] for _ in range(complex_dim + 1)]

    BM_index = initialise_structure()
    for dim in range(complex_dim + 1):
        for simplex, id in id_maps[dim].items():  # key, value
            if dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary in coboundaries[dim][simplex]:
                    id_co = id_maps[dim + 1][coboundary]
                    BM_index[dim + 1].extend([[id, id_co]])

    return BM_index

def build_adj_version2(coboundaries, id_maps, complex_dim = 2):
    def initialise_structure():
        return [[] for _ in range(complex_dim + 1)]

    BM_index = initialise_structure()

    for simplex, id in id_maps[complex_dim-1].items():  # key, value
        if complex_dim-1 < complex_dim and simplex in coboundaries[complex_dim-1]:
            for coboundary in coboundaries[complex_dim-1][simplex]:
                id_co = id_maps[complex_dim][coboundary]
                BM_index[complex_dim].extend([[id, id_co]])

    return BM_index

def construct_BM(BM_index, simplex_tables, dim):
    up_index = (torch.tensor(BM_index[dim], dtype=torch.long).t()
                if len(BM_index[dim]) > 0 else None)

    A = torch_geometric.utils.to_scipy_sparse_matrix(up_index).toarray()
    if A.shape[0] >= len(simplex_tables[dim - 1]):
        temp_1 = np.delete(A, slice(len(simplex_tables[dim - 1]), A.shape[0]), 0)
        if temp_1.shape[1] >= len(simplex_tables[dim]):
            temp_2 = np.delete(temp_1, slice(len(simplex_tables[dim]), temp_1.shape[1]), 1)
        elif temp_1.shape[1] < len(simplex_tables[dim]):
            zero_arr = np.zeros((temp_1.shape[0], len(simplex_tables[dim])-temp_1.shape[1]))
            temp_2 = np.insert(temp_1, temp_1.shape[1], zero_arr, axis=1)
    elif A.shape[0] < len(simplex_tables[dim - 1]):
        zero_arr = np.zeros((len(simplex_tables[dim-1]) - A.shape[0], A.shape[1]))
        temp_1 = np.insert(A, A.shape[0], zero_arr, axis=0)
        if temp_1.shape[1] >= len(simplex_tables[dim]):
            temp_2 = np.delete(temp_1, slice(len(simplex_tables[dim]), temp_1.shape[1]), 1)
        elif temp_1.shape[1] < len(simplex_tables[dim]):
            zero_arr = np.zeros((temp_1.shape[0], len(simplex_tables[dim])-temp_1.shape[1]))
            temp_2 = np.insert(temp_1, temp_1.shape[1], zero_arr, axis=1)

    boundary_matrix = np.matrix(temp_2)
    boundary_matrix_T = boundary_matrix.T

    return boundary_matrix, boundary_matrix_T

def construct_BM_version2(BM_index, simplex_tables, dim):
    up_index = (torch.tensor(BM_index[dim], dtype=torch.long).t()
                if len(BM_index[dim]) > 0 else None)
    row = up_index[0]
    col = up_index[1]
    data = [1.0] * len(row)
    sp_matrix = coo_matrix((data, (row, col)), shape=(len(simplex_tables[1]), len(simplex_tables[2]))).toarray()

    return sp_matrix

def construct_BM_version_CSR(BM_index, simplex_tables, dim):
    up_index = (torch.tensor(BM_index[dim], dtype=torch.long).t()
                if len(BM_index[dim]) > 0 else None)
    row = up_index[0]
    col = up_index[1]
    data = [1.0] * len(row)
    sp_matrix = csr_matrix((data, (row, col)), shape=(len(simplex_tables[1]), len(simplex_tables[2])))

    return sp_matrix

def get_zero_dia(matrix):
    np.fill_diagonal(matrix, 0)
    return matrix

def Enumerating_open_triangles(simplex_tables):
    openTriangles = []
    temp = []
    closedTriangles = simplex_tables[2]
    for [u, w] in simplex_tables[1]:
        for [v] in simplex_tables[0]:
            edge_1 = [v, u]
            edge_1.sort()
            edge_2 = [v, w]
            edge_2.sort()
            if edge_1 in simplex_tables[1] and edge_2 in simplex_tables[1]:
                L = list(set(edge_1) | set(edge_2))
                L.sort()
                if L not in closedTriangles:
                    T = tuple(L)
                    temp.append(T)
    temp = set(temp)
    for tri in temp:
        tri = list(tri)
        openTriangles.append(tri)
    return openTriangles, temp

def get_positive_negative_Triangles(openTriangles, openTrianles_set, simplex_tables_old, simplex_tables_new, id_maps):
    closedTriangles_old = simplex_tables_old[2]
    closedTriangles_new = simplex_tables_new[2]
    positive_sample = []
    for L in closedTriangles_new:
        if L not in closedTriangles_old:
            if L in openTriangles:
                positive_sample.append(tuple(L))
    positive_sample = set(positive_sample)
    negative_sample = openTrianles_set - positive_sample
    positive_sample_idmap = []
    negative_sample_idmap = []
    for i, temp in enumerate(positive_sample):
        B = get_simplex_boundaries(temp)
        id_1, id_2, id_3 = id_maps[1][B[0]], id_maps[1][B[1]], id_maps[1][B[2]]
        positive_sample_idmap.append([id_1, id_2, id_3])
    for i, temp in enumerate(negative_sample):
        A = get_simplex_boundaries(temp)
        id_1, id_2, id_3 = id_maps[1][A[0]], id_maps[1][A[1]], id_maps[1][A[2]]
        negative_sample_idmap.append([id_1, id_2, id_3])

    return list(positive_sample), positive_sample_idmap, list(negative_sample), negative_sample_idmap

def evaluate(y_true, y_prob, random_Baseline):
    precision, recall, threshold = met.precision_recall_curve(y_true, y_prob)
    AUC_PR = met.auc(recall, precision)
    eval = float(AUC_PR / random_Baseline)
    return eval

def get_Degree_and_DegreeInv(matrix):
    diag = np.diag(matrix)
    Degree = np.matrix(np.diag(diag))
    diag_inverse = []
    for element in diag:
        if element != 0:
            diag_inverse.append(1 / element)
        else:
            diag_inverse.append(element)
    Degree_inverse = np.matrix(np.diag(diag_inverse))
    return Degree, Degree_inverse

def delete_naninf(matrix):
    where_are_nan = np.isnan(matrix)
    where_are_inf = np.isinf(matrix)
    matrix[where_are_nan] = 0
    matrix[where_are_inf] = 0
    return matrix

def read_txt(file_path):
    node_pair = []
    file = open(file_path, 'r')
    file_data = file.readlines()
    for element in file_data:
        temp = list(map(int, element.split()))
        node_pair.append(temp)
    return node_pair