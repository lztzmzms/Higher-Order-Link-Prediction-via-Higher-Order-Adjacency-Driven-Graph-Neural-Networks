import math
from root_dir import ROOT_DIR

from utils import *
from dataset.get_timestamped import *
import os
import pickle
import scipy.sparse as sp

#input parameters
dataset_name = 'email-Enron'
root_nverts = f"../dataset/raw_data/{dataset_name}/{dataset_name}-nverts.txt"
root_timestamp = f"../dataset/raw_data/{dataset_name}/{dataset_name}-times.txt"
root_simplices = f"../dataset/raw_data/{dataset_name}/{dataset_name}-simplices.txt"
Saving_folder = os.path.join(ROOT_DIR, 'Heuristics_using_HodgeLaplacian', 'dataset', f'{dataset_name}')
if os.path.exists(Saving_folder):
    pass
else:
    os.mkdir(Saving_folder)

#get timestamped simplices
_, data_simplices = get_timestamped_simplices(root_nverts, root_timestamp, root_simplices)

#construct training dataset and save as .pt format
S_old = data_simplices[0: int(0.8*len(data_simplices))]
print(len(S_old))
S_new = data_simplices[int(0.8*len(data_simplices)): int(1*len(data_simplices))]
print(len(S_new))

st_old = pyg_to_simplex_tree(S_old)
st_new = pyg_to_simplex_tree(S_new)
complex_dim = st_old.dimension()

node_list_old = get_nodes(st_old)
node_list_new = get_nodes(st_new)

simplex_tables_old, id_maps = build_tables_ids_version2(st_old, node_list_old)
simplex_tables_new, _ = build_tables_ids_version2(st_new, node_list_new)

simplex_tables_old_dir = os.path.join(Saving_folder, 'simplex_tables_old.pkl')
id_maps_dir = os.path.join(Saving_folder, 'id_maps.pkl')
simplex_tables_new_dir = os.path.join(Saving_folder, 'simplex_tables_new.pkl')

with open(simplex_tables_old_dir, 'wb') as f:
    pickle.dump(simplex_tables_old, f)
with open(simplex_tables_new_dir, 'wb') as f:
    pickle.dump(simplex_tables_new, f)
with open(id_maps_dir, 'wb') as f:
    pickle.dump(id_maps, f)

coboundaries = extract_coboundaries_version2(st_old, complex_dim)
BM_index = build_adj_version2(coboundaries, id_maps)
B2_sp = construct_BM_version2(BM_index, simplex_tables_old, 2)

A = np.matmul(B2_sp, B2_sp.T)
dia = A.diagonal()
diag = []
for i in range(dia.shape[0]):
    diag.append(dia[i])

print(diag)

lower_adj = get_zero_dia(A)
lower_adj = sp.csc_matrix(lower_adj)
print(diag)

row = []
for i in range(len(diag)):
    row.append(i)

Degree = sp.csr_matrix((diag, (row, row)), shape=A.shape)
print(Degree)

diag_inverse = []
for element in diag:
    if element != 0:
        diag_inverse.append(1 / element)
    else:
        diag_inverse.append(element)
Degree_inverse = sp.csr_matrix((diag_inverse, (row, row)), shape=A.shape)


diag_log_inverse = []
for element in diag:
    if element == 1 or element == 0:
        diag_log_inverse.append(0)
    else:
        diag_log_inverse.append(1 / math.log(element))
Degree_log_inverse = sp.csr_matrix((diag_log_inverse, (row, row)), shape=A.shape)


lower_adj_dir = os.path.join(Saving_folder, 'lower_adj.npz')
Degree_dir = os.path.join(Saving_folder, 'Degree.npz')
Degree_inverse_dir = os.path.join(Saving_folder, 'Degree_inverse.npz')
Degree_log_inverse_dir = os.path.join(Saving_folder, 'Degree_log_inverse.npz')

sp.save_npz(lower_adj_dir, lower_adj)
sp.save_npz(Degree_dir, Degree)
sp.save_npz(Degree_inverse_dir, Degree_inverse)
sp.save_npz(Degree_log_inverse_dir, Degree_log_inverse)
