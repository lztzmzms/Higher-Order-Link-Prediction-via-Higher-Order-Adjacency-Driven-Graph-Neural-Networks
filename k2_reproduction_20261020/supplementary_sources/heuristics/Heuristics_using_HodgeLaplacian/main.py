import scipy.sparse as sp
from root_dir import ROOT_DIR
from utils import *
from random import sample
from tqdm import tqdm
import os
import pickle
import os.path as osp


#input parameters
dataset_name = 'email-Enron'

Saving_folder = os.path.join(ROOT_DIR, 'Heuristics_using_HodgeLaplacian', 'dataset', f'{dataset_name}')
simplex_tables_old_dir = os.path.join(Saving_folder, 'simplex_tables_old.pkl')
id_maps_dir = os.path.join(Saving_folder, 'id_maps.pkl')
simplex_tables_new_dir = os.path.join(Saving_folder, 'simplex_tables_new.pkl')

with open(simplex_tables_old_dir, 'rb') as f:
    simplex_tables_old = pickle.load(f)
with open(simplex_tables_new_dir, 'rb') as f:
    simplex_tables_new = pickle.load(f)
with open(id_maps_dir, 'rb') as f:
    id_maps = pickle.load(f)

lower_adj_dir = os.path.join(Saving_folder, 'lower_adj.npz')
Degree_dir = os.path.join(Saving_folder, 'Degree.npz')
Degree_inverse_dir = os.path.join(Saving_folder, 'Degree_inverse.npz')
Degree_log_inverse_dir = os.path.join(Saving_folder, 'Degree_log_inverse.npz')

lower_adj = sp.load_npz(lower_adj_dir)
Degree = sp.load_npz(Degree_dir)
Degree_inverse = sp.load_npz(Degree_inverse_dir)
Degree_log_inverse = sp.load_npz(Degree_log_inverse_dir)

Degree = Degree.toarray()

# get testing data

ratio = 1     #选择用于测试的开三角形的比例
tri_all, tri_all_set = Enumerating_open_triangles(simplex_tables_old)
positive_sample, positive_sample_idmap, negative_sample, negative_sample_idmap = get_positive_negative_Triangles(openTriangles=tri_all, simplex_tables_old=simplex_tables_old, simplex_tables_new=simplex_tables_new,
                                                             id_maps=id_maps, openTrianles_set=tri_all_set)

testing_triangles = sample(tri_all, k=int(ratio*len(tri_all)))
print(len(testing_triangles))

#构建分数矩阵
def choosing_sim(adj, method='Simplicial_CN'):
    if method == 'Simplicial_CN':
        Matrix = adj @ adj
    elif method == 'Local Path':
        Simplicial_CN = adj @ adj
        CN_3 = adj @ adj @ adj
        Matrix = Simplicial_CN + 0.001 * CN_3
    elif method == 'katz':
        temp = sp.eye(adj.shape[0], format='csr') - 0.01*adj
        Matrix = sp.linalg.inv(temp) - sp.eye(adj.shape[0], format='csr')
    elif method == 'RA':
        Matrix = adj @ Degree_inverse @ adj
    elif method == 'Global RA':
        temp = sp.eye(adj.shape[0], format='csr') - 0.01 * Degree_inverse @ adj
        Matrix = adj @ sp.linalg.inv(temp) - adj
    elif method == 'AA':
        Matrix = adj @ Degree_log_inverse @ adj
    elif method == 'Global AA':
        temp = sp.eye(adj.shape[0], format='csr') - 0.01 * Degree_log_inverse @ adj
        Matrix = adj @ sp.linalg.inv(temp) - adj
    elif method == 'PA':
        diag = Degree.diagonal()
        diag = csr_matrix(diag)
        Matrix = diag.transpose() @ diag
    elif method == 'Degree':
        Matrix = Degree

    return Matrix

# 计算eval
def get_evaluation(method):
    Score_Matrix = choosing_sim(adj=lower_adj, method=method)
    print('计算eval')
    number_of_positive = 0
    y_true = []
    y_score = []
    for tri in tqdm(testing_triangles):
        tri = tuple(tri)
        if tri in positive_sample:
            number_of_positive = number_of_positive + 1
            index = positive_sample.index(tri)
            ids = positive_sample_idmap[index]
            y_true.append(1)
            position = itertools.combinations(ids, 2)
            score = 0
            if method == 'Degree':
                for id in ids:
                    score = score + Score_Matrix[(id, id)]
            else:
                for x in position:
                    score = score + Score_Matrix[x]
            y_score.append(score)
        if tri in negative_sample:
            index = negative_sample.index(tri)
            ids = negative_sample_idmap[index]
            y_true.append(0)
            position = itertools.combinations(ids, 2)
            score = 0
            if method == 'Degree':
                for id in ids:
                    score = score + Score_Matrix[(id, id)]
            else:
                for x in position:
                    score = score + Score_Matrix[x]
            y_score.append(score)

    print(number_of_positive)
    random_Baseline = float(number_of_positive / len(testing_triangles))
    ap = calculate_ap(y_true, y_score)
    eval = evaluate(y_true, y_score, random_Baseline)

    result_folder = os.path.join(ROOT_DIR, 'Heuristics_using_HodgeLaplacian', 'results')
    result_folder = os.path.join(result_folder, f'{dataset_name}',f'{method}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')
    msg = (
        f'========== Result ============\n'
        f'Dataset:        {dataset_name}\n'
        f'------------ parameters -----------\n'
        f'method:{method}\n'
        f'random_Baseline:{random_Baseline}\n'
        f'AP (AUC-PR):{ap}\n'
        f'Lift:{eval}\n'
        f'Evaluation:{eval}\n'
        '-------------------------------\n\n')
    print(msg)
    with open(filename, 'a') as handle:
        handle.write(msg)

get_evaluation('Simplicial_CN')
get_evaluation('RA')
get_evaluation('Local Path')
get_evaluation('katz')
get_evaluation('AA')
get_evaluation('PA')
get_evaluation('Global AA')
get_evaluation('Global RA')
