                                                                                  

import os

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'

import csv, hashlib, json, random, sys, time

from collections import defaultdict

from itertools import combinations

from pathlib import Path

import numpy as np

import networkx as nx

import torch

from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.preprocessing import StandardScaler

import dual_mapping_from_raw as dual

import exp.run_all_sparse_sin as sin

from mp.models import SparseSIN

from simplicial_complex.data_loading import DataLoader

from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / 'input'

OUT = SOURCE / 'neural_pilot'

DS = 'tags-math-sx'

sys.path.insert(0, str(SOURCE / 'code'))

import easy_search as sample

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_num_threads(1)

def atomic(name, obj):
    sample.s.atomic(OUT / name, obj)

def log(message):
    print(message, flush=True)
    atomic('STATUS.json', {'message':message, 'utc':sample.s.utc()})

def seed(value=41):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def dual_assets(events, tasks):
    for name,t in tasks.items():
        path=OUT/'cache'/f'dual_{name}.npz'
        if path.exists():
            with np.load(path) as z: t['dual']={k:z[k] for k in z.files}
            continue
        log(f'Dual-Mapping features: {name}')
        history=sample.s.History(events[:t['cut']])
        state={'edges':{e for a,ns in history.adj.items() for e in ((a,b) for b in ns if b>a)},
               'triangles':history.closed}
        q=list(map(tuple,t['q']))
        graph=dual.build_dual_graph(q)
        features,raw=dual.build_features(q,state,graph,return_raw=True)
        rw=dual.build_random_walk_structural_features(graph['edge_index'],len(q),rw_steps=2)
        t['dual']={'local':dual.select_feature_block(features,dual.FEATURE_NAMES,dual.local_feature_names()),
                   'heuristic':dual.build_heuristic_feature_block(raw), 'rw':rw,
                   'edge_index':graph['edge_index']}
        np.savez_compressed(path,**t['dual'])

def sin_assets(events,tasks):
    for name,t in tasks.items():
        path=OUT/'cache'/f'sin_{name}.pt'
        if path.exists():
            t['complexes']=torch.load(path,map_location='cpu',weights_only=False);continue
        log(f'MLSHoL local complexes: {name}')
        history=sample.s.History(events[:t['cut']])
        graph=nx.Graph();graph.add_nodes_from(history.adj)
        graph.add_edges_from((a,b) for a,ns in history.adj.items() for b in ns if b>a)
        bynode=defaultdict(set)
        for _,_,vs in events[:t['cut']]:
            if len(vs)>=3:
                for v in vs:bynode[v].add(vs)
        pyg=[]
        positives=set(map(tuple,t['q'][t['y'].astype(bool)]))
        for i,tri in enumerate(map(tuple,t['q'])):
            random.seed(2026+i)
            local=sin.extract_k_hop_subgraph(graph,1,tri,max_nodes_per_step=30,enforce_node_cap=True).copy()
            nodes=set(local.nodes);old=set()
            for v in nodes:
                for sx in bynode[v]:
                    if sum(n in nodes for n in sx)>=3:old.add(sx)
            removed=sin.remove_target_link(local,tri)
            labeled=sin.construct_labeled_networkx_graph(removed,old,tri,positives,
                        disconnected_policy='unreachable_zero',observed_2_skeleton=True)
            pyg.append(sin.to_pyg_data(labeled))
            if i%500==0:log(f'MLSHoL {name} local graphs {i}/{len(t["q"])}')
        complexes=convert_graph_dataset_with_gudhi(pyg,expansion_dim=2,include_down_adj=False,
                                                 init_method='sum',expand_cliques=False)
        assert len(complexes)==len(t['q'])
        assert np.array_equal([int(c.y.item()) for c in complexes],t['y'])
        torch.save(complexes,path);t['complexes']=complexes

