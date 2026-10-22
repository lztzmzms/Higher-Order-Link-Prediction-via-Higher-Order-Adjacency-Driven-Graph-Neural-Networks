import math
import os
import torch
import pickle
import numpy as np
from root_dir import ROOT_DIR
from glob import glob
from random import sample
import random

from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi
from simplicial_complex.dataset.datasets import InMemoryComplexDataset

labelling_method = 'TRNL'

def load_pyG_dataset(name, split: str='train', num_hops: int=1, filter='*'):
    graph_list = []
    file_path = os.path.join(ROOT_DIR, 'dataset', split, name + '/')
    for file in glob(file_path+filter+'{}_hops.pt'.format(num_hops)):
        graph = torch.load(file)
        graph_list.append(graph)
    return graph_list

class MyDataset(InMemoryComplexDataset):
    def __init__(self, root, name, num_hops=1, max_dim=2, num_classes=2, init_method='sum', split='train', seed=0, include_down_adj=False):
        self.name = name
        self.split = split
        self.num_hops = num_hops
        self.init_method = init_method
        # self.labelling_method = labelling_method
        super(MyDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
                                        init_method=init_method, include_down_adj=include_down_adj)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.seed = seed


    @property
    def raw_file_names(self):
        file_path = os.path.join(ROOT_DIR, 'dataset', self.split, self.name, f'{labelling_method}' + '/')
        # file_path = os.path.join(ROOT_DIR, 'dataset', self.split, self.name + '/')
        filename = '*{}_hops.pt'.format(self.num_hops)
        return file_path

    @property
    def processed_dir(self):
        file_path = os.path.join(ROOT_DIR, 'simplicial_complex', self.split, self.name, f'{labelling_method}')
        # file_path = os.path.join(ROOT_DIR, 'simplicial_complex', self.split, self.name)
        file_path += f"_down_adj" if self.include_down_adj else ""
        return file_path

    @property
    def processed_file_names(self):
        return ['{}_{}_hops.pt'.format(self.name, self.num_hops)]

    def process(self):
        graph_list = []
        file_path = self.raw_file_names
        print(file_path)
        graph_list_path = os.path.join(ROOT_DIR, 'simplicial_complex', self.split, f'{labelling_method}'+f'{self.num_hops}'+'{}_graph_list.pt'.format(self.name))
        # graph_list_path = os.path.join(ROOT_DIR, 'simplicial_complex', self.split,
        #                                '{}_graph_list.pt'.format(self.name))
        filter = '*'
        if not os.path.exists(graph_list_path):
            for file in glob(file_path + filter + '{}_hops.pt'.format(self.num_hops)):
                graph = torch.load(file)
                graph_list.append(graph)
            torch.save(graph_list, graph_list_path)
        graph_list = torch.load(graph_list_path,weights_only=False)
        print('Convert the dataset with gudhi.......')
        print(len(graph_list))
        if self.split == 'test':
            if len(graph_list) > 2000:
                list = sample(graph_list, 2000)
            else:
                list = graph_list
            complexes = convert_graph_dataset_with_gudhi(list, expansion_dim=self.max_dim,
                                                       include_down_adj=self.include_down_adj, init_method=self.init_method)
        if self.split == 'train':
            if len(graph_list) > 2000:
                list = sample(graph_list, 2000)
            else:
                list = graph_list
            complexes = convert_graph_dataset_with_gudhi(list, expansion_dim=self.max_dim,
                                                       include_down_adj=self.include_down_adj, init_method=self.init_method)
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])


