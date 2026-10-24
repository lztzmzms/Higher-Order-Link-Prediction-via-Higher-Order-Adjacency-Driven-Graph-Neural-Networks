import os
import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
                                       
string_classes = str
import collections.abc as container_abcs

from root_dir import ROOT_DIR
from simplicial_complex.complex import Complex, Cochain, ComplexBatch, CochainBatch

int_classes = int

class Collater(object):
    pass                                                                               

         
                                                                                
                                                                                         
       
    def __init__(self, follow_batch, max_dim=2):
        self.follow_batch = follow_batch
        self.max_dim = max_dim

    def collate(self, batch):
        pass                                                   
        elem = batch[0]
        if isinstance(elem, Cochain):
            return CochainBatch.from_cochain_list(batch, self.follow_batch)
        elif isinstance(elem, Complex):
            return ComplexBatch.from_complex_list(batch, self.follow_batch, max_dim=self.max_dim)
        elif isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

class DataLoader(torch.utils.data.DataLoader):
    pass                                                                

         
                                                                   
                                                                       
                               
                                                                         
                                                              
                                                                        
                                                                  
                                                                                   
                        
       
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 max_dim=2, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

                                       
        self.follow_batch = follow_batch

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch, max_dim), **kwargs)

def load_dataset(name, split='train', num_hops=1, max_dim=2, root=os.path.join(ROOT_DIR, 'simplicial_complex'), init_method='sum'):             
    from simplicial_complex.dataset.generate_dataset import load_pyG_dataset, MyDataset
    if name.startswith('email-Enron'):
        print('****')
        print(os.path.join(root, split, name))
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split,)

    elif name == 'coauth-MAG-History':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'coauth-DBLP':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'coauth-MAG-Geology':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'congress-bills':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'contact-high-school':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'contact-primary-school':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'DAWN':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'email-Eu':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'NDC-classes':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'NDC-substances':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'tags-math-sx':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'tags-ask-ubuntu':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'tags-stack-overflow':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'threads-ask-ubuntu':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'threads-math-sx':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'threads-stack-overflow':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    elif name == 'Email-Eu':
        dataset = MyDataset(os.path.join(root, split, name), name, num_hops=num_hops, max_dim=max_dim,
                            init_method=init_method, split=split)
    else:
        raise NotImplementedError(name)
    return dataset

def load_graph_dataset(name, split='train', num_hops=1):
    if name.startswith('email-Enron'):
        graph_list = load_pyG_dataset(name, split=split, num_hops=num_hops, filter='*')
        data = (graph_list, 2)
    elif name.startwith('coauth-MAG-History'):
        graph_list = load_pyG_dataset(name, split=split, num_hops=num_hops)
        data = (graph_list, 2)
    elif name.startwith('NDC-classes'):
        graph_list = load_pyG_dataset(name, split=split, num_hops=num_hops)
        data = (graph_list, 2)
    else:
        raise NotImplementedError
    return data
