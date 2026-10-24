"""
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import torch
import logging
import copy

from torch import Tensor
try:
    from torch_sparse import SparseTensor
except ImportError:
    class SparseTensor:  # type: ignore[no-redef]
        pass
from mp.cell_mp import CochainMessagePassingParams
from torch_geometric.typing import Adj
from typing import List


class Cochain(object):
    pass                                                                                            

         
                                            
                                                                                
                                                                               
                                                                                   
                                                                               
                                                      
                                                                                               
                                                        
                                                                                                 
                                                          
                                                                                          
                                                      
                                                                                        
                                                                        
                                                                                        
                                                                        
                                                               
       
    def __init__(self, dim: int, x: Tensor = None, upper_index: Adj = None, lower_index: Adj = None,
                 shared_boundaries: Tensor = None, shared_coboundaries: Tensor = None, mapping: Tensor = None,
                 boundary_index: Adj = None, upper_orient=None, lower_orient=None, y=None, **kwargs):
        if dim == 0:
            assert lower_index is None
            assert shared_boundaries is None
            assert boundary_index is None

                                                                                    
                                          
        self.__dim__ = dim
                                   
        self.__x = x
        self.upper_index = upper_index
        self.lower_index = lower_index
        self.boundary_index = boundary_index
        self.y = y
        self.shared_boundaries = shared_boundaries
        self.shared_coboundaries = shared_coboundaries
        self.upper_orient = upper_orient
        self.lower_orient = lower_orient
        self.__oriented__ = False
        self.__hodge_laplacian__ = None
                                                   
        self.__mapping = mapping
        for key, item in kwargs.items():
            if key == 'num_cells':
                self.__num_cells__ = item
            elif key == 'num_cells_down':
                self.num_cells_down = item
            elif key == 'num_cells_up':
                self.num_cells_up = item
            else:
                self[key] = item

    @property
    def dim(self):
        pass                                                  

                                                                                          
           
        return self.__dim__
    
    @property
    def x(self):
        pass                                                                 
        return self.__x

    @x.setter
    def x(self, new_x):
        pass                                                              
        if new_x is None:
            logging.warning("Cochain features were set to None. ")
        else:
            assert self.num_cells == len(new_x)
        self.__x = new_x

    @property
    def keys(self):
        pass                                          
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __getitem__(self, key):
        pass                                            
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        pass                                                
        setattr(self, key, value)

    def __contains__(self, key):
        pass                                                                          
        return key in self.keys

    def __cat_dim__(self, key, value):
        pass                                                                 
                                                               
           
        if key in ['upper_index', 'lower_index', 'shared_boundaries',
                   'shared_coboundaries', 'boundary_index']:
            return -1
                                                             
        elif isinstance(value, SparseTensor):
            return (0, 1)
        return 0

    def __inc__(self, key, value):
        pass                                                                        
                                                                  
           
                                                                    
        if key in ['upper_index', 'lower_index']:
            inc = self.num_cells
        elif key in ['shared_boundaries']:
            inc = self.num_cells_down
        elif key == 'shared_coboundaries':
            inc = self.num_cells_up
        elif key == 'boundary_index':
            boundary_inc = self.num_cells_down if self.num_cells_down is not None else 0
            cell_inc = self.num_cells if self.num_cells is not None else 0
            inc = [[boundary_inc], [cell_inc]]
        else:
            inc = 0
        if inc is None:
            inc = 0

        return inc
    
    def __call__(self, *keys):
        pass                                                                          
                                          
                                                                        
                           
           
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @property
    def num_cells(self):
        pass                                             
        if hasattr(self, '__num_cells__'):
            return self.__num_cells__
        if self.x is not None:
            return self.x.size(self.__cat_dim__('x', self.x))
        if self.boundary_index is not None:
            return int(self.boundary_index[1,:].max()) + 1
        assert self.upper_index is None and self.lower_index is None
        return None

    @num_cells.setter
    def num_cells(self, num_cells):
        pass                                          
                                    
        self.__num_cells__ = num_cells

    @property
    def num_cells_up(self):
        pass                                                                                  
        if hasattr(self, '__num_cells_up__'):
            return self.__num_cells_up__
        elif self.shared_coboundaries is not None:
            assert self.upper_index is not None
            return int(self.shared_coboundaries.max()) + 1
        assert self.upper_index is None
        return 0

    @num_cells_up.setter
    def num_cells_up(self, num_cells_up):
        pass                                                                               
                                    
        self.__num_cells_up__ = num_cells_up

    @property
    def num_cells_down(self):
        pass                                                                                 
        if self.dim == 0:
            return None
        if hasattr(self, '__num_cells_down__'):
            return self.__num_cells_down__
        if self.lower_index is None:
            return 0
        raise ValueError('Cannot infer the number of cells in the cochain below.')

    @num_cells_down.setter
    def num_cells_down(self, num_cells_down):
        pass                                                                              
                                    
        self.__num_cells_down__ = num_cells_down
        
    @property
    def num_features(self):
        pass                                                         
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, SparseTensor):
                                                                           
                                                                              
            try:
                return func(item)
            except AttributeError:
                return item
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        pass                                                                     
                                                                                 
                                   
           
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        pass                                                                               
                                                                               
                                            
           
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        pass                                                                            
                         
                                                                                  
                       
           
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def clone(self):
        return self.__class__.from_dict({
            k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })
    
    @property
    def mapping(self):
        return self.__mapping


class CochainBatch(Cochain):
    pass                                               

                                                                                                    
                                                       
       

    def __init__(self, dim, batch=None, ptr=None, **kwargs):
        super(CochainBatch, self).__init__(dim, **kwargs)

        for key, item in kwargs.items():
            if key == 'num_cells':
                self.__num_cells__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Cochain
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_cells_list__ = None
        self.__num_cells_down_list__ = None
        self.__num_cells_up_list__ = None
        self.__num_cochains__ = None

    @classmethod
    def from_cochain_list(cls, data_list, follow_batch=[]):
        pass                                                                
                                     
                                                                     
                                                                          
                                
           
        keys = list(set.union(*[set(data.keys) for data in data_list]))
        assert 'batch' not in keys and 'ptr' not in keys

        batch = cls(data_list[0].dim)
        for key in data_list[0].__dict__.keys():
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None

        batch.__num_cochains__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch']:
            batch[key] = []
        batch['ptr'] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_cells_list = []
        num_cells_up_list = []
        num_cells_down_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                if item is not None:
                                                        
                    cum = cumsum[key][-1]
                    if isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, SparseTensor):
                        value = item.storage.value()
                        if value is not None and value.dtype != torch.bool:
                            if not isinstance(cum, int) or cum != 0:
                                value = value + cum
                            item = item.set_value(value, layout='coo')
                    elif isinstance(item, (int, float)):
                        item = item + cum

                                                                   
                    if isinstance(item, Tensor) and item.dim() == 0:
                        item = item.unsqueeze(0)

                    batch[key].append(item)

                                                             
                    size = 1
                    cat_dim = data.__cat_dim__(key, data[key])
                    cat_dims[key] = cat_dim
                    if isinstance(item, Tensor):
                        size = item.size(cat_dim)
                        device = item.device
                    elif isinstance(item, SparseTensor):
                        size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                        device = item.device()
                    
                                                                                          
                    slices[key].append(size + slices[key][-1])
                    
                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f'{key}_{j}_batch'
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(
                                    torch.full((size, ), i, dtype=torch.long,
                                               device=device))
                        else:
                            tmp = f'{key}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))
                    
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

            if hasattr(data, '__num_cells__'):
                num_cells_list.append(data.__num_cells__)
            else:
                num_cells_list.append(None)

            if hasattr(data, '__num_cells_up__'):
                num_cells_up_list.append(data.__num_cells_up__)
            else:
                num_cells_up_list.append(None)

            if hasattr(data, '__num_cells_down__'):
                num_cells_down_list.append(data.__num_cells_down__)
            else:
                num_cells_down_list.append(None)

            num_cells = data.num_cells
            if num_cells is not None:
                item = torch.full((num_cells, ), i, dtype=torch.long,
                                  device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_cells)

                                   
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_cells_list__ = num_cells_list
        batch.__num_cells_up_list__ = num_cells_up_list
        batch.__num_cells_down_list__ = num_cells_down_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        return batch.contiguous()

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super(CochainBatch, self).__getitem__(idx)
        elif isinstance(idx, int):
                                                               
                                         
            raise NotImplementedError
        else:
                                                                
                                           
            raise NotImplementedError

    def to_cochain_list(self) -> List[Cochain]:
        pass                                                                   
                              
                                                                             
                                                               
                                                               
                                                                       
        raise NotImplementedError


    @property
    def num_cochains(self) -> int:
        pass                                              
        if self.__num_cochains__ is not None:
            return self.__num_cochains__
        return self.ptr.numel() + 1


class Complex(object):
    pass                                                                      

         
                                                                
                                                                                             
                                                
       
    def __init__(self, *cochains: Cochain, y: torch.Tensor = None, dimension: int = None):
        if len(cochains) == 0:
            raise ValueError('At least one cochain is required.')
        if dimension is None:
            dimension = len(cochains) - 1
        if len(cochains) < dimension + 1:
            raise ValueError(f'Not enough cochains passed, '
                             f'expected {dimension + 1}, received {len(cochains)}')

        self.dimension = dimension
        self.cochains = {i: cochains[i] for i in range(dimension + 1)}
        self.nodes = cochains[0]
        self.edges = cochains[1] if dimension >= 1 else None
        self.two_cells = cochains[2] if dimension >= 2 else None

        self.y = y
        
        self._consolidate()
        return
    
    def _consolidate(self):
        for dim in range(self.dimension+1):
            cochain = self.cochains[dim]
            assert cochain.dim == dim
            if dim < self.dimension:
                upper_cochain = self.cochains[dim + 1]
                num_cells_up = upper_cochain.num_cells
                assert num_cells_up is not None
                if 'num_cells_up' in cochain:
                    assert cochain.num_cells_up == num_cells_up
                else:
                    cochain.num_cells_up = num_cells_up
            if dim > 0:
                lower_cochain = self.cochains[dim - 1]
                num_cells_down = lower_cochain.num_cells
                assert num_cells_down is not None
                if 'num_cells_down' in cochain:
                    assert cochain.num_cells_down == num_cells_down
                else:
                    cochain.num_cells_down = num_cells_down
                    
    def to(self, device, **kwargs):
        pass                                                                                 
                                                                                      
        for dim in range(self.dimension + 1):
            self.cochains[dim] = self.cochains[dim].to(device, **kwargs)
        if self.y is not None:
            self.y = self.y.to(device, **kwargs)
        return self

    def get_cochain_params(self,
                           dim : int,
                           max_dim : int=2,
                           include_top_features=True,
                           include_down_features=True,
                           include_boundary_features=True) -> CochainMessagePassingParams:
        pass                                                                                    
                                                          

             
                                                                   
                                                       
                                                                           
                                                                                           
                                                                          
                                                                            
                
                                                         
           
        if dim in self.cochains:
            cells = self.cochains[dim]
            x = cells.x
                             
            upper_index, upper_features = None, None
            upper_weight = None
                                                                                                      
                                                                                            
                                                                               
            if cells.upper_index is not None and (dim+1) in self.cochains:
                upper_index = cells.upper_index
                if self.cochains[dim + 1].x is not None and (dim < max_dim or include_top_features):
                    upper_features = torch.index_select(self.cochains[dim + 1].x, 0,
                                                        self.cochains[dim].shared_coboundaries)
                upper_weight = getattr(cells, "upper_weight", None)

                               
            lower_index, lower_features = None, None
            if include_down_features and cells.lower_index is not None:
                lower_index = cells.lower_index
                if dim > 0 and self.cochains[dim - 1].x is not None:
                    lower_features = torch.index_select(self.cochains[dim - 1].x, 0,
                                                        self.cochains[dim].shared_boundaries)
                                   
            boundary_index, boundary_features = None, None
            if include_boundary_features and cells.boundary_index is not None:
                boundary_index = cells.boundary_index
                if dim > 0 and self.cochains[dim - 1].x is not None:
                    boundary_features = self.cochains[dim - 1].x

            inputs = CochainMessagePassingParams(x, upper_index, lower_index,
                                               up_attr=upper_features, down_attr=lower_features,
                                               up_weight=upper_weight,
                                               boundary_attr=boundary_features, boundary_index=boundary_index)
        else:
            raise NotImplementedError(
                'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return inputs

    def get_all_cochain_params(self,
                               max_dim:int=2,
                               include_top_features=True,
                               include_down_features=True,
                               include_boundary_features=True) -> List[CochainMessagePassingParams]:
        pass                                                                                 

             
                                                                                              
                                                                                       
                                                                                
                                                                                   
                
                                                                   
           
        all_params = []
        return_dim = min(max_dim, self.dimension)
        for dim in range(return_dim+1):
            all_params.append(self.get_cochain_params(dim, max_dim=max_dim,
                                                    include_top_features=include_top_features,
                                                    include_down_features=include_down_features,
                                                    include_boundary_features=include_boundary_features))
        return all_params

    def get_labels(self, dim=None):
        pass                     

                                                                                               
                                                                     
           
        if dim is None:
            y = self.y
        else:
            if dim in self.cochains:
                y = self.cochains[dim].y
            else:
                raise NotImplementedError(
                    'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return y

    def set_xs(self, xs: List[Tensor]):
        pass                                                             
        assert (self.dimension + 1) >= len(xs)
        for i, x in enumerate(xs):
            self.cochains[i].x = x
            
    @property
    def keys(self):
        pass                                          
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys
    
    def __getitem__(self, key):
        pass                                            
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        pass                                                
        setattr(self, key, value)
    
    def __contains__(self, key):
        pass                                                                          
        return key in self.keys


class ComplexBatch(Complex):
    pass                                               

                                                                          

         
                                                                                        
                                                          
                                                             
                                                            
       
    def __init__(self,
                 *cochains: CochainBatch,
                 dimension: int,
                 y: torch.Tensor = None,
                 num_complexes: int = None):
        super(ComplexBatch, self).__init__(*cochains, y=y)
        self.num_complexes = num_complexes
        self.dimension = dimension

    @classmethod
    def from_complex_list(cls, data_list: List[Complex], follow_batch=[], max_dim: int = 2):
        pass                                                  

             
                                                                         
                                                                          
                                    
                                                                                          
                
                                  
           

        dimension = max([data.dimension for data in data_list])
        dimension = min(dimension, max_dim)
        cochains = [list() for _ in range(dimension + 1)]
        label_list = list()
        per_complex_labels = True
        for comp in data_list:
            for dim in range(dimension+1):
                if dim not in comp.cochains:
                                                                                                  
                    cochains[dim].append(Cochain(dim=dim))
                    if dim-1 in comp.cochains:
                                                                                                  
                                                                                                        
                        cochains[dim][-1].num_cells_down = comp.cochains[dim - 1].num_cells
                else:
                    cochains[dim].append(comp.cochains[dim])
            per_complex_labels &= comp.y is not None
            if per_complex_labels:
                label_list.append(comp.y)

        batched_cochains = [CochainBatch.from_cochain_list(cochain_list, follow_batch=follow_batch)
                          for cochain_list in cochains]
        y = None if not per_complex_labels else torch.cat(label_list, 0)
        batch = cls(*batched_cochains, y=y, num_complexes=len(data_list), dimension=dimension)

        return batch
