"""
Based on https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/message_passing.py

MIT License

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

from inspect import Parameter
from typing import List, Optional, Set
from torch_geometric.typing import Adj, Size

import torch
from torch import Tensor
try:
    from torch_sparse import SparseTensor
except ImportError:
    class SparseTensor:  # type: ignore[no-redef]
        pass
from torch_scatter import gather_csr, scatter, segment_csr


from mp.cell_mp_inspector import CellularInspector

def expand_left(src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(0)
    return src
class CochainMessagePassing(torch.nn.Module):
    pass                                                                                                  

                                                   
                                                   
                                          

         
                                                                                                   
                                                                               
                                 
                                                                                       
                                                                       
                                   
                                                                                         
                                                                       
                                   
                                                                                                 
                                                                      
                                                                     
                                                
                                                                    
                                
                                                                                             
                           
                                                                                     
                                                                                            
       

    special_args: Set[str] = {
        'up_index', 'up_adj_t', 'up_index_i', 'up_index_j', 'up_size',
        'up_size_i', 'up_size_j', 'up_ptr', 'agg_up_index', 'up_dim_size',

        'down_index', 'down_adj_t', 'down_index_i', 'down_index_j', 'down_size',
        'down_size_i', 'down_size_j', 'down_ptr', 'agg_down_index', 'down_dim_size',

        'boundary_index', 'boundary_adj_t', 'boundary_index_i', 'boundary_index_j', 'boundary_size',
        'boundary_size_i', 'boundary_size_j', 'boundary_ptr', 'agg_boundary_index', 'boundary_dim_size',
    }

    def __init__(self,
                 up_msg_size,
                 down_msg_size,
                 aggr_up: Optional[str] = "add",
                 aggr_down: Optional[str] = "add",
                 aggr_boundary: Optional[str] = "add",
                 flow: str = "source_to_target",
                 node_dim: int = -2,
                 boundary_msg_size=None,
                 use_down_msg=True,
                 use_boundary_msg=True):

        super(CochainMessagePassing, self).__init__()

        self.up_msg_size = up_msg_size
        self.down_msg_size = down_msg_size
        self.use_boundary_msg = use_boundary_msg
        self.use_down_msg = use_down_msg
                                                                                    
        self.boundary_msg_size = down_msg_size if boundary_msg_size is None else boundary_msg_size
        self.aggr_up = aggr_up
        self.aggr_down = aggr_down
        self.aggr_boundary = aggr_boundary
        assert self.aggr_up in ['add', 'mean', 'max', None]
        assert self.aggr_down in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

                                                                            
                                                                       
        self.node_dim = node_dim

        self.inspector = CellularInspector(self)
                                                                             
                                                                         
                                                                                                
        self.inspector.inspect(self.message_up)
        self.inspector.inspect(self.message_down)
        self.inspector.inspect(self.message_boundary)
        self.inspector.inspect(self.aggregate_up, pop_first_n=1)
        self.inspector.inspect(self.aggregate_down, pop_first_n=1)
        self.inspector.inspect(self.aggregate_boundary, pop_first_n=1)
        self.inspector.inspect(self.message_and_aggregate_up, pop_first_n=1)
        self.inspector.inspect(self.message_and_aggregate_down, pop_first_n=1)
        self.inspector.inspect(self.message_and_aggregate_boundary, pop_first_n=1)
        self.inspector.inspect(self.update, pop_first_n=3)

                                                                                             
                                                                                               
        self.__user_args__ = self.inspector.keys(
            ['message_up', 'message_down', 'message_boundary', 'aggregate_up',
             'aggregate_down', 'aggregate_boundary']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ['message_and_aggregate_up',
             'message_and_aggregate_down',
             'message_and_aggregate_boundary']).difference(self.special_args)
        self.__update_user_args__ = self.inspector.keys(
            ['update']).difference(self.special_args)

                                              
        self.fuse_up = self.inspector.implements('message_and_aggregate_up')
        self.fuse_down = self.inspector.implements('message_and_aggregate_down')
        self.fuse_boundary = self.inspector.implements('message_and_aggregate_boundary')

    def __check_input_together__(self, index_up, index_down, size_up, size_down):
                                                                            
        if (index_up is not None and index_down is not None
                and size_up is not None and size_down is not None):
            assert size_up[0] == size_down[0]
            assert size_up[1] == size_down[1]

    def __check_input_separately__(self, index, size):
        pass                                                                     
        the_size: List[Optional[int]] = [None, None]

        if isinstance(index, Tensor):
            assert index.dtype == torch.long
            assert index.dim() == 2
            assert index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow adjacency "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed sparse tensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = index.sparse_size(1)
            the_size[1] = index.sparse_size(0)
            return the_size

        elif index is None:
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, index, dim):
        if isinstance(index, Tensor):
            index = index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(index, SparseTensor):
            if dim == 1:
                rowptr = index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    def __collect__(self, args, index, size, adjacency, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        assert adjacency in ['up', 'down', 'boundary']

        out = {}
        for arg in args:
                                                                         
                                                   
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            elif index is not None:
                dim = 0 if arg[-2:] == '_j' else 1
                                                                       
                if adjacency == 'up' and arg.startswith('up_'):
                    data = kwargs.get(arg[3:-2], Parameter.empty)
                    size_data = data
                elif adjacency == 'down' and arg.startswith('down_'):
                    data = kwargs.get(arg[5:-2], Parameter.empty)
                    size_data = data
                elif adjacency == 'boundary' and arg.startswith('boundary_'):
                    if dim == 0:
                                                                                                            
                                                                                               
                        data = kwargs.get('boundary_attr', Parameter.empty)
                        size_data = kwargs.get(arg[9:-2], Parameter.empty)
                    else:
                        data = kwargs.get(arg[9:-2], Parameter.empty)
                        size_data = data
                else:
                    continue

                                                                                  
                                                                         
                                                          
                if isinstance(data, (tuple, list)):
                    raise ValueError('This format is not supported for cellular message passing')

                                                                                               
                if isinstance(data, Tensor):
                                                
                    self.__set_size__(size, dim, size_data)
                                                                                             
                    data = self.__lift__(data, index, j if arg[-2:] == '_j' else i)

                out[arg] = data

                                                                                              
                                                                                                   
        if isinstance(index, Tensor):
            out[f'{adjacency}_adj_t'] = None
            out[f'{adjacency}_ptr'] = None
            out[f'{adjacency}_index'] = index
            out[f'{adjacency}_index_i'] = index[i]
            out[f'{adjacency}_index_j'] = index[j]
        elif isinstance(index, SparseTensor):
            out['edge_index'] = None
            out[f'{adjacency}_adj_t'] = index
            out[f'{adjacency}_index_i'] = index.storage.row()
            out[f'{adjacency}_index_j'] = index.storage.col()
            out[f'{adjacency}_ptr'] = index.storage.rowptr()
            out[f'{adjacency}_weight'] = index.storage.value()
            out[f'{adjacency}_attr'] = index.storage.value()
            out[f'{adjacency}_type'] = index.storage.value()

                                                                                            
        if isinstance(index, Tensor) or isinstance(index, SparseTensor):
                                                                                    
            out[f'agg_{adjacency}_index'] = out[f'{adjacency}_index_i']

        out[f'{adjacency}_size'] = size
        out[f'{adjacency}_size_i'] = size[1] or size[0]
        out[f'{adjacency}_size_j'] = size[0] or size[1]
        out[f'{adjacency}_dim_size'] = out[f'{adjacency}_size_i']
        return out

    def get_msg_and_agg_func(self, adjacency):
        if adjacency == 'up':
            return self.message_and_aggregate_up
        if adjacency == 'down':
            return self.message_and_aggregate_down
        elif adjacency == 'boundary':
            return self.message_and_aggregate_boundary
        else:
            return None

    def get_msg_func(self, adjacency):
        if adjacency == 'up':
            return self.message_up
        elif adjacency == 'down':
            return self.message_down
        elif adjacency == 'boundary':
            return self.message_boundary
        else:
            return None

    def get_agg_func(self, adjacency):
        if adjacency == 'up':
            return self.aggregate_up
        elif adjacency == 'down':
            return self.aggregate_down
        elif adjacency == 'boundary':
            return self.aggregate_boundary
        else:
            return None

    def get_fuse_boolean(self, adjacency):
        if adjacency == 'up':
            return self.fuse_up
        elif adjacency == 'down':
            return self.fuse_down
        elif adjacency == 'boundary':
            return self.fuse_boundary
        else:
            return None

    def __message_and_aggregate__(self, index: Adj,
                                  adjacency: str,
                                  size: List[Optional[int]] = None,
                                  **kwargs):
        assert adjacency in ['up', 'down', 'boundary']

                                       
        fuse = self.get_fuse_boolean(adjacency)
        if isinstance(index, SparseTensor) and fuse:
                                                                               
            coll_dict = self.__collect__(self.__fused_user_args__, index, size, adjacency, kwargs)

                                                                    
            msg_aggr_kwargs = self.inspector.distribute(
                f'message_and_aggregate_{adjacency}', coll_dict)
            message_and_aggregate = self.get_msg_and_agg_func(adjacency)
            return message_and_aggregate(index, **msg_aggr_kwargs)

                                                               
        elif isinstance(index, Tensor) or not fuse:
                                                                               
            coll_dict = self.__collect__(self.__user_args__, index, size, adjacency, kwargs)

                                        
            msg_kwargs = self.inspector.distribute(f'message_{adjacency}', coll_dict)
            message = self.get_msg_func(adjacency)
            out = message(**msg_kwargs)
            
                                         
            aggr_kwargs = self.inspector.distribute(f'aggregate_{adjacency}', coll_dict)
            aggregate = self.get_agg_func(adjacency)
            return aggregate(out, **aggr_kwargs)

    def propagate(self, up_index: Optional[Adj],
                  down_index: Optional[Adj],
                  boundary_index: Optional[Adj],                                        
                  up_size: Size = None,
                  down_size: Size = None,
                  boundary_size: Size = None,
                  **kwargs):
        pass                                                 
        up_size = self.__check_input_separately__(up_index, up_size)
        down_size = self.__check_input_separately__(down_index, down_size)
        boundary_size = self.__check_input_separately__(boundary_index, boundary_size)
        self.__check_input_together__(up_index, down_index, up_size, down_size)

        up_out, down_out = None, None
                                      
        if up_index is not None:
            up_out = self.__message_and_aggregate__(up_index, 'up', up_size, **kwargs)

                                        
        if self.use_down_msg and down_index is not None:
            down_out = self.__message_and_aggregate__(down_index, 'down', down_size, **kwargs)

                                            
        boundary_out = None
        if self.use_boundary_msg and 'boundary_attr' in kwargs and kwargs['boundary_attr'] is not None:
            boundary_out = self.__message_and_aggregate__(boundary_index, 'boundary', boundary_size, **kwargs)

        coll_dict = {}
        up_coll_dict = self.__collect__(self.__update_user_args__, up_index, up_size, 'up',
                                        kwargs)
        down_coll_dict = self.__collect__(self.__update_user_args__,
                                          down_index, down_size, 'down', kwargs)
        coll_dict.update(up_coll_dict)
        coll_dict.update(down_coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update(up_out, down_out, boundary_out, **update_kwargs)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        pass                                                                                
                                                                                         
                                                                                                   
                                                                                 
                                                                                        
                                                                           
           
        return up_x_j

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        pass                                                                                
                                                                                           
                                                                                                   
                                                                                 
                                                                                        
                                                                           
           
        return down_x_j

    def message_boundary(self, boundary_x_j: Tensor):
        pass                                                                                   
                                                                                               
                                                                                                   
                                                                                 
                                                                          
           
        return boundary_x_j

    def aggregate_up(self, inputs: Tensor, agg_up_index: Tensor,
                     up_ptr: Optional[Tensor] = None,
                     up_dim_size: Optional[int] = None) -> Tensor:
        pass                                              

                                                                            
                                                                 

                                                                             
                                                                       
                                                     
           
        if up_ptr is not None:
            up_ptr = expand_left(up_ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, up_ptr, reduce=self.aggr_up)
        else:
            return scatter(inputs, agg_up_index, dim=self.node_dim, dim_size=up_dim_size,
                           reduce=self.aggr_up)

    def aggregate_down(self, inputs: Tensor, agg_down_index: Tensor,
                       down_ptr: Optional[Tensor] = None,
                       down_dim_size: Optional[int] = None) -> Tensor:
        pass                                              

                                                                            
                                                                 

                                                                             
                                                                       
                                                     
           
        if down_ptr is not None:
            down_ptr = expand_left(down_ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, down_ptr, reduce=self.aggr_down)
        else:
            return scatter(inputs, agg_down_index, dim=self.node_dim, dim_size=down_dim_size,
                           reduce=self.aggr_down)

    def aggregate_boundary(self, inputs: Tensor, agg_boundary_index: Tensor,
                       boundary_ptr: Optional[Tensor] = None,
                       boundary_dim_size: Optional[int] = None) -> Tensor:
        pass                                            

                                                                            
                                                                 

                                                                             
                                                                       
                                                     
           
                                     
        if boundary_ptr is not None:
            down_ptr = expand_left(boundary_ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, down_ptr, reduce=self.aggr_boundary)
        else:
            return scatter(inputs, agg_boundary_index, dim=self.node_dim, dim_size=boundary_dim_size,
                           reduce=self.aggr_boundary)

    def message_and_aggregate_up(self, up_adj_t: SparseTensor) -> Tensor:
        pass                                                                        
                        
                                                                            
                                           
                                                                         
                                                                            
           
        raise NotImplementedError

    def message_and_aggregate_down(self, down_adj_t: SparseTensor) -> Tensor:
        pass                                                                            
                        
                                                                            
                                           
                                                                         
                                                                            
           
        raise NotImplementedError

    def message_and_aggregate_boundary(self, boundary_adj_t: SparseTensor) -> Tensor:
        pass                                                                                    
                        
                                                                            
                                           
                                                                         
                                                                            
           
        raise NotImplementedError

    def update(self, up_inputs: Optional[Tensor], down_inputs: Optional[Tensor],
               boundary_inputs: Optional[Tensor], x: Tensor) -> (Tensor, Tensor, Tensor):
        pass                                                                               
                                                                                               
                          
           
        if up_inputs is None:
            up_inputs = torch.zeros(x.size(0), self.up_msg_size).to(device=x.device)
        if down_inputs is None:
            down_inputs = torch.zeros(x.size(0), self.down_msg_size).to(device=x.device)
        if boundary_inputs is None:
            boundary_inputs = torch.zeros(x.size(0), self.boundary_msg_size).to(device=x.device)

        return up_inputs, down_inputs, boundary_inputs


class CochainMessagePassingParams:
    pass                                                                              

                                                                                                 
                                                                                        
                                                                                      
         
                                                                               
                                                                     
                                                                       
       
    def __init__(self, x: Tensor, up_index: Adj = None, down_index: Adj = None, **kwargs):
        self.x = x
        self.up_index = up_index
        self.down_index = down_index
        self.kwargs = kwargs
        if 'boundary_index' in self.kwargs:
            self.boundary_index = self.kwargs['boundary_index']
        else:
            self.boundary_index = None
        if 'boundary_attr' in self.kwargs:
            self.boundary_attr = self.kwargs['boundary_attr']
        else:
            self.boundary_attr = None
