import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import BatchNorm1d as BN, LayerNorm as LN, Identity

def get_nonlinearity(nonlinearity, return_module=True):
    if nonlinearity == 'relu':
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == 'elu':
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == 'sigmoid':
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == 'tanh':
        module = torch.nn.Tanh
        function = F.tanh
    else:
        raise NotImplementedError('Nonlinearity {} is not currently supported.'.format(nonlinearity))
    if return_module:
        return module
    return function

def get_pooling_fn(readout):
    if readout == 'sum':
        return global_add_pool
    elif readout == 'mean':
        return global_mean_pool
    else:
        raise NotImplementedError('Readout {} is not currently supported.'.format(readout))

def get_graph_norm(norm):
    if norm == 'bn':
        return BN
    elif norm == 'ln':
        return LN
    elif norm == 'id':
        return Identity
    else:
        raise ValueError('Graph Normalisation {} not currently supported'.format(norm))

def pool_complex(xs, data, max_dim, readout_type):
    pooling_fn = get_pooling_fn(readout_type)
    batch_size = data.cochains[0].batch.max() + 1
    pooled_xs = torch.zeros(max_dim+1, batch_size, xs[0].size(-1), device=batch_size.device)
    for i in range(len(xs)):
        pooled_xs[i, :, :] = pooling_fn(data.cochains[i].batch, size=batch_size)
    return pooled_xs

