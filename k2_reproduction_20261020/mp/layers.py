import torch
from typing import Callable, Optional
from torch import Tensor
from mp.cell_mp import CochainMessagePassing, CochainMessagePassingParams
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN, Identity
from simplicial_complex.complex import Cochain
from torch_scatter import scatter

class SparseCINCochainConv(CochainMessagePassing):
    pass
    def __init__(self, dim: int,
                 up_msg_size: int,
                 down_msg_size: int,
                 boundary_msg_size: Optional[int],
                 msg_up_nn: Callable,
                 msg_boundaries_nn: Callable,
                 update_up_nn: Callable,
                 update_boundaries_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 train_eps: bool = False):
        super(SparseCINCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                 use_down_msg=False)
        self.dim = dim
        self.msg_up_nn = msg_up_nn
        self.msg_boundaries_nn = msg_boundaries_nn
        self.update_up_nn = update_up_nn
        self.update_boundaries_nn = update_boundaries_nn
        self.combine_nn = combine_nn
        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, _, out_boundaries = self.propagate(cochain.up_index, cochain.down_index,
                                              cochain.boundary_index, x=cochain.x,
                                              up_attr=cochain.kwargs['up_attr'],
                                              up_weight=cochain.kwargs.get('up_weight'),
                                              boundary_attr=cochain.kwargs['boundary_attr'])


        out_up += (1 + self.eps1) * cochain.x
        out_boundaries += (1 + self.eps2) * cochain.x
        out_up = self.update_up_nn(out_up)
        out_boundaries = self.update_boundaries_nn(out_boundaries)


        return self.combine_nn(torch.cat([out_up, out_boundaries], dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor, up_weight: Tensor = None) -> Tensor:
        message = self.msg_up_nn((up_x_j, up_attr))
        if up_weight is not None:
            message = message * up_weight.view(-1, 1).to(message.dtype)
        return message

    def message_boundary(self, boundary_x_j: Tensor) -> Tensor:
        return self.msg_boundaries_nn(boundary_x_j)

class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=-1)

class SparseCINConv(torch.nn.Module):
    pass


    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable], passed_msg_boundaries_nn: Optional[Callable],
                 passed_update_up_nn: Optional[Callable],
                 passed_update_boundaries_nn: Optional[Callable],
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_coboundaries=False, **kwargs):
        super(SparseCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    msg_up_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            combine_nn = Sequential(
                Linear(kwargs['hidden']*2, kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            mp = SparseCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps)
            self.mp_levels.append(mp)

    def reset_parameters(self):
        for mp in self.mp_levels:
            mp.reset_parameters()

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out
