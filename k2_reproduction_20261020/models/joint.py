"""Single-head H/structure neural mechanisms; no output score skip or score mixing.

The structural modules are the frozen-source original DualMappingGNN/SparseSIN
classes. Their unused classifier layers are disabled, while their encoders train.
H tokens are transformations of the same scalar H, never historical observations.
"""
import math
import types
import numpy as np
import torch
from torch import nn


class MonotonicH(nn.Module):
    """Nondecreasing basis in the scalar empirical H quantile coordinate.

    Only this branch is monotonic. The joint neural head has no such guarantee.
    """
    def __init__(self, hidden):
        super().__init__()
        self.raw_slopes = nn.Parameter(torch.full((hidden,), -.5))
        self.raw_weights = nn.Parameter(torch.full((hidden,), -1.))
        self.offset = nn.Parameter(torch.linspace(-2, 2, hidden))

    def forward(self, h):
        x = h[:, 2:3]
        return torch.nn.functional.softplus(x * torch.nn.functional.softplus(self.raw_slopes)
                                          + self.offset) * torch.nn.functional.softplus(self.raw_weights)


class JointNetwork(nn.Module):
    def __init__(self, api, model, mode, geometry, h_arrays, cfg, device):
        super().__init__()
        self.api, self.model_name, self.mode = api, model, mode
        self.cfg, self.device = cfg, device
        self.h_arrays = h_arrays
        self.geometry = geometry
        self.hidden = n = int(cfg.get('hidden', 32))
        self.h_encoder = MonotonicH(n) if mode == 'monotonic_joint' else nn.Sequential(
            nn.Linear(4, n), nn.SiLU(), nn.Linear(n, n), nn.SiLU())
        self.core = None
        self._context = None
        self.router_last = None
        if mode != 'h_only':
            if model == 'Dual-Mapping':
                b = geometry['fit']
                *_, Core = api.dual.make_model_classes()
                self.core = Core(b['local'].shape[1], b['heuristic'].shape[1], b['rw'].shape[1], 0,
                                 feature_proj_dim=16, hidden_dim=64, num_layers=int(cfg.get('depth', 3)),
                                 dropout=float(cfg.get('dropout', .05)), max_residual_alpha=1.)
                for p in self.core.base_head[-1].parameters(): p.requires_grad_(False)
                for p in self.core.residual_head.parameters(): p.requires_grad_(False)
                self.core.alpha_param.requires_grad_(False)
                sd = 128
                self.layer_film = nn.Linear(n, 128)
                self.edge_router = nn.Sequential(nn.Linear(2*n, n), nn.SiLU(), nn.Linear(n, 1))
            else:
                self.core = api.SparseSIN(num_input_features=1, num_classes=2,
                                         num_layers=int(cfg.get('depth', 3)), hidden=16,
                                         dropout_rate=float(cfg.get('dropout', .05)), max_dim=2,
                                         readout='mean', final_readout='sum')
                for p in self.core.lin2.parameters(): p.requires_grad_(False)
                sd = 32
                self.layer_film = nn.Linear(n, 32)
                self.relation_router = nn.Linear(n, 6)
                self._install_mls_hooks()
            self.s_encoder = nn.Sequential(nn.Linear(sd, n), nn.LayerNorm(n), nn.SiLU())
        else:
            self.s_encoder = nn.Identity()
        self.final_film = nn.Linear(n, 2*n)
        self.h_tokens = nn.ModuleList([nn.Linear(1, n) for _ in range(4)])
        self.s_tokens = nn.ModuleList([nn.Linear(n, n) for _ in range(3)])
        heads = int(cfg.get('mha_heads', 1))
        self.cross_h = nn.MultiheadAttention(n, heads, dropout=0., batch_first=True)
        self.cross_s = nn.MultiheadAttention(n, heads, dropout=0., batch_first=True)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(3*n, n), nn.SiLU(), nn.Linear(n, n)),
            nn.Sequential(nn.Linear(n, n), nn.SiLU(), nn.Linear(n, n)),
            nn.Sequential(nn.Linear(n, n), nn.SiLU(), nn.Linear(n, n))])
        self.router = nn.Linear(2*n, 3)
        # H, structure, and multiplicative interaction reach ONE nonlinear head.
        self.head = nn.Sequential(nn.Linear(3*n, 2*n), nn.SiLU(), nn.Linear(2*n, 1))
        # Small, nonzero structural influence avoids an initially destructive
        # random classifier without creating an output H-score bypass.
        with torch.no_grad():
            self.head[0].weight[:, n:] *= .05
        self.to(device)

    def _film(self, x, he, layer=False):
        pars = self.layer_film(he) if layer else self.final_film(he)
        gamma, beta = pars.chunk(2, dim=-1)
        return x * (1 + .1*torch.tanh(gamma)) + .1*beta

    def _install_mls_hooks(self):
        """H gates actual upper/boundary messages before aggregation.

        The original propagate and relation indices stay intact. Separate gates
        for each order/relation are indexed by the batched candidate identity.
        """
        for conv_i, conv in enumerate(self.core.convs):
            for dim, mp in enumerate(conv.mp_levels):
                original_up, original_boundary = mp.message_up, mp.message_boundary
                owner = self
                def up(module, up_x_j, up_attr, up_weight=None, _original=original_up, _dim=dim):
                    msg = _original(up_x_j, up_attr, up_weight)
                    if owner.mode == 'h_message' and owner._context is not None:
                        ctx = owner._context
                        index = ctx['data'].cochains[_dim].upper_index
                        if index is not None and len(msg):
                            group = ctx['data'].cochains[_dim].batch[index[1]]
                            msg = msg * ctx['gate'][group, 2*_dim:2*_dim+1]
                    return msg
                def boundary(module, boundary_x_j, _original=original_boundary, _dim=dim):
                    msg = _original(boundary_x_j)
                    if owner.mode == 'h_message' and owner._context is not None:
                        ctx = owner._context
                        index = ctx['data'].cochains[_dim].boundary_index
                        if index is not None and len(msg):
                            group = ctx['data'].cochains[_dim].batch[index[1]]
                            msg = msg * ctx['gate'][group, 2*_dim+1:2*_dim+2]
                    return msg
                mp.message_up = types.MethodType(up, mp)
                mp.message_boundary = types.MethodType(boundary, mp)
            def hook(module, args, output, _i=conv_i):
                if self.mode != 'film' or self._context is None: return output
                loc = self.cfg.get('film_location', 'final')
                use = (loc == 'intermediate' and _i == 0) or (loc == 'readout' and _i == len(self.core.convs)-1)
                if not use: return output
                return [self._film(x, self._context['he'][self._context['data'].cochains[d].batch], True)
                        for d, x in enumerate(output)]
            conv.register_forward_hook(hook)

    def hfeatures(self, role, indices=None, condition='normal'):
        h = self.h_arrays[role]
        if condition == 'h_mask': h = torch.zeros_like(h)
        elif condition.startswith('h_shuffle'):
            order = np.random.default_rng(int(condition.split('_')[-1])).permutation(len(h))
            h = h[torch.tensor(order, device=h.device)]
        return h if indices is None else h[torch.as_tensor(indices, device=h.device)]

    def encode_structure(self, role, indices, condition='normal'):
        ix = np.arange(len(self.h_arrays[role])) if indices is None else np.asarray(indices)
        if self.core is None: return torch.zeros((len(ix), self.hidden), device=self.device)
        if self.model_name == 'Dual-Mapping':
            b = self.geometry[role]
            hh = self.hfeatures(role, None, condition)
            he = self.h_encoder(hh)
            empty = b['local'].new_empty((len(b['local']), 0))
            bh = self.core.base_head[:-1](torch.cat([b['heuristic'], b['rw']], 1))
            h0 = self.core.input_linear(self.core.block_encoder(b['local'], b['heuristic'], b['rw'], empty))
            h = h0
            ew = b.get('edge_weight')
            if self.mode == 'h_message' and b['edge_index'].numel():
                src, dst = b['edge_index']
                ew = torch.sigmoid(self.edge_router(torch.cat([he[src], he[dst]], 1))).flatten()*2
                if b.get('edge_weight') is not None: ew = ew*b['edge_weight']
            for i, layer in enumerate(self.core.gnn_layers):
                h = layer(h, b['edge_index'], ew)
                loc = self.cfg.get('film_location', 'final')
                if self.mode == 'film' and ((loc == 'intermediate' and i == 0) or
                    (loc == 'readout' and i == len(self.core.gnn_layers)-1)):
                    h = self._film(h, he, True)
            z = self.s_encoder(torch.cat([bh, h+self.core.residual_projection(h0)], 1))[torch.as_tensor(ix, device=self.device)]
        else:
            comps = self.geometry[role]
            # ComplexBatch mutates x internally; source objects retain immutable x.
            for i in ix:
                for chain in comps[int(i)].cochains.values():
                    if chain.x is not None:
                        if not hasattr(chain, 'joint_original_x'): chain.joint_original_x = chain.x[:,:1].clone()
                        chain.x = chain.joint_original_x.clone()
                    if chain.y is not None: chain.y = torch.zeros_like(chain.y)
                comps[int(i)].y = torch.tensor([0], dtype=torch.long)
            loader = self.api.DataLoader([comps[int(i)] for i in ix], batch_size=int(self.cfg.get('batch_size', 32)),
                                         shuffle=False, max_dim=2)
            pieces, offset = [], 0
            for data in loader:
                data = data.to(self.device)
                count = int(data.cochains[0].batch.max())+1
                he = self.h_encoder(self.hfeatures(role, ix[offset:offset+count], condition))
                self._context = dict(data=data, he=he, gate=2*torch.sigmoid(self.relation_router(he)))
                pieces.append(self.s_encoder(self.core(data, return_embedding=True)))
                offset += count
            self._context = None
            z = torch.cat(pieces, 0)
        return z

    def fuse(self, h, s, condition='normal', teacher=False):
        he = self.h_encoder(h)
        if condition == 'structure_mask' or teacher or self.mode == 'h_only': s = s*0
        if self.mode == 'structure_only': he = he*0
        if self.mode == 'film' and self.cfg.get('film_location', 'final') == 'final':
            s = self._film(s, he)
        if self.mode == 'cross_attention':
            # Four scalar transforms, explicitly NOT a historical time series.
            ht = torch.stack([p(h[:, k:k+1]) for k, p in enumerate(self.h_tokens)], 1)
            st = torch.stack([p(s) for p in self.s_tokens], 1)
            learned, _ = self.cross_h(ht, st, st, need_weights=False)
            he = he + learned.mean(1)
            if self.cfg.get('attention_direction', 'one_way') == 'bidirectional':
                learned_s, _ = self.cross_s(st, ht, ht, need_weights=False)
                s = s + learned_s.mean(1)
        if self.mode == 'moe':
            joined = torch.cat([he, s], 1)
            route = torch.softmax(self.router(joined), 1)
            experts = torch.stack([self.experts[0](torch.cat([he,s,he*s],1)),
                                   self.experts[1](s), self.experts[2](s*s)], 1)
            s = (route[:,:,None]*experts).sum(1)
            self.router_last = route
        # Mask after transformations so a bias cannot defeat the perturbation.
        if condition == 'structure_mask' or teacher or self.mode == 'h_only': s = s*0
        return self.head(torch.cat([he, s, he*s], 1)).flatten()

    def forward(self, role, indices=None, condition='normal', teacher=False):
        ix = np.arange(len(self.h_arrays[role])) if indices is None else np.asarray(indices)
        h = self.hfeatures(role, ix, condition)
        if teacher: s = torch.zeros((len(ix), self.hidden), device=self.device)
        else: s = self.encode_structure(role, ix, condition)
        if condition == 'structure_shuffle':
            order = np.random.default_rng(9301).permutation(len(s))
            s = s[torch.tensor(order, device=self.device)]
        return self.fuse(h, s, condition, teacher)

    def architecture_description(self):
        return dict(mechanism=self.mode, single_neural_prediction_head=True,
                    raw_H_output_skip=False, output_score_fusion=False,
                    structural_core=('original GraphSAGE DualMappingGNN hidden blocks' if self.model_name=='Dual-Mapping'
                                     else 'original SparseSIN boundary/upper cochain propagation, max_dim2'),
                    H_token_semantics='four deterministic views of one H scalar; not observed history sequence',
                    H_message_implementation=('learned Hsrc/Hdst weights in original GraphSAGE aggregation' if self.model_name=='Dual-Mapping'
                                               else 'separate candidate H gates on actual upper/boundary messages per cochain order'),
                    monotonic_branch_does_not_guarantee_joint_monotonicity=True)
