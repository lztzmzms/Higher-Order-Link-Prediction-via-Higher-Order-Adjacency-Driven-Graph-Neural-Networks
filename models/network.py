import numpy as np
import torch
from torch import nn
from models import dual
from mp.models import SparseSIN
from simplicial_complex.data_loading import DataLoader
from historical_components.encoders.positive import PositiveComponentEncoder, JointHead

DEVICE = torch.device('cpu')

class StructuralEncoder(nn.Module):
    pass
    def __init__(self, model, fit, cfg):
        super().__init__()
        if model not in ('Dual-Mapping', 'MLSHoL'):
            raise ValueError(f'Unknown structural model: {model}')
        self.model, self.cfg = model, cfg
        self.hidden = width = int(cfg.get("hidden", 32))
        if width < 2 or width % 2:
            raise ValueError("hidden must be a positive even representation width")
        self.rw_enabled = bool(cfg.get("rw_enabled", True))
        if model == 'Dual-Mapping':
            block = fit['block']
            if str(cfg.get('gnn', 'GAT')).upper() != 'GAT':
                raise ValueError('Dual-Mapping requires GAT in this release')
            *_, Core = dual.make_model_classes()
            self.core = Core(
                block['local'].shape[1], block['heuristic'].shape[1], (block['rw'].shape[1] if self.rw_enabled else 0), 0,
                feature_proj_dim=max(1, width // 2), hidden_dim=2 * width, num_layers=cfg['depth'],
                encoder='gat', gat_heads=cfg.get('gat_heads', 4),
                dropout=cfg['dropout'], max_residual_alpha=1.0)

            for p in self.core.base_head[-1].parameters():
                p.requires_grad_(False)
            for p in self.core.residual_head.parameters():
                p.requires_grad_(False)
            self.core.alpha_param.requires_grad_(False)
            self.structure_projection = nn.Sequential(nn.Linear(4 * width, width), nn.LayerNorm(width), nn.SiLU())
        else:
            self.core = SparseSIN(1, 2, cfg['depth'], width // 2,
                dropout_rate=cfg['dropout'], max_dim=2, readout='mean', final_readout='sum')
            for p in self.core.lin2.parameters():
                p.requires_grad_(False)
            self.structure_projection = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU())
            if self.rw_enabled:
                rw_width = fit['block']['rw'].shape[1]
                self.rw_projection = nn.Sequential(nn.LayerNorm(rw_width), nn.Linear(rw_width, width), nn.ReLU(), nn.Dropout(cfg['dropout']))
                self.rw_fusion = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.SiLU())
        self.to(DEVICE)

    def structure_embedding(self, role, ids):
        index = torch.as_tensor(ids, dtype=torch.long, device=DEVICE)
        if self.model == 'Dual-Mapping':
            block = dict(role['block'])
            if not self.rw_enabled:
                block['rw'] = block['rw'][:, :0]
            empty = block['local'].new_empty((len(block['local']), 0))
            base = self.core.base_head[:-1](torch.cat([block['heuristic'], block['rw']], 1))
            h0 = self.core.input_linear(self.core.block_encoder(block['local'], block['heuristic'], block['rw'], empty))
            hidden = h0
            for layer in self.core.gnn_layers:
                hidden = layer(hidden, block['edge_index'], block.get('edge_weight'))
            return self.structure_projection(torch.cat([base, hidden + self.core.residual_projection(h0)], 1))[index]
        complexes = role['complexes']
        data = next(iter(DataLoader(
            [complexes[int(i)] for i in ids], batch_size=max(1, len(ids)),
            shuffle=False, max_dim=2))).to(DEVICE)
        structural = self.structure_projection(self.core(data, return_embedding=True))
        if not self.rw_enabled:
            return structural
        return self.rw_fusion(torch.cat([structural, self.rw_projection(role['rw'][index])], 1))

class Network(StructuralEncoder):
    pass
    def __init__(self, model, fit, cfg, component_dim=None, mask_indices=None):
        super().__init__(model, fit, cfg)
        dim = fit['x'].shape[1] if component_dim is None else component_dim
        masks = fit.get('component_masks', []) if mask_indices is None else mask_indices
        self.h_encoder = PositiveComponentEncoder(dim, masks, hidden=self.hidden).double()
        self.head = JointHead(hidden=self.hidden).double()
        self.control = 'joint'
        self.to(DEVICE)

    def forward(self, role, ids):
        index = torch.as_tensor(ids, dtype=torch.long, device=DEVICE)
        history = self.h_encoder(role['x'][index].double())
        structure = self.structure_embedding(role, ids).double()
        return self.head(history, structure)

class StructureNetwork(StructuralEncoder):
    pass
    def __init__(self, model, fit, cfg):
        super().__init__(model, fit, cfg)
        self.head = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.SiLU(), nn.Linear(self.hidden, 1)).double().to(DEVICE)
        self.control = 'structure'

    def forward(self, role, ids):
        return self.head(self.structure_embedding(role, ids).double()).flatten()

JointPredictionModel = Network
