"""Fixed F joint model with RW enabled for both structural encoders."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from models import dual
from models.joint import JointNetwork
from mp.models import SparseSIN
from simplicial_complex.data_loading import DataLoader

API = SimpleNamespace(dual=dual, SparseSIN=SparseSIN, DataLoader=DataLoader)

class Network(JointNetwork):
    def __init__(self, model, geometry, h_arrays, cfg, device, rw_arrays=None):
        if not cfg['rw_enabled']:
            raise ValueError('The release requires RW enabled for both models')
        super().__init__(API, model, 'teacher_joint', geometry, h_arrays, cfg, device)
        self.rw_arrays = rw_arrays
        if model == 'MLSHoL':
            if rw_arrays is None or rw_arrays['fit'].shape[1] == 0:
                raise ValueError('MLSHoL requires real candidate RW inputs')
            width = rw_arrays['fit'].shape[1]
            self.rw_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.hidden),
                                              nn.ReLU(), nn.Dropout(cfg['dropout']))
            self.rw_fusion = nn.Sequential(nn.Linear(2*self.hidden, self.hidden),
                                          nn.LayerNorm(self.hidden), nn.SiLU())
            self.to(device)

    def encode_structure(self, role, indices, condition='normal'):
        if self.model_name == 'MLSHoL' and indices is None and not self.training:
            batch = int(self.cfg['batch_size'])
            return torch.cat([self.encode_structure(role, np.arange(start, min(start+batch, len(self.h_arrays[role]))))
                              for start in range(0, len(self.h_arrays[role]), batch)], 0)
        encoded = super().encode_structure(role, indices, condition)
        if self.model_name == 'MLSHoL':
            ix = np.arange(len(self.h_arrays[role])) if indices is None else indices
            rw = self.rw_arrays[role][torch.as_tensor(ix, device=self.device)]
            encoded = self.rw_fusion(torch.cat([encoded, self.rw_projection(rw)], 1))
        return encoded
