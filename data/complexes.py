from pathlib import Path
import numpy as np
import torch
from simplicial_complex.complex import Cochain, Complex

FIELDS = ('x', 'upper_index', 'lower_index', 'boundary_index', 'shared_boundaries',
          'shared_coboundaries', 'upper_orient', 'lower_orient')

class PackedComplexes:
    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as z:
            self.arrays = {k:z[k] for k in z.files}

    def __len__(self):
        return len(self.arrays['dimensions'])

    def original_nodes(self, index):
        start, stop = self.arrays['original_node_ids_ptr'][index:index+2]
        return self.arrays['original_node_ids'][start:stop].copy()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        chains = []
        for dim in range(int(self.arrays['dimensions'][index]) + 1):
            fields = {}
            for field in FIELDS:
                key = f'd{dim}_{field}'
                if not self.arrays[key + '_present'][index]:
                    fields[field] = None
                    continue
                start, stop = self.arrays[key + '_ptr'][index:index+2]
                value = self.arrays[key]
                part = value[:, start:stop] if field.endswith('index') else value[start:stop]
                fields[field] = torch.from_numpy(part.copy())
            chains.append(Cochain(dim=dim, **fields))
        return Complex(*chains, y=torch.zeros(1, dtype=torch.long))
