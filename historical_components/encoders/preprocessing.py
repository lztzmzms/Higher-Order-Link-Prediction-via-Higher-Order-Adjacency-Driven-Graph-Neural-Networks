import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

class ComponentPreprocessor:
    pass

    def __init__(self, names):
        self.names = [str(name) for name in names]
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Component names must be nonempty and unique")
        self.mask_indices = [j for j, name in enumerate(self.names) if 'mask' in name.lower()]
        if len(self.mask_indices) > 1 and any('if_active' in name for name in self.names):
            raise ValueError('The preserved if_active preprocessing uses one shared rule mask; provide an explicit schema for multiple rules')
        self.scaler = None

    def rank_normalize(self, raw):
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.names):
            raise ValueError("Component matrix and ordered names do not align")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Component cohort must be nonempty and finite")
        encoded = values.copy()
        for j, name in enumerate(self.names):
            if j not in self.mask_indices:
                encoded[:, j] = (rankdata(values[:, j], method='average') - .5) / len(values)
                if 'if_active' in name and self.mask_indices:
                    encoded[:, j] *= values[:, self.mask_indices[0]]
        return encoded

    def fit(self, raw, ids=None):
        encoded = self.rank_normalize(raw)
        index = np.arange(len(encoded)) if ids is None else np.asarray(ids, dtype=np.int64)
        if index.ndim != 1 or not len(index) or np.any(index < 0) or np.any(index >= len(encoded)):
            raise ValueError("Invalid permitted training IDs for component scaler")
        self.scaler = StandardScaler().fit(encoded[index].astype(np.float64))
        return self

    def transform(self, raw):
        if self.scaler is None:
            raise RuntimeError("Fit component scaler on permitted historical training IDs first")
        return np.clip(self.scaler.transform(self.rank_normalize(raw)), -8., 8.).astype(np.float64)

    def fit_transform(self, raw, ids=None):
        return self.fit(raw, ids).transform(raw)

    def state_dict(self):
        if self.scaler is None:
            raise RuntimeError("Component scaler has not been fitted")
        return dict(names=self.names, mask_indices=self.mask_indices,
                    mean=self.scaler.mean_.tolist(), scale=self.scaler.scale_.tolist(),
                    var=self.scaler.var_.tolist(), n_features_in=int(self.scaler.n_features_in_),
                    n_samples_seen=np.asarray(self.scaler.n_samples_seen_).tolist(),
                    clip=[-8., 8.], rank_formula='(average_rank-.5)/N',
                    rank_scope='full fixed prediction-origin candidate cohort')

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['names'])
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = np.asarray(state['mean'], dtype=np.float64)
        obj.scaler.scale_ = np.asarray(state['scale'], dtype=np.float64)
        obj.scaler.var_ = np.asarray(state['var'], dtype=np.float64)
        obj.scaler.n_features_in_ = int(state['n_features_in'])
        obj.scaler.n_samples_seen_ = np.asarray(state['n_samples_seen'])
        return obj
