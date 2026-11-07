from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

@dataclass
class ComponentBatch:
    q: np.ndarray
    values: np.ndarray
    names: Sequence[str]
    mask_indices: Sequence[int] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=np.int64)
        self.values = np.asarray(self.values)
        self.names = tuple(map(str, self.names))
        self.mask_indices = tuple(map(int, self.mask_indices))
        if self.q.ndim != 2 or self.q.shape[1] != 3:
            raise ValueError('Components require explicit sorted three-node candidate IDs')
        if len({tuple(row) for row in self.q}) != len(self.q):
            raise ValueError('Duplicate component candidate IDs')
        if np.any(np.sort(self.q, axis=1) != self.q):
            raise ValueError('Candidate IDs must be sorted triples')
        if self.values.shape != (len(self.q), len(self.names)) or not np.isfinite(self.values).all():
            raise ValueError('Invalid component matrix, names, or nonfinite component output')
        if len(set(self.names)) != len(self.names):
            raise ValueError('Component names must be unique')
        if any(i < 0 or i >= len(self.names) for i in self.mask_indices):
            raise ValueError('Invalid component mask index')
        if self.mask_indices and not np.isin(self.values[:, self.mask_indices], [0, 1]).all():
            raise ValueError('Activity masks must be binary and are not prediction strengths')

@dataclass
class CandidateContext:
    pass

    dataset: str
    prefix_events: Sequence
    candidates: np.ndarray
    prediction_cut: int
    completed_windows: Sequence[Mapping[str, Any]] = ()
    recipe: Mapping[str, Any] = field(default_factory=dict)
    role: str = 'target'

    def __post_init__(self):
        if len(self.prefix_events) != self.prediction_cut:
            raise ValueError('CandidateContext must contain exactly the permitted prefix')
        for window in self.completed_windows:
            if int(window['label_end']) > self.prediction_cut:
                raise ValueError('Historical labels exceed prediction time')

class ComponentGenerator(ABC):
    def generate_for_context(self, context: CandidateContext):
        return self.generate(context=context)

    @abstractmethod
    def generate(self, output_dir: Path | None = None, context: CandidateContext | None = None):
        pass
        raise NotImplementedError
