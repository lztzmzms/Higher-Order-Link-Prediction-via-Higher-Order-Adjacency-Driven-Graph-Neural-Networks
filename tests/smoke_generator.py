from collections import Counter
from itertools import combinations
import numpy as np
from historical_components.interfaces import ComponentBatch, ComponentGenerator

class SyntheticRule(ComponentGenerator):
    def __init__(self, config=None):
        self.config = config or {}

    def generate(self, output_dir=None, context=None):
        counts = Counter(e for _, _, vs in context.prefix_events for e in combinations(vs, 2))
        scores = np.asarray([[min(counts[e] for e in combinations(q,2))] for q in context.candidates], dtype=float)
        return ComponentBatch(context.candidates, scores, ['synthetic_test_pair_min_rule'])
