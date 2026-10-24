#!/usr/bin/env python3
                                                                                  

from __future__ import annotations

import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import argparse

import csv

import hashlib

import json

import math

import signal

import subprocess

import sys

import time

import traceback

import platform

from collections import Counter, defaultdict

from itertools import combinations, groupby

from pathlib import Path

import numpy as np

from scipy.stats import rankdata

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import average_precision_score, roc_auc_score

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

class History:
    pass                                                     
    def __init__(self, prefix):
        self.adj = defaultdict(set)
        self.closed = set()
        self.node = Counter()
        self.edge = Counter()
        self.group = Counter()
        self.nc = Counter()
        self.ec = Counter()
        self.last = {}
        self.decay = {t: Counter() for t in (.01, .05, .2, 1.)}
        self.start = prefix[0][0]
        self.end = prefix[-1][0]
        self.span = max(1, self.end - self.start)
        for timestamp, iterator in groupby(prefix, key=lambda v: v[0]):
            block = list(iterator)
            new = {q for _, _, vs in block for q in combinations(vs, 3)} - self.closed
                                                                                        
            eligible = [q for q in new if all(b in self.adj[a] for a, b in combinations(q, 2))]
            for q in eligible:
                self.nc.update(q)
                self.ec.update(combinations(q, 2))
            for _, _, vs in block:
                self.node.update(vs)
                edges = list(combinations(vs, 2))
                self.edge.update(edges)
                if len(vs) >= 3:
                    self.group.update(edges)
                for edge in edges:
                    x, y = edge
                    self.adj[x].add(y)
                    self.adj[y].add(x)
                    self.last[edge] = timestamp
                    for tau, count in self.decay.items():
                        count[edge] += math.exp(-(self.end - timestamp) / (tau * self.span))
            self.closed.update(new)

    def candidates(self):
        result = []
        for x in sorted(self.adj):
            for y in sorted(v for v in self.adj[x] if v > x):
                for z in sorted(v for v in self.adj[x] & self.adj[y] if v > y):
                    q = (x, y, z)
                    if q not in self.closed:
                        result.append(q)
        return np.asarray(result, dtype=np.int64).reshape(-1, 3)

    def proxies(self, q):
        activity, recent = [], []
        for tri in q:
            edges = list(combinations(map(int, tri), 2))
            activity.append(sum(math.log1p(self.edge[e]) for e in edges))
            recent.append(sum(self.decay[.05][e] for e in edges))
        return np.asarray(activity), np.asarray(recent)

    def features(self, queries):
        output = defaultdict(list)
        for tri in queries:
            tri = tuple(map(int, tri))
            edges = list(combinations(tri, 2))
            groups = {
                'edge_count': [self.edge[e] for e in edges],
                'edge_group': [self.group[e] for e in edges],
                'node_count': [self.node[v] for v in tri],
                'node_degree': [len(self.adj[v]) for v in tri],
                'node_closure': [self.nc[v] for v in tri],
                'edge_closure': [self.ec[e] for e in edges],
                'edge_age': [(self.end - self.last[e]) / self.span for e in edges],
                'edge_common': [len(self.adj[x] & self.adj[y]) for x, y in edges],
                'edge_jaccard': [len(self.adj[x] & self.adj[y]) / max(1, len(self.adj[x] | self.adj[y])) for x, y in edges],
                'edge_coactivity': [self.edge[x, y] / math.sqrt(max(1, self.node[x] * self.node[y])) for x, y in edges],
            }
            for tau, counts in self.decay.items():
                groups[f'decay_{tau}'] = [counts[e] for e in edges]
            for name, values in groups.items():
                v = np.asarray(values, dtype=float)
                for suffix, val in [('min', v.min()), ('mean', v.mean()), ('max', v.max()),
                        ('std', v.std()), ('logsum', np.log1p(v).sum()),
                        ('harmonic', 3 / np.sum(1 / (v + 1e-12)))]:
                    output[name + '_' + suffix].append(val)
            output['common_three'].append(len(set.intersection(*(self.adj[v] for v in tri))))
        names = sorted(output)
        x = np.column_stack([output[n] for n in names])
        return np.log1p(np.maximum(x, 0)).astype(np.float32), names

def constraint(y, population_y, reference_size, config):
    p, n = int(np.sum(y)), len(y) - int(np.sum(y))
    pp, nn = int(np.sum(population_y)), len(population_y) - int(np.sum(population_y))
    factor = (p / n) / (pp / nn) if n and pp and nn else None
    size_factor = len(y) / reference_size
    return {'sample_candidates': len(y), 'sample_positives': p, 'sample_negatives': n,
        'population_candidates': len(population_y), 'population_positives': pp,
        'population_positive_negative_ratio': pp / nn if nn else None,
        'sample_positive_negative_ratio': p / n if n else None, 'ratio_factor': factor,
        'size_factor': size_factor,
        'ratio_valid': factor is not None and config['ratio_min'] <= factor <= config['ratio_max'],
        'size_valid': config['size_min'] <= size_factor <= config['size_max']}

def quota_sample(y, n, seed):
    pass                                                                              
    rng = np.random.default_rng(seed)
    n = min(n, len(y))
    pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
    np_ = int(round(n * len(pos) / len(y)))
    selected = np.r_[rng.choice(pos, min(np_, len(pos)), replace=False),
        rng.choice(neg, n - min(np_, len(pos)), replace=False)]
    return np.sort(selected)

