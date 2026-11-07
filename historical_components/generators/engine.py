import ast
from collections import Counter, defaultdict
from itertools import combinations, groupby, product
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import networkx as nx
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UPSTREAM = Path(__file__).resolve().parents[1] / 'upstream'

def definitions(module, names, extra=None, raw_members=False):
    path = UPSTREAM / (module + '.py')
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing upstream predictive definitions: {module}, {names}')
    if raw_members:
        fn = next(n for n in nodes if n.name == 'predict')

        stack_branch = next(n for n in fn.body if isinstance(n, ast.If)
                            and 'OOF_STACK' in ast.unparse(n.test))
        stack_branch.body = ast.parse("raise ValueError('OOF_STACK requires an explicitly recovered component schema')").body
        loop_index = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.For)
                          and isinstance(n.target, ast.Name) and n.target.id == 'seed')
        fn.body.insert(loop_index, ast.parse('_raw_members = []').body[0])
        loop = fn.body[loop_index + 1]

        append_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr)
                            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                            and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'values')
        loop.body[append_index:] = ast.parse('_raw_members.append(np.asarray(prediction).copy())').body
        fn.body[loop_index + 2:] = ast.parse('return np.column_stack(_raw_members), gate').body
    ns = dict(globals())
    ns.update(extra or {})
    ns['__file__'] = str(path)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), ns)
    return SimpleNamespace(**{name: ns[name] for name in names})

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

class TrainingNamespace:
    pass
    def __init__(self, namespace):
        object.__setattr__(self, '_namespace', namespace)

    def __getattr__(self, name):
        try:
            return self._namespace[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self._namespace[name] = value

def build_engine(dataset, runtime, raw_members=True):
    core = definitions('core', ['aggregate', 'finite_matrix', 'top_features', 'score_metrics'])
    temporal = definitions('temporal', ['sha', 'cutoff', 'uniform_open_candidates', 'attach_labels'],
                           dict(CAP=25000, r=core, STARTS=(.2, .3, .4, .5, .6)))
    temporal.r = core
    temporal.STARTS = (.2, .3, .4, .5, .6)
    event = definitions('event_features', ['History', 'prefix_state'], dict(s=temporal))
    training_data = {}
    source_impl = definitions('source_training', ['train_data', 'columns'],
                              dict(DATA=training_data, CURRENT=dataset, CURRENT_WEIGHT=None, s=temporal))
    source = TrainingNamespace(source_impl.train_data.__globals__)
    source.e = event
    source.train_data = source_impl.train_data
    source.columns = source_impl.columns
    legacy = definitions('legacy', ['train', 'cols', 'estimator', 'predict'], dict(a=source))
    b = SimpleNamespace(a=source, s=temporal, r=core)
    rank_data = {}
    rank = definitions('rank_predictor', ['clock_features', 'community_features', 'feature_indices',
                                        'training', 'pair_indices', 'predict', 'stack_predict', 'robust_quality'],
                       dict(b=b, DATA=rank_data, CURRENT=dataset), raw_members=raw_members)
    rank.DATA = rank_data
    rank.b = b
    broad = definitions('broad', ['novel_features'],
                        dict(s=temporal, TAUS=(.005, .01, .02, .03, .05, .075, .1, .15, .2, .4)))
    def high_pool(score, count):
        return np.argsort(-score, kind='stable')[:min(len(score), int(count))]
    local = definitions('enron_local', ['eligible', 'fit_score'],
                        dict(OUT=Path(runtime), high_pool=high_pool, n=SimpleNamespace(atomic=atomic)))
    bounded = definitions('enron_bounded', ['localized', 'bounded_order', 'rearrange'], dict(high_pool=high_pool))
    return SimpleNamespace(core=core, temporal=temporal, event=event, source=source,
                           legacy=legacy, rank=rank, broad=broad, local=local, bounded=bounded)

def feature_matrix(engine, dataset, prefix, q):
    query = [tuple(map(int, row)) for row in q]
    raw = engine.event.prefix_state(prefix).features(query)
    base_names = sorted(raw)
    base_x = engine.core.finite_matrix(raw, base_names).astype(np.float32)
    extra = engine.broad.novel_features(prefix, query)
    if dataset in ('email-Eu', 'tags-ask-ubuntu'):
        raw.update(extra)
        units = 1. if dataset == 'email-Eu' else 1000.
        raw.update(engine.rank.clock_features(prefix, query, units, (1., 6., 24., 168., 720., 2160.)))
        raw.update(engine.rank.community_features(prefix, query))
        names = sorted(raw)
        x = engine.core.finite_matrix(raw, names).astype(np.float32)
        return x, np.asarray(names), x.copy(), base_x, np.asarray(base_names)
    extra_names = sorted(extra)
    old_x = np.column_stack([base_x, engine.core.finite_matrix(extra, extra_names)]).astype(np.float32)
    names = base_names + extra_names
    units = 1000. if dataset == 'email-Enron' else 1.
    hours = (24., 168., 720., 2160.) if dataset == 'email-Enron' else (.25, 1., 3., 6., 24.)
    new = engine.rank.clock_features(prefix, query, units, hours)
    corrected = old_x.copy()
    corrected[:, names.index('event_same_day_recent_edges')] = np.log1p(new.pop('event_same_day_recent_edges'))
    new.update(engine.rank.community_features(prefix, query))
    new_names = sorted(new)
    x = np.column_stack([corrected, engine.core.finite_matrix(new, new_names)]).astype(np.float32)
    return x, np.asarray(names + new_names), old_x, base_x, np.asarray(base_names)
