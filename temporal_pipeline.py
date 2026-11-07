import importlib
import json
from itertools import combinations
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from preprocessing.events import load_events, open_triangles, labels
from models import dual, network
from train import seed_everything, roles_for, fit_model

def load_plugin(path):
    if not isinstance(path, str) or ':' not in path:
        raise ValueError('Custom implementation must use module:object syntax')
    module, name = path.rsplit(':', 1)
    return getattr(importlib.import_module(module), name)

def structures(prefix, q, cfg, seed, need_complexes=True):
    from data.local_complex import remove_target_link, construct_labeled_networkx_graph, to_pyg_data
    from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi
    edges = {e for _, _, vs in prefix for e in combinations(vs, 2)}
    triangles = {t for _, _, vs in prefix for t in combinations(vs, 3)}
    candidates = list(map(tuple, q))
    dg = dual.build_dual_graph(candidates)
    features, raw = dual.build_features(candidates, dict(edges=edges, triangles=triangles), dg, return_raw=True)
    block = dict(local=dual.select_feature_block(features, dual.FEATURE_NAMES, dual.local_feature_names()),
                 heuristic=dual.build_heuristic_feature_block(raw),
                 rw=dual.build_random_walk_structural_features(dg['edge_index'], len(q),
                     rw_steps=cfg.get('random_walk', {}).get('steps', 2)), edge_index=dg['edge_index'])
    if not need_complexes:
        return block, None
    graph = nx.Graph()
    graph.add_edges_from(sorted(edges))
    rng, items = np.random.default_rng(seed), []
    observed = [vs for _, _, vs in prefix if len(vs) >= 3]
    for tri in candidates:
        extra = sorted({n for v in tri for n in graph.neighbors(v)} - set(tri))
        cap = cfg.get('mls_neighbors', 30)
        if len(extra) > cap:
            extra = sorted(rng.choice(extra, cap, replace=False).tolist())
        local = remove_target_link(graph.subgraph(sorted(set(tri) | set(extra))), tri)
        labeled = construct_labeled_networkx_graph(local, observed, tri, set(),
            disconnected_policy='unreachable_zero', observed_2_skeleton=cfg.get('mls_observed_2_skeleton', True))
        pyg = to_pyg_data(labeled)
        pyg.simplices.extend([[i] for i in range(pyg.num_nodes)])
        if pyg.edge_index.numel() == 0:
            pyg.edge_index = torch.empty((2, 0), dtype=torch.long)
        items.append(convert_graph_dataset_with_gudhi([pyg], expansion_dim=2, include_down_adj=False,
            init_method='sum', expand_cliques=cfg.get('mls_expand_cliques', False))[0])
    return block, items

def run(args):
    from run import write_json
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg.get('K') != 2:
        raise ValueError('This preserved structural implementation supports K=2 (three nodes)')
    mode = args.mode or cfg.get('mode', 'structure_only')
    if mode == 'component_joint' and not cfg.get('component_generator'):
        raise ValueError('New component_joint datasets require an explicit component generator recipe')
    if mode == 'component_joint' and ':' not in cfg['component_generator']:
        raise ValueError('The six registered recipes preserve their original cohorts. New datasets need module:Generator')
    seed = args.seed if args.seed is not None else cfg.get('seed', 9101)
    seed_everything(seed)
    network.DEVICE = torch.device(args.device)
    event_path = args.input or args.config.parent/cfg['input']
    rows, node_ids = load_events(event_path)
    start, fit, cut, end = (cfg[k] for k in ('observation_start', 'train_cut', 'prediction_cut', 'prediction_end'))
    if not start < fit < cut < end:
        raise ValueError('Expected observation_start < train_cut < prediction_cut < prediction_end')
    if any(not start < c < e <= cut for c, e in cfg.get('history_windows', [])):
        raise ValueError('Every upstream history window must end by the final prediction cut')
    q, prefixes = {}, {}
    for role, boundary in (('fit', fit), ('target', cut)):
        prefixes[role] = [r for r in rows if start <= r[0] < boundary]
        builder = load_plugin(cfg['candidate_builder']) if cfg.get('candidate_builder') else open_triangles
        q[role] = np.asarray(builder(prefixes[role]), dtype=np.int64).reshape(-1, 3)
        from data.prepare import validate_candidates
        validate_candidates(prefixes[role], q[role])
        if not len(q[role]):
            raise ValueError(f'No open-triangle candidates at {role} origin')
    args.output.mkdir(parents=True, exist_ok=False)
    components, masks = None, ()
    if mode == 'component_joint':
        from historical_components.encoders import ComponentPreprocessor

        from historical_components.generators import create_generator
        generator = create_generator(cfg['component_generator'], cfg.get('component_generator_config', {}))
        batches = {}
        from historical_components.interfaces import CandidateContext
        for role, boundary in (('fit', fit), ('target', cut)):
            windows = cfg.get('history_windows', [])
            permitted = [dict(cut=sum(r[0] < c for r in prefixes[role]),
                              label_end=sum(r[0] < e for r in prefixes[role]),
                              timestamp_cut=c, timestamp_label_end=e)
                         for c, e in windows if start < c < e <= boundary]
            context = CandidateContext(dataset=cfg.get('dataset', 'custom'), prefix_events=prefixes[role],
                candidates=q[role], prediction_cut=len(prefixes[role]), completed_windows=permitted,
                recipe=dict(cfg.get('component_generator_config', {}),
                            observation_start=start, timestamp_cut=boundary), role=role)
            batch = generator.generate_for_context(context)
            np.testing.assert_array_equal(batch.q, q[role])
            batches[role] = batch
        names = list(batches['fit'].names)
        if names != list(batches['target'].names):
            raise ValueError('Component names/order must match across prediction origins')
        processor = ComponentPreprocessor(names).fit(batches['fit'].values)
        if list(batches['fit'].mask_indices) != processor.mask_indices or list(batches['target'].mask_indices) != processor.mask_indices:
            raise ValueError('Component activity mask names/indices must agree at both origins')
        components = {r: processor.transform(batches[r].values) for r in q}
        masks = batches['fit'].mask_indices
        for role in q:
            np.savez_compressed(args.output/f'{role}_components.npz', q=q[role],
                                components=batches[role].values, names=np.asarray(names))
        write_json(args.output/'component_preprocessing.json', processor.state_dict())
    blocks, complexes = {}, {}
    for role in q:
        blocks[role], complexes[role] = structures(prefixes[role], q[role], cfg, seed,
                                                  need_complexes='MLSHoL' in args.models)
    rw = {}
    for key in ('local', 'heuristic', 'rw'):
        scaler = StandardScaler().fit(blocks['fit'][key])
        for role in q:
            blocks[role][key] = torch.tensor(scaler.transform(blocks[role][key]), dtype=torch.float32, device=args.device)
            if key == 'rw':
                rw[role] = blocks[role][key]
    for block in blocks.values():
        block['edge_index'] = torch.tensor(block['edge_index'], dtype=torch.long, device=args.device)
    train_y = labels([r for r in rows if fit <= r[0] < cut], q['fit'])
    report = []
    for model in args.models:
        roles = roles_for(model, blocks if model == 'Dual-Mapping' else complexes, rw, q, components, masks)
        mc = cfg.get('models', {}).get(model, cfg.get('model'))
        if mc is None:
            raise ValueError(f'Missing model configuration: {model}')
        net, score = fit_model(model, roles, train_y, mc, seed, mode, args.device)
        payload = dict(q=q['target'], logits=score, predictions=score,
                       probabilities=torch.sigmoid(torch.from_numpy(score)).numpy())
        row = dict(model=model, mode=mode, seed=seed, candidates=len(score))
        if cfg.get('future_labels_available', False):
            if cfg.get('observed_until', float('-inf')) < end:
                raise ValueError('A complete future observation horizon is required for metrics')
            from evaluate import evaluate
            payload['y'] = labels([r for r in rows if cut <= r[0] < end], q['target'])
            row.update(evaluate(payload['y'], score))
        np.savez_compressed(args.output/f'{model}_{seed}.npz', **payload)
        torch.save(net.state_dict(), args.output/f'{model}_{seed}.pt')
        write_json(args.output/f'{model}_{seed}_training.json', net.training_history)
        report.append(row)
    write_json(args.output/'metrics.json', report)
    write_json(args.output/'node_ids.json', node_ids)
    write_json(args.output/'config.json', dict(cfg, requested_mode=mode))
    print(json.dumps(report), flush=True)
