import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def resolve_device(value):
    if value != 'auto':
        return value
    import torch
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def configured_dataset(dataset, epochs=None):
    from data.prepare import config
    cfg = config(dataset)
    if epochs is not None:
        if epochs < 1:
            raise ValueError('epochs must be positive')
        for model in cfg['models'].values():
            model['epochs'] = epochs
    return cfg


def existing(args):
    import numpy as np
    cfg = configured_dataset(args.dataset, getattr(args, 'epochs', None))
    if cfg.get('pipeline') == 'historynet_joint':
        from pipelines.historynet_joint import run
        return run(args, cfg)
    mode = args.mode or ('structure_only' if cfg.get('pipeline') == 'native' else 'component_joint')
    if cfg.get('pipeline') == 'native' and mode != 'structure_only':
        raise ValueError('This original dataset uses no historical predictor; choose structure_only')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', dict(cfg, requested_mode=mode))
    report = []

    if cfg.get('pipeline') == 'native':
        from pipelines.inputs import native_check
        from pipelines.dispatch import run_model
        from evaluate import evaluate
        folder = native_check(args.dataset)
        with np.load(folder/'selected_sample.npz', allow_pickle=False) as z:
            q = z['q'].copy()
        for model in args.models:
            for seed in ([args.seed] if args.seed is not None else cfg['models'][model]['seeds']):
                y, score = run_model(args.dataset, model, seed, args.device)
                if len(score) != len(q):
                    raise ValueError('Native candidate/prediction count differs')
                np.savez_compressed(args.output/f'{model}_{seed}.npz', q=q, y=y, predictions=score)
                row = dict(dataset=args.dataset, model=model, seed=seed, **evaluate(y, score))
                report.append(row)
                print(json.dumps(row), flush=True)
        write_json(args.output/'metrics.json', report)
        return
    from data.prepare import geometry, evaluation_labels
    from train import roles_for, fit_model
    from models import network
    from evaluate import evaluate
    import torch
    network.DEVICE = torch.device(args.device)
    with np.load(ROOT/'data'/args.dataset/'candidates.npz', allow_pickle=False) as z:
        q = {'fit': z['train_q'].copy(), 'target': z['target_q'].copy()}
        train_y = z['train_y'].copy()
    components, masks = None, ()
    if mode == 'component_joint':
        from historical_components.generators import create_generator
        from historical_components.encoders import ComponentPreprocessor
        if cfg.get('component_generator_config', {}).get('recipe_policy', 'frozen_fit') != 'frozen_fit':
            raise ValueError('Formal component_joint requires frozen_fit; original_stage is for separate H audit')
        batches = create_generator(cfg['component_generator'], cfg.get('component_generator_config', {})).generate(output_dir=args.output/'components')
        names = list(batches['fit'].names)
        for role in ('fit', 'target'):
            if list(batches[role].names) != names:
                raise ValueError('Component names/order differ across origins')
            np.testing.assert_array_equal(q[role], batches[role].q)
        processor = ComponentPreprocessor(names)
        if list(batches['fit'].mask_indices) != processor.mask_indices or list(batches['target'].mask_indices) != processor.mask_indices:
            raise ValueError('Component activity mask names/indices must agree at both origins')
        processor.fit(batches['fit'].values)
        components = {r: processor.transform(batches[r].values) for r in q}
        masks = batches['fit'].mask_indices
        write_json(args.output/'component_preprocessing.json', processor.state_dict())
    for model in args.models:
        structure, rw = geometry(args.dataset, model, network.DEVICE)
        roles = roles_for(model, structure, rw, q, components, masks)
        for seed in ([args.seed] if args.seed is not None else cfg['models'][model]['seeds']):
            net, score = fit_model(model, roles, train_y, cfg['models'][model], seed, mode, args.device)

            y = evaluation_labels(args.dataset, q['target'])
            probability = torch.sigmoid(torch.from_numpy(score)).numpy()
            np.savez_compressed(args.output/f'{model}_{seed}.npz', q=q['target'], y=y,
                                logits=score, probabilities=probability, predictions=score)
            torch.save(net.state_dict(), args.output/f'{model}_{seed}.pt')
            write_json(args.output/f'{model}_{seed}_training.json', net.training_history)
            row = dict(dataset=args.dataset, model=model, seed=seed, mode=mode, **evaluate(y, score))
            report.append(row)
            print(json.dumps(row), flush=True)
    write_json(args.output/'metrics.json', report)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', help='One of the eight original dataset names')
    source.add_argument('--config', type=Path, help='Explicit configuration for a new dataset')
    p.add_argument('--input', type=Path, help='Override new-dataset events path')
    p.add_argument('--mode', choices=['component_joint', 'structure_only'])
    p.add_argument('--model', choices=['dual', 'mlshol', 'both'], default='both')
    p.add_argument('--seed', type=int)
    p.add_argument('--epochs', type=int, help='Override downstream training epochs for bundled datasets only')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    a = p.parse_args()
    if a.epochs is not None and (a.epochs < 1 or a.config is not None):
        p.error('--epochs must be positive and applies only to --dataset')
    a.models = {'dual': ['Dual-Mapping'], 'mlshol': ['MLSHoL'], 'both': ['Dual-Mapping', 'MLSHoL']}[a.model]
    a.device = resolve_device(a.device)
    if a.dataset:
        existing(a)
    else:
        from temporal_pipeline import run
        run(a)

if __name__ == '__main__':
    main()
