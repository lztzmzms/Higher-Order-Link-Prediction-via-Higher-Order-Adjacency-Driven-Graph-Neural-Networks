import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school',
            'tags-ask-ubuntu', 'tags-math-sx', 'NDC-substances', 'threads-math-sx')
MODELS = {'both': ('Dual-Mapping', 'MLSHoL'), 'dual': ('Dual-Mapping',), 'mlshol': ('MLSHoL',)}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return number


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def source_hashes():
    paths = {ROOT/name for name in ('run.py', 'train.py', 'evaluate.py', 'requirements.txt',
                                   'scripts/run_server.py')}
    for folder in ('configs', 'data', 'features', 'historical_components', 'models', 'mp',
                   'pipelines', 'simplicial_complex', 'torch_scatter'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.json', '.npz', '.zip')
                     and '__pycache__' not in p.parts and '.cache' not in p.parts)
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def make_plan(args):
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError('Dataset names must be unique')
    epochs = {}
    for dataset in args.datasets:
        cfg = json.loads((ROOT/'configs'/f'{dataset}.json').read_text(encoding='utf-8'))
        epochs[dataset] = {m: args.epochs if args.epochs is not None else cfg['models'][m]['epochs']
                           for m in MODELS[args.model]}
    return dict(version=1, datasets=args.datasets, models=list(MODELS[args.model]),
                model_argument=args.model, seeds=list(range(args.seed_start, args.seed_start+args.runs)),
                epochs=epochs, epoch_override=args.epochs, device=args.device, mode='component_joint',
                upstream_recipe='Unchanged configured predictor members and training budgets',
                files=source_hashes())


@lru_cache(maxsize=8)
def reference_candidates(dataset):
    import numpy as np
    if dataset in ('NDC-substances', 'threads-math-sx'):
        from pipelines.inputs import unpack
        with np.load(unpack(dataset)/'selected_sample.npz', allow_pickle=False) as values:
            return values['q'].copy(), values['y'].copy()
    from data.prepare import evaluation_labels, events
    with np.load(ROOT/'data'/dataset/'candidates.npz', allow_pickle=False) as values:
        q = values['target_q'].copy()
    try:
        labels = evaluation_labels(dataset, q)
    finally:
        events.cache_clear()
    return q, labels


def validate_run(directory, dataset, seed, plan):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np
    from evaluate import evaluate
    directory = Path(directory)
    rows = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != len(plan['models']):
        raise ValueError('Incomplete or duplicate model metrics')
    if {r['model'] for r in rows} != set(plan['models']):
        raise ValueError('Model identities differ from the execution plan')
    output = []
    for row in rows:
        model = row['model']
        if row['dataset'] != dataset or row['seed'] != seed or row['mode'] != 'component_joint':
            raise ValueError('Prediction identity differs from the execution plan')
        epochs = plan['epochs'][dataset][model]
        history = json.loads((directory/f'{model}_{seed}_training.json').read_text(encoding='utf-8'))
        if len(history) != epochs or [r['epoch'] for r in history] != list(range(1, epochs+1)):
            raise ValueError('Incomplete training history')
        for record in history:
            visits = record['candidate_visits']
            if (record['optimizer_updates'] != record['epoch'] or visits < 1 or
                    visits != record['epoch']*history[0]['candidate_visits'] or
                    not math.isfinite(record['weighted_BCE'])):
                raise ValueError('Training budget or loss check failed')
        checkpoint = directory/f'{model}_{seed}.pt'
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise ValueError('Missing model checkpoint')
        with np.load(directory/f'{model}_{seed}.npz', allow_pickle=False) as values:
            q, y, score = values['q'], values['y'], values['predictions']
            if q.shape != (len(y), 3) or not len(y):
                raise ValueError('Candidate and prediction dimensions differ')
            expected_q, expected_y = reference_candidates(dataset)
            if not np.array_equal(q, expected_q) or not np.array_equal(y, expected_y):
                raise ValueError('Predictions do not match the fixed evaluation candidates and labels')
            metrics = evaluate(y, score)
            for key in ('AP', 'Lift'):
                if not math.isclose(metrics[key], row[key], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Saved metrics differ from prediction-level recomputation')
            counts = dict(test_candidates=len(y), test_positives=int(y.sum()))
        output.append(dict(dataset=dataset, model=model, seed=seed, epochs=epochs,
                           **metrics, **counts, output_dir=str(directory)))
    return output


def collect(output, plan):
    rows, errors = [], []
    for dataset in plan['datasets']:
        for seed in plan['seeds']:
            job = output/dataset/f'seed_{seed}'
            marker = job/'complete.json'
            if not marker.is_file():
                continue
            try:
                saved = json.loads(marker.read_text(encoding='utf-8'))
                directory = job/saved['attempt']
                if directory.parent.resolve() != job.resolve():
                    raise ValueError('Invalid attempt path')
                rows.extend(validate_run(directory, dataset, seed, plan))
            except Exception as exc:
                errors.append(dict(dataset=dataset, seed=seed, error=str(exc)))
    write_csv(output/'run_metrics.csv', ['dataset', 'model', 'seed', 'epochs', 'AP', 'Lift',
              'test_candidates', 'test_positives', 'output_dir'], rows)
    summary = []
    for dataset in plan['datasets']:
        for model in plan['models']:
            values = [r for r in rows if r['dataset'] == dataset and r['model'] == model]
            row = dict(dataset=dataset, model=model, n_runs=len(values), expected_runs=len(plan['seeds']),
                       epochs=plan['epochs'][dataset][model])
            for metric in ('AP', 'Lift'):
                samples = [r[metric] for r in values]
                row[metric+'_mean'] = statistics.mean(samples) if samples else None
                row[metric+'_std'] = statistics.stdev(samples) if len(samples) > 1 else None
            summary.append(row)
    write_csv(output/'summary.csv', ['dataset', 'model', 'n_runs', 'expected_runs', 'epochs',
              'AP_mean', 'AP_std', 'Lift_mean', 'Lift_std'], summary)
    write_json(output/'summary.json', dict(updated_at=now(), seed_std_ddof=1,
               completed_model_runs=len(rows), expected_model_runs=len(plan['datasets'])*len(plan['models'])*len(plan['seeds']),
               validation_errors=errors, results=summary))
    return rows, errors


def execute(command, log):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    with log.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(command=command, started_at=now()))+'\n')
        handle.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def environment():
    packages = {}
    for name in ('torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn', 'networkx',
                 'gudhi', 'joblib', 'tqdm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(created_at=now(), python=sys.version, executable=sys.executable,
                platform=platform.platform(), packages=packages)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sequential, resumable eight-dataset component-joint experiments')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--model', choices=tuple(MODELS), default='both')
    parser.add_argument('--runs', type=positive, default=10, help='Independent downstream model seeds per dataset')
    parser.add_argument('--seed-start', type=positive, default=9101)
    parser.add_argument('--epochs', type=positive, help='Downstream epochs; omit to retain configured budgets')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Skip validated completed dataset/seed jobs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    plan = make_plan(args)
    total = len(plan['datasets'])*len(plan['seeds'])
    print(json.dumps(dict(jobs=total, model_runs=total*len(plan['models']), seeds=plan['seeds'],
                          epochs=plan['epochs'], output=str(args.output.resolve())), indent=2), flush=True)
    if args.dry_run:
        return 0
    output = args.output.resolve()
    if output.exists():
        if not args.resume:
            parser.error('Output already exists; choose a new directory or use --resume')
        if json.loads((output/'plan.json').read_text(encoding='utf-8')) != plan:
            parser.error('Resume plan differs: use identical settings, source code and input files')
    else:
        output.mkdir(parents=True)
        write_json(output/'plan.json', plan)
        write_json(output/'environment.json', environment())
    lock = output/'run.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error('Another launcher may be active; inspect run.lock before removing a stale lock')
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), host=platform.node(), started_at=now()), handle)
    failed = 0
    try:
        for index, (dataset, seed) in enumerate(((d, s) for d in plan['datasets'] for s in plan['seeds']), 1):
            job = output/dataset/f'seed_{seed}'
            job.mkdir(parents=True, exist_ok=True)
            marker = job/'complete.json'
            if marker.is_file():
                try:
                    previous = json.loads(marker.read_text(encoding='utf-8'))
                    directory = job/previous['attempt']
                    if directory.parent.resolve() != job.resolve():
                        raise ValueError('Invalid attempt path')
                    validate_run(directory, dataset, seed, plan)
                    print(f'[{index}/{total}] SKIP {dataset} seed={seed}', flush=True)
                    continue
                except Exception as exc:
                    print(f'[{index}/{total}] RETRY {dataset} seed={seed}: {exc}', flush=True)
                    marker.unlink()
            attempt = 1
            while (job/f'attempt_{attempt:04d}').exists() or (job/f'attempt_{attempt:04d}.log').exists():
                attempt += 1
            directory = job/f'attempt_{attempt:04d}'
            log = job/f'attempt_{attempt:04d}.log'
            command = [sys.executable, '-u', str(ROOT/'run.py'), '--dataset', dataset,
                       '--model', args.model, '--mode', 'component_joint', '--seed', str(seed),
                       '--device', args.device, '--output', str(directory)]
            if args.epochs is not None:
                command += ['--epochs', str(args.epochs)]
            write_json(job/'status.json', dict(status='RUNNING', started_at=now(), attempt=directory.name))
            print(f'[{index}/{total}] RUN {dataset} seed={seed} log={log}', flush=True)
            try:
                code = execute(command, log)
                if code:
                    raise RuntimeError(f'Worker exited with status {code}; inspect {log}')
                validate_run(directory, dataset, seed, plan)
            except KeyboardInterrupt:
                write_json(job/'status.json', dict(status='INTERRUPTED', finished_at=now(), attempt=directory.name))
                raise
            except Exception as exc:
                failed += 1
                write_json(job/'status.json', dict(status='FAILED', finished_at=now(),
                           attempt=directory.name, error=str(exc)))
                print(f'[{index}/{total}] FAILED {dataset} seed={seed}: {exc}', flush=True)
            else:
                completed = dict(status='COMPLETE', finished_at=now(), attempt=directory.name)
                write_json(marker, completed)
                write_json(job/'status.json', completed)
                print(f'[{index}/{total}] COMPLETE {dataset} seed={seed}', flush=True)
            collect(output, plan)
        rows, errors = collect(output, plan)
        expected = total*len(plan['models'])
        print(f'Completed {len(rows)}/{expected} model runs; summary: {output / "summary.csv"}', flush=True)
        return 1 if failed or errors or len(rows) != expected else 0
    except KeyboardInterrupt:
        collect(output, plan)
        print('Interrupted; rerun the same command with --resume', flush=True)
        return 130
    finally:
        lock.unlink(missing_ok=True)


def interrupt(signum, frame):
    raise KeyboardInterrupt()


if __name__ == '__main__':
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
