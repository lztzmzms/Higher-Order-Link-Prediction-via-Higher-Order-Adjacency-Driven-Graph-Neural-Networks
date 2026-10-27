"""Reproduce both RW-enabled final models on one original dataset protocol."""
import argparse
from data.prepare import DATASETS, MODELS, ROOT, config, material, evaluation_labels
from evaluate import evaluate, save_results
from pipelines.dispatch import run_model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:0')
    parser.add_argument('--model', choices=['dual','mlshol','Dual-Mapping','MLSHoL'])
    parser.add_argument('--check', action='store_true', help='Check original inputs and configuration without training')
    args = parser.parse_args()
    cfg = config(args.dataset)
    models=MODELS if args.model is None else (dict(dual='Dual-Mapping',mlshol='MLSHoL').get(args.model,args.model),)
    if args.check:
        if cfg.get('pipeline')=='native':
            from pipelines.inputs import native_check
            native_check(args.dataset)
        else:
            for model in models:material(args.dataset,model)
        print(f'{args.dataset}: configuration and original input checks passed')
        return
    for model in models:
        data = None if cfg.get('pipeline')=='native' else material(args.dataset, model)
        summary = None
        for seed in cfg['models'][model]['seeds']:
            labels, predictions = run_model(args.dataset, model, seed, args.device, data)
            row = dict(dataset=args.dataset, model=model, seed=seed,
                       **evaluate(labels, predictions))
            print(f"{args.dataset},{model},{seed},AP={row['AP']:.9f},Lift={row['Lift']:.9f}", flush=True)
            summary = save_results(ROOT/'results', row)
        if summary is not None:
            print(f"{args.dataset},{model},AP_mean={summary['AP_mean']:.9f},"
                  f"AP_std={summary['AP_std']:.9f},Lift_mean={summary['Lift_mean']:.9f}", flush=True)

if __name__ == '__main__':
    main()
