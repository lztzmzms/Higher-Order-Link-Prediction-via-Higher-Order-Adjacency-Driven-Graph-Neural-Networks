import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import rankdata

from .builtin import DatasetSpecificComponentGenerator, SEEDS, SIGNALS, recipe
from .engine import build_engine, definitions

def regenerate_final_h(dataset, output_dir):
    pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    gen = DatasetSpecificComponentGenerator(dataset)
    from .builtin import read_events
    with tempfile.TemporaryDirectory(prefix='audit-H-', dir=output_dir) as runtime:
        events, cohorts = read_events(dataset), gen._cohorts()
        engine = build_engine(dataset, runtime, raw_members=False)
        scores = {}
        if dataset in ('email-Eu', 'tags-ask-ubuntu', 'contact-primary-school'):
            snaps, _ = gen._snapshots(engine, events, cohorts)
            for f, role in ((.6, 'fit'), (.8, 'target')):
                cfg = recipe('primary')['config'] if dataset == 'contact-primary-school' else recipe(dataset + '_' + role)['route']['config']
                h, gate = engine.rank.predict(snaps[f], cfg, SEEDS)
                scores[role] = dict(q=snaps[f]['q'], h=h, metadata=dict(parameter_gate=gate))
        elif dataset == 'email-Enron':

            snaps, bases = gen._snapshots(engine, events, cohorts)
            batches = gen._enron(engine, snaps, bases)
            selected = recipe('enron')['selected']
            for role, batch in batches.items():
                raw = batch.values
                mean_hg = raw[:, :3].mean(axis=1)
                backbone = .8 * rankdata(mean_hg) / len(raw) + .2 * rankdata(raw[:, 3]) / len(raw)
                h, movement = engine.bounded.rearrange(backbone, raw[:, 4:7].mean(axis=1),
                    selected['config']['pool'], selected['alpha'], selected['max_move'])
                scores[role] = dict(q=batch.q, h=h, metadata=dict(movement=movement))
        elif dataset == 'contact-high-school':
            batches = gen._highschool(engine, events, cohorts)
            rules = definitions('highschool_rules', ['rank', 'replay'], dict(SIGNALS=SIGNALS))
            for role, batch in batches.items():

                source = {name: batch.values[:, j] for j, name in enumerate(SIGNALS)}
                for model, model_recipe in recipe('highschool')[dataset]['recipes'].items():
                    h = rules.replay(model_recipe['recipe'], source, dict.fromkeys(SIGNALS, 1))
                    scores[role + '_' + model] = dict(q=batch.q, h=h, metadata=dict(
                        final_labels_used_for_original_recipe_selection=False))
        elif dataset == 'tags-math-sx':
            batches = gen._tags(engine, events, cohorts)
            for role, batch in batches.items():
                mean = batch.values.mean(axis=1)
                h = 1. / (1. + np.exp(-np.clip(mean, -40, 40)))
                scores[role] = dict(q=batch.q, h=h, metadata={})
        else:
            raise ValueError('No recovered original H route')
        for name, item in scores.items():
            np.savez_compressed(output_dir / (name + '.npz'), q=item['q'], h=item['h'])
            (output_dir / (name + '_AUDIT.json')).write_text(json.dumps(dict(
                dataset=dataset, role=name, **item['metadata'], optional_audit_output_only=True,
                not_a_joint_model_input=True, final_evaluation_labels_used=False,
                status='generated_by_explicit_user_command; no release performance claim'), ensure_ascii=False, indent=2), encoding='utf-8')
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    regenerate_final_h(args.dataset, args.output)
