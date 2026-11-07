import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import rankdata

from historical_components.interfaces import ComponentBatch, ComponentGenerator
from .engine import UPSTREAM, build_engine, definitions, feature_matrix

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school',
            'tags-ask-ubuntu', 'tags-math-sx')
SEEDS = (41, 42, 43)
ENRON_LEGACY = dict(profile='dense', view='decay', family='HG', leaf=40, depth=3,
                    fraction=.5, balanced=True, rank=True, recent_weight=0., C=.001,
                    learning_rate=.02, l2=0., iterations=150)
SIGNALS = ('temporal/edge_ratio_logprod', 'calendar/clock_overlap_min',
           'temporal/edge_temporal_overlap_w0.01_logsum',
           'temporal/edge_temporal_overlap_w0.005_mean')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (keys if keys is not None else z.files)}

def read_events(dataset):
    ev = arrays(ROOT / 'data' / dataset / 'events.npz', ('times', 'counts', 'vertices'))
    offsets = np.r_[0, np.cumsum(ev['counts'])]
    if len(ev['times']) != len(ev['counts']) or offsets[-1] != len(ev['vertices']):
        raise ValueError('Malformed preserved temporal event data')
    if np.any(ev['times'][1:] < ev['times'][:-1]):
        raise ValueError('Events must preserve chronological order and empty positions')
    return [(int(t), i, tuple(map(int, ev['vertices'][offsets[i]:offsets[i + 1]])))
            for i, t in enumerate(ev['times'])]

def recipe(name):
    return read_json(UPSTREAM / 'recipes' / (name + '.json'))

def member_names():
    return [f'supervised_preensemble_seed_{s}' for s in SEEDS]

class DatasetSpecificComponentGenerator(ComponentGenerator):
    def __init__(self, dataset, config=None):
        if dataset not in DATASETS:
            raise ValueError(f'No recovered original recipe for dataset {dataset!r}')
        self.dataset = dataset
        self.config = config or {}

    def generate_for_context(self, context):
        raise ValueError('Built-in recipes bind the original event cohort, historical windows and candidate IDs. '
                         'Use an explicit custom ComponentGenerator for a new dataset or cohort.')

    def _cohorts(self):
        rows = read_json(UPSTREAM / 'cohorts/INDEX.json')[self.dataset]
        result = {}
        for row in rows:
            data = arrays(UPSTREAM / 'cohorts' / self.dataset / row['file'], ('q', 'cut'))
            digest = hashlib.sha256(np.ascontiguousarray(data['q']).tobytes()).hexdigest()
            if digest != row['candidate_sha256'] or int(data['cut']) != row['cut']:
                raise ValueError('Original historical candidate identity/order was changed')
            result[row['file'].removesuffix('.npz')] = {**data, **row}
        return result

    def _snapshots(self, engine, events, cohorts):
        snaps, bases = {}, {}
        max_cut = int(cohorts['80']['cut'])
        for key, record in sorted(cohorts.items(), key=lambda item: item[1]['fraction']):
            f, cut = float(record['fraction']), int(record['cut'])
            if cut > max_cut:
                raise ValueError('Snapshot crosses final prediction time')
            q = record['q']
            x, names, old_x, base_x, base_names = feature_matrix(engine, self.dataset, events[:cut], q)
            end = record['label_end']
            if end is None and f < .8:
                end = engine.temporal.cutoff(events, round(f + .2, 2))
            if end is not None and int(end) > max_cut:
                raise ValueError('Attempt to use final evaluation labels in component generation')
            prior = engine.temporal.cutoff(events, round(f - .2, 2)) if f > .2 else 0
            units = 1000. if self.dataset in ('email-Enron', 'tags-ask-ubuntu') else 1.
            snap = dict(q=q, cut=np.asarray(cut), fraction=f, label_end=end, x=x, names=names,
                        old_x=old_x,
                        duration_proxy_seconds=max(1., (events[cut - 1][0] - events[prior][0]) / units),
                        completed_label_duration_seconds=(max(1., (events[int(end) - 1][0] - events[cut][0]) / units)
                                                          if end is not None else None))
            if end is not None:
                snap['y'] = engine.temporal.attach_labels(events[cut:int(end)], [tuple(map(int, row)) for row in q])
            base = {**snap, 'x': base_x, 'names': base_names}
            snaps[f], bases[f] = snap, base
        engine.rank.DATA[self.dataset] = dict(snaps=snaps, events=events[:max_cut])
        engine.source.DATA[self.dataset] = {'dense': [v for f, v in bases.items() if f < .8]}
        return snaps, bases

    def _align(self, outputs):
        target = arrays(ROOT / 'data' / self.dataset / 'candidates.npz', ('train_q', 'target_q'))
        aligned = {}
        for role, batch in outputs.items():
            wanted = target['train_q' if role == 'fit' else 'target_q']
            lookup = {tuple(row): i for i, row in enumerate(batch.q)}
            try:
                ids = np.asarray([lookup[tuple(row)] for row in wanted], dtype=np.int64)
            except KeyError as exc:
                raise ValueError('Original model candidate absent from generated component cohort') from exc
            if not np.array_equal(batch.q[ids], wanted):
                raise ValueError('Candidate ID alignment failed')
            aligned[role] = ComponentBatch(wanted, batch.values[ids], batch.names, batch.mask_indices,
                {**batch.metadata, 'component_cohort_rows': len(batch.q), 'output_rows': len(wanted),
                 'candidate_alignment': 'exact sorted tuple lookup; output in model candidate order'})
        return aligned

    def _transfer(self, engine, snaps):
        outputs = {}
        for f, role in ((.6, 'fit'), (.8, 'target')):
            target = snaps[f]
            policy = self.config.get('recipe_policy', 'frozen_fit')
            if policy not in ('frozen_fit', 'original_stage'):
                raise ValueError('Unknown recipe policy')
            recipe_role = 'fit' if policy == 'frozen_fit' else role
            route_lock = recipe(self.dataset + '_' + recipe_role)
            route = route_lock['route']
            if route['kind'] != 'current':
                raise ValueError('Unsupported changed archived route; provide an explicit recovered generator')
            cfg = route['config']
            members, gate = engine.rank.predict(target, cfg, SEEDS)
            if int(gate['max_training_label_end']) > int(target['cut']):
                raise ValueError('Supervised component training exceeds prediction cutoff')
            active = float(bool(cfg['fusion']))
            group = target['x'][:, target['names'].tolist().index('continue_edge_sync_triple_group_64')]
            values = np.column_stack([members, group * active, np.full(len(group), active)])
            if self.dataset == 'tags-ask-ubuntu':

                values = values.astype(np.float32)
            names = member_names() + ['group_rule_if_active', 'group_rule_active_mask']
            outputs[role] = ComponentBatch(target['q'], values, names, (4,), dict(
                dataset=self.dataset, role=role, route=route, parameter_gate=gate,
                recipe_policy=policy, recipe_origin=recipe_role,
                archived_stage_route=recipe(self.dataset + "_" + role)["route"],
                raw_members_before_rank_mean_fusion=True, rule_active=bool(active),
                route_selection_label_end=route_lock.get('maximum_hyperparameter_label_end', route_lock.get('max_selection_label_end')),
                inherited_node_cohort_bias=True, strict_pipeline_causality_proven=False,
                final_evaluation_labels_used=False, status='generated'))
        return outputs

    def _primary(self, engine, snaps):
        cfg = recipe('primary')['config']
        outputs = {}
        for f, role in ((.6, 'fit'), (.8, 'target')):
            target = snaps[f]
            values, gate = engine.rank.predict(target, cfg, SEEDS)
            if gate['max_training_label_end'] > int(target['cut']):
                raise ValueError('Historical component training exceeds prediction time')
            outputs[role] = ComponentBatch(target['q'], values,
                [f'PAIR_LR_decision_seed_{s}' for s in SEEDS], (), dict(
                    dataset=self.dataset, role=role, upstream_config=cfg, parameter_gate=gate,
                    inherited_outcome_stopping_or_selection_bias=True,
                    strict_pipeline_causality_proven=False, final_evaluation_labels_used=False,
                    status='generated'))
        return outputs

    def _enron(self, engine, snaps, bases):
        records, components, gates = [], {}, {}
        for f in (.35, .4, .45, .5, .55, .6, .8):
            target = snaps[f]
            scores, per_seed = [], []
            for seed in SEEDS:
                score, gate = engine.legacy.predict(bases[f], ENRON_LEGACY, (seed,))
                if gate['max_training_label_end'] > int(target['cut']):
                    raise ValueError('HG component label boundary exceeded')
                scores.append(score)
                per_seed.append(gate)
            names = target['names'].tolist()
            rule = target['x'][:, names.index('continue_edge_sync_triple_group_64')]
            hg = np.mean(scores, axis=0)

            backbone = .8 * rankdata(hg) / len(hg) + .2 * rankdata(rule) / len(rule)
            columns = [i for i, n in enumerate(names)
                       if n.startswith(('event_', 'clock_', 'community_', 'continue_edge_sync_', 'continue_decay_'))
                       and n.endswith(('_min', '_mean', '_harmonic', '_logsum'))]
            forced = [names.index(n) for n in ('community_same_pairs', 'continue_edge_sync_triple_group_64')]
            columns = list(dict.fromkeys(forced + columns))
            record = dict(x=np.column_stack([rankdata(backbone) / len(backbone), target['x'][:, columns]]),
                          names=['backbone_percentile'] + [names[i] for i in columns], base=backbone,
                          q=target['q'], cut=int(target['cut']), label_end=target['label_end'], fraction=f)
            if 'y' in target:
                record['y'] = target['y']
            records.append(record)
            components[f] = scores + [rule]
            gates[f] = per_seed
        selected = recipe('enron')['selected']
        outputs = {}
        for f, role in ((.6, 'fit'), (.8, 'target')):
            target = next(r for r in records if r['fraction'] == f)
            pool = records if f == .6 else [engine.bounded.localized(r, selected['config']) for r in records]
            fit_target = target if f == .6 else engine.bounded.localized(target, selected['config'])
            local, local_gates = [], []
            for seed in SEEDS:
                score, gate = engine.local.fit_score(engine.rank, pool, fit_target, selected['config'], (seed,), force=True)
                if gate['max_meta_label_end'] > target['cut']:
                    raise ValueError('Local PAIR_LR label boundary exceeded')
                if gate.get('fallback'):
                    raise ValueError('Original predictor fallback cannot be presented as a genuine PAIR_LR component')
                local.append(score)
                local_gates.append(gate)
            names = [f'HG_probability_seed_{s}' for s in SEEDS] + ['group_sync64_rule'] + [f'PAIR_LR_decision_seed_{s}' for s in SEEDS]
            domain = np.argsort(-target['base'], kind='stable')[:min(len(target['base']), selected['config']['pool'])]
            outputs[role] = ComponentBatch(target['q'], np.column_stack(components[f] + local), names, (), dict(
                dataset=self.dataset, role=role, hg_parameter_gates=gates[f], local_parameter_gates=local_gates,
                pair_lr_training_scope='original historical top200',
                all_candidate_component_prediction=True,
                pair_lr_score_scope='original fit_score decision_function evaluates all target rows',
                original_final_H_rerank_scope_ids=domain.tolist(),
                extrapolation_note='Scores outside original top200 are genuine model extrapolations; their predictive validity is not independently established.',
                original_final_H_not_generated=True, final_evaluation_labels_used=False,
                inherited_recipe_selection_max_label_end=8706, strict_pipeline_causality_proven=False,
                status='generated'))
        return outputs

    def _highschool(self, engine, events, cohorts):
        rules = definitions('highschool_rules', ['signal_values', 'rank', 'direction_selection', 'replay'], dict(SIGNALS=SIGNALS))
        snapshots = {int(round(r['fraction'] * 100)): r for r in cohorts.values()}
        values = {origin: rules.signal_values(events[:int(record['cut'])], record['q'])
                  for origin, record in snapshots.items()}
        outputs = {}
        for origin, prior, role in ((60, 40, 'fit'), (80, 60, 'target')):
            previous, current = snapshots[prior], snapshots[origin]
            labels = engine.temporal.attach_labels(events[int(previous['cut']):int(current['cut'])],
                                                    [tuple(map(int, q)) for q in previous['q']])
            directions, gate = rules.direction_selection(values[prior], labels, int(current['cut']), int(current['cut']))

            signed = np.column_stack([directions[name] * np.nan_to_num(values[origin][name]) for name in SIGNALS])
            outputs[role] = ComponentBatch(current['q'], signed, SIGNALS, (), dict(
                dataset=self.dataset, role=role, directions=directions, direction_gate=gate,
                original_recipes=recipe('highschool')[self.dataset]['recipes'],
                component_scope='union of actual Dual-Mapping/MLSHoL source rule members, before nested rank fusion',
                final_labels_used_for_original_recipe_selection=False,
                final_evaluation_labels_used_for_new_component_generation=False,
                strict_pipeline_causality_proven=False, status='generated'))
        return outputs

    def _tags(self, engine, events, cohorts):
        from historical_components.upstream import tag_predictor
        from historical_components.upstream.tag_sampling import capsnap
        features = definitions('tag_features', ['History'])
        records = {}
        max_cut = int(cohorts['test']['cut'])
        for key, row in cohorts.items():
            cut = int(row['cut'])
            x, names = features.History(events[:cut]).features(row['q'])

            allowed_end = row['label_end'] if key != 'test' else None
            item = dict(q=row['q'], x=x, names=names, cut=cut, label_end=allowed_end)
            if allowed_end is not None:
                end = int(allowed_end)
                if end > max_cut:
                    raise ValueError('Historical tags labels exceed final prediction time')
                item['y'] = engine.temporal.attach_labels(events[cut:end], [tuple(map(int, q)) for q in row['q']])
            records[key] = item
        historical = [v for key, v in records.items() if key not in ('train', 'test')]
        historical.sort(key=lambda row: row['cut'])
        cfg = dict(ensemble_seeds=[41, 42], epochs=20, width=128, weight_power=.5)
        outputs = {}
        for key, role in (('train', 'fit'), ('test', 'target')):
            target = records[key]
            x, y, gate = capsnap(historical, target['cut'], 25000)
            members = []
            for seed in cfg['ensemble_seeds']:
                _, _, prediction, _ = tag_predictor.history_fit(x, y, target['x'], cfg, seed, cfg['epochs'])
                members.append(prediction)
            outputs[role] = ComponentBatch(target['q'], np.column_stack(members),
                [f'HistoryNet_logit_seed_{s}' for s in cfg['ensemble_seeds']], (), dict(
                    dataset=self.dataset, role=role, upstream_config=cfg, parameter_gate=gate,
                    source_features_rebuilt=True, final_H_sigmoid_mean_not_used=True,
                    final_evaluation_labels_used=False, inherited_cohort_limitations=True,
                    status='generated'))
        return outputs

    def generate(self, output_dir=None, context=None):
        if context is not None:
            return self.generate_for_context(context)
        output = Path(output_dir) if output_dir is not None else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='component-runtime-', dir=output) as runtime:
            engine = build_engine(self.dataset, runtime)
            events, cohorts = read_events(self.dataset), self._cohorts()
            if self.dataset == 'contact-high-school':
                result = self._highschool(engine, events, cohorts)
            elif self.dataset == 'tags-math-sx':
                result = self._tags(engine, events, cohorts)
            else:
                snaps, bases = self._snapshots(engine, events, cohorts)
                if self.dataset == 'email-Enron':
                    result = self._enron(engine, snaps, bases)
                elif self.dataset == 'contact-primary-school':
                    result = self._primary(engine, snaps)
                else:
                    result = self._transfer(engine, snaps)
            result = self._align(result)
            if output is not None:
                for role, batch in result.items():
                    np.savez_compressed(output / (role + '.npz'), q=batch.q, component_x=batch.values,
                                        component_names=np.asarray(batch.names), mask_indices=np.asarray(batch.mask_indices, dtype=np.int64))
                    (output / (role + '_METADATA.json')).write_text(json.dumps(batch.metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            return result
