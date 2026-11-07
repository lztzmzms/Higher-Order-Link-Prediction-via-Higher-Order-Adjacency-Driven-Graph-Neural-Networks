import ast
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from historical_components.generators import create_generator
from historical_components.interfaces import ComponentGenerator
from pipelines import historynet_joint

ROOT = Path(__file__).resolve().parents[1]

class HistoryNetGeneratorContracts(unittest.TestCase):
    def fixture_prepare(self, dataset, dest):
        dest.mkdir(parents=True)
        for role in ('fit', 'target'):
            np.savez_compressed(dest/f'{role}.npz', q=np.array([[0, 1, 2], [1, 2, 3]]))

    def fixture_generate(self, dataset, prepared, folder, config):
        folder.mkdir()
        names = np.array(['HistoryNet_logit_seed_41', 'HistoryNet_logit_seed_42'])
        for role in ('fit', 'target'):
            np.savez_compressed(folder/f'{role}_raw.npz', q=np.array([[0, 1, 2], [1, 2, 3]]),
                                component_x=np.array([[-2., 4.], [1., -3.]]), names=names)
        recipe = dict(recipe='tags-math-sx HistoryNet explicit transfer',
                      parameter_gates={role:dict(prediction_cut=cut, max_training_label_end=cut)
                                       for role,cut in [('fit', 60), ('target', 80)]})
        (folder/'recipe.json').write_text(json.dumps(recipe), encoding='utf-8')

    def test_registry_routes_resolve_for_all_eight_datasets(self):
        registry = json.loads((ROOT/'configs/dataset_registry.json').read_text())
        for row in registry['datasets']:
            with self.subTest(dataset=row['dataset']):
                cfg = json.loads((ROOT/row['config']).read_text())
                generator = create_generator(row['component_generator'], cfg.get('component_generator_config'))
                self.assertIsInstance(generator, ComponentGenerator)
                self.assertEqual(generator.dataset, row['dataset'])
                self.assertEqual(cfg['component_generator'], row['component_generator'])

    def test_transfer_generator_returns_raw_aligned_batches(self):
        for dataset in ('NDC-substances', 'threads-math-sx'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'output'
                with patch.object(historynet_joint, 'prepare', side_effect=self.fixture_prepare), \
                     patch.object(historynet_joint, 'generate', side_effect=self.fixture_generate):
                    batches = create_generator(dataset).generate(output_dir=output)
                for role in ('fit', 'target'):
                    np.testing.assert_array_equal(batches[role].values, [[-2., 4.], [1., -3.]])
                    self.assertEqual(batches[role].mask_indices, ())
                    self.assertEqual(len(batches[role].names), 2)
                    self.assertLessEqual(batches[role].metadata['max_training_label_end'],
                                         batches[role].metadata['prediction_cut'])
                with self.assertRaises(FileExistsError):
                    create_generator(dataset).generate(output_dir=output)

    def test_standalone_cli_accepts_both_transfer_datasets(self):
        for dataset in ('NDC-substances', 'threads-math-sx'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'cli'
                with patch.object(historynet_joint, 'prepare', side_effect=self.fixture_prepare), \
                     patch.object(historynet_joint, 'generate', side_effect=self.fixture_generate), \
                     patch('sys.argv', ['generate_components.py', '--dataset', dataset, '--output', str(output)]):
                    runpy.run_path(str(ROOT/'scripts/generate_components.py'), run_name='__main__')
                self.assertTrue((output/'components/target_raw.npz').is_file())

    def test_transfer_rejects_misaligned_ids_and_arbitrary_context(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'output'
            generator = create_generator('NDC-substances')
            def misaligned(*args):
                self.fixture_generate(*args)
                path = args[2]/'target_raw.npz'
                with np.load(path) as values:
                    payload = {k:values[k].copy() for k in values.files}
                payload['q'] = payload['q'][::-1]
                np.savez_compressed(path, **payload)
            with patch.object(historynet_joint, 'prepare', side_effect=self.fixture_prepare), \
                 patch.object(historynet_joint, 'generate', side_effect=misaligned):
                with self.assertRaises(AssertionError):
                    generator.generate(output_dir=output)
            with self.assertRaises(ValueError):
                generator.generate(context=object())

class SupplementaryServerPlanContracts(unittest.TestCase):
    def test_scheduling_is_separate_from_observed_state(self):
        paths = list(ROOT.glob('supplementary_sources/*/src/project/run_strict_timestamp_t2_mlp_r1_dose_local_v1.py'))
        self.assertEqual(len(paths), 2)
        for path in paths:
            with self.subTest(bundle=path.parts[-5]):
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_server_fallback')
                primary, optional = (2, 'primary'), (3, 'optional')
                writes = {}
                namespace = dict(ALL_CASES=[primary, optional], PRIMARY_CASES=[primary], OPTIONAL_CASES=[optional],
                    case_key=lambda order,dataset:f'{order}:{dataset}', OUT_ROOT=Path('output'),
                    write_json=lambda name,value:writes.update({str(name):value}))
                exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
                run = namespace['generate_server_fallback']
                pending = run({})
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0]['execution_plan'], 'SERVER')
                self.assertEqual(pending[0]['execution_status'], 'NOT_RUN')
                self.assertFalse(pending[0]['resource_limit_observed'])
                self.assertNotIn('LOCAL_RESOURCE_LIMIT', pending[0])
                self.assertEqual(run({'3:optional':dict(status='COMPLETE')}), [])
                failed = run({'2:primary':dict(status='RESOURCE_LIMIT', reason='observed failure'),
                              '3:optional':dict(status='COMPLETE')})
                self.assertEqual(len(failed), 1)
                self.assertTrue(failed[0]['resource_limit_observed'])
                self.assertFalse(failed[0]['scheduled_server_case'])
                self.assertEqual(failed[0]['reason'], 'observed failure')
                engineering = run({'3:optional':dict(status='ENGINEERING_FAILURE', reason='input error')})
                self.assertEqual(engineering[0]['execution_status'], 'ENGINEERING_FAILURE')
                self.assertFalse(engineering[0]['resource_limit_observed'])

if __name__ == '__main__':
    unittest.main()
