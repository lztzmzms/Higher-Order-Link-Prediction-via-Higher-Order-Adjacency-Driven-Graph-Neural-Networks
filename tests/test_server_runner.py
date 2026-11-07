import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from evaluate import evaluate
from run import configured_dataset, existing

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('server_runner', ROOT/'scripts/run_server.py')
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class ServerRunnerContracts(unittest.TestCase):
    def test_epoch_override_keeps_configs_and_upstream_budgets(self):
        for dataset in server.DATASETS:
            with self.subTest(dataset=dataset):
                path = ROOT/'configs'/f'{dataset}.json'
                before = path.read_bytes()
                original = configured_dataset(dataset)
                current = configured_dataset(dataset, 10)
                self.assertTrue(all(m['epochs'] == 10 for m in current['models'].values()))
                self.assertEqual(current['component_generator_config'], original['component_generator_config'])
                self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            configured_dataset(server.DATASETS[0], 0)

    def test_epoch_override_reaches_historynet_dispatch(self):
        from types import SimpleNamespace
        for dataset in ('NDC-substances', 'threads-math-sx'):
            with patch('pipelines.historynet_joint.run') as worker:
                existing(SimpleNamespace(dataset=dataset, epochs=10))
                cfg = worker.call_args.args[1]
                self.assertTrue(all(m['epochs'] == 10 for m in cfg['models'].values()))
                self.assertEqual(cfg['component_generator_config']['epochs'], 20)

    def test_dry_run_counts_both_interpretations_without_writing(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch.object(server, 'source_hashes', return_value={}):
            output = Path(directory)/'output'
            self.assertEqual(server.main(['--runs', '10', '--output', str(output), '--dry-run']), 0)
            self.assertFalse(output.exists())
            self.assertEqual(server.main(['--runs', '1', '--epochs', '10', '--output', str(output), '--dry-run']), 0)
            self.assertFalse(output.exists())

    def fixtures(self, root):
        q = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        y = np.array([0, 1, 0, 1])
        for dataset in ('email-Enron', 'email-Eu'):
            path = root/'configs'/f'{dataset}.json'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((ROOT/'configs'/f'{dataset}.json').read_bytes())
        calls = []
        failures = {('email-Enron', 9102)}
        def execute(command, log):
            def argument(name):
                return command[command.index(name)+1]
            dataset, seed = argument('--dataset'), int(argument('--seed'))
            calls.append((dataset, seed))
            output = Path(argument('--output'))
            output.mkdir()
            log.write_text('worker fixture', encoding='utf-8')
            if (dataset, seed) in failures:
                failures.remove((dataset, seed))
                return 9
            rows = []
            for model in server.MODELS[argument('--model')]:
                score = np.array([.1, .7, .3, .5])+seed*.00001
                np.savez_compressed(output/f'{model}_{seed}.npz', q=q, y=y, predictions=score)
                (output/f'{model}_{seed}.pt').write_bytes(b'checkpoint-fixture')
                history = [dict(epoch=e, optimizer_updates=e, candidate_visits=4*e, weighted_BCE=.7)
                           for e in range(1, int(argument('--epochs'))+1)]
                (output/f'{model}_{seed}_training.json').write_text(json.dumps(history))
                rows.append(dict(dataset=dataset, model=model, seed=seed, mode='component_joint', **evaluate(y, score)))
            (output/'metrics.json').write_text(json.dumps(rows))
            return 0
        return q, y, calls, execute

    def test_failed_pairs_resume_without_repeating_completed_pairs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            q, y, calls, worker = self.fixtures(root)
            output = root/'results'
            command = ['--datasets', 'email-Enron', 'email-Eu', '--runs', '2', '--epochs', '2',
                       '--device', 'cpu', '--output', str(output)]
            with patch.object(server, 'ROOT', root), patch.object(server, 'source_hashes', return_value={'source': 'fixed'}), \
                 patch.object(server, 'execute', side_effect=worker), \
                 patch.object(server, 'reference_candidates', return_value=(q, y)):
                self.assertEqual(server.main(command), 1)
                first = json.loads((output/'summary.json').read_text())
                self.assertEqual(first['completed_model_runs'], 6)
                self.assertEqual(server.main(command+['--resume']), 0)
                self.assertEqual(len(calls), 5)
                self.assertEqual(calls.count(('email-Enron', 9101)), 1)
                self.assertEqual(calls.count(('email-Enron', 9102)), 2)
                self.assertTrue((output/'email-Enron/seed_9102/attempt_0001.log').is_file())
                self.assertTrue((output/'email-Enron/seed_9102/attempt_0002').is_dir())
                final = json.loads((output/'summary.json').read_text())
                self.assertEqual(final['completed_model_runs'], 8)
                self.assertTrue(all(r['n_runs'] == 2 for r in final['results']))
                self.assertFalse((output/'run.lock').exists())
                with self.assertRaises(SystemExit):
                    server.main(command[:-2]+['--output', str(output)])
                with patch.object(server, 'source_hashes', return_value={'source': 'changed'}):
                    with self.assertRaises(SystemExit):
                        server.main(command+['--resume'])

    def test_resume_rejects_wrong_candidate_ids_and_incomplete_history(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            q, y, calls, worker = self.fixtures(root)
            output = root/'run'
            command = ['worker', '--dataset', 'email-Enron', '--seed', '9101', '--model', 'both',
                       '--epochs', '2', '--output', str(output)]
            worker(command, root/'run.log')
            plan = dict(models=list(server.MODELS['both']), epochs={'email-Enron':dict.fromkeys(server.MODELS['both'], 2)})
            with patch.object(server, 'reference_candidates', return_value=(q, y)):
                rows = server.validate_run(output, 'email-Enron', 9101, plan)
                self.assertEqual(len(rows), 2)
                path = output/'Dual-Mapping_9101.npz'
                np.savez_compressed(path, q=q[::-1], y=y, predictions=np.array([.1, .7, .3, .5]))
                with self.assertRaises(ValueError):
                    server.validate_run(output, 'email-Enron', 9101, plan)
                np.savez_compressed(path, q=q, y=y, predictions=np.array([.1, .7, .3, .5]))
                (output/'Dual-Mapping_9101_training.json').write_text('[]')
                with self.assertRaises(ValueError):
                    server.validate_run(output, 'email-Enron', 9101, plan)


if __name__ == '__main__':
    unittest.main()
