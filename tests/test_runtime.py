import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from train import fit_model, predict, roles_for, update, optimizer
from models import network
from data.prepare import geometry
from historical_components.encoders import PositiveComponentEncoder
from historical_components.generators.builtin import DatasetSpecificComponentGenerator
from simplicial_complex.data_loading import DataLoader

class RuntimeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = dict(hidden=16, depth=1, dropout=0., gat_heads=4, gnn='GAT',
                       lr=.001, structure_lr_ratio=.1, weight_decay=0.,
                       batch_size=3, epochs=2, loss_weight=1., rw_enabled=True)
        cls.q = {r: np.array([[i, i+1, i+2] for i in range(6)]) for r in ('fit','target')}
        cls.y = np.array([1, 0, 0, 0, 0, 1])
        cls.x = {r: np.random.default_rng(41).normal(size=(6, 3)) for r in cls.q}

    def roles(self, model):
        structure, rw = geometry('email-Enron', model, torch.device('cpu'))
        rw = {r: rw[r][:6] for r in rw}
        if model == 'Dual-Mapping':
            for r, block in structure.items():
                for key in ('local','heuristic','rw'):
                    block[key] = block[key][:6]
                edge = block['edge_index']
                block['edge_index'] = edge[:, (edge < 6).all(0)]
        else:

            pass
        return roles_for(model, structure, rw, self.q, self.x)

    def test_both_models_train_save_reload_predict_and_budget(self):
        for model in ('Dual-Mapping','MLSHoL'):
            for enabled in (True, False):
                with self.subTest(model=model, rw=enabled):
                    roles = self.roles(model)
                    cfg = dict(self.cfg, rw_enabled=enabled)
                    net, score = fit_model(model, roles, self.y, cfg, 17)
                    self.assertEqual(net.training_history[-1]['candidate_visits'], 12)
                    self.assertEqual(net.training_history[-1]['optimizer_updates'], 2)
                    self.assertTrue(np.isfinite(score).all())
                    stream = io.BytesIO()
                    torch.save(net.state_dict(), stream)
                    stream.seek(0)
                    restored = network.Network(model, roles['fit'], cfg)
                    restored.load_state_dict(torch.load(stream, weights_only=True))
                    from models.singleton_batchnorm import install
                    _, reset = install()
                    try:
                        np.testing.assert_array_equal(score, predict(restored, roles['target']))
                    finally:
                        reset()
                    self.assertEqual(net.h_encoder.first.raw_weight.shape[0], 16)
                    if not enabled:
                        before = predict(net, roles['target'])
                        roles['target']['rw'] = roles['target']['rw'] * 100 + 999
                        roles['target']['block']['rw'] = roles['target']['block']['rw'] * 100 + 999
                        np.testing.assert_array_equal(before, predict(net, roles['target']))

    def test_sparse_sin_default_forward(self):
        roles = self.roles('MLSHoL')
        net = network.StructureNetwork('MLSHoL', roles['fit'], self.cfg).eval()
        packed = roles['fit']['complexes']
        data = next(iter(DataLoader([packed[i] for i in range(3)], batch_size=3, max_dim=2)))
        with torch.no_grad():
            output = net.core(data)
        self.assertEqual(tuple(output.shape), (3, 2))
        torch.testing.assert_close(output.exp().sum(-1), torch.ones(3))

    def test_initialization_breaks_hidden_symmetry(self):
        torch.manual_seed(41)
        encoder = PositiveComponentEncoder(7).double()
        x = torch.randn(96, 7, dtype=torch.float64)
        z = encoder(x).detach()
        self.assertGreater(float(z.std(dim=1).mean()), 1e-4)
        self.assertGreater(int(torch.linalg.matrix_rank(z-z.mean(0), atol=1e-9)), 1)
        self.assertGreater(float(encoder.first.raw_weight.std()), .01)

    def test_loss_weight_and_microbatch_full_objective(self):
        roles = self.roles('Dual-Mapping')

        a = network.Network('Dual-Mapping', roles['fit'], dict(self.cfg, batch_size=2))
        b = network.Network('Dual-Mapping', roles['fit'], dict(self.cfg, batch_size=5, loss_weight=2.))
        b.load_state_dict(a.state_dict())
        la = update(a, optimizer(a, a.cfg), roles['fit'], self.y)
        lb = update(b, optimizer(b, b.cfg), roles['fit'], self.y)
        self.assertAlmostEqual(lb, 2*la, places=10)

    def test_dual_backward_traverses_shared_graph_once(self):
        roles = self.roles('Dual-Mapping')
        cfg = dict(self.cfg, batch_size=2)
        net = network.Network('Dual-Mapping', roles['fit'], cfg)
        calls, shared_gradient_calls = [], []
        original_backward = torch.Tensor.backward
        def counted(tensor, *args, **kwargs):
            calls.append(kwargs)
            return original_backward(tensor, *args, **kwargs)
        parameter = next(p for p in net.core.parameters() if p.requires_grad)
        hook = parameter.register_hook(lambda gradient: shared_gradient_calls.append(1))
        try:
            with patch.object(torch.Tensor, 'backward', counted):
                loss = update(net, optimizer(net, cfg), roles['fit'], self.y)
        finally:
            hook.remove()
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(shared_gradient_calls), 1)
        self.assertFalse(calls[0].get('retain_graph', False))
        self.assertEqual(net.epoch_execution['backward_calls'], 1)
        self.assertEqual(net.epoch_execution['effective_batch_size'], len(self.y))

    def test_mlshol_accumulates_microbatch_gradients_and_bn_statistics(self):
        import copy
        from torch.nn import functional as F
        roles = self.roles('MLSHoL')
        cfg = dict(self.cfg, batch_size=4, dropout=0., loss_weight=1.3)
        torch.manual_seed(17)
        net = network.Network('MLSHoL', roles['fit'], cfg)

        with torch.no_grad():
            net.head.s.weight.normal_(std=.5)
        reference, full_batch = copy.deepcopy(net), copy.deepcopy(net)
        np.random.seed(47)
        torch.manual_seed(58)
        opt = torch.optim.SGD(net.parameters(), lr=0.)
        measured = update(net, opt, roles['fit'], self.y)

        np.random.seed(47)
        torch.manual_seed(58)
        reference.train()
        order = np.random.permutation(len(self.y))
        p = self.y.mean()
        pos_weight = torch.tensor(min(10., ((1-p)/p)**.5), dtype=torch.float64)
        terms = []
        for start in range(0, len(order), cfg['batch_size']):
            ids = order[start:start+cfg['batch_size']]
            terms.append(cfg['loss_weight'] * F.binary_cross_entropy_with_logits(
                reference(roles['fit'], ids), torch.tensor(self.y[ids], dtype=torch.float64),
                pos_weight=pos_weight, reduction='sum') / len(self.y))
        aggregate = sum(terms)
        aggregate.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        self.assertAlmostEqual(measured, float(aggregate.detach()), places=10)
        for (name, actual), (_, expected) in zip(net.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                if actual.grad is None or expected.grad is None:
                    self.assertIs(actual.grad, expected.grad)
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)

        tracked = dict(net.named_buffers())
        self.assertTrue(any(name.endswith('num_batches_tracked') and int(value) == 2
                            for name, value in tracked.items()))
        full_batch.train()
        with torch.no_grad():
            full_batch(roles['fit'], np.arange(len(self.y)))
        full_buffers = dict(full_batch.named_buffers())
        self.assertTrue(any(name.endswith('num_batches_tracked') and
                            int(value) == 2 and int(full_buffers[name]) == 1
                            for name, value in tracked.items()))
        self.assertTrue(any(name.endswith('running_mean') and
                            not torch.equal(value, full_buffers[name])
                            for name, value in tracked.items()))
        self.assertEqual(net.epoch_execution['gradient_scope'], 'microbatch_forwards')
        self.assertEqual(net.epoch_execution['backward_calls'], 2)
        self.assertEqual(net.epoch_execution['effective_batch_size'], 4)

    def test_transfer_recipes_frozen_without_test_outcomes(self):
        from types import SimpleNamespace
        for dataset in ('email-Eu','tags-ask-ubuntu'):
            generator = DatasetSpecificComponentGenerator(dataset)
            seen = []
            def member_predict(target, cfg, seeds):
                seen.append(cfg)
                return np.ones((3,3)), {'max_training_label_end': 1}
            engine = SimpleNamespace(rank=SimpleNamespace(predict=member_predict))
            snap = dict(q=np.array([[0,1,2],[0,1,3],[1,2,3]]), cut=10,
                        x=np.ones((3,1)), names=np.array(['continue_edge_sync_triple_group_64']))
            result = generator._transfer(engine, {.6:snap,.8:snap})
            self.assertEqual(seen[0], seen[1])
            self.assertEqual(result['fit'].metadata['rule_active'], result['target'].metadata['rule_active'])
            self.assertEqual(result['target'].metadata['recipe_origin'], 'fit')

    def test_real_supervised_transfer_members_on_small_past_windows(self):
        import tempfile
        from pathlib import Path
        from historical_components.generators.engine import build_engine
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root, prefix='member-check-') as directory:
            self.assertEqual(Path(directory).resolve().parent, root.resolve())
            for dataset in ('email-Eu', 'tags-ask-ubuntu'):
                engine = build_engine(dataset, directory)
                rng = np.random.default_rng(41)
                snaps = {}
                for f in (.2,.4,.6,.8):
                    x = rng.normal(size=(8,3)).astype(np.float32)
                    snaps[f] = dict(q=np.array([[i,i+1,i+2] for i in range(8)]),
                        x=x, old_x=x.copy(), y=np.array([0,1]*4),
                        names=np.array(['continue_edge_sync_triple_group_64',
                            'event_decay_mock','event_edge_count_harmonic']),
                        fraction=f,cut=int(f*100),label_end=int(f*100)+20 if f<.8 else None,
                        duration_proxy_seconds=10.,completed_label_duration_seconds=10.)
                engine.rank.DATA[dataset] = dict(snaps=snaps)
                batches = DatasetSpecificComponentGenerator(dataset)._transfer(engine,snaps)
                self.assertEqual(batches['fit'].values.shape, (8,5))
                self.assertTrue(np.isfinite(batches['target'].values).all())
                self.assertEqual(batches['fit'].metadata['route'],batches['target'].metadata['route'])

if __name__ == '__main__':
    unittest.main()
