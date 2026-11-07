import unittest
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv
from historical_components.encoders import (
    ComponentPreprocessor, PositiveComponentEncoder, JointHead,
)
from models import dual

class ComponentContracts(unittest.TestCase):
    def test_positive_scores_at_fixed_structure_and_free_activity_mask(self):
        encoder = PositiveComponentEncoder(3, masks=[2]).double()
        head = JointHead().double()

        self.assertIsInstance(encoder.context, nn.Linear)
        self.assertGreater(float(encoder.context.weight.abs().sum()), 0.)
        with torch.no_grad():
            encoder.context.weight.fill_(-.2)
        values = torch.tensor([[.1, -.4, .8], [-.2, .3, .1]], dtype=torch.float64, requires_grad=True)
        fixed_structure = torch.zeros(2, 32, dtype=torch.float64)
        derivative = torch.autograd.grad(head(encoder(values), fixed_structure).sum(), values)[0]
        self.assertTrue(bool((derivative[:, :2] > 0).all()))
        self.assertTrue(bool((derivative[:, 2] < 0).all()))
        self.assertEqual(encoder.scores, [0, 1])
        self.assertEqual(encoder.masks, [2])

    def test_ties_activity_and_training_only_scaling(self):
        processor = ComponentPreprocessor(['member', 'rule_if_active', 'rule_active_mask'])
        raw = np.array([[2, 8, 1], [2, 8, 1], [8, 0, 0], [0, 4, 1]], dtype=np.float64)
        rank = processor.rank_normalize(raw)
        np.testing.assert_array_equal(rank[:, 0], [.5, .5, .875, .125])
        np.testing.assert_array_equal(rank[:, 1], [.75, .75, 0., .375])
        np.testing.assert_array_equal(rank[:, 2], raw[:, 2])

        processor.fit(raw, ids=[0, 3])
        np.testing.assert_array_equal(processor.scaler.mean_, [.3125, .5625, 1.])
        np.testing.assert_array_equal(processor.scaler.scale_, [.1875, .1875, 1.])
        state = processor.state_dict()
        target = np.array([[1, 7, 1], [3, 0, 0]], dtype=np.float64)
        result = processor.transform(target)
        np.testing.assert_allclose(result[:, 0], [-1/3, 7/3])
        np.testing.assert_array_equal(result[:, 2], [0., -1.])
        self.assertEqual(processor.state_dict(), state)
        restored = ComponentPreprocessor.from_state_dict(state)
        np.testing.assert_array_equal(restored.transform(target), result)
        self.assertEqual(result.dtype, np.float64)

    def test_transformed_components_clip_and_reject_invalid_inputs(self):
        processor = ComponentPreprocessor(['member'])
        raw = np.arange(100, dtype=np.float64).reshape(-1, 1)
        processor.fit(raw, ids=[0, 1])
        result = processor.transform(raw)
        self.assertEqual(float(result[-1, 0]), 8.)
        with self.assertRaises(ValueError):
            processor.transform(np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            processor.transform(np.ones((4, 2)))

    def test_actual_gat_layers_preserve_candidate_graph_inputs(self):
        *_, Core = dual.make_model_classes()
        core = Core(3, 4, 2, 0, feature_proj_dim=16, hidden_dim=64,
                    num_layers=3, encoder='gat', gat_heads=4, dropout=0.)
        self.assertEqual(core.encoder, 'gat')
        self.assertEqual(len(core.gnn_layers), 3)
        self.assertTrue(all(isinstance(layer.attention, GATConv) for layer in core.gnn_layers))
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        original = edge_index.clone()
        hidden = torch.zeros((3, 64))
        output = core.gnn_layers[0](hidden, edge_index)
        self.assertEqual(tuple(output.shape), (3, 64))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(torch.equal(edge_index, original))
        with self.assertRaises(ValueError):
            Core(3, 4, 2, 0, encoder='sage')
        with self.assertRaises(ValueError):
            Core(3, 4, 2, 0, hidden_dim=64, gat_heads=3)

if __name__ == '__main__':
    unittest.main()
