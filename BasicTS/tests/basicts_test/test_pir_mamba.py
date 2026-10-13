"""CPU regression checks; run with unittest, no optional pytest dependency."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from basicts.models.PIRMamba import PIRMamba, PIRMambaConfig
from basicts.models.PIRMamba.arch.attention import PatternRetrieval
from basicts.models.PIRMamba.utils import load_checkpoint


class TestPIRMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = PIRMamba(PIRMambaConfig(mamba_backend="torch")).eval()

    def test_architecture_and_invalid_shapes(self):
        self.assertEqual(sum(parameter.numel() for parameter in self.model.parameters()), 837813)
        self.assertEqual(len(self.model.state_dict()), 186)
        with self.assertRaises(ValueError):
            PIRMamba(PIRMambaConfig(output_len=24, mamba_backend="torch"))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 95, 170), torch.zeros(1, 95, 2))

    def test_calendar_is_required_and_range_checked(self):
        history = torch.zeros(1, 96, 170)
        with self.assertRaises(ValueError):
            self.model(history, None)
        timestamps = torch.zeros(1, 96, 2)
        timestamps[:, -1, 0] = 1.
        with self.assertRaises(ValueError):
            self.model(history, timestamps)

    def test_pattern_retrieval_is_history_causal(self):
        history = torch.randn(2, 96, 170)
        changed = history.clone()
        changed[:, 40:] += 100
        retrieval = self.model.pattern_retrieval
        self.assertIsInstance(retrieval, PatternRetrieval)
        with torch.no_grad():
            original = retrieval(history)
            altered = retrieval(changed)
        self.assertTrue(torch.equal(original[:, :, :40], altered[:, :, :40]))
        self.assertFalse(torch.equal(original[:, :, 40:], altered[:, :, 40:]))

    def test_native_weights_and_inputs_are_preserved(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pt"
            state = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
            torch.save({"model_state_dict": state}, checkpoint)
            loaded = load_checkpoint(self.model, checkpoint)
            self.assertTrue(all(torch.equal(value, self.model.state_dict()[name])
                                for name, value in state.items()))
        self.assertFalse(loaded["legacy_names_converted"])
        history = torch.randn(1, 96, 170)
        saved = history.clone()
        with torch.no_grad():
            prediction = self.model(history, torch.zeros(1, 96, 2))
        self.assertEqual(tuple(prediction.shape), (1, 12, 170))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.equal(history, saved))


if __name__ == "__main__":
    unittest.main()
