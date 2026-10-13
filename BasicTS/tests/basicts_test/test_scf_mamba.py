"""SCF-Mamba architecture, masked data and native BasicTS lifecycle checks."""
import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from basicts import BasicTSLauncher
from basicts.models.SCFMamba import SCFMamba, SCFMambaConfig
from basicts.models.SCFMamba.arch.cross_attention import CrossSequenceAttention
from basicts.models.SCFMamba.data import SCFMambaDataset, masked_mae
from basicts.models.SCFMamba.train import MaskedMetricsCallback, SCFMambaAdam, build_config
from basicts.models.SCFMamba.utils import load_checkpoint


class TestSCFMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, nodes=16, horizon=12):
        return SCFMamba(SCFMambaConfig(num_features=nodes, output_len=horizon,
                                       mamba_backend="torch"))

    @staticmethod
    def dataset(folder, nodes=16):
        rng = np.random.default_rng(4321)
        for split in ("train", "val", "test"):
            values = rng.uniform(10, 30, (109, nodes)).astype(np.float32)
            values[97, 0] = 0
            np.save(Path(folder) / f"{split}_data.npy", values)
            np.save(Path(folder) / f"{split}_timestamps.npy", np.zeros((109, 2), np.float64))

    def test_public_config_shapes_and_parameter_counts(self):
        counts = {170: (865311, 866463, 868767, 873375),
                  307: (1334112, 1335264, 1337568, 1342176)}
        for nodes, expected in counts.items():
            for horizon, count in zip((12, 24, 48, 96), expected):
                with self.subTest(nodes=nodes, horizon=horizon):
                    model = self.model(nodes, horizon).eval()
                    history = torch.randn(1, 96, nodes)
                    saved = history.clone()
                    with torch.no_grad():
                        prediction = model(history, torch.zeros(1, 96, 2))
                    self.assertEqual(tuple(prediction.shape), (1, horizon, nodes))
                    self.assertTrue(prediction.isfinite().all())
                    self.assertTrue(torch.equal(history, saved))
                    self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                    self.assertFalse(hasattr(model, "pattern_retrieval"))

    def test_input_and_calendar_validation(self):
        model = self.model()
        with self.assertRaises(ValueError):
            self.model(horizon=0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 95, 16), torch.zeros(1, 95, 2))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 96, 16), None)
        time = torch.zeros(1, 96, 2)
        time[:, -1, 0] = 1
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 96, 16), time)

    def test_shared_attention_three_temporal_sites(self):
        model = self.model().eval()
        calls = []
        hook = model.fusion.register_forward_pre_hook(lambda _, args: calls.append(args))
        with torch.no_grad():
            model(torch.randn(1, 96, 16), torch.zeros(1, 96, 2))
        hook.remove()
        self.assertEqual(sorted(args[0].shape[1] for args in calls), [24, 24, 48])
        self.assertTrue(all(set(args[2]).isdisjoint(args[3]) for args in calls))

    def test_attention_dropout_checkpoint_continuation(self):
        fusion = CrossSequenceAttention(16, 96).train()
        with torch.no_grad():
            fusion.even_gate.fill_(0.2)
            fusion.odd_gate.fill_(0.2)
        args = (torch.randn(2, 24, 16), torch.randn(2, 24, 16),
                tuple(range(0, 48, 2)), tuple(range(1, 48, 2)))
        global_rng = torch.get_rng_state()
        fusion(*args)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        saved = copy.deepcopy(fusion.state_dict())
        expected = fusion(*args)
        replica = CrossSequenceAttention(16, 96).train()
        replica.load_state_dict(saved)
        before = torch.get_rng_state()
        actual = replica(*args)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

    def test_mask_uses_raw_targets_and_train_only_statistics(self):
        prediction = torch.tensor([99., 3., 3.], requires_grad=True)
        targets = torch.tensor([-1., 0., 1.])
        raw = torch.tensor([0., 5., 7.])
        loss = masked_mae(prediction, targets, raw != 0)
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        self.assertEqual(prediction.grad[0].item(), 0.)
        self.assertGreater(prediction.grad[1:].abs().sum().item(), 0.)
        self.assertEqual(masked_mae(torch.ones(2), torch.zeros(2),
                                    torch.zeros(2, dtype=torch.bool)).item(), 0.)
        with TemporaryDirectory() as folder:
            self.dataset(folder)
            dataset = SCFMambaDataset("fixture", 96, 12, "val", data_file_path=folder)
            train = np.load(Path(folder) / "train_data.npy")
            np.testing.assert_array_equal(dataset.mean.numpy(), train.mean(0).astype(np.float64))
            item = dataset[0]
            np.testing.assert_array_equal(item["targets_mask"], item["raw_targets"] != 0)
            self.assertFalse(item["targets_mask"][1, 0])
            self.assertEqual(len(dataset), 2)

    def test_optimizer_groups_gradients_and_empty_batch(self):
        model = self.model().train()
        optimizer = SCFMambaAdam(model.parameters())
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [1e-5, 0.])
        params = [p for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len({id(p) for p in params}), len(list(model.parameters())))
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.randn(2, 96, 16), torch.zeros(2, 96, 2))
            loss = masked_mae(prediction, torch.ones_like(prediction),
                              torch.ones_like(prediction, dtype=torch.bool))
            loss.backward()
            self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in params))
            optimizer.step()
        self.assertGreater(model.fusion.attention.in_proj_weight.grad.abs().sum().item(), 0.)
        callback = MaskedMetricsCallback()
        runner = type("Runner", (), {})()
        callback.on_compute_loss(runner, {"targets_mask": torch.zeros(1, dtype=torch.bool)}, data={})
        self.assertFalse(runner.should_optimizer_step)
        replica = SCFMambaAdam(model.parameters())
        replica.load_state_dict(optimizer.state_dict())
        self.assertEqual([g["weight_decay"] for g in replica.param_groups], [1e-5, 0.])

    def test_native_checkpoint_and_global_evaluation_metrics(self):
        with TemporaryDirectory() as folder:
            self.dataset(folder)
            model = self.model().eval()
            checkpoint = Path(folder) / "model.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, checkpoint)
            replica = self.model().eval()
            loaded = load_checkpoint(replica, checkpoint)
            self.assertFalse(loaded["legacy_names_converted"])
            script = Path(__file__).resolve().parents[2] / "scripts/forecasting/evaluate_scf_mamba.py"
            spec = importlib.util.spec_from_file_location("scf_evaluation", script)
            evaluation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluation)
            dataset = SCFMambaDataset("fixture", 96, 12, "test", data_file_path=folder)
            report = evaluation.evaluate(replica, dataset, batch_size=1)
            self.assertEqual(report["windows"], 2)
            self.assertEqual(len(report["all_steps"]), 12)
            self.assertEqual(report["all_H_steps"]["valid_elements"], 2 * 12 * 16 - 2)
            self.assertAlmostEqual(report["all_H_steps"]["RMSE"] ** 2,
                                   report["all_H_steps"]["MSE"], places=10)
            # Global accumulation must be invariant to evaluation batch size.
            other = evaluation.evaluate(replica, dataset, batch_size=2)
            self.assertAlmostEqual(other["all_H_steps"]["MAE"],
                                   report["all_H_steps"]["MAE"], places=5)

    def test_basic_ts_training_and_resume(self):
        with TemporaryDirectory() as folder:
            self.dataset(folder)
            config = build_config("fixture", folder, epochs=1, batch_size=1,
                                  gpus=None, backend="torch", checkpoint_dir=str(Path(folder) / "ckpt"))
            BasicTSLauncher.launch_training(config)
            checkpoint_dir = Path(config.ckpt_save_dir) / config.md5
            last = checkpoint_dir / "SCFMamba_1.pt"
            saved = torch.load(last, map_location="cpu", weights_only=False)
            self.assertIn("fusion._extra_state", saved["model_state_dict"])
            self.assertEqual(len(saved["optim_state_dict"]["param_groups"]), 2)
            self.assertEqual(saved["scheduler_state_dict"]["last_epoch"], 1)
            # Keep the same config namespace to exercise the framework's resume path.
            config.num_epochs = 2
            config.test_interval = 3
            from basicts.runners import BasicTSRunner
            runner = BasicTSRunner(config)
            runner.train()
            self.assertEqual(runner.start_epoch, 1)
            self.assertEqual(runner.epoch, 2)
            self.assertEqual(len(runner.optimizer.param_groups), 2)


if __name__ == "__main__":
    unittest.main()
