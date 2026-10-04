"""Focused checks for the STIDGCN-derived model and its two replacements."""

import torch

from basicts.metrics.metric_meter import RMSEMeter
from basicts.models.MambaTCNCA import MambaTCNCAConfig, MambaTCNCADG
from basicts.models.MambaTCNCA.arch.temporal import CausalTCNBlock
from basicts.models.STIDGCN.architecture import DGCN, GLU, STIDGCN, TemporalEmbedding


def test_tcn_does_not_read_future_steps():
    torch.manual_seed(1)
    block = CausalTCNBlock(channels=8, dilation=2, dropout=0).eval()
    inputs = torch.randn(6, 9, 8)
    changed = inputs.clone()
    changed[:, 5:] += 100
    torch.testing.assert_close(block(inputs)[:, :5], block(changed)[:, :5])


def test_full_model_keeps_stidgcn_tree_and_backpropagates():
    torch.manual_seed(2)
    config = MambaTCNCAConfig(
        num_nodes=5, input_len=12, output_len=3, channels=4,
        num_heads=2, dropout=0, mamba_backend="torch",
    )
    model = MambaTCNCADG(config)
    assert isinstance(model.temporal_embedding, TemporalEmbedding)
    assert isinstance(model.tree.stage1.dgcn, DGCN)
    assert isinstance(model.glu, GLU)
    timestamps = torch.rand(2, 12, 2)
    prediction = model(torch.randn(2, 12, 5), timestamps)
    assert prediction.shape == (2, 3, 5)
    prediction.square().mean().backward()
    assert model.tree.stage1.temporal1.mamba.A_log.grad is not None
    assert model.tree.stage1.cross_odd.attention.in_proj_weight.grad is not None
    assert model.tree.stage1.dgcn.generator.memory.grad is not None


def test_without_ca_restores_source_hadamard_path():
    config = MambaTCNCAConfig(
        num_nodes=5, channels=4, num_heads=2, use_ca=False,
        use_mamba=False, use_tcn=True, dropout=0,
    )
    model = MambaTCNCADG(config)
    assert model.tree.stage1.cross_odd is None
    prediction = model(torch.randn(1, 12, 5), torch.rand(1, 12, 2))
    assert prediction.shape == (1, 12, 5)


def test_unchanged_backbone_matches_stidgcn_when_operators_are_swapped_back():
    """Prove the tree, embedding, graph and decoder preserve source behavior."""
    torch.manual_seed(3)
    source = STIDGCN(
        input_dim=3, num_nodes=5, channels=4, granularity=288,
        input_len=12, output_len=3, dropout=0,
    ).eval()
    replacement = MambaTCNCADG(MambaTCNCAConfig(
        num_nodes=5, channels=4, output_len=3,
        use_mamba=False, use_ca=False, dropout=0,
    )).eval()
    replacement.temporal_embedding = source.Temb
    replacement.start_conv = source.start_conv
    replacement.glu = source.glu
    replacement.regression_layer = source.regression_layer
    for stage, original in zip(
        (replacement.tree.stage1, replacement.tree.stage2, replacement.tree.stage3),
        (source.tree.IDGCN1, source.tree.IDGCN2, source.tree.IDGCN3),
    ):
        stage.temporal1 = original.conv1
        stage.temporal2 = original.conv2
        stage.temporal3 = original.conv3
        stage.temporal4 = original.conv4
        stage.dgcn = original.dgcn
    inputs = torch.randn(1, 12, 5)
    timestamps = torch.rand(1, 12, 2)
    history = torch.cat((
        inputs.unsqueeze(-1), timestamps.unsqueeze(2).expand(-1, -1, 5, -1)
    ), dim=-1)
    with torch.no_grad():
        expected = source(history).squeeze(-1)
        actual = replacement(inputs, timestamps)
    torch.testing.assert_close(actual, expected)


def test_rmse_meter_exposes_last_value_for_step_training():
    meter = RMSEMeter()
    meter.update(2.0, n=2)
    meter.update(4.0, n=1)
    assert meter.last == 4.0
    assert abs(meter.value - (8.0 ** 0.5)) < 1e-6
    meter.reset()
    assert meter.last == 0.0
