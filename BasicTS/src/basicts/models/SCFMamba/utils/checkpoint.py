"""Strictly load native SCF-Mamba weights or the historical B-model names."""
from pathlib import Path

import torch


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    legacy = "fusion.gate_query" in state
    if legacy:
        replacements = {
            ".phi.": ".scale_odd.", ".psi.": ".scale_even.",
            ".node_encoder.": ".node_mamba.",
            ".history_projection.": ".input_proj.",
            ".input_projection.": ".input_proj.",
            ".output_projection.": ".output_proj.",
            ".time_projection.": ".time_proj.",
            ".feedforward_in.": ".ffn_in.",
            ".feedforward_out.": ".ffn_out.",
            ".feedforward_norm.": ".ffn_norm.",
            "backbone.projection.": "backbone.head.",
            "fusion.gate_query": "fusion.even_gate",
            "fusion.gate_memory": "fusion.odd_gate",
            "fusion.positions": "fusion.position_encoding",
        }
        converted = {}
        for name, value in state.items():
            for old, new in replacements.items():
                name = name.replace(old, new)
            if name in converted:
                raise ValueError(f"Duplicate checkpoint key: {name}")
            if name == "fusion.position_encoding":
                value = value[:, :model.input_len].clone()
            converted[name] = value
        converted["fusion._extra_state"] = {"seed": 54321, "dropout_states": {}}
        state = converted
    model.load_state_dict(state, strict=True)
    return {"path": str(Path(path)), "epoch": checkpoint.get("epoch"),
            "legacy_names_converted": legacy}
