"""Load native BasicTS weights or translate the historical V-12 tensor names."""
from pathlib import Path
import re

import torch

DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[5]
                      / "checkpoints/PIRMamba/PEMS08_10_96_12/PIRMamba_best_val_MAE_unmasked.pt")


def convert_legacy_state(state):
    """Rename tensors only: no values, shapes or precision are changed."""
    converted = {}
    names = {
        "backbone.time_in_day_emb": "time_embedding.time_of_day",
        "backbone.day_in_week_emb": "time_embedding.day_of_cycle",
        "backbone.core.inv_timescales": "backbone.position_frequencies",
        "backbone.core.projection1.weight": "backbone.projection.weight",
    }
    for name, tensor in state.items():
        if name in names:
            target = names[name]
        elif name.startswith("backbone.core.blocks1.SCINet_Tree."):
            target = name.replace("backbone.core.blocks1.SCINet_Tree.", "backbone.tree.", 1)
            target = target.replace("SCINet_Tree_odd.", "odd.").replace("SCINet_Tree_even.", "even.")
            target = target.replace("workingblock.interact.level.", "interaction.")
            target = target.replace(".P.", ".update_odd.").replace(".U.", ".update_even.")
            target = target.replace(".seasonal.", ".node_encoder.")
            replacements = {
                "embedding.ValueEmb.": "history_projection.",
                "encoder.ssm_layers.0.ssm_r.": "encoder.backward_mamba.",
                "encoder.ssm_layers.0.ssm.": "encoder.forward_mamba.",
                "encoder.ssm_layers.0.conv1.": "encoder.feedforward_in.",
                "encoder.ssm_layers.0.conv2.": "encoder.feedforward_out.",
                "encoder.ssm_layers.0.norm1.": "encoder.mamba_norm.",
                "encoder.ssm_layers.0.norm2.": "encoder.feedforward_norm.",
                "encoder.norm_layer.": "output_norm.",
                "projector.": "output_projection.",
            }
            for old, new in replacements.items():
                target = target.replace(old, new)
        elif name.startswith("cross_attention."):
            target = name.replace(".query_value.", ".query_projection.")
            target = target.replace(".history_value.", ".history_projection.")
            target = target.replace(".readout.", ".output_projection.")
        elif name.startswith("dft."):
            target = "pattern_retrieval." + name[4:]
            target = re.sub(r"\.(embedding|query|key|value)\.",
                            lambda match: "." + {"embedding": "window_projection", "query": "query_projection",
                                                "key": "key_projection", "value": "value_projection"}[match[1]] + ".",
                            target)
        else:
            raise ValueError(f"Unknown historical V tensor: {name}")
        if target in converted:
            raise ValueError(f"Duplicate tensor mapping: {target}")
        converted[target] = tensor
    return converted


def load_checkpoint(model, path=DEFAULT_CHECKPOINT):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    legacy = "backbone.time_in_day_emb" in state
    if legacy:
        state = convert_legacy_state(state)
    model.load_state_dict(state, strict=True)
    return {"path": str(path), "epoch": checkpoint.get("epoch"),
            "legacy_names_converted": legacy, "state_tensors": len(state)}

