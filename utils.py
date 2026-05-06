from pathlib import Path

import torch


def safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def state_dict_from_checkpoint(ckpt):
    return ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt


def load_state(model, ckpt, strict=True):
    model.load_state_dict(state_dict_from_checkpoint(ckpt), strict=strict)
    return model


def save_checkpoint(path, model, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": dict(config)}, path)


def last_hidden(seq_output, item_seq_ids=None):
    if item_seq_ids is None:
        return seq_output[:, -1]
    pos = (item_seq_ids != 0).long().sum(dim=1).sub(1).clamp_min(0)
    batch = torch.arange(item_seq_ids.size(0), device=item_seq_ids.device)
    return seq_output[batch, pos]


def checkpoint_config(ckpt, defaults):
    if isinstance(ckpt, dict) and "config" in ckpt:
        out = dict(defaults)
        out.update(ckpt["config"])
        return out
    return dict(defaults)
