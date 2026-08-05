"""Safetensors export (raw checkpoint and LLaMA-converted AR weights)."""

from __future__ import annotations

import os


from ..config import Config
from .compat import llama_metadata, to_llama_state


def _require_safetensors():
    try:
        from safetensors.torch import save_file

        return save_file
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "safetensors is required for this export format.\n"
            "Install it with:  pip install safetensors"
        ) from e


def export_safetensors(state_dict: dict, cfg: Config, out_path: str,
                       llama_compat: bool = True) -> str:
    save_file = _require_safetensors()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if llama_compat:
        tensors = to_llama_state(state_dict, cfg)
        meta = {"format": "llama", **{k: str(v) for k, v in llama_metadata(cfg).items()}}
    else:
        tensors = {k: v.float().contiguous() for k, v in state_dict.items()}
        meta = {"format": "mariana"}

    # clone: looped core layers share storage, which safetensors rejects
    tensors = {k: v.cpu().clone().contiguous() for k, v in tensors.items()}
    save_file(tensors, out_path, metadata=meta)

    cfg_path = os.path.splitext(out_path)[0] + ".config.json"
    with open(cfg_path, "w") as f:
        f.write(cfg.to_json())
    print(f"Wrote {out_path}")
    print(f"Wrote {cfg_path}")
    return out_path
