"""Configuration for the Mariana DRV-LM model.

The dataclass is the single source of truth for model hyper-parameters.
It serializes to/from plain dicts (JSON in checkpoints, GGUF metadata on
export) so checkpoints stay portable across versions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- vocabulary / sequence -------------------------------------------------
    vocab_size: int = 258          # 256 byte tokens + BOS + MASK
    ctx_len: int = 256
    bos_token: int = 256
    mask_token: int = 257

    # --- backbone --------------------------------------------------------------
    dim: int = 256
    n_heads: int = 4
    n_pre: int = 2                 # encoder blocks before the recurrent core
    n_core: int = 2                # shared core blocks (looped max_loops times)
    n_post: int = 2                # decoder blocks after the core
    mlp_mult: int = 4
    max_loops: int = 2             # recurrent-depth refinement iterations
    dropout: float = 0.0

    # --- diffusion -------------------------------------------------------------
    block_size: int = 32           # diffusion block width (block-causal attention)
    n_time_buckets: int = 64       # discrete noise-level buckets

    # --- conditioning ----------------------------------------------------------
    rope_base_pos: float = 10000.0
    rope_base_time: float = 100.0
    use_adaln: bool = True         # adaLN-zero FiLM timestep conditioning

    # --- optional capacity -----------------------------------------------------
    use_moe: bool = False          # sparse-MoE FFN in every block
    moe_experts: int = 4
    moe_topk: int = 2

    # --- auxiliary heads -------------------------------------------------------
    use_mtp: bool = True           # multi-token-prediction auxiliary head

    # --- loop regularisation ---------------------------------------------------
    loop_dropout: bool = True      # sample loop count in {1..max_loops} per step

    # --- loss weights ------------------------------------------------------------
    w_ar: float = 1.0
    w_diff: float = 1.0
    w_align: float = 0.1           # diffusion<->AR consistency (acceptance lever)
    w_ver: float = 0.1
    w_mtp: float = 0.1
    w_moe_lb: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        # tolerate checkpoints written by older/newer versions
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls.from_dict(json.loads(s))

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0
        return self.dim // self.n_heads

    @property
    def n_layers(self) -> int:
        """Flattened decoder depth (core blocks are weight-shared across loops)."""
        return self.n_pre + self.n_core * self.max_loops + self.n_post
