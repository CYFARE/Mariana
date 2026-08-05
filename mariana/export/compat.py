"""Convert a Mariana checkpoint into the *exact* LLaMA-architecture equivalent
of its autoregressive decoding path.

Why this is exact (lossless):

* Committed/AR tokens always use diffusion time bucket 0 and region 0, and
  the time/region embeddings are defined relative to that baseline inside
  the model itself -- so the exported token table is used verbatim and the
  tied LM head stays exact.
* adaLN modulation is defined relative to the t=0 baseline, so at bucket 0
  it is the identity transform -- nothing to fold.
* The position RoPE axis uses the standard LLaMA frequency schedule
  (``theta=cfg.rope_base_pos``); the time axis contributes zero rotation at
  bucket 0.  Only the pair layout differs (interleaved vs half-split), which
  is fixed by permuting the Q/K projection rows.
* The recurrent core is unrolled into ``n_core * max_loops`` distinct GGUF
  layers with shared weights.

The result is a plain LLaMA transformer (RMSNorm + RoPE + SwiGLU, tied
embeddings) that llama.cpp / Ollama / ONNX runtimes can execute directly.
"""

from __future__ import annotations

import torch

from ..config import Config
from ..rope import interleaved_to_neox_permutation


def fold_token_embedding(sd: dict, cfg: Config) -> torch.Tensor:
    # nothing to fold: the model already defines time/region embeddings
    # relative to the t=0 / region=0 baseline
    return sd["tok_emb.weight"].float().clone()


def _rope_perm(dim: int, n_heads: int) -> torch.Tensor:
    head_dim = dim // n_heads
    p = interleaved_to_neox_permutation(head_dim)
    return torch.cat([p + h * head_dim for h in range(n_heads)])


def layer_sources(cfg: Config):
    """Flattened layer list: ``[(state_dict_prefix, index), ...]``."""
    src = [("pre", i) for i in range(cfg.n_pre)]
    for _ in range(cfg.max_loops):
        src += [("core", i) for i in range(cfg.n_core)]
    src += [("post", i) for i in range(cfg.n_post)]
    return src


def to_llama_state(sd: dict, cfg: Config) -> dict[str, torch.Tensor]:
    """Return ``{llama_tensor_name: tensor}`` in float32."""
    if cfg.use_moe:
        raise ValueError(
            "GGUF/ONNX export of MoE checkpoints is not supported yet; "
            "export a dense checkpoint (use_moe=False)."
        )
    out: dict[str, torch.Tensor] = {}
    perm = _rope_perm(cfg.dim, cfg.n_heads)

    out["token_embd.weight"] = fold_token_embedding(sd, cfg)

    for l, (prefix, i) in enumerate(layer_sources(cfg)):
        p = f"{prefix}.{i}."
        qkv = sd[p + "qkv.weight"].float()  # (3*dim, dim), row order (q|k|v)
        q, k, v = qkv.chunk(3, dim=0)
        out[f"blk.{l}.attn_q.weight"] = q[perm].contiguous()
        out[f"blk.{l}.attn_k.weight"] = k[perm].contiguous()
        out[f"blk.{l}.attn_v.weight"] = v.contiguous()
        out[f"blk.{l}.attn_output.weight"] = sd[p + "o.weight"].float()
        out[f"blk.{l}.attn_norm.weight"] = sd[p + "ln1.weight"].float()
        out[f"blk.{l}.ffn_norm.weight"] = sd[p + "ln2.weight"].float()
        out[f"blk.{l}.ffn_gate.weight"] = sd[p + "mlp.fc1.weight"].float()
        out[f"blk.{l}.ffn_up.weight"] = sd[p + "mlp.fc2.weight"].float()
        out[f"blk.{l}.ffn_down.weight"] = sd[p + "mlp.fc3.weight"].float()

    out["output_norm.weight"] = sd["ln.weight"].float()
    out["output.weight"] = out["token_embd.weight"].clone()  # tied head
    return out


def llama_metadata(cfg: Config) -> dict:
    return {
        "llama.context_length": cfg.ctx_len,
        "llama.embedding_length": cfg.dim,
        "llama.block_count": cfg.n_layers,
        "llama.feed_forward_length": cfg.mlp_mult * cfg.dim,
        "llama.attention.head_count": cfg.n_heads,
        "llama.attention.head_count_kv": cfg.n_heads,
        "llama.attention.layer_norm_rms_epsilon": 1e-6,
        "llama.rope.dimension_count": cfg.head_dim,
        "llama.rope.freq_base": float(cfg.rope_base_pos),
        "llama.vocab_size": cfg.vocab_size,
    }
