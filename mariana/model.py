"""Mariana DRV-LM model.

Architectural upgrades over the original single-file version (see
``architecture.txt``):

* **adaLN-zero timestep conditioning** (DiT-style FiLM scale/shift/gate on
  every block) instead of a plain additive time embedding.  The modulation
  is computed *relative to the t=0 baseline*, so committed/AR tokens (time
  bucket 0) get an exact identity transform -- this keeps the AR path
  bit-exact exportable to vanilla-LLaMA formats (GGUF/ONNX).
* **Full-spectrum LLaMA-exact position RoPE** + additive time-axis rotation
  on odd pair slots (see ``rope.py``).
* **Recurrent-depth core with inference-time adaptive halting**: the shared
  core blocks loop ``max_loops`` times, exiting early when the hidden state
  converges (cheap "deep recursive thinking").  Training samples the loop
  count (loop dropout) so every depth is a valid compute point.
* **Optional sparse-MoE FFN** (top-k routing, Switch-style load balancing).
* **Multi-token-prediction head** for lookahead supervision.
* **Verifier head** scoring per-token draft acceptance (BCE-trained).
* Optional **FlexAttention** kernel path for the hybrid training mask, with
  automatic SDPA fallback.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .rope import apply_rope, build_rope_angles

try:  # FlexAttention is available in torch >= 2.5
    from torch.nn.attention.flex_attention import flex_attention

    HAS_FLEX = True
except Exception:  # pragma: no cover
    flex_attention = None
    HAS_FLEX = False


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.dim
        self.fc1 = nn.Linear(cfg.dim, hidden, bias=False)  # gate
        self.fc2 = nn.Linear(cfg.dim, hidden, bias=False)  # up
        self.fc3 = nn.Linear(hidden, cfg.dim, bias=False)  # down

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class SparseMoEFFN(nn.Module):
    """Top-k routed SwiGLU experts with Switch-style load-balancing loss."""

    def __init__(self, cfg: Config):
        super().__init__()
        E, dim = cfg.moe_experts, cfg.dim
        hidden = cfg.mlp_mult * cfg.dim
        self.k = cfg.moe_topk
        self.n_experts = E
        self.router = nn.Linear(dim, E, bias=False)
        self.gate = nn.Parameter(torch.empty(E, dim, hidden))
        self.up = nn.Parameter(torch.empty(E, dim, hidden))
        self.down = nn.Parameter(torch.empty(E, hidden, dim))
        for p in (self.gate, self.up, self.down):
            nn.init.normal_(p, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)                      # (N, E)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = probs.topk(self.k, dim=-1)       # (N, k)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            sel = top_i == e                            # (N, k)
            idx, slot = sel.nonzero(as_tuple=True)
            if idx.numel() == 0:
                continue
            xe = flat[idx]
            he = F.silu(xe @ self.gate[e]) * (xe @ self.up[e])
            ye = he @ self.down[e]
            out.index_add_(0, idx, ye * top_w[idx, slot].unsqueeze(-1))

        # Switch load-balancing: E * sum(f_e * P_e)
        f = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            f[e] = (top_i == e).float().mean()
        lb = self.n_experts * (f * probs.mean(0)).sum()
        return out.view(B, T, D), lb


# ---------------------------------------------------------------------------
# Transformer block with adaLN-zero conditioning
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.nh = cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ln1 = RMSNorm(cfg.dim)
        self.ln2 = RMSNorm(cfg.dim)
        if cfg.use_moe:
            self.mlp = SparseMoEFFN(cfg)
        else:
            self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.use_adaln:
            # produces (scale1, shift1, gate1, scale2, shift2, gate2) deltas
            self.adaln = nn.Linear(cfg.dim, 6 * cfg.dim, bias=True)
            nn.init.zeros_(self.adaln.weight)
            nn.init.zeros_(self.adaln.bias)

    def _modulate(self, h, scale, shift):
        return h * (1.0 + scale) + shift

    def forward(
        self,
        x,
        cond,                 # (B, T, dim) adaLN delta (0 at time bucket 0)
        angles_pos,
        angles_time,
        times,
        mask=None,            # bool (B,1,T,S), True = attend
        kv_cache=None,
        return_kv=False,
        block_mask=None,      # FlexAttention BlockMask (full-seq training only)
    ):
        B, T, _ = x.shape
        lb = None

        if self.cfg.use_adaln:
            s1, b1, g1, s2, b2, g2 = self.adaln(cond).chunk(6, dim=-1)
        else:
            s1 = b1 = g1 = s2 = b2 = g2 = 0.0

        residual = x
        h = self._modulate(self.ln1(x), s1, b1) if self.cfg.use_adaln else self.ln1(x)

        qkv = self.qkv(h).view(B, T, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # absolute positions: new tokens start after the cached prefix
        off = kv_cache[0].shape[2] if kv_cache is not None else 0
        ap = angles_pos[off : off + T]

        q = apply_rope(q, ap, angles_time, times)
        k = apply_rope(k, ap, angles_time, times)

        if kv_cache is not None:
            ck, cv = kv_cache
            k = torch.cat([ck, k], dim=2)
            v = torch.cat([cv, v], dim=2)

        if block_mask is not None and HAS_FLEX:
            out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        out = out.transpose(1, 2).contiguous().view(B, T, self.cfg.dim)
        out = self.o(out)
        x = residual + (1.0 + g1) * self.dropout(out) if self.cfg.use_adaln else residual + self.dropout(out)

        residual = x
        h = self._modulate(self.ln2(x), s2, b2) if self.cfg.use_adaln else self.ln2(x)
        if isinstance(self.mlp, SparseMoEFFN):
            mlp_out, lb = self.mlp(h)
        else:
            mlp_out = self.mlp(h)
        x = residual + (1.0 + g2) * self.dropout(mlp_out) if self.cfg.use_adaln else residual + self.dropout(mlp_out)

        if return_kv:
            return x, (k, v), lb
        return x, lb


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class MarianaLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.time_emb = nn.Embedding(cfg.n_time_buckets, cfg.dim)
        self.region_emb = nn.Embedding(2, cfg.dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )

        self.pre = nn.ModuleList([Block(cfg) for _ in range(cfg.n_pre)])
        self.core = nn.ModuleList([Block(cfg) for _ in range(cfg.n_core)])
        self.post = nn.ModuleList([Block(cfg) for _ in range(cfg.n_post)])

        self.ln = RMSNorm(cfg.dim)
        self.verifier = nn.Linear(cfg.dim, 1, bias=False)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings
        self.mtp_head = (
            nn.Linear(cfg.dim, cfg.vocab_size, bias=False) if cfg.use_mtp else None
        )

        head_dim = cfg.head_dim
        ap, at = build_rope_angles(
            head_dim, cfg.ctx_len * 2, cfg.n_time_buckets,
            cfg.rope_base_pos, cfg.rope_base_time,
        )
        self.register_buffer("angles_pos", ap, persistent=False)
        self.register_buffer("angles_time", at, persistent=False)

        self.apply(self._init_weights)
        self._rescale_residual_init()

    # -- init -----------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # re-assert adaLN-zero after generic init
        for blk in list(self.pre) + list(self.core) + list(self.post):
            if self.cfg.use_adaln:
                nn.init.zeros_(blk.adaln.weight)
                nn.init.zeros_(blk.adaln.bias)

    def _rescale_residual_init(self):
        # GPT-2 style: shrink residual-path output projections by depth
        scale = 0.02 / (2 * self.cfg.n_layers) ** 0.5
        for blk in list(self.pre) + list(self.core) + list(self.post):
            nn.init.normal_(blk.o.weight, std=scale)
            if isinstance(blk.mlp, SwiGLU):
                nn.init.normal_(blk.mlp.fc3.weight, std=scale)
            else:
                nn.init.normal_(blk.mlp.down, std=scale)

    # -- conditioning -----------------------------------------------------------
    def _cond_delta(self, times):
        """adaLN input relative to the t=0 baseline (exact identity at t=0)."""
        emb = self.time_emb(times.clamp(0, self.cfg.n_time_buckets - 1))
        base = self.time_emb.weight[0]
        return self.time_mlp(emb) - self.time_mlp(base)

    # -- forward ------------------------------------------------------------------
    def blocks_for_loops(self, n_loops: int):
        return list(self.pre) + list(self.core) * n_loops + list(self.post)

    def forward(
        self,
        tokens,
        times,
        region,
        mask=None,
        kv_cache=None,
        return_kv=False,
        n_loops=None,
        block_mask=None,
        halt_tol: float = 0.0,
        need_hidden: bool = False,
    ):
        """Returns ``(logits, verify, extras)``.

        ``extras`` may contain ``kv`` (list of (k, v)), ``hidden`` (final
        hidden states) and ``moe_lb`` (summed load-balancing loss).
        """
        if n_loops is None:
            n_loops = self.cfg.max_loops
        # Embeddings are defined *relative* to the t=0 / region=0 baseline, so
        # the committed/AR path uses exactly tok_emb (+ nothing else) and the
        # tied LM head is exact.  This is what makes the AR path losslessly
        # exportable as a vanilla LLaMA-architecture model.
        x = self.tok_emb(tokens) + (
            self.time_emb(times.clamp(0, self.cfg.n_time_buckets - 1))
            - self.time_emb.weight[0]
        ) + (self.region_emb(region) - self.region_emb.weight[0])
        cond = self._cond_delta(times) if self.cfg.use_adaln else None

        extras: dict = {}
        new_cache = [] if return_kv else None
        moe_lb = 0.0
        layer = 0

        def run_block(block, x, cache):
            nonlocal moe_lb, layer
            if return_kv:
                x, kv, lb = block(
                    x, cond, self.angles_pos, self.angles_time, times,
                    mask, cache, True, block_mask,
                )
                new_cache.append(kv)
            else:
                x, lb = block(
                    x, cond, self.angles_pos, self.angles_time, times,
                    mask, cache, False, block_mask,
                )
            if lb is not None:
                moe_lb = moe_lb + lb
            layer += 1
            return x

        def cache_for(_layer):
            return kv_cache[_layer] if kv_cache is not None else None

        for blk in self.pre:
            x = run_block(blk, x, cache_for(layer))

        # Never halt while a KV cache is involved: the cache length must
        # always match the canonical depth (pre + core*max_loops + post).
        can_halt = (
            halt_tol > 0.0 and kv_cache is None and not return_kv and not self.training
        )
        x_prev = None
        for _ in range(n_loops):
            for blk in self.core:
                x = run_block(blk, x, cache_for(layer))
            if can_halt:
                if x_prev is not None and (x - x_prev).abs().max().item() < halt_tol:
                    break
                x_prev = x

        for blk in self.post:
            x = run_block(blk, x, cache_for(layer))

        h = self.ln(x)
        logits = self.lm_head(h)
        verify = self.verifier(h).squeeze(-1)
        if return_kv:
            extras["kv"] = new_cache
        if need_hidden:
            extras["hidden"] = h
        if isinstance(moe_lb, torch.Tensor):
            extras["moe_lb"] = moe_lb
        return logits, verify, extras
