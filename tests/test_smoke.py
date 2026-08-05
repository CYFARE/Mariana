"""Smoke tests for the Mariana package (CPU, tiny config, fast).

Run:  python -m pytest tests/ -q   (or plain: python tests/test_smoke.py)
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mariana.config import Config
from mariana.model import MarianaLM
from mariana.train import corrupt_batch, hybrid_losses, make_block_mask
from mariana.generate import ar_generate, draft_refine_verify


def tiny_cfg() -> Config:
    return Config(
        ctx_len=32, dim=64, n_heads=4, n_pre=1, n_core=1, n_post=1,
        max_loops=2, block_size=16, mlp_mult=2,
    )


def test_forward_shapes():
    cfg = tiny_cfg()
    model = MarianaLM(cfg)
    x = torch.randint(0, 256, (2, cfg.ctx_len))
    times = torch.zeros(2, cfg.ctx_len, dtype=torch.long)
    region = torch.zeros(2, cfg.ctx_len, dtype=torch.long)
    mask = make_block_mask(cfg.ctx_len, 16, cfg.block_size).unsqueeze(0).unsqueeze(0)
    logits, verify, extras = model(x, times, region, mask, need_hidden=True)
    assert logits.shape == (2, cfg.ctx_len, cfg.vocab_size)
    assert verify.shape == (2, cfg.ctx_len)
    assert "hidden" in extras


def test_kv_cache_consistency():
    """Cached decode must match a full forward pass (AR path exactness)."""
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    T = 12
    x = torch.randint(0, 256, (1, T))
    zeros = torch.zeros(1, T, dtype=torch.long)
    causal = torch.tril(torch.ones(T, T)).bool().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        full_logits, _, _ = model(x, zeros, zeros, causal)
        # prefill first half, decode rest token by token
        t0 = torch.zeros(1, 6, dtype=torch.long)
        m0 = torch.tril(torch.ones(6, 6)).bool().unsqueeze(0).unsqueeze(0)
        logits, _, ex = model(x[:, :6], t0, t0, m0, return_kv=True)
        kv = ex["kv"]
        outs = [logits[:, -1]]
        for i in range(6, T):
            tok = x[:, i : i + 1]
            t1 = torch.zeros(1, 1, dtype=torch.long)
            m1 = torch.ones(1, 1, 1, i + 1, dtype=torch.bool)
            logits, _, ex = model(tok, t1, t1, m1, kv_cache=kv, return_kv=True)
            kv = ex["kv"]
            outs.append(logits[:, -1])
        cached = torch.stack(outs, dim=1)  # logits at positions 5..T-1
    torch.testing.assert_close(cached, full_logits[:, 5:T], atol=1e-4, rtol=1e-4)


def test_corrupt_and_loss():
    cfg = tiny_cfg()
    model = MarianaLM(cfg)
    x = torch.randint(0, 256, (4, cfg.ctx_len + 1))
    inputs, targets, originals, times, region, attn_mask, flex_bm = corrupt_batch(x, cfg)
    assert inputs.shape == (4, cfg.ctx_len)
    assert (inputs[:, 0] == cfg.bos_token).all()
    assert (times[region == 0] == 0).all()
    causal = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).bool().unsqueeze(0).unsqueeze(0)
    losses = hybrid_losses(model, inputs, targets, originals, times, region,
                           attn_mask, None, cfg, causal, cfg.max_loops)
    for k in ("ar", "diff", "align", "ver", "total"):
        assert k in losses and torch.isfinite(losses[k])
    losses["total"].backward()


def test_training_reduces_loss():
    cfg = tiny_cfg()
    model = MarianaLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 256, (8, cfg.ctx_len + 1))
    causal = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).bool().unsqueeze(0).unsqueeze(0)
    first = last = None
    for _ in range(8):
        inputs, targets, originals, times, region, attn_mask, _ = corrupt_batch(x, cfg)
        losses = hybrid_losses(model, inputs, targets, originals, times, region,
                               attn_mask, None, cfg, causal, cfg.max_loops)
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        first = first or float(losses["total"])
        last = float(losses["total"])
    assert last < first, f"loss did not decrease: {first} -> {last}"


def test_generation_runs():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    out, acc, tps, _ = draft_refine_verify(
        model, b"HELLO: ", max_new_tokens=16, n_steps=2,
        device=torch.device("cpu"), cfg=cfg,
    )
    assert len(out) > 0
    out_ar, _, _ = ar_generate(model, b"HELLO: ", max_new_tokens=8,
                               device=torch.device("cpu"), cfg=cfg)
    assert len(out_ar) > 0


def test_loop_dropout_and_halting():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    x = torch.randint(0, 256, (1, cfg.ctx_len))
    zeros = torch.zeros(1, cfg.ctx_len, dtype=torch.long)
    causal = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).bool().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        for n in (1, 2):
            logits, _, _ = model(x, zeros, zeros, causal, n_loops=n)
            assert logits.shape[-1] == cfg.vocab_size
        logits, _, _ = model(x, zeros, zeros, causal, halt_tol=1e-3)
        assert torch.isfinite(logits).all()


def test_export_gguf_and_safetensors():
    from mariana.export.gguf_export import export_gguf
    from mariana.export.safetensors import export_safetensors

    cfg = tiny_cfg()
    model = MarianaLM(cfg)
    sd = model.state_dict()
    with tempfile.TemporaryDirectory() as d:
        gguf_path = export_gguf(sd, cfg, os.path.join(d, "m.gguf"), dtype="f32")
        assert os.path.getsize(gguf_path) > 1000
        with open(gguf_path, "rb") as f:
            assert f.read(4) == b"GGUF"
        assert os.path.exists(os.path.join(d, "Modelfile"))

        st_path = export_safetensors(sd, cfg, os.path.join(d, "m.safetensors"))
        from safetensors import safe_open

        with safe_open(st_path, framework="pt") as f:
            keys = list(f.keys())
        assert any(k.startswith("blk.") for k in keys)
        assert "token_embd.weight" in keys
        n_layers = cfg.n_pre + cfg.n_core * cfg.max_loops + cfg.n_post
        assert f"blk.{n_layers - 1}.attn_q.weight" in keys


def test_llama_compat_parity():
    """Converted LLaMA-format weights must reproduce the AR forward exactly."""
    from mariana.export.compat import to_llama_state

    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    sd = model.state_dict()
    llama_sd = to_llama_state(sd, cfg)

    T = 16
    x = torch.randint(0, 256, (1, T))
    zeros = torch.zeros(1, T, dtype=torch.long)
    causal = torch.tril(torch.ones(T, T)).bool().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        ref, _, _ = model(x, zeros, zeros, causal)

    # manual LLaMA-architecture forward using the converted weights
    import torch.nn.functional as F

    def rms(h, w, eps=1e-6):
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w

    h = llama_sd["token_embd.weight"][x]
    half = cfg.head_dim // 2
    pos = torch.arange(T, dtype=torch.float32)
    freqs = 1.0 / (cfg.rope_base_pos ** (torch.arange(half).float() / half))
    ang = torch.outer(pos, freqs)  # (T, half) NeoX layout
    cos, sin = torch.cos(ang)[None, None], torch.sin(ang)[None, None]

    def rope(t):  # (1, nh, T, hd) half-split rotation
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

    n_layers = cfg.n_layers
    for l in range(n_layers):
        p = f"blk.{l}."
        r = h
        hn = rms(h, llama_sd[p + "attn_norm.weight"])
        q = (hn @ llama_sd[p + "attn_q.weight"].T).view(1, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = (hn @ llama_sd[p + "attn_k.weight"].T).view(1, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = (hn @ llama_sd[p + "attn_v.weight"].T).view(1, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        q, k = rope(q), rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(1, T, cfg.dim) @ llama_sd[p + "attn_output.weight"].T
        h = r + o
        r = h
        hn = rms(h, llama_sd[p + "ffn_norm.weight"])
        h = r + (F.silu(hn @ llama_sd[p + "ffn_gate.weight"].T)
                 * (hn @ llama_sd[p + "ffn_up.weight"].T)) @ llama_sd[p + "ffn_down.weight"].T
    logits = rms(h, llama_sd["output_norm.weight"]) @ llama_sd["output.weight"].T
    torch.testing.assert_close(logits, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            import traceback

            print(f"FAIL {name}: {e}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
