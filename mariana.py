#!/usr/bin/env python3
"""
Mariana: a tiny hybrid Diffusion-AR language model (DRV-LM).

Single-file, dependency-light trainer/generator.  It combines:
  * Causal next-token training on a clean prefix
  * Masked diffusion training on a noisy draft block
  * Draft-Refine-Verify generation with speculative acceptance

Run with no arguments to train on TinyShakespeare and generate a sample.

Examples:
    python mariana.py                       # train + demo
    python mariana.py --steps 3000
    python mariana.py --steps 3000 --compile   # faster training, slow first inference
    python mariana.py --gen --prompt "KING: "

Notes:
    * Designed to run on CPU, CUDA, or Intel XPU; auto-detects the best device.
    * On your machine (no CUDA driver), it will transparently use the CPU.
    * --compile is optional and mostly benefits training; inference may recompile
      for changing sequence lengths on the first run.
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# CPU / thread tuning before torch import
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", str(min(24, os.cpu_count() or 4)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    print("PyTorch is required. Install it with:")
    print("    pip install torch")
    print(
        "For CPU-only wheels (smaller): pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    raise e

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(1)
try:
    torch.set_flush_denormal(True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 258  # 256 bytes + BOS + MASK
    ctx_len: int = 256
    dim: int = 256
    n_heads: int = 4
    n_pre: int = 2
    n_core: int = 2
    n_post: int = 2
    mlp_mult: int = 4
    max_loops: int = 2
    dropout: float = 0.0
    bos_token: int = 256
    mask_token: int = 257


# ---------------------------------------------------------------------------
# Tokenizer / data
# ---------------------------------------------------------------------------
FALLBACK_TEXT = (
    "Mariana Trench, the deepest known point of Earth's oceans, "
    "lies in the western Pacific Ocean near Guam. Its floor reaches "
    "nearly eleven thousand meters below the surface. The pressure "
    "there is more than a thousand times that at sea level, yet life "
    "persists. Microbes, amphipods, and even strange translucent "
    "snailfish have been filmed in the hadal zone. Expeditions use "
    "specialized submersibles with titanium hulls and thick acrylic "
    "viewports to withstand the crushing weight. Light never reaches "
    "the bottom, so creatures rely on chemosynthesis near seeps and "
    "falls of organic matter from above. The water is near freezing, "
    "and the sediment is soft with fine ooze accumulated over millions "
    "of years. Studying these extremes teaches us about the limits of "
    "life and informs the search for organisms on icy moons. "
    "Engineers draw inspiration from deep-sea animals to design better "
    "materials and robots. Every dive reveals new species and reminds "
    "us how much of our own planet remains unexplored. "
)


def fetch_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache = ".tinyshakespeare.txt"
    try:
        if os.path.exists(cache) and os.path.getsize(cache) > 10000:
            return open(cache, "rb").read()
        print(f"Downloading {url} ...")
        data = urllib.request.urlopen(url, timeout=15).read()
        open(cache, "wb").write(data)
        return data
    except Exception as e:
        print(f"Download failed ({e}); using fallback corpus.")
        return FALLBACK_TEXT.encode("utf-8")


def build_dataset(data: bytes, ctx: int, val_frac=0.05):
    # tile small corpora to reach a usable size
    while len(data) < 500_000:
        data = data + data
    ids = torch.tensor(bytearray(data), dtype=torch.uint8).long().clamp(0, 255)
    n = (len(ids) // (ctx + 1)) * (ctx + 1)
    ids = ids[:n]
    ids = ids.view(-1, ctx + 1)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:]


# ---------------------------------------------------------------------------
# 2-D RoPE helpers
# ---------------------------------------------------------------------------
def build_rope_angles(
    dim: int, pos_len: int, time_len: int, base_pos=10000.0, base_time=100.0
):
    """
    Build standard 1-D RoPE angles for each head dimension.
    Even-indexed frequency slots encode position, odd-indexed slots encode time.
    Returns tensors of shape (pos_len, dim/2) and (time_len, dim/2).
    """
    half = dim // 2
    quarter = dim // 4
    i_pos = torch.arange(quarter, dtype=torch.float32)
    i_time = torch.arange(quarter, dtype=torch.float32)
    freq_pos = 1.0 / (base_pos ** (i_pos / half))
    freq_time = 1.0 / (base_time ** (i_time / half))

    pos = torch.arange(pos_len, dtype=torch.float32)
    time = torch.arange(time_len, dtype=torch.float32)

    angles_pos = torch.zeros(pos_len, half, dtype=torch.float32)
    angles_pos[:, 0::2] = torch.outer(pos, freq_pos)
    angles_time = torch.zeros(time_len, half, dtype=torch.float32)
    angles_time[:, 1::2] = torch.outer(time, freq_time)

    return angles_pos, angles_time


def apply_rope(x, angles_pos, angles_time, times):
    """
    Standard 2-D RoPE: rotate adjacent dimension pairs.
    x:      (B, H, T, D)
    times:  (B, T) integer time bucket indices
    """
    B, H, T, D = x.shape
    half = D // 2

    # position angles for each token (broadcast to batch)
    ang_pos = (
        angles_pos[:T].unsqueeze(0).unsqueeze(1).expand(B, 1, T, half).to(x.device)
    )
    ang_time = angles_time[times].unsqueeze(1)  # (B, 1, T, half)
    angles = ang_pos + ang_time  # (B, 1, T, half)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x1 = x[..., 0::2]  # (B, H, T, half)
    x2 = x[..., 1::2]

    x_out = torch.empty_like(x)
    x_out[..., 0::2] = x1 * cos - x2 * sin
    x_out[..., 1::2] = x1 * sin + x2 * cos
    return x_out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.dim % cfg.n_heads == 0
        self.head_dim = cfg.dim // cfg.n_heads
        self.nh = cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ln1 = RMSNorm(cfg.dim)
        self.ln2 = RMSNorm(cfg.dim)
        hidden = cfg.mlp_mult * cfg.dim
        self.fc1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.fc2 = nn.Linear(cfg.dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x, angles_pos, angles_time, times, mask, kv_cache=None, return_kv=False
    ):
        B, T, _ = x.shape
        residual = x
        x = self.ln1(x)

        qkv = self.qkv(x).view(B, T, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = apply_rope(q, angles_pos, angles_time, times)
        k = apply_rope(k, angles_pos, angles_time, times)

        if kv_cache is not None:
            ck, cv = kv_cache
            k = torch.cat([ck, k], dim=2)
            v = torch.cat([cv, v], dim=2)

        # SDPA on CPU supports efficient causal + custom masks in 2.3+
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.cfg.dim)
        out = self.o(out)
        x = residual + self.dropout(out)

        # SwiGLU MLP
        residual = x
        x = self.ln2(x)
        gate = F.silu(self.fc1(x))
        x = self.fc3(gate * self.fc2(x))
        x = residual + self.dropout(x)

        if return_kv:
            return x, (k, v)
        return x


class MarianaLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.time_emb = nn.Embedding(64, cfg.dim)
        self.region_emb = nn.Embedding(2, cfg.dim)

        self.pre = nn.ModuleList([Block(cfg) for _ in range(cfg.n_pre)])
        self.core = nn.ModuleList([Block(cfg) for _ in range(cfg.n_core)])
        self.post = nn.ModuleList([Block(cfg) for _ in range(cfg.n_post)])

        self.ln = RMSNorm(cfg.dim)
        self.verifier = nn.Linear(cfg.dim, 1, bias=False)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie weights

        self.register_buffer("angles_pos", None)
        self.register_buffer("angles_time", None)
        self._init_angles()
        self.apply(self._init_weights)

    def _init_angles(self):
        head_dim = self.cfg.dim // self.cfg.n_heads
        ap, at = build_rope_angles(head_dim, self.cfg.ctx_len * 2, 64)
        self.register_buffer("angles_pos", ap, persistent=False)
        self.register_buffer("angles_time", at, persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens, times, region, mask=None, kv_cache=None, return_kv=False):
        B, T = tokens.shape
        x = (
            self.tok_emb(tokens)
            + self.time_emb(times.clamp(0, 63))
            + self.region_emb(region)
        )

        # flatten loop count to a static graph
        blocks = list(self.pre) + list(self.core) * self.cfg.max_loops + list(self.post)

        new_cache = [] if return_kv else None
        for i, block in enumerate(blocks):
            cache = kv_cache[i] if kv_cache is not None else None
            if return_kv:
                x, kv = block(
                    x, self.angles_pos, self.angles_time, times, mask, cache, True
                )
                new_cache.append(kv)
            else:
                x = block(
                    x, self.angles_pos, self.angles_time, times, mask, cache, False
                )

        h = self.ln(x)
        logits = self.lm_head(h)
        verify = torch.sigmoid(self.verifier(h).squeeze(-1))
        if return_kv:
            return logits, verify, new_cache
        return logits, verify


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def make_hybrid_mask(L, cutoff):
    """
    Lower-triangular for positions < cutoff (autoregressive prefix).
    Bidirectional for positions >= cutoff (draft block) attending to all prefix + draft.
    """
    idx = torch.arange(L)
    is_draft = idx >= cutoff
    # causal if both row and col are in prefix and col <= row
    causal = (
        (~is_draft.unsqueeze(1))
        & (~is_draft.unsqueeze(0))
        & (idx.unsqueeze(1) >= idx.unsqueeze(0))
    )
    # draft rows see prefix + all draft
    draft_rows = is_draft.unsqueeze(1) & torch.ones(L, L, dtype=torch.bool)
    return causal | draft_rows


def corrupt_batch(x, mask_token, bos_token, ctx_len, max_t=1.0):
    B, L1 = x.shape
    x = x.clone()
    # Always start sequences with BOS
    x[:, 0] = bos_token

    originals = x[:, :-1].clone()  # self-prediction targets for diffusion
    targets = x[:, 1:].clone()  # next-token targets for AR
    inputs = x[:, :-1].clone()

    cutoff = torch.randint(ctx_len // 4, ctx_len - 4, (B,))
    times = torch.zeros(B, ctx_len, dtype=torch.long)
    region = torch.zeros(B, ctx_len, dtype=torch.long)
    masks = []

    for b in range(B):
        k = cutoff[b].item()
        region[b, k:] = 1
        # sample corruption level
        t = random.random() * max_t
        bucket = int(t * 63) + 1
        times[b, k:] = bucket
        # mask/replace draft tokens
        p_mask = 0.1 + 0.8 * t
        for pos in range(k, ctx_len):
            r = random.random()
            if r < p_mask:
                inputs[b, pos] = mask_token
            elif r < p_mask + 0.1:
                inputs[b, pos] = random.randint(0, 255)
        masks.append(make_hybrid_mask(ctx_len, k))

    attn_mask = torch.stack(masks).unsqueeze(1)  # (B,1,T,T)
    return inputs, targets, originals, times, region, attn_mask


def train_step(model, xb, optimizer, scaler, device, cfg, use_amp):
    inputs, targets, originals, times, region, attn_mask = corrupt_batch(
        xb, cfg.mask_token, cfg.bos_token, cfg.ctx_len
    )
    inputs, targets, originals = (
        inputs.to(device),
        targets.to(device),
        originals.to(device),
    )
    times, region, attn_mask = times.to(device), region.to(device), attn_mask.to(device)

    # Causal mask for the clean autoregressive pass
    T = cfg.ctx_len
    causal_mask = (
        torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0).unsqueeze(0)
    )
    zeros = torch.zeros_like(region)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        # 1) Causal AR pass on the clean sequence: teaches standard next-token prediction
        logits_ar, _ = model(originals, zeros, zeros, causal_mask)
        ar_loss = F.cross_entropy(
            logits_ar.reshape(-1, cfg.vocab_size), targets.reshape(-1)
        )

        # 2) Hybrid diffusion pass on the corrupted sequence
        logits, verify = model(inputs, times, region, attn_mask)

        diff_mask = region == 1
        diff_loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), originals.reshape(-1), reduction="none"
        )
        diff_loss = (
            diff_loss * diff_mask.reshape(-1).float()
        ).sum() / diff_mask.sum().clamp_min(1)

        # Verifier loss: predict whether draft token equals original
        with torch.no_grad():
            pred_tok = logits.argmax(-1)
            correct = (pred_tok == originals).float()
        ver_loss = F.binary_cross_entropy(verify, correct, reduction="none")
        ver_loss = (ver_loss * diff_mask.float()).sum() / diff_mask.sum().clamp_min(1)

        loss = ar_loss + 0.2 * diff_loss + 0.05 * ver_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), ar_loss.item(), diff_loss.item(), ver_loss.item()


@torch.no_grad()
def evaluate(model, val_loader, device, cfg, use_amp):
    model.eval()
    tot = ar = diff = ver = n = 0
    for xb in val_loader:
        inputs, targets, originals, times, region, attn_mask = corrupt_batch(
            xb[0], cfg.mask_token, cfg.bos_token, cfg.ctx_len, max_t=0.5
        )
        inputs, targets, originals = (
            inputs.to(device),
            targets.to(device),
            originals.to(device),
        )
        times, region, attn_mask = (
            times.to(device),
            region.to(device),
            attn_mask.to(device),
        )
        T = cfg.ctx_len
        causal_mask = (
            torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0).unsqueeze(0)
        )
        zeros = torch.zeros_like(region)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            logits_ar, _ = model(originals, zeros, zeros, causal_mask)
            logits, verify = model(inputs, times, region, attn_mask)
        ar_loss = F.cross_entropy(
            logits_ar.reshape(-1, cfg.vocab_size), targets.reshape(-1)
        )

        diff_mask = region == 1
        diff_loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), originals.reshape(-1), reduction="none"
        )
        diff_loss = (
            diff_loss * diff_mask.reshape(-1).float()
        ).sum() / diff_mask.sum().clamp_min(1)

        pred_tok = logits.argmax(-1)
        correct = (pred_tok == originals).float()
        ver_loss = F.binary_cross_entropy(verify, correct, reduction="none")
        ver_loss = (ver_loss * diff_mask.float()).sum() / diff_mask.sum().clamp_min(1)

        tot += (ar_loss + 0.2 * diff_loss + 0.05 * ver_loss).item()
        ar += ar_loss.item()
        diff += diff_loss.item()
        ver += ver_loss.item()
        n += 1
    model.train()
    return tot / n, ar / n, diff / n, ver / n


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def sample_logits(logits, temperature=0.9, top_p=0.95, greedy=False, banned=None):
    if banned is not None:
        logits = logits.clone()
        logits[..., banned] = float("-inf")
    if greedy:
        return logits.argmax(-1), None
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_p, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probs = probs.scatter(-1, sorted_i, sorted_p.masked_fill(remove, 0.0))
        probs = probs / probs.sum(-1, keepdim=True)
    tok = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(
        probs.shape[:-1]
    )
    return tok, probs


def prefill(model, tokens, device, cfg):
    B, T = tokens.shape
    times = torch.zeros(B, T, dtype=torch.long, device=device)
    region = torch.zeros(B, T, dtype=torch.long, device=device)
    mask = torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0).unsqueeze(0)
    logits, _, kv = model(tokens, times, region, mask, return_kv=True)
    return logits[:, -1, :], kv


def draft_refine_verify(
    model,
    prompt_bytes,
    max_new_tokens=200,
    draft_len=16,
    n_steps=4,
    temperature=0.9,
    top_p=0.95,
    device="cpu",
    cfg=None,
):
    model.eval()
    ctx = cfg.ctx_len
    output = [cfg.bos_token] + list(prompt_bytes[: ctx - max_new_tokens - 8])
    accepted_total = 0
    drafted_total = 0
    t0 = time.time()

    # prefill prompt
    tokens = torch.tensor([output], dtype=torch.long, device=device)
    last_logits, kv_cache = prefill(model, tokens, device, cfg)

    while len(output) - 1 - len(prompt_bytes) < max_new_tokens:
        start = len(output)
        end = min(start + draft_len, ctx)
        if start >= ctx - 1:
            break
        if end - start < 2:
            break
        draft_len_real = end - start

        # Draft block: all MASK tokens, uniform time
        draft = torch.full(
            (1, draft_len_real), cfg.mask_token, dtype=torch.long, device=device
        )
        mask_positions = list(range(draft_len_real))
        committed = torch.full(
            (1, draft_len_real), cfg.mask_token, dtype=torch.long, device=device
        )

        # Refine with decreasing noise schedule
        for step in range(n_steps):
            frac = (step + 1) / n_steps
            bucket = max(1, int((1 - frac) * 63))
            t_block = torch.full(
                (1, draft_len_real), bucket, dtype=torch.long, device=device
            )
            r_block = torch.ones((1, draft_len_real), dtype=torch.long, device=device)

            # Draft sees the full cached prefix + all draft positions bidirectionally
            T_total = start + draft_len_real
            mask = torch.ones(
                1, 1, draft_len_real, T_total, dtype=torch.bool, device=device
            )

            logits_d, verify_d = model(
                draft, t_block, r_block, mask, kv_cache=kv_cache, return_kv=False
            )
            logits_d = logits_d[:, -draft_len_real:]
            verify_d = verify_d[:, -draft_len_real:]

            target_unmask = max(1, int(len(mask_positions) * frac))
            if mask_positions and target_unmask > 0:
                logp = F.log_softmax(logits_d / max(temperature, 1e-6), dim=-1)
                conf, _ = logp.max(dim=-1)  # (1, draft_len_real)
                # only consider still-masked positions
                conf_mask = torch.full_like(conf, -1e9)
                conf_mask[0, mask_positions] = conf[0, mask_positions]
                _, topk_idx = torch.topk(
                    conf_mask[0], min(target_unmask, len(mask_positions))
                )

                for idx in topk_idx:
                    idx = idx.item()
                    tok, _ = sample_logits(
                        logits_d[:, idx],
                        temperature,
                        top_p,
                        greedy=False,
                        banned=[cfg.bos_token, cfg.mask_token],
                    )
                    tok_i = tok.item()
                    committed[0, idx] = tok_i
                    draft[0, idx] = tok_i
                    if idx in mask_positions:
                        mask_positions.remove(idx)

            # Optional verifier remask on low-confidence committed tokens
            if step < n_steps - 1 and mask_positions:
                low = verify_d[0, :] < 0.1
                for idx in range(draft_len_real):
                    if low[idx].item() and idx not in mask_positions:
                        draft[0, idx] = cfg.mask_token
                        mask_positions.append(idx)

        # Verification pass over draft in causal order
        draft_tokens = committed[0].tolist()
        drafted_total += draft_len_real
        accepted = 0
        for i, dtok in enumerate(draft_tokens):
            pos = start + i
            if pos >= ctx:
                break
            t_single = torch.zeros((1, 1), dtype=torch.long, device=device)
            r_single = torch.zeros((1, 1), dtype=torch.long, device=device)
            # mask covers cached prefix (length pos) + current token
            mask_causal = torch.ones(
                (1, 1, 1, pos + 1), dtype=torch.bool, device=device
            )

            # Evaluate draft token WITHOUT updating KV cache
            logits_ar, _ = model(
                torch.tensor([[dtok]], device=device),
                t_single,
                r_single,
                mask_causal,
                kv_cache=kv_cache,
                return_kv=False,
            )
            logits_ar = logits_ar[:, -1, :]
            p_ar = F.softmax(logits_ar / max(temperature, 1e-6), dim=-1)
            p_dtok = p_ar[0, dtok].item()

            # Speculative-style acceptance: accept with probability p_ar(dtok)
            if random.random() < p_dtok:
                output.append(dtok)
                accepted += 1
                # Commit accepted token to KV cache
                _, _, kv_cache = model(
                    torch.tensor([[dtok]], device=device),
                    t_single,
                    r_single,
                    mask_causal,
                    kv_cache=kv_cache,
                    return_kv=True,
                )
                if len(output) - 1 - len(prompt_bytes) >= max_new_tokens:
                    break
            else:
                # Reject: sample a replacement from the AR distribution and
                # commit it to the cache, then start a fresh draft block.
                p_ar_banned = p_ar.clone()
                p_ar_banned[0, [cfg.bos_token, cfg.mask_token]] = 0.0
                p_ar_banned = p_ar_banned / p_ar_banned.sum(-1, keepdim=True)
                ar_tok = torch.multinomial(p_ar_banned[0], num_samples=1).item()
                output.append(ar_tok)
                _, _, kv_cache = model(
                    torch.tensor([[ar_tok]], device=device),
                    t_single,
                    r_single,
                    mask_causal,
                    kv_cache=kv_cache,
                    return_kv=True,
                )
                break
        accepted_total += accepted

    elapsed = time.time() - t0
    gen_bytes = bytes(output[1:])  # strip BOS
    return (
        gen_bytes,
        accepted_total / max(drafted_total, 1),
        len(output) / max(elapsed, 1e-6),
        elapsed,
    )


# ---------------------------------------------------------------------------
# Benchmark AR baseline
# ---------------------------------------------------------------------------
def ar_generate(
    model,
    prompt_bytes,
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95,
    device="cpu",
    cfg=None,
):
    model.eval()
    output = [cfg.bos_token] + list(prompt_bytes[: cfg.ctx_len - max_new_tokens - 8])
    tokens = torch.tensor([output], dtype=torch.long, device=device)
    last_logits, kv_cache = prefill(model, tokens, device, cfg)

    t0 = time.time()
    for _ in range(max_new_tokens):
        tok, _ = sample_logits(
            last_logits,
            temperature,
            top_p,
            banned=[cfg.bos_token, cfg.mask_token],
        )
        tok = tok.item()
        output.append(tok)
        if len(output) >= cfg.ctx_len:
            break
        t_single = torch.zeros((1, 1), dtype=torch.long, device=device)
        r_single = torch.zeros((1, 1), dtype=torch.long, device=device)
        mask_causal = torch.ones(
            (1, 1, 1, len(output)), dtype=torch.bool, device=device
        )
        last_logits, _, kv_cache = model(
            torch.tensor([[tok]], device=device),
            t_single,
            r_single,
            mask_causal,
            kv_cache=kv_cache,
            return_kv=True,
        )
        last_logits = last_logits[:, -1, :]
    elapsed = time.time() - t0
    return bytes(output[1:]), len(output) / max(elapsed, 1e-6), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mariana tiny hybrid Diffusion-AR LM")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gen", action="store_true", help="Generate from checkpoint")
    parser.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    parser.add_argument("--checkpoint", type=str, default="mariana.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--loops", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--gen-len", type=int, default=200)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}  Threads: {torch.get_num_threads()}")

    # Load checkpoint config when generating/resuming so dimensions match
    if (args.gen or args.resume) and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = Config(**ckpt["cfg"])
        if args.loops is not None:
            cfg.max_loops = args.loops
        if args.gen:
            print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")
    else:
        cfg = Config(ctx_len=args.ctx, dim=args.dim, n_heads=args.heads)
        if args.loops is not None:
            cfg.max_loops = args.loops

    model = MarianaLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    if args.gen:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint {args.checkpoint} not found. Train first.")
            sys.exit(1)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        out, acc_rate, tps, elapsed = draft_refine_verify(
            model,
            args.prompt.encode("utf-8"),
            max_new_tokens=args.gen_len,
            device=device,
            cfg=cfg,
        )
        print(f"\n--- Generated ({tps:.1f} tok/s, accept {acc_rate:.2%}) ---")
        print(out.decode("utf-8", errors="replace"))
        return

    # Data
    data = fetch_shakespeare()
    train_ids, val_ids = build_dataset(data, cfg.ctx_len)
    train_loader = DataLoader(
        TensorDataset(train_ids), batch_size=args.batch, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_ids), batch_size=args.batch, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
    )

    start_step = 0
    if args.resume and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    use_amp = device.type in ("cuda", "xpu")
    scaler = None

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile unavailable: {e}")

    model.train()
    t_start = time.time()
    train_iter = iter(train_loader)
    for step in range(start_step, args.steps):
        lr = (
            args.lr
            * min(1.0, (step + 1) / args.warmup)
            * 0.5
            * (1 + math.cos(math.pi * step / args.steps))
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        try:
            xb = next(train_iter)[0]
        except StopIteration:
            train_iter = iter(train_loader)
            xb = next(train_iter)[0]

        loss, ar_loss, diff_loss, ver_loss = train_step(
            model, xb, optimizer, scaler, device, cfg, use_amp
        )

        if (step + 1) % 25 == 0 or step == start_step:
            tok_per_sec = (
                (args.batch * cfg.ctx_len)
                / max(time.time() - t_start, 1e-6)
                * min(step + 1 - start_step, 25)
            )
            print(
                f"step {step + 1:5d}/{args.steps} | loss {loss:.3f} | ar {ar_loss:.3f} "
                f"| diff {diff_loss:.3f} | ver {ver_loss:.3f} | lr {lr:.2e} | "
                f"tok/s {tok_per_sec:.0f}"
            )

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            vloss, var, vdiff, vver = evaluate(model, val_loader, device, cfg, use_amp)
            print(
                f"VAL step {step + 1}: loss {vloss:.3f} | ar {var:.3f} | diff {vdiff:.3f} | ver {vver:.3f}"
            )
            torch.save(
                {
                    "step": step + 1,
                    "model": model.state_dict()
                    if not args.compile
                    else model._orig_mod.state_dict()
                    if hasattr(model, "_orig_mod")
                    else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": asdict(cfg),
                },
                args.checkpoint,
            )
            print(f"Saved {args.checkpoint}")
            model.train()

    # Final demo generation
    print("\n--- Final generation demo ---")
    model.eval()
    prompt = args.prompt.encode("utf-8")
    out, acc_rate, tps, elapsed = draft_refine_verify(
        model, prompt, max_new_tokens=args.gen_len, device=device, cfg=cfg
    )
    print(f"Hybrid: {tps:.1f} tok/s, acceptance {acc_rate:.2%}, time {elapsed:.2f}s")
    print(out.decode("utf-8", errors="replace"))

    if args.bench:
        out_ar, tps_ar, elapsed_ar = ar_generate(
            model, prompt, max_new_tokens=args.gen_len, device=device, cfg=cfg
        )
        print(f"\nAR baseline: {tps_ar:.1f} tok/s, time {elapsed_ar:.2f}s")
        print(out_ar.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
