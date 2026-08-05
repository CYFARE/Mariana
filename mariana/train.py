"""Training: hybrid causal + block-diffusion objective.

Upgrades over the original:

* **Block-wise corruption & timesteps** -- the draft region is split into
  ``block_size`` blocks, each with an independent noise level.  Attention is
  *block-causal* (bidirectional inside a block, causal across blocks), which
  is the modern block-diffusion design and stays KV-cache compatible.
* **Attention-mask annealing** -- early in training a fraction of batches use
  a purely causal mask, annealing to the block-hybrid mask (stabilises the
  causal -> bidirectional transition).
* **L_align consistency loss** -- KL pulling diffusion marginals toward the
  (detached) AR conditionals; directly optimises speculative acceptance.
* **Loop dropout** -- the recurrent core depth is sampled per step so every
  loop count is a valid operating point (enables inference-time halting).
* **Auxiliary MTP loss** and optional MoE load-balancing loss.
"""

from __future__ import annotations

import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import build_dataset, fetch_corpus
from .device import Amp, empty_cache
from .model import HAS_FLEX, MarianaLM

if HAS_FLEX:
    from torch.nn.attention.flex_attention import create_block_mask


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def make_causal_mask(L: int) -> torch.Tensor:
    return torch.tril(torch.ones(L, L, dtype=torch.bool))


def make_block_mask(L: int, cutoff: int, block_size: int) -> torch.Tensor:
    """Causal prefix + block-causal draft (bidirectional inside each block)."""
    idx = torch.arange(L)
    # causal everywhere by default
    allow = idx.unsqueeze(1) >= idx.unsqueeze(0)
    # draft rows additionally see their whole block (future positions inside it)
    if cutoff < L:
        block_end = ((idx - cutoff).clamp_min(0) // block_size + 1) * block_size + cutoff
        block_end = block_end.clamp_max(L)
        draft_rows = idx >= cutoff
        allow = allow | (draft_rows.unsqueeze(1) & (idx.unsqueeze(0) < block_end.unsqueeze(1)))
    return allow


def build_flex_block_mask(cutoffs: torch.Tensor, ctx: int, block_size: int, device):
    """FlexAttention BlockMask equivalent of :func:`make_block_mask` (batched)."""
    cutoffs = cutoffs.to(device)

    def mask_mod(b, h, q, kv):
        k = cutoffs[b]
        causal = q >= kv
        qd = q >= k
        block_end = ((q - k).clamp_min(0) // block_size + 1) * block_size + k
        return causal | (qd & (kv < block_end))

    return create_block_mask(mask_mod, B=len(cutoffs), H=None, Q_LEN=ctx, KV_LEN=ctx, device=device)


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------
def corrupt_batch(x, cfg: Config, max_t: float = 1.0, p_causal: float = 0.0,
                  use_flex: bool = False, device=None):
    """Return inputs/targets/etc for one hybrid training batch.

    * prefix [0, cutoff): clean, causal, AR loss
    * draft  [cutoff, L): split into blocks with independent noise levels
    """
    B, L1 = x.shape
    L = cfg.ctx_len
    x = x.long()  # uint8 storage -> int64 token ids
    x[:, 0] = cfg.bos_token

    originals = x[:, :-1].clone()  # self-prediction targets (diffusion)
    targets = x[:, 1:].clone()     # next-token targets (AR)
    inputs = x[:, :-1].clone()

    cutoffs = torch.randint(L // 4, L - 4, (B,))
    times = torch.zeros(B, L, dtype=torch.long)
    region = torch.zeros(B, L, dtype=torch.long)
    masks = []

    for b in range(B):
        k = int(cutoffs[b])
        region[b, k:] = 1
        causal_only = random.random() < p_causal  # mask-annealing batch
        pos = k
        while pos < L:
            bend = min(pos + cfg.block_size, L)
            t = random.random() * max_t
            bucket = int(t * (cfg.n_time_buckets - 1)) + 1
            times[b, pos:bend] = bucket
            p_mask = 0.1 + 0.8 * t
            for p in range(pos, bend):
                r = random.random()
                if r < p_mask:
                    inputs[b, p] = cfg.mask_token
                elif r < p_mask + 0.1:
                    inputs[b, p] = random.randint(0, 255)
            pos = bend
        if causal_only:
            masks.append(make_causal_mask(L))
        else:
            masks.append(make_block_mask(L, k, cfg.block_size))

    attn_mask = torch.stack(masks).unsqueeze(1)  # (B,1,L,L)

    flex_bm = None
    if use_flex and HAS_FLEX and device is not None:
        flex_bm = build_flex_block_mask(cutoffs, L, cfg.block_size, device)
        attn_mask = None  # FlexAttention path takes precedence

    return inputs, targets, originals, times, region, attn_mask, flex_bm


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _to(device, *tensors):
    return [t.to(device, non_blocking=True) if t is not None else None for t in tensors]


def hybrid_losses(model, inputs, targets, originals, times, region, attn_mask,
                  flex_bm, cfg: Config, causal_mask, n_loops_diff: int):
    """One AR pass + one block-diffusion pass; returns the loss dict."""
    zeros = torch.zeros_like(region)

    # 1) Causal AR pass on the clean sequence (canonical full depth)
    logits_ar, _, extras = model(originals, zeros, zeros, causal_mask,
                                 need_hidden=cfg.use_mtp)
    ar_loss = F.cross_entropy(logits_ar.reshape(-1, cfg.vocab_size), targets.reshape(-1))

    # 2) Hybrid block-diffusion pass on the corrupted sequence
    logits, verify, extras_d = model(
        inputs, times, region, attn_mask, n_loops=n_loops_diff, block_mask=flex_bm
    )

    diff_mask = region == 1
    flat_mask = diff_mask.reshape(-1).float()
    n_diff = diff_mask.sum().clamp_min(1)

    diff_loss = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), originals.reshape(-1), reduction="none"
    )
    diff_loss = (diff_loss * flat_mask).sum() / n_diff

    # 3) L_align: pull diffusion marginals toward detached AR conditionals.
    #    AR logit at position j-1 is the conditional for token j.
    # earliest draft position across the batch
    k_min = int((region == 1).float().argmax(dim=1).min().item())
    with torch.no_grad():
        ar_cond = F.log_softmax(logits_ar[:, k_min - 1:-1].detach(), dim=-1)
    diff_logp = F.log_softmax(logits[:, k_min:], dim=-1)
    align_loss = F.kl_div(diff_logp, ar_cond, log_target=True, reduction="none").sum(-1)
    align_loss = (align_loss * diff_mask[:, k_min:].float()).sum() / n_diff

    # 4) Verifier: predict whether the drafted token equals the original
    with torch.no_grad():
        correct = (logits.argmax(-1) == originals).float()
    ver_loss = F.binary_cross_entropy_with_logits(verify, correct, reduction="none")
    ver_loss = (ver_loss * diff_mask.float()).sum() / n_diff

    losses = {
        "ar": ar_loss,
        "diff": diff_loss,
        "align": align_loss,
        "ver": ver_loss,
        "total": cfg.w_ar * ar_loss + cfg.w_diff * diff_loss
        + cfg.w_align * align_loss + cfg.w_ver * ver_loss,
    }

    # 5) Multi-token prediction: hidden state at i predicts token i+2
    if cfg.use_mtp and model.mtp_head is not None:
        mtp_logits = model.mtp_head(extras["hidden"][:, :-2])
        mtp_target = originals[:, 2:]
        mtp_loss = F.cross_entropy(
            mtp_logits.reshape(-1, cfg.vocab_size), mtp_target.reshape(-1)
        )
        losses["mtp"] = mtp_loss
        losses["total"] = losses["total"] + cfg.w_mtp * mtp_loss

    if "moe_lb" in extras_d:
        losses["total"] = losses["total"] + cfg.w_moe_lb * extras_d["moe_lb"]

    return losses


def train_step(model, xb, device, cfg: Config, amp: Amp, causal_mask,
               use_flex: bool, p_causal: float, grad_accum: int, do_step: bool,
               optimizer, step: int, total_steps: int):
    inputs, targets, originals, times, region, attn_mask, flex_bm = corrupt_batch(
        xb, cfg, p_causal=p_causal, use_flex=use_flex, device=device
    )
    inputs, targets, originals, times, region, attn_mask = _to(
        device, inputs, targets, originals, times, region, attn_mask
    )

    n_loops = (
        random.randint(1, cfg.max_loops) if cfg.loop_dropout else cfg.max_loops
    )

    with amp.autocast():
        losses = hybrid_losses(
            model, inputs, targets, originals, times, region, attn_mask,
            flex_bm, cfg, causal_mask, n_loops,
        )
        loss = losses["total"] / grad_accum

    amp.scale_backward(loss)
    if do_step:
        if amp.use_scaler:
            amp.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        amp.step(optimizer)
        optimizer.zero_grad(set_to_none=True)

    return {k: float(v.detach()) for k, v in losses.items()}


@torch.no_grad()
def evaluate(model, val_loader, device, cfg: Config, amp: Amp, causal_mask,
             max_batches: int = 20):
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    for xb in val_loader:
        inputs, targets, originals, times, region, attn_mask, _ = corrupt_batch(
            xb[0], cfg, max_t=0.5
        )
        inputs, targets, originals, times, region, attn_mask = _to(
            device, inputs, targets, originals, times, region, attn_mask
        )
        with amp.autocast():
            losses = hybrid_losses(
                model, inputs, targets, originals, times, region, attn_mask,
                None, cfg, causal_mask, cfg.max_loops,
            )
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
        if n >= max_batches:
            break
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, cfg: Config, step: int, amp: Amp,
                    compiled: bool):
    sd = model._orig_mod.state_dict() if compiled and hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(
        {
            "step": step,
            "model": sd,
            "optimizer": optimizer.state_dict(),
            "amp": amp.state_dict(),
            "cfg": cfg.to_dict(),
            "format": 2,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def train(args, cfg: Config, device) -> MarianaLM:
    model = MarianaLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M  Depth: {cfg.n_layers} layers "
          f"({cfg.n_pre}+{cfg.n_core}x{cfg.max_loops}+{cfg.n_post})")

    data = fetch_corpus(getattr(args, "data", None))
    train_ids, val_ids = build_dataset(data, cfg.ctx_len)
    train_loader = DataLoader(
        TensorDataset(train_ids), batch_size=args.batch, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(TensorDataset(val_ids), batch_size=args.batch)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
    )
    amp = Amp(device, force=getattr(args, "amp", False))

    start_step = 0
    if args.resume and os.path.exists(args.checkpoint):
        ckpt = load_checkpoint(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        amp.load_state_dict(ckpt.get("amp", {}))
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    compiled = False
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compiled = True
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile unavailable: {e}")

    use_flex = getattr(args, "flex", False) and HAS_FLEX and device.type in ("cuda", "xpu")
    if getattr(args, "flex", False) and not use_flex:
        # FlexAttention has no CPU backward kernel; SDPA masks are equivalent
        print("FlexAttention unavailable on this device/torch; using SDPA masks.")

    causal_mask = (
        make_causal_mask(cfg.ctx_len).to(device).unsqueeze(0).unsqueeze(0)
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()
    t_last = t_start
    train_iter = iter(train_loader)

    for step in range(start_step, args.steps):
        progress = step / max(args.steps, 1)
        lr = (
            args.lr
            * min(1.0, (step + 1) / max(args.warmup, 1))
            * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # mask annealing: start ~50% causal-only batches, anneal to 0
        p_causal = 0.5 * max(0.0, 1.0 - progress / max(args.anneal_frac, 1e-6))

        metrics: dict[str, float] = {}
        for accum_idx in range(args.grad_accum):
            try:
                xb = next(train_iter)[0]
            except StopIteration:
                train_iter = iter(train_loader)
                xb = next(train_iter)[0]
            losses = train_step(
                model, xb, device, cfg, amp, causal_mask, use_flex, p_causal,
                args.grad_accum, accum_idx == args.grad_accum - 1,
                optimizer, step, args.steps,
            )
            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0.0) + v / args.grad_accum

        if (step + 1) % 25 == 0 or step == start_step:
            now = time.time()
            tok_per_sec = (
                args.batch * cfg.ctx_len * args.grad_accum * min(step + 1 - start_step, 25)
                / max(now - t_last, 1e-6)
            )
            t_last = now
            msg = " | ".join(f"{k} {v:.3f}" for k, v in metrics.items())
            print(f"step {step + 1:5d}/{args.steps} | {msg} | lr {lr:.2e} | tok/s {tok_per_sec:.0f}")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            vlosses = evaluate(model, val_loader, device, cfg, amp, causal_mask)
            msg = " | ".join(f"{k} {v:.3f}" for k, v in vlosses.items())
            print(f"VAL step {step + 1}: {msg}")
            save_checkpoint(args.checkpoint, model, optimizer, cfg, step + 1, amp, compiled)
            print(f"Saved {args.checkpoint}")
            model.train()

    empty_cache(device)
    return model
