"""Inference: iterative Draft -> Refine -> Verify loop with speculative acceptance.

Upgrades over the original:

* **Lossless speculative sampling** -- each draft token d is verified against
  the AR conditional *at its own position* and accepted with
  ``min(1, p_AR(d) / q_draft(d))``; rejected positions are resampled from the
  residual distribution ``relu(p_AR - q_draft)``.  This preserves the exact
  AR sampling distribution while committing accepted tokens in bulk.
* **Adaptive draft length** -- N grows with the rolling acceptance rate and
  shrinks when the verifier is uncertain (the "speed law" from
  ``architecture.txt``: tokens/step ~ acceptance x N, so optimise both).
* **Verifier-consensus early exit** -- refinement stops as soon as every
  position is unmasked and the verifier agrees, saving diffusion steps.
* **Entropy-ordered unmasking** with nucleus sampling (kept, now block aware).
* Device-agnostic autocast (CUDA/XPU/MPS bf16, fp32 CPU fallback).
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from .config import Config


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


def _autocast(device):
    enabled = device.type in ("cuda", "xpu", "mps")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def prefill(model, tokens, device, cfg: Config):
    B, T = tokens.shape
    times = torch.zeros(B, T, dtype=torch.long, device=device)
    region = torch.zeros(B, T, dtype=torch.long, device=device)
    mask = torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0).unsqueeze(0)
    with _autocast(device):
        logits, _, extras = model(tokens, times, region, mask, return_kv=True)
    return logits[:, -1, :], extras["kv"]


@torch.no_grad()
def draft_refine_verify(
    model,
    prompt_bytes,
    max_new_tokens: int = 200,
    draft_len: int | None = None,
    n_steps: int = 2,
    temperature: float = 0.9,
    top_p: float = 0.95,
    device="cpu",
    cfg: Config = None,
    adaptive_draft: bool = True,
    draft_temperature: float | None = None,
):
    """draft_temperature: sampling temperature for the diffusion draft.
    ``None`` -> match the target ``temperature``; ``0`` -> greedy drafting
    (usually much higher acceptance; output stays lossless thanks to the
    residual-rejection sampling against the AR head)."""
    model.eval()
    ctx = cfg.ctx_len
    if draft_len is None:
        draft_len = cfg.block_size
    dt_temp = temperature if draft_temperature is None else draft_temperature
    banned = [cfg.bos_token, cfg.mask_token]

    def _draft_sample(logits_col):
        """Return (token, q_distribution) for one draft position."""
        if dt_temp <= 0.0:  # greedy: q is a point mass at the argmax
            lg = logits_col.clone().float()
            lg[..., banned] = float("-inf")
            tok = int(lg.argmax(-1).item())
            q = torch.zeros(logits_col.shape[-1], device=logits_col.device)
            q[tok] = 1.0
            return tok, q
        tok, probs = sample_logits(logits_col, dt_temp, top_p, banned=banned)
        return int(tok.item()), probs[0].float()
    draft_len_min, draft_len_max = 4, min(2 * cfg.block_size, ctx // 4)

    output = [cfg.bos_token] + list(prompt_bytes[: ctx - max_new_tokens - 8])
    accepted_total = 0
    drafted_total = 0
    rolling_accept = 0.5
    t0 = time.time()

    tokens = torch.tensor([output], dtype=torch.long, device=device)
    pending, kv_cache = prefill(model, tokens, device, cfg)

    while len(output) - 1 - len(prompt_bytes) < max_new_tokens:
        start = len(output)
        end = min(start + draft_len, ctx)
        if start >= ctx - 1 or end - start < 2:
            break
        draft_len_real = end - start

        # --- A. Draft: all-MASK block, bidirectional within the block -------
        draft = torch.full(
            (1, draft_len_real), cfg.mask_token, dtype=torch.long, device=device
        )
        mask_positions = list(range(draft_len_real))
        committed = torch.full(
            (1, draft_len_real), cfg.mask_token, dtype=torch.long, device=device
        )
        # per-position draft distributions q(d) for speculative sampling
        draft_probs = torch.zeros(
            1, draft_len_real, cfg.vocab_size, dtype=torch.float32, device=device
        )

        # --- B. Refine: decreasing noise schedule + verifier remasking ------
        for step in range(n_steps):
            frac = (step + 1) / n_steps
            bucket = max(1, int((1 - frac) * (cfg.n_time_buckets - 1)))
            t_block = torch.full(
                (1, draft_len_real), bucket, dtype=torch.long, device=device
            )
            r_block = torch.ones((1, draft_len_real), dtype=torch.long, device=device)

            T_total = start + draft_len_real
            mask = torch.ones(
                1, 1, draft_len_real, T_total, dtype=torch.bool, device=device
            )

            with _autocast(device):
                logits_d, verify_d, _ = model(
                    draft, t_block, r_block, mask, kv_cache=kv_cache
                )
            logits_d = logits_d[:, -draft_len_real:]
            verify_d = verify_d[:, -draft_len_real:]

            target_unmask = max(1, int(len(mask_positions) * frac))
            if mask_positions and target_unmask > 0:
                logp = F.log_softmax(logits_d / max(dt_temp, 1e-6), dim=-1)
                conf, _ = logp.max(dim=-1)
                conf_mask = torch.full_like(conf, -1e9)
                conf_mask[0, mask_positions] = conf[0, mask_positions]
                _, topk_idx = torch.topk(
                    conf_mask[0], min(target_unmask, len(mask_positions))
                )
                for idx in topk_idx:
                    idx = idx.item()
                    tok, q = _draft_sample(logits_d[:, idx])
                    committed[0, idx] = tok
                    draft[0, idx] = tok
                    draft_probs[0, idx] = q
                    if idx in mask_positions:
                        mask_positions.remove(idx)

            # verifier consensus early exit: everything committed + trusted
            if not mask_positions:
                min_conf = torch.sigmoid(verify_d[0]).min().item()
                if min_conf > 0.9:
                    break
            # re-noise low-confidence committed tokens for the next step
            elif step < n_steps - 1:
                low = torch.sigmoid(verify_d[0, :]) < 0.1
                for idx in range(draft_len_real):
                    if low[idx].item() and idx not in mask_positions:
                        draft[0, idx] = cfg.mask_token
                        mask_positions.append(idx)

        # fill any positions the refinement left masked
        if mask_positions:
            for idx in mask_positions:
                tok, q = _draft_sample(logits_d[:, idx])
                committed[0, idx] = tok
                draft_probs[0, idx] = q

        # --- C/D. Verify: single-pass parallel speculative sampling ----------
        # One causal forward over the whole draft block yields the AR
        # distribution at every draft position at once; accepted tokens reuse
        # the block's KV entries, and only the first rejection needs one
        # repair forward.  ``pending`` is the AR distribution for the first
        # draft position (from prefill or the previous block).
        draft_tokens = committed[0].tolist()
        n = len(draft_tokens)
        drafted_total += n

        dt = torch.tensor([draft_tokens], dtype=torch.long, device=device)
        zeros_tn = torch.zeros((1, n), dtype=torch.long, device=device)
        vmask = torch.ones(1, 1, n, start + n, dtype=torch.bool, device=device)
        vmask[0, 0, :, start:] = torch.tril(
            torch.ones(n, n, dtype=torch.bool, device=device)
        )
        with _autocast(device):
            vlogits, _, vextras = model(
                dt, zeros_tn, zeros_tn, vmask, kv_cache=kv_cache, return_kv=True
            )
        block_kv = vextras["kv"]

        commit_toks: list[int] = []
        rejected = False
        for i, dtok in enumerate(draft_tokens):
            pos = start + i
            if pos >= ctx or len(output) - 1 - len(prompt_bytes) >= max_new_tokens:
                break
            p = F.softmax(
                (pending if i == 0 else vlogits[:, i - 1])
                / max(temperature, 1e-6),
                dim=-1,
            )
            p[0, [cfg.bos_token, cfg.mask_token]] = 0.0
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
            q = draft_probs[0, i]
            q_dtok = q[dtok].item()

            # accept with min(1, p/q); reject from the residual distribution
            accept_prob = min(1.0, p[0, dtok].item() / max(q_dtok, 1e-9))
            if (
                dtok not in (cfg.bos_token, cfg.mask_token)
                and q_dtok > 0.0
                and random.random() < accept_prob
            ):
                commit_toks.append(dtok)
            else:
                resid = (p[0] - q).clamp_min(0.0)
                if resid.sum() < 1e-9:
                    resid = p[0]
                resid = resid / resid.sum()
                commit_toks.append(torch.multinomial(resid, num_samples=1).item())
                rejected = True
                break

        output.extend(commit_toks)
        m = len(commit_toks)
        accepted = m - (1 if rejected else 0)
        accepted_total += accepted

        def _truncate(kv, length):
            return [(k[:, :, :length], v[:, :, :length]) for k, v in kv]

        if not rejected:
            # everything committed: block KV entries are exact
            kv_cache = _truncate(block_kv, start + m)
            pending = vlogits[:, m - 1]
        else:
            # keep KV for the accepted prefix; the resampled token differs
            # from the draft, so it needs one repair forward
            kv_cache = _truncate(block_kv, start + m - 1)
            t_single = torch.zeros((1, 1), dtype=torch.long, device=device)
            r_single = torch.zeros((1, 1), dtype=torch.long, device=device)
            mask_causal = torch.ones(
                (1, 1, 1, start + m), dtype=torch.bool, device=device
            )
            with _autocast(device):
                logits_next, _, extras = model(
                    torch.tensor([[commit_toks[-1]]], device=device),
                    t_single, r_single, mask_causal,
                    kv_cache=kv_cache, return_kv=True,
                )
            kv_cache = extras["kv"]
            pending = logits_next[:, -1, :]

        # adaptive draft length: follow the rolling acceptance rate
        if adaptive_draft and drafted_total > 0:
            rate = accepted / max(draft_len_real, 1)
            rolling_accept = 0.7 * rolling_accept + 0.3 * rate
            if rolling_accept > 0.8 and draft_len < draft_len_max:
                draft_len = min(draft_len + 4, draft_len_max)
            elif rolling_accept < 0.4 and draft_len > draft_len_min:
                draft_len = max(draft_len - 4, draft_len_min)

    elapsed = time.time() - t0
    gen_bytes = bytes(output[1:])  # strip BOS
    return (
        gen_bytes,
        accepted_total / max(drafted_total, 1),
        len(output) / max(elapsed, 1e-6),
        elapsed,
    )


@torch.no_grad()
def ar_generate(
    model,
    prompt_bytes,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.95,
    device="cpu",
    cfg: Config = None,
):
    """Plain autoregressive baseline (what the exported GGUF/ONNX model does)."""
    model.eval()
    output = [cfg.bos_token] + list(prompt_bytes[: cfg.ctx_len - max_new_tokens - 8])
    tokens = torch.tensor([output], dtype=torch.long, device=device)
    last_logits, kv_cache = prefill(model, tokens, device, cfg)

    t0 = time.time()
    for _ in range(max_new_tokens):
        tok, _ = sample_logits(
            last_logits, temperature, top_p,
            banned=[cfg.bos_token, cfg.mask_token],
        )
        tok = tok.item()
        output.append(tok)
        if len(output) >= cfg.ctx_len:
            break
        t_single = torch.zeros((1, 1), dtype=torch.long, device=device)
        r_single = torch.zeros((1, 1), dtype=torch.long, device=device)
        mask_causal = torch.ones((1, 1, 1, len(output)), dtype=torch.bool, device=device)
        with _autocast(device):
            logits, _, extras = model(
                torch.tensor([[tok]], device=device),
                t_single, r_single, mask_causal,
                kv_cache=kv_cache, return_kv=True,
            )
        kv_cache = extras["kv"]
        last_logits = logits[:, -1, :]
    elapsed = time.time() - t0
    return bytes(output[1:]), len(output) / max(elapsed, 1e-6), elapsed
