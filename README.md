# Mariana

A tiny hybrid **Diffusion-AR language model** (DRV-LM) — single-command training,
generation, and export. It combines:

* **Causal next-token training** on a clean prefix (standard AR objective)
* **Block-wise masked-diffusion training** on a noisy draft region
  (bidirectional inside a block, causal across blocks)
* **Draft → Refine → Verify generation** with speculative acceptance against
  the AR head (lossless w.r.t. the AR distribution)
* **Lossless export** of the AR decoding path to **GGUF** (llama.cpp / Ollama),
  **ONNX**, and **safetensors**

## Install

```bash
python -m venv venv && source venv/bin/activate

pip install -r requirements.txt          # cross-platform, GPU default
pip install -r requirements-cpu.txt      # smaller CPU-only wheels
pip install -r requirements-xpu.txt      # Intel Arc / XPU
pip install -r requirements-export.txt   # GGUF / ONNX / safetensors toolchain
```

Device selection is automatic — **CUDA → Intel XPU → Apple MPS → CPU** — with
mixed precision (bf16/fp16 + GradScaler) chosen per device. Everything falls
back to CPU transparently; `--device cuda|xpu|mps|cpu|auto` overrides.

## Quick start

```bash
python mariana.py                        # train on TinyShakespeare + demo
python mariana.py --steps 3000           # legacy flag style still works
python -m mariana train --steps 3000 --compile
python -m mariana gen --prompt "KING: "
python -m mariana bench                  # hybrid vs AR speed comparison
```

On ≤ 4 GB VRAM use smaller batches + gradient accumulation:

```bash
python -m mariana train --batch 4 --grad-accum 4 --ctx 128 --no-gen
```

## Architecture (v2)

See `architecture.txt` for the full design spec. Implemented highlights:

| Feature | Flag / default |
| --- | --- |
| adaLN-zero timestep conditioning (DiT-style FiLM), exact identity at t=0 | default on (`--no-adaln`) |
| Full-spectrum LLaMA-exact position RoPE + time-axis rotation | built in |
| Block-causal hybrid attention (bidirectional in block, causal across) | `block_size` (`--block-size`) |
| Attention-mask annealing (causal → hybrid) | `--anneal-frac` |
| Block-wise independent diffusion timesteps | built in |
| Recurrent-depth core + loop dropout + inference-time halting | `--loops` |
| L_align consistency loss (trains speculative acceptance rate) | `w_align` |
| Verifier head + verifier-consensus early exit at inference | default on |
| Lossless speculative sampling (min(1, p/q), residual rejection) | built in |
| Single-pass parallel block verification with KV reuse | built in |
| Draft sampling temperature control | `gen --draft-temperature` |
| Multi-token-prediction auxiliary head | default on (`--no-mtp`) |
| Optional sparse-MoE FFN (top-2, load-balanced) | `--moe` |
| FlexAttention kernel path (GPU), SDPA fallback everywhere | `--flex` |
| Adaptive draft length driven by rolling acceptance | default on (`gen --no-adaptive`) |

The package layout:

```
mariana/
  config.py    device.py    rope.py    model.py
  data.py      train.py     generate.py cli.py
  export/      compat.py  gguf_export.py  onnx_export.py  safetensors.py
mariana.py     # thin backward-compatible wrapper
tests/         # CPU smoke tests (python tests/test_smoke.py)
```

## Export & test with Ollama / llama.cpp / ONNX

```bash
pip install -r requirements-export.txt
python -m mariana export --checkpoint mariana.pt --format all --out exports
```

produces

```
exports/mariana-f16.gguf     # LLaMA-architecture GGUF (F16; F32/BF16 via --dtype)
exports/Modelfile            # ready for Ollama
exports/mariana.onnx         # dynamic batch/seq axes, onnxruntime-verified
exports/mariana-llama.safetensors
```

The export is **lossless by construction**: committed/AR tokens always use
diffusion time bucket 0, where (a) the time/region embeddings are defined
relative to their baseline, (b) adaLN modulation is the identity, and (c) the
RoPE time axis contributes zero rotation — so the AR path *is* a vanilla
LLaMA transformer (RMSNorm + RoPE θ=10000 + SwiGLU + tied embeddings). The
converter only unrolls the recurrent core and permutes Q/K rows from the
interleaved to the half-split RoPE layout. A numeric parity test
(`tests/test_smoke.py::test_llama_compat_parity`) verifies bit-exactness.

Run it:

```bash
# llama.cpp
llama-cli -m exports/mariana-f16.gguf -p "KING: " -n 64
llama-quantize exports/mariana-f16.gguf exports/mariana-q4.gguf Q4_K_M   # optional

# Ollama
cd exports && ollama create mariana -f Modelfile && ollama run mariana "KING: "

# ONNX Runtime (Python)
python - <<'PY'
import onnxruntime as ort, numpy as np
s = ort.InferenceSession("exports/mariana.onnx")
ids = np.array([[256] + list(b"KING: ")])          # BOS + prompt bytes
print(s.run(["logits"], {"tokens": ids})[0][0, -1].argmax())
PY
```

Tokenizer: byte-level (256 byte tokens + `<|bos|>` + `<|mask|>`) embedded in
the GGUF via the GPT-2 bytes-to-unicode table, so prompts encode/decode as
raw bytes end-to-end.

## Training on Common Corpus (or any large corpus)

[Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) is ~2.3
trillion tokens across 10,000 parquet shards (~4 TB) -- stream a subset and
train on it as plain text:

```bash
pip install -r requirements-data.txt

# materialize N GiB of text (English by default; '' = all languages):
python scripts/prepare_common_corpus.py --gb 2 \
    --out /run/media/plz/CFN-ARCHIVE/LLMTrain/common_corpus_en.txt

# train on it:
python -m mariana train \
    --data /run/media/plz/CFN-ARCHIVE/LLMTrain/common_corpus_en.txt \
    --steps 100000 --eval-every 5000 --checkpoint mariana-cc.pt --no-gen
```

The corpus is stored as uint8 (1 byte/token), so multi-GiB corpora fit in
RAM.  One training epoch is `corpus_bytes / (batch x ctx)` steps; 1-2 epochs
on a multi-GB corpus is a good budget (see the discussion below for what is
realistic on a given GPU).

## Tests

```bash
python tests/test_smoke.py     # or: python -m pytest tests/ -q
```

Covers forward shapes, KV-cache exactness, corruption/losses, training
convergence, both generation modes, loop dropout/halting, GGUF + safetensors
export, and LLaMA-conversion numeric parity.

## Notes

* **On hybrid vs AR speed**: speculative drafting only pays off when the
  rolling acceptance rate is high (rule of thumb: > ~50%). Acceptance is
  maximised when the draft distribution matches the target temperature
  (acceptance = 1 - total-variation distance), which is the default; it
  improves with training because `L_align` explicitly optimises it. On an
  undertrained checkpoint expect the hybrid path to be *slower* than AR —
  that is the algorithm working as designed, not a bug. Use
  `python -m mariana bench --refine-steps 1` for the cheapest hybrid config.
* `--compile` mostly benefits training; inference may recompile for changing
  sequence lengths on the first run.
* `--flex` (FlexAttention) only engages on CUDA/XPU; on CPU/MPS the code
  automatically uses the numerically-equivalent SDPA mask path.
* Checkpoints store the full config (`cfg`), optimizer and AMP state —
  `--resume` restores everything.
