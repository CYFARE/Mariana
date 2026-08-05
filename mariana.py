#!/usr/bin/env python3
"""Mariana: a tiny hybrid Diffusion-AR language model (DRV-LM).

This is a thin backward-compatible wrapper around the ``mariana`` package.

Examples:
    python mariana.py                      # train + demo (same as: -m mariana train)
    python mariana.py --steps 3000
    python mariana.py gen --prompt "KING: "
    python mariana.py bench
    python mariana.py export --format all  # GGUF + ONNX + safetensors -> exports/

Device selection is automatic: CUDA -> Intel XPU -> Apple MPS -> CPU.
"""

import os

# Mitigate CUDA fragmentation on small GPUs by using expandable segments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", str(min(24, os.cpu_count() or 4)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

try:
    import torch
except ImportError as e:
    print("PyTorch is required. Install it with:")
    print("    pip install -r requirements.txt          # cross-platform (GPU default)")
    print("    pip install -r requirements-cpu.txt      # CPU-only wheels")
    raise e

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(1)
try:
    torch.set_flush_denormal(True)
except Exception:
    pass

from mariana.cli import main

if __name__ == "__main__":
    main()
