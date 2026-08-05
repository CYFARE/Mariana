"""Cross-platform device & mixed-precision selection.

Preference order for ``--device auto``: CUDA -> Intel XPU -> Apple MPS -> CPU.
Any explicitly requested device that is unavailable falls back to CPU with a
warning, so the same command line works on every machine.
"""

from __future__ import annotations

import contextlib

import torch


def _available(name: str) -> bool:
    try:
        if name == "cuda":
            return torch.cuda.is_available()
        if name == "xpu":
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        if name == "mps":
            return (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
        if name == "cpu":
            return True
    except Exception:
        return False
    return False


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve a device string to an available torch.device (GPU first)."""
    if requested != "auto":
        if _available(requested):
            return torch.device(requested)
        print(f"[device] requested '{requested}' is unavailable; falling back to CPU")
        return torch.device("cpu")
    for name in ("cuda", "xpu", "mps", "cpu"):
        if _available(name):
            return torch.device(name)
    return torch.device("cpu")  # pragma: no cover


def bf16_supported(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type == "xpu":
        # XPU supports bf16 on Arc/Data-Centre GPUs
        return True
    if device.type == "mps":
        return True  # autocast bf16 is supported on Apple silicon
    if device.type == "cpu":
        try:
            return bool(torch.cpu.is_bf16_supported())
        except Exception:
            return False
    return False


class Amp:
    """Uniform autocast + GradScaler handling across devices.

    dtype resolution:
      * CUDA/XPU: bf16 when supported, else fp16 (with GradScaler)
      * MPS:      bf16 autocast (no scaler; unsupported on MPS)
      * CPU:      fp32 by default (bf16 if ``force`` and supported)
    """

    def __init__(self, device: torch.device, force: bool = False):
        self.device = device
        if device.type in ("cuda", "xpu"):
            if bf16_supported(device):
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        elif device.type == "mps":
            self.dtype = torch.bfloat16
        else:  # cpu
            self.dtype = torch.bfloat16 if (force and bf16_supported(device)) else torch.float32

        self.enabled = self.dtype != torch.float32
        self.use_scaler = self.dtype == torch.float16 and device.type in ("cuda", "xpu")
        self.scaler = torch.GradScaler(device.type) if self.use_scaler else None

    def autocast(self):
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def scale_backward(self, loss: torch.Tensor) -> None:
        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.use_scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> dict:
        return self.scaler.state_dict() if self.use_scaler else {}

    def load_state_dict(self, sd: dict) -> None:
        if self.use_scaler and sd:
            self.scaler.load_state_dict(sd)


def empty_cache(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def describe(device: torch.device) -> str:
    parts = [f"Device: {device}", f"Threads: {torch.get_num_threads()}"]
    try:
        if device.type == "cuda":
            p = torch.cuda.get_device_properties(device)
            parts.append(f"GPU: {p.name} ({p.total_memory / 1024**3:.1f} GiB)")
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            p = torch.xpu.get_device_properties(device)
            parts.append(f"GPU: {p.name} ({p.total_memory / 1024**3:.1f} GiB)")
    except Exception:
        pass
    return "  ".join(parts)
