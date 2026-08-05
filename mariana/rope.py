"""2-D Rotary Position Embedding (position axis + diffusion-time axis).

Design notes
------------
* **Position axis is full-spectrum and LLaMA-exact.**  Every dimension pair
  ``j`` rotates with frequency ``base_pos ** (-j / half)`` which is identical
  to the standard LLaMA schedule ``theta ** (-2j / head_dim)``.  When the
  diffusion time bucket is 0 (committed / AR tokens) the model therefore
  reduces *exactly* to a vanilla LLaMA transformer -- this is what makes
  lossless GGUF/ONNX export of the AR decoding path possible.

* **Time axis rotates only the odd pair slots** (frequency schedule with a
  much smaller base).  It acts as an additive extra rotation, so it carries
  the noise-level signal without disturbing the positional geometry.

Internally the interleaved (GPT-J style) pair layout is used; the export
code permutes Q/K weights into the half-split (NeoX/LLaMA) layout.
"""

from __future__ import annotations

import torch


def build_rope_angles(
    dim: int,
    pos_len: int,
    time_len: int,
    base_pos: float = 10000.0,
    base_time: float = 100.0,
):
    """Return ``(angles_pos, angles_time)`` of shape ``(len, dim//2)``.

    ``angles_pos[j]  = pos * base_pos  ** (-j / half)``          (all slots)
    ``angles_time[j] = t   * base_time ** (-(j//2) / half)``     (odd slots only)
    """
    half = dim // 2
    quarter = dim // 4
    j = torch.arange(half, dtype=torch.float32)
    freq_pos = 1.0 / (base_pos ** (j / half))

    i_time = torch.arange(quarter, dtype=torch.float32)
    freq_time = 1.0 / (base_time ** (i_time / half))

    pos = torch.arange(pos_len, dtype=torch.float32)
    time = torch.arange(time_len, dtype=torch.float32)

    angles_pos = torch.outer(pos, freq_pos)  # (pos_len, half)

    angles_time = torch.zeros(time_len, half, dtype=torch.float32)
    angles_time[:, 1::2] = torch.outer(time, freq_time)

    return angles_pos, angles_time


def apply_rope(x, angles_pos, angles_time, times):
    """Rotate adjacent (interleaved) dimension pairs.

    x:      (B, H, T, D)
    times:  (B, T) integer time-bucket indices
    """
    B, H, T, D = x.shape
    half = D // 2

    ang_pos = (
        angles_pos[:T].unsqueeze(0).unsqueeze(1).expand(B, 1, T, half).to(x.device)
    )
    ang_time = angles_time[times].unsqueeze(1)  # (B, 1, T, half)
    angles = ang_pos + ang_time  # (B, 1, T, half)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    x_out = torch.empty_like(x)
    x_out[..., 0::2] = x1 * cos - x2 * sin
    x_out[..., 1::2] = x1 * sin + x2 * cos
    return x_out


def interleaved_to_neox_permutation(head_dim: int) -> torch.Tensor:
    """Permutation that maps interleaved pair layout to half-split (NeoX).

    Applying ``w[perm]`` to the rows of a Q/K projection converts weights
    trained with the interleaved convention into the layout llama.cpp /
    GGUF / HF-LLaMA expect, without changing the math.
    """
    perm = list(range(0, head_dim, 2)) + list(range(1, head_dim, 2))
    assert len(perm) == head_dim
    return torch.tensor(perm, dtype=torch.long)
