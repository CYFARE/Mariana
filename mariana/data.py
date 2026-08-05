"""Byte-level data pipeline (platform-independent, no tokenizer deps)."""

from __future__ import annotations

import os
import urllib.request

import torch

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

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def fetch_corpus(path: str | None = None, cache: str = ".tinyshakespeare.txt") -> bytes:
    """Load a local corpus or download TinyShakespeare (with cache + fallback)."""
    if path:
        with open(path, "rb") as f:
            return f.read()
    try:
        if os.path.exists(cache) and os.path.getsize(cache) > 10000:
            with open(cache, "rb") as f:
                return f.read()
        print(f"Downloading {SHAKESPEARE_URL} ...")
        data = urllib.request.urlopen(SHAKESPEARE_URL, timeout=15).read()
        with open(cache, "wb") as f:
            f.write(data)
        return data
    except Exception as e:
        print(f"Download failed ({e}); using fallback corpus.")
        return FALLBACK_TEXT.encode("utf-8")


def build_dataset(data: bytes, ctx: int, val_frac: float = 0.05):
    """Chunk raw bytes into (ctx+1) rows; kept as uint8 (1 byte/token) so
    multi-GB corpora fit in RAM -- batches are promoted to int64 inside
    ``corrupt_batch`` instead."""
    # tile small corpora to reach a usable size
    while len(data) < 500_000:
        data = data + data
    ids = torch.tensor(bytearray(data), dtype=torch.uint8)
    n = (len(ids) // (ctx + 1)) * (ctx + 1)
    ids = ids[:n].view(-1, ctx + 1)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:]
