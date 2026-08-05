"""Mariana: a tiny hybrid Diffusion-AR language model (DRV-LM)."""

from .config import Config
from .model import MarianaLM

__version__ = "2.0.0"
__all__ = ["Config", "MarianaLM", "__version__"]
