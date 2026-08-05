"""ONNX export of the AR decoding path (dynamic batch/sequence axes)."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from ..config import Config
from ..model import MarianaLM


class ARWrapper(nn.Module):
    """tokens (B, T) -> logits (B, T, V) for the pure causal AR path."""

    def __init__(self, model: MarianaLM):
        super().__init__()
        # no weight surgery needed: time/region embeddings are relative to
        # the t=0 / region=0 baseline, so the AR path is already vanilla
        self.model = copy.deepcopy(model).float().cpu().eval()
        self.cfg = model.cfg

    def forward(self, tokens):
        B, T = tokens.shape
        device = tokens.device
        times = torch.zeros(B, T, dtype=torch.long, device=device)
        region = torch.zeros(B, T, dtype=torch.long, device=device)
        mask = torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0).unsqueeze(0)
        logits, _, _ = self.model(tokens, times, region, mask)
        return logits


def export_onnx(model: MarianaLM, cfg: Config, out_path: str,
                verify: bool = True, dynamo: bool | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wrapper = ARWrapper(model)
    dummy = torch.randint(0, 256, (1, 8), dtype=torch.long)

    kwargs = dict(
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={"tokens": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
    )
    try:
        if dynamo:
            torch.onnx.export(wrapper, (dummy,), out_path, dynamo=True, **kwargs)
        else:
            torch.onnx.export(wrapper, (dummy,), out_path, dynamo=False, **kwargs)
    except Exception as e:
        if dynamo is not False:
            print(f"dynamo exporter failed ({e}); retrying legacy exporter")
            torch.onnx.export(wrapper, (dummy,), out_path, dynamo=False, **kwargs)
        else:
            raise
    print(f"Wrote {out_path}")

    if verify:
        _verify(wrapper, out_path)
    return out_path


def _verify(wrapper: ARWrapper, out_path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping numeric verification.")
        return
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = torch.randint(0, 256, (1, 16), dtype=torch.long)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    got = sess.run(["logits"], {"tokens": x.numpy()})[0]
    max_err = float(abs(ref - got).max())
    print(f"onnxruntime check: max |logit err| = {max_err:.5f}")
    if max_err > 1e-2:
        print("WARNING: ONNX logits diverge from PyTorch reference.")
