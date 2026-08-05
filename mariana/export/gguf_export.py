"""GGUF export (llama.cpp / Ollama compatible) of the AR decoding path.

The converted model is a vanilla LLaMA-architecture transformer (see
``compat.py``), so any GGUF consumer can load it:

    llama-cli -m mariana-f16.gguf -p "KING: "
    ollama create mariana -f Modelfile && ollama run mariana

Quantisation note: this writer emits F16/F32.  For Q8_0/Q4_K_M etc. run
``llama-quantize mariana-f16.gguf mariana-q4.gguf Q4_K_M`` from llama.cpp.
"""

from __future__ import annotations

import os


from ..config import Config
from .compat import llama_metadata, to_llama_state


def _require_gguf():
    try:
        import gguf

        return gguf
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "The 'gguf' package is required for GGUF export.\n"
            "Install it with:  pip install gguf"
        ) from e


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte<->unicode table (lets llama.cpp BPE encode raw bytes)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def export_gguf(state_dict: dict, cfg: Config, out_path: str,
                dtype: str = "f16", name: str = "mariana",
                write_modelfile: bool = True) -> str:
    gguf = _require_gguf()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    raw_dtype = {
        "f16": gguf.GGMLQuantizationType.F16,
        "f32": gguf.GGMLQuantizationType.F32,
        "bf16": gguf.GGMLQuantizationType.BF16,
    }[dtype.lower()]

    tensors = to_llama_state(state_dict, cfg)
    meta = llama_metadata(cfg)

    writer = gguf.GGUFWriter(out_path, arch="llama")
    writer.add_name(name)
    writer.add_description(
        "Mariana DRV-LM (hybrid diffusion-AR) -- AR decoding path, "
        "losslessly converted to the LLaMA architecture."
    )
    writer.add_context_length(meta["llama.context_length"])
    writer.add_embedding_length(meta["llama.embedding_length"])
    writer.add_block_count(meta["llama.block_count"])
    writer.add_feed_forward_length(meta["llama.feed_forward_length"])
    writer.add_head_count(meta["llama.attention.head_count"])
    writer.add_head_count_kv(meta["llama.attention.head_count_kv"])
    writer.add_layer_norm_rms_eps(meta["llama.attention.layer_norm_rms_epsilon"])
    writer.add_rope_dimension_count(meta["llama.rope.dimension_count"])
    writer.add_rope_freq_base(meta["llama.rope.freq_base"])
    writer.add_file_type(raw_dtype)

    # --- byte-level tokenizer via the GPT-2 bytes-to-unicode table --------
    b2u = _bytes_to_unicode()
    tokens = [b2u[i] for i in range(256)] + ["<|bos|>", "<|mask|>"]
    types = [gguf.TokenType.NORMAL] * 256 + [
        gguf.TokenType.CONTROL,
        gguf.TokenType.CONTROL,
    ]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges([])
    writer.add_bos_token_id(cfg.bos_token)
    writer.add_eos_token_id(cfg.bos_token)  # byte model: BOS doubles as stop
    if hasattr(writer, "add_mask_token_id"):
        writer.add_mask_token_id(cfg.mask_token)

    for tname, t in tensors.items():
        writer.add_tensor(tname, t.numpy(), raw_dtype=raw_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {out_path} ({dtype})")

    if write_modelfile:
        modelfile = os.path.join(
            os.path.dirname(os.path.abspath(out_path)), "Modelfile"
        )
        base = os.path.basename(out_path)
        with open(modelfile, "w") as f:
            f.write(
                f"# Ollama Modelfile for the Mariana DRV-LM (AR decoding path)\n"
                f"# Usage:  ollama create {name} -f Modelfile && ollama run {name}\n"
                f"FROM ./{base}\n"
                f"PARAMETER temperature 0.9\n"
                f"PARAMETER top_p 0.95\n"
                f"PARAMETER num_ctx {cfg.ctx_len}\n"
                f"PARAMETER stop \"<|bos|>\"\n"
            )
        print(f"Wrote {modelfile}")
    return out_path
