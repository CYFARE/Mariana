"""Mariana command line interface.

Subcommands
-----------
    python -m mariana train  [--steps N] [--batch B] [--device auto] ...
    python -m mariana gen    --prompt "KING: " [--gen-len 50]
    python -m mariana bench  [--gen-len 50]          # hybrid vs AR speed test
    python -m mariana export --format gguf|onnx|safetensors|all [--out DIR]

Backward compatibility: ``python mariana.py --steps 3000`` (no subcommand)
still works and is treated as ``train``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from .config import Config
from .device import describe, pick_device


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-pre", type=int, default=2)
    p.add_argument("--n-core", type=int, default=2)
    p.add_argument("--n-post", type=int, default=2)
    p.add_argument("--loops", type=int, default=None, help="max_loops override")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--moe", action="store_true", help="enable sparse-MoE FFN")
    p.add_argument("--no-mtp", action="store_true", help="disable MTP head")
    p.add_argument("--no-adaln", action="store_true", help="disable adaLN conditioning")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=str, default="mariana.pt")
    p.add_argument("--device", type=str, default="auto",
                   help="auto|cuda|xpu|mps|cpu (auto prefers GPU, falls back to CPU)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mariana", description="Mariana tiny hybrid Diffusion-AR LM (DRV-LM)"
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- train ---------------------------------------------------------------
    p = sub.add_parser("train", help="train the hybrid model")
    _add_common_args(p)
    _add_model_args(p)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--flex", action="store_true", help="use FlexAttention kernel")
    p.add_argument("--amp", action="store_true", help="force AMP on CPU too")
    p.add_argument("--data", type=str, default=None, help="local corpus file")
    p.add_argument("--anneal-frac", type=float, default=0.5,
                   help="fraction of training over which causal-mask annealing decays")
    p.add_argument("--gen-len", type=int, default=50)
    p.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    p.add_argument("--no-gen", action="store_true", help="skip final demo generation")
    p.add_argument("--bench", action="store_true", help="run AR baseline after training")

    # -- gen ------------------------------------------------------------------
    p = sub.add_parser("gen", help="generate from a checkpoint")
    _add_common_args(p)
    p.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    p.add_argument("--gen-len", type=int, default=50)
    p.add_argument("--draft-len", type=int, default=None)
    p.add_argument("--refine-steps", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--draft-temperature", type=float, default=None,
                   help="draft sampling temperature (0 = greedy drafting; "
                        "default: match --temperature)")
    p.add_argument("--ar", action="store_true", help="plain AR decoding (no diffusion)")
    p.add_argument("--no-adaptive", action="store_true", help="fixed draft length")

    # -- bench ----------------------------------------------------------------
    p = sub.add_parser("bench", help="hybrid vs AR speed comparison")
    _add_common_args(p)
    p.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    p.add_argument("--gen-len", type=int, default=50)
    p.add_argument("--refine-steps", type=int, default=2)
    p.add_argument("--draft-temperature", type=float, default=None,
                   help="draft sampling temperature (0 = greedy drafting)")

    # -- export ---------------------------------------------------------------
    p = sub.add_parser("export", help="convert checkpoint to GGUF/ONNX/safetensors")
    _add_common_args(p)
    p.add_argument("--format", type=str, default="gguf",
                   choices=["gguf", "onnx", "safetensors", "all"])
    p.add_argument("--out", type=str, default="exports")
    p.add_argument("--dtype", type=str, default="f16", choices=["f16", "f32", "bf16"])
    p.add_argument("--name", type=str, default="mariana")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--raw", action="store_true",
                   help="safetensors: save raw Mariana weights instead of LLaMA-converted")

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"train", "gen", "bench", "export"}
    if argv and argv[0] not in cmds and not argv[0].startswith("-h") and argv[0] != "--help":
        argv = ["train"] + argv  # legacy flag style -> train
    elif not argv:
        argv = ["train"]
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_cfg_and_model(args, device):
    from .model import MarianaLM
    from .train import load_checkpoint

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint {args.checkpoint} not found. Train first.")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = Config.from_dict(ckpt["cfg"])
    if getattr(args, "loops", None) is not None:
        cfg.max_loops = args.loops
    model = MarianaLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")
    return cfg, model, ckpt


def cfg_from_args(args) -> Config:
    cfg = Config(
        ctx_len=args.ctx,
        dim=args.dim,
        n_heads=args.heads,
        n_pre=args.n_pre,
        n_core=args.n_core,
        n_post=args.n_post,
        block_size=args.block_size,
        use_moe=args.moe,
        use_mtp=not args.no_mtp,
        use_adaln=not args.no_adaln,
    )
    if args.loops is not None:
        cfg.max_loops = args.loops
    return cfg


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_train(args) -> None:
    from .generate import ar_generate, draft_refine_verify
    from .train import load_checkpoint, train

    device = pick_device(args.device)
    print(describe(device))
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if vram < 6:
            print("Low VRAM detected. Consider --batch 4 --grad-accum 4 --ctx 128.")

    # when resuming, prefer the checkpoint config so dimensions match
    if args.resume and os.path.exists(args.checkpoint):
        ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
        cfg = Config.from_dict(ckpt["cfg"])
        if args.loops is not None:
            cfg.max_loops = args.loops
    else:
        cfg = cfg_from_args(args)

    model = train(args, cfg, device)

    if not args.no_gen:
        from .device import empty_cache

        empty_cache(device)
        print("\n--- Final generation demo (Draft-Refine-Verify) ---")
        model.eval()
        out, acc, tps, elapsed = draft_refine_verify(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            device=device, cfg=cfg,
        )
        print(f"Hybrid: {tps:.1f} tok/s, acceptance {acc:.2%}, time {elapsed:.2f}s")
        print(out.decode("utf-8", errors="replace"))

    if args.bench:
        out_ar, tps_ar, elapsed_ar = ar_generate(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            device=device, cfg=cfg,
        )
        print(f"\nAR baseline: {tps_ar:.1f} tok/s, time {elapsed_ar:.2f}s")
        print(out_ar.decode("utf-8", errors="replace"))


def cmd_gen(args) -> None:
    from .generate import ar_generate, draft_refine_verify

    device = pick_device(args.device)
    print(describe(device))
    cfg, model, _ = load_cfg_and_model(args, device)

    if args.ar:
        out, tps, elapsed = ar_generate(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            temperature=args.temperature, top_p=args.top_p, device=device, cfg=cfg,
        )
        print(f"\n--- AR generation ({tps:.1f} tok/s) ---")
    else:
        out, acc, tps, elapsed = draft_refine_verify(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            draft_len=args.draft_len, n_steps=args.refine_steps,
            temperature=args.temperature, top_p=args.top_p, device=device, cfg=cfg,
            adaptive_draft=not args.no_adaptive,
            draft_temperature=args.draft_temperature,
        )
        print(f"\n--- Hybrid generation ({tps:.1f} tok/s, accept {acc:.2%}) ---")
    print(out.decode("utf-8", errors="replace"))


def cmd_bench(args) -> None:
    from .generate import ar_generate, draft_refine_verify

    device = pick_device(args.device)
    print(describe(device))
    cfg, model, _ = load_cfg_and_model(args, device)
    prompt = args.prompt.encode()

    out, acc, tps, el = draft_refine_verify(
        model, prompt, max_new_tokens=args.gen_len, device=device, cfg=cfg,
        n_steps=args.refine_steps, draft_temperature=args.draft_temperature,
    )
    print(f"Hybrid DRV: {tps:.1f} tok/s | acceptance {acc:.2%} | {el:.2f}s")
    print(out.decode("utf-8", errors="replace"))
    out_ar, tps_ar, el_ar = ar_generate(
        model, prompt, max_new_tokens=args.gen_len, device=device, cfg=cfg
    )
    print(f"\nAR baseline: {tps_ar:.1f} tok/s | {el_ar:.2f}s")
    print(out_ar.decode("utf-8", errors="replace"))
    print(f"\nSpeedup: {tps / max(tps_ar, 1e-9):.2f}x")


def cmd_export(args) -> None:
    device = torch.device("cpu")  # export is device-independent; keep it CPU
    cfg, model, ckpt = load_cfg_and_model(args, device)
    sd = ckpt["model"]
    os.makedirs(args.out, exist_ok=True)
    fmts = ["gguf", "onnx", "safetensors"] if args.format == "all" else [args.format]

    for fmt in fmts:
        if fmt == "gguf":
            from .export.gguf_export import export_gguf

            export_gguf(sd, cfg, os.path.join(args.out, f"{args.name}-{args.dtype}.gguf"),
                        dtype=args.dtype, name=args.name)
        elif fmt == "onnx":
            from .export.onnx_export import export_onnx

            export_onnx(model, cfg, os.path.join(args.out, f"{args.name}.onnx"),
                        verify=not args.no_verify)
        elif fmt == "safetensors":
            from .export.safetensors import export_safetensors

            suffix = "raw" if args.raw else "llama"
            export_safetensors(sd, cfg, os.path.join(args.out, f"{args.name}-{suffix}.safetensors"),
                               llama_compat=not args.raw)


def main(argv=None) -> None:
    args = parse_args(argv)
    {"train": cmd_train, "gen": cmd_gen, "bench": cmd_bench, "export": cmd_export}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
