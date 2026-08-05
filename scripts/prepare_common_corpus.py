#!/usr/bin/env python3
"""Download a subset of PleIAs/common_corpus and save it as plain text.

The full dataset is ~2.3 trillion tokens in 10,000 parquet files (~4 TB) --
far too large to mirror, and HF's streaming endpoint caps out early for this
repo layout.  This script instead downloads *whole* parquet shards with
``hf_hub_download`` (resumable, interrupt-safe), extracts the ``text`` column
with pyarrow, applies optional language/collection filters, and writes raw
text that ``mariana train --data`` can consume directly.

Examples
--------
# 2 GB of English text onto the archive drive (default paths):
python scripts/prepare_common_corpus.py --gb 2

# multilingual + code, custom location, keep the parquet shards for later:
python scripts/prepare_common_corpus.py --gb 10 --language "" --keep-parquet

Requires:  pip install -r requirements-data.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

DEFAULT_DIR = "/run/media/plz/CFN-ARCHIVE/LLMTrain"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(DEFAULT_DIR, "common_corpus_en.txt"),
                   help="output text file")
    p.add_argument("--gb", type=float, default=2.0,
                   help="gigabytes (GiB) of filtered text to write")
    p.add_argument("--language", default="English",
                   help="keep only this language ('' = keep everything)")
    p.add_argument("--collections", default="",
                   help="comma-separated open_type filter, e.g. 'Open Culture,Open Web'"
                        " ('' = all six collections)")
    p.add_argument("--shard-dir", default=os.path.join(DEFAULT_DIR, "parquet"),
                   help="where to download parquet shards")
    p.add_argument("--keep-parquet", action="store_true",
                   help="keep downloaded shards after extraction (allows re-filtering)")
    p.add_argument("--seed", type=int, default=0,
                   help="shard selection shuffle seed (0 = deterministic order)")
    p.add_argument("--append", action="store_true",
                   help="append to --out instead of overwriting")
    args = p.parse_args()

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("Missing deps:  pip install -r requirements-data.txt")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.shard_dir, exist_ok=True)
    target = int(args.gb * 1024**3)
    lang = args.language or None
    colls = {c.strip() for c in args.collections.split(",") if c.strip()} or None

    api = HfApi()
    shards = [f for f in api.list_repo_files("PleIAs/common_corpus", repo_type="dataset")
              if f.endswith(".parquet")]
    shards.sort()
    if args.seed:
        random.Random(args.seed).shuffle(shards)
    print(f"{len(shards)} shards available | target {args.gb} GiB | "
          f"language={lang or 'ALL'} | collections={colls or 'ALL'}")

    written = 0
    docs = 0
    t0 = time.time()
    mode = "ab" if args.append else "wb"
    out = open(args.out, mode)
    try:
        for shard in shards:
            if written >= target:
                break
            print(f"downloading {shard} ...")
            local = hf_hub_download(
                "PleIAs/common_corpus", shard, repo_type="dataset",
                local_dir=args.shard_dir,
            )
            pf = pq.ParquetFile(local)
            cols = ["text", "language", "open_type"]
            for batch in pf.iter_batches(batch_size=64, columns=cols):
                d = batch.to_pydict()
                for text, language, open_type in zip(
                    d["text"], d["language"], d["open_type"]
                ):
                    if not text or len(text) < 200:
                        continue
                    if lang and language != lang:
                        continue
                    if colls and open_type not in colls:
                        continue
                    b = text.encode("utf-8") + b"\n\n"
                    out.write(b)
                    written += len(b)
                    docs += 1
                if written >= target:
                    break
            if not args.keep_parquet:
                os.remove(local)
            rate = written / 1024**2 / max(time.time() - t0, 1e-6)
            print(f"  {written / 1024**3:.2f}/{args.gb} GiB after {shard} "
                  f"({docs} docs, {rate:.0f} MiB/s)")
    finally:
        out.close()

    el = time.time() - t0
    print(f"Done: {written / 1024**3:.2f} GiB, {docs} documents in {el:.0f}s")
    print(f"Train with:  python mariana.py train --data {args.out} ...")


if __name__ == "__main__":
    main()
