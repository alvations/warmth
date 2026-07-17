#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish WARMTH to the HuggingFace Hub. Prepared for running *later*, from an
environment where ``huggingface.co`` is reachable and you are authenticated
(``huggingface-cli login`` or ``HF_TOKEN``).

Two modes:

1. Upload the repo as-is (the parquet under ``data/`` + this ``README.md`` with
   its ``configs:`` block ARE a loadable dataset — no conversion needed)::

       python push_to_hub.py --repo-id alvations/warmth

   This uploads ``data/**`` and ``README.md`` via ``huggingface_hub`` so
   ``load_dataset("alvations/warmth")`` and every per-collection / per-release
   config work immediately.

2. Round-trip through the ``datasets`` library (validates the schema, rewrites
   canonical shards) before pushing::

       python push_to_hub.py --repo-id alvations/warmth --via-datasets

Neither mode fetches source data; build/enrich the parquet first (see build.py,
enrich.py). Use ``--dry-run`` to print what would be uploaded.
"""

import argparse
import glob
import os


def _upload_files(repo_id, out_root, dry_run, private):
    from huggingface_hub import HfApi
    api = HfApi()
    files = sorted(glob.glob(os.path.join(out_root, "**", "*.parquet"), recursive=True))
    if os.path.isfile("README.md"):
        files.append("README.md")
    print("Would upload %d files to %s:" % (len(files), repo_id))
    for f in files:
        print("  ", f)
    if dry_run:
        return
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for f in files:
        path_in_repo = f if f == "README.md" else f  # keep data/<collection>/<shard>.parquet
        api.upload_file(path_or_fileobj=f, path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="dataset")
    print("Uploaded. Try: load_dataset(%r)" % repo_id)


def _push_via_datasets(repo_id, out_root, dry_run, private):
    from datasets import load_dataset
    from adapters import REGISTRY
    for key in REGISTRY:
        shards = glob.glob(os.path.join(out_root, key, "*.parquet"))
        if not shards:
            continue
        print("Loading config %s (%d shards)" % (key, len(shards)))
        if dry_run:
            continue
        ds = load_dataset(os.path.join(out_root, key), split="train")
        ds.push_to_hub(repo_id, config_name=key, private=private)
    print("Pushed per-collection configs to %s" % repo_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="e.g. alvations/warmth")
    ap.add_argument("--out", default="data")
    ap.add_argument("--via-datasets", action="store_true",
                    help="round-trip through datasets.push_to_hub per collection")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.via_datasets:
        _push_via_datasets(args.repo_id, args.out, args.dry_run, args.private)
    else:
        _upload_files(args.repo_id, args.out, args.dry_run, args.private)


if __name__ == "__main__":
    main()
