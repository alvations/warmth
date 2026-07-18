#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialise WARMTH collections into Parquet shards under ``data/<collection>/``
that ``datasets.load_dataset`` reads directly (no loading script, which matters
for ``datasets`` >= 4/5).

Each adapter (see :mod:`adapters`) yields the shared superset schema; this script
streams its records into one parquet file per ``release`` (e.g. the WMT metrics
collection shards into ``wmt08.parquet`` … ``wmt14.parquet``; NTREX into
``ntrex-128.parquet``).

Usage::

    python build.py                       # build every locally-available collection
    python build.py --collections ntrex   # just one
    python build.py --collections ntrex --fetch          # git-clone NTREX first
    python build.py --collections wmt24pp --root /data/wmt24pp   # fetch-only source
    python build.py --list                # show the registry + availability
    python build.py --print-config        # emit the README `configs:` YAML

`fetch` collections (HF / GCS / statmt / task repos) are skipped unless you pass
``--fetch`` (for those whose adapter can self-fetch) or ``--root`` (a directory
you already downloaded). See adapters/ for what each needs.
"""

import argparse
import glob
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

from warmth_schema import ARROW_SCHEMA, FIELDS
from adapters import REGISTRY

BATCH = 50_000


def _shard_name(release):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (release or "all")).strip("-").lower()
    return slug or "all"


class _Writers:
    """Lazily-opened one-parquet-per-release writers for a collection."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.writers = {}
        self.buffers = {}
        self.counts = {}
        os.makedirs(out_dir, exist_ok=True)

    def add(self, rec):
        shard = _shard_name(rec.release)
        buf = self.buffers.setdefault(shard, [])
        buf.append(rec)
        if len(buf) >= BATCH:
            self._flush(shard)

    def _flush(self, shard):
        buf = self.buffers[shard]
        if not buf:
            return
        if shard not in self.writers:
            path = os.path.join(self.out_dir, shard + ".parquet")
            self.writers[shard] = pq.ParquetWriter(path, ARROW_SCHEMA, compression="zstd")
        cols = {name: [getattr(r, name) for r in buf] for name in FIELDS}
        self.writers[shard].write_table(pa.table(cols, schema=ARROW_SCHEMA))
        self.counts[shard] = self.counts.get(shard, 0) + len(buf)
        self.buffers[shard] = []

    def close(self):
        for shard in list(self.buffers):
            self._flush(shard)
        for w in self.writers.values():
            w.close()
        return dict(self.counts)


def build_collection(key, out_root="data", fetch=False, root=None):
    entry = REGISTRY[key]
    module = entry["module"]

    if root is None and fetch and hasattr(module, "fetch"):
        print("  fetching %s ..." % key)
        root = module.fetch(data_root=out_root) if "data_root" in module.fetch.__code__.co_varnames else module.fetch(os.path.join(out_root, "_sources", key))

    kwargs = {}
    if root is not None:
        kwargs["root"] = root
    out_dir = os.path.join(out_root, key)
    writers = _Writers(out_dir)
    n = 0
    for rec in module.iter_records(**kwargs):
        writers.add(rec)
        n += 1
    counts = writers.close()
    if n == 0:
        # leave nothing behind for an empty build
        for p in glob.glob(os.path.join(out_dir, "*.parquet")):
            os.remove(p)
    for shard, c in sorted(counts.items()):
        print("    %s/%s.parquet: %d rows" % (key, shard, c))
    return n


def _config_block(out_root="data"):
    lines = ["configs:", "- config_name: default", "  data_files:",
             "  - split: train", "    path: data/*/*.parquet"]
    for key in REGISTRY:
        shards = sorted(glob.glob(os.path.join(out_root, key, "*.parquet")))
        if not shards:
            continue
        lines += ["- config_name: %s" % key, "  data_files:",
                  "  - split: train", "    path: data/%s/*.parquet" % key]
        # also expose each release as its own config, but only when there are
        # few, clean shards (avoid dozens of per-langpair sub-configs)
        if 1 < len(shards) <= 12:
            for sp in shards:
                name = os.path.splitext(os.path.basename(sp))[0]
                if name == key:
                    continue  # avoid a per-shard config colliding with the collection config
                lines += ["- config_name: %s" % name, "  data_files:",
                          "  - split: train", "    path: data/%s/%s.parquet" % (key, name)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--collections", nargs="*", default=None,
                    help="registry keys to build (default: all 'local' ones)")
    ap.add_argument("--fetch", action="store_true",
                    help="allow adapters that can self-fetch to download first")
    ap.add_argument("--root", default=None,
                    help="pre-downloaded source dir (single-collection builds)")
    ap.add_argument("--list", action="store_true", help="list the registry and exit")
    ap.add_argument("--print-config", action="store_true",
                    help="print the README configs: YAML for what is built")
    args = ap.parse_args()

    if args.list:
        for key, e in REGISTRY.items():
            print("%-16s %-6s %s" % (key, e["availability"], e["note"]))
        return
    if args.print_config:
        print(_config_block(args.out))
        return

    keys = args.collections
    if keys is None:
        keys = [k for k, e in REGISTRY.items() if e["availability"] == "local"]

    total = 0
    for key in keys:
        print("Building %s ..." % key)
        try:
            total += build_collection(key, args.out, fetch=args.fetch, root=args.root)
        except Exception as exc:  # noqa: BLE001 - one blocked source shouldn't kill the rest
            print("  SKIPPED %s: %s" % (key, exc))
    print("Total rows built: %d" % total)


if __name__ == "__main__":
    main()
