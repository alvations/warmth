#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot consistency pass over the built parquet: canonicalise the
``src_lang`` / ``tgt_lang`` columns to a single convention across every shared
task (via :func:`warmth_schema.norm_lang`) while leaving ``langpair`` — which
holds the raw direction — untouched, so no information is lost. Idempotent.

    python normalize_data.py            # rewrite every data/*/*.parquet in place
"""

import glob
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from warmth_schema import ARROW_SCHEMA, norm_lang


def normalize_file(path):
    t = pq.read_table(path)
    cols = t.to_pydict()
    cache = {}

    def nm(c):
        if c not in cache:
            cache[c] = norm_lang(c)
        return cache[c]

    changed = 0
    for field in ("src_lang", "tgt_lang"):
        new = [nm(c) for c in cols[field]]
        changed += sum(1 for a, b in zip(cols[field], new) if a != b)
        cols[field] = new
    pq.write_table(pa.table(cols, schema=ARROW_SCHEMA), path, compression="zstd")
    return t.num_rows, changed


def main(argv):
    targets = argv or sorted(glob.glob("data/*/*.parquet"))
    total = 0
    for path in targets:
        n, changed = normalize_file(path)
        if changed:
            print("  %-45s %d rows, %d codes normalised" % (path, n, changed))
        total += n
    print("normalized %d files, %d rows" % (len(targets), total))


if __name__ == "__main__":
    main(sys.argv[1:])
