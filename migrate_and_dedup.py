#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add the ``row_hash`` content fingerprint to every built parquet, drop exact
duplicate rows globally (so combining sources never inflates the data), and
report empty source/reference counts per release for verification.

    python migrate_and_dedup.py          # rewrite data/*/*.parquet in place
    python migrate_and_dedup.py --check  # report only, write nothing
"""

import collections
import glob
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from warmth_schema import ARROW_SCHEMA, _HASH_FIELDS, compute_row_hash


def run(check=False):
    seen = set()
    kept = dropped = 0
    empty_src = collections.Counter()
    empty_ref = collections.Counter()
    per_release = collections.Counter()
    files = sorted(glob.glob("data/*/*.parquet"))
    for path in files:
        t = pq.read_table(path)
        cols = t.to_pydict()
        n = t.num_rows
        keep = []
        hashes = []
        for i in range(n):
            row = {f: cols[f][i] for f in _HASH_FIELDS}
            h = compute_row_hash(row)
            rel = "%s/%s" % (row.get("collection"), row.get("release"))
            if not row.get("source"):
                empty_src[rel] += 1
            if not row.get("reference"):
                empty_ref[rel] += 1
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            keep.append(i)
            hashes.append(h)
            per_release[rel] += 1
        kept += len(keep)
        if not check:
            newcols = {f: [cols[f][i] for i in keep] for f in _HASH_FIELDS}
            newcols["row_hash"] = hashes
            pq.write_table(pa.table(newcols, schema=ARROW_SCHEMA), path,
                           compression="zstd")
        d = n - len(keep)
        if d:
            print("  %-48s %d -> %d  (-%d dup)" % (path, n, len(keep), d))
    print("\nTOTAL kept=%d dropped_duplicates=%d" % (kept, dropped))
    print("\nEmpty SOURCE by release (WMT14 has no local source -> expected):")
    for r, c in sorted(empty_src.items()):
        print("  %-32s %d" % (r, c))
    print("\nEmpty REFERENCE by release:")
    for r, c in sorted(empty_ref.items()) or [("(none)", 0)]:
        print("  %-32s %d" % (r, c))
    if not empty_ref:
        print("  (none)")


if __name__ == "__main__":
    run(check="--check" in sys.argv)
