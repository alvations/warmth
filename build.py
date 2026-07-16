#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialise the consolidated WMT Metrics-task data into Parquet shards that
``datasets.load_dataset`` can read directly (no loading script required, which
matters for ``datasets`` >= 4/5 where remote-code loading was removed).

Usage
-----
    python build.py                       # write data/wmt2008.parquet ... wmt2014.parquet
    python build.py --out data --years 2013 2014
    python build.py --push-to-hub alvations/warmth   # requires HF auth + network

After building, the dataset loads with::

    from datasets import load_dataset
    ds = load_dataset("data")                 # local parquet dir
    ds = load_dataset("alvations/warmth")     # once pushed to the Hub
    ds = load_dataset("alvations/warmth", "wmt14")   # a single edition

The Parquet layout plus the ``configs:`` block written into ``README.md`` give
every edition its own selectable config, with a ``default`` config spanning all
of them.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq

from warmth_core import iter_records, available_years, Record, DEFAULT_DATA_ROOT

# Explicit Arrow schema so nullable string / float columns stay nullable even
# when a whole shard happens to have no nulls.
SCHEMA = pa.schema([
    ("year", pa.int32()),
    ("wmt", pa.string()),
    ("testset", pa.string()),
    ("langpair", pa.string()),
    ("src_lang", pa.string()),
    ("tgt_lang", pa.string()),
    ("system", pa.string()),
    ("segment_id", pa.int32()),
    ("doc_id", pa.string()),
    ("source", pa.string()),
    ("reference", pa.string()),
    ("hypothesis", pa.string()),
    ("human_score", pa.float32()),
    ("human_score_level", pa.string()),
])

BATCH = 50_000


def _flush(writer, rows):
    cols = {name: [getattr(r, name) for r in rows] for name in Record._fields}
    writer.write_table(pa.table(cols, schema=SCHEMA))


def build_year(year, out_dir, data_root=DEFAULT_DATA_ROOT):
    """Stream one edition to ``<out_dir>/wmt<year>.parquet``; return row count."""
    path = os.path.join(out_dir, "wmt%d.parquet" % year)
    writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
    n, rows = 0, []
    try:
        for r in iter_records(data_root, years=[year]):
            rows.append(r)
            if len(rows) >= BATCH:
                _flush(writer, rows)
                n += len(rows)
                rows = []
        if rows:
            _flush(writer, rows)
            n += len(rows)
    finally:
        writer.close()
    if n == 0:  # nothing on disk for this edition; don't leave an empty shard
        os.remove(path)
    return n


def build(out_dir, years=None, data_root=DEFAULT_DATA_ROOT):
    os.makedirs(out_dir, exist_ok=True)
    if years is None:
        years = available_years(data_root)
    counts = {}
    for year in years:
        counts[year] = build_year(year, out_dir, data_root)
        print("  wmt%d.parquet: %d rows" % (year, counts[year]))
    return counts


def _config_block(years):
    lines = ["configs:"]
    lines.append("- config_name: default")
    lines.append("  data_files:")
    lines.append("  - split: train")
    lines.append("    path: data/wmt*.parquet")
    for year in years:
        lines.append("- config_name: wmt%s" % str(year)[-2:])
        lines.append("  data_files:")
        lines.append("  - split: train")
        lines.append("    path: data/wmt%d.parquet" % year)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory for parquet shards")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--push-to-hub", metavar="REPO_ID", default=None,
                    help="push the built dataset to this HF Hub repo (needs auth+network)")
    ap.add_argument("--print-config", action="store_true",
                    help="print the README `configs:` YAML block for the built years")
    args = ap.parse_args()

    print("Building parquet shards into %s/ ..." % args.out)
    counts = build(args.out, args.years, args.data_root)
    print("Total rows: %d" % sum(counts.values()))

    if args.print_config:
        print("\n--- README configs block ---")
        print(_config_block([y for y in counts if counts[y]]))

    if args.push_to_hub:
        from datasets import load_dataset
        print("Loading built parquet for push ...")
        ds = load_dataset(args.out)
        print("Pushing to %s ..." % args.push_to_hub)
        ds.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
