#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_benchmark.py — load the combined MT shared-task benchmark from the parquet
files in `data/`. No `datasets` library required (pure pyarrow), but a
`datasets` helper is also provided.

Examples:
    # one split
    python load_benchmark.py --split wmt14 --head 3

    # everything as one table
    python load_benchmark.py --split all --head 3
"""
import argparse
import glob
import os

DATA = os.path.join(os.path.dirname(__file__), "data")


def load_split_pyarrow(split: str):
    """Return a pyarrow.Table for one split, or the concatenation of every
    per-split file when split == 'all'."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    if split == "all":
        files = sorted(f for f in glob.glob(os.path.join(DATA, "*.parquet"))
                       if os.path.basename(f) != "all.parquet")
        return pa.concat_tables([pq.read_table(f) for f in files])
    path = os.path.join(DATA, f"{split}.parquet")
    return pq.read_table(path)


def load_as_hf_datasetdict():
    """Return a datasets.DatasetDict with one split per parquet file."""
    from datasets import Dataset, DatasetDict
    dd = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*.parquet"))):
        name = os.path.splitext(os.path.basename(f))[0]
        dd[name] = Dataset.from_parquet(f)
    return DatasetDict(dd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", help="split name, e.g. wmt14, flores200, or 'all'")
    ap.add_argument("--head", type=int, default=3)
    args = ap.parse_args()
    tbl = load_split_pyarrow(args.split)
    print(f"{args.split}: {tbl.num_rows:,} rows, {tbl.num_columns} columns")
    print("columns:", tbl.column_names)
    rows = tbl.slice(0, args.head).to_pylist()
    for r in rows:
        print("-" * 60)
        print("pair    :", r["pair"], "| dataset:", r["dataset"], "| domain:", r["domain"])
        print("source  :", r["source"][:120])
        print("reference:", r["reference"][:120])


if __name__ == "__main__":
    main()
