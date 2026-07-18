#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pivot the ``scores/*.jsonl`` checkpoints written by ``score_population.py`` into
score **columns** and join them onto every row by key, producing a companion
table ``data_scores/<collection>/<shard>.parquet`` aligned 1:1 with
``data/<collection>/<shard>.parquet`` (same row order) and keyed by ``row_hash``.

Each metric becomes one float column:
  bleu, chrf, ter, comet, metricx, bertscore, sentbert, cometkiwi_hyp,
  metricxqe_hyp, cometkiwi_ref, metricxqe_ref, difficulty_src, sentinel_src

Row-space metrics join by ``row_hash``; ``*_ref`` join by hash(source|reference);
``*_src`` join by hash(source) — so a reference/source scored once is reused for
every system sharing it. Missing scores are ``null``.

    python merge_scores.py                       # build data_scores/ companion
    python merge_scores.py --inplace             # instead, add columns into data/

Load joined::

    import pyarrow.parquet as pq
    d = pq.read_table("data/wmt-metrics/wmt14.parquet")
    s = pq.read_table("data_scores/wmt-metrics/wmt14.parquet")   # same order + row_hash
    # d and s are row-aligned; or join any two tables on `row_hash`.
"""

import argparse
import glob
import io
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from score_population import METRICS, key_for, SCORES_DIR

ALL_METRIC_COLS = list(METRICS.keys())


def load_scores():
    """{space: {key: {metric: score}}} from scores/*.jsonl."""
    store = {}
    for sp in ("row", "refqe", "src"):
        path = os.path.join(SCORES_DIR, sp + ".jsonl")
        d = {}
        if os.path.isfile(path):
            with io.open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    d.setdefault(r["k"], {})[r["m"]] = r["s"]
        store[sp] = d
    return store


def score_columns_for(row, store):
    out = {}
    for name, spec in METRICS.items():
        sp = spec["space"]
        try:
            k = key_for(sp, row)
        except Exception:  # noqa: BLE001
            continue
        v = store[sp].get(k, {}).get(name)
        out[name] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data_scores")
    ap.add_argument("--inplace", action="store_true",
                    help="add score columns into data/*.parquet instead of a companion dir")
    args = ap.parse_args()

    store = load_scores()
    present = [m for m in ALL_METRIC_COLS
               if any(m in v for sp in store for v in store[sp].values())]
    if not present:
        print("no scores found in %s/ — run score_population.py first" % SCORES_DIR)
        return
    print("score columns present:", present)

    n_files = 0
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*", "*.parquet"))):
        tbl = pq.read_table(path)
        rows = tbl.to_pylist()
        cols = {m: [] for m in present}
        any_scored = False
        for row in rows:
            sc = score_columns_for(row, store)
            for m in present:
                v = sc.get(m)
                cols[m].append(v)
                if v is not None:
                    any_scored = True
        if not any_scored:
            continue  # don't waste space writing an all-null companion shard
        if args.inplace:
            new = tbl
            for m in present:
                new = new.append_column(m, pa.array(cols[m], type=pa.float32()))
            pq.write_table(new, path, compression="zstd")
            outp = path
        else:
            outp = path.replace(args.data_dir, args.out_dir, 1)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            data = {"row_hash": tbl.column("row_hash").to_pylist()}
            data.update({m: cols[m] for m in present})
            schema = pa.schema([("row_hash", pa.string())]
                               + [(m, pa.float32()) for m in present])
            pq.write_table(pa.table(data, schema=schema), outp, compression="zstd")
        n_files += 1
        print("  %s (%d rows)" % (outp, tbl.num_rows))
    print("merged scores into %d files (%s)" % (n_files, "inplace" if args.inplace else args.out_dir))


if __name__ == "__main__":
    main()
