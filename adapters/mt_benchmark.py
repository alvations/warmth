#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge adapter for the ``mt-eval-benchmark`` branch of this repo.

That branch froze a 33-split, ~870K-row union of MT shared-task *test sets*
(source + reference only, 15 columns) pulled via sacrebleu / HuggingFace /
GitHub. Several of those splits are ones WARMTH's main branch could **not** reach
directly from its firewalled build env — most importantly the **WMT15-20** news
test sets (sacrebleu), plus IWSLT17, Multi30k, mTEDx, MTNT, Tatoeba, DiaBLa,
the WMT'23 terminology set and the wmt-mqm evaluation segments.

This adapter pulls those split parquet files (Git-LFS, via the
``media.githubusercontent.com`` endpoint that works here) and maps the branch's
15-column schema onto WARMTH's superset schema. The branch splits are
references-only, so ``system`` / ``hypothesis`` / ``human_score`` are ``None``;
all provenance (dataset_full_name, paper, url, citation, loader, challenge_type)
is preserved in ``annotations`` so nothing is lost.

Splits where the main branch already has a *richer* version (WMT08-14 & 21-25
with system outputs / human eval, wmt24pp's full 55 pairs, FLORES-200's 203
languages, NTREX-128) are intentionally **not** re-imported here.
"""

import io
import json
import os
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

from warmth_schema import record, ARROW_SCHEMA, FIELDS

BRANCH = "mt-eval-benchmark"
MEDIA_BASE = ("https://media.githubusercontent.com/media/alvations/warmth/%s/"
              "mt_benchmark/data" % BRANCH)

# split -> (collection, release). Only splits not already richer on main.
SPLITS = {
    "wmt15": ("wmt-general", "WMT15"),
    "wmt16": ("wmt-general", "WMT16"),
    "wmt17": ("wmt-general", "WMT17"),
    "wmt18": ("wmt-general", "WMT18"),
    "wmt19": ("wmt-general", "WMT19"),
    "wmt20": ("wmt-general", "WMT20"),
    "wmt_mqm": ("wmt-mqm", "wmt-mqm"),
    "wmt_terminology_2023": ("wmt-terminology", "wmt-terminology-2023"),
    "iwslt17": ("iwslt", "IWSLT17"),
    "multi30k_2016": ("multi30k", "Multi30k-2016"),
    "multi30k_2017": ("multi30k", "Multi30k-2017"),
    "multi30k_2018": ("multi30k", "Multi30k-2018"),
    "mtedx_test": ("mtedx", "mTEDx"),
    "mtnt1_1_test": ("mtnt", "MTNT-1.1"),
    "mtnt2019": ("mtnt", "MTNT-2019"),
    "tatoeba": ("tatoeba", "Tatoeba-MT"),
    "diabla": ("diabla", "DiaBLa"),
}

_PROVENANCE = ("dataset_full_name", "paper", "dataset_url", "citation",
               "loader", "challenge_type", "dataset")


def fetch_split(split, dest_dir):
    """Download one branch split parquet (Git-LFS) into ``dest_dir``."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, split + ".parquet")
    if not os.path.isfile(path):
        url = "%s/%s.parquet" % (MEDIA_BASE, split)
        ctx = None
        req = urllib.request.Request(url, headers={"User-Agent": "warmth"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
    return path


def _map_row(row, collection, release):
    ann = {k: row[k] for k in _PROVENANCE if row.get(k) not in (None, "")}
    yr = row.get("year")
    try:
        year = int(yr) if yr not in (None, "") else None
    except (ValueError, TypeError):
        year = None
    return record(
        collection=collection,
        release=release,
        year=year,
        testset=row.get("dataset"),
        domain=row.get("domain") or None,
        langpair=row.get("pair"),
        src_lang=row.get("source_lang"),
        tgt_lang=row.get("target_lang"),
        system=None,
        segment_id=int(row.get("idx", 0)) + 1,
        doc_id=None,
        source=row.get("source"),
        reference=row.get("reference"),
        hypothesis=None,
        human_score=None,
        human_score_level=None,
        annotations=json.dumps(ann, ensure_ascii=False) if ann else None,
    )


def iter_split(path, collection, release):
    tbl = pq.read_table(path)
    for row in tbl.to_pylist():
        yield _map_row(row, collection, release)


def iter_records(root=None, splits=None):
    """Fetch (if needed) and yield records for the selected branch splits.

    ``root`` is a directory of already-downloaded branch parquet (default:
    ``data/_sources/mt_benchmark``); missing splits are fetched there.
    """
    root = root or os.path.join("data", "_sources", "mt_benchmark")
    for split in (splits or SPLITS):
        collection, release = SPLITS[split]
        path = os.path.join(root, split + ".parquet")
        if not os.path.isfile(path):
            fetch_split(split, root)
        yield from iter_split(path, collection, release)
