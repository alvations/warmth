#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The single, uniform record schema shared by every WARMTH source adapter.

WARMTH gathers heterogeneous machine-translation evaluation resources — WMT
metrics/general-MT task data, WMT24++ post-edits, NTREX-128, FLORES+, Meta's
BOUQuET, the WMT terminology task — into one table so they can live in one
HuggingFace dataset. They differ a lot (some have MT hypotheses and human
scores, some are pure multi-parallel test sets), so the schema is a *superset*:
fields that don't apply to a given source are simply ``None``.

Every adapter yields :class:`Record` objects (build them with :func:`record`,
which defaults every unspecified field to ``None``). :data:`ARROW_SCHEMA` is the
matching pyarrow schema used to write the parquet shards, with all optional
columns kept nullable.
"""

from collections import namedtuple

import pyarrow as pa

FIELDS = [
    "collection",        # source family: "wmt-metrics", "wmt24pp", "ntrex",
                         #   "flores-plus", "bouquet", "wmt-terminology"
    "release",           # specific release: "WMT13", "NTREX-128", "FLORES+",
                         #   "wmt24pp", "wmt23-terminology"
    "year",              # int | None
    "testset",           # e.g. "newstest2013", "flores-devtest" | None
    "domain",            # e.g. "news", "speech", "social" | None
    "langpair",          # "de-en" (as distributed)
    "src_lang",          # normalised source language code
    "tgt_lang",          # normalised target language code
    "system",            # MT system id | None (None for pure test sets)
    "segment_id",        # 1-indexed segment / line number
    "doc_id",            # document id | None
    "source",            # source segment | None
    "reference",         # reference translation | None
    "hypothesis",        # MT output | None (None for pure test sets)
    "human_score",       # float | None
    "human_score_level", # "system" | "segment" | None
    "annotations",       # JSON string for anything structured (MQM spans,
                         #   post-edits, term matches, quality labels) | None
]

Record = namedtuple("Record", FIELDS)


def record(**kwargs):
    """Create a :class:`Record`, defaulting any unspecified field to ``None``.

    ``langpair`` is derived from ``src_lang``/``tgt_lang`` when omitted (and
    vice-versa) so adapters can pass whichever is convenient.
    """
    data = {f: kwargs.get(f) for f in FIELDS}
    if data["langpair"] is None and data["src_lang"] and data["tgt_lang"]:
        data["langpair"] = "%s-%s" % (data["src_lang"], data["tgt_lang"])
    if (data["src_lang"] is None or data["tgt_lang"] is None) and data["langpair"] and "-" in data["langpair"]:
        s, t = data["langpair"].split("-", 1)
        data["src_lang"] = data["src_lang"] or s
        data["tgt_lang"] = data["tgt_lang"] or t
    unknown = set(kwargs) - set(FIELDS)
    if unknown:
        raise TypeError("unknown record field(s): %s" % ", ".join(sorted(unknown)))
    return Record(**data)


ARROW_SCHEMA = pa.schema([
    ("collection", pa.string()),
    ("release", pa.string()),
    ("year", pa.int32()),
    ("testset", pa.string()),
    ("domain", pa.string()),
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
    ("annotations", pa.string()),
])

# Common language-code normalisation across sources (WMT Czech drift, etc.).
LANG_NORMALISE = {"cz": "cs"}


def norm_lang(code):
    return LANG_NORMALISE.get(code, code) if code else code
