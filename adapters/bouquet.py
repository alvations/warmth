#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOUQuET adapter — Meta's multi-way, multi-centric evaluation benchmark of
hand-crafted sentences translated across many languages, drawn from non-English
sources (a multi-parallel test set for MT evaluation).

Source: HuggingFace ``facebook/bouquet`` (the code lives at
``github.com/facebookresearch/bouquet``; the data is on the Hub). Rows carry a
sentence ``id``/index, a language field and the ``text``, plus provenance/domain
metadata. Like FLORES this is references-only, so ``hypothesis``/``system`` are
``None`` and we pivot on a chosen source language (default English).

Column names on the Hub have shifted across revisions; ``COLS`` lists the
aliases we accept for id / language / text / domain so the adapter keeps working.
"""

import json

from warmth_schema import record, norm_lang
from adapters._hf import load_hf, config_names

COLLECTION = "bouquet"
HF_PATH = "facebook/bouquet"
DEFAULT_PIVOT = "eng"

COLS = {
    "id": ("id", "sentence_id", "index", "idx"),
    "lang": ("lang", "language", "iso", "iso_639_3", "lang_code"),
    "text": ("text", "sentence", "translation"),
    "domain": ("domain", "source", "topic", "register"),
}


def _pick(row, key):
    for alias in COLS[key]:
        if alias in row and row[alias] is not None:
            return row[alias]
    return None


def _load_split(cfg, split):
    rows = {}
    for i, r in enumerate(load_hf(HF_PATH, cfg, split=split)):
        sid = _pick(r, "id")
        rows[sid if sid is not None else i] = r
    return rows


def iter_records(root=None, pivot=DEFAULT_PIVOT, splits=("test",), configs=None):
    langs = configs or config_names(HF_PATH)
    for split in splits:
        try:
            pivot_rows = _load_split(pivot, split)
        except Exception:  # noqa: BLE001 - pivot config name may differ per revision
            pivot_rows = {}
        for cfg in langs:
            if cfg == pivot:
                continue
            for sid, tr in _load_split(cfg, split).items():
                sr = pivot_rows.get(sid)
                tgt_lang = norm_lang(_pick(tr, "lang") or cfg)
                dom = _pick(tr, "domain")
                yield record(
                    collection=COLLECTION,
                    release="BOUQuET",
                    year=None,
                    testset="bouquet-%s" % split,
                    domain=str(dom) if dom is not None else None,
                    langpair="%s-%s" % (pivot, tgt_lang),
                    src_lang=pivot,
                    tgt_lang=tgt_lang,
                    system=None,
                    segment_id=int(sid) + 1 if isinstance(sid, int) else 1,
                    doc_id=None,
                    source=_pick(sr, "text") if sr else None,
                    reference=_pick(tr, "text"),
                    hypothesis=None,
                    human_score=None,
                    human_score_level=None,
                    annotations=None,
                )
