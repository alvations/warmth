#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLORES+ adapter — the community-maintained continuation of FLORES-200, a
multi-parallel evaluation benchmark (dev / devtest) covering 200+ languages.

Source: HuggingFace ``openlanguagedata/flores_plus`` (CC BY-SA 4.0). One config
per language (``<iso_639_3>_<iso_15924>``, e.g. ``eng_Latn``); rows carry
``id``, ``text``, ``url``, ``domain``, ``topic``, ``has_hyperlink`` etc.

FLORES is references-only (no MT outputs / human scores). To produce
directed rows we pair a chosen pivot source language (default ``eng_Latn``) with
every other language: ``source`` = pivot text, ``reference`` = target text, at
matching ``id``. ``hypothesis``/``system``/``human_score`` are ``None``.
"""

import json

from warmth_schema import record
from adapters._hf import load_hf, config_names

COLLECTION = "flores-plus"
HF_PATH = "openlanguagedata/flores_plus"
DEFAULT_PIVOT = "eng_Latn"


def _load_split(cfg, split):
    rows = {}
    for r in load_hf(HF_PATH, cfg, split=split):
        rows[r["id"]] = r
    return rows


def iter_records(root=None, pivot=DEFAULT_PIVOT, splits=("dev", "devtest"), configs=None):
    langs = configs or config_names(HF_PATH)
    for split in splits:
        pivot_rows = _load_split(pivot, split)
        for cfg in langs:
            if cfg == pivot:
                continue
            tgt_rows = _load_split(cfg, split)
            for sid, tr in tgt_rows.items():
                sr = pivot_rows.get(sid)
                ann = {k: tr.get(k) for k in ("domain", "topic", "url") if tr.get(k) is not None}
                yield record(
                    collection=COLLECTION,
                    release="FLORES+",
                    year=None,
                    testset="flores-%s" % split,
                    domain=tr.get("domain"),
                    langpair="%s-%s" % (pivot, cfg),
                    src_lang=pivot,
                    tgt_lang=cfg,
                    system=None,
                    segment_id=int(sid),
                    doc_id=None,
                    source=sr.get("text") if sr else None,
                    reference=tr.get("text"),
                    hypothesis=None,
                    human_score=None,
                    human_score_level=None,
                    annotations=json.dumps(ann) if ann else None,
                )
