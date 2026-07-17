#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT24++ adapter — post-edited references for WMT24 (Deutsch et al., 2025),
English into 55 languages/variants, with the original MT output and human
post-edit for every segment.

Source: HuggingFace ``google/wmt24pp`` (Apache-2.0). One config per language
pair, e.g. ``en-de_DE``. Columns (per the dataset card): ``lp``, ``domain``,
``document_id``, ``segment_id``, ``is_bad_source``, ``source``, ``target``
(the human post-edit = reference), ``original_target`` (the raw MT hypothesis).

Mapping to the WARMTH schema: ``source`` -> source, ``target`` -> reference,
``original_target`` -> hypothesis (system ``"wmt24pp-postedited-MT"``);
``is_bad_source`` and ``domain`` are preserved.
"""

import json

from warmth_schema import record
from adapters._hf import load_hf, config_names

COLLECTION = "wmt24pp"
HF_PATH = "google/wmt24pp"


def _lp(cfg):
    # config names look like "en-de_DE"; keep as-is for langpair.
    src, _, tgt = cfg.partition("-")
    return cfg, src, tgt


def iter_records(root=None, configs=None):
    cfgs = configs or config_names(HF_PATH)
    for cfg in cfgs:
        langpair, src_lang, tgt_lang = _lp(cfg)
        ds = load_hf(HF_PATH, cfg, split="train")
        for i, row in enumerate(ds):
            ann = {}
            if row.get("is_bad_source") is not None:
                ann["is_bad_source"] = bool(row["is_bad_source"])
            yield record(
                collection=COLLECTION,
                release="wmt24pp",
                year=2024,
                testset="wmt24",
                domain=row.get("domain"),
                langpair=langpair,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                system="wmt24pp-postedited-MT",
                segment_id=int(row.get("segment_id", i + 1) or (i + 1)),
                doc_id=row.get("document_id"),
                source=row.get("source"),
                reference=row.get("target"),          # human post-edit
                hypothesis=row.get("original_target"),  # raw MT output
                human_score=None,
                human_score_level=None,
                annotations=json.dumps(ann) if ann else None,
            )
