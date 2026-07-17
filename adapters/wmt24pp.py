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

import glob
import io
import json
import os

from warmth_schema import record

COLLECTION = "wmt24pp"
HF_PATH = "google/wmt24pp"


def _lp(cfg):
    # config / filename stems look like "en-de_DE"; keep as-is for langpair.
    src, _, tgt = cfg.partition("-")
    return cfg, src, tgt


def _emit(row, langpair, src_lang, tgt_lang, i):
    ann = {}
    if row.get("is_bad_source") is not None:
        ann["is_bad_source"] = bool(row["is_bad_source"])
    return record(
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
        reference=row.get("target"),           # human post-edit (recommended ref)
        hypothesis=row.get("original_target"),  # original MT reference / output
        human_score=None,
        human_score_level=None,
        annotations=json.dumps(ann) if ann else None,
    )


def iter_records(root=None, configs=None):
    """Local dir of ``en-xx_XX.jsonl`` files (default), else the HuggingFace Hub."""
    if root and os.path.isdir(root):
        files = sorted(glob.glob(os.path.join(root, "*.jsonl")))
        for path in files:
            langpair, src_lang, tgt_lang = _lp(os.path.splitext(os.path.basename(path))[0])
            with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    yield _emit(json.loads(line), langpair, src_lang, tgt_lang, i)
        return
    # Hub fallback
    from adapters._hf import load_hf, config_names
    for cfg in (configs or config_names(HF_PATH)):
        langpair, src_lang, tgt_lang = _lp(cfg)
        for i, row in enumerate(load_hf(HF_PATH, cfg, split="train")):
            yield _emit(row, langpair, src_lang, tgt_lang, i)
