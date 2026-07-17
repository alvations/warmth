#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bio-MQM adapter — segment-level MQM error-span annotations of biomedical MT
(Amazon Science, ``amazon-science/bio-mqm-dataset``).

Layout: ``data/<version>/target/main_phase/<round>/<lp>/<system>.json``, each a
JSON array of segments with ``source``, ``target`` (the annotated MT output),
``SEG_ID``, ``DOC_ID``, ``MT_Engine``, ``source_locale``/``target_locale`` and
``source_errors`` / ``target_errors`` (the MQM span annotations). Language pair
comes from the directory name (e.g. ``fr2en`` -> ``fr-en``); the ``reference*``
files are the human references (kept, with ``system`` = the file stem).

MQM spans are preserved verbatim in ``annotations``; ``human_score_level`` is set
to ``segment:mqm``.
"""

import glob
import json
import os
import re

from warmth_schema import record, norm_lang

COLLECTION = "bio-mqm"
REPO_URL = "https://github.com/amazon-science/bio-mqm-dataset"


def _lp_from_dir(name):
    m = re.match(r"([a-z]{2,3})2([a-z]{2,3})$", name)
    return (m.group(1), m.group(2)) if m else (None, None)


def _loc(code):
    # "frfr"/"enus" -> "fr"/"en"
    return norm_lang(code[:2]) if code else None


def iter_records(root=None, version=None):
    versions = [version] if version else ["v1", "v2"]
    for ver in versions:
        base = os.path.join(root, "data", ver, "target")
        if not os.path.isdir(base):
            continue
        for path in sorted(glob.glob(os.path.join(base, "**", "*.json"), recursive=True)):
            lp_dir = os.path.basename(os.path.dirname(path))
            src_lang, tgt_lang = _lp_from_dir(lp_dir)
            if not src_lang:
                continue
            system = os.path.splitext(os.path.basename(path))[0]
            try:
                data = json.load(open(path, encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if not isinstance(data, list):
                continue
            for i, seg in enumerate(data):
                if not isinstance(seg, dict):
                    continue
                ann = {}
                for k in ("target_errors", "source_errors", "Annotator_ID", "MT_Engine"):
                    if seg.get(k) not in (None, "", "[]"):
                        ann[k] = seg[k]
                has_mqm = "target_errors" in ann or "source_errors" in ann
                try:
                    seg_id = int(seg.get("SEG_ID", i + 1))
                except (ValueError, TypeError):
                    seg_id = i + 1
                is_ref = system.lower().startswith("reference")
                yield record(
                    collection=COLLECTION, release="bio-mqm-%s" % ver, year=None,
                    testset="bio-mqm", domain="biomedical",
                    langpair="%s-%s" % (src_lang, tgt_lang),
                    src_lang=_loc(seg.get("source_locale")) or src_lang,
                    tgt_lang=_loc(seg.get("target_locale")) or tgt_lang,
                    system=None if is_ref else system,
                    segment_id=seg_id, doc_id=seg.get("DOC_ID"),
                    source=seg.get("source"),
                    reference=seg.get("target") if is_ref else None,
                    hypothesis=None if is_ref else seg.get("target"),
                    human_score=None,
                    human_score_level="segment:mqm" if has_mqm else None,
                    annotations=json.dumps(ann, ensure_ascii=False) if ann else None)
