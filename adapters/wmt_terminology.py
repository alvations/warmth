#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT Terminology shared-task adapter (WMT21 / WMT23 / WMT25 terminology tracks).

The terminology task ships, per language pair, segments that carry a source, a
reference, and a **terminology constraint** — a set of (source term -> target
term) pairs the translation is expected to honour — plus, for the evaluation,
participating system hypotheses. The official data is distributed per edition
(statmt.org / the task organisers' repos) rather than from one canonical place,
so this adapter is format-driven: point it at a directory of the task's JSONL
files (or set ``--repo-url`` for :func:`fetch` to clone one).

Accepted JSONL fields (aliases tried in order)::

    source        : source, src, source_text
    reference     : reference, target, ref, tgt, post_edit
    hypothesis    : hypothesis, translation, output, mt        (optional)
    system        : system, system_name, sysid                 (optional)
    terms         : terms, terminology, term_dict, constraints  -> annotations
    doc_id        : doc_id, document_id, docid                  (optional)
    langpair      : lp, langpair, language_pair                 (optional; else
                    inferred from the file name, e.g. en-de.jsonl)

The terminology constraints are preserved verbatim under
``annotations = {"terms": ...}``.
"""

import glob
import io
import json
import os
import re
import subprocess

from warmth_schema import record

COLLECTION = "wmt-terminology"
DEFAULT_REPO_URL = None  # set via --repo-url; no single canonical upstream

_ALIASES = {
    "source": ("source", "src", "source_text", "source_segment"),
    "reference": ("reference", "target", "ref", "tgt", "post_edit", "reference_translation"),
    "hypothesis": ("hypothesis", "translation", "output", "mt", "candidate"),
    "system": ("system", "system_name", "sysid", "submission"),
    "terms": ("terms", "terminology", "term_dict", "constraints", "term_pairs"),
    "doc_id": ("doc_id", "document_id", "docid"),
    "langpair": ("lp", "langpair", "language_pair", "direction"),
    "segment_id": ("segment_id", "segid", "id", "line_id"),
    "year": ("year", "edition"),
}

_LP_RE = re.compile(r"([a-z]{2,3}(?:[-_][A-Za-z]{2,4})?)-([a-z]{2,3}(?:[-_][A-Za-z]{2,4})?)")


def _pick(row, key):
    for a in _ALIASES[key]:
        if a in row and row[a] is not None:
            return row[a]
    return None


def fetch(dest, repo_url=DEFAULT_REPO_URL):
    if not repo_url:
        raise RuntimeError("set --repo-url to the terminology task data repo to fetch")
    if not os.path.isdir(dest):
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, dest])
    return dest


def _langpair_from_name(path):
    m = _LP_RE.search(os.path.basename(path))
    return m.group(0) if m else None


def iter_records(root, year=None):
    for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        file_lp = _langpair_from_name(path)
        with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                lp = _pick(row, "langpair") or file_lp
                if not lp or "-" not in lp:
                    continue
                src_lang, _, tgt_lang = lp.partition("-")
                terms = _pick(row, "terms")
                yr = _pick(row, "year") or year
                yield record(
                    collection=COLLECTION,
                    release="wmt-terminology" + ("-%s" % yr if yr else ""),
                    year=int(yr) if yr else None,
                    testset="wmt-terminology",
                    domain=None,
                    langpair=lp,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    system=_pick(row, "system"),
                    segment_id=int(_pick(row, "segment_id") or (i + 1)),
                    doc_id=_pick(row, "doc_id"),
                    source=_pick(row, "source"),
                    reference=_pick(row, "reference"),
                    hypothesis=_pick(row, "hypothesis"),
                    human_score=None,
                    human_score_level=None,
                    annotations=json.dumps({"terms": terms}) if terms is not None else None,
                )
