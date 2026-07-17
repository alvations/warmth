#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTREX-128 adapter — News Test References for MT Evaluation, English source
(WMT19 newstest) translated into 128 target languages, with document-level
information (Federmann, Kocmi & Xin, 2022).

Source: https://github.com/MicrosoftTranslator/NTREX (CC BY-SA 4.0).

Layout::

    NTREX-128/newstest2019-src.eng.txt   original English source (1997 lines)
    NTREX-128/newstest2019-ref.<code>.txt human reference per language
    NTREX-additional/newstest2019-ref.<code>.txt  extra languages
    DOCUMENT_IDS.tsv                      line-aligned document boundaries
    LANGUAGES.tsv                         <code>\t<language name>

NTREX is a multi-parallel *test set*: it has a source and references but no MT
system outputs and no human scores, so ``system``/``hypothesis``/``human_score``
are ``None``. Each ``(eng -> code)`` direction becomes one segment stream with
``doc_id`` filled from ``DOCUMENT_IDS.tsv``.
"""

import io
import json
import os
import re
import subprocess

from warmth_schema import record

COLLECTION = "ntrex"
REPO_URL = "https://github.com/MicrosoftTranslator/NTREX"

_REF_RE = re.compile(r"^newstest2019-ref(?:-(?P<variant>\d+))?\.(?P<code>.+)\.txt$")
_SRC_NAME = "newstest2019-src.eng.txt"


def default_root(data_root="data"):
    return os.path.join(data_root, "_sources", "NTREX")


def fetch(dest=None, data_root="data"):
    """git-clone NTREX into ``dest`` (default ``data/_sources/NTREX``)."""
    dest = dest or default_root(data_root)
    if os.path.isdir(os.path.join(dest, "NTREX-128")):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, dest])
    return dest


def _read_lines(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [l.rstrip("\n") for l in fh]


def _load_languages(root):
    path = os.path.join(root, "LANGUAGES.tsv")
    names = {}
    if os.path.isfile(path):
        for line in _read_lines(path):
            parts = line.split("\t")
            if len(parts) >= 2:
                names[parts[0]] = parts[1]
    return names


def iter_records(root=None, data_root="data", include_additional=True):
    root = root or default_root(data_root)
    lang_names = _load_languages(root)
    docids = _read_lines(os.path.join(root, "DOCUMENT_IDS.tsv"))

    subdirs = ["NTREX-128"]
    if include_additional and os.path.isdir(os.path.join(root, "NTREX-additional")):
        subdirs.append("NTREX-additional")

    for sub in subdirs:
        sdir = os.path.join(root, sub)
        if not os.path.isdir(sdir):
            continue
        src = _read_lines(os.path.join(root, "NTREX-128", _SRC_NAME))
        release = "NTREX-128" if sub == "NTREX-128" else "NTREX-additional"
        for fname in sorted(os.listdir(sdir)):
            m = _REF_RE.match(fname)
            if not m or fname == _SRC_NAME:
                continue
            code = m.group("code")
            variant = m.group("variant")
            ref = _read_lines(os.path.join(sdir, fname))
            ann = None
            if variant:
                ann = json.dumps({"reference_variant": int(variant)})
            for i, r in enumerate(ref):
                yield record(
                    collection=COLLECTION,
                    release=release,
                    year=2019,
                    testset="newstest2019",
                    domain="news",
                    langpair="eng-%s" % code,
                    src_lang="eng",
                    tgt_lang=code,
                    system=None,
                    segment_id=i + 1,
                    doc_id=docids[i] if i < len(docids) else None,
                    source=src[i] if i < len(src) else None,
                    reference=r,
                    hypothesis=None,
                    human_score=None,
                    human_score_level=None,
                    annotations=ann,
                )
