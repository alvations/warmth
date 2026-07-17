#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT Biomedical test-set adapter (``fyvo/WMT-Biomed-Test``).

The ``Sentence level test sets`` directory holds line-aligned parallel files
named ``<dataset>_<src>2<tgt>.<lang>`` (e.g. ``medline18_en2fr.en`` /
``medline18_en2fr.fr``). For direction ``X2Y`` the ``.X`` file is the source and
the ``.Y`` file is the reference. These are references-only biomedical test
sets, so ``hypothesis`` / ``human_score`` are ``None``.
"""

import io
import os
import re

from warmth_schema import record, norm_lang

COLLECTION = "wmt-biomed"
REPO_URL = "https://github.com/fyvo/WMT-Biomed-Test"
TESTSET_DIR = "Sentence level test sets"

_NAME_RE = re.compile(r"^(?P<ds>.+?)_(?P<src>[a-z]{2,3})2(?P<tgt>[a-z]{2,3})\.(?P<lang>[a-z]{2,3})$")


def _read(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [l.rstrip("\n") for l in fh]


def iter_records(root=None):
    tdir = os.path.join(root, TESTSET_DIR)
    if not os.path.isdir(tdir):
        tdir = root
    # group files by (dataset, direction); each has a .src and .tgt member
    groups = {}
    for fname in os.listdir(tdir):
        m = _NAME_RE.match(fname)
        if not m:
            continue
        key = (m.group("ds"), m.group("src"), m.group("tgt"))
        groups.setdefault(key, {})[m.group("lang")] = os.path.join(tdir, fname)

    for (ds, src, tgt), files in sorted(groups.items()):
        if src not in files or tgt not in files:
            continue
        source = _read(files[src])
        reference = _read(files[tgt])
        ym = re.search(r"(\d{2,4})$", ds)
        year = None
        if ym:
            y = int(ym.group(1))
            year = 2000 + y if y < 100 else y
        for i, s in enumerate(source):
            yield record(
                collection=COLLECTION, release="WMT-Biomed", year=year,
                testset=ds, domain="biomedical",
                langpair="%s-%s" % (src, tgt),
                src_lang=norm_lang(src), tgt_lang=norm_lang(tgt),
                system=None, segment_id=i + 1, doc_id=None,
                source=s, reference=reference[i] if i < len(reference) else None,
                hypothesis=None, human_score=None, human_score_level=None,
                annotations=None)
