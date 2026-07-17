#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT15-25 metrics adapter (the "high" years) via the ``mt-metrics-eval`` package.

The modern WMT Metrics / General-MT evaluation data — system outputs, per-segment
human scores (DA / DA+SQM / MQM / ESA), document ids and domains — is packaged by
Google's ``mt-metrics-eval`` toolkit and downloaded from Google Cloud Storage
(``storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz``), which may be
firewalled in some build environments. Extract it, then point this adapter at the
``mt-metrics-eval-v2`` directory.

Expected EvalSet layout (per test set, e.g. ``wmt22/``)::

    sources/<lp>.txt
    references/<lp>.<refname>.txt
    documents/<lp>.docs             # per segment: "<domain>\t<docid>"
    system-outputs/<lp>/<system>.txt
    human-scores/<lp>.<name>.seg.score   # per segment block: "<system>\t<score>"

Each system's output is line-aligned to the source; human-score files repeat the
source length once per system. ``<name>`` (mqm, da-sqm, wmt-z, esa, …) is recorded
in ``human_score_level`` as ``"segment:<name>"``.
"""

import glob
import io
import json
import os
import re

from warmth_schema import record

COLLECTION = "wmt-metrics"
GCS_TARBALL = "https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz"

def _year_from_testset(ts):
    """'wmt22' -> 2022, 'newstest2019' -> 2019, else None."""
    m = re.search(r"(20\d\d)", ts)
    if m:
        return int(m.group(1))
    m = re.search(r"wmt(\d{2})", ts)
    if m:
        return 2000 + int(m.group(1))
    return None


def _read(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [l.rstrip("\n") for l in fh]


def _load_docs(path):
    """documents/<lp>.docs -> ([domain per seg], [docid per seg])."""
    domains, docids = [], []
    if not os.path.isfile(path):
        return domains, docids
    for line in _read(path):
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) == 2:
            domains.append(parts[0]); docids.append(parts[1])
        elif parts:
            domains.append(None); docids.append(parts[0])
        else:
            domains.append(None); docids.append(None)
    return domains, docids


def _load_seg_scores(hs_dir, lp, n_src):
    """Return {name: {system: [score per seg]}} for one langpair."""
    out = {}
    for path in glob.glob(os.path.join(hs_dir, "%s.*.seg.score" % lp)):
        name = os.path.basename(path)[len(lp) + 1:-len(".seg.score")]
        by_sys = {}
        rows = _read(path)
        # files are "<system>\t<score>" repeated in blocks of n_src segments
        for idx, line in enumerate(rows):
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            sysname, score = parts[0], parts[1]
            seg = idx % n_src if n_src else idx
            by_sys.setdefault(sysname, {})[seg + 1] = score
        out[name] = by_sys
    return out


def iter_records(root, testsets=None):
    """Walk an extracted ``mt-metrics-eval-v2`` directory.

    ``root`` is that directory; ``testsets`` optionally restricts to e.g.
    ``["wmt22", "wmt23"]`` (default: every subdir that has ``system-outputs``).
    """
    if testsets is None:
        testsets = sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d, "system-outputs")))
    for ts in testsets:
        base = os.path.join(root, ts)
        year = _year_from_testset(ts)
        so_dir = os.path.join(base, "system-outputs")
        src_dir = os.path.join(base, "sources")
        ref_dir = os.path.join(base, "references")
        doc_dir = os.path.join(base, "documents")
        hs_dir = os.path.join(base, "human-scores")
        if not os.path.isdir(so_dir):
            continue
        for lp in sorted(os.listdir(so_dir)):
            lp_dir = os.path.join(so_dir, lp)
            if not os.path.isdir(lp_dir):
                continue
            src = _read(os.path.join(src_dir, lp + ".txt")) if os.path.isdir(src_dir) else []
            n = len(src)
            # first reference file for this lp (there can be several)
            refs = sorted(glob.glob(os.path.join(ref_dir, lp + ".*.txt"))) if os.path.isdir(ref_dir) else []
            reference = _read(refs[0]) if refs else []
            domains, docids = _load_docs(os.path.join(doc_dir, lp + ".docs"))
            seg_scores = _load_seg_scores(hs_dir, lp, n) if os.path.isdir(hs_dir) else {}
            src_lang, _, tgt_lang = lp.partition("-")

            for sysfile in sorted(os.listdir(lp_dir)):
                system = re.sub(r"\.txt$", "", sysfile)
                hyp = _read(os.path.join(lp_dir, sysfile))
                for i, h in enumerate(hyp):
                    score = None
                    level = None
                    for name, by_sys in seg_scores.items():
                        s = by_sys.get(system, {}).get(i + 1)
                        if s not in (None, "None", "none"):
                            try:
                                score = float(s); level = "segment:%s" % name
                                break
                            except ValueError:
                                pass
                    yield record(
                        collection=COLLECTION,
                        release="WMT%s" % (str(year)[-2:] if year else "?"),
                        year=year,
                        testset=ts,
                        domain=domains[i] if i < len(domains) else None,
                        langpair=lp,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        system=system,
                        segment_id=i + 1,
                        doc_id=docids[i] if i < len(docids) else None,
                        source=src[i] if i < len(src) else None,
                        reference=reference[i] if i < len(reference) else None,
                        hypothesis=h,
                        human_score=score,
                        human_score_level=level,
                        annotations=None,
                    )
