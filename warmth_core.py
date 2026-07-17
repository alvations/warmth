#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
warmth_core: consolidate the WMT Metrics-task data under ``metric_data/`` into a
single, uniform stream of records.

The WMT (Workshop on [Statistical] Machine Translation / Conference on Machine
Translation) shared tasks distribute their evaluation data (via statmt.org) as a
loose pile of parallel text files: one file per reference, one per source, and
one per participating system output, all line-aligned by segment.  The exact
directory and file-naming conventions drift from year to year.  This module
hides that drift behind one function, :func:`iter_records`, that yields a flat
record for every ``(year, langpair, system, segment)`` combination with the
source, reference and hypothesis stitched back together plus whatever human
annotations we have locally.

The output schema (one dict per yielded record)::

    year          int     e.g. 2014
    wmt           str     e.g. "WMT14"
    testset       str     e.g. "newstest2014"
    langpair      str     e.g. "de-en"   (as it appears on disk)
    src_lang      str     normalised source language code (cz -> cs)
    tgt_lang      str     normalised target language code
    system        str     participating system id, e.g. "uedin-wmt14.3025"
    segment_id    int     1-indexed segment / line number within the test set
    doc_id        str|None document id, when recoverable (None for these
                          plain-text distributions -- see NOTE on doc ids)
    source        str|None source segment (None when the source side is not
                          available locally, e.g. WMT14)
    reference     str|None reference translation
    hypothesis    str     the system's output segment
    human_score   float|None  human judgement attached to this segment/system,
                          when available (WMT14 ships system-level DA scores)
    human_score_level str|None  granularity of ``human_score``:
                          "system" or "segment" or None

NOTE on doc ids: the metric-task packages committed to this repo are the plain
line-aligned ``.txt`` distributions, which carry no document boundaries.  The
document id lives only in the original SGML (``*-src.sgm`` / ``*-ref.sgm``)
packages on statmt.org.  ``doc_id`` is therefore ``None`` for every record here;
the field exists so that a future enrichment pass (parsing the SGM, or the
``mt-metrics-eval`` packages) can populate it without a schema change.
"""

import io
import os
import re

from warmth_schema import Record, record, LANG_NORMALISE, norm_lang

COLLECTION = "wmt-metrics"

# Repository-root-relative default location of the raw WMT files.
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric_data")

# Per-edition layout description.  Everything else (langpairs, systems) is
# discovered by walking the ``system-outputs`` tree so new files are picked up
# automatically.
#
#   ref_by      : "lang"     -> one reference file per language  (multi-parallel)
#                 "langpair" -> one reference file per direction (bilingual)
#   ref_tmpl    : filename template inside ``references/``
#   src         : "from_ref" -> source is the source-language reference file
#                               (only valid for the multi-parallel editions)
#                 "src_dir"  -> a dedicated ``sources/`` directory exists
#                 "none"     -> source side not distributed locally
#   src_tmpl    : filename template inside ``sources/`` (when src == "src_dir")
YEARS = {
    2008: dict(wmt="WMT08", testset="newstest2008", sysroot="newstest2008",
               ref_by="lang", ref_tmpl="newstest2008.{lang}", src="from_ref"),
    2009: dict(wmt="WMT09", testset="newstest2009", sysroot="newstest2009",
               ref_by="lang", ref_tmpl="newstest2009.{lang}", src="from_ref"),
    2010: dict(wmt="WMT10", testset="newssyscombtest2010", sysroot="newssyscombtest",
               ref_by="lang", ref_tmpl="newssyscombtest2010.{lang}", src="from_ref"),
    2011: dict(wmt="WMT11", testset="newstest2011", sysroot="newstest2011",
               ref_by="lang", ref_tmpl="newstest2011-ref.{lang}",
               src="src_dir", src_tmpl="newstest2011-src.{lang}"),
    2012: dict(wmt="WMT12", testset="newstest2012", sysroot="newstest2012",
               ref_by="lang", ref_tmpl="newstest2012-ref.{lang}",
               src="src_dir", src_tmpl="newstest2012-src.{lang}"),
    2013: dict(wmt="WMT13", testset="newstest2013", sysroot="newstest2013",
               ref_by="lang", ref_tmpl="newstest2013-ref.{lang}",
               src="src_dir", src_tmpl="newstest2013-src.{lang}"),
    2014: dict(wmt="WMT14", testset="newstest2014", sysroot="newstest2014",
               ref_by="langpair", ref_tmpl="newstest2014-ref.{langpair}",
               src="none"),
}

# Language codes are written inconsistently across editions (Czech is "cz" in
# 2008-2011 and "cs" from 2012).  ``LANG_NORMALISE`` / ``norm_lang`` come from
# warmth_schema and map e.g. cz -> cs for the src_lang/tgt_lang fields, while the
# on-disk ``langpair`` string is kept untouched.
_norm_lang = norm_lang


def _read_lines(path):
    """Read a text file into a list of newline-stripped unicode strings."""
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def _system_id(filename, testset, langpair):
    """Recover the bare system id from a system-output filename.

    Filenames embed the test-set token and the langpair token in a
    year-dependent order, e.g.::

        de-en.newstest2008.rbmt1          -> rbmt1
        newstest2013.es-en.cu-zeman.2734  -> cu-zeman.2734
        newstest2014.rbmt1.0.de-en        -> rbmt1.0

    Strategy: split on ".", drop the tokens equal to the test-set name and to
    the langpair, and re-join the remainder.  The langpair token itself contains
    a "-" and no ".", so it survives the split intact.
    """
    tokens = filename.split(".")
    keep = [t for t in tokens if t != testset and t != langpair]
    return ".".join(keep)


def _load_wmt14_human_scores(year_dir):
    """Parse WMT14's system-level human (DA) scores.

    Format (tab-separated): ``human  <langpair>  <testset>  <system>  <score>``
    Returns a dict keyed by ``(langpair, system)`` -> float.
    """
    scores = {}
    if not os.path.isdir(year_dir):
        return scores
    for name in os.listdir(year_dir):
        if not name.endswith(".scores"):
            continue
        for line in _read_lines(os.path.join(year_dir, name)):
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            _, langpair, _testset, system, score = parts[:5]
            try:
                scores[(langpair, system)] = float(score)
            except ValueError:
                continue
    return scores


def _human_scores_for(year, year_dir):
    """Return (scores_dict, level) for the human annotations shipped with an
    edition.  Only WMT14 carries them locally, at system level."""
    if year == 2014:
        return _load_wmt14_human_scores(year_dir), "system"
    return {}, None


def available_years(data_root=DEFAULT_DATA_ROOT):
    """Years for which a ``metric_data/WMT##`` directory exists on disk."""
    return [y for y in sorted(YEARS) if os.path.isdir(os.path.join(data_root, YEARS[y]["wmt"]))]


def _reference_path(cfg, year_dir, langpair, tgt):
    refs = os.path.join(year_dir, "references")
    if cfg["ref_by"] == "langpair":
        path = os.path.join(refs, cfg["ref_tmpl"].format(langpair=langpair))
    else:
        path = os.path.join(refs, cfg["ref_tmpl"].format(lang=tgt))
    return path if os.path.isfile(path) else None


def _source_path(cfg, year_dir, src):
    if cfg["src"] == "src_dir":
        path = os.path.join(year_dir, "sources", cfg["src_tmpl"].format(lang=src))
    elif cfg["src"] == "from_ref":
        path = os.path.join(year_dir, "references", cfg["ref_tmpl"].format(lang=src))
    else:  # "none"
        return None
    return path if os.path.isfile(path) else None


def iter_records(data_root=DEFAULT_DATA_ROOT, years=None):
    """Yield :class:`Record` objects for every system output segment.

    Parameters
    ----------
    data_root : str
        Path to the ``metric_data`` directory.
    years : iterable[int] or None
        Restrict to these WMT editions (e.g. ``[2013, 2014]``).  ``None`` means
        every edition present on disk.
    """
    if years is None:
        years = available_years(data_root)

    for year in years:
        cfg = YEARS[year]
        year_dir = os.path.join(data_root, cfg["wmt"])
        sys_root = os.path.join(year_dir, "system-outputs", cfg["sysroot"])
        if not os.path.isdir(sys_root):
            continue

        human_scores, human_level = _human_scores_for(year, year_dir)

        # Cache reference / source line lists so we read each file once.
        ref_cache = {}
        src_cache = {}

        for langpair in sorted(os.listdir(sys_root)):
            lp_dir = os.path.join(sys_root, langpair)
            if not os.path.isdir(lp_dir):
                continue
            if "-" not in langpair:
                continue
            src, tgt = langpair.split("-", 1)

            if langpair not in ref_cache:
                rpath = _reference_path(cfg, year_dir, langpair, tgt)
                ref_cache[langpair] = _read_lines(rpath) if rpath else []
            reference = ref_cache[langpair]

            if langpair not in src_cache:
                spath = _source_path(cfg, year_dir, src)
                src_cache[langpair] = _read_lines(spath) if spath else []
            source = src_cache[langpair]

            for filename in sorted(os.listdir(lp_dir)):
                fpath = os.path.join(lp_dir, filename)
                if not os.path.isfile(fpath):
                    continue
                system = _system_id(filename, cfg["testset"], langpair)
                hyp_lines = _read_lines(fpath)
                score = human_scores.get((langpair, system))

                for i, hyp in enumerate(hyp_lines):
                    yield record(
                        collection=COLLECTION,
                        release=cfg["wmt"],
                        year=year,
                        testset=cfg["testset"],
                        domain="news",
                        langpair=langpair,
                        src_lang=_norm_lang(src),
                        tgt_lang=_norm_lang(tgt),
                        system=system,
                        segment_id=i + 1,
                        doc_id=None,
                        source=source[i] if i < len(source) else None,
                        reference=reference[i] if i < len(reference) else None,
                        hypothesis=hyp,
                        human_score=score,
                        human_score_level=human_level if score is not None else None,
                    )


def _summarise(data_root=DEFAULT_DATA_ROOT):
    """Quick stats for a manual sanity check / CLI use."""
    from collections import Counter
    per_year = Counter()
    langpairs = set()
    systems = set()
    with_src = 0
    with_ref = 0
    with_score = 0
    total = 0
    for r in iter_records(data_root):
        total += 1
        per_year[r.release] += 1
        langpairs.add((r.release, r.langpair))
        systems.add((r.release, r.langpair, r.system))
        with_src += r.source is not None
        with_ref += r.reference is not None
        with_score += r.human_score is not None
    return dict(total=total, per_year=dict(sorted(per_year.items())),
                n_langpairs=len(langpairs), n_systems=len(systems),
                with_source=with_src, with_reference=with_ref,
                with_human_score=with_score)


if __name__ == "__main__":
    import json
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_ROOT
    print(json.dumps(_summarise(root), indent=2, ensure_ascii=False))
