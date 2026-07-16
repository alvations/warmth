#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich the WARMTH parquet shards with the two fields that the plain-text
metric-task distributions cannot carry: ``doc_id`` (document boundaries) and
segment-level ``human_score``.

Both live in packages that are *not* in this repo and are served from
statmt.org (which may be firewalled in some environments). This script is the
turnkey merge step: point it at those files wherever you can reach them and it
rewrites ``data/*.parquet`` in place, without changing the schema.

--------------------------------------------------------------------------
1. Document ids  (``--sgm-dir``)
--------------------------------------------------------------------------
The WMT ``test`` tarballs (e.g. ``https://statmt.org/wmt13/test.tgz``) contain
SGML source files such as ``newstest2013-src.de.sgm`` whose structure is::

    <srcset setid="newstest2013" srclang="de">
      <doc docid="spiegel/2012/..." genre="news" origlang="de">
        <seg id="1">...</seg>
        <seg id="2">...</seg>
    ...

The segments, concatenated in document order, are exactly the line order of the
plain ``sources/`` / ``references/`` files this dataset was built from (verified:
segment 1 of ``newstest2013-src.en`` is the SGM's first ``<seg>``). So the
``doc_id`` for ``(year, src_lang, segment_id)`` is recoverable by walking the
source SGM of that year/language.

Usage::

    # extract each year's test.tgz so the .sgm files sit under one directory
    python enrich.py --sgm-dir /path/to/wmt_test_sgm --parquet-dir data

--------------------------------------------------------------------------
2. Segment-level human scores  (``--human-scores FILE ...``)
--------------------------------------------------------------------------
Pass one or more tab/CSV files of per-segment human judgements. Each must have
a header naming (case-insensitive, aliases accepted) the columns:

    year | langpair | system | segment_id | score   [| level]

Rows are matched to the dataset on ``(year, langpair, system, segment_id)`` and
written into ``human_score`` (with ``human_score_level`` set to ``level`` or
``"segment"``). This is the format WMT's Direct-Assessment seg-score releases
reduce to; a tiny adapter is all most releases need.

Run ``python enrich.py --self-test`` to check the SGM parser without any files.
"""

import argparse
import csv
import glob
import io
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from warmth_core import LANG_NORMALISE, YEARS

# ---------------------------------------------------------------------------
# SGML source parsing -> {(year, norm_src_lang): [docid_for_seg1, docid2, ...]}
# ---------------------------------------------------------------------------

_SETID_RE = re.compile(r'setid="([^"]+)"')
_SRCLANG_RE = re.compile(r'srclang="([^"]+)"')
_DOCID_RE = re.compile(r'docid="([^"]+)"')
_SEG_OPEN_RE = re.compile(r"<seg\b")
# testset name -> year, e.g. "newstest2013" -> 2013, "newssyscombtest2010" -> 2010
_YEAR_RE = re.compile(r"(20\d\d)")


def _norm_lang(code):
    return LANG_NORMALISE.get(code, code)


def parse_sgm(path):
    """Return (year, norm_srclang, [docid per segment in order]) for a source SGM.

    Robust to the SGML being upper/lower case and to ``<seg>`` and ``<doc>``
    tags appearing anywhere on a line.
    """
    year = None
    srclang = None
    docids = []
    current_doc = None
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            low = line.lower()
            if year is None and ("<srcset" in low or "<refset" in low or "<tstset" in low):
                m = _SETID_RE.search(line)
                if m:
                    ym = _YEAR_RE.search(m.group(1))
                    if ym:
                        year = int(ym.group(1))
                m = _SRCLANG_RE.search(line)
                if m and m.group(1) != "any":
                    srclang = m.group(1)
            if "<doc" in low:
                m = _DOCID_RE.search(line)
                if m:
                    current_doc = m.group(1)
                    # origlang lets us recover srclang when the header said "any"
                    if srclang is None:
                        om = re.search(r'origlang="([^"]+)"', line)
                        if om:
                            srclang = om.group(1)
            # a <seg ...> opens a scored segment
            for _ in _SEG_OPEN_RE.finditer(line):
                docids.append(current_doc)
    if srclang is not None:
        srclang = _norm_lang(srclang)
    return year, srclang, docids


def build_docid_map(sgm_dir):
    """Walk every ``*src*.sgm`` under ``sgm_dir`` -> {(year, src_lang): [docids]}."""
    docmap = {}
    patterns = ["*src*.sgm", "*src*.sgml", "*-src.*.sgm"]
    seen = set()
    files = []
    for pat in patterns:
        for p in glob.glob(os.path.join(sgm_dir, "**", pat), recursive=True):
            if p not in seen:
                seen.add(p)
                files.append(p)
    for path in sorted(files):
        year, srclang, docids = parse_sgm(path)
        if year is None or srclang is None or not docids:
            continue
        key = (year, srclang)
        # Prefer the longest parse if the same (year,lang) appears twice.
        if key not in docmap or len(docids) > len(docmap[key]):
            docmap[key] = docids
    return docmap


# ---------------------------------------------------------------------------
# Human-score files -> {(year, langpair, system, segment_id): (score, level)}
# ---------------------------------------------------------------------------

_COL_ALIASES = {
    "year": {"year", "wmt_year", "yr"},
    "langpair": {"langpair", "lp", "lang_pair", "direction"},
    "system": {"system", "sysid", "system_id", "sys"},
    "segment_id": {"segment_id", "segid", "seg", "seg_id", "sid"},
    "score": {"score", "human_score", "z", "raw", "da", "avg_score"},
    "level": {"level", "granularity"},
}


def _resolve_columns(header):
    idx = {}
    lowered = [h.strip().lower() for h in header]
    for canonical, aliases in _COL_ALIASES.items():
        for i, h in enumerate(lowered):
            if h in aliases:
                idx[canonical] = i
                break
    return idx


def _sniff_reader(fh):
    sample = fh.read(4096)
    fh.seek(0)
    delim = "\t" if sample.count("\t") >= sample.count(",") else ","
    return csv.reader(fh, delimiter=delim)


def load_human_scores(paths):
    scores = {}
    for path in paths:
        with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
            reader = _sniff_reader(fh)
            rows = iter(reader)
            header = next(rows, None)
            if header is None:
                continue
            cols = _resolve_columns(header)
            required = {"year", "langpair", "system", "segment_id", "score"}
            if not required.issubset(cols):
                sys.stderr.write(
                    "warning: %s missing columns %s; skipped\n"
                    % (path, sorted(required - set(cols)))
                )
                continue
            for row in rows:
                try:
                    year = int(re.search(r"20\d\d", row[cols["year"]]).group(0))
                    langpair = row[cols["langpair"]].strip()
                    system = row[cols["system"]].strip()
                    seg = int(row[cols["segment_id"]])
                    score = float(row[cols["score"]])
                except (ValueError, AttributeError, IndexError):
                    continue
                level = "segment"
                if "level" in cols and cols["level"] < len(row) and row[cols["level"]].strip():
                    level = row[cols["level"]].strip()
                scores[(year, langpair, system, seg)] = (score, level)
    return scores


# ---------------------------------------------------------------------------
# Rewrite parquet in place
# ---------------------------------------------------------------------------

def enrich_parquet(parquet_dir, docmap, human_scores, dry_run=False):
    stats = {"rows": 0, "docid_filled": 0, "score_filled": 0, "files": 0}
    for path in sorted(glob.glob(os.path.join(parquet_dir, "wmt*.parquet"))):
        table = pq.read_table(path)
        cols = table.to_pydict()
        n = table.num_rows
        stats["rows"] += n
        stats["files"] += 1

        doc_col = list(cols["doc_id"])
        score_col = list(cols["human_score"])
        level_col = list(cols["human_score_level"])

        for i in range(n):
            year = cols["year"][i]
            seg = cols["segment_id"][i]
            src_lang = cols["src_lang"][i]
            langpair = cols["langpair"][i]
            system = cols["system"][i]

            docids = docmap.get((year, src_lang))
            if docids and 1 <= seg <= len(docids) and docids[seg - 1] is not None:
                if doc_col[i] is None:
                    stats["docid_filled"] += 1
                doc_col[i] = docids[seg - 1]

            hs = human_scores.get((year, langpair, system, seg))
            if hs is not None:
                if score_col[i] is None:
                    stats["score_filled"] += 1
                score_col[i], level_col[i] = hs[0], hs[1]

        cols["doc_id"] = doc_col
        cols["human_score"] = score_col
        cols["human_score_level"] = level_col

        if not dry_run:
            pq.write_table(pa.table(cols, schema=table.schema), path, compression="zstd")
    return stats


# ---------------------------------------------------------------------------
# Self-test (no external files needed)
# ---------------------------------------------------------------------------

_SELFTEST_SGM = """<srcset setid="newstest2013" srclang="en">
<doc docid="cyberpresse/2012/12/01/1564248" genre="news" origlang="fr">
<seg id="1">A Republican strategy to counter the re-election of Obama</seg>
<seg id="2">Republican leaders justified their policy by the need to combat electoral fraud.</seg>
</doc>
<doc docid="bbc.co.uk/2012/11/30/999" genre="news" origlang="en">
<seg id="1">Second document, first segment.</seg>
</doc>
</srcset>
"""


def _self_test():
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "newstest2013-src.en.sgm")
    with io.open(p, "w", encoding="utf-8") as fh:
        fh.write(_SELFTEST_SGM)
    year, lang, docids = parse_sgm(p)
    assert year == 2013, year
    assert lang == "en", lang
    assert docids == [
        "cyberpresse/2012/12/01/1564248",
        "cyberpresse/2012/12/01/1564248",
        "bbc.co.uk/2012/11/30/999",
    ], docids
    m = build_docid_map(d)
    assert m[(2013, "en")][0] == "cyberpresse/2012/12/01/1564248"
    print("self-test OK: parsed", len(docids), "segments across 2 documents")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", default="data")
    ap.add_argument("--sgm-dir", default=None,
                    help="directory with extracted WMT *-src.*.sgm files")
    ap.add_argument("--human-scores", nargs="*", default=[],
                    help="TSV/CSV files with per-segment human scores")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing parquet")
    ap.add_argument("--self-test", action="store_true",
                    help="run the SGM-parser self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.sgm_dir and not args.human_scores:
        ap.error("nothing to do: pass --sgm-dir and/or --human-scores "
                 "(or --self-test)")

    docmap = build_docid_map(args.sgm_dir) if args.sgm_dir else {}
    if args.sgm_dir:
        print("doc_id map: %d (year,lang) groups, e.g. %s"
              % (len(docmap), sorted(docmap)[:5]))
    human = load_human_scores(args.human_scores) if args.human_scores else {}
    if args.human_scores:
        print("human scores: %d segment entries loaded" % len(human))

    stats = enrich_parquet(args.parquet_dir, docmap, human, dry_run=args.dry_run)
    print(("[dry-run] " if args.dry_run else "") +
          "rows=%(rows)d files=%(files)d doc_id_filled=%(docid_filled)d "
          "score_filled=%(score_filled)d" % stats)


if __name__ == "__main__":
    main()
