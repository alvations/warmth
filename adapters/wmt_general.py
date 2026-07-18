#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT General / News MT task adapter (WMT21-25), from the official
``wmt-conference`` GitHub repositories:

* ``wmt{21,22,23,24}-news-systems`` — the classic per-segment layout under
  ``txt/`` (``sources/<lp>.txt``, ``references/<lp>.<ref>.txt``,
  ``system-outputs/<lp>/<system>.txt``, ``documents/<lp>.docs`` = domain\\tdocid).
* ``wmt25-general-mt`` — document-level JSONL (``data/wmt25-genmt.jsonl`` with
  ``src_text`` + ``refs`` per document, and ``data/systems/<system>.jsonl`` with
  one ``hypothesis`` per document).

All map onto the WARMTH schema under ``collection="wmt-general"``. These are the
task submissions (source / reference / system output, with doc ids and domains);
segment-level human scores live in the mt-metrics-eval packages
(:mod:`adapters.wmt_metrics_hi`).
"""

import ast
import glob
import io
import json
import os
import re
import subprocess

from warmth_schema import record
from adapters.wmt_metrics_hi import _read, _load_docs

COLLECTION = "wmt-general"

# repo per edition for --fetch
NEWS_REPOS = {
    2021: "https://github.com/wmt-conference/wmt21-news-systems",
    2022: "https://github.com/wmt-conference/wmt22-news-systems",
    2023: "https://github.com/wmt-conference/wmt23-news-systems",
    2024: "https://github.com/wmt-conference/wmt24-news-systems",
}
WMT25_REPO = "https://github.com/wmt-conference/wmt25-general-mt"


# --------------------------------------------------------------------------
# WMT21-24 news-systems  (segment-level ``txt/`` tree)
# --------------------------------------------------------------------------

def iter_news_systems(txt_dir, year):
    """Ingest one ``txt/`` (or ``txt-ts/``) directory of a news-systems repo."""
    src_dir = os.path.join(txt_dir, "sources")
    ref_dir = os.path.join(txt_dir, "references")
    so_dir = os.path.join(txt_dir, "system-outputs")
    doc_dir = os.path.join(txt_dir, "documents")
    if not os.path.isdir(so_dir):
        return
    release = "WMT%s" % str(year)[-2:]
    testset = "wmttest%d" % year
    for lp in sorted(os.listdir(so_dir)):
        lp_dir = os.path.join(so_dir, lp)
        if not os.path.isdir(lp_dir):
            continue
        src = _read(os.path.join(src_dir, lp + ".txt")) if os.path.isfile(os.path.join(src_dir, lp + ".txt")) else []
        refs = sorted(glob.glob(os.path.join(ref_dir, lp + ".*.txt")))
        reference = _read(refs[0]) if refs else []
        domains, docids = _load_docs(os.path.join(doc_dir, lp + ".docs"))
        src_lang, _, tgt_lang = lp.partition("-")
        for sysfile in sorted(os.listdir(lp_dir)):
            system = re.sub(r"\.txt$", "", sysfile)
            for i, h in enumerate(_read(os.path.join(lp_dir, sysfile))):
                yield record(
                    collection=COLLECTION, release=release, year=year,
                    testset=testset, domain=domains[i] if i < len(domains) else None,
                    langpair=lp, src_lang=src_lang, tgt_lang=tgt_lang, system=system,
                    segment_id=i + 1, doc_id=docids[i] if i < len(docids) else None,
                    source=src[i] if i < len(src) else None,
                    reference=reference[i] if i < len(reference) else None,
                    hypothesis=h, human_score=None, human_score_level=None,
                    annotations=None)


# --------------------------------------------------------------------------
# WMT21 news-systems  (flat "<testset>.<lp>.{src,ref.A,hyp.<system>}.<lang>")
# --------------------------------------------------------------------------

def iter_wmt21(txt_dir, year=2021):
    """WMT21's flat naming: sources/``<ts>.<lp>.src.<lang>``,
    references/``<ts>.<lp>.ref.<X>.<lang>``, system-outputs/
    ``<ts>.<lp>.hyp.<system>.<lang>``. Covers both ``newstest2021`` and
    ``florestest2021`` test sets."""
    src_dir = os.path.join(txt_dir, "sources")
    ref_dir = os.path.join(txt_dir, "references")
    so_dir = os.path.join(txt_dir, "system-outputs")
    if not os.path.isdir(so_dir):
        return
    release = "WMT%s" % str(year)[-2:]

    def _src_ref(testset, lp):
        s = os.path.join(src_dir, "%s.%s.src.%s" % (testset, lp, lp.split("-")[0]))
        src = _read(s) if os.path.isfile(s) else []
        refs = sorted(glob.glob(os.path.join(ref_dir, "%s.%s.ref.*" % (testset, lp))))
        ref = _read(refs[0]) if refs else []
        return src, ref

    cache = {}
    for fname in sorted(os.listdir(so_dir)):
        parts = fname.split(".")
        if len(parts) < 5 or parts[2] != "hyp":
            continue
        testset, lp = parts[0], parts[1]
        system = ".".join(parts[3:-1])
        if lp.count("-") != 1:
            continue
        src_lang, tgt_lang = lp.split("-", 1)
        if (testset, lp) not in cache:
            cache[(testset, lp)] = _src_ref(testset, lp)
        src, ref = cache[(testset, lp)]
        domain = "news" if testset.startswith("news") else \
                 ("flores" if testset.startswith("flores") else None)
        for i, h in enumerate(_read(os.path.join(so_dir, fname))):
            yield record(
                collection=COLLECTION, release=release, year=year,
                testset=testset, domain=domain, langpair=lp,
                src_lang=src_lang, tgt_lang=tgt_lang, system=system,
                segment_id=i + 1, doc_id=None,
                source=src[i] if i < len(src) else None,
                reference=ref[i] if i < len(ref) else None,
                hypothesis=h, human_score=None, human_score_level=None,
                annotations=None)


# --------------------------------------------------------------------------
# WMT25 general-mt  (document-level JSONL)
# --------------------------------------------------------------------------

def _ref_text(refs):
    """refs is {refname: {'ref': text}} — values are sometimes repr-strings."""
    if not isinstance(refs, dict) or not refs:
        return None
    v = next(iter(refs.values()))
    if isinstance(v, dict):
        return v.get("ref")
    if isinstance(v, str):
        try:
            d = ast.literal_eval(v)
            return d.get("ref") if isinstance(d, dict) else v
        except (ValueError, SyntaxError):
            return v
    return None


def iter_wmt25(repo_dir):
    data_dir = os.path.join(repo_dir, "data")
    genmt = os.path.join(data_dir, "wmt25-genmt.jsonl")
    if not os.path.isfile(genmt):
        return
    docs = {}
    order = {}
    counter = {}
    with io.open(genmt, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            d = json.loads(line)
            docs[d["doc_id"]] = d
            lp = "%s-%s" % (d["src_lang"], d["tgt_lang"])
            counter[lp] = counter.get(lp, 0) + 1
            order[d["doc_id"]] = counter[lp]
    sys_dir = os.path.join(data_dir, "systems")
    sysfiles = sorted(glob.glob(os.path.join(sys_dir, "*.jsonl"))) if os.path.isdir(sys_dir) else []
    for path in sysfiles:
        system = os.path.splitext(os.path.basename(path))[0]
        with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                row = json.loads(line)
                doc = docs.get(row.get("doc_id"))
                if not doc:
                    continue
                lp = "%s-%s" % (doc["src_lang"], doc["tgt_lang"])
                ann = {"level": "document"}
                if row.get("paragraphs_match") is not None:
                    ann["paragraphs_match"] = str(row.get("paragraphs_match"))
                # preserve WMT25 multimodal / prompt metadata so nothing is lost
                for k in ("video", "screenshot", "prompt_instruction"):
                    if doc.get(k):
                        ann[k] = doc[k]
                yield record(
                    collection=COLLECTION, release="WMT25", year=2025,
                    testset="wmttest2025", domain=doc.get("domain"),
                    langpair=lp, src_lang=doc["src_lang"], tgt_lang=doc["tgt_lang"],
                    system=system, segment_id=order.get(row["doc_id"], 1),
                    doc_id=row.get("doc_id"), source=doc.get("src_text"),
                    reference=_ref_text(doc.get("refs")), hypothesis=row.get("hypothesis"),
                    human_score=None, human_score_level=None,
                    annotations=json.dumps(ann))


# --------------------------------------------------------------------------
# unified entry point
# --------------------------------------------------------------------------

def iter_records(root=None, year=None):
    """Dispatch by directory shape.

    ``root`` may be: a news-systems repo (has ``txt/``), such a repo's ``txt/``
    dir directly, or the wmt25-general-mt repo (has ``data/wmt25-genmt.jsonl``).
    ``year`` is required for the news-systems layout (it isn't in the files).
    """
    if root is None:
        raise ValueError("wmt-general needs --root (a wmt-conference repo dir)")
    if os.path.isfile(os.path.join(root, "data", "wmt25-genmt.jsonl")):
        yield from iter_wmt25(root)
        return
    txt = root
    if os.path.isdir(os.path.join(root, "txt")):
        txt = os.path.join(root, "txt")
    if year is None:
        m = re.search(r"wmt(\d{2})", root)
        year = 2000 + int(m.group(1)) if m else None
    if year is None:
        raise ValueError("could not infer year for %s; pass year=" % root)
    yield from iter_news_systems(txt, year)


def fetch(dest, year=None, repo_url=None):
    url = repo_url or (NEWS_REPOS.get(year) if year else None)
    if not url:
        raise RuntimeError("pass year (2021-24) or repo_url to fetch wmt-general")
    if not os.path.isdir(dest):
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])
    return dest
