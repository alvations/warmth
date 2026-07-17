#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLORES adapter — the multi-parallel FLORES evaluation benchmark (dev / devtest),
200+ languages, each sentence aligned across every language with per-sentence
metadata (source URL, domain, topic).

Two ingest paths:

* **local** (default here): the classic ``flores200_dataset/`` layout —
  ``dev/<code>.dev``, ``devtest/<code>.devtest`` and ``metadata_{dev,devtest}.tsv``
  (URL / domain / topic / has_image / has_hyperlink). This is FLORES-200; the
  data is byte-identical to the official release and is what WARMTH ships,
  fetched from a public mirror because ``dl.fbaipublicfiles.com`` /
  ``huggingface.co`` are firewalled in some environments.
* **hub** (:func:`iter_records_hf`): the newer FLORES+ superset on the Hub
  (``openlanguagedata/flores_plus``), for environments that can reach it.

FLORES is references-only, so ``hypothesis``/``system``/``human_score`` are
``None``. Directed rows are produced by pairing a pivot language (default
``eng_Latn``) with every other language at the same sentence index; the source
URL is stored as ``doc_id`` (sentences from one article share a URL), and
``domain``/``topic`` are preserved.
"""

import io
import json
import os
import re
import subprocess

from warmth_schema import record

COLLECTION = "flores-plus"
# A public mirror that vendors the exact official ``flores200_dataset/`` tree.
MIRROR_REPO = "https://github.com/AleksandarPetrov/tokenization-fairness"
HF_PATH = "openlanguagedata/flores_plus"
DEFAULT_PIVOT = "eng_Latn"


def default_root(data_root="data"):
    return os.path.join(data_root, "_sources", "flores200", "flores200_dataset")


def fetch(dest=None, data_root="data", repo_url=MIRROR_REPO):
    """Clone a mirror and return the ``flores200_dataset`` directory inside it."""
    clone_dir = dest or os.path.join(data_root, "_sources", "flores200")
    if not os.path.isdir(clone_dir):
        os.makedirs(os.path.dirname(clone_dir), exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, clone_dir])
    inner = os.path.join(clone_dir, "flores200_dataset")
    return inner if os.path.isdir(inner) else clone_dir


def _read(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [l.rstrip("\n") for l in fh]


def _resolve(root):
    if root and os.path.isdir(os.path.join(root, "devtest")):
        return root
    if root and os.path.isdir(os.path.join(root, "flores200_dataset", "devtest")):
        return os.path.join(root, "flores200_dataset")
    return root


def _load_metadata(root, split):
    """metadata_<split>.tsv -> list of dicts aligned to segments (header skipped)."""
    path = os.path.join(root, "metadata_%s.tsv" % split)
    if not os.path.isfile(path):
        return []
    rows = _read(path)
    if not rows:
        return []
    header = rows[0].split("\t")
    out = []
    for line in rows[1:]:
        out.append(dict(zip(header, line.split("\t"))))
    return out


def iter_records(root=None, pivot=DEFAULT_PIVOT, splits=("dev", "devtest")):
    root = _resolve(root or default_root())
    for split in splits:
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            continue
        meta = _load_metadata(root, split)
        pivot_path = os.path.join(sdir, "%s.%s" % (pivot, split))
        pivot_text = _read(pivot_path) if os.path.isfile(pivot_path) else []
        for fname in sorted(os.listdir(sdir)):
            if not fname.endswith("." + split):
                continue
            code = fname[:-(len(split) + 1)]
            if code == pivot:
                continue
            ref = _read(os.path.join(sdir, fname))
            for i, r in enumerate(ref):
                m = meta[i] if i < len(meta) else {}
                ann = {k: m[k] for k in ("topic", "has_image", "has_hyperlink") if m.get(k)}
                yield record(
                    collection=COLLECTION,
                    release="FLORES-200",
                    year=None,
                    testset="flores-%s" % split,
                    domain=m.get("domain"),
                    langpair="%s-%s" % (pivot, code),
                    src_lang=pivot,
                    tgt_lang=code,
                    system=None,
                    segment_id=i + 1,
                    doc_id=m.get("URL"),
                    source=pivot_text[i] if i < len(pivot_text) else None,
                    reference=r,
                    hypothesis=None,
                    human_score=None,
                    human_score_level=None,
                    annotations=json.dumps(ann) if ann else None,
                )


def iter_records_hf(root=None, pivot=DEFAULT_PIVOT, splits=("dev", "devtest"), configs=None):
    """FLORES+ (superset) via the HuggingFace Hub — for network-enabled runs."""
    from adapters._hf import load_hf, config_names
    langs = configs or config_names(HF_PATH)

    def load(cfg, split):
        return {r["id"]: r for r in load_hf(HF_PATH, cfg, split=split)}

    for split in splits:
        pivot_rows = load(pivot, split)
        for cfg in langs:
            if cfg == pivot:
                continue
            for sid, tr in load(cfg, split).items():
                sr = pivot_rows.get(sid)
                yield record(
                    collection=COLLECTION, release="FLORES+", year=None,
                    testset="flores-%s" % split, domain=tr.get("domain"),
                    langpair="%s-%s" % (pivot, cfg), src_lang=pivot, tgt_lang=cfg,
                    system=None, segment_id=int(sid), doc_id=tr.get("url"),
                    source=sr.get("text") if sr else None, reference=tr.get("text"),
                    hypothesis=None, human_score=None, human_score_level=None,
                    annotations=None)
