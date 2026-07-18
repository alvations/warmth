#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mt_benchmark.py
=====================

Combine open-source **Machine Translation shared-task test sets** into a single,
uniform evaluation benchmark. Every source below is an official / community MT
*evaluation* set (dev/test/challenge test suites) — there are **no training
corpora**, and there is **nothing adversarial / red-team / robustness-attack**
in here. It is a clean union of public MT benchmarks with a shared schema.

Each row is normalised to 15 columns (see SCHEMA.md). Output is a HuggingFace
`datasets.DatasetDict` saved with `save_to_disk`, one split per source plus an
`all` split, and an `all.jsonl` export.

Run everything:
    python build_mt_benchmark.py --output mt_data

Run a subset of sources:
    python build_mt_benchmark.py --sources wmt,terminology --output mt_data

Sources (source-key -> what it pulls):
    wmt          sacrebleu wmt08..wmt24 news test sets
    mtnt         sacrebleu mtnt1.1/test, mtnt2019 (noisy user-generated text)
    iwslt        sacrebleu iwslt17 (TED talks, spoken)
    multi30k     sacrebleu multi30k/2016,2017,2018 (image-caption MT)
    mtedx        sacrebleu mtedx/test (multilingual TED)
    terminology  HF zouharvi/wmt-terminology-2023
    mqm          HF RicardoRei/wmt-mqm-human-evaluation (MQM-annotated refs)
    wmt24pp      HF google/wmt24pp (post-edited, 2024)
    wmt25        local wmt-conference/wmt25-general-mt (doc-level, refA)
    wmt25_humeval  local wmt25-general-mt human-eval JSONL (refA)
    flores200    HF facebook/flores devtest (en -> 30 langs)
    ntrex128     GitHub MicrosoftTranslator/NTREX (WMT19 in 128 langs)
    diabla       HF rbawden/DiaBLa (en<->fr bilingual dialogue)
    tatoeba      HF Helsinki-NLP/tatoeba_mt (30 en-X configs)

Dependencies: `datasets`, `sacrebleu`. No auth token required for the public HF
datasets. `wmt25*` sources need a local clone of wmt-conference/wmt25-general-mt
(`git lfs pull` its data/*.jsonl); if absent they are skipped cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path("mt_data")

# ----------------------------------------------------------------------------
# sacrebleu test-set groups
# ----------------------------------------------------------------------------
WMT_TESTSETS = [
    "wmt08", "wmt09", "wmt10", "wmt11", "wmt12", "wmt13", "wmt14",
    "wmt15", "wmt16", "wmt17", "wmt18", "wmt19", "wmt20", "wmt21",
    "wmt22", "wmt23", "wmt24",
]
MTNT_TESTSETS = ["mtnt1.1/test", "mtnt2019"]
IWSLT_TESTSETS = ["iwslt17"]
MULTI30K_TESTSETS = ["multi30k/2016", "multi30k/2017", "multi30k/2018"]
MTEDX_TESTSETS = ["mtedx/test"]

# ----------------------------------------------------------------------------
# Provenance: full name, paper, url, canonical citation, loader used.
# ----------------------------------------------------------------------------
PROVENANCE: dict[str, dict[str, str]] = {
    # ---- WMT news (sacrebleu) ----
    "wmt08": {"full_name": "WMT 2008 Shared Task: Third Workshop on Statistical Machine Translation",
              "paper": "Callison-Burch et al., 2008", "url": "https://www.statmt.org/wmt08/",
              "citation": "statmt.org/wmt08", "loader": "sacrebleu"},
    "wmt09": {"full_name": "WMT 2009 Shared Task", "paper": "Callison-Burch et al., 2009",
              "url": "https://www.statmt.org/wmt09/", "citation": "statmt.org/wmt09", "loader": "sacrebleu"},
    "wmt10": {"full_name": "WMT 2010 Shared Task", "paper": "Callison-Burch et al., 2010",
              "url": "https://www.statmt.org/wmt10/", "citation": "statmt.org/wmt10", "loader": "sacrebleu"},
    "wmt11": {"full_name": "WMT 2011 Shared Task", "paper": "Callison-Burch et al., 2011",
              "url": "https://www.statmt.org/wmt11/", "citation": "statmt.org/wmt11", "loader": "sacrebleu"},
    "wmt12": {"full_name": "WMT 2012 Shared Task", "paper": "Callison-Burch et al., 2012",
              "url": "https://www.statmt.org/wmt12/", "citation": "statmt.org/wmt12", "loader": "sacrebleu"},
    "wmt13": {"full_name": "WMT 2013 Shared Task", "paper": "Bojar et al., 2013",
              "url": "https://www.statmt.org/wmt13/", "citation": "statmt.org/wmt13", "loader": "sacrebleu"},
    "wmt14": {"full_name": "WMT 2014 Shared Task: Ninth Workshop on Statistical Machine Translation",
              "paper": "Bojar et al., 2014. Also: Vaswani et al., 2017 (Transformer paper used wmt14 en-de)",
              "url": "https://www.statmt.org/wmt14/", "citation": "statmt.org/wmt14", "loader": "sacrebleu"},
    "wmt15": {"full_name": "WMT 2015 Shared Task", "paper": "Bojar et al., 2015",
              "url": "https://www.statmt.org/wmt15/", "citation": "statmt.org/wmt15", "loader": "sacrebleu"},
    "wmt16": {"full_name": "WMT 2016 Shared Task: First Conference on Machine Translation",
              "paper": "Bojar et al., 2016", "url": "https://www.statmt.org/wmt16/",
              "citation": "statmt.org/wmt16", "loader": "sacrebleu"},
    "wmt17": {"full_name": "WMT 2017 Shared Task", "paper": "Bojar et al., 2017",
              "url": "https://www.statmt.org/wmt17/", "citation": "statmt.org/wmt17", "loader": "sacrebleu"},
    "wmt18": {"full_name": "WMT 2018 Shared Task", "paper": "Bojar et al., 2018",
              "url": "https://www.statmt.org/wmt18/", "citation": "statmt.org/wmt18", "loader": "sacrebleu"},
    "wmt19": {"full_name": "WMT 2019 Shared Task: Fourth Conference on Machine Translation",
              "paper": "Barrault et al., 2019", "url": "https://www.statmt.org/wmt19/",
              "citation": "statmt.org/wmt19", "loader": "sacrebleu"},
    "wmt20": {"full_name": "WMT 2020 Shared Task", "paper": "Barrault et al., 2020",
              "url": "https://www.statmt.org/wmt20/", "citation": "statmt.org/wmt20", "loader": "sacrebleu"},
    "wmt21": {"full_name": "WMT 2021 Shared Task", "paper": "Akhbardeh et al., 2021",
              "url": "https://www.statmt.org/wmt21/", "citation": "statmt.org/wmt21", "loader": "sacrebleu"},
    "wmt22": {"full_name": "WMT 2022 Shared Task", "paper": "Kocmi et al., 2022",
              "url": "https://www.statmt.org/wmt22/", "citation": "statmt.org/wmt22", "loader": "sacrebleu"},
    "wmt23": {"full_name": "WMT 2023 Shared Task", "paper": "Kocmi et al., 2023",
              "url": "https://www.statmt.org/wmt23/", "citation": "statmt.org/wmt23", "loader": "sacrebleu"},
    "wmt24": {"full_name": "WMT 2024 Shared Task: Ninth Conference on Machine Translation",
              "paper": "Kocmi et al., 2024. Findings of the WMT24 General MT Shared Task",
              "url": "https://www2.statmt.org/wmt24/translation-task.html",
              "citation": "aclanthology.org/2024.wmt-1.1", "loader": "sacrebleu"},
    # ---- MTNT (noisy UGC) ----
    "mtnt1.1_test": {"full_name": "MTNT: Machine Translation of Noisy Text", "paper": "Michel & Neubig, EMNLP 2018",
                     "url": "https://www.cs.cmu.edu/~pmichel1/mtnt/", "citation": "aclanthology.org/D18-1050",
                     "loader": "sacrebleu"},
    "mtnt2019": {"full_name": "MTNT 2019: WMT Robustness Task Extension", "paper": "Li et al., WMT 2019",
                 "url": "https://www.statmt.org/wmt19/robustness.html",
                 "citation": "statmt.org/wmt19/robustness.html", "loader": "sacrebleu"},
    # ---- IWSLT / Multi30k / mTEDx ----
    "iwslt17": {"full_name": "IWSLT 2017 Evaluation Campaign - TED talks", "paper": "Cettolo et al., IWSLT 2017",
                "url": "https://iwslt.org/2017", "citation": "iwslt.org/2017", "loader": "sacrebleu"},
    "multi30k": {"full_name": "Multi30K Image Caption Translation", "paper": "Elliott et al., 2016. 30k Flickr captions translated",
                 "url": "https://github.com/multi30k/dataset", "citation": "aclanthology.org/W16-3210", "loader": "sacrebleu"},
    "mtedx": {"full_name": "mTEDx Multilingual TED Talks", "paper": "Salesky et al., Interspeech 2021",
              "url": "https://www.openslr.org/100", "citation": "openslr.org/100", "loader": "sacrebleu"},
    # ---- Terminology / MQM / WMT24++ / WMT25 ----
    "wmt_terminology_2023": {"full_name": "WMT 2023 Shared Task on MT with Terminologies",
                             "paper": "Zouhar et al., WMT 2023. Three modes: no-term, proper-term, random-term",
                             "url": "https://github.com/wmt-terminology-task/data-2023",
                             "citation": "aclanthology.org/2023.wmt-1.54",
                             "loader": "load_dataset('zouharvi/wmt-terminology-2023')"},
    "wmt_mqm": {"full_name": "WMT MQM Human Evaluation Annotations",
                "paper": "Freitag et al., 2021. Multidimensional Quality Metrics error annotations",
                "url": "https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation",
                "citation": "aclanthology.org/2021.wmt-1.73",
                "loader": "load_dataset('RicardoRei/wmt-mqm-human-evaluation')"},
    "wmt24pp": {"full_name": "WMT24++ Extended Test Sets (Google)",
                "paper": "Freitag et al., 2025. 55 languages, 4 domains. arXiv:2502.12404",
                "url": "https://huggingface.co/datasets/google/wmt24pp",
                "citation": "arxiv.org/abs/2502.12404",
                "loader": "load_dataset('google/wmt24pp', 'en-de_DE')"},
    "wmt25": {"full_name": "WMT 2025 General MT Shared Task",
              "paper": "Kocmi et al., WMT 2025. 31 lang pairs, 6 domains, doc-level",
              "url": "https://github.com/wmt-conference/wmt25-general-mt",
              "citation": "statmt.org/wmt25", "loader": "local: wmt25-general-mt/data/wmt25-genmt.jsonl"},
    "wmt25_humeval": {"full_name": "WMT 2025 General MT Human Evaluation (MQM)",
                      "paper": "Kocmi et al., WMT 2025. 40 systems scored with MQM, 14 lang pairs",
                      "url": "https://github.com/wmt-conference/wmt25-general-mt",
                      "citation": "statmt.org/wmt25", "loader": "local: wmt25-general-mt/data/wmt25-genmt-humeval*.jsonl"},
    # ---- Multilingual reference benchmarks ----
    "flores200": {"full_name": "FLORES-200 (NLLB) Multilingual MT Benchmark",
                  "paper": "NLLB Team et al., 2022. 200+ languages, Wikipedia-derived",
                  "url": "https://huggingface.co/datasets/facebook/flores",
                  "citation": "arxiv.org/abs/2207.04672",
                  "loader": "load_dataset('facebook/flores', '<lang_script>', split='devtest')"},
    "ntrex128": {"full_name": "NTREX-128: WMT19 Test Set in 128 Languages",
                 "paper": "Federmann et al., 2022. Microsoft professional translation",
                 "url": "https://github.com/MicrosoftTranslator/NTREX",
                 "citation": "aclanthology.org/2022.sumeval-1.4", "loader": "download zip, parse TXT files"},
    "diabla": {"full_name": "DiaBLa: Bilingual Dialogue Corpus", "paper": "Bawden et al., LREC 2020",
               "url": "https://huggingface.co/datasets/rbawden/DiaBLa",
               "citation": "aclanthology.org/2020.lrec-1.62", "loader": "load_dataset('rbawden/DiaBLa', split='test')"},
    "tatoeba": {"full_name": "Tatoeba MT Challenge Test Sets", "paper": "Tiedemann, WMT 2020. Crowdsourced sentence pairs",
                "url": "https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt",
                "citation": "aclanthology.org/2020.wmt-1.139",
                "loader": "load_dataset('Helsinki-NLP/tatoeba_mt', '<src-tgt>', split='test')"},
}


def _provenance(dataset_key: str) -> dict[str, str]:
    """Provenance for a dataset key, with a base-key fallback so that
    per-year sacrebleu keys (`multi30k_2016`, `mtedx_test`) inherit from
    their base entry (`multi30k`, `mtedx`)."""
    p = PROVENANCE.get(dataset_key)
    if p is None:
        base = dataset_key.rsplit("_", 1)[0]           # multi30k_2016 -> multi30k
        p = PROVENANCE.get(base, {})
        if not p and dataset_key.endswith("_test"):    # mtedx_test -> mtedx
            p = PROVENANCE.get(dataset_key[:-5], {})
    return {
        "dataset_full_name": p.get("full_name", dataset_key),
        "paper": p.get("paper", ""),
        "dataset_url": p.get("url", ""),
        "citation": p.get("citation", ""),
        "loader": p.get("loader", ""),
    }


# 639-3 (and a few 639-1) -> 2-letter, for datasets that carry long codes.
ISO_MAP = {
    "eng": "en", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es",
    "rus": "ru", "zho": "zh", "chi": "zh", "cmn": "zh", "jpn": "ja", "kor": "ko",
    "ara": "ar", "heb": "he", "hin": "hi", "ben": "bn", "tam": "ta", "tel": "te",
    "urd": "ur", "vie": "vi", "tha": "th", "ind": "id", "tur": "tr", "swa": "sw",
    "yor": "yo", "ibo": "ig", "hau": "ha", "amh": "am", "zul": "zu", "som": "so",
    "ell": "el", "pol": "pl", "ukr": "uk", "ces": "cs", "cze": "cs", "nld": "nl",
    "dut": "nl", "swe": "sv", "fin": "fi", "por": "pt", "ita": "it", "ron": "ro",
    "rum": "ro", "hun": "hu", "dan": "da", "nor": "no", "nob": "nb", "isl": "is",
    "cym": "cy", "eus": "eu",
}


def _norm(code: str) -> str:
    return ISO_MAP.get(code, code)


# ----------------------------------------------------------------------------
# 1) sacrebleu sets: WMT, MTNT, IWSLT, Multi30k, mTEDx
# ----------------------------------------------------------------------------
def pull_sacrebleu_sets(testsets: list[str], pairs: list[str] | None = None) -> list[dict]:
    import sacrebleu
    records: list[dict] = []
    for ts in testsets:
        try:
            available_pairs = sacrebleu.get_langpairs_for_testset(ts)
        except Exception:
            print(f"  [skip] {ts}: no pairs found")
            continue
        for pair in available_pairs:
            if pairs and pair not in pairs:
                continue
            try:
                src_file = sacrebleu.get_source_file(ts, pair)
                ref_files = sacrebleu.get_reference_files(ts, pair)
                if not ref_files:
                    continue
                with open(src_file, encoding="utf-8") as f:
                    sources = [line.strip() for line in f]
                with open(ref_files[0], encoding="utf-8") as f:
                    references = [line.strip() for line in f]
                if len(sources) != len(references):
                    n = min(len(sources), len(references))
                    sources, references = sources[:n], references[:n]
                src_lang, tgt_lang = pair.split("-")
                domain = "news"
                if "mtnt" in ts:
                    domain = "noisy_ugc"
                elif "robust" in ts:
                    domain = "robustness"
                elif "iwslt" in ts:
                    domain = "spoken"
                elif "multi30k" in ts:
                    domain = "multimodal"
                year = ""
                for y in range(2008, 2030):
                    if str(y)[-2:] in ts:
                        year = str(y)
                        break
                ds_key = ts.replace("/", "_")
                prov = _provenance(ds_key)
                for i, (s, r) in enumerate(zip(sources, references)):
                    records.append({
                        "source": s, "reference": r,
                        "source_lang": src_lang, "target_lang": tgt_lang,
                        "pair": pair, "dataset": ds_key,
                        "domain": domain, "year": year, "idx": i,
                        "challenge_type": "standard", **prov,
                    })
                print(f"  [ok] {ts} {pair}: {len(sources)} segs")
            except Exception as e:
                print(f"  [skip] {ts} {pair}: {e}")
    return records


# ----------------------------------------------------------------------------
# 2) WMT terminology 2023
# ----------------------------------------------------------------------------
def pull_wmt_terminology() -> list[dict]:
    from datasets import load_dataset
    records: list[dict] = []
    try:
        ds = load_dataset("zouharvi/wmt-terminology-2023", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  [skip] terminology: {e}")
        return records
    prov = _provenance("wmt_terminology_2023")
    counts: Counter = Counter()
    for i, row in enumerate(ds):
        src = (row.get("src") or "").strip()
        ref = (row.get("ref") or "").strip()
        pair = (row.get("langs") or "").strip()   # real pair, e.g. de-en, en-cs, zh-en
        if not (src and ref and pair and "-" in pair):
            continue
        src_lang, tgt_lang = pair.split("-", 1)
        counts[pair] += 1
        records.append({
            "source": src, "reference": ref,
            "source_lang": src_lang, "target_lang": tgt_lang,
            "pair": pair, "dataset": "wmt_terminology_2023",
            "domain": "terminology", "year": "2023", "idx": i,
            "challenge_type": "terminology", **prov,
        })
    print(f"  [ok] terminology: {dict(counts)}")
    return records


# ----------------------------------------------------------------------------
# 3) WMT MQM human evaluation (dedup to one row per (src, pair))
# ----------------------------------------------------------------------------
def pull_mqm_annotations() -> list[dict]:
    from datasets import load_dataset
    records: list[dict] = []
    try:
        ds = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  [skip] mqm: {e}")
        return records
    prov = _provenance("wmt_mqm")
    seen: set = set()
    for i, row in enumerate(ds):
        src = (row.get("src") or "").strip()
        ref = (row.get("ref") or "").strip()
        pair = (row.get("lp") or "en-de").strip()
        if not (src and ref):
            continue
        key = (src, pair)
        if key in seen:
            continue
        seen.add(key)
        src_lang, tgt_lang = (pair.split("-", 1) + [""])[:2]
        records.append({
            "source": src, "reference": ref,
            "source_lang": src_lang, "target_lang": tgt_lang,
            "pair": pair, "dataset": "wmt_mqm",
            "domain": "mqm_evaluated", "year": str(row.get("year", "")),
            "idx": i, "challenge_type": "mqm_human_eval", **prov,
        })
    print(f"  [ok] mqm: {len(records)} unique (src, pair)")
    return records


# ----------------------------------------------------------------------------
# 4) WMT24++ (Google), post-edited targets
# ----------------------------------------------------------------------------
def pull_wmt24pp(pairs: list[str] | None = None) -> list[dict]:
    from datasets import load_dataset
    configs = [
        ("en-de_DE", "en-de"), ("en-zh_CN", "en-zh"), ("en-fr_FR", "en-fr"),
        ("en-ja_JP", "en-ja"), ("en-ru_RU", "en-ru"), ("en-es_MX", "en-es"),
        ("en-cs_CZ", "en-cs"), ("en-hi_IN", "en-hi"), ("en-ko_KR", "en-ko"),
        ("en-it_IT", "en-it"), ("en-pt_BR", "en-pt"), ("en-ar_SA", "en-ar"),
    ]
    prov = _provenance("wmt24pp")
    records: list[dict] = []
    for config, pair in configs:
        if pairs and pair not in pairs:
            continue
        try:
            ds = load_dataset("google/wmt24pp", name=config, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"  [skip] wmt24pp {config}: {e}")
            continue
        src_lang, tgt_lang = pair.split("-")
        kept = 0
        for i, row in enumerate(ds):
            if (row.get("domain") == "canary") or row.get("is_bad_source"):
                continue
            src = (row.get("source") or "").strip()
            ref = (row.get("target") or "").strip()   # post-edited human target
            if not (src and ref):
                continue
            records.append({
                "source": src, "reference": ref,
                "source_lang": src_lang, "target_lang": tgt_lang,
                "pair": pair, "dataset": "wmt24pp",
                "domain": row.get("domain") or "mixed", "year": "2024",
                "idx": i, "challenge_type": "standard", **prov,
            })
            kept += 1
        print(f"  [ok] wmt24pp {pair}: {kept} segs")
    return records


# ----------------------------------------------------------------------------
# 5) WMT25 general + human-eval (local wmt25-general-mt clone)
# ----------------------------------------------------------------------------
def _normalize_wmt25_lang(lang: str) -> str:
    """de_DE -> de, es_MX -> es, sr_Cyrl_RS -> sr_Cyrl, sr_Latn_RS -> sr_Latn."""
    if not lang:
        return lang
    parts = lang.split("_")
    if len(parts) >= 3:               # sr_Cyrl_RS -> sr_Cyrl
        return "_".join(parts[:2])
    if len(parts) == 2 and len(parts[1]) == 2:   # de_DE -> de
        return parts[0]
    return lang


def pull_wmt25_general(data_dir: str = "wmt25-general-mt/data") -> list[dict]:
    path = Path(data_dir) / "wmt25-genmt.jsonl"
    if not path.exists():
        print(f"  [skip] wmt25: {path} not found (clone wmt-conference/wmt25-general-mt + git lfs pull)")
        return []
    prov = _provenance("wmt25")
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            src = (row.get("src_text") or "").strip()
            refs = row.get("refs") or {}
            ref_obj = refs.get("refA") or (next(iter(refs.values())) if refs else {})
            ref = (ref_obj.get("ref", "") if isinstance(ref_obj, dict) else "").strip()
            if not (src and ref):
                continue
            src_lang = _normalize_wmt25_lang(row.get("src_lang", ""))
            tgt_lang = _normalize_wmt25_lang(row.get("tgt_lang", ""))
            records.append({
                "source": src, "reference": ref,
                "source_lang": src_lang, "target_lang": tgt_lang,
                "pair": f"{src_lang}-{tgt_lang}", "dataset": "wmt25",
                "domain": row.get("domain") or "general", "year": "2025",
                "idx": i, "challenge_type": "doc_level", **prov,
            })
    print(f"  [ok] wmt25: {len(records)} segs with refs")
    return records


def pull_wmt25_humeval(data_dir: str = "wmt25-general-mt/data") -> list[dict]:
    prov = _provenance("wmt25_humeval")
    records: list[dict] = []
    for fname, challenge_type in [
        ("wmt25-genmt-humeval.jsonl", "humeval"),
        ("wmt25-genmt-humeval_control.jsonl", "humeval_control"),
    ]:
        path = Path(data_dir) / fname
        if not path.exists():
            print(f"  [skip] wmt25_humeval: {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                src = (row.get("src_text") or "").strip()
                tgt = row.get("tgt_text") or {}
                ref = (tgt.get("refA", "") if isinstance(tgt, dict) else "").strip()
                doc_id = row.get("doc_id") or ""
                if not (src and ref and "_#_" in doc_id):
                    continue
                parts = doc_id.split("_#_")
                pair_raw, domain = parts[0], parts[1] if len(parts) > 1 else ""
                if "-" not in pair_raw:
                    continue
                s_raw, t_raw = pair_raw.split("-", 1)
                src_lang = _normalize_wmt25_lang(s_raw)
                tgt_lang = _normalize_wmt25_lang(t_raw)
                records.append({
                    "source": src, "reference": ref,
                    "source_lang": src_lang, "target_lang": tgt_lang,
                    "pair": f"{src_lang}-{tgt_lang}", "dataset": "wmt25_humeval",
                    "domain": domain or "general", "year": "2025",
                    "idx": i, "challenge_type": challenge_type, **prov,
                })
    print(f"  [ok] wmt25_humeval: {len(records)} segs")
    return records


# ----------------------------------------------------------------------------
# 6) FLORES-200 (en -> 30 langs)
# ----------------------------------------------------------------------------
def pull_flores200(target_langs: list[str] | None = None) -> list[dict]:
    from datasets import load_dataset
    default_targets = {
        "deu_Latn": "de", "fra_Latn": "fr", "spa_Latn": "es", "rus_Cyrl": "ru",
        "zho_Hans": "zh", "jpn_Jpan": "ja", "kor_Hang": "ko", "arb_Arab": "ar",
        "heb_Hebr": "he", "hin_Deva": "hi", "ben_Beng": "bn", "tam_Taml": "ta",
        "tel_Telu": "te", "urd_Arab": "ur", "vie_Latn": "vi", "tha_Thai": "th",
        "ind_Latn": "id", "tur_Latn": "tr", "swh_Latn": "sw", "yor_Latn": "yo",
        "ibo_Latn": "ig", "hau_Latn": "ha", "amh_Ethi": "am", "zul_Latn": "zu",
        "som_Latn": "so", "ell_Grek": "el", "pol_Latn": "pl", "ukr_Cyrl": "uk",
        "ces_Latn": "cs", "nld_Latn": "nl",
    }
    prov = _provenance("flores200")
    records: list[dict] = []
    try:
        eng = load_dataset("facebook/flores", "eng_Latn", split="devtest", trust_remote_code=True)
    except Exception as e:
        print(f"  [skip] flores200 (english): {e}")
        return records
    eng_by_id = {r["id"]: r["sentence"] for r in eng}
    for tgt_config, tgt_lang in default_targets.items():
        if target_langs and tgt_lang not in target_langs:
            continue
        try:
            tgt = load_dataset("facebook/flores", tgt_config, split="devtest", trust_remote_code=True)
        except Exception as e:
            print(f"  [skip] flores200 {tgt_config}: {e}")
            continue
        kept = 0
        for i, row in enumerate(tgt):
            src = (eng_by_id.get(row["id"], "") or "").strip()
            ref = (row.get("sentence", "") or "").strip()
            if not (src and ref):
                continue
            records.append({
                "source": src, "reference": ref,
                "source_lang": "en", "target_lang": tgt_lang,
                "pair": f"en-{tgt_lang}", "dataset": "flores200",
                "domain": row.get("domain", "") or "wikimedia", "year": "2022",
                "idx": i, "challenge_type": "standard", **prov,
            })
            kept += 1
        print(f"  [ok] flores200 en-{tgt_lang}: {kept} segs")
    return records


# ----------------------------------------------------------------------------
# 7) NTREX-128 (WMT19 test in 128 langs) — download + parse GitHub zip
# ----------------------------------------------------------------------------
def pull_ntrex128(cache_dir: str = ".cache") -> list[dict]:
    prov = _provenance("ntrex128")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ntrex_dir = cache / "NTREX-128"
    if not ntrex_dir.exists():
        url = "https://github.com/MicrosoftTranslator/NTREX/archive/refs/heads/main.zip"
        zip_path = cache / "ntrex.zip"
        try:
            print(f"  [dl] {url}")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(cache)
            extracted = cache / "NTREX-main" / "NTREX-128"
            if extracted.exists():
                extracted.rename(ntrex_dir)
        except Exception as e:
            print(f"  [skip] ntrex128: {e}")
            return []
    eng_file = None
    for f in ntrex_dir.glob("newstest2019-*.eng.txt"):
        eng_file = f
        break
    if not eng_file:
        print("  [skip] ntrex128: english file not found")
        return []
    with open(eng_file, encoding="utf-8") as f:
        sources = [line.rstrip("\n") for line in f]
    records: list[dict] = []
    for ref_file in sorted(ntrex_dir.glob("newstest2019-*.txt")):
        if ref_file == eng_file:
            continue
        raw_code = ref_file.stem.split(".")[-1]
        if raw_code == "eng":
            continue
        tgt_lang = _norm(raw_code)
        with open(ref_file, encoding="utf-8") as f:
            refs = [line.rstrip("\n") for line in f]
        n = min(len(sources), len(refs))
        kept = 0
        for i in range(n):
            src, ref = sources[i].strip(), refs[i].strip()
            if not (src and ref):
                continue
            records.append({
                "source": src, "reference": ref,
                "source_lang": "en", "target_lang": tgt_lang,
                "pair": f"en-{tgt_lang}", "dataset": "ntrex128",
                "domain": "news", "year": "2019", "idx": i,
                "challenge_type": "standard", **prov,
            })
            kept += 1
        if kept:
            print(f"  [ok] ntrex128 en-{tgt_lang}: {kept} segs")
    return records


# ----------------------------------------------------------------------------
# 8) DiaBLa (en<->fr dialogue)
# ----------------------------------------------------------------------------
def pull_diabla() -> list[dict]:
    from datasets import load_dataset
    prov = _provenance("diabla")
    records: list[dict] = []
    try:
        ds = load_dataset("rbawden/DiaBLa", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  [skip] diabla: {e}")
        return records
    lang_map = {"english": "en", "french": "fr"}
    for i, row in enumerate(ds):
        src = (row.get("orig", "") or "").strip()
        ref = (row.get("ref", "") or "").strip()
        if not (src and ref):
            continue
        meta = row.get("utterance_meta") or {}
        src_lang = lang_map.get(meta.get("source_language", "english") if isinstance(meta, dict) else "english", "en")
        tgt_lang = lang_map.get(meta.get("target_language", "french") if isinstance(meta, dict) else "french", "fr")
        records.append({
            "source": src, "reference": ref,
            "source_lang": src_lang, "target_lang": tgt_lang,
            "pair": f"{src_lang}-{tgt_lang}", "dataset": "diabla",
            "domain": "dialogue", "year": "2020", "idx": i,
            "challenge_type": "dialogue", **prov,
        })
    print(f"  [ok] diabla: {len(records)} segs")
    return records


# ----------------------------------------------------------------------------
# 9) Tatoeba MT challenge (30 en-X configs, capped)
# ----------------------------------------------------------------------------
def pull_tatoeba(pairs_whitelist: list[str] | None = None, max_per_config: int = 1000) -> list[dict]:
    from datasets import load_dataset
    default_configs = [
        "eng-deu", "eng-fra", "eng-spa", "eng-rus", "eng-zho", "eng-jpn",
        "eng-kor", "eng-ara", "eng-heb", "eng-hin", "eng-ben", "eng-tha",
        "eng-vie", "eng-tur", "eng-pol", "eng-ukr", "eng-ces", "eng-nld",
        "eng-swe", "eng-fin", "eng-ell", "eng-por", "eng-ita", "eng-ron",
        "eng-hun", "eng-dan", "eng-nor", "eng-isl", "eng-cym", "eng-eus",
    ]
    prov = _provenance("tatoeba")
    records: list[dict] = []
    for config in default_configs:
        try:
            ds = load_dataset("Helsinki-NLP/tatoeba_mt", config, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  [skip] tatoeba {config}: {e}")
            continue
        count = 0
        for i, row in enumerate(ds):
            if count >= max_per_config:
                break
            src = (row.get("sourceString", "") or "").strip()
            ref = (row.get("targetString", "") or "").strip()
            src_lang = _norm(row.get("sourceLang", ""))
            tgt_lang = _norm(row.get("targetlang", ""))
            if not (src and ref and src_lang and tgt_lang):
                continue
            pair = f"{src_lang}-{tgt_lang}"
            if pairs_whitelist and pair not in pairs_whitelist:
                continue
            records.append({
                "source": src, "reference": ref,
                "source_lang": src_lang, "target_lang": tgt_lang,
                "pair": pair, "dataset": "tatoeba",
                "domain": "mixed", "year": "2020", "idx": i,
                "challenge_type": "standard", **prov,
            })
            count += 1
        print(f"  [ok] tatoeba {config}: {count} segs")
    return records


# ----------------------------------------------------------------------------
# Assemble + save
# ----------------------------------------------------------------------------
DISPATCH = {
    "wmt": lambda pairs: pull_sacrebleu_sets(WMT_TESTSETS, pairs),
    "mtnt": lambda pairs: pull_sacrebleu_sets(MTNT_TESTSETS, pairs),
    "iwslt": lambda pairs: pull_sacrebleu_sets(IWSLT_TESTSETS, pairs),
    "multi30k": lambda pairs: pull_sacrebleu_sets(MULTI30K_TESTSETS, pairs),
    "mtedx": lambda pairs: pull_sacrebleu_sets(MTEDX_TESTSETS, pairs),
    "terminology": lambda pairs: pull_wmt_terminology(),
    "mqm": lambda pairs: pull_mqm_annotations(),
    "wmt24pp": lambda pairs: pull_wmt24pp(pairs),
    "wmt25": lambda pairs: pull_wmt25_general(),
    "wmt25_humeval": lambda pairs: pull_wmt25_humeval(),
    "flores200": lambda pairs: pull_flores200(),
    "ntrex128": lambda pairs: pull_ntrex128(),
    "diabla": lambda pairs: pull_diabla(),
    "tatoeba": lambda pairs: pull_tatoeba(),
}

DEFAULT_SOURCES = list(DISPATCH.keys())


def build_dataset(sources: list[str] | None = None, pairs: list[str] | None = None,
                  output_dir: Path = OUTPUT_DIR) -> None:
    from datasets import Dataset, DatasetDict

    all_sources = sources or DEFAULT_SOURCES
    all_records: list[dict] = []
    for key in all_sources:
        fn = DISPATCH.get(key)
        if fn is None:
            print(f"  [warn] unknown source '{key}' — skipped")
            continue
        print(f"\n=== {key} ===")
        try:
            all_records.extend(fn(pairs))
        except Exception:
            print(f"  [error] {key}:\n{traceback.format_exc()}")

    if not all_records:
        print("No records pulled; nothing to save.")
        return

    if pairs:
        all_records = [r for r in all_records if r["pair"] in pairs]

    before = len(all_records)
    all_records = [r for r in all_records if r["source"].strip() and r["reference"].strip()]
    print(f"\nDropped {before - len(all_records)} empty rows.")

    by_dataset: dict[str, list[dict]] = {}
    for r in all_records:
        by_dataset.setdefault(r["dataset"], []).append(r)

    splits = {"all": Dataset.from_list(all_records)}
    for ds_name, recs in sorted(by_dataset.items()):
        safe = ds_name.replace(".", "_").replace("/", "_").replace("-", "_")
        splits[safe] = Dataset.from_list(recs)

    dd = DatasetDict(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(output_dir))

    with open(output_dir / "all.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records):,} rows to {output_dir} across {len(by_dataset)} datasets + 'all'.")
    for ds_name, recs in sorted(by_dataset.items()):
        n_pairs = len({r["pair"] for r in recs})
        print(f"  {ds_name:24s} {len(recs):>8,} rows  {n_pairs:>3} pairs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the combined MT shared-task benchmark.")
    ap.add_argument("--sources", "-s", default=None,
                    help="comma-separated source keys (default: all). "
                         "Options: " + ",".join(DEFAULT_SOURCES))
    ap.add_argument("--pairs", "-p", default=None,
                    help="comma-separated language pairs to keep (e.g. en-de,de-en)")
    ap.add_argument("--output", "-o", default=str(OUTPUT_DIR), help="output directory")
    args = ap.parse_args()
    sources = args.sources.split(",") if args.sources else None
    pairs = args.pairs.split(",") if args.pairs else None
    build_dataset(sources=sources, pairs=pairs, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
