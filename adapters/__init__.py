#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter registry. Each adapter yields :class:`warmth_schema.Record` objects for
one source family, mapping that source onto WARMTH's single superset schema.

``REGISTRY`` maps a collection key to metadata:
    module        the adapter module (has ``iter_records`` and maybe ``fetch``)
    availability  "local"   — materialisable from this repo now
                  "fetch"   — needs a network fetch (HF / GCS / statmt / git)
    note          one-line human description of what/where it is
"""

from adapters import (
    wmt_metrics, wmt_metrics_hi, wmt_general, ntrex, wmt24pp, flores_plus,
    bouquet, wmt_terminology, bio_mqm, wmt_biomed, mt_benchmark,
)

REGISTRY = {
    "wmt-metrics": dict(
        module=wmt_metrics, availability="local",
        note="WMT08-14 news metric task (source/ref/hyp; WMT14 system-level DA) from metric_data/"),
    "wmt-metrics-hi": dict(
        module=wmt_metrics_hi, availability="local",
        note="WMT22-23 mt-metrics-eval slices with seg-level MQM/DA-SQM, docids, domains"),
    "wmt-general": dict(
        module=wmt_general, availability="local",
        note="WMT15-25 general/news: WMT21/24/25 submissions+ESA humeval (wmt-conference), WMT15-20 test sets (mt-eval-benchmark branch)"),
    "ntrex": dict(
        module=ntrex, availability="local",
        note="NTREX-128: eng source -> 128 refs with doc ids"),
    "flores-plus": dict(
        module=flores_plus, availability="local",
        note="FLORES-200 dev+devtest multi-parallel (203 langs) with URL/domain"),
    "wmt24pp": dict(
        module=wmt24pp, availability="local",
        note="WMT24++ post-edits + original MT, all 55 en->xx pairs (google/wmt24pp)"),
    "bio-mqm": dict(
        module=bio_mqm, availability="local",
        note="Bio-MQM: biomedical MT with segment MQM error spans (amazon-science/bio-mqm-dataset)"),
    "wmt-biomed": dict(
        module=wmt_biomed, availability="local",
        note="WMT biomedical parallel test sets, en<->fr (fyvo/WMT-Biomed-Test)"),
    "wmt-terminology": dict(
        module=wmt_terminology, availability="local",
        note="WMT25 (en-zh docs) + WMT23 terminology task (mt-eval-benchmark branch)"),
    "wmt-mqm": dict(
        module=mt_benchmark, availability="merged",
        note="WMT MQM human-evaluation segments (mt-eval-benchmark branch, RicardoRei/wmt-mqm)"),
    "iwslt": dict(
        module=mt_benchmark, availability="merged",
        note="IWSLT17 TED-talk (spoken) test sets (mt-eval-benchmark branch, sacrebleu)"),
    "multi30k": dict(
        module=mt_benchmark, availability="merged",
        note="Multi30k 2016-18 image-caption test sets (mt-eval-benchmark branch, sacrebleu)"),
    "mtedx": dict(
        module=mt_benchmark, availability="merged",
        note="mTEDx multilingual TED-x test sets (mt-eval-benchmark branch, sacrebleu)"),
    "mtnt": dict(
        module=mt_benchmark, availability="merged",
        note="MTNT noisy user-generated-text test sets (mt-eval-benchmark branch, sacrebleu)"),
    "tatoeba": dict(
        module=mt_benchmark, availability="merged",
        note="Tatoeba-MT test sets, 23 pairs (mt-eval-benchmark branch, HF Helsinki-NLP)"),
    "diabla": dict(
        module=mt_benchmark, availability="merged",
        note="DiaBLa bilingual dialogue test set (mt-eval-benchmark branch, HF rbawden/DiaBLa)"),
    "bouquet": dict(
        module=bouquet, availability="fetch",
        note="Meta BOUQuET multi-parallel eval set — HF facebook/bouquet (not reachable here)"),
}


def get(key):
    return REGISTRY[key]["module"]
