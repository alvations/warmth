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
    wmt_metrics, wmt_metrics_hi, ntrex, wmt24pp, flores_plus, bouquet,
    wmt_terminology,
)

REGISTRY = {
    "wmt-metrics": dict(
        module=wmt_metrics, availability="local",
        note="WMT08-14 news metric task (source/ref/hyp; WMT14 system-level DA) from metric_data/"),
    "wmt-metrics-hi": dict(
        module=wmt_metrics_hi, availability="fetch",
        note="WMT15-25 via mt-metrics-eval (seg-level DA/MQM, docids, domains) — GCS"),
    "ntrex": dict(
        module=ntrex, availability="local",
        note="NTREX-128: eng source -> 128 refs with doc ids (git clone) "),
    "wmt24pp": dict(
        module=wmt24pp, availability="fetch",
        note="WMT24++ post-edits + original MT for 55 langs — HF google/wmt24pp"),
    "flores-plus": dict(
        module=flores_plus, availability="fetch",
        note="FLORES+ dev/devtest multi-parallel (200+ langs) — HF openlanguagedata/flores_plus"),
    "bouquet": dict(
        module=bouquet, availability="fetch",
        note="Meta BOUQuET multi-parallel eval set — HF facebook/bouquet"),
    "wmt-terminology": dict(
        module=wmt_terminology, availability="fetch",
        note="WMT terminology task (source/ref/hyp + term constraints) — task repos"),
}


def get(key):
    return REGISTRY[key]["module"]
