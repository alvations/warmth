#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT metrics / general-MT task adapter.

WMT08-14 are shipped in this repo under ``metric_data/`` and are read directly
by :mod:`warmth_core`. WMT15-25 (system outputs + segment-level human DA/MQM
plus doc ids and domains) are not in this repo; they live in the
``mt-metrics-eval`` package and on statmt.org. Use :mod:`adapters.wmt_metrics_hi`
to fetch and ingest those editions where the network allows.
"""

import warmth_core

COLLECTION = "wmt-metrics"


def iter_records(root=None, data_root="data"):
    yield from warmth_core.iter_records(root or warmth_core.DEFAULT_DATA_ROOT)
