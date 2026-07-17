#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small helper for adapters that pull from the HuggingFace Hub.

The Hub is unreachable from some build environments (e.g. the one WARMTH was
assembled in); these adapters are therefore written to run *later*, wherever
``huggingface.co`` is reachable. ``load_hf`` centralises the import + a clear
error message so a blocked run fails loudly instead of silently.
"""


def load_hf(path, name=None, split=None, streaming=False):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the `datasets` package is required to ingest %s; pip install datasets"
            % path) from exc
    return load_dataset(path, name, split=split, streaming=streaming)


def config_names(path):
    from datasets import get_dataset_config_names
    return get_dataset_config_names(path)
