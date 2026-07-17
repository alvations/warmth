---
pretty_name: WARMTH — WMT & multilingual MT evaluation data, consolidated
license: other
language:
- en
- de
- fr
- es
- cs
- ru
- hi
- zh
- ja
task_categories:
- translation
tags:
- machine-translation
- wmt
- statmt
- mt-evaluation
- mt-metrics
- human-evaluation
- ntrex
- flores
- terminology
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
- config_name: wmt-metrics
  data_files:
  - split: train
    path: data/wmt-metrics/*.parquet
- config_name: wmt08
  data_files:
  - split: train
    path: data/wmt-metrics/wmt08.parquet
- config_name: wmt09
  data_files:
  - split: train
    path: data/wmt-metrics/wmt09.parquet
- config_name: wmt10
  data_files:
  - split: train
    path: data/wmt-metrics/wmt10.parquet
- config_name: wmt11
  data_files:
  - split: train
    path: data/wmt-metrics/wmt11.parquet
- config_name: wmt12
  data_files:
  - split: train
    path: data/wmt-metrics/wmt12.parquet
- config_name: wmt13
  data_files:
  - split: train
    path: data/wmt-metrics/wmt13.parquet
- config_name: wmt14
  data_files:
  - split: train
    path: data/wmt-metrics/wmt14.parquet
- config_name: ntrex
  data_files:
  - split: train
    path: data/ntrex/*.parquet
- config_name: ntrex-128
  data_files:
  - split: train
    path: data/ntrex/ntrex-128.parquet
- config_name: ntrex-additional
  data_files:
  - split: train
    path: data/ntrex/ntrex-additional.parquet
---

# WARMTH — WMT & multilingual MT evaluation data, consolidated

**WARMTH** gathers heterogeneous machine-translation **evaluation** resources —
the WMT metrics / general-MT shared tasks, WMT24++ post-edits, NTREX-128,
FLORES+, Meta's BOUQuET, and the WMT terminology task — into **one
HuggingFace-loadable dataset** with a single uniform schema. Every row is one
**segment × system** translation (or, for pure test sets, one **segment**), so
source, reference and hypothesis sit side by side, tagged with collection,
release, language pair, segment id, document id, domain and any human
annotations.

```python
from datasets import load_dataset

ds  = load_dataset("alvations/warmth")               # everything built
wmt = load_dataset("alvations/warmth", "wmt-metrics") # one collection
n19 = load_dataset("alvations/warmth", "ntrex")       # NTREX-128
w14 = load_dataset("alvations/warmth", "wmt14")       # one WMT edition
```

## Schema (one superset for every source)

| field | type | description |
|-------|------|-------------|
| `collection` | string | source family: `wmt-metrics`, `wmt24pp`, `ntrex`, `flores-plus`, `bouquet`, `wmt-terminology` |
| `release` | string | specific release, e.g. `WMT13`, `NTREX-128`, `FLORES+`, `wmt24pp` |
| `year` | int32 \| null | edition year |
| `testset` | string \| null | e.g. `newstest2013`, `flores-devtest` |
| `domain` | string \| null | e.g. `news`, `speech`, `social` |
| `langpair` | string | direction as distributed, e.g. `de-en`, `eng-spa` |
| `src_lang` / `tgt_lang` | string | normalised language codes |
| `system` | string \| null | MT system id (`null` for pure test sets) |
| `segment_id` | int32 | 1-indexed segment within the test set |
| `doc_id` | string \| null | document id |
| `source` | string \| null | source segment |
| `reference` | string \| null | reference translation |
| `hypothesis` | string \| null | MT output (`null` for pure test sets) |
| `human_score` | float32 \| null | human judgement |
| `human_score_level` | string \| null | `system`, `segment`, or `segment:<name>` (e.g. `segment:mqm`) |
| `annotations` | string \| null | JSON for anything structured (MQM spans, post-edits, term constraints, quality flags) |

## Collections

| key | availability | rows built here | what it is |
|-----|--------------|----------------:|------------|
| `wmt-metrics` | **local** | 2,461,040 | WMT08–14 news metric task — source/ref/hyp; WMT14 system-level DA |
| `ntrex` | **local** | 259,610 | NTREX-128: English source → 128 refs, with document ids |
| `wmt-metrics-hi` | fetch | — | WMT15–25 via `mt-metrics-eval` — seg-level DA/MQM/ESA, doc ids, domains (GCS) |
| `wmt24pp` | fetch | — | WMT24++ post-edits + original MT, 55 langs (HF `google/wmt24pp`) |
| `flores-plus` | fetch | — | FLORES+ dev/devtest, 200+ langs (HF `openlanguagedata/flores_plus`) |
| `bouquet` | fetch | — | Meta BOUQuET multi-parallel eval set (HF `facebook/bouquet`) |
| `wmt-terminology` | fetch | — | WMT terminology task: source/ref/hyp + term constraints (task repos) |

**local** collections are materialised in `data/` and load out of the box.
**fetch** collections need a network download and are built with `build.py`
(the environment this repo was assembled in could not reach statmt.org /
huggingface.co / the mt-metrics-eval GCS bucket, so they are shipped as
ready-to-run adapters rather than pre-built parquet).

## Building the fetch collections

```bash
python build.py --list                       # registry + availability
python build.py                              # (re)build all local collections
python build.py --collections ntrex --fetch  # git-clone NTREX, then build

# WMT15-25 metrics: download mt-metrics-eval-v2.tgz where GCS is reachable, extract, then
python build.py --collections wmt-metrics-hi --root /path/to/mt-metrics-eval-v2

# HF-hosted sources (run where huggingface.co is reachable)
python build.py --collections wmt24pp        # google/wmt24pp
python build.py --collections flores-plus    # openlanguagedata/flores_plus
python build.py --collections bouquet        # facebook/bouquet

# WMT terminology task (point at the task's JSONL data)
python build.py --collections wmt-terminology --root /path/to/terminology_data
```

Each adapter lives in [`adapters/`](adapters) and maps its source onto the
schema above; add a new source by writing one `iter_records()` and registering
it in `adapters/__init__.py`.

## WMT metrics: doc ids & more human scores

WMT08–14 ship as plain line-aligned text (no document boundaries; only WMT14 has
system-level DA). [`enrich.py`](enrich.py) fills `doc_id` from the WMT `test.tgz`
**SGML** and merges per-segment human scores — including a one-command `--fetch`:

```bash
python enrich.py --fetch --parquet-dir data/wmt-metrics      # download SGML → doc_id
python enrich.py --human-scores da-seg-scores.tsv --parquet-dir data/wmt-metrics
python enrich.py --self-test
```

For WMT15–25 the `wmt-metrics-hi` adapter already brings doc ids, domains and
segment-level DA/MQM/ESA from `mt-metrics-eval`.

## Publishing to the Hub

The `data/**` parquet plus this card's `configs:` block already are a loadable
dataset. To push it (run where `huggingface.co` is reachable and authenticated):

```bash
python push_to_hub.py --repo-id alvations/warmth              # upload data/ + README
python push_to_hub.py --repo-id alvations/warmth --via-datasets --dry-run
```

## Provenance & license

Redistributed for MT and MT-evaluation research; each source keeps its own
terms — WMT/statmt.org shared-task data (cite the relevant *Findings of the WMT*
papers), NTREX-128 (CC BY-SA 4.0), FLORES+ (CC BY-SA 4.0), WMT24++ (Apache-2.0),
BOUQuET (Meta), and the WMT terminology task. Please cite the originating papers
and respect the upstream licenses.
