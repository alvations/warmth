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
- zh
- ja
- ko
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
- wmt24pp
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
- config_name: wmt-metrics-hi
  data_files:
  - split: train
    path: data/wmt-metrics-hi/*.parquet
- config_name: wmt22
  data_files:
  - split: train
    path: data/wmt-metrics-hi/wmt22.parquet
- config_name: wmt23
  data_files:
  - split: train
    path: data/wmt-metrics-hi/wmt23.parquet
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
- config_name: wmt24pp
  data_files:
  - split: train
    path: data/wmt24pp/*.parquet
- config_name: flores-plus
  data_files:
  - split: train
    path: data/flores-plus/*.parquet
---

# WARMTH — WMT & multilingual MT evaluation data, consolidated

**WARMTH** gathers heterogeneous machine-translation **evaluation** resources —
the WMT metrics / general-MT shared tasks, WMT24++ post-edits, NTREX-128 and
FLORES — into **one HuggingFace-loadable dataset** with a single uniform schema.
Every row is one **segment × system** translation (or, for pure test sets, one
**segment**), so source, reference and hypothesis sit side by side, tagged with
collection, release, language pair, segment id, document id, domain and any
human annotations.

```python
from datasets import load_dataset

ds   = load_dataset("alvations/warmth")                 # everything (~3.26M rows)
wmt  = load_dataset("alvations/warmth", "wmt-metrics")  # WMT08-14
flo  = load_dataset("alvations/warmth", "flores-plus")  # FLORES-200
pp   = load_dataset("alvations/warmth", "wmt24pp")      # WMT24++ post-edits
n19  = load_dataset("alvations/warmth", "ntrex")        # NTREX-128
w23  = load_dataset("alvations/warmth", "wmt23")        # one edition
```

## Schema (one superset for every source)

| field | type | description |
|-------|------|-------------|
| `collection` | string | `wmt-metrics`, `wmt24pp`, `ntrex`, `flores-plus` |
| `release` | string | e.g. `WMT13`, `WMT23`, `NTREX-128`, `FLORES-200`, `wmt24pp` |
| `year` | int32 \| null | edition year |
| `testset` | string \| null | e.g. `newstest2013`, `wmt23`, `flores-devtest` |
| `domain` | string \| null | e.g. `news`, `speech`, `social`, `literary`, `wikinews` |
| `langpair` | string | direction as distributed, e.g. `de-en`, `eng-spa`, `en-ja_JP` |
| `src_lang` / `tgt_lang` | string | normalised language codes |
| `system` | string \| null | MT system id (`null` for pure test sets) |
| `segment_id` | int32 | 1-indexed segment within the test set |
| `doc_id` | string \| null | document id |
| `source` | string \| null | source segment |
| `reference` | string \| null | reference translation |
| `hypothesis` | string \| null | MT output (`null` for pure test sets) |
| `human_score` | float32 \| null | human judgement |
| `human_score_level` | string \| null | `system`, `segment`, or `segment:<name>` (e.g. `segment:mqm`, `segment:da-sqm`) |
| `annotations` | string \| null | JSON: post-edit flags (`is_bad_source`), FLORES topic/flags, etc. |

## What is materialised here (~3.26M rows)

| config | rows | source | ref | hyp | doc_id | human score | notes |
|--------|-----:|:--:|:--:|:--:|:--:|:--:|-------|
| `wmt-metrics` (WMT08–14) | 2,461,040 | ✅* | ✅ | ✅ | via `enrich.py` | WMT14 system-DA | news task |
| `wmt-metrics-hi` (WMT22–23) | 75,530 | ✅ | ✅ | ✅ | ✅ | ✅ seg MQM/DA-SQM | **partial** mt-metrics-eval slices (see below) |
| `wmt24pp` | 54,890 | ✅ | ✅ post-edit | ✅ orig ref | ✅ | — | all 55 en→xx pairs |
| `ntrex` (NTREX-128) | 259,610 | ✅ | ✅ | — | ✅ | — | eng → 128 langs |
| `flores-plus` (FLORES-200) | 407,827 | ✅ | ✅ | — | ✅ (URL) | — | dev + devtest, 203 langs |

`*` WMT14 has no source side locally (per-direction test sets).

Language coverage spans `en↔{cs,de,es,fr,ru,hi,hu,zh,he,…}` (WMT), 128 NTREX
languages, 200+ FLORES languages, and 55 WMT24++ locale pairs.

## Provenance & how each collection is obtained

The build environment could **not** reach `statmt.org`, `huggingface.co`, or the
`mt-metrics-eval` GCS bucket (all firewalled), so wherever possible the data was
fetched from public GitHub copies and re-assembled. Each adapter lives in
[`adapters/`](adapters) and maps its source onto the schema above.

- **wmt-metrics (WMT08–14)** — the plain metric-task files under
  [`metric_data/`](metric_data), read by [`warmth_core.py`](warmth_core.py).
- **wmt-metrics-hi (WMT15–25)** — the `mt-metrics-eval` layout (system outputs,
  seg-level MQM/DA/ESA, doc ids, domains). **WMT22 and WMT23 are materialised
  from community/official mirrors** (`NJUNLP/lost_in_the_src`,
  `wmt-conference/ErrorSpanAnnotation`) and are **partial slices** — the full
  release lives on the GCS bucket. Point `build.py` at a full
  `mt-metrics-eval-v2` extract to complete WMT15–25.
- **wmt24pp** — all 55 `en-xx_XX.jsonl` files (`google/wmt24pp`).
- **ntrex** — `MicrosoftTranslator/NTREX` (CC BY-SA 4.0), doc ids from `DOCUMENT_IDS.tsv`.
- **flores-plus** — the official `flores200_dataset/` (dev + devtest) with
  per-sentence URL / domain / topic metadata.

### Still to fetch (blocked hosts / no GitHub mirror found)

`bouquet` (Meta BOUQuET, HF `facebook/bouquet`) and `wmt-terminology` (WMT
terminology task) have working adapters but no reachable data in this
environment — run their `build.py` step where the Hub / task repos are reachable.
Likewise WMT15–21 / WMT24–25 metrics need a full `mt-metrics-eval-v2` extract.

## (Re)building

```bash
python build.py --list                       # registry + availability
python build.py                              # local collections
python build.py --collections ntrex --fetch  # git-clone then build
python build.py --collections flores-plus --fetch
python build.py --collections wmt24pp  --root /path/to/wmt24pp
python build.py --collections wmt-metrics-hi --root /path/to/mt-metrics-eval-v2
python build.py --collections bouquet        # needs huggingface.co
python enrich.py --fetch --parquet-dir data/wmt-metrics   # WMT08-14 doc ids
```

Add a source by writing one `iter_records()` and registering it in
`adapters/__init__.py`.

## Publishing to the Hub

The `data/**` parquet plus this card's `configs:` block already are a loadable
dataset. To push (run where `huggingface.co` is reachable and authenticated):

```bash
python push_to_hub.py --repo-id alvations/warmth              # upload data/ + README
python push_to_hub.py --repo-id alvations/warmth --via-datasets --dry-run
```

## License

Redistributed for MT and MT-evaluation research; each source keeps its own
terms — WMT/statmt.org shared-task data (cite the relevant *Findings of the WMT*
papers), NTREX-128 (CC BY-SA 4.0), FLORES-200 (CC BY-SA 4.0), WMT24++
(google/wmt24pp, Apache-2.0). Please cite the originating papers and respect the
upstream licenses.
