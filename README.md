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
- uk
- hi
task_categories:
- translation
tags:
- machine-translation
- wmt
- statmt
- mt-evaluation
- mt-metrics
- human-evaluation
- mqm
- ntrex
- flores
- wmt24pp
- terminology
- biomedical
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
- config_name: wmt-general
  data_files:
  - split: train
    path: data/wmt-general/*.parquet
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
- config_name: flores-plus
  data_files:
  - split: train
    path: data/flores-plus/*.parquet
- config_name: wmt24pp
  data_files:
  - split: train
    path: data/wmt24pp/*.parquet
- config_name: bio-mqm
  data_files:
  - split: train
    path: data/bio-mqm/*.parquet
- config_name: wmt-biomed
  data_files:
  - split: train
    path: data/wmt-biomed/*.parquet
- config_name: wmt-terminology
  data_files:
  - split: train
    path: data/wmt-terminology/*.parquet
---

# WARMTH — WMT & multilingual MT evaluation data, consolidated

**WARMTH** gathers heterogeneous machine-translation **evaluation** resources —
the WMT metrics, general/news, terminology and biomedical shared tasks, plus
WMT24++, NTREX-128 and FLORES — into **one HuggingFace-loadable dataset** with a
single uniform schema. Every row is one **segment × system** translation (or,
for pure test sets, one **segment**), so source, reference and hypothesis sit
side by side, tagged with collection, release, language pair, segment id,
document id, domain and any human annotation (system DA, segment MQM/DA-SQM,
MQM error spans, post-edits, terminology constraints).

```python
from datasets import load_dataset

ds  = load_dataset("alvations/warmth")                 # everything (~3.71M rows)
m   = load_dataset("alvations/warmth", "wmt-metrics")  # WMT08-14
g   = load_dataset("alvations/warmth", "wmt-general")  # WMT24-25 submissions
flo = load_dataset("alvations/warmth", "flores-plus")  # FLORES-200
pp  = load_dataset("alvations/warmth", "wmt24pp")      # WMT24++ post-edits
bio = load_dataset("alvations/warmth", "bio-mqm")      # biomedical MQM spans
```

## Schema (one superset for every source)

| field | type | description |
|-------|------|-------------|
| `collection` | string | source family (see table below) |
| `release` | string | e.g. `WMT13`, `WMT23`, `WMT25`, `NTREX-128`, `FLORES-200`, `wmt24pp`, `bio-mqm-v2` |
| `year` | int32 \| null | edition year |
| `testset` | string \| null | e.g. `newstest2013`, `wmttest2025`, `flores-devtest`, `medline18`, `bio-mqm` |
| `domain` | string \| null | `news`, `speech`, `social`, `literary`, `biomedical`, `wikinews`, … |
| `langpair` | string | direction as distributed, e.g. `de-en`, `eng-spa`, `en-ja_JP`, `en-zh` |
| `src_lang` / `tgt_lang` | string | normalised language codes |
| `system` | string \| null | MT system id (`null` for pure test sets / references) |
| `segment_id` | int32 | 1-indexed segment (or document, for WMT25) |
| `doc_id` | string \| null | document id |
| `source` | string \| null | source segment |
| `reference` | string \| null | reference translation |
| `hypothesis` | string \| null | MT output (`null` for pure test sets) |
| `human_score` | float32 \| null | scalar human judgement |
| `human_score_level` | string \| null | `system`, `segment`, `segment:mqm`, `segment:da-sqm`, … |
| `annotations` | string \| null | JSON: MQM error spans, post-edit flags, terminology constraints, FLORES topic, … |

## Collections (~3.71M rows materialised)

| config | rows | hyp | human annotation | notes |
|--------|-----:|:---:|------------------|-------|
| `wmt-metrics` (WMT08–14) | 2,461,040 | ✅ | WMT14 system-DA | news metric task |
| `wmt-metrics-hi` (WMT22–23) | 75,530 | ✅ | ✅ segment MQM / DA-SQM | mt-metrics-eval slices |
| `wmt-general` (WMT24–25) | 372,031 | ✅ | — | official task submissions; WMT25 doc-level |
| `wmt24pp` | 54,890 | ✅ orig | post-edit reference | all 55 en→xx pairs |
| `bio-mqm` | 62,173 | ✅ | ✅ segment MQM error spans | biomedical |
| `wmt-biomed` | 13,691 | — | — | en↔fr biomedical test sets |
| `ntrex` (NTREX-128) | 259,610 | — | — | eng → 128 langs, doc ids |
| `flores-plus` (FLORES-200) | 407,827 | — | — | dev+devtest, 203 langs, URL+domain |
| `wmt-terminology` | 111 | — | term constraints | en-zh, `proper`/`random` term dicts |

Language coverage spans WMT `en↔{cs,de,es,fr,ru,zh,ja,uk,hi,he,is,…}`, 128 NTREX
languages, 203 FLORES languages, and 55 WMT24++ locale pairs.

## Provenance & how each collection is obtained

The build environment could **not** reach `statmt.org`, `huggingface.co`, or the
`mt-metrics-eval` GCS bucket (all firewalled), so the data was pulled from public
**GitHub** copies and re-assembled. Each adapter lives in [`adapters/`](adapters)
and maps its source onto the schema above.

- **wmt-metrics (WMT08–14)** — plain metric-task files in [`metric_data/`](metric_data).
- **wmt-metrics-hi (WMT22–23)** — `mt-metrics-eval` slices with segment MQM/DA-SQM
  (`wmt-conference/ErrorSpanAnnotation`, `NJUNLP/lost_in_the_src`). Partial vs the
  full GCS release.
- **wmt-general (WMT24–25)** — official `wmt-conference/wmt24-news-systems` (segment
  level) and `wmt-conference/wmt25-general-mt` (document level).
- **wmt24pp** — all 55 `en-xx_XX.jsonl` (`google/wmt24pp`).
- **bio-mqm** — `amazon-science/bio-mqm-dataset` (biomedical MQM error spans).
- **wmt-biomed** — `fyvo/WMT-Biomed-Test` (en↔fr biomedical parallel test sets).
- **ntrex** — `MicrosoftTranslator/NTREX` (CC BY-SA 4.0), doc ids from `DOCUMENT_IDS.tsv`.
- **flores-plus** — official `flores200_dataset/` (dev+devtest) with URL/domain/topic.
- **wmt-terminology** — `wmt-conference/wmt25-terminology` (en-zh docs + term dicts).

### Known gaps

- **WMT15–21 general/metrics** — need a full `mt-metrics-eval-v2` extract (GCS) or
  the `wmt21-news-systems` repo (idiosyncratic `florestest2021` layout, not yet
  parsed).
- **bouquet** — Meta BOUQuET is HF-only (`facebook/bouquet`); adapter ready, data
  not reachable here.
- WMT22/23 in `wmt-metrics-hi` are the human-scored slices, so they are kept out
  of `wmt-general` to avoid duplicating system outputs.

## (Re)building & extending

```bash
python build.py --list                        # registry + availability
python build.py                               # local collections (from metric_data/)
python build.py --collections ntrex --fetch   # git-clone then build
python build.py --collections flores-plus --fetch
python build.py --collections wmt24pp --root /path/to/wmt24pp
python build.py --collections wmt-metrics-hi --root /path/to/mt-metrics-eval-v2
python build.py --collections bouquet         # needs huggingface.co
python enrich.py --fetch --parquet-dir data/wmt-metrics   # WMT08-14 doc ids
```

Add a source by writing one `iter_records()` and registering it in
`adapters/__init__.py`.

## Publishing to the Hub

The `data/**` parquet plus this card's `configs:` block already are a loadable
dataset (every shard is < 40 MB). To push (run where `huggingface.co` is reachable
and authenticated):

```bash
python push_to_hub.py --repo-id alvations/warmth
```

## License

Redistributed for MT and MT-evaluation research; each source keeps its own terms
— WMT/statmt.org shared tasks (cite the relevant *Findings of the WMT* papers),
NTREX-128 (CC BY-SA 4.0), FLORES-200 (CC BY-SA 4.0), WMT24++ (Apache-2.0),
Bio-MQM (amazon-science, CC BY-NC 4.0), WMT-Biomed (fyvo), and the WMT
terminology task. Please cite the originating papers and respect upstream licenses.
