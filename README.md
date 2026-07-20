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
- config_name: wmt-terminology-2023
  data_files:
  - split: train
    path: data/wmt-terminology/wmt-terminology-2023.parquet
- config_name: wmt-mqm
  data_files:
  - split: train
    path: data/wmt-mqm/*.parquet
- config_name: iwslt
  data_files:
  - split: train
    path: data/iwslt/*.parquet
- config_name: multi30k
  data_files:
  - split: train
    path: data/multi30k/*.parquet
- config_name: multi30k-2016
  data_files:
  - split: train
    path: data/multi30k/multi30k-2016.parquet
- config_name: multi30k-2017
  data_files:
  - split: train
    path: data/multi30k/multi30k-2017.parquet
- config_name: multi30k-2018
  data_files:
  - split: train
    path: data/multi30k/multi30k-2018.parquet
- config_name: mtedx
  data_files:
  - split: train
    path: data/mtedx/*.parquet
- config_name: mtnt
  data_files:
  - split: train
    path: data/mtnt/*.parquet
- config_name: mtnt-1-1
  data_files:
  - split: train
    path: data/mtnt/mtnt-1-1.parquet
- config_name: mtnt-2019
  data_files:
  - split: train
    path: data/mtnt/mtnt-2019.parquet
- config_name: tatoeba
  data_files:
  - split: train
    path: data/tatoeba/*.parquet
- config_name: diabla
  data_files:
  - split: train
    path: data/diabla/*.parquet
---

# WARMTH — WMT & multilingual MT evaluation data, consolidated

> **Continuing this project?** Read **[`HANDOFF.md`](HANDOFF.md)** — it tells any agent exactly how to resume and finish the metric-score population (neural metrics need a GPU box that can reach huggingface.co). ~258k string scores are already checkpointed under `scores/`.

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

ds  = load_dataset("alvations/warmth")                 # everything (~4.40M rows)
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
| `langpair` | string | **raw** direction as distributed (preserves the source's exact codes), e.g. `de-en`, `eng-spa`, `eng_Latn-ace_Arab`, `en-ja_JP` |
| `src_lang` / `tgt_lang` | string | **canonical** base language subtag, consistent across every task (`eng`/`eng_Latn`/`en`→`en`, `ar_EG`→`ar`, `spa`→`es`); the raw code is always recoverable from `langpair` |
| `system` | string \| null | MT system id (`null` for pure test sets / references) |
| `segment_id` | int32 | 1-indexed segment (or document, for WMT25) |
| `doc_id` | string \| null | document id |
| `source` | string \| null | source segment |
| `reference` | string \| null | reference translation |
| `hypothesis` | string \| null | MT output (`null` for pure test sets) |
| `human_score` | float32 \| null | scalar human judgement |
| `human_score_level` | string \| null | `system`, `segment`, `segment:mqm`, `segment:da-sqm`, … |
| `annotations` | string \| null | JSON: MQM error spans, post-edit flags, terminology constraints, FLORES topic, … |
| `row_hash` | string | 16-hex content fingerprint (blake2b-64) of all other fields; **unique across the whole dataset** — dedup key, guards against inflation |

## Collections (~4.40M rows materialised)

| config | rows | hyp | human annotation | notes |
|--------|-----:|:---:|------------------|-------|
| `wmt-metrics` (WMT08–14) | 2,461,040 | ✅ | WMT14 system-DA | news metric task |
| `wmt-general` (WMT15–25) | 980,756 | ✅ 21/24/25 | ✅ WMT24–25 ESA (143,528) | WMT21/24/25 submissions+ESA humeval; WMT15–20 test sets |
| `wmt-metrics-hi` (WMT22–23) | 75,530 | ✅ | ✅ segment MQM / DA-SQM | mt-metrics-eval slices |
| `flores-plus` (FLORES-200) | 407,827 | — | — | dev+devtest, 203 langs, URL+domain |
| `ntrex` (NTREX-128) | 259,610 | — | — | eng → 128 langs, doc ids |
| `bio-mqm` | 62,173 | ✅ | ✅ segment MQM error spans | biomedical |
| `wmt24pp` | 54,890 | ✅ orig | post-edit reference | all 55 en→xx pairs |
| `tatoeba` | 18,000 | — | — | Tatoeba-MT, 23 pairs (mixed) |
| `iwslt` | 16,786 | — | — | IWSLT17 TED talks (spoken) |
| `wmt-biomed` | 13,691 | — | — | en↔fr biomedical test sets |
| `mtedx` | 12,626 | — | — | mTEDx spoken |
| `wmt-mqm` | 10,973 | — | — | WMT MQM-evaluated segments |
| `mtnt` | 9,181 | — | — | MTNT noisy user-generated text |
| `multi30k` | 8,213 | — | — | Multi30k 2016–18 image captions |
| `wmt-terminology` | 7,595 | — | term constraints | WMT23 (3 pairs) + WMT25 en-zh |
| `diabla` | 5,748 | — | — | bilingual dialogue |
| **total** | **4,404,639** | | | 16 collections, every row_hash unique |

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
- **wmt-general (WMT21, 24–25)** — official `wmt-conference/wmt21-news-systems`
  and `wmt24-news-systems` (segment level) + `wmt25-general-mt` (document level),
  plus the WMT24/25 `*-genmt-humeval` ESA human scores (WMT25's is Git-LFS, fetched
  via the media.githubusercontent endpoint).
- **wmt24pp** — all 55 `en-xx_XX.jsonl` (`google/wmt24pp`).
- **bio-mqm** — `amazon-science/bio-mqm-dataset` (biomedical MQM error spans).
- **wmt-biomed** — `fyvo/WMT-Biomed-Test` (en↔fr biomedical parallel test sets).
- **ntrex** — `MicrosoftTranslator/NTREX` (CC BY-SA 4.0), doc ids from `DOCUMENT_IDS.tsv`.
- **flores-plus** — official `flores200_dataset/` (dev+devtest) with URL/domain/topic.
- **wmt-terminology** — `wmt-conference/wmt25-terminology` (en-zh docs + term dicts).
- **wmt-general WMT15–20, wmt-mqm, iwslt, multi30k, mtedx, mtnt, tatoeba, diabla,
  wmt-terminology-2023** — merged from this repo's **`mt-eval-benchmark`** branch
  (source+reference test sets pulled there via sacrebleu / HuggingFace / GitHub),
  fetched as Git-LFS parquet via `media.githubusercontent.com` and mapped onto the
  superset schema. All provenance (paper, url, citation, loader) kept in `annotations`.

### Known gaps

- **WMT15–20** — now included as source+reference test sets, merged from this
  repo's `mt-eval-benchmark` branch (sacrebleu-pulled), so WMT08–25 is complete
  for source/reference. System outputs + human scores for WMT15–20 still live only
  on the firewalled GCS bucket (run `wmt-metrics-hi` with a full mt-metrics-eval
  extract to add them).
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

## Metric scores (`score_population.py` + `merge_scores.py`)

Automatic MT-metric scores are populated per row via **[lightyear](https://github.com/alvations/lightyear)**
(BLEU, CHRF, TER, COMET, CometKiwi, MetricX-24, MetricX-24-QE, BERTScore,
SentenceBERTScore, PreCOMET difficulty, Sentinel-src) into three key-spaces so
neural models never re-score identical text:

| space | key | metrics | applies to |
|-------|-----|---------|-----------|
| `row` | `row_hash` | `bleu chrf ter comet metricx bertscore sentbert` (ref-based) · `cometkiwi_hyp metricxqe_hyp` (QE on hyp) | rows with a hypothesis |
| `refqe` | hash(source‖reference) | `cometkiwi_ref metricxqe_ref` (QE on the reference itself) | any row with source+reference |
| `src` | hash(source) | `difficulty_src sentinel_src` | any row with a source |

**Reference-only rows** (test sets, no hypothesis) therefore get the QE-on-reference
and source-only scores; the reference-based / hyp-QE metrics are left **NaN** (null).

The scorer is **resumable and checkpointing**: scores append to `scores/<space>.jsonl`
(`{"k","m","s"}`, fsync-flushed every `--flush-every`), already-computed `(key, metric)`
pairs are skipped on restart (metric-granular resume), and every `--commit-every` rows
the `scores/` dir is git-committed (and pushed with `--push`). SIGINT/SIGTERM flush
and exit cleanly — interrupt any time and re-run to continue.

```bash
pip install "git+https://github.com/alvations/lightyear"      # neural metrics also need torch+GPU+HF
python score_population.py --preset fast                        # BLEU/CHRF/TER, no GPU/network
python score_population.py --preset all --commit-every 50000 --push
python merge_scores.py                                          # pivot -> data_scores/<coll>/<shard>.parquet (row_hash-joined columns)
```

## License

Redistributed for MT and MT-evaluation research; each source keeps its own terms
— WMT/statmt.org shared tasks (cite the relevant *Findings of the WMT* papers),
NTREX-128 (CC BY-SA 4.0), FLORES-200 (CC BY-SA 4.0), WMT24++ (Apache-2.0),
Bio-MQM (amazon-science, CC BY-NC 4.0), WMT-Biomed (fyvo), and the WMT
terminology task. Please cite the originating papers and respect upstream licenses.
