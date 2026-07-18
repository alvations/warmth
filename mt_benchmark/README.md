# Combined MT Shared-Task Benchmark

A single, uniform evaluation benchmark built by **combining the open-source
Machine Translation shared-task test sets** into one schema. **≈870K segments,
33 splits, 189 language pairs, 37 source / 139 target languages** — all
evaluation-only, **non-adversarial** shared-task data (WMT, terminology, FLORES,
NTREX, IWSLT, Multi30k, mTEDx, MTNT, Tatoeba, DiaBLa, MQM).

> **Handoff note (read me first).** This branch is a self-contained package: the
> frozen data (`data/*.parquet`), the builder that produced it
> (`build_mt_benchmark.py`), a loader (`load_benchmark.py`), and full provenance
> (`DATA_SOURCES.md`, `SCHEMA.md`, `stats.json`). If you are an agent picking
> this up: everything you need to *use*, *verify*, or *rebuild* the benchmark is
> in this directory. Start with `DATA_SOURCES.md` for where every row came from.

---

## Contents

| file | what it is |
|------|-----------|
| `data/<split>.parquet` | one parquet per split (git-lfs) |
| `data/all.parquet` | all splits concatenated into one table (git-lfs) |
| `build_mt_benchmark.py` | **the builder** — one loader per source, pulls & normalises everything |
| `load_benchmark.py` | load a split (pyarrow) or the whole thing as a `datasets.DatasetDict` |
| `DATA_SOURCES.md` | **where & how every dataset was pulled** (sacrebleu / HF / GitHub / local) |
| `SCHEMA.md` | the 15-column schema + invariants |
| `stats.json` | frozen per-split row counts, pair lists, language-pair totals |
| `requirements.txt` | `datasets`, `pyarrow`, `sacrebleu` |

## Quick start

```bash
pip install -r requirements.txt

# peek at a split
python load_benchmark.py --split wmt14 --head 3
python load_benchmark.py --split all   --head 3
```

```python
# load one split
import pyarrow.parquet as pq
tbl = pq.read_table("data/wmt24.parquet")          # 12,010 rows

# or every split as a HuggingFace DatasetDict
from load_benchmark import load_as_hf_datasetdict
dd = load_as_hf_datasetdict()
print(dd["flores200"][0]["source"], "->", dd["flores200"][0]["reference"])
```

Each row has the same 15 columns; the two you almost always want are `source`
and `reference`, with `pair` / `dataset` / `domain` for filtering. Full column
docs in [`SCHEMA.md`](SCHEMA.md).

## Splits (frozen snapshot)

| split | rows | pairs | domain | source |
|-------|-----:|------:|--------|--------|
| `wmt08` | 20,508 | 10 | news | sacrebleu |
| `wmt09` | 36,324 | 12 | news | sacrebleu |
| `wmt10` | 19,912 | 8 | news | sacrebleu |
| `wmt11` | 24,024 | 8 | news | sacrebleu |
| `wmt12` | 24,024 | 8 | news | sacrebleu |
| `wmt13` | 30,000 | 10 | news | sacrebleu |
| `wmt14` | 28,772 | 10 | news | sacrebleu |
| `wmt15` | 21,026 | 10 | news | sacrebleu |
| `wmt16` | 33,990 | 12 | news | sacrebleu |
| `wmt17` | 38,042 | 14 | news | sacrebleu |
| `wmt18` | 41,924 | 14 | news | sacrebleu |
| `wmt19` | 31,387 | 19 | news | sacrebleu |
| `wmt20` | 35,945 | 22 | news | sacrebleu |
| `wmt21` | 19,008 | 20 | news | sacrebleu |
| `wmt22` | 37,060 | 21 | news | sacrebleu |
| `wmt23` | 24,994 | 14 | news | sacrebleu |
| `wmt24` | 12,010 | 11 | news | sacrebleu |
| `wmt25` | 1,662 | 15 | doc-level | local clone (wmt25-general-mt) |
| `wmt25_humeval` | 3,039 | 13 | doc-level | local clone (wmt25-general-mt) |
| `wmt24pp` | 11,520 | 12 | post-edited | HF `google/wmt24pp` |
| `wmt_mqm` | 10,973 | 3 | mqm | HF `RicardoRei/wmt-mqm-human-evaluation` |
| `wmt_terminology_2023` | 7,484 | 3 | terminology | HF `zouharvi/wmt-terminology-2023` |
| `iwslt17` | 16,786 | 12 | spoken | sacrebleu |
| `multi30k_2016` | 3,000 | 3 | multimodal | sacrebleu |
| `multi30k_2017` | 2,000 | 2 | multimodal | sacrebleu |
| `multi30k_2018` | 3,213 | 3 | multimodal | sacrebleu |
| `mtedx_test` | 12,626 | 13 | spoken | sacrebleu |
| `flores200` | 30,360 | 30 | wikimedia | HF `facebook/flores` |
| `ntrex128` | 255,616 | 127 | news | GitHub zip (Microsoft NTREX) |
| `diabla` | 5,748 | 1 | dialogue | HF `rbawden/DiaBLa` |
| `tatoeba` | 18,000 | 23 | mixed | HF `Helsinki-NLP/tatoeba_mt` |
| `mtnt1_1_test` | 4,045 | 4 | noisy_ugc | sacrebleu |
| `mtnt2019` | 5,136 | 4 | noisy_ugc | sacrebleu |
| **total** | **870,158** | **189** | | |

Top language pairs by volume: `en-de` 50,349 · `en-cs` 50,043 · `de-en` 41,150
· `en-fr` 38,740 · `en-ru` 35,464 · `cs-en` 34,331 · `ru-en` 26,674 · `fr-en`
25,844 · `zh-en` 23,922 · `en-es` 23,536. Full breakdown in `stats.json`.

## Rebuilding from scratch

The parquet files are a frozen snapshot. To regenerate from upstream:

```bash
pip install -r requirements.txt
# optional, only for the wmt25 splits:
git clone https://github.com/wmt-conference/wmt25-general-mt
(cd wmt25-general-mt && git lfs install && git lfs pull)

python build_mt_benchmark.py --output mt_data           # everything
python build_mt_benchmark.py --sources wmt,flores200    # a subset
python build_mt_benchmark.py --pairs en-de,de-en        # filter by pair
```

No auth token is needed — every HF dataset used is public. `sacrebleu` downloads
and caches the WMT/IWSLT/Multi30k/MTNT/mTEDx files itself; NTREX is a plain
GitHub zip. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for the exact origin, loader
code, and citation of every split. **Note:** upstream sets can be re-released, so
a fresh rebuild may differ by a handful of rows from this snapshot — the numbers
above and in `stats.json` describe the committed parquet files.

## Scope: non-adversarial, evaluation-only

This benchmark is intentionally limited to **straight MT shared-task test sets**.
It excludes challenge/robustness/bias suites (`aces`, `winomt`, `mt_geneval`,
`halomi`) and contains **no adversarial, prompt-injection, unicode-attack, or
red-team content**. The only `challenge_type` values present are `standard`,
`terminology`, `dialogue`, `doc_level`, `humeval`, `humeval_control`, and
`mqm_human_eval`.

## Relationship to the rest of `warmth`

The repo root already hosts raw WMT metric-task data (`metric_data/`,
`wmt_metric_task_data_indices.py`) — system outputs + references for WMT08–14.
This `mt_benchmark/` directory is the modern, unified, source+reference
evaluation benchmark spanning WMT08→WMT25 and beyond, in one schema.
