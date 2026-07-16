---
pretty_name: WARMTH — WMT Metrics-task translations, references & human judgements
license: other
language:
- en
- de
- fr
- es
- cs
- ru
- hi
- hu
task_categories:
- translation
tags:
- machine-translation
- wmt
- statmt
- mt-evaluation
- mt-metrics
- human-evaluation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/wmt*.parquet
- config_name: wmt08
  data_files:
  - split: train
    path: data/wmt2008.parquet
- config_name: wmt09
  data_files:
  - split: train
    path: data/wmt2009.parquet
- config_name: wmt10
  data_files:
  - split: train
    path: data/wmt2010.parquet
- config_name: wmt11
  data_files:
  - split: train
    path: data/wmt2011.parquet
- config_name: wmt12
  data_files:
  - split: train
    path: data/wmt2012.parquet
- config_name: wmt13
  data_files:
  - split: train
    path: data/wmt2013.parquet
- config_name: wmt14
  data_files:
  - split: train
    path: data/wmt2014.parquet
---

# WARMTH — WMT data in Python

**WARMTH** consolidates the [WMT](https://www.statmt.org/) (Workshop / Conference
on Machine Translation) **Metrics shared-task** distributions — the `newstest`
source segments, human reference translations, every participating system's
output, and the human evaluation scores — into **one HuggingFace-loadable
dataset** with a single, uniform schema.

Each row is one **segment × system** translation, so the source, reference and
hypothesis for a given segment sit side by side, tagged with the edition,
language pair, system id and (where available) the human judgement.

```python
from datasets import load_dataset

# everything (WMT08–WMT14), ~2.46M rows
ds = load_dataset("alvations/warmth", split="train")

# one edition
ds14 = load_dataset("alvations/warmth", "wmt14", split="train")

ds[0]
# {'year': 2008, 'wmt': 'WMT08', 'testset': 'newstest2008',
#  'langpair': 'cz-en', 'src_lang': 'cs', 'tgt_lang': 'en',
#  'system': 'dcu', 'segment_id': 1, 'doc_id': None,
#  'source': '...', 'reference': '...', 'hypothesis': '...',
#  'human_score': None, 'human_score_level': None}
```

## Schema

| field | type | description |
|-------|------|-------------|
| `year` | int32 | WMT edition year, e.g. `2014` |
| `wmt` | string | edition tag, e.g. `"WMT14"` |
| `testset` | string | test-set name, e.g. `"newstest2014"` |
| `langpair` | string | translation direction as distributed on disk, e.g. `"de-en"` |
| `src_lang` | string | normalised source language code (`cz`→`cs`) |
| `tgt_lang` | string | normalised target language code |
| `system` | string | participating system id, e.g. `"uedin-wmt14.3025"` |
| `segment_id` | int32 | 1-indexed segment / line number within the test set |
| `doc_id` | string \| null | document id — see *Known limitations* |
| `source` | string \| null | source segment |
| `reference` | string \| null | reference translation |
| `hypothesis` | string | the system's output segment |
| `human_score` | float32 \| null | human judgement for this system/segment |
| `human_score_level` | string \| null | granularity of `human_score`: `"system"`, `"segment"`, or `null` |

## Coverage

| edition | rows | source side | reference | human score |
|---------|-----:|:-----------:|:---------:|:-----------:|
| WMT08 | 180,488 | ✅ (multi-parallel) | ✅ | — |
| WMT09 | 214,617 | ✅ (multi-parallel) | ✅ | — |
| WMT10 | 357,984 | ✅ (multi-parallel) | ✅ | — |
| WMT11 | 516,516 | ✅ (`sources/`) | ✅ | — |
| WMT12 | 309,309 | ✅ (`sources/`) | ✅ | — |
| WMT13 | 567,000 | ✅ (`sources/`) | ✅ | — |
| WMT14 | 315,126 | — (not distributed locally) | ✅ | ✅ system-level DA |
| **total** | **2,461,040** | 2,139,812 rows | all rows | 315,126 rows |

Language pairs span `en↔{cs, de, es, fr, ru, hi, hu}` and a few non-English
directions (e.g. `de-es`), plus WMT10's `xx-en` system-combination track.

## How it is built

The raw WMT files live under [`metric_data/`](metric_data) exactly as
distributed by statmt.org. The naming and layout drift year to year (references
per-language vs per-direction, a dedicated `sources/` dir only from 2011, Czech
written `cz` before 2012 and `cs` after, different filename token orders for
system outputs). [`warmth_core.py`](warmth_core.py) hides that drift behind one
generator, `iter_records()`, that line-aligns source/reference/hypothesis and
recovers the bare system id. [`build.py`](build.py) streams that into the
`data/*.parquet` shards this card points at.

```bash
python build.py                      # rebuild data/wmt2008.parquet … wmt2014.parquet
python warmth_core.py                # print coverage stats as JSON
python build.py --push-to-hub alvations/warmth   # needs HF auth + network
```

### Enriching `doc_id` and segment-level `human_score`

Document ids and most human judgements are not in the plain-text distributions
this dataset is built from — they live in the WMT `test.tgz` **SGML** files and
in separate manual-evaluation packages on statmt.org. [`enrich.py`](enrich.py)
merges them into the parquet in place (schema unchanged), so run it wherever
those files are reachable:

```bash
# one command: download + extract the WMT test tarballs, then fill doc_id
python enrich.py --fetch --parquet-dir data
python enrich.py --fetch --years 2013 2014          # just some editions
python enrich.py --fetch --base-url https://mirror/wmt{yy}/{name}   # if statmt is blocked

# ...or point at .sgm files you already extracted
python enrich.py --sgm-dir /path/to/wmt_test_sgm --parquet-dir data
# add per-segment human scores from a year|langpair|system|segment_id|score file
python enrich.py --human-scores da-seg-scores-2013.tsv --parquet-dir data
python enrich.py --self-test         # verify the SGM parser, no files needed
```

`--fetch` downloads through the standard `HTTPS_PROXY` / CA-bundle env and caches
the tarballs, so re-runs are incremental.

The segment order of the `.txt` files matches the SGML `<seg>` order exactly
(verified against `newstest2013`), so doc ids map cleanly onto `segment_id`.

Building in-memory without the parquet shards:

```python
from datasets import Dataset
from warmth_core import iter_records
ds = Dataset.from_generator(lambda: (r._asdict() for r in iter_records()))
```

## Known limitations & how to extend

- **Document ids (`doc_id`) are `null`.** The metric-task packages here are the
  plain line-aligned `.txt` distributions, which carry no document boundaries.
  Doc ids live only in the original SGML (`*-src.sgm` / `*-ref.sgm`) packages on
  statmt.org. Run [`enrich.py --sgm-dir`](enrich.py) against those files to
  populate the field (no schema change).
- **Human judgements are sparse.** Only WMT14 ships evaluation scores in this
  repo, at **system level** (Direct Assessment). Segment-level human judgements
  (ranking / DA / MQM) for the other editions are separate statmt.org packages;
  merge them with [`enrich.py --human-scores`](enrich.py), which sets
  `human_score` / `human_score_level` (`"segment"`). Multiple annotations per
  segment can be stored by emitting multiple rows or extending the schema with
  an `annotations` list.
- **WMT14 has no source side locally.** WMT14 moved to per-direction (bilingual)
  test sets, so the source cannot be reconstructed from a reverse-direction
  reference (different sentence counts). Add a `sources/` dir to
  `metric_data/WMT14/` to populate it.

To ingest more editions, drop the statmt.org files under
`metric_data/WMT<NN>/` in the same layout and add a one-line entry to the
`YEARS` table in `warmth_core.py`; system/langpair discovery is automatic.

## Provenance & license

Data originates from the WMT Metrics shared tasks (2008–2014), distributed via
[statmt.org](https://www.statmt.org/). It is redistributed here for machine-
translation and MT-evaluation research; please cite the relevant WMT *Findings
of the … Workshop on Statistical Machine Translation* papers and respect the
original terms of the shared-task data.
