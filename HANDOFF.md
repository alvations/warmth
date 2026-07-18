# WARMTH — handoff for finishing the metric-score population

**Read this first if you are an agent picking this up.** The dataset is built and
committed; what remains is populating automatic MT-metric scores for every row.
The string metrics (BLEU/CHRF/TER) may already be running/partly done; the
**neural metrics still need a machine that can reach `huggingface.co` and
(ideally) a GPU** — the environment this repo was assembled in is firewalled off
from statmt.org / huggingface.co / the mt-metrics-eval GCS bucket, so it can only
run the sacrebleu string metrics.

## What is already done

- `data/<collection>/<shard>.parquet` — **4,404,639 rows, 16 collections**, one
  uniform 18-column schema (see `README.md` and `warmth_schema.py`). Every row has
  a unique `row_hash` (blake2b-64 content fingerprint) — the join key for scores.
- Adapters in `adapters/` rebuild any collection; `build.py --list` shows them.
- Scoring pipeline: `score_population.py` (compute) + `merge_scores.py` (pivot to
  columns). Documented in `README.md` under *Metric scores*.

## What remains: finish scoring, then merge

### 1. Install lightyear (and ryokai) + neural deps
```bash
pip install "git+https://github.com/alvations/lightyear"   # torch, transformers, sacrebleu, sentence-transformers
# GPU strongly recommended for COMET / MetricX / BERTScore
# first run downloads models from huggingface.co (must be reachable)
```

### 2. Run the scorer to completion (resumable)
```bash
# all metrics, all rows; commits+pushes gz checkpoints every 50k scores
python score_population.py --preset all --commit-every 50000 --push
```
- It is **resumable and metric-granular**: on restart it reads the committed
  `scores/<space>.NNNN.jsonl.gz` parts + the live `scores/<space>.jsonl` and skips
  every `(key, metric)` already computed. Just re-run the same command to continue.
- Interrupt any time (Ctrl-C / SIGTERM) — it flushes and exits cleanly.
- Presets: `fast` (bleu/chrf/ter, no GPU/network), `neural`, `qe`, `all`. Or
  `--metrics comet,cometkiwi_hyp,cometkiwi_ref,metricx,metricxqe_hyp,metricxqe_ref,difficulty_src,sentinel_src,bertscore,sentbert`.
- Scope with `--collections wmt-metrics wmt-general …` to parallelise across boxes.

### 3. What gets scored (three key-spaces; no wasted neural compute)
| space | key | metrics | applies to |
|-------|-----|---------|-----------|
| `row` | `row_hash` | bleu, chrf, ter, comet, metricx, bertscore, sentbert (ref-based); cometkiwi_hyp, metricxqe_hyp (QE on hyp) | rows **with a hypothesis** |
| `refqe` | hash(source‖reference) | cometkiwi_ref, metricxqe_ref (QE on the reference) | any row with source+reference |
| `src` | hash(source) | difficulty_src, sentinel_src | any row with a source |

**Reference-only rows** (test sets, `hypothesis` is null) get the `refqe` +
`src` scores; the reference-based / hyp-QE metrics are correctly left **NaN**.

### 4. Merge scores into columns
```bash
python merge_scores.py            # -> data_scores/<collection>/<shard>.parquet, row_hash-joined, 1:1 with data/
# or fold columns straight into data/:
python merge_scores.py --inplace
```
`data_scores/` is gitignored (derived); commit it explicitly if you want the
scored columns in the repo, or publish via `push_to_hub.py`.

## Checkpoint format (so you can trust resume)
`scores/<space>.jsonl` (working, gitignored) and `scores/<space>.NNNN.jsonl.gz`
(committed, append-only parts). Each line: `{"k": <key>, "m": <metric>, "s": <score|null>}`.
Losing the working file loses at most the scores since the last commit; the gz
parts in the repo are the durable record.

## Publishing
The `data/**` parquet + `README.md` `configs:` block already are a HuggingFace
dataset. `python push_to_hub.py --repo-id alvations/warmth` uploads it (needs an
authenticated, reachable `huggingface.co`).

## Known gaps (need an unfirewalled machine)
- WMT15–20 **system outputs + human scores** (source/reference are already in) —
  run `adapters/wmt_metrics_hi` against a full `mt-metrics-eval-v2` extract.
- BOUQuET (`facebook/bouquet`, HF-only) — `adapters/bouquet` is ready.
- All neural metric scores (this document's main task).

## Map of the code
- `warmth_schema.py` — the 18-col Record + `row_hash` + language canonicalisation.
- `adapters/` — one `iter_records()` per source; `adapters/__init__.py` is the registry.
- `build.py` — materialise collections to parquet; `--list`, `--print-config`.
- `enrich.py` — WMT08–14 doc ids (SGML) + segment human scores.
- `normalize_data.py` / `migrate_and_dedup.py` — lang canonicalisation, `row_hash`, global dedup.
- `score_population.py` / `merge_scores.py` — this task.
- `push_to_hub.py` — publish.
