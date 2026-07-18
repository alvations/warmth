# Schema

Every row in every split — and in the combined `all` table — has the **same 15
columns**. All string columns are guaranteed to be `str` (never `None`), and
`source` / `reference` are guaranteed non-empty after the empty-row filter.

| # | column | type | description |
|---|--------|------|-------------|
| 1 | `source` | str | Source-language sentence/segment to translate. Never empty. |
| 2 | `reference` | str | Gold human reference translation. Never empty. See per-dataset note below on what "reference" means. |
| 3 | `source_lang` | str | Source language code (2-letter where possible; may carry a script tag for WMT25, e.g. `sr_Cyrl`). |
| 4 | `target_lang` | str | Target language code. |
| 5 | `pair` | str | `"{source_lang}-{target_lang}"`, e.g. `en-de`. |
| 6 | `dataset` | str | Short split key, e.g. `wmt14`, `flores200`, `wmt_terminology_2023`. Matches the split name. |
| 7 | `domain` | str | Text domain: `news`, `spoken`, `multimodal`, `noisy_ugc`, `terminology`, `dialogue`, `wikimedia`, `mqm_evaluated`, `mixed`, or a WMT25 doc domain. |
| 8 | `year` | str | 4-digit year of the shared task / release (may be `""` for a few sacrebleu sets). |
| 9 | `idx` | int | Row index within its `(dataset, pair)` group as pulled. |
| 10 | `challenge_type` | str | One of `standard`, `terminology`, `dialogue`, `doc_level`, `humeval`, `humeval_control`, `mqm_human_eval`. **No adversarial/robustness-attack types exist in this benchmark.** |
| 11 | `dataset_full_name` | str | Human-readable dataset name (provenance). |
| 12 | `paper` | str | Citing paper / authors (provenance). |
| 13 | `dataset_url` | str | Canonical URL for the source (provenance). |
| 14 | `citation` | str | Short citation handle, e.g. `aclanthology.org/2024.wmt-1.1` (provenance). |
| 15 | `loader` | str | How the row was pulled: `sacrebleu`, `load_dataset('...')`, `local: ...`, or `download zip, parse TXT files` (provenance). |

## What `reference` means per source

Most sources give a straightforward human reference translation. A few are worth
calling out:

- **WMT sacrebleu sets (wmt08–wmt24):** the **first** official reference only.
  Multi-reference WMT years keep just `ref_files[0]`.
- **wmt24pp:** the `target` column, which is a **post-edited** human translation
  (not a from-scratch reference). `is_bad_source` rows and the `canary` domain
  are dropped.
- **wmt_mqm:** the `ref` field; deduplicated to **one row per `(source, pair)`**
  (the raw dataset has many MQM annotations per segment).
- **wmt25 / wmt25_humeval:** `refA` — the primary human reference. WMT25 is
  document-level; only segments that carry a reference are kept.
- **flores200:** English `devtest` is the source; each target language's
  `devtest` sentence (aligned by `id`) is the reference. All pairs are `en-X`.
- **ntrex128:** English `newstest2019` is the source; each of the 127 other
  language files is a reference, aligned line-by-line. All pairs are `en-X`.

## Invariants

- `pair == f"{source_lang}-{target_lang}"` for every row.
- `source.strip()` and `reference.strip()` are both non-empty.
- The `all` table equals the row-wise concatenation of every per-split parquet
  (except `all.parquet` itself).
