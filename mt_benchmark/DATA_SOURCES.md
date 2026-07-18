# Data sources — where every dataset came from and how it was pulled

This benchmark is a **union of public MT shared-task evaluation sets**. Nothing
here was scraped, synthesised, or hand-written; every row comes from an official
or community-maintained MT test/dev/challenge set, pulled through one of four
mechanisms:

| mechanism | what it is | auth |
|-----------|-----------|------|
| **sacrebleu** | `sacrebleu` downloads & caches the official WMT/IWSLT/Multi30k/MTNT/mTEDx test files on first use | none |
| **HuggingFace `load_dataset`** | public HF datasets, pulled with `trust_remote_code=True` | none (public) |
| **GitHub zip** | direct download of a release zip, unzipped and parsed | none |
| **local clone** | a git-lfs repo cloned locally, JSONL parsed from disk | none |

Every loader lives in [`build_mt_benchmark.py`](build_mt_benchmark.py). The table
below maps each split to its exact origin. Row counts are from the snapshot in
[`stats.json`](stats.json) (≈870K rows total, 189 language pairs).

---

## 1. WMT news test sets — `wmt08` … `wmt24`  (via sacrebleu)

- **Origin:** the official WMT News Translation shared-task test sets
  (`newstestYYYY`), served through the `sacrebleu` package.
- **How pulled** (`pull_sacrebleu_sets`):
  ```python
  import sacrebleu
  pairs = sacrebleu.get_langpairs_for_testset("wmt14")     # all pairs for that year
  src   = sacrebleu.get_source_file("wmt14", "en-de")      # source file path
  refs  = sacrebleu.get_reference_files("wmt14", "en-de")  # reference file paths
  # source = lines of src; reference = lines of refs[0] (FIRST reference only)
  ```
- **testset names used:** `wmt08, wmt09, wmt10, wmt11, wmt12, wmt13, wmt14,
  wmt15, wmt16, wmt17, wmt18, wmt19, wmt20, wmt21, wmt22, wmt23, wmt24`.
- **Language pairs:** every pair sacrebleu offers for that year (WMT started at
  ~10 en↔{cs,de,es,fr,hu} pairs in 2008 and grew to 20+ pairs incl. ja/zh/ru by
  the 2020s).
- **domain** = `news`; **challenge_type** = `standard`.
- **Home page:** https://www.statmt.org/wmtNN/  (e.g. wmt24 →
  https://www2.statmt.org/wmt24/translation-task.html). Findings paper for
  wmt24: `aclanthology.org/2024.wmt-1.1` (Kocmi et al., 2024).
- **Note:** sacrebleu is the canonical, checksum-verified way to get these test
  sets — it is what MT metric papers use, so numbers are comparable.

## 2. MTNT — `mtnt1_1_test`, `mtnt2019`  (via sacrebleu)

- **Origin:** *Machine Translation of Noisy Text* (Reddit user-generated text).
  `mtnt1.1/test` (Michel & Neubig, EMNLP 2018) and `mtnt2019` (the WMT19
  Robustness-task extension, Li et al., WMT 2019).
- **How pulled:** `pull_sacrebleu_sets(["mtnt1.1/test", "mtnt2019"])` — same
  sacrebleu source/reference file mechanism as WMT.
- **Pairs:** en↔fr, en↔ja.  **domain** = `noisy_ugc`.
- **URLs:** https://www.cs.cmu.edu/~pmichel1/mtnt/ ·
  https://www.statmt.org/wmt19/robustness.html
- (MTNT is *noisy*, not *adversarial* — it is naturally occurring internet text,
  a standard robustness benchmark, not a crafted attack.)

## 3. IWSLT17 — `iwslt17`  (via sacrebleu)

- **Origin:** IWSLT 2017 evaluation campaign, TED-talk translation (spoken).
- **How pulled:** `pull_sacrebleu_sets(["iwslt17"])`.
- **domain** = `spoken`.  **URL:** https://iwslt.org/2017 (Cettolo et al., 2017).

## 4. Multi30k — `multi30k_2016/2017/2018`  (via sacrebleu)

- **Origin:** Multi30K image-caption translation test sets (Flickr captions).
- **How pulled:** `pull_sacrebleu_sets(["multi30k/2016","multi30k/2017","multi30k/2018"])`.
- **domain** = `multimodal` (the images are not included — this is the text side).
- **URL:** https://github.com/multi30k/dataset (Elliott et al., 2016,
  `aclanthology.org/W16-3210`).

## 5. mTEDx — `mtedx_test`  (via sacrebleu)

- **Origin:** multilingual TED-x talk translations.
- **How pulled:** `pull_sacrebleu_sets(["mtedx/test"])`.
- **URL:** https://www.openslr.org/100 (Salesky et al., Interspeech 2021).

## 6. WMT Terminology 2023 — `wmt_terminology_2023`  (HuggingFace)

- **Origin:** WMT 2023 shared task on MT with terminologies.
- **How pulled** (`pull_wmt_terminology`):
  ```python
  from datasets import load_dataset
  ds = load_dataset("zouharvi/wmt-terminology-2023", split="test", trust_remote_code=True)
  # source = row["src"]; reference = row["ref"]; pair = row["langs"]
  ```
- **Pairs (from the `langs` column):** `de-en`, `en-cs`, `zh-en` (3 pairs).
  ⚠️ Earlier code wrongly hardcoded `en-de`; the real pair is in `langs`.
- **domain** = `terminology`; **challenge_type** = `terminology`.
- **URL:** https://github.com/wmt-terminology-task/data-2023
  (`aclanthology.org/2023.wmt-1.54`, Zouhar et al., 2023).

## 7. WMT MQM human evaluation — `wmt_mqm`  (HuggingFace)

- **Origin:** MQM (Multidimensional Quality Metrics) human error annotations
  over WMT system outputs, aggregated by Ricardo Rei.
- **How pulled** (`pull_mqm_annotations`):
  ```python
  ds = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train", trust_remote_code=True)
  # source = row["src"]; reference = row["ref"]; pair = row["lp"]
  # dedup: keep first row per (src, pair)
  ```
- **Pairs:** `zh-en`, `en-de`, `en-ru`.  **domain** = `mqm_evaluated`.
- **URL:** https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation
  (`aclanthology.org/2021.wmt-1.73`, Freitag et al., 2021).

## 8. WMT24++ — `wmt24pp`  (HuggingFace, Google)

- **Origin:** Google's extended, **post-edited** WMT24 test sets (55 languages).
- **How pulled** (`pull_wmt24pp`): iterate 12 configs
  `en-de_DE, en-zh_CN, en-fr_FR, en-ja_JP, en-ru_RU, en-es_MX, en-cs_CZ,
  en-hi_IN, en-ko_KR, en-it_IT, en-pt_BR, en-ar_SA`:
  ```python
  ds = load_dataset("google/wmt24pp", name="en-de_DE", split="train", trust_remote_code=True)
  # source = row["source"]; reference = row["target"] (post-edited)
  # DROP rows where domain == "canary" or is_bad_source is truthy
  ```
- **Pairs:** 12 `en-X` pairs.  **domain** = the row's own domain.
- **URL:** https://huggingface.co/datasets/google/wmt24pp
  (`arxiv.org/abs/2502.12404`, Freitag et al., 2025).

## 9. WMT25 general + human-eval — `wmt25`, `wmt25_humeval`  (local clone)

- **Origin:** WMT 2025 General MT shared task (document-level, 31 pairs).
- **How pulled** (`pull_wmt25_general`, `pull_wmt25_humeval`): read JSONL from a
  **local clone** of the task repo — these are **not** on sacrebleu/HF:
  ```bash
  git clone https://github.com/wmt-conference/wmt25-general-mt
  cd wmt25-general-mt && git lfs install && git lfs pull
  ```
  ```python
  # wmt25:         data/wmt25-genmt.jsonl          -> source=src_text, reference=refs["refA"]["ref"]
  # wmt25_humeval: data/wmt25-genmt-humeval*.jsonl -> source=src_text, reference=tgt_text["refA"]
  ```
  Language codes are normalised (`de_DE→de`, `sr_Cyrl_RS→sr_Cyrl`). Only
  segments that carry a reference are kept (WMT25 also ships untranslated test
  inputs). `challenge_type` = `doc_level` / `humeval` / `humeval_control`.
- **URL:** https://github.com/wmt-conference/wmt25-general-mt (Kocmi et al., 2025).
- Default `data_dir` in the builder is `wmt25-general-mt/data` (override with the
  function arg). If the clone is missing, both loaders print `[skip]` and return
  nothing — the rest of the benchmark still builds.

## 10. FLORES-200 — `flores200`  (HuggingFace, Meta/NLLB)

- **Origin:** FLORES-200 `devtest`, the NLLB multilingual benchmark (Wikipedia).
- **How pulled** (`pull_flores200`): load English once, then each target; align
  by `id`:
  ```python
  eng = load_dataset("facebook/flores", "eng_Latn", split="devtest", trust_remote_code=True)
  eng_by_id = {r["id"]: r["sentence"] for r in eng}
  tgt = load_dataset("facebook/flores", "deu_Latn", split="devtest", trust_remote_code=True)
  # source = eng_by_id[row["id"]]; reference = row["sentence"]
  ```
- **Pairs:** English → **30 curated targets**
  (de, fr, es, ru, zh, ja, ko, ar, he, hi, bn, ta, te, ur, vi, th, id, tr, sw,
  yo, ig, ha, am, zu, so, el, pl, uk, cs, nl).
- **domain** = `wikimedia`.  **URL:** https://huggingface.co/datasets/facebook/flores
  (`arxiv.org/abs/2207.04672`, NLLB Team, 2022).
- Note: `facebook/flores` is **public** (not the gated `openlanguagedata/flores_plus`).

## 11. NTREX-128 — `ntrex128`  (GitHub zip, Microsoft)

- **Origin:** NTREX-128 — the WMT19 `newstest2019` set professionally
  translated by Microsoft into **128 languages**.
- **How pulled** (`pull_ntrex128`): download + unzip the GitHub archive, then
  align each target file to the shared English source line-by-line:
  ```
  https://github.com/MicrosoftTranslator/NTREX/archive/refs/heads/main.zip
  -> NTREX-128/newstest2019-*.txt   (one file per language, parallel by line)
  source = English file; reference = each other language's file
  ```
- **Pairs:** 127 `en-X` pairs → the biggest split (255,616 rows).
- **domain** = `news`.  **URL:** https://github.com/MicrosoftTranslator/NTREX
  (`aclanthology.org/2022.sumeval-1.4`, Federmann et al., 2022).

## 12. DiaBLa — `diabla`  (HuggingFace)

- **Origin:** DiaBLa bilingual English↔French dialogue corpus.
- **How pulled** (`pull_diabla`):
  ```python
  ds = load_dataset("rbawden/DiaBLa", split="test", trust_remote_code=True)
  # source = row["orig"]; reference = row["ref"]
  # direction from row["utterance_meta"] (english/french), default en->fr
  ```
- **Pairs:** en-fr.  **domain** = `dialogue`; **challenge_type** = `dialogue`.
- **URL:** https://huggingface.co/datasets/rbawden/DiaBLa
  (`aclanthology.org/2020.lrec-1.62`, Bawden et al., 2020).

## 13. Tatoeba MT Challenge — `tatoeba`  (HuggingFace, Helsinki-NLP)

- **Origin:** Tatoeba Translation Challenge test sets (crowdsourced pairs).
- **How pulled** (`pull_tatoeba`): iterate 30 English configs, **cap 1000 rows
  per config**:
  ```python
  ds = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-deu", split="test", trust_remote_code=True)
  # source = row["sourceString"]; reference = row["targetString"]
  # langs from sourceLang/targetlang (639-3), normalised to 2-letter
  ```
- **Pairs:** up to 30 en-X pairs (some configs empty at build time → 23 present).
- **domain** = `mixed`.  **URL:** https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt
  (`aclanthology.org/2020.wmt-1.139`, Tiedemann, 2020).

---

## Reproducing the exact snapshot

```bash
pip install -r requirements.txt
# (optional) for wmt25*:
git clone https://github.com/wmt-conference/wmt25-general-mt && \
  (cd wmt25-general-mt && git lfs install && git lfs pull)

# build everything
python build_mt_benchmark.py --output mt_data

# or a subset
python build_mt_benchmark.py --sources wmt,terminology --output mt_data
python build_mt_benchmark.py --pairs en-de,de-en --output mt_data_ende
```

Row counts can drift slightly over time: sacrebleu occasionally re-releases a
test file, and the HF datasets (tatoeba, wmt24pp, wmt-mqm) can be updated
upstream. The `data/*.parquet` files in this branch are the **frozen snapshot**
whose counts are recorded in `stats.json`; rebuild only if you want fresh
upstream data.

## What is deliberately NOT here

Excluded by design — these are challenge/robustness/bias sets, not
straight translation shared-task test sets, and were dropped from this benchmark:
`aces`, `winomt`, `mt_geneval`, `halomi`, and any hand-written adversarial /
prompt-injection / unicode-attack rows. This benchmark is **evaluation-only,
non-adversarial, shared-task MT data**.
