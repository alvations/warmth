#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Populate automatic MT-metric scores for every WARMTH row, resumably.

Metrics come from **lightyear** (https://github.com/alvations/lightyear) — BLEU,
CHRF, TER, COMET, CometKiwi, MetricX-24, MetricX-24-QE, BERTScore,
SentenceBERTScore, PreCOMET difficulty, Sentinel-src — and optionally **ryokai**
(https://github.com/alvations/ryokai). Install:

    pip install "git+https://github.com/alvations/lightyear"
    # neural metrics also need torch + transformers + a reachable huggingface.co

Three score **key-spaces**, so expensive neural models never re-score identical
text:

  * ``row``   keyed by ``row_hash`` — reference-based metrics on (hyp, ref) and
              QE metrics on the **hypothesis** (hyp, src). Only rows with a
              hypothesis.  columns: bleu, chrf, ter, comet, metricx, bertscore,
              sentbert, cometkiwi_hyp, metricxqe_hyp
  * ``refqe`` keyed by hash(src⋁ref) — QE metrics on the **reference** itself
              (ref, src): reference quality. Any row with src+ref, deduped.
              columns: cometkiwi_ref, metricxqe_ref
  * ``src``   keyed by hash(src) — source-only metrics on the **source**, deduped.
              columns: difficulty_src, sentinel_src

Checkpointing / resume / commit:
  * scores are appended to ``scores/<space>.jsonl`` as ``{"k","m","s"}`` lines and
    flushed every ``--flush-every`` rows — crash-safe, append-only.
  * on start, existing lines are read so already-computed (key, metric) pairs are
    skipped: **resume is automatic and metric-granular** (adding a new metric only
    computes the missing ones).
  * every ``--commit-every`` rows the ``scores/`` dir is git-committed and (with
    ``--push``) pushed, so partials land in the repo as we chug along.
  * SIGINT/SIGTERM flush the buffer and exit cleanly.

Run ``merge_scores.py`` afterwards to pivot the JSONL into score **columns**
joined onto the rows by ``row_hash``.

Examples::

    python score_population.py --preset fast                 # BLEU/CHRF/TER (no GPU/network)
    python score_population.py --preset fast --commit-every 50000 --push
    python score_population.py --metrics comet,cometkiwi_hyp,cometkiwi_ref,metricx,metricxqe_hyp,metricxqe_ref,difficulty_src
    python score_population.py --preset all --collections wmt-metrics wmt-general
"""

import argparse
import glob
import hashlib
import io
import json
import os
import signal
import subprocess
import sys

import pyarrow.parquet as pq

SCORES_DIR = "scores"

# name -> (space, needs, key_of, lightyear_metric_key, factory_spec)
#   space  : "row" | "refqe" | "src"
#   needs  : tuple of fields that must be non-empty for this metric to apply
#   call   : how to feed the lightyear metric's .score(hyp, ref, src)
# factory_spec is (ClassName, kwargs) resolved lazily so the "fast" preset never
# imports torch.
METRICS = {
    # reference-based, keyed by row_hash
    "bleu":       dict(space="row", needs=("hypothesis", "reference"), key="bleu_score",
                       cls="BLEUScore", kw={}, call="ref"),
    "chrf":       dict(space="row", needs=("hypothesis", "reference"), key="chrf_score",
                       cls="CHRFScore", kw={}, call="ref"),
    "ter":        dict(space="row", needs=("hypothesis", "reference"), key="ter_score",
                       cls="TERScore", kw={}, call="ref"),
    "comet":      dict(space="row", needs=("hypothesis", "reference", "source"), key="comet_score",
                       cls="COMETScore", kw={}, call="ref"),
    "metricx":    dict(space="row", needs=("hypothesis", "reference", "source"), key="metricx_score",
                       cls="MetricXScore", kw={}, call="ref"),
    "bertscore":  dict(space="row", needs=("hypothesis", "reference"), key="bert_score",
                       cls="BERTScore", kw={}, call="ref"),
    "sentbert":   dict(space="row", needs=("hypothesis", "reference"), key="sentbert_score",
                       cls="SentenceBERTScore", kw={}, call="ref"),
    # QE on the hypothesis, keyed by row_hash
    "cometkiwi_hyp": dict(space="row", needs=("hypothesis", "source"), key="cometkiwi_score",
                          cls="COMETScore", kw={"qe": True}, call="qe_hyp"),
    "metricxqe_hyp": dict(space="row", needs=("hypothesis", "source"), key="metricxqe_score",
                          cls="MetricXScore", kw={"qe": True}, call="qe_hyp"),
    # QE on the reference (reference quality), keyed by hash(src|ref)
    "cometkiwi_ref": dict(space="refqe", needs=("reference", "source"), key="cometkiwi_score",
                          cls="COMETScore", kw={"qe": True}, call="qe_ref"),
    "metricxqe_ref": dict(space="refqe", needs=("reference", "source"), key="metricxqe_score",
                          cls="MetricXScore", kw={"qe": True}, call="qe_ref"),
    # source-only, keyed by hash(src)
    "difficulty_src": dict(space="src", needs=("source",), key="difficulty_score",
                           cls="DifficultyScore", kw={}, call="src"),
    "sentinel_src":   dict(space="src", needs=("source",), key="sentinel_src_score",
                           cls="SentinelSrcScore", kw={}, call="src"),
}

PRESETS = {
    "fast": ["bleu", "chrf", "ter"],
    "neural": ["comet", "metricx", "bertscore", "sentbert",
               "cometkiwi_hyp", "metricxqe_hyp"],
    "qe": ["cometkiwi_hyp", "metricxqe_hyp", "cometkiwi_ref", "metricxqe_ref",
           "difficulty_src", "sentinel_src"],
    "all": list(METRICS.keys()),
}


def _h(*parts):
    b = "\x1f".join(parts).encode("utf-8")
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def key_for(space, row):
    if space == "row":
        return row["row_hash"]
    if space == "refqe":
        return _h(row.get("source") or "", row.get("reference") or "")
    if space == "src":
        return _h(row.get("source") or "")
    raise ValueError(space)


# --------------------------------------------------------------------------
# lightyear metric instances (lazy, cached so cometkiwi is built once)
# --------------------------------------------------------------------------
_INSTANCES = {}


def get_metric(name):
    spec = METRICS[name]
    cache_key = (spec["cls"], tuple(sorted(spec["kw"].items())))
    if cache_key not in _INSTANCES:
        import lightyear.metrics as M
        cls = getattr(M, spec["cls"])
        _INSTANCES[cache_key] = cls(**spec["kw"])
    return _INSTANCES[cache_key]


def score_one(name, row):
    """Return a float score (or None) for one metric on one row."""
    spec = METRICS[name]
    m = get_metric(name)
    hyp = (row.get("hypothesis") or "").strip()
    ref = (row.get("reference") or "").strip()
    src = (row.get("source") or "").strip()
    try:
        if spec["call"] == "ref":
            res = m.score(hyp=hyp, ref=ref, src=src or None)
        elif spec["call"] == "qe_hyp":
            res = m.score(hyp=hyp, ref=None, src=src)
        elif spec["call"] == "qe_ref":
            res = m.score(hyp=ref, ref=None, src=src)
        else:  # src-only
            res = m.score(hyp=src, ref=None, src=src)
        return float(res[spec["key"]]["score"])
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
        sys.stderr.write("  score fail %s: %s\n" % (name, str(exc)[:80]))
        return None


# --------------------------------------------------------------------------
# checkpoint store
# --------------------------------------------------------------------------

def load_done(spaces):
    """Return {space: {(key, metric)}} already computed, from scores/*.jsonl."""
    done = {sp: set() for sp in spaces}
    for sp in spaces:
        path = os.path.join(SCORES_DIR, sp + ".jsonl")
        if not os.path.isfile(path):
            continue
        with io.open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    done[sp].add((d["k"], d["m"]))
                except Exception:  # noqa: BLE001
                    continue
    return done


class Writer:
    def __init__(self):
        os.makedirs(SCORES_DIR, exist_ok=True)
        self._fh = {}

    def _f(self, space):
        if space not in self._fh:
            self._fh[space] = io.open(os.path.join(SCORES_DIR, space + ".jsonl"),
                                      "a", encoding="utf-8")
        return self._fh[space]

    def write(self, space, key, metric, score):
        self._f(space).write(json.dumps({"k": key, "m": metric, "s": score}) + "\n")

    def flush(self):
        for fh in self._fh.values():
            fh.flush()
            os.fsync(fh.fileno())

    def close(self):
        self.flush()
        for fh in self._fh.values():
            fh.close()
        self._fh = {}


def git_commit_push(n, push):
    subprocess.run(["git", "add", SCORES_DIR], check=False)
    r = subprocess.run(["git", "commit", "-q", "-m",
                        "scores: checkpoint %d scored keys" % n], check=False)
    if r.returncode == 0 and push:
        subprocess.run(["git", "push", "-q", "origin", "HEAD"], check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--metrics", default=None, help="comma list overriding --preset")
    ap.add_argument("--collections", nargs="*", default=None,
                    help="restrict to these data/<collection> dirs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--flush-every", type=int, default=2000)
    ap.add_argument("--commit-every", type=int, default=50000)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max keys to score (testing)")
    args = ap.parse_args()

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    elif args.preset:
        metrics = PRESETS[args.preset]
    else:
        ap.error("pass --preset or --metrics")
    for m in metrics:
        if m not in METRICS:
            ap.error("unknown metric %r (known: %s)" % (m, ", ".join(METRICS)))
    spaces = sorted({METRICS[m]["space"] for m in metrics})

    done = load_done(spaces)
    seen_this_run = {sp: set() for sp in spaces}  # avoid dup within one run
    writer = Writer()

    stop = {"flag": False}

    def _sig(signum, frame):
        sys.stderr.write("\n[signal %d] flushing and exiting...\n" % signum)
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.parquet")))
    if args.collections:
        keep = set(args.collections)
        files = [f for f in files if f.split(os.sep)[-2] in keep]

    scored = 0
    since_commit = 0
    since_flush = 0
    try:
        for path in files:
            cols = ["row_hash", "source", "reference", "hypothesis"]
            tbl = pq.read_table(path, columns=cols).to_pylist()
            for row in tbl:
                if stop["flag"]:
                    raise KeyboardInterrupt
                for name in metrics:
                    spec = METRICS[name]
                    if any(not (row.get(f) and str(row[f]).strip()) for f in spec["needs"]):
                        continue
                    sp = spec["space"]
                    k = key_for(sp, row)
                    if (k, name) in done[sp] or (k, name) in seen_this_run[sp]:
                        continue
                    s = score_one(name, row)
                    writer.write(sp, k, name, s)
                    seen_this_run[sp].add((k, name))
                    scored += 1
                    since_flush += 1
                    since_commit += 1
                    if since_flush >= args.flush_every:
                        writer.flush()
                        since_flush = 0
                        print("  scored=%d (%s)" % (scored, os.path.basename(path)))
                    if since_commit >= args.commit_every:
                        writer.flush()
                        git_commit_push(scored, args.push)
                        since_commit = 0
                    if args.limit and scored >= args.limit:
                        raise StopIteration
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        writer.close()
    if since_commit:
        git_commit_push(scored, args.push)
    print("DONE. new scores written this run: %d" % scored)


if __name__ == "__main__":
    main()
