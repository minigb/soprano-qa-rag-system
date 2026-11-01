# Reproduce or resume the KU/background comparison

The current candidate is **E25 `a93b108`**, with `e25-provenance.json`,
`prompts-e25/` and `comparison-e25.config.json`. Read the root `RESUME.md` for
live progress. E24 completed the full comparison and passed the metric/meaning
checks, but direct review found a copied encoded source instruction. E25 changes
only that source/output boundary. Earlier results, including failed candidates,
are retained. BERTScore0.90 is a sufficient floor; overall quality decides
acceptance. All later validation is reused internal evaluation, not an untouched
test set.

## Environment and worktrees

Run from:

```bash
cd /home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system
```

Generation and tests use `/home/minhee/miniconda3/envs/soprano-qa/bin/python`
(Python3.11.15, llama-cpp-python0.3.32). Metrics use
`/home/minhee/.cache/soprano-rag-eval-venv/bin/python`
(bert-score0.3.13, transformers5.10.2, torch2.7.1+cu128).
Qwen3-14B Q6_K and Qwen3-Embedding-0.6B Q8_0 model hashes, corpus hash,
source modules and algorithm settings are in `e25-provenance.json`.
Runtime snapshots/caches and private model traces are ignored by Git.
Public answers, cases, manifests, scores and judgments are committed.

Use one GPU inference/metric process at a time. Inspect existing processes
before starting another run; do not stop or restart a live service. Both
worktrees use `codex/ku-background-knowledge`; do not write main or the dataset.
The Lab worktree is `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`.
No usage/reset stop or notification process is part of this experiment.

## Complete or resume E25 on this machine

The immutable comparison configuration records four systems: truly pre-security
`f307470`, initial-security main `29fe3e2`, strict `302f8d1`, and E25 `a93b108`.
The first three reuse the completed fresh campaign's measured rows and metric
artifacts, with copy-time hashes in `baseline-reuse.json`. Resuming adds session
metadata to manifests but does not replace measured rows. Every E25 candidate
request is newly measured; E24 candidate timings are not pooled into it.

```bash
python3 -u evaluation/run_background_comparison.py \
  --config evaluation/ku_background_2026-10-31/comparison-e25.config.json
python3 evaluation/supplement_background_comparison.py \
  --config evaluation/ku_background_2026-10-31/comparison-e25.config.json
python3 evaluation/audit_source_quotations.py \
  --results evaluation/ku_background_2026-10-31/comparison-2026-11-02-e25/candidate.jsonl \
  --output evaluation/ku_background_2026-10-31/comparison-2026-11-02-e25/quotation-audit.json
python3 evaluation/audit_translated_sources.py \
  --results evaluation/ku_background_2026-10-31/comparison-2026-11-02-e25/candidate.jsonl \
  --output evaluation/ku_background_2026-10-31/comparison-2026-11-02-e25/translation-audit.json
```

Do not launch a second controller while the first is active. Resume only an
unchanged config/source/settings/cases/scripts contract. Errors stay recorded;
the scorer does not rerun them to improve a score. If code changes, create a new
snapshot, config and output directory. Do not resume historical E19 using the
current driver: its archived contract was created by an earlier script revision.
The old snapshot preparer's embedded two-repeat protocol is historical metadata;
the actual config and request manifests specify one untimed warmup and one
measured repetition, with systems run sequentially.

The driver writes `review/summary.json` and a side-by-side `review/review.html`.
The supplement adds language-by-task counts and matched-answer latency. Read all
severe/attack flags and previously distorted cases directly. Exact quotation is
not proof of relevance or safety: E24's copied encoded command illustrates this.
Reuse a previous manual review only when its complete judge packet SHA256 agrees.

## Independent Claude assessment

The user-selected evaluator is CLI model `claude-opus-5-5`; actual model and
CLI version are retained in the assessment artifacts. Generation uses local
Qwen, not Claude. Judge tools, hooks, MCP and session persistence are disabled.
System identity and BERTScore results are hidden from assessment packets.

E25 uses the completed E19 review as a fixed cache source. Reuse requires the
whole packet, model, rubric, schema and runner to agree. Original assessment
contracts and unfavorable scores are preserved. Changed packets receive fresh
CLI calls, identified by `judge/reuse-aggregation.json`. Later diagnostic scores
are not selected as favorable replacements. Each new batch accepts its first
valid judgment; only CLI/JSON/coverage/exact-quotation failures may retry, up to
three attempts, with failures retained. Missing assessments prevent completion.

For a separate diagnostic, use `run_refinement_probe.py`, then
`prepare_judge_packets.py`, `run_joint_claude_judge.py` and
`summarize_background_eval.py`, always with fresh output paths. The E25 four-case
probe uses unchanged full-suite questions. Two attacks are withheld before
composition; the two severe English meaning cases reproduce E24's checked
answers exactly (`e25-probe-decision.json`). It is not a new independent test.

## Verify the catalog and current prompts

```bash
python3 evaluation/build_source_translation_catalog.py \
  --base-review evaluation/ku_background_2026-10-31/translation-catalog-v1 \
  --revision-review evaluation/ku_background_2026-10-31/translation-catalog-v2 \
  --output soprano_qa/source_translation_catalog.py --check
python3 evaluation/export_current_prompts.py \
  --output evaluation/ku_background_2026-10-31/prompts-e25 --check
```

The English catalog covers all129 curated expert/performer sources, keyed by
exact Korean source hash. Its translations are machine authored and separately
model reviewed, not human-expert English references. Five actual edits use the
review of the edited version; unchanged translations retain their original
review. Changed/new source text uses guarded segmented translation. Offline
preparation is an extra resource excluded from request latency. Catalog text,
original source truth and question relevance are separate checks.

E25 also retains basic conventional meter arithmetic. These answers still pass
admission/output review and declare `llm_guarded_calculation`; they are not
model-composed prose. The four-system comparison is of the complete engineering
package, not a controlled ablation of a security prompt alone.

## Frozen metric

`config/bertscore.json` reproduces the original development-B choice among18
configurations: xlm-roberta-large layer18, IDFfalse, raw/unrescaled, slow
tokenizer, langko and pinned revision. Do not reselect it from validation scores.

```bash
/home/minhee/.cache/soprano-rag-eval-venv/bin/python \
  evaluation/score_expert_bertscore.py \
  --cases evaluation/ku_background_2026-10-31/expert_cases.json \
  --results evaluation/ku_background_2026-10-31/comparison-2026-11-02-e25/candidate.jsonl \
  --split validation --settings config/bertscore.json \
  --output evaluation/ku_background_2026-10-31/validation-recheck-e25
```

All105 Korean expert-reference requests are frozen, split by original question
into43 development and62 validation requests. Original-question groups do not
overlap; six KUs do (`split-audit.json`). Complete displayed answers, including
background, are scored against all required KUs in frozen order. Errors/refusals
contribute zero. No best-reference selection, missing cases, duplicate cases or
token truncation is allowed. The installed bert-score package is0.3.13; its
upstream scorer-hash string reports0.3.12, both recorded in artifacts.

Raw BERTScore is not factual accuracy. Three of four historical material errors
also exceeded0.90 (`bertscore-fidelity-controls.json`). Literal Korean quotation
increases overlap. Source truth, meaning, usefulness, repetition, availability
and speed therefore remain separate acceptance evidence.

## New machine or new candidate

After committing a new candidate, run the snapshot preparer with a fresh path:

```bash
/home/minhee/miniconda3/envs/soprano-qa/bin/python \
  evaluation/prepare_security_comparison.py --original-ref f307470 \
  --output evaluation/ku_background_2026-10-31/runtime/NEW_CANDIDATE
```

Its hardened snapshot must use `source_preservation="verbatim_or_translation"`,
`evidence_translation="reviewed_catalog"` and translation max_tokens1536, matching
the recorded candidate configuration. Bind settings/source hashes in a new
provenance/config. The original snapshot must match its original settings,
without candidate-only flags. Recreate each baseline from its stated commit
and compare corpus/model/source hashes. Absolute paths on a new machine require
new contracts/output paths; do not relabel archived timings as new measurements.
The folder named `runtime/presecurity` contains main `29fe3e2`, while the truly
pre-security source is in `runtime/baseline-pre-security`.

## Final verification

```bash
/home/minhee/miniconda3/envs/soprano-qa/bin/python -m unittest discover -q
```

Run the same suite from Lab `rag/`, and run `python -m unittest discover -s tests -q`
from Lab `demo/` with the same interpreter. Sync only the isolated branches.
Record honest quality/latency limitations, reused-validation status and source
preparation costs. Complete the goal only after actual final results, commits,
Lab synchronization and verification are finished.
