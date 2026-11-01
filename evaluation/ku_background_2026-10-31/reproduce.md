# Reproduce or resume the quality comparison

The November 1 resume supersedes the October 31 shutdown checkpoint. Read the
root `RESUME.md` for live progress. E15 was rejected for English meaning errors
despite passing development BERTScore. E17 source is `ba65680`, using reviewed
source translations plus a teaching-context relevance repair. It is experimental
until the complete comparison is accepted. The fixed target is BERTScore F1
at least 0.90 with meaning and overall quality preserved, not maximized F1.
Validation has been reused for refinement; it is not an untouched test set.

## Working directory and environments

Run the commands from:

```bash
cd /home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system
```

Generation and tests use `/home/minhee/miniconda3/envs/soprano-qa/bin/python`
(Python 3.11.15, llama-cpp-python 0.3.32). Metrics use
`/home/minhee/.cache/soprano-rag-eval-venv/bin/python`
(bert-score 0.3.13, transformers 5.10.2, torch 2.7.1+cu128).
Models are Qwen3-14B Q6_K and Qwen3-Embedding-0.6B Q8_0; checkpoint/corpus/source
hashes are in `e17-provenance.json`. Runtime snapshots and private
traces are local and ignored by Git. Public outputs, manifests, frozen cases
and provenance are committed.

Before inference, inspect GPU processes and compare the source/settings/case hashes in
the current run manifests and provenance (the old `resume-state.json` is historical). Do not stop or restart the existing live service. Run one
evaluation GPU worker at a time. No usage/reset stop or notifier is needed.

## Verify the reviewed source catalog

The catalog covers all 129 curated expert/performer sources, not selected test
questions. It is machine authored and model reviewed, not a new expert-reference
set. The original review and the five actual translation revisions are retained.
Rebuild verification uses the first review for unchanged translations and the
review of each edited version; it never selects the most favorable score.

```bash
python3 evaluation/build_source_translation_catalog.py \
  --base-review evaluation/ku_background_2026-10-31/translation-catalog-v1 \
  --revision-review evaluation/ku_background_2026-10-31/translation-catalog-v2 \
  --output soprano_qa/source_translation_catalog.py --check
```

Exact original source SHA256 selects an English translation at request time;
changed/new source text uses guarded segmented translation. Offline preparation
and review are added resources excluded from request latency. Source fidelity,
source factuality and question relevance must be evaluated separately.

## E17 diagnostic probe

Use the frozen E17 snapshot and the same 70 diagnostic cases as E15. Six are
explicitly reused earlier validation failures; this is a diagnostic cohort.
Never start another GPU job while an existing worker is running.

```bash
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u \
  evaluation/run_refinement_probe.py \
  --system-root evaluation/ku_background_2026-10-31/runtime/probe-e17/snapshots/hardened \
  --settings evaluation/ku_background_2026-10-31/runtime/probe-e17/snapshots/hardened/settings.json \
  --cases evaluation/ku_background_2026-10-31/e15_probe_cases.json \
  --output evaluation/ku_background_2026-10-31/probe-e17.jsonl \
  --variant hardened --warmup-case expert-kim-die-forelle-03__no-range
python3 evaluation/prepare_judge_packets.py \
  --cases evaluation/ku_background_2026-10-31/e15_probe_cases.json \
  --results evaluation/ku_background_2026-10-31/probe-e17.jsonl \
  --output evaluation/ku_background_2026-10-31/judge-e17
python3 evaluation/run_joint_claude_judge.py \
  --directory evaluation/ku_background_2026-10-31/judge-e17 \
  --workers 3 --model claude-opus-5-5
python3 evaluation/summarize_background_eval.py \
  --cases evaluation/ku_background_2026-10-31/e15_probe_cases.json \
  --results evaluation/ku_background_2026-10-31/probe-e17.jsonl \
  --judge-directory evaluation/ku_background_2026-10-31/judge-e17 \
  --output evaluation/ku_background_2026-10-31/review-e17
```

The judge records the actual returned model, CLI version and input/rubric hashes;
tools, MCP and session persistence are disabled. Invalid CLI/JSON/coverage/exact
quotation results may be retried at most three times with errors retained. The
first valid judgment is accepted even if unfavorable. No score-based retries.
Prior completed reviews retain their original contracts and are not replaced.

## Full comparison after candidate review

The fresh campaign output is `comparison-2026-11-01/`. Pre-security `f307470`
and main `29fe3e2` precede the candidate diagnostic probe; strict `302f8d1` follows.
A final candidate configuration will reuse those completed frozen baseline rows
and run 290 candidate requests, then jointly blind/deduplicate all four systems
for Claude assessment. The historical `comparison-e14.config.json` was never
started and must not be used as the final candidate configuration.

Use `python3 -u evaluation/run_background_comparison.py --config FINAL_CONFIG`
only after the selected candidate and its config are recorded. Each system has
one untimed warmup and one measured run. The driver scores development and
validation after generation exits, then writes a complete JSON summary and
side-by-side HTML. Generation uses Qwen, and assessment uses Claude.

Every source/settings/case/script contract is hash-bound. Do not change runner
scripts or snapshots while this command runs. If code changes, create a new
snapshot/config/output directory before measuring; never overwrite an earlier
result or pool historical latency. Resume of an unchanged config retains measured
rows, including errors. Missing/failed judge assessments prevent a complete report.
The generic preparer's embedded two-repeat protocol is historical; these actual
comparison configs explicitly use one repeat, as do their request manifests.

## Rebuilding a candidate snapshot

After committing a new candidate, use a **fresh** output directory:

```bash
/home/minhee/miniconda3/envs/soprano-qa/bin/python \
  evaluation/prepare_security_comparison.py --original-ref 302f8d1 \
  --output evaluation/ku_background_2026-10-31/runtime/NEW_CANDIDATE
```

Use its `snapshots/hardened` as the candidate. E17 needs these settings overrides
before any request is run: `llm.evidence_translation="reviewed_catalog"` and
`llm.evidence_translation_max_tokens=1536`. The default repository configuration
has not enabled them. Record overrides and settings hashes in provenance, copy
the comparison config, and set a new output path. `source_preservation` already
uses `verbatim_or_translation`. Preserve the three baseline snapshots unchanged.

For a new machine, use the same preparer with each baseline ref and take its
`original` snapshot. Compare corpus/model/source/settings hashes with committed
provenance; remove candidate-only flags from baseline runtime settings to match
the originals. `runtime/presecurity` in the current config is the main baseline,
not the genuinely pre-security source; do not infer identity from folder names.

## Frozen metric

The reusable setting is `config/bertscore.json`, identical in encoder/layer/IDF/
tokenizer/rescaling/revision to the original development-B selection used by
the comparison. Do not select a new metric using observed validation scores.

```bash
/home/minhee/.cache/soprano-rag-eval-venv/bin/python \
  evaluation/score_expert_bertscore.py \
  --cases evaluation/ku_background_2026-10-31/expert_cases.json \
  --results evaluation/ku_background_2026-10-31/comparison-2026-11-01/candidate.jsonl \
  --split validation --settings config/bertscore.json \
  --output evaluation/ku_background_2026-10-31/validation-recheck-final
```

The target set has 105 Korean requests, split into 43 development and 62
validation requests by original question group. Six KUs overlap across splits;
see `split-audit.json`. All applicable frozen expert KUs form each reference;
no best-reference choice, answer truncation, discarded errors or refusal exclusion
is permitted. The scorer rejects missing/duplicate requests and truncation under
the frozen setting. Errors/refusals contribute zero to the primary mean.

Raw BERTScore F1 is not factual accuracy. Three of four historical material errors
also exceeded 0.90 (`bertscore-fidelity-controls.json`). Extractive Korean KU
quotation increases literal overlap, so compare relevance, concision, refusal,
meaning and general-knowledge usefulness before accepting a high score.

## Verification and reporting

```bash
/home/minhee/miniconda3/envs/soprano-qa/bin/python -m unittest discover -q
python3 evaluation/export_current_prompts.py \
  --output evaluation/ku_background_2026-10-31/prompts-e17 --check
```

Run equivalent tests from the lab worktree's `rag/`, and run
`python -m unittest discover -s tests -q` from its `demo/` with the same interpreter.
Synchronize only the isolated branches, not main/live. The final report must
include reused-validation and six-KU-overlap limits, all-Korean BERTScore targets,
source fidelity versus source truth, fallible judge/manual review, forced indirect
attack exposure, and single-run latency limitations and additional offline source preparation. Mark the goal
complete only after the target and overall quality comparison are actually done.
