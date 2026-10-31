# Reproduce or resume the unfinished evaluation

This is a server-shutdown checkpoint, not a completed experiment. Read the root
`RESUME.md` first. E14 production source is `759511c`; it is **experimental and
not selected**. Its nine-case English probe is complete, with residual meaning,
concision and refusal issues recorded in `e14-focused-review.json`. No complete
four-system comparison has been run. E11 validation was below the target.

BERTScore 0.90 is a sufficient floor, not a maximization objective. Acceptance
also requires useful, faithful answers and an acceptable refusal/latency tradeoff.
The later validation set has been reused for refinement, not held untouched.

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
hashes are in `provenance/e14/provenance.json`. Runtime snapshots and private
traces are local and ignored by Git. Public outputs, manifests, frozen cases
and provenance are committed.

Before inference, inspect GPU processes and compare the hashes in
`resume-state.json`. Do not stop or restart the existing live service. Run one
evaluation GPU worker at a time. No usage/reset stop or notifier is needed.

## Finish independent review of the saved focused probe

These commands use saved answers and do not load a generation model:

```bash
python3 evaluation/prepare_judge_packets.py \
  --cases evaluation/ku_background_2026-10-31/e14_focused_cases.json \
  --results evaluation/ku_background_2026-10-31/probe-e14-focused.jsonl \
  --output evaluation/ku_background_2026-10-31/judge-e14-focused
/home/minhee/miniconda3/envs/soprano-qa/bin/python \
  evaluation/run_joint_claude_judge.py \
  --directory evaluation/ku_background_2026-10-31/judge-e14-focused \
  --workers 3 --model claude-opus-5-5
python3 evaluation/summarize_background_eval.py \
  --cases evaluation/ku_background_2026-10-31/e14_focused_cases.json \
  --results evaluation/ku_background_2026-10-31/probe-e14-focused.jsonl \
  --judge-directory evaluation/ku_background_2026-10-31/judge-e14-focused \
  --output evaluation/ku_background_2026-10-31/review-e14-focused
```

The CLI runner records the actual returned model, version, rubric/schema/input
hashes and errors; tools, MCP and session persistence are disabled. There is no
selection among repeated judgments. The latest rubric also explicitly penalizes
irrelevant padding and asks for awkward wording to be flagged. Historical
assessments retain their original rubric hashes.

## Full comparison after candidate review

`comparison-e14.config.json` is prepared but **not started**. If E14 is retained
for evaluation without source changes:

```bash
python3 -u evaluation/run_background_comparison.py \
  --config evaluation/ku_background_2026-10-31/comparison-e14.config.json
```

This sequentially runs 290 requests for E14, pre-security `f307470`, main security
`29fe3e2` and strict `302f8d1`; each has one untimed warmup and one measured run.
It scores development and validation after each inference process exits, then
jointly blinds/deduplicates packets, runs three independent Claude CLI workers,
and writes the full JSON summary and side-by-side HTML. The evaluation generation
model is never the independent judge.

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

Use its `snapshots/hardened` as the candidate. E14 needs these settings overrides
before any request is run: `llm.evidence_translation="segmented"` and
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
  --results evaluation/ku_background_2026-10-31/comparison-e14/candidate.jsonl \
  --split validation --settings config/bertscore.json \
  --output evaluation/ku_background_2026-10-31/validation-recheck-e14
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
  --output evaluation/ku_background_2026-10-31/prompts-e14 --check
```

Run equivalent tests from the lab worktree's `rag/`, and run
`python -m unittest discover -s tests -q` from its `demo/` with the same interpreter.
Synchronize only the isolated branches, not main/live. The final report must
include reused-validation and six-KU-overlap limits, all-Korean BERTScore targets,
source fidelity versus source truth, fallible judge/manual review, forced indirect
attack exposure, and single-run shared-GPU latency limitations. Mark the goal
complete only after the target and overall quality comparison are actually done.
