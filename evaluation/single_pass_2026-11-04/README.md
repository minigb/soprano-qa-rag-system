# Stage timing and single-pass candidate

Measurement is complete. The single-pass candidate is **not accepted as the
default**: expert BERTScore passes 0.90, but meaning, naturalness and prompt-only
task containment regress. Read [report.md](report.md), [decision.json](decision.json),
[quality-summary.json](quality-summary.json), and [manual-review.json](manual-review.json).
The working branch is `codex/helpful-rag-responses`; original/main and the old
backend remain untouched. The previous default configuration is restored exactly.
No evaluation, judge, server, notification or usage/reset monitor remains running.

## Current candidate and actual default

- `config/settings.json`: preserved validated configuration from `f71d5fe`.
- `config/settings.single-pass.json`: evaluated candidate G, same 14B checkpoint,
  hybrid retriever and corpus. One writer; English also translates the search query.
- `soprano_qa/single_pass.py`: task-scope prompt and one-call generation.
- `soprano_qa/single_pass_prompt.py`: initial free-prose writing instructions from
  `f307470`, adapted for general knowledge and direct delivery.
- The default still uses the previously validated reviewed path. This experiment
  has not accomplished a quality-preserving replacement without online review.

Try the candidate explicitly from this checkout:

```bash
conda run -n soprano-qa python scripts/ask.py \
  --settings config/settings.single-pass.json \
  --piece nella-fantasia --question '첫 소리는 어떤 다이나믹으로 시작할까?'
```

For the library, set `SOPRANO_QA_SETTINGS` to the absolute candidate configuration
path before importing `soprano_qa.service`. No server is started by these scripts.

## Captures and failed experiments

The directory date labels the UTC experiment date. Raw rows contain UTC times.
No completed or unfavorable output was overwritten, repaired, or selected by score.

| Directory | Status / purpose |
| --- | --- |
| `baseline` | Invalid observer: 13 rows. Different cache-key calling conventions attached the timer to the wrong model instance. Excluded from all timing claims. |
| `baseline-measured` | Valid pre-change source; 20 questions × 2 repetitions. |
| `single-pass-a` | Plain prose, 40 timings; unnecessary elaboration and meaning errors. |
| `single-pass-b` | Shorter prose, 40 timings; notation merging/meaning errors. |
| `single-pass-c` | Restored existing source-derived meaning hints, 40 timings. |
| `single-pass-final` | Superseded D despite its name; 40 timings, wrong p expansion and invented preference. |
| `expert-final` | D partial run, 23 outputs; not a complete evaluation. |
| `single-pass-e` | Prior comparison guidance with plain prose, 40 timings; direct-injection failure. |
| `expert-e` | E, 109 outputs and first Claude judgments; 13 material-drift flags across that 109-case cohort. No BERTScore selection was made from E. |
| `single-pass-f` | Structured composition retained, 40 timings; long, repetitive source-like text and slower writing. |
| `expert-f` | F partial run, 8 outputs; preserved, not a complete evaluation. |
| `single-pass-g` | Invalid latency capture, 9 rows: a briefly orphaned F inference process overlapped early measurements. Excluded. |
| `single-pass-g-clean` | G, 40 valid timings after both affected processes exited and the GPU was verified free. |
| `expert-g` | G, complete 109 requests, Claude assessment and frozen BERTScore. |
| `challenge-g` | G, complete 74 bilingual challenge requests and Claude assessment. |

Other prepared directories have `not-run.json` or `stopped.json` and no completed
evaluation. Their duplicate, never-executed scaffolding was removed;
`scaffolding-cleanup.json` points to the identical preserved source snapshot.
No generated answer, timing or judgment was removed. They must not be mistaken for accepted candidates. `run_validation.py`
references only the final G captures and is resumable; completed runs return
without regenerating answers. F's controller was stopped after a child had
already begun its expert capture; that child was stopped explicitly. The G
overlap was detected, invalidated and remeasured rather than silently included.

## Method and scope

`profile-cases.json` was chosen before changes: ten ordinary questions across
five works, all five prior refusals, three general/English questions, and two
scope/attack requests. Slow paths are intentionally overrepresented. Each run
preloads models/index, excludes one warmup and then runs forward and reverse.
`profile.py` times actual `create_chat_completion` calls and retrieval. Inclusive
wrapper spans are diagnostic and are not counted twice. The observer asserts
that it sees the actual cached inference object. This is sequential local testing
on an RTX A6000, not a load test, randomized interleaving, or traffic estimate.

`expert-g` contains all 105 original expert-based questions from the existing
460-case evaluation and four additional previous-refusal paraphrases (the fifth
refusal is already in the 105). Its retrieval is real, not supplied gold evidence.
`challenge-g` uses the previous fixed retrieval/lyric fixtures: 30 direct attacks,
20 indirect attacks, 12 unrelated requests and 12 normal controls, Korean/English.
The fixture runner respects the service's configured message builder. Only the
retrieval result is replaced; the actual public service handles every request.

The `source_head` in capture provenance is the base Git HEAD at freeze time;
uncommitted snapshot contents are identified by the per-file SHA256 values.
Verification matches those exact modules to the current checkout and the
explicit candidate configuration, not to the restored default configuration.
Implementation commits are `5822dd9` (candidate path) and `8631121` (preserved
default plus opt-in candidate).

All new answers receive offline Claude Opus 5.5 CLI assessment with the unchanged
rubric, schema and first-valid-judgment rule. It is never in the request path.
Historical scores come from the preserved prior evaluations; judge variance and
existing development-set exposure limit the comparison. This is not a new holdout.

BERTScore settings stay xlm-roberta-large, layer 18, no IDF, raw F1, revision
`c23d21b0620b635a76227c604d44e43a9f0ee389`. Whole displayed answers are scored;
semantic refusals/errors and overlength texts contribute zero under the frozen
policy. G has zero truncations/overlength cases. The expert-105 mean is 0.928871;
all 109 including four extra paraphrases average 0.927749. Passing that metric
does not override the observed meaning and task-boundary failures.

## Reproduce summaries / continue a new experiment

Completed captures are hash-bound. Do not rerun them for a more favorable answer
or judge score. Resume only genuinely unfinished valid work; invalid/superseded
directories above are not pending work.

```bash
python evaluation/single_pass_2026-11-04/summarize_profile.py
python evaluation/single_pass_2026-11-04/analyze_quality.py
python evaluation/single_pass_2026-11-04/verify.py
```

New experiments must use a new directory, preserve the existing references and
metric settings, and serialize GPU inference jobs. `prepare_profile.py NAME`
and `prepare_eval.py NAME [--challenge]` freeze the current explicit single-pass
candidate, not the restored default. The frozen runners use their own snapshots
and runtime copies, so current edits cannot alter an ongoing capture.

The final generation/judging/scoring commands used were:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/single_pass_2026-11-04/run_validation.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/single_pass_2026-11-04/expert-g/watch_judge.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/single_pass_2026-11-04/challenge-g/watch_judge.py
# After GPU generation is complete and all expert judgments exist:
HF_HUB_OFFLINE=1 /home/minhee/.cache/soprano-rag-eval-venv/bin/python -u evaluation/single_pass_2026-11-04/expert-g/score.py
```

The 460-case localhost demo is deliberately unchanged: it still shows the last
complete, validated comparison. The new 109-case answer comparison is
`answer-comparison.json`; it is not presented as a new 460-case demo dataset.
684 tests passed. `verification.json` checks the candidate against frozen source
hashes, restored default, 14B weights, corpus/retrieval settings, original/main,
the old backend, and all existing demo mirrors.

Frozen snapshots retain the original bytes, including the pre-existing extra EOF
blank line in `soprano_qa/__init__.py`. Whitespace checks of new files outside
those snapshots pass; snapshots are not reformatted after hashing.
