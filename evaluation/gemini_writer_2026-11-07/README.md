# Gemini writer comparison — 2026-11-07

**Archived, incomplete, do not resume.** The user stopped additional API-model
experiments. The active comparison is
[`../gemini_flash_writer_2026-11-07/`](../gemini_flash_writer_2026-11-07/README.md),
Gemini 3.8 Flash only versus local Qwen3-14B. Historical commands below describe
the abandoned capture and are not current resume instructions.

Worktree: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`.
Branch: `codex/helpful-rag-responses`.

This is a controlled writer substitution using candidate J's exact reconstructed
system/user messages. The same corpus records, retrieved KU order, measure scope,
score context and original questions are retained. No expert reference answer,
judge result or baseline answer is sent as an additional generation input.

`prepare.py` executes the real service with recorded retrieval and translation
results, capturing its writer messages. All 182 successful historical service
responses reproduce exactly apart from model path metadata. For source exclusion
cases, the already-recorded recovery branch is replayed with no source evidence.
This does not establish defense against passing the raw excluded injection to
Gemini. The original source screens remain part of the tested application.

The inventory contains 109 expert/regression cases (105 original expert questions
+ four historical-refusal paraphrases) and 74 Korean/English challenge cases.
`out-of-scope-01-en` previously failed before writing in Qwen query translation;
it remains excluded, not counted as successful defense. Paired challenge N=73.
These are reused development questions, not a new holdout or the full460.

## Frozen contract

`contract.json`, `writer-settings.json`, `snapshot/`, and each cohort's
`writer-inputs.json` identify the code and input hashes. `current.jsonl` stores the
first output/error per case. `inflight.json` prevents an ambiguous interrupted
request from being silently submitted again. Existing results are not overwritten.

Gemini uses `gemini-4-argon`, LOW thinking and max_output_tokens2048. J used
Qwen3-14B-Q6_K, non-thinking and max_tokens768. Common sampling fields are sent
unchanged, but the [official guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-4-argon)
says Argon ignores temperature/top_p/top_k. The note in `contract.json` that these
values are retained describes the submitted request, **not effective matching**.
This comparison isolates the writer backend at fixed context, not model size,
training, quantization, reasoning or sampler behavior separately.

The first34 outcomes used sequential Standard API calls. Because of long API
waiting times, the remaining calls use a bounded pool of three workers, with
per-case journals and ordered aggregation. `scheduling-amendment.json` records
this change and one deliberately interrupted request with no received answer;
that same input was resubmitted, while every already-stored success/failure was
retained. The provider may bill the interrupted request. `execution-contract.json`
identifies the new runner without altering the original frozen contract.

Recorded wall time excludes retrieval/query translation. Historical end-to-end times must not be presented
as directly comparable generation-only latency. API finish reasons, resolved
model and token usage are saved. No tools, web grounding, output review, source
substitution or automatic retries are used. Mixed-concurrency quality-capture
timings are diagnostic only. `timing_sequential.py` separately repeats the same
ten ordinary-music questions serially after generation finishes; its additional
answers are never selected for judging or BERTScore. A technical failure is kept separate
from an assistant refusal or attack defense.

## Run / resume

```bash
source ~/gemini-vertex-3/env.sh
cd /home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/gemini_writer_2026-11-07/run_parallel.py
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/gemini_writer_2026-11-07/expert/watch_judge.py
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/gemini_writer_2026-11-07/challenge/watch_judge.py
HF_HUB_OFFLINE=1 /home/minhee/.cache/soprano-rag-eval-venv/bin/python -u \
  evaluation/gemini_writer_2026-11-07/expert/score.py
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/gemini_writer_2026-11-07/timing_sequential.py
/home/minhee/miniconda3/envs/soprano-qa/bin/python evaluation/gemini_writer_2026-11-07/analyze.py
```

Do not run `prepare.py` or the superseded sequential `run.py` again. Inspect active processes before
resuming generation or judging. Claude Opus5.5 uses the unchanged historical
rubric and first valid assessment rule, with identity-blinded packets. Fixed
BERTScore is XLM-R-large/layer18/noIDF/rawF1 with the frozen revision, complete
answers, no truncation and no parameter tuning. Failures/refusals/encoder overflow
retain zero contribution in conservative means.

The original/main/Lab backend and existing460-demo remain preserved. No usage-reset
stop rule or Discord/process-stop notifier is used.
