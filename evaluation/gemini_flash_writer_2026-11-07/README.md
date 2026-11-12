# Gemini Flash versus local Qwen3-14B

Worktree: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`.
Branch: `codex/helpful-rag-responses`. **Completed and verified.** See
[the report](report.md), [metrics](summary.json), and [decision](decision.json).
Full105 effective BERTScore is0.897379 (target unmet); returned99 score0.951765.
Do not rerun completed captures. Only
`gemini-3.8-flash` is used; the earlier Argon capture is an unfinished archive.

## Comparison contract

The writer receives candidate J's exact system/user messages, questions, KU order,
measure scope and score context. Retrieval is the existing hybrid BM25 + Qwen
embedding search. English query translation remains local Qwen3-14B. The captured
retrieval/translation results are replayed to keep generation inputs identical.
Reference answers, baseline answers and judge results are not added to the prompt.
There is no online output judge, answer replacement or quality-based retry.

`contract.json`, `writer-settings.json`, `snapshot/` and each cohort's
`writer-inputs.json` contain the input/code hashes. The service reconstruction
reproduced 182 historical successful responses except for model path metadata.
Cases whose retrieved sources were screened out retain J's empty-evidence
recovery input; this is not a test of passing those raw excluded attacks to Gemini.

The reused development inventory contains:

- 105 questions based on expert annotations, plus four previous-refusal paraphrases.
- 74 bilingual challenge cases: 30 direct attacks, 20 source-injection cases,
  12 benign music controls and 12 out-of-scope requests.
- One historical English translation failure has no writer input and is excluded
  from the paired challenge comparison (73 paired). It is not a defense success.

This is not a new holdout and is not the full 460-case evaluation. Local comparison
outputs are frozen candidate J results in `../prompt_fidelity_2026-11-06/`.

Gemini uses LOW thinking, 2048 output tokens and the original prompt, including
the Qwen `/no_think` suffix. Qwen used non-thinking generation with 768 tokens.
The common sampling fields are sent unchanged, but Google documents that Flash
ignores temperature/top_p/top_k. Tokenization, training and reasoning differ.
This comparison tests a writer replacement at fixed context, not model size alone.

## Results and failures

`current.jsonl` preserves each case's first received answer or technical error.
Per-case journals save finished calls before ordered aggregation; `.pending`
files guard against unobserved interrupted calls. Do not silently resubmit those.
Two exact-input Flash probe answers were reused at the beginning, not regenerated.

Claude Opus 5.5 CLI uses the existing identity-blinded rubric and first valid
assessment. It judges source meaning, factuality, completeness, naturalness,
refusal and attack success offline. BERTScore uses the existing frozen
XLM-R-large revision, layer18, noIDF, raw F1. Complete displayed answers are scored
without truncation or metric tuning. API failures and refusals contribute zero
to effective means. Conditional answer-quality results are reported separately.

API errors are not counted as refusals, semantic errors or successful defenses.
Matched successful-pair quality accompanies full-inventory availability metrics.
Never select a later or better answer to improve a metric.

Quality capture initially used concurrency2/5-second starts, then concurrency1/
10-second starts after429s, then concurrency2 at the same10-second spacing to
overlap long provider waits; another429 led back to serial10-second starts.
Separate serial latency observations use the same ten
ordinary music questions as J, excluding retrieval/translation and spacing waits.
Those extra answers never enter quality scoring. Historical local timing used two
repeats; API timing uses one. Shared YouTube work continues on the API project,
so measured latency is descriptive and cannot isolate provider/model speed.

## Historical run / recovery procedure

Only `~/gemini-vertex-3/env.sh` is authorized for further calls. The user withdrew
account4 fallback after authentication/permission failures. Keep all setup errors
and switching records. Do not stop the unrelated YouTube jobs.

Check for existing runners and `.pending` journal files before resuming:

```bash
cd /home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system
source ~/gemini-vertex-3/env.sh
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u \
  evaluation/gemini_flash_writer_2026-11-07/run_parallel.py \
  --workers 1 --start-interval 10
```

`finish.py` waits for generation, performs the offline judging and fixed metric,
then the separate ten-case serial timing and analysis. Do not start a duplicate
controller. `verify.py` also requires the actual selected-model Korean/English
service smoke records. Reports must explicitly label any incomplete capture.
There is no account/model fallback, usage-reset stop rule or notifier.

The original/main, local settings, corpus, prompt and existing 460-case demo are
preserved. The API backend is opt-in via `config/settings.gemini.json`.

## References

- [Google Gemini 3.8 Flash guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Google 429 guidance](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deploy/error-code-429)
