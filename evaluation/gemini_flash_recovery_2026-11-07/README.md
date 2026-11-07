# Missing-answer recovery on the newly configured account

Completed on 2026-11-07 (Asia/Seoul), branch `codex/helpful-rag-responses`.
The user supplied a new Google API account and authorized its use if needed.
`~/gemini-vertex-4/env.sh` now has updated ADC for its own project; it is no longer
the old copied account3 credential configuration.

Only the **7 failed Gemini requests** were retried, once each: six expert questions
and one English direct-injection question. All seven returned answers. The model
remained `gemini-3.8-flash`, LOW, with identical frozen messages and generation
settings. There were no health-probe, successful-answer regeneration, extra model,
or repeat latency-benchmark calls. Requests were serial, at least10 seconds apart.
The unrelated YouTube jobs were not modified.

All464 tracked files in the original comparison remain byte-for-byte unchanged.
`contract.json` records their hashes, the seven eligible IDs, exact input hashes,
account routing and the one-retry policy before generation. `requests/` contains
each retry outcome. `combined/` is a separate view of the175 unchanged original
answers plus the seven recovered answers; it does not replace the original run.
The original translation-failure exclusion remains excluded.

Claude Opus5.5 used the unchanged rubric for the seven new answers only; all175
existing judgments were reused unchanged. Only the six new expert answers were
scored with the original XLM-R-large/layer18/noIDF/rawF1 configuration and encoder
revision. No-IDF per-answer scores for the other99 expert answers were reused
verbatim. No metric tuning, truncation or answer replacement was performed.

See [the supplemental report](report.md), [summary](summary.json),
[expert answer comparison](expert-comparison.json),
[challenge comparison](challenge-comparison.json), and [verification](verification.json).

`run.py` is the frozen historical capture/judge/metric script. Do not rerun completed
API requests. `analyze.py` can reconstruct the combined report without API calls:

```bash
cd /home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system
/home/minhee/miniconda3/envs/soprano-qa/bin/python \
  evaluation/gemini_flash_recovery_2026-11-07/analyze.py
```

No runtime code, default model configuration, original baseline, corpus or web
demo was changed. This offline recovery does not add automatic retries or account
fallback to the serving path. Original first-attempt reliability and latency
measurements remain valid historical observations, not measurements of this new
account. These seven cases were selected because they had failed, so their timings
must not be presented as a new representative latency benchmark.
