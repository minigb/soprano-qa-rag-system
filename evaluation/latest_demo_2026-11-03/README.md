# Current 14B outputs for the existing answer comparison demo

This is a fresh output capture, not a new RAG implementation or an ablation arm.
The source is the current uncommitted semantic-generation WIP in
`/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`.
`provenance.json` records the exact source, configuration and Qwen3-14B-Q6_K hashes.
The tracked `snapshot/soprano_qa` is byte-exact; production files stay unchanged.

`run.py` calls that snapshot's public `service.ask` directly for every original
460-case request, including its real retrieval, routing, generation and checks.
It uses no model proxy, source-text substitution, ablation intervention, usage
monitor or notifier. Results, failures and refusals checkpoint once per case.
The existing initial-system answers are retained verbatim on the left of the UI.

From this worktree, resume without replacing completed cases:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/latest_demo_2026-11-03/run.py
```

Evaluate completed batches with the existing Claude CLI rubric:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/latest_demo_2026-11-03/watch_judge.py
```

Once generation and all 460 Claude judgments are complete, compute the fixed
metric before exporting:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 /home/minhee/.cache/soprano-rag-eval-venv/bin/python -u evaluation/latest_demo_2026-11-03/score.py
python sites/answer-review/export_latest.py
python sites/answer-review/verify_latest.py
node --check sites/answer-review/dist/app.js
```

The exporter requires complete coverage and matching result/source hashes.
`watch_judge.py` evaluates finished chunks with Claude Opus5.5 and the exact
same rubric/runner as the historical demo. After all judgments and generation,
run `score.py` using `/home/minhee/.cache/soprano-rag-eval-venv/bin/python` with
`HF_HUB_OFFLINE=1`. It retains the existing BERTScore setting and full answers;
unavailable, semantic-refusal and overlength cases contribute zero.
No older Claude judgment or BERTScore is attached to a new answer. Outputs not
yet evaluated are explicitly **not evaluated**; this is distinct from zero
quality or an unscorable metric. The historical E37 view, deep links and notes remain available.
The 460-case initial answers, references and existing judgments are unchanged.

Serving instructions are in `sites/answer-review/README.md`. No HTTP server or
hosted deployment is started by this capture or exporter.

## Completed capture

All 460 requests and their Claude assessments completed on 2026-11-03. The
public service returned 455 generated answers and 5 unavailable responses;
Claude also identified those 5 as refusals. No request was discarded or rerun
to replace its answer. BERTScore used the full delivered text, with those five
refusals retained as zero contributions. There were no overlength answers.

| Measure | Preserved initial system | Current 14B snapshot |
| --- | ---: | ---: |
| Requests | 460 | 460 |
| Claude material source drift | 27 | 18 |
| Effective naturalness / 4 | 3.5022 | 3.5239 |
| Expert original cohort BERTScore F1 (105 cases) | 0.9468 | 0.9366 |
| Mean request latency, seconds | 3.0370 | 8.6630 |
| Median request latency, seconds | 2.7131 | 8.0042 |

These are the historical initial run and a fresh full-service capture, not a
controlled component timing experiment or a new independent evaluation set.
Claude and BERTScore evaluation time is excluded from request latency. The
expert similarity threshold is met, while 5 unanswered music requests and 18
material-drift judgments remain; this is not a deployment acceptance result.

`data-verification.json` records exact comparison of all 920 full answers,
unchanged historical data, new score mappings and unchanged backend source.
`browser-verification.json` records the passing browser checks, including
version selection, isolated notes, filters, source panels, links and four
viewport widths. The browser test served local files through request
interception without starting an HTTP server.
