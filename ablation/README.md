# Isolated 14B component comparison

Working branch: `codex/rag-component-ablation`.
Root: `/home/minhee/.codex/worktrees/rag-component-ablation/soprano-qa-rag-system`.

Completed: all 233 cases, 12 conditions, 970 unique Claude packets, and the frozen
expert105 BERTScore comparison. Start with [FINDINGS.md](FINDINGS.md),
[actual answer examples](examples.md), or [all answers](review.html).
All experiment jobs have finished. No candidate policy was adopted into production.

This experiment does not edit either protected checkout:

- Original: `/home/minhee/.codex/worktrees/rag-pre-injection/soprano-qa-rag-system`
- Current: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`

The user fixed generation to Qwen3-14B-Q6_K. The checkpoint SHA256 is
`ec1f1d1421d7636a23e17645cac6fda39ee4b5f1f11e42eb09238089705ef699`.
No new model is downloaded or selected. Claude Opus5.5 CLI is the separately authorized
offline evaluator, not a production generator or output gate.

Read [PROTOCOL.md](PROTOCOL.md) and [ANALYSIS-AMENDMENT.md](ANALYSIS-AMENDMENT.md).
All 233 questions and contexts are frozen in `cases.json`; source/model hashes are in
`manifest.json`. Source snapshots are byte exact. A separate execution copy renames the
current package namespace to coexist with the original package. Runtime imports do not
edit either source checkout. No retrieval/translation search is repeated across conditions.

Commands from this worktree:

```bash
python ablation/restore_runtime.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u ablation/run.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u ablation/watch_judge.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python ablation/audit_static_output.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python ablation/analyze.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python ablation/write_examples.py
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 /home/minhee/.cache/soprano-rag-eval-venv/bin/python ablation/score_fixed_metric.py
```

Generation and Claude evaluation can overlap: GPU generation is serial, Claude evaluation uses three
CLI workers on completed immutable batches. Resume reuses exact previous calls and the
first valid exact-packet judgment, not favorable scores. `run.py --limit 2` was the first
two cases of the same shuffled cohort, not a filtered quality pretest; those cases remain.
The idle watcher was restarted before its first batch to enable exact reuse across batches.

Results checkpoint one complete case at a time in `results.jsonl`. Saved draft text,
input prompts and per-call information are in ignored mode0600 `*.private.jsonl` files.
The completed analyzer writes `report.md`, `summary.json`, `changed-cases.json`, all
variant JSONL files, and a standalone `review.html`. No HTTP server or hosted site starts.
The analyzer refuses incomplete generation/judging and verifies protected source hashes.
No production adoption, usage/reset stopping, notifications, or main-branch edits occur.
Final checks verified all 2,796 HTML answer strings against saved outputs, metric
input hashes against their contracts, and all protected source hashes. Whitespace
checks pass for experiment code and generated artifacts. The two byte-exact
snapshot `__init__.py` files retain their existing blank final line; they were not
edited merely to silence Git's `blank-at-eof` diagnostic.
The supplementary metric uses the previously frozen setting on all 105 expert cases.
It never searches settings or changes answers; Claude refusals and unavailable outputs
contribute zero. The generated `fixed-bertscore/PROTOCOL.json` records the full contract.

Important interpretation: this is an original-writer/frozen-context intervention, not a
performance estimate of the entire current service. In particular, the current JSON
renderer automatically supplies citations, whereas the original prose writer often puts
one label only at the end of a paragraph. Strictly blocking that original draft tests the
warning/withholding decision; it must not be mistaken for the current service's refusal rate.
