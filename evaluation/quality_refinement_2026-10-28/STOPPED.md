# Stopped at the requested usage boundary

The usage guard latched a terminal stop when the primary resets_at changed from
1793602085 to 1793602086. Reported usage was 49%, not 95%. This one-second
change may be reporting jitter; the run conservatively treated it as a reset
and stopped rather than using a possibly reset allowance. Do not automatically
resume or delete the terminal usage state.

Work is isolated on codex/refine-rag-quality in the RAG and soprano-lab worktrees.
No main checkout was edited, no live server restarted, and nothing published.
Changes are uncommitted and retained for review.

Completed: baseline and Candidate A development probes (18 each); all120 final
cases and their106 reference texts audited against the current corpus; focused
security tests for A (148), probe runner tests (11), lab integration tests (5),
and latest refusal-detector focused tests (77). Candidate B is frozen with
source hashes and archive, but has not received model evaluation or a full
post-change test run. Two intermediate prompt-string test failures were fixed
before B was frozen; the complete focused suite has not been rerun since.

The broader baseline run checkpointed58/120 rows before stopping, including a
usage-interrupted worker error. This is not a completed security comparison.
Do not report attack rates or quality improvements from this partial run.
Candidate A still had invented source-conflict causes, unrelated detail, and
false refusals; it was not accepted as a successful refinement. Candidate B
uses a compact shared prompt and further detector corrections, awaiting tests.

Pending on explicit continuation: review exact validator diagnoses; run full
unit suite and Candidate B development probes; select/freeze a satisfactory
candidate; complete a fair120-case baseline/refined comparison with independent
semantic review; sync approved RAG changes into the lab worktree; regenerate
and inspect the25 static example answers; write final metrics/report. The lab
worktree currently contains only UI refusal-label and export-provenance changes.
Saved static answers and the live server remain unchanged.
