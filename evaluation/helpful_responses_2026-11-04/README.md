# Useful-response implementation and evaluation

Completed: all 460 requests regenerated and assessed, 74 bilingual challenge
requests assessed, fixed-setting BERTScore computed, and the comparison demo
updated and browser-checked. The five previous refusals now answer faithfully;
their naturalness is 3/4, below the initial five-answer mean of 3.6/4. Expert-105
BERTScore is 0.945026. Existing material meaning problems remain in 17/460 cases;
two previously answered cases have lower naturalness ratings. See
[report.md](report.md), [comparison.json](comparison.json), and
[manual-review.json](manual-review.json) for the improvements and limitations.

Implementation: `codex/helpful-rag-responses`, commits `2f01ba7` and `e9d523c`, in
`/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`.
The original `codex/pre-injection-baseline` and the previous background-knowledge
working tree are preserved. Only the latter's static comparison-demo assets are
to be refreshed after the final evaluation; its backend is not this implementation.

The real library and CLI share `soprano_qa/response_policy.py`. A quality failure
gets one rewrite, then a distinct instruction/domain check can release sanitized
prose with a visible qualification. An empty or unusable music answer gets one
general-knowledge attempt. Unsafe source material is excluded before a general
answer is attempted. Out-of-domain requests receive a service-scope explanation.
Exhausted safety/backend failures remain explicit errors; this is not a guarantee
that every possible request always gets a substantive musical answer.

Classification `topic` metadata stays in the retrieval corpus but is omitted from
answer-generation inputs and generation evidence. The bilingual `indirect-07`
case exposed why this matters: a command in `topic` made the old gate discard
an entirely clean expert answer, after which general advice reversed its meaning.
Removing the nonessential category field preserves the full source answer, its
question, attribution and scope, all of which still receive complete screening.
No source fact is trimmed or restored from a hidden reference answer.

Generation is fixed to Qwen3-14B-Q6_K, semantic synthesis, temperature 0, seed 42,
8192 context and the existing retrieval configuration. There is no source-text
substitution and no BERTScore setting search.

## Frozen captures

- `pilot-a`, `pilot-b`, `pilot-d`, `pilot-e`, `pilot-f`, `pilot-g`: all five prior
  refusals, including unsuccessful wording experiments. `pilot-summary.json`
  records every completed candidate's judgments. G is the final writer-policy
  candidate; the final implementation also includes L's metadata correction.
  F introduced an unsupported general extension; G separates scope corrections
  from broader quality rewrites and removes that extension in these examples.
- `pilot-c`: setup mistake (wrong interpreter, 460-case inventory); generation
  never started. It is not an evaluated candidate.
- `security`: incomplete preliminary E security run, explicitly stopped when
  the writer prompt changed. `stopped.json` records its incomplete status.
- `security-final`: the complete pre-metadata-fix 74-case capture, which exposed
  the meaning reversal. It is preserved as a failed intermediate evaluation.
- `source-recovery-probe`, `source-recovery-probe-i`, `source-recovery-probe-j`:
  unsuccessful general-prompt/reasoning experiments on the bilingual regression.
  Those generation changes were reverted. A format-placeholder usability check
  was retained. `source-recovery-probe-k` showed that static pattern matching was
  insufficient for this semantic metadata injection. `source-recovery-probe-l`
  omits the category field entirely and passes both cases without a general fallback.
- `full`: an 18-row preliminary capture stopped when the security evaluation
  exposed that source-loss regression. It is not the final demo dataset.
- `security-hardened`: the final 74 cases, using the frozen attack inventory from the prior
  component-ablation work: 30 direct attacks, 20 indirect attacks, 12 unrelated
  requests (including Korean/English kimchi-stew requests), and 12 normal controls.
  The real service handles them; only retrieval/score context are replaced with
  the fixed attack fixtures. This measures a fixed challenge set, not all attacks.
- `full-final`: all existing 460 requests, with actual hybrid retrieval and the same
  953-unit corpus/model/settings used in the November 3 capture. Every request
  is regenerated. This is the sole new right-hand dataset for the final demo.

Claude Opus 5.5 CLI uses the unchanged prior rubric. First valid judgments are
accepted, regardless of their score. In the full cohort, an identical complete
question/answer/evidence/reference packet can reuse its prior judgment only when
model, rubric, schema and input/output hashes all match. Changed packets are
judged afresh. Reuse counts and the exact provenance are recorded per batch.
This avoids judging an unchanged answer again merely to obtain a different score.

BERTScore remains xlm-roberta-large, layer 18, no IDF, raw F1, with the existing
pinned model revision/settings. Full displayed answers, including qualifications,
are scored. Refusals/errors and unscorable lengths contribute zero to the effective
mean. The target applies to the existing 105 expert-based requests; meaning and
readability are evaluated independently. A high score does not establish accuracy.

## Resume and reproduce

Do not run two GPU generation captures simultaneously. Each script appends only
unfinished case IDs and refuses changed frozen inputs. Preserve completed rows
and all unsuccessful candidates. `prepare.py` creates a new directory rather than
overwriting a capture. It copies the prior runtime and the current source snapshot.

From the implementation checkout, run generation sequentially:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/helpful_responses_2026-11-04/security-hardened/run.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/helpful_responses_2026-11-04/full-final/run.py
```

Claude judging can run concurrently with GPU generation:

```bash
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/helpful_responses_2026-11-04/security-hardened/watch_judge.py
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/helpful_responses_2026-11-04/full-final/watch_judge.py
```

After all 460 outputs and judgments are complete and the generation GPU is free:

```bash
HF_HUB_OFFLINE=1 /home/minhee/.cache/soprano-rag-eval-venv/bin/python -u evaluation/helpful_responses_2026-11-04/full-final/score.py
python sites/answer-review/export_helpful.py
python sites/answer-review/verify_helpful.py
python evaluation/helpful_responses_2026-11-04/analyze.py
```

Before mirroring, review the five answers, new material-drift flags, naturalness
declines, security judgments and fixed-setting expert BERTScore. Browser QA uses
`sites/answer-review/tests/browser.cjs` with `REVIEW_STATIC_DIR`; no HTTP server is
needed. `mirror_helpful.py` checks captured destination hashes and replaces only
static assets and their serving documentation, with the catalog written last.
The user starts the localhost server themselves. No external hosting, Discord
notification, process-stop notifier, or usage/reset stopping rule is used.

Protected-checkout hashes/status are recorded in `protected-checkouts-before.json`.
The full suite passed 674 tests after the final source changes. Final evaluation
and demo verification results are in `verification.json` and
`browser-validation.json`. The three existing static demo mirrors were refreshed;
the old Lab backend was preserved. No generation or judge work remains pending.

The two-case first recovery probe exposed an evaluator aggregation bug with an
empty third worker. Its empty assessment file was supplied only after verifying
that worker's packet list was empty; existing judgments were retained unchanged.
New short cohorts and fresh-packet reviews limit workers to available packets.
The judge rubric, schema and ordinary joint runner remain byte-identical to the
prior evaluation so exact-packet reuse contracts can still be verified.
