# Qualitative evaluation viewer

This local, read-only viewer presents the selected paraphrased questions and
their saved retrieval and generation results from
`three_piece_results.json`.

## Synthesized development retrieval evaluation

`run_synthesized_retrieval.py` measures retrieval robustness on three unindexed
Korean reformulations of every schema-1.3 question for Die Forelle, In Flowery
Clouds, and La Capinera. This is a development set that informed tuning, not an
untouched holdout. It expands the same 51 canonical source/range cases as the
strict evaluator into 153 cases. The known invalid La Capinera
`kim-la-capinera-01` / measures 78-81 pairing remains excluded.

The benchmark dataset and evaluated system are separate arguments. This makes
it possible to evaluate clean or dirty code worktrees without copying the
runner into them:

```bash
SQA_VARIANT_DATASET=/home/minhee/soprano-qa-dataset-evaluation-set-synthesized

conda run -n soprano-qa python \
  evaluation/run_synthesized_retrieval.py \
  --system-root /home/minhee/soprano-qa-rag-system \
  --dataset-root "$SQA_VARIANT_DATASET" \
  --output evaluation/synthesized_retrieval_lexical.json \
  --top-k 6 \
  --require-retrieval-mode lexical

conda run -n soprano-qa python \
  evaluation/run_synthesized_retrieval.py \
  --system-root /home/minhee/soprano-qa-rag-system-dense-retrieval \
  --dataset-root "$SQA_VARIANT_DATASET" \
  --output evaluation/synthesized_retrieval_hybrid.json \
  --top-k 6 \
  --require-retrieval-mode hybrid
```

`--dataset-root` supplies only benchmark inventories, reviews, and
`expert_curation/evaluation_question_variants/*.json`. By default it does not
replace the target service's configured corpus dataset. This separation is
intentional: pointing corpus construction at the development dataset worktree
would change the corpus fingerprint merely because its absolute path differs.
Use `--pipeline-dataset-root` only when a different corpus source is an
explicit part of the experiment.

The runner imports `soprano_qa` exclusively from `--system-root`, performs all
153 calls in that process with generation and internal-knowledge fallback
disabled, and redirects corpus, stats, embedding-cache, and bytecode writes
away from the target worktree. It fingerprints target code, configuration,
corpus, and dense assets before and after the run. Outputs checkpoint after
each case, resume only when input and system fingerprints match, and refuse to
mix lexical, hybrid, fallback, or differently configured runs.

Each case records ordered evidence IDs, first expected and range-applicable
ranks, reciprocal rank, hit@1, hit@k, complete expected/applicable retrieval,
target-source grounding, and the pipeline's retrieval diagnostics. Summary
rollups are provided overall, by piece, and by synthesized variant index.
`formulation_consistency` reports whether all three formulations of the same
source/range case hit at `k`; its rate is the most direct robustness measure
for comparing lexical and hybrid retrieval.

## Five-piece qualitative run

The completed five-piece annotations can be evaluated separately without
pretending that the older schema-1.1 inventories have the strict claim-scope
contract used below. The first command saves real range-aware pipeline
answers and linked expert references; the second runs conservative local
LLM-as-a-judge triage whose default completed-run gate is 90%:

```bash
conda run -n soprano-qa python \
  evaluation/run_five_piece_qualitative.py --generate

conda run -n recsys-chall python \
  evaluation/judge_five_piece_qualitative.py
```

Both commands checkpoint each case and resume safely. Their default artifacts
are `five_piece_qualitative.json` and
`five_piece_qualitative_judged.json`. Retrieval ranks remain diagnostics;
the qualitative judge evaluates final-answer fidelity against the expert
source and range-applicable linked knowledge units. This protocol is useful
for all-five coverage and manual review, but it is intentionally distinct
from the stricter schema-1.3 protocol below.

## Reproducible three-piece reliability evaluation

`run_question_evaluation.py` evaluates all 40 annotator questions (51
measure-range inference cases) for Die Forelle, In Flowery Clouds, and La
Capinera. Nella fantasia and Una voce poco fa are deliberately excluded. One
La Capinera question/range pairing is also excluded because its lyric anchor
does not match the score in that reviewed range; the source annotation and
reviewed range remain unchanged.

The phases are separate and resumable:

```bash
SQA_JUDGE_MODEL=/home/minhee/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

conda run -n soprano-qa python \
  evaluation/run_question_evaluation.py \
  --phase retrieval \
  --judge-backend transformers \
  --judge-model-path "$SQA_JUDGE_MODEL"

conda run -n soprano-qa python \
  evaluation/run_question_evaluation.py \
  --phase generate \
  --judge-backend transformers \
  --judge-model-path "$SQA_JUDGE_MODEL"

conda run -n recsys-chall python \
  /home/minhee/soprano-qa-rag-system/evaluation/run_question_evaluation.py \
  --phase judge \
  --judge-backend transformers \
  --judge-model-path "$SQA_JUDGE_MODEL"
```

All three commands must use the same judge backend and model path because the
snapshot fingerprints its dataset, corpus, implementation, generator model,
judge model, and evaluation settings. If an input changes, use a new
`--output`; the runner refuses to mix incompatible results.

The generation phase must run in the `soprano-qa` Conda environment, where
`llama-cpp-python` is installed. Running it with a Python interpreter that
lacks `llama_cpp` produces an extractive fallback, which deliberately cannot
pass the grounded RAG+LLM metric.

`--dataset-root` may point at the separate development-variant worktree.
`--pipeline-dataset-root`, however, must resolve to the selected system's
configured `dataset_root`: the full evaluator fingerprints and authenticates
the already-built corpus before any pipeline call and will not rebuild it in
place. To evaluate a genuinely different corpus dataset, first prepare a
separate system worktree whose config, corpus, and stats all use that dataset.

The default output is the schema-8
`evaluation/three_piece_results.json`, produced by the v13 dual-frame judge
protocol. The runner:

- requires schema-1.3 question inventories for the three target pieces and
  verifies each question-level `reference_claim_scope` against the exact
  source-answer SHA-256 and complete ordered sentence/atomic-claim partition;
- uses curator-scoped, verbatim source-answer claims as the primary
  authority: `direct_required` is hard completeness,
  `optional_background` remains factuality authority but may be omitted, and a
  mixed source sentence is split into independently scoped atomic claims
  before judging;
- when an original source's legacy hint belongs to a different confirmed
  range inside a merged unit, retains overlapping contributor or unit prose
  only as factuality support, never as a new required claim, and records a
  mandatory reference-review exception;
- passes confirmed selected ranges to the pipeline as metadata and never adds
  measure numbers to paraphrased question text;
- disables the pipeline's internal-model-knowledge fallback, so a no-hit
  cannot masquerade as a grounded RAG answer;
- authenticates the expert records actually supplied to generation against
  the corpus ID, piece, source IDs, answer, ranges, review states, and warning,
  then reconstructs only the expert text that was visible in that exact
  range-selected prompt;
- authenticates confirmed non-overlapping records separately as
  `accepted_secondary_context`; they remain auditable but never become `S`
  factuality authority, never satisfy grounded-RAG eligibility by themselves,
  and do not create a review disclosure for an otherwise unused annotation;
- reconstructs the exact expected leading `검토 주의:` disclosure only from
  those authenticated corpus records, validates its exact boundary, and
  strips it exactly once before candidate `C` segmentation and all semantic,
  normalized-full-answer, atomic-claim, range, and NLI checks. The result is
  persisted separately as `review_disclosure_validation`, outside the
  semantic `R`/`S`/`C` packet;
- blocks an automatic pass when that disclosure is missing, mismatched,
  non-leading, extra, unexpected, or derived from untrusted evidence. A
  disclosure-only answer has no candidate `C` content and therefore cannot
  cover an `R` item;
- represents the question-scoped target claims as `R` items and authenticated
  retrieved-expert support as `S` items. Only `R` items are completeness
  targets; `S` items can establish that an additional generated claim is
  expert-grounded, but can never replace a missing `R` item;
- keeps a range-partitioned unit's combined multi-range rewrite out of `S`
  authority and admits only prompt-visible, range-applicable source material.
  Reliance on provisional retrieved support is sent to human review, while an
  evidence-integrity mismatch blocks an automatic pass;
- checkpoints by atomic replacement after every pipeline case and every judge
  attempt, and skips completed work when restarted;
- keeps expected-knowledge-unit and source-grounding retrieval data as
  diagnostics, separate from answer quality;
- applies claim-alignment and contradiction-first judge frames that
  exhaustively classify every scoped source claim and every generated
  answer segment, with bidirectional link auditing;
- independently guards literal pronunciation anchors and novel quantitative
  facts, so semantic similarity cannot hide a missing expert example or an
  invented regimen;
- independently corroborates required `R` coverage with candidate-to-reference
  KLUE NLI over the full answer, raw answer segments, and segments with only
  the Korean question subject restored for referent resolution. Every
  Qwen-covered curator-split atomic `R` is checked; when authenticated `S`
  context is present, every covered non-atomic `direct_required` or
  `mixed_or_ambiguous` `R` is also checked, preventing factuality support from
  being substituted for target completeness even when the judge omits an
  `S` link or the candidate paraphrases `S`;
- sends the semantic judge compact, positive, case-applicable `R` items plus
  authenticated `S` items with immutable IDs, authority roles, and review
  flags; raw or unverified retrieval payloads are not judge authority;
- runs a separate Qwen range-scope frame, then conservatively postprocesses
  each excluded atomic claim with local Qwen3-Embedding-0.6B contrast
  similarity and Korean KLUE NLI scores;
- computes all scores and pass/fail decisions in Python;
- preserves every raw judge response and keeps JSON parsing fail-closed. Only
  after the standard decoder points to one invalid `\'` escape may the
  evaluator remove that single backslash and retry, recording the repair in
  `deterministic_normalizations`; code fences, trailing text, arbitrary
  invalid escapes, and other malformed JSON remain errors;
- sends frame disagreement or confidence below 0.7 to human review and never
  counts it as a pass;
- reports expert-claim answer fidelity separately from the stricter
  auto-adjudicated reliability result. An internal-knowledge or extractive
  fallback cannot pass either grounded RAG+LLM metric even if its text happens
  to resemble the reference;
- persists per-piece case and question status counts and pass rates for answer
  fidelity, final judge adjudication, and retrieval diagnostics. Headline
  pass rates remain null until the run is complete; partial snapshots expose
  completed-only and end-to-end counts separately;
- exits with status 2 after a complete judge phase when the expert-claim
  answer-fidelity case pass rate is below the default 80% quality gate.

Use `--case-id SOURCE__mSTART-END` (or `SOURCE__no-range`) for a focused retry,
`--limit N` for a smoke test, and the explicit `--rerun` option to invalidate
the chosen phase and its downstream results for selected cases.

The Qwen3-4B Transformers judge is distinct from the Qwen3-8B GGUF answer
generator, and its chat template runs with `enable_thinking=False`. It is
still an automated judge from the same model family, not independent human
evidence. Its result is conservative triage; human review remains
authoritative, especially for annotations whose rewrite or measure status is
pending.

The primary evaluation objective is the semantic reliability of the generated
answer against the original expert content and applicable range-scoped source
content. `answer_quality_metric` is the primary content-quality result: it
normally records fidelity before a separate unresolved-curation gate. A
support-only recurrence is the deliberate exception: when no question-scoped
direct `R` authority applies, an otherwise supported answer is
`human_review` in both answer-quality and final adjudication rather than a
hard content failure or an automatic pass. Structured contradictions,
unsupported material claims, and failed range-scope checks still fail.
`primary_metric` is retained in the snapshot as the stricter, fully
auto-adjudicated operational result. A case sent to review is never silently
counted as an automatic pass in either status inventory. Exact source or
knowledge-unit recall is a retrieval diagnostic, not a substitute for judging
the generated answer.

Calibrate the judge before a full run:

```bash
conda run -n recsys-chall python evaluation/calibrate_judge.py \
  --judge-model-path "$SQA_JUDGE_MODEL"
```

The twenty-five controls cover faithful answers, an ambiguous secondary
omission that may pass or escalate but may never be falsely claimed as
covered, direct contradictions, unsupported mandatory vocal advice, cautious
versus overconfident handling of unresolved annotations, exhaustive
reference/candidate classification, merged-unit range disambiguation in both
directions, shared-vocabulary and negated-correction controls, paraphrased
range leakage, natural negation, double-negation ambiguity, and mixed
allowed/excluded range assertions. They also verify both sides of the
retrieved-expert boundary: an authenticated `S` item may support a truthful
extra claim, while an `S`-only answer cannot satisfy a missing target `R`
claim. The quality gate requires every pass/fail/review and risk-signal
expectation to match. A failed, stale, empty, wrong-contract, or
model-mismatched calibration artifact
blocks the evaluator's judge phase. This is a fixed regression gate, not a
statistical estimate of judge accuracy.

The unresolved meter/key simultaneity control alone accepts either an exact
claim-linked contradiction or a preserved unanchored
`contradicts_expert` warning as its risk signal. That warning is explicitly
review-only and pass-clamped. All other contradiction controls continue to
require the strict linked contradiction signal.

The semantic judge receives the normalized generated answer separately from a
validated expert packet. Target `R` IDs come from the dataset's exact scoped
source claims, including curator-split atomic claims. Supplemental `S` IDs are
built only from expert evidence recorded as actually used for generation,
authenticated against `data/corpus.json`, checked for piece and selected-range
applicability, and reduced to the material that the generation prompt could
see. The judge does not trust arbitrary evidence text or pipeline metadata.
Candidate `C` IDs come from deterministic answer segmentation.

Review disclosures are transport-level review metadata, not candidate answer
claims. For each case, the evaluator reconstructs the expected disclosure
from the same authenticated corpus records used to establish `S` authority.
The raw generated answer must begin with that exact `검토 주의:` text and a
valid boundary. The evaluator removes exactly one authenticated leading copy,
then gives the remaining body to `C` segmentation, semantic judging,
full-answer and atomic-claim checks, range reconciliation, and every NLI
guard. It persists the authentication outcome, expected text, contributing
evidence IDs, stripping status, and any failure reason separately in
`review_disclosure_validation`.

Generic warning-looking prose is never stripped. A missing, altered,
non-leading, repeated, extra, unexpected, or untrusted disclosure makes the
case ineligible for automatic pass even when diagnostics continue. If
stripping leaves no substantive body, the answer has no `C` item and cannot
satisfy `R` completeness. The viewer exposes this disclosure audit beside the
other evidence-integrity diagnostics.

Each `R` and `S` ID must be classified exactly once as covered, missing,
contradicted, or not required, and each candidate ID exactly once as
supported, contradicted, unsupported, mixed, or irrelevant. Only `R` items
participate in completeness. `S` items are factuality-only support: omission
is allowed, and covering an `S` item cannot cover or replace an `R` item. A
model cannot relabel an `R` item scoped as `direct_required` or
`mixed_or_ambiguous` as `not_required`.
Missing or `not_required` optional background does not block completeness,
but any candidate assertion about that background is still checked for
support or contradiction. Every semantic link is checked in both directions.
Redundant one-way links are reported but cannot overturn otherwise exhaustive
classifications; an assessment with an unlinked covered, contradicted,
supported, or mixed item cannot pass. Cross-field validation likewise
prevents an equivalence verdict from passing when an exhaustive
classification records a required omission, contradiction, or unsupported
material claim.

An independent required-claim NLI gate checks entailment in the useful
direction: the generated answer (or linked `C` segment) is the premise and the
required `R` claim is the hypothesis. It also scores a deterministic form that
restores only the Korean subject named in the question to an anaphoric answer
segment; this resolves references such as bare `악센트가` without supplying
any answer content. Every curator-split atomic `R` classified as covered is
guarded this way.

When authenticated retrieved `S` context exists, the gate additionally guards
every covered non-atomic `R` scoped as `direct_required` or
`mixed_or_ambiguous`. It does not depend on the judge linking the candidate to
`S` or on the candidate repeating `S` verbatim. This catches paraphrased
supplemental substitutions and omitted `S` links that would otherwise transfer
reliable retrieved factuality into target completeness. Entailment of at least
0.8 corroborates the `R` coverage. Neutral or contradictory local NLI does not
rewrite the Qwen label or independently force semantic failure; it blocks an
automatic pass and routes the case to human review. The reverse direction is
deliberately not required because a more specific expert sentence need not
entail a faithful concise paraphrase.

The range-scope judge is a distinct Qwen call. It receives the allowed
case-range claims, the excluded other-range claims, and the generated answer
segments, but excluded claims are contrast-only and cannot increase semantic
coverage. Its raw relation remains in the artifact. A deterministic hybrid
guard also compares the full candidate answer with every X claim and the
best-matching A claim using Qwen3-Embedding-0.6B, and runs a local Korean KLUE
NLI classifier both on the full answer and on every individual candidate
segment. Clause-level evidence may localize a relation to an exact candidate
ID; a full-answer signal alone is never allowed to fabricate those links. The
default entailment/contradiction threshold is 0.8 and the embedding contrast
delta threshold is 0.05. Asserted claims fail; residual ambiguity goes to
human review; confirmed negation and absence pass. Unresolved curator notes
may support a cautious caveat, but never resolve the underlying source
conflict. Duplicate X rows are preserved; exact duplicates are deduplicated,
while conflicting rows use clause NLI conservatively and can never become an
automatic pass. Model paths and thresholds are CLI/environment configurable
and their exact local records and hashes are bound into calibration and
evaluation fingerprints. Cases without an explicit contrast are marked not
applicable.

An answer-quality RAG+LLM pass requires LLM generation from authenticated
retrieved expert evidence plus a semantic-fidelity pass; it does not require
one exact target source or knowledge-unit ID. Expected-source and expected-KU
recall remain separate retrieval diagnostics. They help explain failures and
confirm pipeline routing, but they are not the quality verdict.

For a disconnect-safe sequential run, use
`evaluation/run_three_piece_detached.sh`. It first enforces judge calibration,
then resumes the same atomic snapshot and writes progress to
`evaluation/three_piece_run_status.json`. For example:

```bash
tmux new-session -d -s soprano-qa-3piece-eval \
  -c /home/minhee/soprano-qa-rag-system \
  "bash -lc './evaluation/run_three_piece_detached.sh \
  >> evaluation/three_piece_run.log 2>&1'"
```

From the repository root, run:

```bash
python3 evaluation/server.py
```

Open <http://127.0.0.1:8766/>. Use `--port` to choose another loopback port:

```bash
python3 evaluation/server.py --port 9000
```

The viewer supports:

- filtering by piece, annotator, review status, measure-range shape, answer
  basis, semantic result, and manual semantic-fidelity state;
- side-by-side comparison of the original and paraphrased question;
- separate tabs for every configured measure-range inference;
- an authority-first comparison of the exact per-case R items used by the
  semantic judge against the generated answer;
- a separate view of authenticated retrieved-expert S items that may support
  candidate factuality but never count toward target completeness, including
  evidence-validation and provisional-review diagnostics;
- the separate `review_disclosure_validation` audit, including the trusted
  expected prefix, contributing evidence IDs, exact-strip status, automatic
  pass eligibility, and failure reason;
- explicit reference-curation review reasons when automated results require
  manual adjudication;
- the global original annotation as source lineage, clearly separated from
  the exact range-scoped case authority;
- exhaustive per-claim and per-answer-segment semantic classifications for
  both judge frames, including their validated R/C links;
- literal, numeric-fact, and required-claim NLI guard decisions that prevented
  a raw judge classification—including an `S`-to-`R` transfer—from becoming
  an automatic pass;
- the separate hybrid range decision, including reviewed A/X claims, the raw
  LLM relation, full-answer and clause-level NLI probabilities, embedding
  contrast delta, link provenance, decision reasons, and guard thresholds;
- collapsed secondary KU, retrieval, evidence, grounding, and raw-JSON
  diagnostics;
- question wording and inference semantic-fidelity ratings, issue tags, and
  notes;
- export and import of a compact manual-review JSON file.

The server prefers `evaluation/three_piece_results.json` when it exists and
falls back to the legacy manual snapshot. Use `--results-file PATH` to select
an exact read-only snapshot.

Manual semantic-fidelity review is enabled only after the run declares itself
complete and every inference case has an aggregate judge result. This prevents
an in-progress checkpoint from invalidating or silently migrating local review
state. Once enabled, review is automatically saved in the current browser's
`localStorage`, keyed to the exact inference snapshot. It is not written into
the result snapshot. Export the review JSON if it needs to be retained, shared,
or version-controlled.

Focused checks:

```bash
python3 -m unittest -v \
  tests.test_question_evaluation \
  tests.test_evaluation_viewer_server
node --test evaluation/test_review_state.js
```
