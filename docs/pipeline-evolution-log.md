# Soprano QA pipeline evolution log

This is the living handoff document for generation and retrieval experiments.
Update it **before starting a materially different pipeline version** and again
after that version has been evaluated. It records why an approach existed,
what failed, concrete failure samples, preserved rollback points, and what must
be true before a newer approach replaces it.

The stable method description remains in
[`rag-llm-pipeline-method.md`](rag-llm-pipeline-method.md). This file is the
chronological engineering record, including unsuccessful approaches.

## Current status (2026-08-04)

- Active repository work is based on `5eed0fc`, with pipeline, evaluation,
  viewer, corpus, documentation, and test changes in progress.
- Generation model: `Qwen3-14B-Q6_K.gguf` through `llama-cpp-python` in the
  `soprano-qa` Conda environment. The 14B size is the project ceiling.
- Retrieval model: `Qwen3-Embedding-0.6B-Q8_0.gguf` in required hybrid mode.
- Evaluation contract: 105 original cases plus 355 synthesized-family cases
  across all five pieces. The latter consists of 306 paraphrase cases and 49
  questions derived from finalized knowledge units that lacked a related
  original question.
- Active baseline run directory:
  `/home/minhee/soprano-qa-evaluation-final-20260804-q6-v7-7DgWml`.
- The detached baseline run has finished. All 460 cases returned non-empty
  `llm/retrieved_evidence` answers and had no refusal, footer, record-ID leak,
  or prompt-field leak.
- Direct Codex review covers exactly 105 original and 355 synthesized-family
  cases, without a missing, extra, duplicate, or piece-mismatched assessment.
  It flagged 28 original answers and 65 synthesized-family answers: 93 of 460
  overall (28 high, 43 medium, and 22 low severity).
- The largest recurring classes are lost qualifiers and unsupported causal or
  meaning changes. Combined flags by piece are Die Forelle 6/58, In Flowery
  Clouds 17/74, La Capinera 22/118, Nella fantasia 13/75, and Una voce poco fa
  35/135.
- Therefore this one-pass run is a **baseline, not a promotable final result**.
  The generated answers remain available for comparison and red-flag review.

## RAG+LLM approach branch/worktree registry

Only branches and worktrees created to preserve or develop a distinct RAG+LLM
pipeline approach belong in this registry. Dataset, demo, and ordinary
evaluation-output directories are deliberately excluded.

| Experimental snapshot | Branch | Worktree | Commit | State and purpose |
| --- | --- | --- | --- | --- |
| Strict grounding experiment | `codex/strict-grounded-pipeline-snapshot` | `/home/minhee/codex/strict-grounded-pipeline-snapshot` | `e3816ad` | Clean committed snapshot of the strict one-pass design that could reject/abstain. It is intentionally unmerged and retained only as a reproducible comparison/rollback point. |
| Pre-fidelity-editor baseline | `codex/pre-fidelity-editor-snapshot` | `/home/minhee/codex/pre-fidelity-editor-snapshot` | `7e364b6` | Clean committed snapshot of the answer-required one-pass code immediately before the proposed fidelity-editor change. It is intentionally unmerged and is the primary rollback point for the next experiment. |
| Atomic knowledge-unit experiment | `codex/atomic-claim-generation` | `/home/minhee/codex/atomic-claim-generation` | `3864df0` | Clean committed two-stage Qwen experiment with exhaustive full-KU planning and conservative rendering. It fixes the targeted meaning-drift cases but remains unmerged because broad-answer completeness and natural cohesion regress. |

All three snapshot worktrees are committed and clean. Their experimental branches
are local and have not been pushed.

For every future material RAG+LLM approach:

1. create a `codex/...` branch and linked worktree before modifying the active
   approach;
2. commit the complete reproducible approach in that worktree—do not leave a
   rollback worktree dirty;
3. add its branch, path, commit, purpose, evaluation artifact, known failures,
   and merge/removal status to this table;
4. do not remove the worktree or branch until the replacement has been fully
   evaluated, committed on `main`, and the rollback is no longer useful.

## Non-negotiable product behavior

These constraints survive every approach change:

1. A user-facing request must receive a useful answer; semantic uncertainty is
   represented by a review red flag, not by an abstention message.
2. The answer may use only retrieved, finalized evidence. There is no internal
   model-knowledge, lexical, extractive, or alternate-model fallback.
3. Missing embedding or generation checkpoints and missing native backends are
   startup errors, not reasons to silently change the pipeline.
4. Answers use Korean formal-polite speech consistently and expose no
   knowledge-unit IDs, temporary evidence labels, retrieval metadata, footer,
   or editorial state.
5. A measure-local finalized unit may answer a no-range question when the
   semantics match. It must not be promoted into a whole-piece frequency,
   universality, or location claim.
6. Exact linked-KU retrieval is a diagnostic, not an answer-quality verdict.
   A semantically correct answer may use another compatible finalized unit;
   retrieving an expected ID cannot make an incorrect answer pass.
7. Every final artifact must cover all five pieces: 105 original and 355
   synthesized-family cases. Semantic review covers all 460 answers and is
   bound to each exact case with `semantic_review_case_hash`.

## Approach 1: legacy hybrid RAG with presentation cleanup

Relevant main commits include `35c257c` (tuned hybrid RAG evaluation),
`a80d24a` (remove retrieval footers), and `6d38a3d` (fail fast when hybrid
retrieval is unavailable).

### What it tried

- Retrieve a small set of expert/web records with lexical and embedding
  signals.
- Let a local Qwen model compose an answer.
- Retain fallback paths when retrieval or generation was weak.
- Append or expose retrieval provenance in answer text in some earlier output
  formats.

### Why it was replaced

- A fallback could hide a missing model and silently change the system being
  evaluated.
- Some user-visible answers included `제공된 검색 근거`, opaque KU IDs, scope
  notices, or other pipeline details.
- Intermediate annotation fields could reach the corpus even though only the
  finalized reviewed knowledge-unit answer was authoritative.
- Three-piece and duplicated footer/no-footer evaluations made it difficult to
  tell which artifact represented the product.

### Durable decisions from this phase

- Models and native backends are mandatory and validated before corpus work.
- Runtime logs and `*.json.lock` files are ignored, not committed.
- Only one current original artifact, one synthesized-family artifact, and one
  semantic-review artifact are promoted.
- The release corpus is generated from finalized reviewed knowledge units and
  excludes raw `measure_notes`, conflicts, and mid-review prose.

## Approach 2: strict one-pass grounding experiment

Rollback point: worktree
`/home/minhee/codex/strict-grounded-pipeline-snapshot`, branch
`codex/strict-grounded-pipeline-snapshot`, commit `e3816ad`.

### What it tried

- Add detailed scope, causality, modality, agent, negation, and citation rules
  to one system prompt.
- Select a compact evidence bundle and require sentence-level temporary
  evidence labels.
- Reject a draft when deterministic validation considered it ungrounded.

### Problems

- Rejecting a draft recreated the product behavior the project is trying to
  avoid: some questions received no useful answer.
- Highly defensive selection could discard complementary expert units and make
  broad performance answers incomplete.
- A long list of safeguards did not reliably stop a 14B generator from
  strengthening a possibility into a fact or inventing a connective relation.
- The prose became less natural as case-specific constraints accumulated.

### Decision

The strict state was preserved in its own worktree rather than deleted. Main
returned to an answer-required design: one draft is always displayed, while a
separate direct semantic review records concerns.

## Approach 3: five-piece answer-required one-pass baseline

Rollback point before the next architecture experiment: worktree
`/home/minhee/codex/pre-fidelity-editor-snapshot`, branch
`codex/pre-fidelity-editor-snapshot`, commit `7e364b6`.

### What it tries

- Use the Qwen3 14B Q6 model for one answer-generation call.
- Retrieve finalized expert units with equal semantic authority regardless of
  whether their confirmed scope is whole-piece or measure-local; range remains
  an applicability boundary rather than a quality hierarchy.
- Let several directly useful units form a complete answer, while semantic
  focus removes unrelated neighbors.
- Always return the sanitized model draft. Deterministic grounding concerns
  become `generation_validation_warning`; they never trigger an extractive or
  internal-knowledge substitute.
- Keep semantic acceptance outside the generation code. Direct Codex review
  marks red flags, and `/red-flags` displays them without hiding the answer.

### What improved

- The preflight retrieval audit covers all 460 cases with zero empty evidence
  bundles. Expected-ID diagnostics are Hit@1 346/355, Hit@3 352/355, and
  Hit@6 352/355; the three misses have semantically valid alternative units.
- All 105 original baseline calls returned grounded LLM answers.
- Mechanical presentation checks found no refusal, empty answer, record ID,
  evidence footer, or internal prompt vocabulary.
- Whole-song semantic retrieval now finds local expert advice for cases such
  as the Una voce interlude and the In Flowery Clouds opening fermata.
- Broad multi-facet routing preserves requested facets rather than allowing an
  embedding-only neighbor to replace the router bundle.

### Why it is not final

The one-pass model still paraphrases beyond the reviewed claim. The errors are
not merely exact-ID misses; most occur even when the correct single KU is the
only evidence.

#### Sample A: unsupported decision procedure and lost local scope

- Question: `악보의 페르마타 표기와 음원에서 들리는 음의 길이가 다르면 어떻게 해야 할까?`
- Reviewed claim: the **opening** fermata's presence or absence may be an
  arrangement difference, and both versions are allowed.
- Baseline answer adds: `연주 시 상황에 따라 유연하게 처리하면 됩니다.`
- Problem: the answer drops the opening-only qualifier and invents a general
  decision procedure that the reviewed unit never gives.

#### Sample B: interpretation promoted to composer intent

- Question: `박자가 바뀌는 이유는 무엇이며, 곡의 분위기는 어떻게 달라질까?`
- Reviewed claim: meter/key/accompaniment changes accompany an atmosphere
  transition and hardship imagery; the composer changes meter and key to
  transition the atmosphere.
- Baseline answer says the composer changes them `이미지를 표현하고자`.
- Problem: it joins two supported relations into a new intentional causal
  chain. This is the same high-risk agent/meaning change that motivated the
  strict experiment.

#### Sample C: permissive qualifier changed into a prescription

- Question: `크레센도 부분이 잘되지 않으면 어떻게 해야 할까?`
- Reviewed claim: the voice naturally tends to strengthen toward a high note,
  and it is acceptable if that produces a crescendo effect.
- Baseline answer says the singer should progressively increase the sound and
  that doing so is important.
- Problem: `자연스럽게 ... 커진다고 해도 괜찮다` becomes an active rule.

#### Sample D: condition boundary reversed

- Question: `장식음은 어떻게 처리할까?`
- Reviewed claim: **unslashed** ornaments usually divide the beat with the
  principal note; **slashed** ornaments are often sung before it.
- Baseline answer opens with equal division as the general ornament rule and
  later suggests applying the same method elsewhere.
- Problem: the condition-specific rule is generalized before the contrast is
  stated, making the answer unsafe for slashed ornaments.

#### Sample E: qualifier reversal in diction advice

- Question: `‘linguaggio’의 겹자음을 더 잘 발음하려면 어떻게 해야 할까?`
- Reviewed claim: a doubled consonant is often produced like one consonant but
  should feel like two articulations, with a short pause between them.
- Baseline answer says it should **not** flow like one consonant.
- Problem: a subtle production qualifier is reversed.

#### Sample F: malformed but user-visible Korean

- `끌어 부르다` became `2~3배로 끌어내리는 것` for a fermata.
- `이탈리아어` became `이탈리어`.
- `끝맺음을 짓는 것` became `끝맺음을 지는 것`.
- These are not metadata leaks, but they can change the requested action or
  make a polished demo look unreliable.

### Repeated error classes

1. `가능하다`, `볼 수 있다`, `경우가 있다`, and recommendations become
   categorical facts.
2. Separate evidence relations are joined into a new purpose, cause, or
   composer intention.
3. A local example loses its passage qualifier in a no-range answer.
4. Contrasting rules are flattened into one universal technique.
5. A related but unrequested unit contributes distracting advice.
6. The model appends an unsupported emotional payoff to finish a paragraph.
7. Korean synonyms or conjugations become less accurate than the reviewed
   sentence they replace.

## Approach 4: conservative fidelity-editor experiments

Status: **tested on a targeted A/B set and rejected**. Neither experimental
prompt was merged into the production pipeline.

### Reason for a new version

Adding more rules to the already long first-pass prompt has not prevented the
same semantic transformations. A second, concise task may be easier for a 14B
instruction-following model: edit an already useful draft against the exact
reviewed evidence rather than solve retrieval, synthesis, style, scope, and
faithfulness simultaneously.

### Tested design

1. The first arm kept the current retrieval bundle and gave the same
   Qwen3-14B-Q6 model the exact question, finalized evidence, and first-pass
   draft. It asked the model to act only as a conservative fidelity editor.
2. The second arm removed the first draft to avoid anchoring. It asked the
   model to produce a short answer directly from the same evidence while
   preserving scope, modality, conditions, causal direction, and multiple
   useful units.
3. Both arms used flagged examples plus clean multi-unit controls. Neither arm
   had an extractive, alternate-model, or internal-knowledge fallback.

### Result and decision

- The draft-editing arm repaired the unsupported generic advice in the
  opening-fermata example, but copied most causal, condition, and qualifier
  errors from the first draft. It also introduced plain-register endings in
  two cases.
- Removing the draft improved several cases: it preserved the permissive
  crescendo wording, retained the two ornament conditions, and combined the
  three Die Forelle guidance units without the earlier repetitive framing.
- It still fused atmosphere imagery into a new purpose for changing meter,
  changed `2~3배로 부르는` into `2~3배로 끌어내리는`, lost the opening-only
  scope, omitted a material doubled-consonant qualifier, and produced malformed
  Korean in the Nella fantasia control.
- A second general-purpose prompt therefore does not supply an independent
  enough semantic safeguard when it uses the same 14B model. The approach is
  rejected rather than expanded into a 460-case rerun.
- Enabling Qwen's native thinking mode did not rescue the short direct arm. It
  added unsupported darker/tension imagery and emotional flow, strengthened a
  common interpretation into composer intention, repeated the doubled-
  consonant reversal, and invented extra performance instructions. Reasoning
  mode is therefore rejected for this answer path as well.

## Approach 5: atomic knowledge-unit planning and rendering

Status: **targeted experiment implemented, committed, and not accepted** on
branch `codex/atomic-claim-generation`, worktree
`/home/minhee/codex/atomic-claim-generation`, commit `3864df0`. It is not
merged and has not replaced the one-pass baseline.

### Reason for a new version

The failed editor and thinking experiments show that more prose instructions
do not lock meaning. The directly retrieved expert unit is usually correct,
but a multi-relation paragraph lets the model join two valid relations into a
new invalid one. Key predicates and qualifiers also remain free to drift.

### Candidate design

1. Treat every selected finalized knowledge unit as one indivisible U block.
   This avoids splitting `D.S.`, `Op.`, names, or anaphoric sequences such as
   `이는` and `따라서`, and guarantees that rendering cannot silently delete a
   sentence inside a selected unit.
2. The first call makes an exhaustive plan. It classifies every U block once
   and must return at least one ordered `selected_unit_id`; temporary labels
   are never real KU IDs and never reach the user.
3. The second call receives only the selected full texts. It returns one C
   block per selected unit, preserving all sentences and changing only formal
   speech level, spacing, or minimal grammar.
4. Schema checks enforce complete ordered accounting and one-to-one hidden
   mappings. A 0.99 lexical-fidelity threshold plus protected semantic markers
   catches changes such as `부르는` to `끌어내리는`; protocol labels, schema
   keys, JSON, and opaque IDs are rejected at the display boundary.
5. A nonempty answer that fails a semantic/fidelity check is still displayed
   with `generation_validation_warning` for red-flag review. There is no
   extractive, internal-knowledge, alternate-model, or first-draft fallback.

### First targeted result

The integrated targeted run preserved the opening-only fermata scope, kept
the Flowery meter-change and imagery relations separate, retained the
permissive crescendo wording, kept `2~3배로 부르는`, preserved the doubled-
consonant qualifier and popular-song condition, retained the common-
interpretation wording for Rossini's dotted rhythm, and kept the
slashed/unslashed ornament contrast. The Una accent case initially produced
an empty plan; requiring at least one model-selected unit repaired that
answer without choosing a retrieval result in application code. The affected
unit/service suites pass 137 tests, and the snapshot is clean and committed.

The experiment is not accepted. The planner still omits directly useful
support in broad questions: the whole-piece word-boundary breathing rule in
Nella fantasia, and—depending on the run—La Capinera opening diction or entry
timing. The Die Forelle control retains all three expert units but reproduces
the same adjacent `매우 중요합니다` framing that the user identified as
unnatural. In other words, exact full-KU rendering greatly reduces dangerous
meaning changes but sacrifices stable completeness and natural multi-unit
cohesion. It was therefore committed as a comparison snapshot without a full
460-case rerun or a merge into `main`.

## Evaluation and promotion checklist

For each new approach, record all of the following here before calling it
final:

- code/worktree/commit used for rollback;
- exact model and quantization;
- original and synthesized case counts for all five pieces;
- zero empty/refusal/unavailable/mechanical-leak cases;
- retrieval Hit@1, Hit@3, Hit@6, and MRR as diagnostics;
- direct semantic-review count and severity distribution;
- representative new flags and regressions from clean controls;
- whether every red-flag assessment matches the current semantic hash;
- `/api/results`, `/`, and `/red-flags` validation with the promoted files;
- complete system test result and the relevant dataset-repository test result.

Do not overwrite the promoted artifacts or declare a new approach current
until this checklist is supported by fresh evidence.
