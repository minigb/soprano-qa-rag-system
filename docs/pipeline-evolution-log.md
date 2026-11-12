# Soprano QA pipeline evolution log

This is the living handoff document for generation and retrieval experiments.
Update it **before starting a materially different pipeline version** and again
after that version has been evaluated. It records why an approach existed,
what failed, concrete failure samples, preserved rollback points, and what must
be true before a newer approach replaces it.

The stable method description remains in
[`rag-llm-pipeline-method.md`](rag-llm-pipeline-method.md). This file is the
chronological engineering record, including unsuccessful approaches.

## Single-pass service and stage timing (2026-11-04)

The user requested existing hybrid retrieval, prompt-based task scope, a fixed
14B generator, and no online output assessment. Work stays on
`codex/helpful-rag-responses`; `f71d5fe` preserves the preceding implementation.
The initial pre-injection worktree and both main repositories are unchanged.

Before changing the service, 20 fixed questions were measured twice with actual
completion timers. The initial observer attached to a differently keyed cached
model instance; that capture is explicitly invalidated and retained. The valid
`baseline-measured` capture uses identical cache arguments and asserts that
inference is observed. Ordinary questions average 8.720 s: 6.322 s writing,
0.422 s admission, 0.339 s extra KU selection, 1.556 s output assessment,
0.033 s retrieval, and 0.049 s other work. All five prior refusals trigger the
slow path (23.289 s average, usually seven completions). This enriched cohort
is diagnostic, not a traffic-weighted latency estimate.

`single_pass.py` removes model-based admission, second KU selection, output
review and retries from the default service/CLI. Korean uses one writing call;
English retains the existing query translation and one writing call. Retrieval
and the 14B checkpoint are unchanged. Plain prose is not replaced by source
quotes, scores or a canned musical answer. Legacy paths remain explicit.

Candidates and all observed failures are retained under
`evaluation/single_pass_2026-11-04`. A simplified writing prompt produced excess
general commentary; shorter B improved latency but merged dynamics notation.
C restored existing source-derived meaning hints. D (directory
`single-pass-final`, a superseded name) fixed bar/verse wording but expanded p
incorrectly and invented a preference between source accounts. Its profile
and partial expert run are preserved; the name does not make it a final result.
E also retains the previous comparison-writing instructions inside the sole
generation call. It adds no source selector, output reviewer or retry.

E's plain-prose version was also rejected: the profile included a cookie recipe
under a direct injection, and expert review found source contradictions such as
breathing inside a word where the source warns against doing so. Its full 109
expert/regression outputs and first Claude judgments are retained. F therefore
keeps the previous semantic composition/comparison instructions and structured
draft format, with a prompt-based task boundary. The application only joins
generated text fields; it does not validate citations or evaluate the answer,
and it never substitutes a source text. There is still exactly one writing call.

F's profile still produced long repetitive source-like paragraphs and substantial
writing latency despite one call. It was not adopted. G restores the initial
free-prose writing instructions from `f307470`, adapting their source-only and
internal-citation restrictions to the user's general-knowledge/direct-delivery
requirements. Its source-disagreement and meaning hints remain input preparation
only. One G timing capture was invalidated because an orphaned F process briefly
overlapped it; `single-pass-g-clean` starts after confirming the GPU is free.
All invalid and unsuccessful captures remain separately labelled.

G's offline checks cover the unchanged expert 105 plus four additional former
refusal paraphrases, and the prior 74 bilingual challenge fixtures. BERTScore
settings and references stay fixed. These are regression checks, not a fresh
holdout or a new 460-case demo capture. The final report records measured
latency, quality differences and limitations; no test result is used to alter
an individual delivered answer.

Final G results: expert-105 BERTScore 0.928871, but material meaning drift rises
from 2 to 11/105 and naturalness falls from 3.543 to 3.448/4. Direct attack success
rises from 0 to 8/30 and unrelated task completion from 0 to 3/12. Ordinary
diagnostic latency falls from 8.720 to 3.968 s; previous-refusal latency from
23.289 to 4.367 s. The candidate is not promoted. The validated default config
is restored byte-for-byte, and `config/settings.single-pass.json` explicitly
selects the experimental path. The old backend and all 460-case demo assets
remain unchanged. 684 tests and source/model/corpus/preservation checks pass.
See `evaluation/single_pass_2026-11-04/report.md` for the final comparison.

## Active experiment: bounded general knowledge (2026-10-31)

Development checkpoints on `codex/ku-background-knowledge`:

- A (24-question pilot): typed source/background/lyric parts, but general
  concept questions were often rejected and related KUs were used as if they
  answered a different concept. Preserved under `ku_background_2026-10-31`.
- B (`6420d2c`, 55 development requests): explicit general-music admission
  wording. Expert answers: 38/43 delivered; Claude found 3 material and 14 minor
  source drifts. General answerable questions: 4/10 delivered. The best of 18
  BERTScore settings was XLM-R large layer 18 without IDF: 0.838086 effective
  F1, 0.948361 conditional on delivered answers. No validation answers were
  used to choose this setting. B is not a successful release.
- C (`9c30037`, 55 development requests): admission sees the original
  request rather than retrieval/score contents, and generation preserves
  complete relevant explanations. Literal scoped source sentences can pass
  the coarse frequency screen without authorizing new whole-piece claims.
  Expert delivery 37/43 and effective F1 0.823354: it did not improve B.
  Lyric diction needed explicit admission context.
- D (`a9fee80`, 55 development requests): fix Unicode ellipsis citation boundaries
  and colon-in-parentheses register checks; normalize only unambiguous
  `있어요`/`없어요` endings. Clarify music diction admission and narrative
  source scope. An output-review denial may trigger at most one fresh draft
  with the same frozen evidence, schema and policies; the replacement needs a
  new affirmative output verdict. Admission failures, malformed verdicts,
  input/output text screening and backend failures never use this retry.
  Expert delivery 41/43 and effective F1 0.909693 under B's frozen metric.
  Claude v2 found 2 material expert drifts; general delivery was only 6/10.

The 555 system tests pass for the prepared D source. The experiment's frozen
protocol, rather than the old source-only release checklist below, governs
general-knowledge evaluation and legitimate refusals of unknown local facts.
No experiment here changes the live service or published demo answer files.

Next development hypothesis E: D's early general-concept answers still attach
related but non-answering passages (for example, legato advice to a fermata
definition). A semantic selection step will be evaluated for bundles containing
multiple primary sources. It can only retain existing primary citation labels
or return an empty list; it cannot write new facts or promote context-only
records. It runs after task admission and source screening, before composition.
The generator may then use marked general knowledge when nothing directly
answers the question. A malformed selection fails closed. This is prompted
relevance filtering, not a trained Self-RAG implementation or a security proof.
Whether its extra inference cost and possible selection omissions are worth it
must be decided on the same frozen development questions before validation.
E (`5f75039`) delivered 41/43 expert answers, F1 0.912229, with one material
expert drift and one material general drift under Claude v2. E2 (`35d47af`)
had the same delivery and metric, two material expert judgments and one
material general judgment. Separate judge calls have variance, so the final
comparison will deduplicate identical packets and judge them only once.

E2 hypothesis, before running it: E's selector correctly returned no passages
for note-value definitions, but the source-oriented composer then returned an
empty answer. Use a dedicated general-music system prompt when the selected
bundle has no primary source. It retains the task/injection boundary, explicit
unknown-score limits, requested language and typed output; it does not ask a
general explanation to satisfy irrelevant KU synthesis instructions. Limit the
schema's kind choices to those actually possible for the supplied bundle.

E3 hypothesis, before running it: E2 still returned empty arrays for elementary
note-value/glossary questions. An isolated development diagnostic compared the
current grammar, a required item, a background example and unconstrained JSON.
Only the required item produced the correct substantive answer. In the general
path require an explanation **or an explicit limitation**, using a matching
format prompt; unknown local facts must still be withheld and output review
remains mandatory. No minimum factual claim is required. Also make explicit
that replacing a source's technique with another term is not a faithful
definition, and that omitting a condition can widen a permission. These changes
are hypotheses to evaluate, not grounds to dismiss earlier failed cases.

E4 development hypothesis: E3 now answers both note-value questions, but its
English answer added an unrequested, false 60-BPM example. Restrict general
explanations to requested details and ask for units/proportions to be checked.
Do not add assumed numerical demonstrations unless needed. This changes only
the general prompt; it is not a special answer rule for a benchmark case.
Record the E3 error and evaluate E4 rather than silently replacing its output.

E3 completed 55 requests: 41/43 expert answers, effective F1 0.912330. Its
general answers still confused Korean articulation terminology. E4 was tested
in isolated empty-bundle development diagnostics, not a full service run.
Greedy constrained generation, the Qwen-recommended sampled settings, and
parsed thinking-mode JSON all retained serious Korean glossary errors. Adding
original Italian term names helped portamento but did not reliably fix tenuto
or notation. These diagnostics are not held-out measurements.

E5 hypothesis before evaluation: keep the current structured generation and
add a short bilingual terminology aid, checked against a published music
theory lesson, to composition and output review. This is explicit general
reference guidance, not new expert KUs or references for BERTScore. It cannot
establish local score facts or overwrite source-specific performance advice.
The frozen expert references and BERTScore setting are unchanged. This is
development-informed domain guidance; no claim of unseen glossary performance
is warranted from these development questions alone.

E6 deterministic scope correction, identified from an E3 withheld draft before
validation: a faithful "in phrases where ornaments often appear" clause was
misread as a new whole-piece frequency assertion. Permit only a source-matching
relative clause with the same subject, frequency and predicate; mask just its
frequency word and continue screening the rest of the claim. Broader claims,
changed subjects and unrequested measure numbers remain rejected. This does
not change generation prompts or model settings, and does not weaken semantic
review. The separate mixed-register Nella fantasia draft remains withheld;
it also contains questionable generalization, so fixing its register alone
would not establish a correct answer.

E7 routing hypothesis: E5 fixed the basic definitions, but some mixed drafts
cited piece-specific legato advice as if it defined portamento. Put the
terminology aid only in the general composer (and output reviewer). Extend
relevance selection to every nonempty primary bundle and require an explicit
general-definition versus evidence mode. Abstract definitions use an empty
bundle and the general prompt; performance/history/local-score questions retain
relevant sources. This is semantic routing, not a list of benchmark questions.
The generator cannot cite a discarded source. Full development evaluation
must check that expert questions are not incorrectly routed away from their KUs.

The E6 scope regression test also exposed missing formal verb forms in the
old broadening pattern: it caught `자주 나타나는` but missed `자주 나타납니다`.
Both forms are now covered; only a source-matching relative clause receives
the narrow exception described above.

E8 hypothesis: E7's combined intent/relevance call let triplet and legato
passages redefine an abstract fermata-duration question. Separate question
routing from passage selection. The router sees only the raw question and
selected range, after task admission; it never sees a piece title, source or
lyric that could supply an irrelevant interpretation. Only explicit abstract
definitions/comparisons/general theory rules route to the general composer.
Performance/history/interpretation questions and uncertainty default to
evidence. Multi-passage relevance selection then retains its original label-only
contract. This adds a bounded question-only call for nonempty bundles.
Also request one coherent general paragraph (two items only when separate
score lyrics are needed); longer multi-item defaults had encouraged unrequested
and sometimes false notation examples. The source composer still allows six
items. Both changes require fresh development measurement before validation.

E7 also showed that the E6 source-matching exception had not fixed the original
leggiero refusal: its source uses different but compatible relative clauses.
E8 therefore separates the scope check from entailment: an explicit "in phrases
where X often appears" construction is not itself a whole-piece assertion.
Mask only its frequency token and still reject other broadening in the claim.
The actual historical source and generated clause now form the regression
test. Citation authority, selected-measure checks and semantic support review
remain required; this syntax rule does not establish that X or the advice is true.

The user authorized general musical knowledge as a supplement to curated KU
answers, a separate Claude Opus 5.5 CLI judge, and a BERTScore F1 target of 0.90
on expert-annotation questions. This explicitly supersedes the historical
internal-knowledge prohibition below for this experiment. The baseline remains
preserved at `302f8d1`; work is isolated on `codex/ku-background-knowledge`. The
[protocol](../evaluation/ku_background_2026-10-31/protocol.md) fixes provenance,
development/validation separation, metric settings and quality/security checks.
Results are pending; no claim of improvement or promotion is made yet.

## Latest meaning-fidelity diagnostic (2026-10-31)

The security refinement branch `codex/refine-rag-quality` at `79cbd7b` was
compared with the secured pre-refinement baseline `29fe3e2` on 41 targeted
cases per system. The [report](../evaluation/meaning_fidelity_2026-10-31/report.md)
and [side-by-side answers](../evaluation/meaning_fidelity_2026-10-31/review.html)
record 75 delivered-answer reviews and seven abstentions. On the 39 service
requests per variant, material meaning errors increased from 14 to 19; the two
fixed-evidence conflict probes are reported separately. Ten new material
regressions received the same evidence text, order and scope in both systems.

Reproduced failures include consonant omission becoming emphasis, the
one-consonant qualifier being negated, invented vowel connections, and a
duration instruction becoming `2~3배로 끌어내리는`. The output guard passed all
20 materially wrong current answers across the 41-case inventory. Some older
problems improved, but the current candidate is not validated for meaning
preservation. Earlier aggregate quality scores do not override this finding.

This is a frozen diagnostic, not a new generation approach or a full rerun of
the 460 historical cases. No generator, main branch, running service, or
published static answer was changed by the audit. Prompt exports include the
structured-format prefix that `llm.generate()` prepends to `system_prompt()`.

## Historical status (2026-08-04)

- Active `main` work remains the answer-required one-pass baseline, with
  pipeline, evaluation, viewer, corpus, documentation, and test changes in
  progress.
- Generation model: `Qwen3-14B-Q6_K.gguf` through `llama-cpp-python` in the
  `soprano-qa` Conda environment. The 14B size is the project ceiling.
- Retrieval model: `Qwen3-Embedding-0.6B-Q8_0.gguf` in required hybrid mode.
- Evaluation contract: 105 original cases plus 355 synthesized-family cases
  across all five pieces. The latter consists of 306 paraphrase cases and 49
  questions derived from finalized knowledge units that lacked a related
  original question.
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
- 2026-10-17: Approach 7 adds the dataset's `web_coaching` lineage and a
  gated evidence supplement on top of this baseline; see its section below.

## RAG+LLM approach branch/worktree registry

This registry is intentionally limited to committed snapshots of distinct,
new RAG+LLM pipeline approaches. Do not list routine development branches,
baseline-only rollback points, dataset-repository branches, demo worktrees,
evaluation-output directories, or historical cleanup work here.

| Experimental snapshot | Branch | Worktree | Commit | Dataset commit | State and purpose |
| --- | --- | --- | --- | --- | --- |
| Strict grounding experiment | `codex/strict-grounded-pipeline-snapshot` | `/home/minhee/codex/strict-grounded-pipeline-snapshot` | `e3816ad` | not pinned | Clean committed snapshot of the strict one-pass design that could reject/abstain. It is intentionally unmerged and retained only as a reproducible comparison/rollback point. |
| Atomic knowledge-unit experiment | `codex/atomic-claim-generation` | `/home/minhee/codex/atomic-claim-generation` | `3864df0` | `93415f0` | Clean committed two-stage Qwen experiment with exhaustive full-KU planning and conservative rendering. It fixes the targeted meaning-drift cases but remains unmerged because broad-answer completeness and natural cohesion regress. |
| Source-locked KU rendering experiment | `codex/source-locked-ku-rendering` | `/home/minhee/codex/source-locked-ku-rendering` | `90fbddb` | `93415f0` | Clean committed experiment that retains every selected primary KU, renders each KU in an isolated structured call, and composes multiple units deterministically. Targeted real-model checks pass; it remains unmerged pending a full 460-case run and direct review. |

Every snapshot listed above is committed and clean. These experimental branches
are local and have not been pushed.

A code commit alone does not reproduce an approach. Because every current
design renders reviewed knowledge-unit prose closely, the finalized dataset is
part of the experiment, so each row pins a `soprano-qa-dataset` commit as well.
`93415f0` is the first commit that contains the finalized five-piece review
states and the derived knowledge-unit question layer that approaches 5 and 6
were developed against. The strict experiment predates those curation commits
and its exact dataset state was never captured; treat its output as
non-reproducible at the sentence level and compare it only structurally.

Two related worktrees are deliberately **not** registry entries, because the
rule above excludes baseline rollback points and dataset repositories. They are
recorded here only so that they are discoverable:

- `/home/minhee/codex/pre-fidelity-editor-snapshot`, branch
  `codex/pre-fidelity-editor-snapshot`, commit `7e364b6` — clean committed
  rollback point for the Approach 3 one-pass baseline as it stood before the
  fidelity-editor experiments.
- `/home/minhee/codex/pre-ku-question-coverage-dataset`, branch
  `codex/pre-ku-question-coverage-dataset`, commit `d728ca8` — dataset-side
  rollback point preceding the derived knowledge-unit question layer.

Neither the pipeline nor the dataset branches have been pushed to `origin`.
Every experiment in this log currently exists on one local disk only.

Worktrees do not carry the gitignored generation and embedding checkpoints, and
`config/settings.json` resolves its relative paths against the worktree's own
`config/` directory. Supply the main checkout's assets explicitly when running
an experiment:

```
SOPRANO_QA_RAG_DATASET_ROOT=/home/minhee/soprano-qa-dataset \
SOPRANO_QA_MODEL_PATH=/home/minhee/soprano-qa-rag-system/models/Qwen3-14B-Q6_K.gguf \
SOPRANO_QA_EMBEDDING_MODEL_PATH=/home/minhee/soprano-qa-rag-system/models/Qwen3-Embedding-0.6B-Q8_0.gguf
```

For every future material RAG+LLM approach:

1. create a `codex/...` branch and linked worktree before modifying the active
   approach;
2. commit the complete reproducible approach in that worktree—do not leave a
   rollback worktree dirty, and commit the dataset repository before the run so
   the approach is pinned to reviewed prose that cannot silently change;
3. add its branch, path, commit, dataset commit, purpose, evaluation artifact,
   known failures, and merge/removal status to this table;
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

This was the active baseline immediately before the fidelity-editor and
knowledge-unit rendering experiments. It remains documented for comparison,
but is not listed in the branch/worktree registry because it is a baseline
checkpoint rather than a distinct experimental RAG+LLM approach.

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

## Approach 6: source-locked full-KU rendering

Status: **implemented, committed, and not yet promoted** on branch
`codex/source-locked-ku-rendering`, worktree
`/home/minhee/codex/source-locked-ku-rendering`, commit `90fbddb`.

### Reason for a new version

The atomic planner reduced semantic drift but sometimes omitted useful units,
while one-pass and editor designs still let Qwen join valid relations into a
new unsupported causal claim. The next experiment therefore removes model-
driven selection and cross-unit synthesis from the answer path.

### Candidate design

1. Retain every finalized primary retrieval result; a measure-local unit has
   the same semantic authority as a whole-piece unit when it answers a no-range
   question, without turning its local example into a universal claim.
2. Give Qwen exactly one full KU per schema-constrained call. The call contains
   no user question, peer KU, stable KU ID, provenance ID, URL, score, range
   notice, or license metadata.
3. Require one ordered output sentence for every source sentence. The model may
   change only the outer sentence ending to formal-polite Korean and make
   minimal spacing or punctuation corrections; quoted text remains unchanged.
4. Keep KU/result mapping in application code. Temporary labels exist only for
   internal validation, are attached at the validator's own claim boundaries,
   and never enter display prose.
5. Present two or more retained KUs as bullets under a generic lead-in. Exact
   repeated piece-title prefixes may be removed as a presentation transform;
   no model performs a cross-unit rewrite.
6. A nonempty semantic or fidelity mutation remains visible with structured
   validation warnings for red-flag review. Invalid JSON, missing sentence
   coverage, a refusal, or a missing finalized primary unit is a hard protocol
   error; there is no extractive, alternate-model, or internal-knowledge
   fallback.
7. A narrow scope-aware retrieval rule may add one strong whole-piece change
   trajectory after a local leader when the no-range question explicitly asks
   why a musical property changes repeatedly across the work. It does not fill
   to K and does not affect ordinary local, yes/no, practice, or ranged queries.

### Targeted result

- All nine real Qwen3-14B targeted cases returned structured, nonempty,
  formal-polite answers with no renderer warning or internal label. These
  include the Flowery opening fermata and Una interlude questions asked without
  a range, the repeated-meter trajectory, the earlier harmful composer-intent
  case, the Die Forelle broad-guidance case, crescendo, fermata-duration, and
  ornament controls.
- Eight of the nine are semantically answer-complete under direct review. The
  Una accent answer is faithful to `una-voce-poco-fa-ku-020`, but that finalized
  dataset KU omitted two explanatory sentences still present in its source
  annotation. This is dataset curation debt, not renderer or retrieval loss;
  production code must not fall back to raw `source_annotations`.
- Real-model regressions for the quoted Una stage direction and the attached
  score abbreviation `dim.은` now preserve the quote, punctuation, Korean
  particle, and outer polite ending with no warning.
- Exact-copy orchestration covers all 223 corpus records and 425 protected
  sentence units with zero protocol errors and zero false fidelity warnings.
  The relevant test inventory passes 410/410 when the sibling dataset and main
  model assets are supplied to the worktree; the real five-piece repeated-
  change retrieval integration also passes.

### Remaining risks and decision

- This is a promising candidate, not a promoted pipeline. The required 105
  original plus 355 synthesized-family inference run and direct Codex semantic
  review have not yet been completed.
- Latency grows with the number and length of retained KUs because isolation
  requires one model call per KU. Observed targeted calls ranged from several
  seconds for short units to roughly 25 seconds for the longest unit; caching
  or pre-rendering may be needed for an interactive demo.
- Fidelity checks detect rather than overwrite a nonempty semantic mutation,
  consistent with the answer-required/no-fallback contract. Such warnings must
  be treated as red flags, not clean answers.
- Faithful rendering inherits source-level defects. In addition to the
  truncated Una accent KU, `nella-fantasia-ku-004` contains repetitive and
  awkward reviewed prose. Those records need curation rather than a more
  creative answer model.
- Whitespace is ignored by lexical fidelity comparison. This avoids false
  flags for benign Korean spacing fixes, but direct semantic review must still
  catch the rare spacing change that affects meaning.

## Approach 7: web_coaching evidence on the one-pass baseline

Worktree `/home/minhee/codex/web-coaching-integration`, branch
`web-coaching-integration` (uncommitted at the time of writing), dataset
`web_coaching/` lineage as of 2026-10-17. Generation is Approach 3 unchanged;
only corpus inputs and evidence selection change.

### Reason for a new version

The answer-eligible web database (101 records) is mostly catalog, edition,
and rights facts; only 3 records are performance guidance. Questions about
singing technique, diction, or interpretation that the 122 expert units do
not cover were answered from loosely related expert units, often with
invented advice. The dataset now has a separate `web_coaching/` lineage: 723
Korean claims from 78 web pages and 27 voice-lesson/masterclass video
transcripts, each tied to a verbatim excerpt that its validator checks.

### Design

- New `web_coaching` record type (`soprano_qa/corpus.py`), built only after
  the dataset's `web_coaching/tools/validate.py` reports no errors; exact
  release schema; global scope, no measure range. `is_web_record()` routes
  existing web-evidence handling to both web lineages.
- The expert-led semantic focus replaces no-range bundles with expert-only
  evidence, so web_coaching claims were retrieved but almost never reached
  generation (gold recall 0.06). `supplement_web_coaching_evidence()` appends
  up to `generation_supplement` web_coaching candidates after the expert
  selection, for no-range questions only.
- Gate `top_ranked`: append only when the best-ranked candidate is itself
  web_coaching. Without it, 131 of 460 expert cases received web_coaching
  evidence and blind review found those answers less faithful.
- The web_coaching description lives in its evidence block, not in
  `SYSTEM_PROMPT`, so prompts for questions without web_coaching evidence
  are byte-identical to the baseline.
- `SOPRANO_QA_SETTINGS` selects evaluation conditions
  (`config/experiments/`). An extra dense-candidate switch for web_coaching
  was tried and removed: web records already receive dense scores through
  the hybrid path, and the switch changed no result.

### Evaluation

The v7 baseline was first reproduced exactly (460/460 identical answers and
evidence). New question set: 58 questions in the dataset's
`web_coaching/evaluation/questions/` (33 narrow, 12 broad, 8 conflicting
sources, 5 unanswerable), written by Claude from the web_coaching claims with
every answering claim listed as gold. Answers were judged by blind Claude
reviewers with shuffled labels (helpfulness and faithfulness 0–2, judged
against each answer's own evidence).

| Condition | Coaching gold recall | Coaching helpfulness | Coaching faithfulness | Coaching answers with an invented claim | Expert cases given web_coaching | Expert answers changed |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (old web only) | 0.00 | 0.67 | 0.83 | 43% | 0 | 0 |
| + web_coaching, supplement 4, ungated | 0.65 | 1.67 | 1.40 | 12% | 131 | 156 |
| + web_coaching, supplement 4, `top_ranked` (adopted) | 0.55 | 1.47 | 1.24 | 21% | 8 | 16 |

Coaching helpfulness/faithfulness come from separate pairwise reviews against
the baseline, so baseline scores vary slightly between rows of the source
reviews (0.64–0.69, 0.78–0.84). A three-way review found no meaningful
difference between keeping and dropping the old web database
(helpfulness 1.64 vs 1.64), so it is kept.

Expert benchmark, adopted configuration: synthesized Hit@1 346/355, Hit@6
352/355, and MRR 0.983 are unchanged; all expected targets retrieved rises
from 318 to 319; target-source grounding stays 300; all 31 original-set
diagnostics are identical. 444/460 answers are byte-identical. Of the 16 that
changed, 8 received web_coaching evidence, 3 had slightly different expert
evidence (BM25 statistics shift with the larger corpus), and 5 had identical
evidence (decoding variation). Blind pairwise review of the 16: helpfulness
32 vs 32, faithfulness 31 vs 31, no substantive errors on either side
(baseline best 5, candidate best 4, tie 7). The ungated variant's sampled
review, by contrast, scored faithfulness 41 (baseline) vs 29.

### Decision and remaining risks

Adopted as the default (`config/settings.json`). Remaining risks:

- The coaching question set and all judgments are model-generated from the
  same claims; they show that the lineage fills a gap, not that the system
  is better overall. Human review of the gold sets and a sample of verdicts
  is the next step before citing these numbers.
- Unanswerable questions are still usually fabricated (4/5) because the
  answer-required contract has no abstention path.
- Answers still mix up evidence about vowel linking and doubled consonants,
  or apply a passage-specific note to another passage.
- Ranged questions never receive web_coaching evidence.
- The synthesized runner aborts on a single exception; one rerun was needed
  for a case whose first draft varied with llama.cpp prompt-cache history.

### Revision (2026-10-17): separate index, dense gate, score context

The first integration put web_coaching records into the main index. The
existing retrieval tests then showed that the 723 extra records shifted BM25
statistics enough to change expert rankings and two ranged accent cases, and
that indexing each claim with its category's search terms let every claim of
a category match any query naming it. The revision:

- ranks web_coaching claims in a separate index (`coaching_index.py`; claim
  text and lyric anchor only), so the main index is exactly the expert and
  web-database corpus and its tests pass unchanged;
- replaces the `top_ranked` gate with `dense_margin`: claims are appended
  only when the best claim's dense cosine beats every main-index candidate
  by 0.05. Offline on the same questions this keeps coaching gold recall at
  0.55 (`top_ranked`: 0.55) while 15 of 262 no-range expert cases would
  receive web_coaching evidence;
- adds score context: selected-range questions receive the sung text of the
  selected bars from the dataset's `sheet_music/lyrics/` (model-transcribed,
  unverified, validated), and may be answered from it alone when nothing
  grounds the bars and retrieval did not deliberately abstain.

The adopted-configuration numbers in the table above predate this revision.
Re-evaluation of the merged `main` pipeline (`b3a283a`; artifacts in
`b7951b0`), with blind pairwise LLM review (labels shuffled, un-blinded
afterwards; indicative only):

- Coaching questions (58): mean gold recall 0.52, gold hit rate 0.70, 2.55
  web_coaching passages per answer, 1 rejected answer. Against the pre-coaching
  baseline: helpfulness 0.76 -> 1.48 and faithfulness 0.88 -> 1.21 (0-2),
  answers with invented claims 43% -> 14%, fabricated content on unanswerable
  questions 4/5 -> 3/5; best or tied 53 vs 18.
- Expert benchmark (460): all 657 retrieval summary metrics are identical to
  the baseline. 290 answers are byte-identical; 155 selected-range answers
  changed only through the lyric context and 15 no-range answers received
  web_coaching evidence; no other drift.
  - The 155 lyric-context answers: new preferred 43, old 35, tie 77;
    faithfulness 1.76 -> 1.79, helpfulness 1.97 -> 1.99, fully unfaithful
    answers 2 -> 0. No answer invented a meaning or musical fact from the
    lyrics.
  - The 15 coaching-supplemented expert answers: old preferred 9, new 4, tie 2;
    faithfulness 1.73 -> 1.33, with two fully unfaithful answers. Both come
    from one ornament question pair, where a web warning about leaning on the
    short note of a dotted ornament was turned into advice.

The dense-margin gate therefore helps questions that the expert units do not
cover but still costs faithfulness on some expert-covered no-range questions.
Candidate follow-ups: a larger margin, suppressing the supplement when a main
candidate already scores highly, or a prompt rule that warnings stay warnings.

## Approach 8: English questions

Worktree `/home/minhee/codex/english-support-rag`, branch `english-support`
(from `fef5f66`), 2026-10-17. `ask(..., language="en")`; `language="ko"`
(the default) is unchanged: `SYSTEM_PROMPT` is byte-identical, the Korean
user prompt is unchanged, and no extra model call is made.

### Design

- Retrieval stays Korean. The corpus, lexical concepts, aliases, and dense
  calibration are Korean, so an English question is first translated to
  Korean with one extra Qwen3-14B call (same settings, `/no_think`, strict
  "translate faithfully, keep titles/lyrics/foreign terms/bar numbers"
  prompt in `soprano_qa/translation.py`, with a short glossary of standard
  Korean music terms). The translation is the retrieval and gating query and
  is returned as `retrieval_query`. A translation without Hangul fails
  closed. The dense query instruction is unchanged.
- Measure mentions are parsed from the original English question
  (`validate_question_measure_contract`). An English serial-comma list
  ("measures 31, 32, 33, and 34") previously raised; it now parses.
- Generation uses the same frozen evidence bundle with `SYSTEM_PROMPT_EN`, a
  separate constant that keeps every grounding, scope, citation, and certainty
  rule, adds faithful-translation rules (no added, dropped, or strengthened
  hedges; lyric words and foreign terms kept in the original language), and
  uses English examples. `build_messages(..., language="en")` also appends a
  two-line English answer contract to the user prompt: without it the model
  labelled only the last sentence of a paragraph, and 4 of 6 smoke answers
  carried an "uncited substantive claim" warning.
- Finalization skips the 합니다체 pass for English and instead rejects a draft
  whose letters are more than 30% Hangul. `NO_CORPUS_EVIDENCE_MESSAGE_EN`
  and `no_corpus_evidence_message(language)` replace the Korean message on
  the English path. `generate=False` still returns Korean evidence text
  (`answer_language: "ko"`).

### Evaluation

Smoke test, 6 English questions (5 no-range, 1 ranged), `generate=True`: all
6 answers are English, none raised, none carries a validation warning. The
ranged answer keeps the hedge ("can be seen as representing the trout's
leaping motion"). Problems found by reading against the evidence: the Rosina
answer turns the expert's exaggerated dotted rhythm (겹점8분음표 and
32분음표) into "exaggerated staccato and rapid note values", and blends a web
claim about a contralto voice; the 'che volano' answer drops that the word
has a single l; "the third verse" (translated "3절") retrieves a generic
verse-colour unit instead of the unit about the changed third stanza (3연).

Retrieval, 105 original evaluation cases (`retrieve_generation_evidence`,
top_k 6; hit = any expected range-applicable KU in the bundle):

| Question | Korean | English via translation |
| --- | --- | --- |
| Inventory paraphrase (102 cases with an English paraphrase) | 102/102 | 100/102 |
| Original annotator question (74 cases where both pass the measure contract) | 73/74 | 73/74 |

The first English run, before the translation glossary, scored 95/102 on
the paraphrases; its misses were terminology (grace note translated as
어절, double-dotted as 이중점음, meter as 박자 구조). The glossary was added
after inspecting those misses, so 100/102 is partly tuned on this set. The two
remaining misses are a dataset translation of 박자 as "tempo" and "text"
fitting under ornamentation. 31 original questions name bars while their case
passes no range and are rejected by the measure contract in both languages.
English answers were not scored by blind review.

### Decision and remaining risks

Available behind `language="en"`; the Korean path is unchanged. Risks:

- Retrieval quality depends on one translation call; mistranslated music
  terms silently change the evidence.
- The English answer is a translation of Korean evidence by the same model,
  so terminology errors (dotted rhythm vs staccato) can occur even with the
  right evidence. No blind faithfulness review of English answers exists yet.
- English requests make two model calls, and `generate=False` English
  requests still load the 14B model for translation.
- `scripts/ask.py` and the `answer.py` CLI have no language option.

## October 31: reject the destructive intent router (E9)

E8 passed effective development BERTScore (0.905950, 42/43 delivered) but
failed semantic review: Claude marked four expert answers with material drift
and three with substantial factual errors. Manual review confirmed that
question-only routing discarded useful KUs for short local questions, then
invented explanations of grace notes and small notation. A high BERTScore did
not justify retaining that design. E8 never entered validation or deployment.

E9 removes that routing call. It retains passage-based relevance selection,
explicitly marked general knowledge, the checked bilingual terminology aid,
one concise general paragraph, and the conditional-scope correction. A
regression test verifies that a short notation question keeps its reviewed
source through composition and output review. No reference or question list
is embedded in the production routing code. Frozen experiment artifacts and
all earlier unsuccessful candidates remain in the background-evaluation
folder; the current candidate still requires a fresh complete comparison.

## Evaluation and promotion checklist

For each new approach, record all of the following here before calling it
final:

- code/worktree/commit used for rollback;
- the `soprano-qa-dataset` commit the run was executed against;
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
