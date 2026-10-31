# Development record

Metric selection used development B only. E9 was selected before its validation
run; subsequent E10/E11 refinement used development outputs and the previously
known targeted meaning cases. E9 validation metric files existed by E11, but
their values were not consulted for E10/E11 selection. The development
cohort has 43 expert-reference requests, ten answerable general-music requests,
and two questions about unavailable local facts. Refusals remain in the expert
BERTScore denominator with a score of zero. Complete answers are scored.

The 18 encoder/layer/IDF combinations in `bertscore-development-b/` were tried
once on development B. XLM-R large, layer 18, no IDF, raw F1 and the slow tokenizer
were selected there and kept for every later run. No validation output was used
to choose the metric. The raw scales of different encoders are not calibrated
accuracy percentages.

| Run | Expert answers / 43 | Effective expert F1 | General answers / 10 | Change or finding |
| --- | ---: | ---: | ---: | --- |
| B | 38 | 0.838086 | 4 | Initial typed source/background answers; too many refusals. |
| C | 37 | 0.823354 | 3 | Separating task admission alone did not fix the refusals. |
| D | 41 | 0.909693 | 6 | Fixed valid-output rejections, bounded output-review retry and scoped source wording. |
| E | 41 | 0.912229 | 6 | Select passages that answer the question before composition. |
| E2 | 41 | 0.912229 | 6 | Correct note-value versus pitch-interval query translation. |
| E3 | 41 | 0.912330 | 9 | Require a general explanation or explicit limitation when no primary passage is selected. More answers exposed general factual errors. |
| E5 | 41 | 0.913376 | 10 | Checked bilingual terminology and a dedicated general explanation prompt. Remaining unrelated-source attribution and conditional-scope issues. |
| E7 | 40 | 0.889482 | 9 | Joint intent/label selection regressed: an invalid mixed result refused a diction question; retrieved context still contaminated a general fermata answer. Rejected. |
| E8 | 42 | 0.905950 | 10 | Question-only routing passed the numeric target but discarded relevant sources for short score/performance questions. Wrong grace-note and small-note explanations were found by inspecting answers. Rejected before validation. |
| E9 | 42 | 0.934070 | 8 | Removed question-only routing; expert development review found no material drift, but manual inspection of the known meaning cohort later found 11 material drifts / 41. |
| E10 | 43 | 0.972074 | 7 | Complete native-language source quotations; full English source translations. All 43 expert development answers judged faithful, but unrelated local advice still contaminated general questions. |
| E11 | 43 | 0.972074 | 10 | Conservative explicit-general routing; no expert development text changes from E10. Claude: zero material drift, zero substantial factual errors, zero false refusals on 55 development requests. Selected before E11 validation. |

A was an earlier 24-request pilot, not the full development cohort. E4 compared
generation modes in isolated diagnostics, not a complete service run; E6's scope
experiment was incorporated into E7 without a separate measured cohort. Their
outputs are not substituted into any later run. See
`diagnostics/general-generation/README.md`. Thinking/sampling did not reliably
fix the observed terminology errors and were not adopted.

E8 separates a question-only general/evidence routing decision from passage
selection, limits general explanations to one paragraph, and corrects a false
whole-piece generalization detection on an explicitly conditional phrase. The
scope fix uses the actual previously rejected answer as a regression case;
semantic review is still required and no source text is rewritten. E8 source
is frozen at `654a86d68176bef7b22fb3d4b3f9307e9e68b59e` in
`provenance/e8-final/`; the exact policies are in `prompts-e8-final/`.

Despite passing the development metric, E8 was rejected on meaning and factual
quality. Its prepared `comparison-e8-not-run.config.json` was never executed.
E9 removes the destructive question-only routing pass, retaining actual-passage
relevance selection, concise general explanations and the conditional-scope
fix. It does not special-case evaluation questions or copy reference answers.

Independent Claude judgments vary. Earlier B/C used the first rubric; D onward
include actual lyric context in the second rubric. Development judgments were
used to diagnose problems, not pooled as independent trials. In particular,
`source_fidelity=not_applicable` on a general answer does **not** mean it is
factually correct. The final summarizer reports substantial factual errors
separately, and the final comparison judges deduplicated packets jointly.

The diagnostic `bertscore-fidelity-controls.json` shows that three of four
previously annotated material meaning errors still exceed raw F1 0.90. These
known historical controls are outside the target cohort and were not used to
retune the metric. BERTScore cannot establish preservation of negation,
conditions, attribution or practical meaning by itself.

## Snapshot metadata caveat

The reusable snapshot preparer retains an older `protocol` object describing
two repetitions and paired order in each `provenance.json`. Those fields do not
describe this experiment. The actual protocol is `protocol.md`, the final
comparison configuration, and each run's manifest: one measured repetition per
system, sequential systems, one untimed warm-up, identical frozen inputs and
settings, and one GPU generation worker. Source/model/corpus hashes in the
snapshot provenance remain applicable. The runtime directory called
`presecurity` contains **main `29fe3e2` with initial security**; the actual
pre-security baseline is `f307470` in `baseline-pre-security`.

## Source-preservation refinement

E10 (`9d9ac23`) changes the answer architecture, not the expert references or
metric scale. The model selects relevant retrieved source labels; for Korean,
the application copies each selected source's complete answer, checks its
identity, and visibly labels it as source text. This avoids paraphrase errors
in negation, scope, vowels and interpretive qualifications. English evidence
items translate one complete source. Optional general explanations remain
generated and separately labelled. Admission, source screening, semantic
output review, citations and scope checks still apply.

This is an extractive/translation approach for source claims. Its higher
BERTScore partly reflects literal overlap with expert KUs that also supply
the references. It is not evidence of improved unrestricted generation or
generalization to new works. Selection, relevance, source errors and English
translation can still fail. The candidate never reads the frozen evaluation
reference file during a request.

E10's independent judge missed an unrelated fermata answer and an English
notation error found by manual inspection. LLM judgments are therefore
reported with their calibration and manual checks, not treated as truth.
E11 (`4788793`) tests a conservative deterministic general-intent check after
admission: explicit definitions/comparisons/universal rules, no local or causal
referent and no selected range. Ambiguous short questions keep the KUs. Eight
general development questions satisfy this check; none of the 43 expert
development requests does. The rule has no case IDs, piece-name list or
music-term whitelist. The test suite passes 582 tests.

The E9 full run completed 290 requests. Its baseline stage failed at import
because the pre-security source has no `source_priority` module. No baseline
response was measured by that failed attempt. E11's runner imports that module
only when present; ordered fixed contexts do not require a synthetic priority
policy. The original failed contract is retained, and subsequent comparisons
use new output directories. The reporting predicate also now includes all 41
meaning cases; the earlier predicate omitted `recent-fresh-benign-05-ko` from
that subgroup, though it remained in overall metrics.

## E11 validation, E12 routing repair and E13 translation repair

E11’s full 290-request candidate run completed before its orchestrator was
terminated at the candidate boundary. No baseline request was made in that run.
Expert validation F1 was **0.892643** (57/62 delivered, zero truncation), below
target. It also produced a serious D.S. al Coda definition error: the general
intent route discarded the directly answering KU and the model invented an
incorrect explanation. E11 was rejected rather than accepted on its passing
development score. Its raw answers, metric artifacts and interruption record
remain in `comparison-e11/`.

Manual source comparison of all 41 known meaning cases found 36 faithful, one
minor drift, two material English drifts and two withheld answers. The two
material drifts were invented lyric romanization and substituting mixed voice
for the source’s falsetto term. This manual audit is separate from Claude’s
scores; it is not a human-expert annotation. E9’s corresponding manual audit
had 11 material drifts, 13 minor drifts, 15 faithful, one withheld and one
uncertain. These diagnostic cohorts do not estimate production prevalence.

E12 (`6953b13`) retains primary evidence when the existing retriever reports a
complete source-question concept match. The signal changes only composition
routing, never safety approval. It also clarifies that singing breath/effort
and pitch/key/tempo comparisons across performances are musical tasks. The
55 development requests plus six explicitly reused failed expert-validation
requests form `e12_probe_cases.json`. Development F1 remains 0.972074 with
43/43 expert answers; all ten general questions are answered. Five of the six
known failures are answered correctly in the probe; the no-range accent
question remains withheld. Claude finds no material drift or substantial
factual error in this probe.

E13 (`ab5d591`) additionally prohibits invented romanization/transliteration and
asks translations to preserve each source-listed technique. The lexical
translation 가성 = falsetto was checked against the Korean Speech-Language &
Hearing Association dictionary; register terminology varies across traditions,
so no universal physiological classification is asserted. The E13 probe adds
all nine previously known English meaning cases to the E12 probe (70 total).
The original 43 development expert requests alone determine development F1.

**Validation-use disclosure:** E12/E13 were refined after inspecting E11
validation failures. The 43/62 question-group split and all references remain
unchanged, but later validation is reused internal evaluation, not an untouched
test set. The encoder/layer/IDF setting remains the original development-B
selection throughout; no reference, denominator or metric was adjusted to
raise the validation result. A new independently annotated test set would be
needed for an unbiased estimate of generalization after this refinement.

## E13 outcome and E14 source-translation separation

E13 retained development F1 0.972074 and answered all ten general development
questions, but Claude and manual inspection still found two material English
meaning errors among the nine known targeted English examples. Prohibiting
romanization in a prompt did not stop invented spellings; an unsupported causal
connection also remained. E13 is not selected as the final implementation.

E14 separates source selection from translation, supplies original source
sentences to a question-free translation call, and protects quoted Korean
literals in application code. Source coverage and literal identity are checked;
semantic output review still compares the translated answer with the original
source. The extra call and its 1536-token cap must be counted in latency and
configuration comparisons. English translation fidelity is not guaranteed by
these structural checks.

The user clarified that **0.90 is a sufficient BERTScore threshold, not a
maximization target**. Final selection must consider meaning, usefulness,
naturalness, false refusals and latency. High scores obtained through full KU
quotation are not evidence that overall RAG quality is maintained.

E14’s focused nine-English-case probe completed before the user’s requested
server-shutdown checkpoint: 9 responses, 8 delivered, no execution errors.
Original Hangul lyric spelling and the falsetto/head/chest list improved, but
the classical-vocal qualifier was omitted in one sentence, a measure/phrase
referent drift remained, and one normal question was still withheld. Redundant
source selection also produced repeated genre advice. E14 is **experimental,
not selected**. The full matrix and independent Claude assessment remain
pending. See `e14-focused-review.json` and the root `RESUME.md`.
