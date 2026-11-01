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

## November 1 resume and E15 quality repair

The shutdown checkpoint hashes all matched, and main/dataset revisions and
working-tree status were unchanged. The GPU had no live model after restart;
no service was restarted. E14’s saved nine English answers were assessed by
Claude Opus 5.5: five faithful, three minor drifts, one refusal, no material
drift under that judge. One invalid batch used a nonliteral source excerpt and
was rejected; the first valid retry was retained. Manual review additionally
identifies the measure/phrase drift missed by Claude. The judge’s 81.82 percent
effective rubric score is not factual accuracy.

E15 uses translation-only lexical/syntax guidance instead of appending a
general-answer glossary to the translator; it distinguishes measure/phrase,
classical vocal production, pronunciation liaison, breathing, and the lyric
phrase/syllable relationship. Source selection is asked to omit purely redundant
passages while preserving complementary and conflicting advice. Output review
now explicitly permits a reviewed source’s own conditional/general advice
without treating that advice as proof of an unprovided local score fact.
No guard or citation check is bypassed. The 598-test suite passes.

The independent judge runner now permits at most three attempts only for CLI,
format, coverage or exact-quotation failures, retaining diagnostic records. It
accepts the first valid batch, even a low-scoring one; unfavorable judgments
never trigger retries. Prior judge artifacts retain their original contracts.

The fresh comparison campaign in `comparison-2026-11-01/` starts with the
immutable pre-security baseline while candidate quality work proceeds. It will
use the same frozen 290 cases and unchanged B-selected BERTScore configuration.
Measured baseline rows will be retained exactly; old historical timings are
not pooled into this campaign. Candidate selection remains subject to overall
quality, not maximizing BERTScore beyond 0.90.

## E15 rejection and E16 source preparation

E15 completed all 70 fixed diagnostic requests without execution errors;
69 were delivered, with the no-range accent question still withheld.
Development BERTScore remained 0.972074 (43/43). Despite that passing score,
Claude identified three material errors among the nine known English meaning
examples: pronunciation advice reversed, a musical rest became a comma, and
each measure became each phrase. E15 was rejected. Its complete outputs,
judgments and `e15-rejection.json` remain available.

E16 (`2ec416d`) uses maintained English translations for every curated expert
or performer source in the corpus: 129 answers across all five works. Codex
authored and checked these translations, and Claude Opus 5.5 independently
assessed their fidelity to the Korean original. The first review found 124
faithful and five minor drifts, with no material drift; five actual text edits
were then assessed again and all were judged faithful. Unchanged translations
retain their first reviews. This is not a selection among scores for the same
translation, and it is not human-expert annotation.

Runtime lookup uses exact source-text SHA256, never evaluation questions or
case IDs. The source selector, admission, screening, citation and output review
remain active; new/changed sources fall back to guarded segmented translation.
General explanations are generated separately. Offline source preparation and
review are added resources, excluded from per-request latency and disclosed in
the comparison. Korean references, all cases and the B-selected scorer are
unchanged. The catalog review also identifies three original-source factual
issues; a faithful translation does not make those original claims correct.

The 605-test suite passes, including exact-hash reuse, changed-source fallback,
mixed-source ordering, screening and rejection of catalog output by the normal
guard. The full RAG candidate is still subject to the 70-case diagnostic probe,
then the four-system comparison. Catalog fidelity alone is not acceptance.

## E17 teaching-context relevance repair before the next probe

Before running E16's prepared probe, manual inspection of E15's general-practice
answers identified a separate inherited defect: the slow-practice questions
received repetitive source collections rather than the requested practice
checks. Claude had also assigned partial relevance/completeness to those
answers. E16's Korean generation was unchanged, so E17 incorporates the repair
before the next 70-case diagnostic run; no E16 runtime responses are claimed.

E17 (`ba65680`) supplies each source's original teaching question alongside its
answer to the existing relevance selector, within the same untrusted JSON data
boundary. The selector must match the requested task, not just the musical
subject. Korean composition also receives the existing instruction to omit
purely redundant passages while preserving distinct qualifications/conflicts.
General quantitative examples must state the BPM beat unit. The general-intent
route, source texts, reviewed English catalog, evaluation questions/references
and frozen metric remain unchanged. Regression tests: 606 passed.


## E17 result and E18 bounded practice routing

E17 completed 70 requests: 69 delivered, one no-range accent refusal, no
execution errors. Development F1 remains 0.972074. Independent Claude judged
59 source answers faithful and 11 not applicable, with zero material drift or
substantial factual error. Manual comparison also found all nine known English
examples faithful. The aggregate rubric score was 95.45/100; this is not factual
accuracy. Remaining redundancy and the two slow-practice questions were scored
as partial relevance/completeness. First-person source wording in English also
needed visible attribution. See `e17-quality-review.json`.

E18 (`7cbf004`) adds a narrow explicit practice-scenario branch after admission.
A generic mention of lyrics alone need not make this scenario local. Selected
ranges, local/causal references, complete original-question matches and names
of the actual selected work retain evidence. Work names are read from the
bundle, not a hardcoded list. On the fixed290 question-only audit, only the two
known slow-practice questions change routing. Eight additional general/local
paraphrases are therefore included in the diagnostic probe to check behavior
beyond those two phrasings; these are not expert metric targets or an independent
human-annotated test. The original70 and full290 remain unchanged.

English source translations are now visibly labelled `Translated reference
text:` so a source's first-person statement does not become the assistant's
own opinion. This application-created metadata cannot be supplied through model
JSON and grants no security, citation, scope or output-review exemption. The
609-test suite passes. E18 is still awaiting its 78-case probe and final full
comparison; no passing final validation result is claimed yet.

## E18 diagnostic failures and E19 metadata/spelling repair

E18 completed 78 requests: 75 delivered, three false refusals, no execution
errors. Development F1 was 0.972074. Claude found no material meaning drift or
substantial factual error, but marked one original source's Larghetto wording
as mixed/uncertain. The effective rubric score was 92.22/100, including the
eight newly added diagnostic cases; this is not directly comparable with the
earlier 70-case aggregate. The English meaning cases remained faithful, and
both slow-practice questions now received useful general practice advice.

Two additional normal-question failures prevent adopting E18. A new-scale
practice question was rejected at admission. An aria-practice answer passed
semantic review but the finalizer rejected the translated spelling `YouTube`:
the Korean source spells it 유튜브/유투브. Inspection also showed that actual
curated KUs store original questions in `retrieval_aliases` and Korean work
names in `piece_title`; E17/E18 had not exposed these fields to the intended
context consumers. The complete E18 outputs and judgments are retained,
including these new failures. `e18-failure-diagnostics.json` records the
finalizer reason and answer/source text, without private model reasoning.

E19 (`93f3552`) passes the actual question/title metadata as untrusted context,
explicitly distinguishing a teaching question's premises from source facts.
Admission now recognizes ordinary music exercises and ambiguous music words
within the active musical context. The spelling heuristic accepts an English
token present in the reviewed translation of an exact-hash Korean source;
changed sources, unknown tokens, invalid citations and scope/security failures
remain rejected. No security or citation exemption was added. All 611 unit
tests passed. E19 is being checked on the same 78 cases, unchanged expert
references and frozen metric before the complete four-system comparison.


## E19 full review and E20/E21 rejection, E22 refinement

The completed fresh four-system E19 comparison contains 290 requests per
system and 919 unique independent Claude packets. E19 passes validation F1
0.9734575 (62/62 delivered), but is not adopted: two material meaning changes
(D.S./Da Capo and microphone-conditioned vocal advice), two independently judged
successful indirect attacks, and further general-knowledge errors remain.
Four answers visibly copied adversarial control text; the independent rubric
counts two of those as achieved attack objectives. Both observations are kept.
Original pre-security has 24 material drifts and 16 successful attacks among
60 judged attacks (10 attack execution errors); E19 has 2 material drifts and
2/70 successes. These are internal benchmark results, not universal guarantees.

E20 adds source-role/template screening but still uses E19's flawed routing.
It was deliberately stopped after 98 completed requests plus a controlled
interrupt row when independent E19 review exposed that flaw. Its partial output
is retained with an interruption record and is not treated as a full comparison.

E21 retains useful KUs for definitions and named-work comparisons. In a frozen
50-question diagnostic probe it corrects D.S. and the microphone conditions,
but adds speculative source-conflict reasoning and still gives incorrect Korean
meter explanations. The independent judge misses the speculative cause in one
answer; direct review records that discrepancy rather than overriding the saved
judge result. A general rule written only in English did not prevent the Korean
mistake, including in a newly authored 9/8 transfer question. E21 is not adopted.

E22 uses concise task-admission rules, explicitly distinguishes definition
passages from local performance advice, discourages repeated source synthesis,
and supplies the same checked meter rule bilingually. Generic practice defaults
to singing. The 50 diagnostic case bodies, 290 full cases, expert references,
translation catalog and BERTScore configuration remain unchanged. Observed
failures inform this development; none of these reused evaluations is an
independent untouched holdout. E22 was subsequently rejected; outcomes follow.

A reusable judge cache now accepts only complete hash-validated prior reviews
under the identical model, rubric, schema and judge runner. Reuse requires the
entire packet (question, answer, sources, references, context and task) to match.
It retains the original assessment contract, including unfavorable scores.
Changed packets receive fresh independent review. The aggregation records which
judgments were reused and does not claim they were new CLI calls.

## E22–E24 outcomes and E25 final boundary repair

E22's unchanged 50-case diagnostic still had five false refusals and three
substantial factual errors, including incorrect beat units. E23 added a bounded
general-definition retry after a valid negative output verdict and a narrow
conventional meter calculator. This recovered English legato and rubato answers,
but the original terse meter comparisons did not reach the calculator. E24
extended its recognizer to terse comparisons in the active music context while
excluding fraction/date questions. Its 14-case diagnostic passed direct meter
checking and independent review. E22/E23 full comparison configs were prepared
but never run; their directories record that fact rather than fabricated results.

E24 (`ec9534c`) then completed all 290 candidate requests, without execution
errors. Validation F1 was **0.9734575238920027** (62/62), with zero truncation
under the unchanged development-selected setting. The independent review had
917 unique packets: 888 exact E19 packets reused and 29 freshly assessed.
Candidate normal results were zero material drifts, zero substantial factual
errors, two mixed/uncertain factual answers, three false refusals out of 216
answerable requests, and effective rubric 94.7391/100. Known meaning cases had
39 faithful and two minor drifts out of 41. The complete artifacts and manual
review are retained, including remaining p/PP attribution/uncertainty omissions,
source wording ambiguity and long repeated quotations.

The 58 injection objectives and 12 plain out-of-scope requests all failed in
E24. Nevertheless, direct reading of all ten delivered attack answers found a
Base64 decoding command copied into `indirect-10-ko`. It did not produce a cookie
recipe, the frozen success criterion, so formal ASR remained zero. This is an
additional answer-contamination failure, also acknowledged in the saved judge
rationale, and prevents adopting E24 as the final version.

E25 (`a93b108`) changes only source/output screening for instructions that turn
a decoded request into the final/actual answer. No payload is decoded; musical
notation decoding remains allowed. Security tests49 passed and all953 original
corpus records remained allowed. A four-case live diagnostic withheld both
encoded indirect sources and reproduced the corrected D.S./microphone English
answers exactly. The full290 candidate then completed under the same cases,
metric, models and first-valid-judgment policy. E24's measurements remain
separate; no candidate latency or answer rows were carried into E25.

E25 achieves validation F1 0.9734575239 (62/62), zero execution errors, zero
judged material/substantial factual errors, three false refusals out of216,
and0/58 observed injection successes. All220 normal packets are identical to
E24; only the contaminated encoded-source answer changes. Its94.8232/100
effective rubric differs slightly from E24 through fresh judge variation, not
changed normal answers. The p/PP2 direct-review limitation remains even though
the new judge labels that identical answer faithful. The branch adopts the
evaluated reviewed_catalog/1536 settings. See the [final report](report.md) for
all baselines, residual limitations and descriptive latency costs.
