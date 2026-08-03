# Synthesized Korean question development evaluation

## Executive summary

This development set contains three intended meaning-preserving Korean
reformulations for each of 40 schema-1.3 expert questions. Canonical
source/range expansion produces 153 cases across 51 three-formulation groups.

Dense-enhanced retrieval is decisively more robust than the existing BM25
system. At `top_k=6`, BM25 retrieves any benchmark-linked expected knowledge
unit for 13.73% of cases, the separate dense/hybrid worktree reaches 97.39%,
and the final tuned system reaches 100.00%. The final system was therefore
selected for generation evaluation. This establishes better retrieval
robustness on this routed development set; by itself it does not prove better
answer correctness.

Retrieval and generation completed all 153 cases without an error and with
validated pipeline integrity. The generator produced 127 answers from the LLM
and used a compact expert-first extractive safeguard for 26. All 153 retrievals
are target-source grounded, and every generated answer cites at least one
benchmark-linked expected unit.

## Dataset and protocol

The synthesized formulations are stored in the dataset repository under
`expert_curation/evaluation_question_variants/`. They are benchmark data tied
to stable expert source IDs and schema validation, so the dataset repository
is the appropriate source of truth. They remain separate from expert source
text and corpus aliases, preventing the retriever from indexing the evaluation
wordings as evidence. Evaluation runners and result artifacts remain in the
RAG repository.

The dataset contains:

- 40 canonical expert questions and 120 synthesized formulations;
- 51 canonical source/range groups, expanding to 153 cases;
- 33 Die Forelle, 30 In Flowery Clouds, and 90 La Capinera cases; and
- 96 lexical substitutions, 95 syntactic reframings, 68
  information-structure changes, 32 word-order changes, and 13 colloquial
  reframings. A variant may carry more than one transformation label.

An AI semantic audit corrected eight high-confidence meaning or wording
problems and rewrote 13 formulations whose structural diversity was too weak.
The variants have not received independent human semantic review. They are
therefore intended meaning-preserving development data, not a human-validated
or untouched benchmark.

Retrieval receives the canonical piece ID and, when applicable, the canonical
measure range as structured metadata. The synthesized wording does not add a
measure locator. These results measure evidence retrieval after oracle
piece/range routing, not free-text piece identification or range extraction.
One La Capinera inference range, measures 78-81 for
`kim-la-capinera-01`, is excluded because its lyric anchor does not apply
there.

All systems use `top_k=6` and the same authenticated 223-record corpus.

Metric definitions:

- Hit@1 and Hit@6 mean that at least one ID from the canonical question's
  `knowledge_unit_ids` appears by that rank.
- “All listed @6” means that every listed ID appears in the first six results.
- MRR uses the rank of the first listed expected unit.
- “Target-source grounded” means that an in-scope unit linked to the canonical
  source appears in the first six results.
- “Three-wording consistency” means that all three formulations in a
  source/range group achieve Hit@6.

These are benchmark-linked metrics, not uniformly curator-certified retrieval
metrics. Nine In Flowery Clouds cases derived from three
`rewrite_review_pending` sources have empty
`retrieval_eligible_knowledge_unit_ids`; the evaluator deliberately scores
their canonical linked IDs for development diagnostics. Those nine cases
remain subject to reference review.

## Retrieval comparison

| System | Hit@1 | Hit@6 | All listed @6 | MRR | Target-source grounded | Three-wording consistency |
|---|---:|---:|---:|---:|---:|---:|
| Existing BM25 | 13.73% | 13.73% | 13.07% | 0.1373 | 13.73% | 0.00% |
| Separate dense/hybrid worktree | 94.12% | 97.39% | 89.54% | 0.9575 | 97.39% | 92.16% |
| Final tuned system | **94.12%** | **100.00%** | **93.46%** | **0.9651** | **100.00%** | **100.00%** |

Compared with BM25, the final system improves Hit@6 by 86.27 percentage
points. Compared with the separate dense/hybrid worktree, it removes all four
Hit@6 failures, improves all-listed coverage by 3.92 points, and raises
three-wording consistency by 7.84 points without reducing overall Hit@1.

Final-system results by piece are:

| Piece | Cases | Hit@1 | Hit@6 | All listed @6 | MRR | Target-source grounded |
|---|---:|---:|---:|---:|---:|---:|
| Die Forelle | 33 | 81.82% | 100.00% | 72.73% | 0.8889 | 100.00% |
| In Flowery Clouds | 30 | 90.00% | 100.00% | 100.00% | 0.9444 | 100.00% |
| La Capinera | 90 | 100.00% | 100.00% | 98.89% | 1.0000 | 100.00% |

The final router executed 151 cases in hybrid mode and dense candidates
contributed in all 151. Two broad-guidance formulations were intentionally
routed through lexical expansion. No case used a dense-error fallback. The
separate dense worktree predates contribution instrumentation, so its 153
contribution values are recorded as unknown, not as zero.

There are no Hit@6 failures. Ten cases do not retrieve every listed unit: six
omit optional Schubert background `die-forelle-ku-003`, two omit optional
verse-colour guidance `die-forelle-ku-015`, one broad Die Forelle formulation
omits direct diction guidance `die-forelle-ku-017`, and one La Capinera opening
formulation omits `la-capinera-ku-002`. Nine cases place their first expected
unit below rank 1; the lowest first-expected rank is 6.

## Generation diagnostics

Generation was evaluated only for the final tuned system; BM25 and the
separate dense/hybrid worktree are compared only at retrieval.

Retrieval and generation both completed 153/153 cases with zero errors:

| Generation path | Cases | Rate |
|---|---:|---:|
| LLM from retrieved evidence | 127 | 83.01% |
| Expert-first extractive safeguard | 26 | 16.99% |

The safeguard was used 18 times because the grounded model returned no usable
answer and eight times because a local example did not support a whole-piece
generalization. By piece, it was used for 11 Die Forelle, 14 In Flowery Clouds,
and one La Capinera case.

The final expert-first selector prefers a direct expert anchor, preserves
same-source split units, admits at most one independent non-local expert
contender, and does not append web or local-example evidence after a non-local
expert anchor. Review warnings are derived only from evidence actually cited
or selected. Retrieval diagnostics still retain the full candidate list.

Across all answers, the serialized answer strings total 30,960 Unicode code
points, including citation text and whitespace, with a mean of 202.35 and
median of 169. Compared with the intermediate selector, overall text is 11.39%
shorter and extractive text is 28.73% shorter. The 26 extractive answers
contain 45 citations, zero web citations, zero `local_example` citations, and
nine warnings that all concern the queried In Flowery Clouds unit directly.

Generated-answer citation coverage is:

- at least one listed expected unit: 153/153;
- all listed expected units: 137/153 (89.54%);
- at least one retrieval-eligible unit among cases with an eligible target:
  144/144; and
- all retrieval-eligible units among those cases: 128/144.

Citation presence is a structural grounding proxy, not proof that every claim
is semantically correct. Sixteen multi-unit answers omit a secondary listed
unit: six omit optional Schubert background, three omit optional verse-colour
guidance, one omits broad-question diction guidance, three omit an additional
In Flowery Clouds unit, and three omit La Capinera opening pronunciation.

The earlier zero-expected fallback failures are fixed:

- `kim-die-forelle-04-syn-03` now leads with `die-forelle-ku-006`;
- `kim-die-forelle-08-syn-01` and `syn-03` include
  `die-forelle-ku-011`; and
- all three In Flowery Clouds meter-return formulations now use
  `in-flowery-clouds-ku-011` and no longer answer the earlier transition.

All three transposition/too-high formulations now use
`in-flowery-clouds-ku-007` to explain that changing key is allowed for this art
song. The unrelated fermata warning and local examples no longer leak into
those answers.

Remaining visible precision issues include generic Die Forelle difficulty or
timbre supplements in several extractive answers, a broad transposition
supplement in one small-notes answer, and six Die Forelle LLM answers with
broad six-ID citation footers. These are useful targets for a future reranker
or evidence-aware answer compressor.

## Semantic-judge results

The exact-code calibration gate passed 25/25 fixed controls with 100% accuracy
and zero terminal frame errors before judging began.

Judging completed 153/153 cases with zero operational errors. Raw conservative
triage assigned 62 cases (40.52%) to `pass`, 80 (52.29%) to `human_review`, and
11 (7.19%) to `fail`. Seven structured-output retry exhaustions are included
in the human-review count as `unjudgeable`.

| Piece | Cases | Semantic pass | Human review | Fail | Unjudgeable (in review) | Reliable RAG+LLM pass |
|---|---:|---:|---:|---:|---:|---:|
| Die Forelle | 33 | 6 | 21 | 6 | 2 | 4 |
| In Flowery Clouds | 30 | 4 | 22 | 4 | 3 | 4 |
| La Capinera | 90 | 52 | 37 | 1 | 2 | 52 |
| **Overall** | **153** | **62** | **80** | **11** | **7** | **60** |

The run's primary `semantically_reliable_rag_llm_answer` metric additionally
requires an LLM answer generated from retrieved expert evidence. It records
60/153 passes (39.22%): two Die Forelle extractive safeguards received a
semantic `pass` but are intentionally excluded. At canonical-question level,
where any failed or human-review expanded case prevents a pass, 9/40 questions
pass, 24 require human review, and seven fail.

The counts must be interpreted as conservative triage rather than direct
answer-accuracy estimates. The judge combines two prompts from the same
Qwen3-4B checkpoint with an atomic Korean NLI guard; the two prompts are not
statistically independent. In the final run, 130 cases received 100/100 raw
weighted scores from both frames; 68 of them were nevertheless routed to human
review, and 50 of those 68 carried atomic-entailment warnings. Across the 61
dual-complete cases with an atomic-entailment warning, the median of each
case's highest warning entailment was 0.00358 against a 0.8 threshold.
Incomplete answers can receive similarly low guard scores, so simply lowering
the threshold is not justified. “Human review” means automation declined to
certify the answer, not that the answer is known to be wrong.

The 25/25 calibration result shows that the implementation matches its fixed
safety controls. It is not a statistical calibration study of Korean
paraphrase entailment.

All seven `unjudgeable` cases exhausted four structured-output validation
attempts and were routed to human review without an invented score.
Deterministic coverage/range guard failures remain operational errors. Retry
budgets and selected system roots are bound into calibration and evaluation
fingerprints.

## Changes made

Retrieval changes include:

- integration of dense/hybrid retrieval and a local embedding cache;
- preventing bare `어느 쪽` from triggering a page constraint while still
  recognizing explicit `어느 페이지` and `몇 쪽`;
- lowering dense relevance/content admission thresholds to 0.44 and 0.43;
- allowing strong dense candidates with ordinary lexical concept overlap to
  compete while preserving source, piece, range, and requested-constraint
  gates;
- recognizing broader Korean performance-guidance constructions; and
- expanding coherent same-source sibling units for broad guidance questions.

Generation now uses an expert-first compact safeguard after grounded LLM
generation fails, scopes review warnings to cited evidence, rejects unsupported
local generalizations, and keeps answer prose bounded while retaining complete
retrieval diagnostics.

The evaluator now distinguishes intentional lexical routing, hybrid searches
with and without dense contribution, and true lexical fallback. It binds the
retry budget, selected system root, all pipeline modules, requirements,
embedding model, and embedding cache; reauthenticates corpus/stats/model/cache
after retrieval and generation; and cannot mark a result complete until its
integrity state is validated.

## Reproducibility and artifacts

Each retrieval artifact fingerprints the variant and canonical question
files, evaluator code, selected worktree code/configuration, exact corpus and
statistics, and embedding checkpoint/cache when applicable. It rejects stale
inputs, incompatible resumes, imports from the wrong worktree, target mutation,
and unintended fallback. Checkpoints are atomic.

Current primary artifacts are:

- `synthesized_retrieval_bm25.json`;
- `synthesized_retrieval_dense.json`;
- `synthesized_retrieval_final.json`;
- `synthesized_question_results_final.json`.

Files whose names contain `_pre_` are archived diagnostics only and are not
current benchmark results.

## Limitations and next evaluation

This is a tuned development result, not an untouched generalization estimate.
The same formulations informed threshold, routing, and answer-selection
changes. Only three pieces have the required schema-1.3 reference contract, La
Capinera contributes 90 of 153 cases, and only 13 of 120 formulations are
explicitly colloquial. A subsequent evaluation should use independently
human-reviewed paraphrases from additional pieces, with more colloquial,
elliptical, misspelled, and adversarial Korean, and should keep that set
untouched until final comparison.

The evaluation assumes oracle piece/range metadata. A separate benchmark is
needed for piece recognition and measure-range extraction from free text.

Finally, the inherited canonical contract for `kim-la-capinera-17`
paraphrases the original question about rapid alternation of `ff` and `pp` as
a more general question about their musical contrast. Its three variants
preserve that canonical paraphrase and therefore do not test robustness to the
original rapid-alternation detail.
