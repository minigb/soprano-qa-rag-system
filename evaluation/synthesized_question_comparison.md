# Five-piece synthesized-question RAG+LLM evaluation

## Executive summary

The current synthesized-question development evaluation covers all five
supported pieces. It applies three intended meaning-preserving Korean
reformulations to each of 79 active human-annotated questions, producing 237
variant formulations and 372 range-expanded inference cases.

The current hybrid RAG+LLM pipeline completed all 372 cases with zero runtime
errors, and the artifact passed its post-run integrity checks. At `top_k=6`,
at least one benchmark-linked expected knowledge unit was retrieved in
359/372 cases (96.51%). All three variants succeeded at Hit@6 in 119/124
canonical source/range groups (95.97%). No case used a dense-error or lexical
fallback.

The pipeline returned 240 grounded LLM answers, 126 extractive safeguards,
and six unavailable answers where the retriever admitted no corpus evidence
and internal model knowledge was intentionally disabled. These generation
path counts and retrieval metrics do not establish semantic answer correctness.
No LLM judge
was run for this synthesized evaluation; compare the human expected answer
and generated answer manually in the local viewer.

## Dataset and protocol

The variant source files live in the dataset worktree under
`expert_curation/evaluation_question_variants/`. They are joined to the human
evaluation-question inventories and reviewed expert knowledge units by stable
source IDs. The synthesized wording is not added to the retrieval corpus or
its aliases.

The active benchmark contains:

| Piece | Active questions | Variant formulations | Inference cases |
|---|---:|---:|---:|
| Die Forelle | 10 | 30 | 33 |
| In Flowery Clouds | 10 | 30 | 30 |
| La Capinera | 20 | 60 | 90 |
| Nella fantasia | 14 | 42 | 60 |
| Una voce poco fa | 25 | 75 | 159 |
| **Overall** | **79** | **237** | **372** |

The stored Nella fantasia question `kim-nella-fantasia-01` remains excluded:
its recorded answer asks for existing web material rather than supplying an
expert answer. It is retained only for provenance and is not synthesized or
sent to the pipeline. One La Capinera range, measures 78–81 for
`kim-la-capinera-01`, is also excluded because the linked lyric anchor does
not apply to that range.

Each inference request receives the canonical piece ID and, when applicable,
the canonical measure range as structured metadata. The benchmark therefore
tests wording robustness after oracle piece/range routing; it does not test
piece recognition or measure extraction from free text.

The runner uses `top_k=6`, requires the configured embedding and generation
GGUF checkpoints, and initializes both before writing results. Hybrid mode is
fail-closed: an unavailable dense backend aborts the run instead of silently
switching to lexical retrieval. Every request uses `generate=True` and
`allow_internal_knowledge=False`.

## Retrieval results

The artifact reports several target lanes because not every linked knowledge
unit is retrieval-eligible and not every linked unit applies to every selected
range:

| Target lane | Eligible cases | Hit@1 | Hit@6 | All targets @6 | MRR |
|---|---:|---:|---:|---:|---:|
| All benchmark-linked units | 372 | 91.13% | 96.51% | 85.48% | 0.9348 |
| Range-applicable linked units | 372 | 91.13% | 96.51% | 88.71% | 0.9348 |
| Retrieval-eligible units | 360 | 90.83% | 96.39% | 88.33% | 0.9331 |
| Range-applicable retrieval-eligible units | 360 | 90.83% | 96.39% | 88.33% | 0.9331 |

Twelve cases have no retrieval-eligible target and are omitted from the two
eligible-lane denominators rather than counted as misses. Target-source
grounding is 359/372 (96.51%).

Results against all benchmark-linked units by piece are:

| Piece | Cases | Hit@1 | Hit@6 | All targets @6 | MRR | Target-source grounded |
|---|---:|---:|---:|---:|---:|---:|
| Die Forelle | 33 | 81.82% | 100.00% | 69.70% | 0.8889 | 100.00% |
| In Flowery Clouds | 30 | 90.00% | 100.00% | 100.00% | 0.9444 | 100.00% |
| La Capinera | 90 | 100.00% | 100.00% | 98.89% | 1.0000 | 100.00% |
| Nella fantasia | 60 | 88.33% | 100.00% | 85.00% | 0.9417 | 100.00% |
| Una voce poco fa | 159 | 89.31% | 91.82% | 78.62% | 0.9030 | 91.82% |
| **Overall** | **372** | **91.13%** | **96.51%** | **85.48%** | **0.9348** | **96.51%** |

The five source/range groups in which not all three variants hit at `k=6` are
all in Una voce poco fa:
`kim-una-voce-poco-fa-13` at measures 62–62 and
`kim-una-voce-poco-fa-15` at measures 68–69, 72–73, 92–93, and 96–97.
The latter four groups miss at Hit@6 for all three variants; the first misses
only its third variant.

Retrieval execution consisted of 363 hybrid searches with a contributing
dense result, six hybrid searches with no admitted dense match, and three
intentional lexical routes inside the configured hybrid pipeline. All 369
dense-attempted cases completed cleanly, and `fallback_used_cases` is zero.

### Miss analysis

All 13 Hit@6 misses are concentrated in two Una voce poco fa questions:

- For `kim-una-voce-poco-fa-13` at measure 62, variants 1 and 2 retrieve the
  expected vowel-change unit, but variant 3 retrieves a different generic
  vowel-sustaining unit. The expected dense candidate exceeds the configured
  relevance-to-content alias-gap guard by 0.0157.
- All 12 cases for `kim-una-voce-poco-fa-15` miss the expected displaced-accent
  unit. Its curated answer describes Rosina's lively character but omits the
  central words “accent” and “first beat”; those concepts occur only in its
  source question/retrieval alias. The conservative content and alias-gap
  guards therefore reject it for all three variants.
- At measures 68–69 and 72–73, no other evidence is admitted, producing the
  six unavailable answers. At measures 92–93 and 96–97, an unrelated
  ossia/alternative-melody unit with an overlapping range is admitted instead,
  producing six extractive answers that do not address the expected accent
  question.

This is a corpus-rewrite and conservative-admission interaction, not a missing
model or an execution fallback.

## Generation diagnostics

| Generation path | Cases | Rate |
|---|---:|---:|
| LLM from retrieved evidence | 240 | 64.52% |
| Retrieval-extractive safeguard | 126 | 33.87% |
| Unavailable because retrieval admitted no corpus evidence | 6 | 1.61% |

The extractive safeguard was used 105 times because the grounded model
returned no usable answer and 21 times because the safety checks rejected an
unsupported whole-piece generalization from a local example. The six
unavailable cases did not use the model's internal knowledge.

No answer used an `internal_knowledge` basis, and the serialized artifact
contains zero occurrences of the removed `제공된 검색 근거` footer.

Generation modes by piece are:

| Piece | LLM | Extractive | Unavailable |
|---|---:|---:|---:|
| Die Forelle | 22 | 11 | 0 |
| In Flowery Clouds | 16 | 14 | 0 |
| La Capinera | 89 | 1 | 0 |
| Nella fantasia | 44 | 16 | 0 |
| Una voce poco fa | 69 | 84 | 6 |
| **Overall** | **240** | **126** | **6** |

An extractive or unavailable mode is an explicit pipeline outcome, not an
operational model fallback: both local checkpoints and the `llama_cpp`
backend passed preflight, and the run recorded zero execution errors.

## Manual answer comparison

No semantic LLM judge was run for this artifact. Retrieval hits, evidence
links, and generation modes are structural diagnostics; they cannot decide
whether an answer is complete, precise, or faithful to the human annotation.

Use the read-only viewer to inspect the human expected/reference answer beside
the generated answer:

```bash
conda run -n soprano-qa python evaluation/server.py \
  --results-file evaluation/synthesized_question_results.json
```

Then open <http://127.0.0.1:8766/>. Filters are available for piece,
generation mode, case status, and synthesized variant. Manual review state is
stored in browser `localStorage` and can be exported from the viewer.

For schema-1.3 inventory records, the artifact also exposes curator-defined
claim scope and per-case reference authority. Schema-1.1 source answers do not
have that claim partition and are marked for manual interpretation where
appropriate.

## Reproducibility and artifact

The current result is
`evaluation/synthesized_question_results.json`. It contains the benchmark
inputs, human references, generated answers, evidence, retrieval diagnostics,
generation modes and reasons, exclusions, model hashes, runtime information,
and integrity state in one resumable artifact.

`evaluation/run_synthesized_questions.py` fingerprints the variant and
canonical input files, evaluator, selected pipeline code and settings,
authenticated corpus and statistics, both model checkpoints, and runtime. It
checkpoints atomically after each case, rejects incompatible resumes, and
reauthenticates the inputs after inference. The current artifact is complete:
372/372 cases, zero errors, and validated integrity.

## Limitations and next review

This is a development evaluation, not an untouched generalization estimate.
The variants are intended to preserve meaning, but structural validation of
wording and protected anchors is not a substitute for independent human
semantic review. Una voce poco fa accounts for all 13 Hit@6 misses and all six
unavailable answers, so those cases are the clearest retrieval and corpus
coverage targets.

The next necessary evaluation step is manual expected-versus-generated answer
review, especially for the 126 extractive safeguards, six unavailable cases,
and the source/range groups without an expected Hit@6 retrieval. A future
holdout should also include independently reviewed paraphrases and should test
free-text piece/range routing separately.
