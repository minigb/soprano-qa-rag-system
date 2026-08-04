# Five-piece synthesized-question hybrid answer evaluation

## Executive summary

The current development evaluation covers all five supported pieces. It
applies three meaning-preserving Korean reformulations to each of 79 active
human-annotated questions, producing 237 variant formulations and 483
inference cases. All 483 cases completed without execution errors, and the
artifact passed its post-run integrity check.

The safe answer policy returned the complete finalized expert answer in 346
cases and explicitly returned unavailable in 137 cases. It never substituted
model knowledge, raw web text, a different retrieval backend, or a different
judge. The 137 abstentions comprise 124 cases with insufficient authoritative
evidence, ten with no range-applicable causal authority, and three ambiguous
dense-grounding cases.

Across the complete benchmark, an expected knowledge unit ranks first in
454/483 cases (93.9959%) and within the stored `top_k=6` in 470/483
(97.3085%); MRR is 0.955487. All three variants hit within six results in
154/161 formulation groups (95.6522%). The purpose-built 114-case no-range
cohort places a required finalized expert unit at rank 1 in 109 cases
(95.6140%), within the first three in all 114, and has MRR 0.978070.

Hit@3 is the minimum retrieval acceptance criterion for the no-range cohort.
Hit@1 and MRR remain ranking-quality signals, while Hit@6 is a broader
diagnostic. Retrieval success does not itself authorize an answer: ten
no-range cohort cases retrieved a required unit within the first three but
were still refused by the stricter answer-admissibility gate. These cases are
the main recall issue to review next.

No semantic LLM judge was run over the synthesized artifact. Use the local
viewer to compare its human reference with the returned answer. The separate
Codex CLI judgment covers the 41-case canonical no-range cohort only.

## Dataset and protocol

The variant source files live in the dataset repository under
`expert_curation/evaluation_question_variants/`. They join to the human
evaluation inventories and finalized expert knowledge units by stable source
ID. Synthesized wording is never added to the retrieval corpus or its aliases.

| Piece | Active questions | Variant formulations | Ordinary contexts | Added no-range contexts | Inference cases |
|---|---:|---:|---:|---:|---:|
| Die Forelle | 10 | 30 | 11 | 3 | 42 |
| In Flowery Clouds | 10 | 30 | 10 | 5 | 45 |
| La Capinera | 20 | 60 | 29 | 16 | 135 |
| Nella fantasia | 14 | 42 | 20 | 4 | 72 |
| Una voce poco fa | 25 | 75 | 53 | 10 | 189 |
| **Overall** | **79** | **237** | **123** | **38** | **483** |

The stored Nella fantasia question `kim-nella-fantasia-01` remains excluded
because its recorded response requests web material instead of supplying an
expert answer. It remains in the dataset for provenance and is never expanded
or sent to the pipeline. The La Capinera measures 78–81 case for
`kim-la-capinera-01` is also excluded because its linked lyric anchor does not
apply there.

The runner uses `top_k=6`, requires both configured GGUF checkpoints, and
loads them before creating output. Hybrid retrieval is fail-closed: a missing
checkpoint or backend aborts the run. Every inference request uses
`generate=True` and `allow_internal_knowledge=False`. Finalized expert
authority is rendered directly; it is not summarized or paraphrased by the
answer model.

## No-range semantic cohort

| Piece | Cases | Hit@1 | Hit@3 | Hit@6 | MRR | Expert answer | Unavailable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Die Forelle | 9 | 8 | 9 | 9 | 0.944444 | 8 | 1 |
| In Flowery Clouds | 15 | 15 | 15 | 15 | 1.000000 | 15 | 0 |
| La Capinera | 48 | 48 | 48 | 48 | 1.000000 | 48 | 0 |
| Nella fantasia | 12 | 9 | 12 | 12 | 0.875000 | 6 | 6 |
| Una voce poco fa | 30 | 29 | 30 | 30 | 0.983333 | 27 | 3 |
| **Overall** | **114** | **109** | **114** | **114** | **0.978070** | **104** | **10** |

The canonical inference artifact contains 38 semantic shadows plus three
exact user-reported regressions, for a separate 41-case no-range cohort. Its
required unit is rank 1 in 41/41 cases, Hit@3 is 41/41, and MRR is 1.000000.
The fresh schema-2.1/protocol-v9 Codex CLI artifact judged all 41 candidate
answers as pass and recorded no semantic-drift category.

## Complete-benchmark retrieval diagnostics

| Target lane | Cases | Hit@1 | Hit@6 | All targets @6 | MRR |
|---|---:|---:|---:|---:|---:|
| All benchmark-linked units | 483 | 93.9959% | 97.3085% | 76.1905% | 0.955487 |
| Range-applicable linked units | 483 | 93.9959% | 97.3085% | 77.4327% | 0.955487 |
| Retrieval-eligible units | 483 | 93.9959% | 97.3085% | 76.1905% | 0.955487 |
| Range-applicable retrieval-eligible units | 483 | 93.9959% | 97.3085% | 77.4327% | 0.955487 |

| Piece | Cases | Hit@1 | Hit@6 | All targets @6 | MRR | Target-source grounded |
|---|---:|---:|---:|---:|---:|---:|
| Die Forelle | 42 | 90.4762% | 100.0000% | 66.6667% | 0.948413 | 100.0000% |
| In Flowery Clouds | 45 | 100.0000% | 100.0000% | 86.6667% | 1.000000 | 100.0000% |
| La Capinera | 135 | 100.0000% | 100.0000% | 99.2593% | 1.000000 | 100.0000% |
| Nella fantasia | 72 | 86.1111% | 95.8333% | 83.3333% | 0.909722 | 95.8333% |
| Una voce poco fa | 189 | 92.0635% | 94.7090% | 56.6138% | 0.932099 | 94.7090% |
| **Overall** | **483** | **93.9959%** | **97.3085%** | **76.1905%** | **0.955487** | **97.3085%** |

Retrieval used 467 hybrid searches with a contributing dense result, 13
hybrid searches with no admitted dense match, and three deliberate lexical
routes within the configured hybrid pipeline. All 480 dense attempts
completed without a backend error. None changed to a fallback retrieval mode.

## Answer-path diagnostics

| Piece | Finalized expert answer | Unavailable |
|---|---:|---:|
| Die Forelle | 22 | 20 |
| In Flowery Clouds | 37 | 8 |
| La Capinera | 127 | 8 |
| Nella fantasia | 31 | 41 |
| Una voce poco fa | 129 | 60 |
| **Overall** | **346** | **137** |

| Unavailable reason | Cases |
|---|---:|
| Insufficient authoritative evidence | 124 |
| Missing range-scoped causal authority | 10 |
| Ambiguous dense grounding | 3 |

The answer GGUF and `llama_cpp` backend passed the mandatory preflight, but no
case needed free-form generation from web evidence. Answerable cases were
satisfied by a finalized expert unit; the remaining cases abstained. This is
intentional safety behavior, not an extractive or model-knowledge substitute.

A strict scan of all 647 completed canonical and synthesized answers found no
knowledge-unit IDs, provenance labels, `제공된 검색 근거`, review/conflict
markers, legacy scope headings, informal Korean endings, or sanitizer changes.

## Manual answer comparison

Run the read-only viewer to inspect the human reference beside the generated
answer:

```bash
conda run -n soprano-qa python evaluation/server.py
```

Then open <http://127.0.0.1:8766/>. The viewer groups all 164 original results
with all 483 synthesized results by matching inference scenario and presents
the question families separately. Every one of the 647 generated outputs is
selectable; no judged or selected-only cohort is substituted. Manual review
state is stored in browser `localStorage` and can be exported from the viewer.

## Reproducibility and limitations

The tracked `evaluation/synthesized_question_results.json` contains benchmark
inputs, human references, answers, evidence, retrieval diagnostics,
exclusions, model hashes, runtime fingerprints, and integrity state. The
runner checkpoints atomically after every case, rejects incompatible resumes,
and reauthenticates its inputs after inference.

This remains a development evaluation rather than an untouched
generalization estimate. The immediate follow-up target is answer-admissibility
recall: inspect the 124 `insufficient_authoritative_evidence` abstentions,
especially the ten no-range semantic cases whose required unit was already in
the top three, without weakening semantic-drift protection.
