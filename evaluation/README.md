# Evaluation

Run every command in this document from the repository root with the
`soprano-qa` Conda environment. The current qualitative workflow covers all
five supported pieces: Die Forelle, In Flowery Clouds, La Capinera, Nella
fantasia, and Una voce poco fa.

## Setup

Install the project dependencies and download both configured GGUF models:

```bash
conda run -n soprano-qa python -m pip install -r requirements.txt
conda run -n soprano-qa python scripts/download_model.py
conda run -n soprano-qa python scripts/download_embedding_model.py
```

Hybrid retrieval fails at startup when the embedding checkpoint or
`llama-cpp-python` backend is unavailable. It does not silently switch to
lexical retrieval. Generated evaluation runs also validate and load the
configured answer model before creating or resuming output. The offline
qualitative judge is separate from production
generation: it requires an installed, authenticated Codex CLI and never uses
the local GGUF for judging. Check authentication with
`conda run -n soprano-qa codex login status` before starting a judgment run.

## Current qualitative evaluation

Generate range-aware RAG answers and save their linked expert references:

```bash
conda run -n soprano-qa python \
  evaluation/run_qualitative.py
```

The default output is `evaluation/qualitative.json`. The tracked snapshot is
complete and currently contains 79 questions and 164 inference cases across
the five supported pieces. The runner checkpoints each case and can resume an
interrupted run. Use `--piece PIECE_ID` or `--limit N` for a focused run.
Of the 164 completed cases, 160 use `expert_verbatim` /
`authoritative_expert`. Four selected-range causal cases return unavailable
because their selected ranges contain only a procedural sibling unit, not
reviewed causal authority. In the 41-case no-range cohort, the required
semantic target is Hit@1 and Hit@3 in all 41 cases, with MRR 1.000000. These
values come from the inference artifact; they are not Codex judge verdicts.

Run conservative answer-fidelity triage with Codex CLI:

```bash
conda run -n soprano-qa python \
  evaluation/judge_qualitative.py \
  --cohort measure_scoped_question_no_range
```

The judge selects 41 no-range cases across all five pieces: 38 semantically
self-contained questions whose finalized expert units have confirmed measure
ranges, plus three exact user-reported regressions. Five deictic or
score-dependent questions are deliberately excluded from this
additional no-range cohort; their ordinary range-selected cases remain in the
benchmark. The judge does not evaluate the other 123 canonical cases or any
synthesized case.

The judge supports only the Codex CLI in the `soprano-qa` environment. It
resolves the configured model, reasoning effort, and service tier, then passes
them explicitly to one noninteractive `codex exec` process per case. The
current configuration resolves to `gpt-5.6-sol`, `ultra` reasoning, and the
`priority` service tier; all three values are recorded and fingerprinted.
Before creating or updating `evaluation/qualitative_judged.json`, the script:

- validates the input artifact;
- requires an installed, authenticated Codex CLI;
- verifies that the configured Codex model, reasoning effort, and service
  tier are available;
- fingerprints the CLI, model, reasoning effort, service tier, input,
  structured-output schema, and judge implementation.

Each response must match the judge's structured JSON schema. The invocation
uses an empty temporary working directory, read-only sandbox, ephemeral
session, disabled web and tools, and a dedicated authenticated provider alias.
The provider explicitly disables WebSocket transport and sets its HTTP-request
and SSE-stream retry counts to zero. The per-case timeout is 600 seconds by
default.
A missing CLI or login, unavailable model, timeout, tool event, nonzero
process exit, missing output, or malformed response aborts the run. Each case
gets exactly one Codex invocation, with no automatic retry and no Qwen,
`llama-cpp-python`, lexical, or heuristic alternate judge. A failed or
interrupted judgment requires a new output file; only exactly matching
completed cases can be resumed. The production answer generator is never used
as a judge. Retrieval
ranks remain diagnostics and do not determine the pass, review, or fail
verdict. A Codex judgment is automated triage, not independent human evidence.

The current schema-2.1/protocol-v9 artifact completed all 41 selected cases:
41 passed, none were marked review or fail, and no semantic-drift category was
recorded.

Useful judge controls include `--cohort
measure_scoped_question_no_range`, `--piece PIECE_ID`, `--case-id CASE_ID`,
`--limit N`, `--codex-timeout-seconds N`, and `--minimum-pass-rate RATE`.
Resume is allowed only when the selected source cases and all fingerprinted
judge inputs match. Judge schema 2.1/protocol v9 includes each finalized
unit's canonical measure status, ranges, required-versus-supporting role, and
proposition-level semantic checks. Those fields let the judge detect agent,
predicate, causality, modality, and scope drift; they are structured authority
metadata and are not a display requirement for no-range answers.

The inference artifact records Hit@1, Hit@3, Hit@6, and MRR; the judgment
artifact records verdict totals and semantic drift categories for the selected
cohort. Hit@3 is the minimum retrieval acceptance criterion; Hit@1 and MRR
show ranking quality, while Hit@6 remains diagnostic. Scope safety is enforced
against canonical range metadata internally, while ranges and provenance
remain in structured output; user-visible prose remains natural and does not
expose an unrequested range or a pipeline scope label.

## Synthesized-question hybrid answer benchmark

`run_synthesized_questions.py` evaluates three Korean reformulations of every
active expert question across all five pieces: 79 source questions and 237
unique variant formulations. It applies them to 123 normal source/range
contexts plus 38 additional no-range semantic contexts, for 161 formulation
groups and 483 inference cases. The no-range cohort contributes 114 cases.
The excluded unanswerable Nella Fantasia source remains in the dataset for
provenance but is never expanded or sent to the pipeline.

The runner requires clean hybrid retrieval and grounded local generation. It
loads both GGUF checkpoints before creating the result artifact, refuses
missing or incompatible models, and accepts only `expert_verbatim` /
`authoritative_expert`, `llm` / `retrieved_evidence`, or an explicit
`unavailable` / `no_corpus_evidence` abstention:

```bash
SQA_VARIANT_DATASET=/home/minhee/soprano-qa-dataset

conda run -n soprano-qa python \
  evaluation/run_synthesized_questions.py \
  --dataset-root "$SQA_VARIANT_DATASET" \
  --output evaluation/synthesized_question_results.json \
  --top-k 6
```

The single JSON artifact contains the human source answer, linked knowledge
units, exact case authority where schema 1.3 provides it, generated answers,
evidence, retrieval diagnostics, generation modes and unavailable reasons,
model hashes, and runtime fingerprints. It checkpoints atomically after every
case and can resume only when the authenticated benchmark, corpus, code,
models, runtime,
and settings still match. `--limit N` is useful for a bounded invocation.

The current completed artifact has validated integrity and contains 483/483
completed inference cases with zero execution errors. It records 346
`expert_verbatim` answers with an `authoritative_expert` basis and 137 explicit
unavailable answers with a `no_corpus_evidence` basis. The abstentions comprise
124 cases with insufficient authoritative evidence, ten with missing
range-scoped causal authority, and three with ambiguous dense grounding. No
case is replaced with an extractive or internal-knowledge answer path.

Across the full mixed benchmark, expected knowledge-unit Hit@1 is 454/483
(93.9959%), Hit@6 is 470/483 (97.3085%), and MRR is 0.955487. All three variants
achieve Hit@6 in 154/161 formulation groups (95.6522%). The purpose-built
no-range cohort is stronger at the acceptance cutoff: Hit@1 is 109/114
(95.6140%), Hit@3 is 114/114 (100%), and MRR is 0.978070 across all five pieces.
Of those 114 cases, 104 return the authoritative expert answer and ten
explicitly abstain under the stricter answer-admissibility gate. See
[`synthesized_question_comparison.md`](synthesized_question_comparison.md) for
the complete retrieval and generation-path report. The tracked run uses no
internal-knowledge answer basis, exposes no evidence/provenance IDs, contains
no `제공된 검색 근거` footer, satisfies formal-polite style, and contains no
repeated horizontal whitespace. No-range answers also omit legacy scope
headings such as `범위 안내` and `확인된 국소 예시` and keep unrequested
canonical ranges in structured evidence metadata instead.

No semantic LLM judge was run for this synthesized result. Retrieval and
generation-path metrics are diagnostics, not answer-accuracy verdicts. Use
the viewer below to compare each human expected/reference answer with the
generated answer manually.

## Current result artifacts

- `qualitative.json` preserves the current five-piece evaluation questions,
  generated answers, expert references, and retrieval diagnostics.
- `qualitative_judged.json` is the separate output target for the 41-case
  Codex CLI answer-fidelity triage. Treat it as a result only when its run
  status is complete; it never modifies the inference artifact.
- `synthesized_question_results.json` preserves the completed five-piece
  synthesized-question hybrid answer run and its human references.

The judgment artifact is separate from both inference artifacts; the
synthesized runner does not create judge verdicts.

Runtime `*.log` and `*.json.lock` files are ignored and should not be
committed.

## Read-only result viewer

The local viewer reads the complete canonical and synthesized result artifacts
without modifying either source file:

```bash
conda run -n soprano-qa python evaluation/server.py
```

Open <http://127.0.0.1:8766/>. The viewer presents 164 original inference
results and 483 synthesized inference results across all five pieces. It joins
them into 164 scenario groups by source question, case kind, and measure range;
161 groups contain one original result and three synthesized variants, while
three reported regression scenarios are original-only. Original and
synthesized question wording is separated in both the detail view and result
tabs, and all 647 generated outputs remain individually selectable. The
Question type filter can show both families or either one alone. Piece,
generation-mode, status, and synthesized-variant filters remain available.
Manual review state is stored in the browser's `localStorage`; export it from
the viewer if it must be retained or shared.

The defaults are `evaluation/qualitative.json` and
`evaluation/synthesized_question_results.json`. To compare compatible files at
other paths, pass both `--original-results-file` and
`--synthesized-results-file`. The server fails closed if either artifact is
missing, incomplete, stale, or cannot be joined exactly.

## Tests

Run the focused evaluation tests:

```bash
conda run -n soprano-qa python -m unittest -v \
  tests.test_qualitative_runner \
  tests.test_qualitative_judge \
  tests.test_annotator_retrieval_coverage \
  tests.test_synthesized_questions_runner \
  tests.test_evaluation_server
```

Run the complete repository suite:

```bash
conda run -n soprano-qa python -m unittest discover -s tests -v
```
