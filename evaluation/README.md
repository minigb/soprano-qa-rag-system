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
lexical retrieval. The qualitative judge applies the same fail-closed policy
to its Qwen checkpoint and backend.

## Current qualitative evaluation

Generate range-aware RAG answers and save their linked expert references:

```bash
conda run -n soprano-qa python \
  evaluation/run_qualitative.py --generate
```

The default output is `evaluation/qualitative.json`. The tracked snapshot is
complete and currently contains 79 questions and 124 inference cases across
the five supported pieces. The runner checkpoints each case and can resume an
interrupted run. Use `--piece PIECE_ID` or `--limit N` for a focused run.

Run conservative answer-fidelity triage with the configured local Qwen GGUF:

```bash
conda run -n soprano-qa python \
  evaluation/judge_qualitative.py
```

The judge supports only `llama-cpp-python` in the `soprano-qa` environment.
Before creating or updating `evaluation/qualitative_judged.json`, it:

- validates the input artifact;
- requires the configured model file and the `llama_cpp` package;
- initializes the GGUF and verifies that its declared architecture is Qwen;
- fingerprints the model, backend, interpreter, generation settings, input,
  and judge implementation.

The judge calls the low-level LLM interface directly, so it cannot enter the
service's extractive fallback path. A backend or programming failure is
checkpointed once and aborts the run. Only a malformed judge response uses
the bounded per-case retry loop. Retrieval ranks remain diagnostics and do
not determine the pass, review, or fail verdict.

The answer artifact records the generator's model path but not its checkpoint
hash. The judgment artifact therefore records whether the configured paths
match, while explicitly leaving same-checkpoint identity unverified. A Qwen
judgment is automated triage, not independent human evidence.

Useful judge controls include `--piece PIECE_ID`, `--case-id CASE_ID`,
`--limit N`, `--max-attempts N`, and `--minimum-pass-rate RATE`. Resume is
allowed only when the source cases and all fingerprinted judge inputs match.

## Synthesized-question hybrid RAG+LLM benchmark

`run_synthesized_questions.py` evaluates three Korean reformulations of every
active expert question across all five pieces: 79 source questions, 237
variant formulations, and 372 range-expanded inference cases. The excluded
unanswerable Nella Fantasia source remains in the dataset for provenance but
is never expanded or sent to the pipeline.

The runner requires clean hybrid retrieval and grounded local generation. It
loads both GGUF checkpoints before creating the result artifact, refuses
lexical fallback or missing/incompatible models, and always calls the service
with generation enabled and internal model knowledge disabled:

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
evidence, retrieval diagnostics, generation modes and reasons, model hashes,
and runtime fingerprints. It checkpoints atomically after every case and can
resume only when the authenticated benchmark, corpus, code, models, runtime,
and settings still match. `--limit N` is useful for a bounded invocation.

The current completed artifact has validated integrity and contains 372/372
successful inference cases. It records 240 grounded LLM answers, 126
retrieval-extractive safeguards, and six unavailable answers where the
retriever admitted no corpus evidence and internal model knowledge remained
disabled. Expected knowledge-unit Hit@6 is 359/372 (96.51%), and all three
variants achieve
Hit@6 in 119/124 canonical source/range groups (95.97%). See
[`synthesized_question_comparison.md`](synthesized_question_comparison.md) for
the complete retrieval and generation-path report. No output uses an internal
knowledge answer basis or contains the removed `제공된 검색 근거` footer.

No semantic LLM judge was run for this synthesized result. Retrieval and
generation-path metrics are diagnostics, not answer-accuracy verdicts. Use
the viewer below to compare each human expected/reference answer with the
generated answer manually.

## Current result artifacts

- `qualitative.json` preserves the current five-piece evaluation questions,
  generated answers, expert references, and retrieval diagnostics.
- `synthesized_question_results.json` preserves the completed five-piece
  synthesized-question hybrid RAG+LLM run and its human references.

`qualitative_judged.json` is created only when the optional qualitative judge
is run. It is separate from both inference artifacts; the synthesized runner
does not create judge verdicts.

Runtime `*.log` and `*.json.lock` files are ignored and should not be
committed.

## Read-only result viewer

The local viewer can inspect a compatible detailed result snapshot without
modifying it. For example:

```bash
conda run -n soprano-qa python evaluation/server.py \
  --results-file evaluation/synthesized_question_results.json
```

Open <http://127.0.0.1:8766/>. The synthesized-result view shows the human
expected/reference answer beside the generated answer and supports piece,
generation-mode, status, and variant filters. Manual review state is stored
in the browser's `localStorage`; export it from the viewer if it must be
retained or shared.

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
