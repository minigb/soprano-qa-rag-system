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

## Synthesized retrieval development benchmark

`run_synthesized_retrieval.py` evaluates three Korean reformulations of each
case in the retained development-variant dataset. That dataset contains 153
cases for Die Forelle, In Flowery Clouds, and La Capinera. Its three-piece
scope describes the available synthesized benchmark data; it does not limit
the five-piece qualitative workflow above.

The main repository is currently configured for hybrid retrieval, so a valid
current run is:

```bash
SQA_VARIANT_DATASET=/home/minhee/soprano-qa-dataset-evaluation-set-synthesized

conda run -n soprano-qa python \
  evaluation/run_synthesized_retrieval.py \
  --system-root /home/minhee/soprano-qa-rag-system \
  --dataset-root "$SQA_VARIANT_DATASET" \
  --output evaluation/synthesized_retrieval_hybrid.json \
  --top-k 6 \
  --require-retrieval-mode hybrid
```

The runner separates the benchmark dataset from the evaluated system, turns
off generation and internal-knowledge fallback, fingerprints the target code,
configuration, corpus, and dense assets, and checkpoints after every case.
Use a separately configured lexical system root for a new lexical-versus-
hybrid comparison; the removed historical worktrees are not required by the
retained result files.

The tracked `synthesized_retrieval_bm25.json`,
`synthesized_retrieval_dense.json`, and `synthesized_retrieval_final.json`
files are retrieval-only development snapshots. They do not contain LLM judge
verdicts.

## Retained artifacts

- `qualitative.json` preserves the current five-piece evaluation questions,
  generated answers, expert references, and retrieval diagnostics.
- `synthesized_question_results_final.json` preserves the completed
  synthesized-question RAG and evaluation results.
- `synthesized_retrieval_bm25.json`, `synthesized_retrieval_dense.json`, and
  `synthesized_retrieval_final.json` preserve retrieval development results.

The synthesized-question snapshot includes the historical verdicts recorded
by that run. They are preserved as part of the result, but they are not output
from the current `judge_qualitative.py` workflow.

Runtime `*.log` and `*.json.lock` files are ignored and should not be
committed.

## Read-only result viewer

The local viewer can inspect a compatible detailed result snapshot without
modifying it. For example:

```bash
conda run -n soprano-qa python evaluation/server.py \
  --results-file evaluation/synthesized_question_results_final.json
```

Open <http://127.0.0.1:8766/>. Manual review state is stored in the browser's
`localStorage`; export it from the viewer if it must be retained or shared.

## Tests

Run the focused evaluation tests:

```bash
conda run -n soprano-qa python -m unittest -v \
  tests.test_qualitative_runner \
  tests.test_qualitative_judge \
  tests.test_annotator_retrieval_coverage
```

Run the complete repository suite:

```bash
conda run -n soprano-qa python -m unittest discover -s tests -v
```
