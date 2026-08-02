# Soprano QA RAG System

Local, measure-aware hybrid retrieval and Qwen-based answer generation for
five soprano works. This repository is the RAG/LLM layer and uses the sibling
dataset and demo repositories described below.

## Setup

### 1. Clone the three repositories

Keep the default directory names as immediate siblings:

```bash
mkdir soprano-qa-workspace
cd soprano-qa-workspace
git clone https://github.com/minigb/soprano-qa-dataset.git
git clone https://github.com/minigb/soprano-qa-rag-system.git
git clone https://github.com/minigb/soprano-qa-demo.git
```

The resulting layout must be:

```text
soprano-qa-workspace/
├── soprano-qa-dataset/
├── soprano-qa-rag-system/
└── soprano-qa-demo/
```

### 2. Create the shared Conda environment

Install Conda (Miniconda or Miniforge) and initialize it for your shell first.
All three repositories then use the single `soprano-qa` environment. Run these
commands from `soprano-qa-workspace/`:

```bash
cd soprano-qa-dataset
conda env create -f environment.yml
conda activate soprano-qa

cd ../soprano-qa-rag-system
python -m pip install -r requirements.txt
```

If `soprano-qa` already exists, run this alternative block from
`soprano-qa-workspace/` to update it without pruning packages:

```bash
cd soprano-qa-dataset
conda env update -n soprano-qa -f environment.yml
conda activate soprano-qa
cd ../soprano-qa-rag-system
python -m pip install -r requirements.txt
```

For NVIDIA CUDA 12.4, replace the `pip install` command in the chosen block
with:

```bash
python -m pip install -r requirements.txt \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Other platforms may build `llama-cpp-python` locally or use an appropriate
platform-specific wheel.

### 3. Build the corpus and download the local models

```bash
python scripts/build_corpus.py
python scripts/download_embedding_model.py  # 639 MB; needed for dense retrieval
python scripts/download_model.py            # Optional; needed for generated answers
```

The default generation checkpoint needs approximately 5 GB. Retrieval-only
queries work without it by passing `--no-generate`. If the embedding checkpoint
is unavailable, the service reports the reason and falls back to lexical BM25
instead of downloading a model during a request.

The embedding model is loaded when the retrieval index first initializes.
`retrieval.n_gpu_layers` defaults to `-1` (all available GPU layers); lower it
in `config/settings.json` when the embedding and generation checkpoints must
share a smaller GPU.

### 4. Verify the installation

```bash
python scripts/ask.py \
  --piece die-forelle \
  --measures 2-5 \
  --question "피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?" \
  --no-generate

python -m unittest discover -v
```

## Usage

Run a generated, measure-specific answer:

```bash
python scripts/ask.py \
  --piece die-forelle \
  --measures 28-30 \
  --question "28마디부터 분위기 변화를 어떻게 표현해야 하나요?"
```

Omit `--measures` for a whole-work question. Add `--json` for structured
retrieval, evidence, rights, citation, and answer metadata.

To inspect the selected evaluation results in a local qualitative-review
workspace, run:

```bash
python3 evaluation/server.py
```

Then open `http://127.0.0.1:8766/`. The page keeps manual ratings and notes in
the browser and can export them as JSON; it does not modify the inference
snapshot. The schema-8/v13 evaluation treats generated-answer reliability as
the primary quality outcome. Curator-scoped target claims are `R` items and are
the only completeness targets; expert claims from the records actually
retrieved for generation are corpus-authenticated, range-checked `S` items
that may support candidate factuality but can never replace a missing `R`
item. An independent required-claim NLI gate always checks curator-split
atomic `R` coverage. Whenever authenticated `S` context exists, it also checks
every covered non-atomic `direct_required` or `mixed_or_ambiguous` `R`,
preventing paraphrased or unlinked `S` factuality from being transferred into
`R` completeness. The 25-control judge calibration exercises this boundary.
An exact leading `검토 주의:` disclosure is reconstructed from authenticated
corpus metadata, validated, and stripped exactly once before candidate `C`
segmentation and every downstream semantic, normalized-full-answer,
atomic-claim, range, and NLI check. Its separate
`review_disclosure_validation` audit is visible in the viewer. Missing,
mismatched, non-leading, extra, unexpected, or untrusted disclosures block an
automatic pass, and a disclosure-only response supplies no `C` coverage.
Judge JSON remains strict: only a single decoder-pointed invalid `\'` escape
may have that backslash removed on retry, with the raw output retained and the
normalization audited; arbitrary malformed JSON is still rejected. Exact
source and knowledge-unit retrieval remain secondary diagnostics. See
[evaluation/README.md](evaluation/README.md) for details.

Applications can import the service facade while this repository is on
`PYTHONPATH`:

```python
from soprano_qa.service import ask

result = ask(
    piece_id="die-forelle",
    question="피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?",
    measure_range=(2, 5),
    generate=True,
)
```

## Configuration

Defaults are repository-relative and assume the sibling layout above.

| Setting | Purpose |
| --- | --- |
| `SOPRANO_QA_RAG_DATASET_ROOT` | Override the dataset/corpus repository |
| `SOPRANO_QA_MODEL_PATH` | Override the local GGUF checkpoint |
| `SOPRANO_QA_EMBEDDING_MODEL_PATH` | Override the local embedding GGUF checkpoint |
| `config/settings.json` | Corpus, model, retrieval, and generation defaults |
| `config/feature_overrides.json` | Optional recurring measure-range overrides |

## Project notes

The corpus combines expert measure annotations with answer-eligible web
database records. Retrieval fuses BM25 and Qwen3 dense ranks in memory; the
corpus vectors are cached locally, and a bounded query cache avoids repeated
embedding work for generation retries. Piece identity, evidence eligibility,
and measure scope remain deterministic constraints. Measure-scoped questions
prioritize overlapping expert evidence; unsupported questions can use a
clearly marked internal-knowledge fallback without corpus citations.

Expert records come from the sibling dataset's active
`expert_curation/review/*.json` files. All included knowledge units remain
searchable, including units awaiting rewrite or measure review, but pending
states and review notes travel with the evidence and are shown to the model.
An unreviewed measure is never treated as a confirmed range: it may support an
unscoped query, but it is excluded from a range-selected query. A
semantically relevant record with a confirmed non-overlapping range is kept
as lower-ranked `other_range_context_only`: it remains inspectable and can
help detect a range mismatch, but it cannot ground, broaden, or receive an
automatic citation in the selected-range answer. Original annotator questions
are retained as retrieval aliases, and verbatim source answers are supplied
as authoritative context for grounded generation. Strongest range-applicable
alias matches remain primary; up to two confirmed other-range alias matches
may follow as explicitly secondary context. Source-level legacy range hints
are retained as non-authoritative disambiguation metadata for merged
multi-range units, while confirmed knowledge-unit ranges remain the only
routing authority. If one raw source was split across several knowledge
units, its full Q&A stays in the corpus for lineage but is withheld from any
single-unit prompt, preventing a claim assigned to an out-of-range sibling
unit from leaking back into the answer.

Unresolved rewrite and measure-review states are not hidden in prompt
metadata. The answer carries a deterministic Korean review disclosure before
the LLM-generated content, including unresolved rewrite notes, unconfirmed
scope, and possible wrong-piece warnings. This prevents a fluent answer from
presenting a provisional annotation as a settled score fact.
Evidence with a pending rewrite receives a second grounded compliance pass:
the warning prefix alone is not accepted if the body later restates one side
of a documented timing, location, notation, or causality conflict as settled.
An unseen code/UI-style CamelCase token in generated Korean triggers a
grounded rewrite retry, which catches visible decoding corruptions before the
answer is returned.
Changes to the review files invalidate the corpus fingerprint, so the next
query rebuilds and reloads the local artifact automatically. No data copy into
the demo repository is required.

See
[docs/rag-llm-pipeline-method.md](docs/rag-llm-pipeline-method.md) for the
retrieval, measure-routing, prompt construction, and generation design.

To run the web application, continue with the
[demo setup](https://github.com/minigb/soprano-qa-demo#setup).
