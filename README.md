# Soprano QA RAG System

Local, measure-aware hybrid retrieval and grounded answer generation for five
soprano works. Every grounded generated request sends the complete selected,
reviewed knowledge-unit answer texts to one local Qwen3-14B call, which
composes a natural singer-facing answer. Retrieval confidence changes which
evidence is supplied, but never switches to a verbatim, extractive, ID-only,
or internal-knowledge answer path. This repository is the RAG/LLM layer and
uses the sibling dataset and demo repositories described below.

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
python scripts/download_model.py            # Required for generated answer requests
```

The default answer checkpoint is the instruction/chat model
`Qwen3-14B-Q6_K.gguf`, from `Qwen/Qwen3-14B-GGUF`, and needs approximately
12.1 GB of disk space. No `chat_format` override is configured:
`llama-cpp-python` uses the native `tokenizer.chat_template` embedded in the
GGUF. Every generated request validates and loads that checkpoint and its
backend before corpus or retrieval work begins. Retrieval-only queries can
omit it by passing `--no-generate`. The embedding checkpoint is mandatory when
`retrieval.mode` is `hybrid`; the system validates it before rebuilding the
corpus or answering a query. Missing, unreadable, or incompatible required
models stop the request with an error. To run intentionally without an
embedding model, explicitly set `retrieval.mode` to `lexical`.

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

Omit `--measures` when no score range is selected. This does not declare that
every retrieved claim applies to the whole work: a directly matching local
expert unit may answer the question, but its confirmed range still constrains
internal routing, while ranges and provenance remain in structured evidence
metadata. User-visible prose stays natural and does not print pipeline labels
or canonical measure numbers that the user did not request; it still must not
turn that local annotation into a whole-work or frequency claim. Add `--json`
for structured retrieval, evidence, rights, citation, and answer metadata.
When the question explicitly names a bar inside a broader `--measures`
selection, that named bar is the effective retrieval and grounding scope; the
broader selection is only its validated envelope.

Run the current five-piece original-question inference with:

```bash
conda run -n soprano-qa python evaluation/run_qualitative.py
```

The runner writes `evaluation/qualitative.json`. The current protocol contains
105 original-question inference cases across all five pieces. A question with
several valid annotated ranges is evaluated at one deterministic,
range-overlapping representative range instead of being repeated for every
range. The representative is selected reproducibly from the sorted,
non-excluded confirmed ranges using a versioned SHA-256 hash of the source ID.
The full range inventory remains in the artifact, and changing the selection
policy invalidates resume fingerprints.

To test wording robustness and knowledge-unit coverage, run the synthesized
family evaluation:

```bash
conda run -n soprano-qa python \
  evaluation/run_synthesized_questions.py \
  --dataset-root ../soprano-qa-dataset \
  --output evaluation/synthesized_question_results.json
```

The synthesized-family artifact contains 355 inference cases: 306 paraphrase
cases and 49 coverage cases derived from 36 finalized knowledge units that had
no original related evaluation question. These derived questions exist only
for evaluation; they are never added to the corpus, retrieval aliases, or
production prompts. The original and synthesized runners share the same
representative-range policy.

Together, the two artifacts contain 460 generated answers. Every successfully
completed evaluation case must use `llm` / `retrieved_evidence`; a verbatim,
ID-selector, extractive, internal-knowledge, or unavailable result is not
accepted as a completed answer. Fresh retrieval and answer-quality metrics are
intentionally pending until both artifacts have been rerun with the final
pipeline code.

The chronological rationale, failed approaches, concrete red-flag samples,
distinct experimental RAG+LLM snapshot branches/worktrees, and current
experiment status are maintained in
[`docs/pipeline-evolution-log.md`](docs/pipeline-evolution-log.md). Update that
living handoff before starting and after evaluating every materially different
pipeline version.

Direct semantic review is stored separately in
`evaluation/semantic_quality_review.json`. Each assessment is bound by a
semantic hash to the exact question, reference authority, generated answer,
ordered evidence bundle, and generation-validation warning.
A flagged answer remains visible with its severity, reason codes, and
rationale; the review never substitutes another answer.

Inspect every original and synthesized-family result with the read-only
viewer:

```bash
conda run -n soprano-qa python evaluation/server.py
```

Open <http://127.0.0.1:8766/> for the complete comparison and
<http://127.0.0.1:8766/red-flags> for the dedicated semantic-red-flag queue.
The viewer keeps original and synthesized-family questions visibly separate
and exposes all 460 results. If the review artifact is absent or stale, the
viewer reports that semantic review is unavailable instead of treating the
cases as passes. See
[evaluation/README.md](evaluation/README.md) for the complete current
workflow.

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
embedding work for repeated normalized questions. Piece identity, evidence
eligibility, and measure scope remain deterministic constraints.
Measure-scoped questions prioritize overlapping expert evidence. Generated
requests fail before corpus work if the configured answer model cannot load;
hybrid retrieval likewise fails before corpus work if its embedding model or
backend is unavailable.

When no range is selected, generation first evaluates a stable six-candidate
pool. A finalized local expert bundle may displace generic whole-piece context
only when its retrieval-text and curated-answer similarities establish a
clear semantic winner, or when a stronger relevance-led guard passes. The
decision is frozen before answer delivery, the direct local unit is ordered
first, and only finalized same-source split units may accompany it. Confirmed
ranges continue to constrain the selected bundle internally, but a no-range
answer does not print those unrequested locators. Its natural wording must
remain conditional or passage-local and cannot present the annotation as a
rule for the whole piece.
A narrower source-question-led path handles cases where the finalized answer
explains the result without repeating the question's central nouns. It requires
a clean local expert record with confirmed ranges and source-question aliases,
a raw relevance score of at least 0.64, answer similarity of at least
0.40, and a 0.10 lead over every other source; ordinary domain and explicit
constraint guards still apply. This path changes neither the global thresholds
nor the range authority of the selected unit. Source aliases recall the
annotation family rather than authorizing every split unit in it. For a pure
causal question, answer-side relation checks keep an explicitly causal sibling
and reject a technique-only sibling; if only the latter applies in the
effective selected range, the service fails closed. A mixed why-and-how
question may retain both reviewed facets.

Expert records come from the sibling dataset's active
`expert_curation/review/*.json` files. Production corpus construction is
fail-closed: every knowledge unit must have `rewrite_status: ready` and a
finalized measure status (`specific`, `whole_piece`, or `unspecified`). The
builder reports every offending knowledge-unit ID and stops instead of
indexing provisional annotations. A request without sufficient finalized
authority returns an explicit unavailable result.
Editorial rewrite and measure notes remain available in the dataset for audit
history but are never supplied to normal grounded generation.

The reviewed `knowledge_units.answer` text is the authoritative expert input.
Original annotator questions remain retrieval aliases, while ordinary raw
source answers, legacy range hints, and editorial notes remain only in the
dataset repository and are not serialized into `data/corpus.json`. The corpus
is generated reproducibly with `python scripts/build_corpus.py`; its strict
release schema rejects unexpected editorial fields instead of relying on the
answer prompt to ignore them. Confirmed knowledge-unit ranges may route a
merged multi-range unit, but they never substitute old source prose for its
curated answer. For every grounded generated request, the complete selected
curated answers are frozen into one prompt. Qwen3-14B then composes one natural
formal-polite response and may combine several compatible units when they add
useful, non-duplicative detail. Retrieval confidence never changes the answer
renderer. A semantically relevant record with a confirmed non-overlapping
range is kept
as lower-ranked `other_range_context_only`: it remains inspectable and can
help detect a range mismatch, but it cannot ground, broaden, or receive an
automatic citation in the selected-range answer. Strongest range-applicable
alias matches remain primary, and up to two confirmed other-range alias
matches may follow as explicitly secondary context.

If range-overlapping dense-only candidates are too close to distinguish
safely, the service fails closed with `unavailable` / `no_corpus_evidence`
rather than choosing one because it happens to rank slightly higher. Such a
result is not accepted as a completed evaluation answer. When grounding is
available, the pipeline makes exactly one local answer-model call against the
frozen evidence bundle. It has no second model call and no verbatim,
extractive, alternate-model, or internal-knowledge fallback. Backend and
context errors propagate as errors. If deterministic validation questions the
sole model draft, the application returns that same draft after mandatory
presentation sanitization and records the reason in
`generation_validation_warning` for semantic review.

Runtime retrieval accepts only the release corpus schema and rejects expert
records containing provisional review fields. Pending-review compatibility is
not part of the answer path. Temporary labels such as `[E1]` are citation
anchors for validation, not a knowledge-unit selection stage; the application
removes them and all opaque record IDs from user-visible prose. Validation
concerns remain structured warnings and semantic-review inputs.
Changes to the review files invalidate the corpus fingerprint, so the next
query rebuilds and reloads the local artifact automatically. No data copy into
the demo repository is required.

See
[docs/rag-llm-pipeline-method.md](docs/rag-llm-pipeline-method.md) for the
retrieval, measure-routing, prompt construction, and generation design.

To run the web application, continue with the
[demo setup](https://github.com/minigb/soprano-qa-demo#setup).
