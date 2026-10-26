# Soprano QA RAG System

Local, measure-aware hybrid retrieval and grounded answer generation for five
soprano works. Every grounded request sends the complete selected, reviewed
knowledge-unit answer texts to one local Qwen3-14B call, which composes a
natural singer-facing Korean answer. Retrieval confidence changes which
evidence is supplied, but never switches to a verbatim, extractive, ID-only,
or internal-knowledge answer path.

This repository is the RAG/LLM layer. It reads reviewed expert content from the
sibling dataset repository and is consumed by the sibling demo application.

It is a Python library with a CLI, not a deployable service: HTTP transport,
process management, and authentication are the caller's responsibility.
Everything runs locally, and two GGUF checkpoints totalling about 12.8 GB must
be on disk before a grounded request can be served.

## Setup

### 1. Clone the repositories as siblings

Directory names matter: the default paths in `config/settings.json` resolve
against this layout.

```bash
mkdir soprano-qa-workspace
cd soprano-qa-workspace
git clone https://github.com/minigb/soprano-qa-dataset.git
git clone https://github.com/minigb/soprano-qa-rag-system.git
git clone https://github.com/minigb/soprano-qa-demo.git
```

```text
soprano-qa-workspace/
├── soprano-qa-dataset/
├── soprano-qa-rag-system/
└── soprano-qa-demo/
```

The dataset repository is required. The demo repository is optional unless you
are running the web application.

### 2. Create the shared Conda environment

Install Conda (Miniconda or Miniforge) and initialize it for your shell. All
three repositories share one `soprano-qa` environment. Run from
`soprano-qa-workspace/`:

```bash
cd soprano-qa-dataset
conda env create -f environment.yml   # or: conda env update -n soprano-qa -f environment.yml
conda activate soprano-qa

cd ../soprano-qa-rag-system
python -m pip install -r requirements.txt
```

For NVIDIA CUDA 12.4, replace the `pip install` command with:

```bash
python -m pip install -r requirements.txt \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Other platforms may build `llama-cpp-python` locally or use an appropriate
platform-specific wheel.

### 3. Build the corpus and download the models

```bash
python scripts/build_corpus.py
python scripts/download_embedding_model.py
python scripts/download_model.py
```

| Checkpoint | Source | Size | Required when |
| --- | --- | --- | --- |
| `Qwen3-14B-Q6_K.gguf` | `Qwen/Qwen3-14B-GGUF` | 12.1 GB | `generate=True` |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | `Qwen/Qwen3-Embedding-0.6B-GGUF` | 639 MB | `retrieval.mode` is `hybrid` |

Both checkpoints are validated and loaded before corpus or retrieval work
begins; a missing or incompatible model stops the request with an error rather
than silently changing the pipeline. `retrieval.n_gpu_layers` and
`llm.n_gpu_layers` default to `-1` (all available GPU layers); lower them in
`config/settings.json` when both checkpoints must share a smaller GPU.

### 4. Verify the installation

```bash
python scripts/ask.py \
  --piece die-forelle \
  --measures 2-5 \
  --question "피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?" \
  --no-generate

python -m unittest discover -v
```

`--no-generate` exercises retrieval only, confirming the corpus and embedding
model without loading the 12.1 GB answer model.

## Usage

```bash
python scripts/ask.py \
  --piece die-forelle \
  --measures 28-30 \
  --question "28마디부터 분위기 변화를 어떻게 표현해야 하나요?"
```

Omit `--measures` when no score range is selected. This does not declare that
every retrieved claim applies to the whole work: a directly matching local
expert unit may answer the question, but its confirmed range still constrains
internal routing while ranges and provenance stay in structured evidence
metadata. When the question explicitly names a bar inside a broader
`--measures` selection, that named bar is the effective grounding scope. Add
`--json` for structured retrieval, evidence, rights, citation, and answer
metadata.

## Python API

```python
from soprano_qa.service import ask

result = ask(
    piece_id="die-forelle",
    question="피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?",
    measure_range=(2, 5),
    generate=True,
)
```

Arguments are keyword-only: `piece_id`, `question`, `measure_range`
(`tuple[int, int] | None`), `generate` (`bool`), `top_k` (default `6`), and
`language` (`"ko"`, the default, or `"en"`; anything else raises
`ValueError`). Valid piece IDs are `die-forelle`, `in-flowery-clouds`,
`la-capinera`, `nella-fantasia`, and `una-voce-poco-fa`.

```python
result = ask(
    piece_id="die-forelle",
    question="What does the piano accompaniment depict in these bars?",
    measure_range=(2, 5),
    generate=True,
    language="en",
)
```

With `language="en"` the question is English and a generated answer is
English. The extractive `generate=False` path still returns the Korean
evidence text (`answer_language: "ko"`). `language="ko"` is unchanged and
makes no extra model call.

`ask()` returns a `dict`. The fields to branch on:

| Field | Values |
| --- | --- |
| `answer` | User-visible prose, already sanitized (Korean, or English for a generated `language="en"` answer) |
| `language` / `answer_language` | Requested language / language of `answer` |
| `retrieval_query` | The text used for retrieval: the question itself, or its Korean translation for `language="en"` |
| `generation_mode` | `llm` / `retrieval_only` / `unavailable` |
| `answer_basis` | `retrieved_evidence` / `retrieved_secondary_context` / `no_corpus_evidence` |
| `unavailable_reason` | Set only when no answer could be grounded |
| `generation_validation_warning` | Set when a deterministic check questioned the draft |

Only `generation_mode: "llm"` with `answer_basis: "retrieved_evidence"` is a
completed generated answer; `unavailable` means none could be grounded. The
remaining fields are audit metadata, including the ordered `evidence` bundle
with ranges and provenance. Record IDs and evidence labels are not for display.

`model_status()` and `corpus_stats()` report checkpoint and corpus state.
`corpus_stats()` initializes the retrieval index, so prefer it at startup.

`GenerationBackendUnavailable`, `DenseRetrievalUnavailable`,
`GeneratedAnswerRejected`, and `ValueError` propagate instead of degrading the
answer path; the first two indicate a misconfigured installation.

Generation holds a process-wide lock, so threads do not increase throughput and
one process answers one generated question at a time. Loaded models are cached
for the life of the process, so a long-lived process is strongly preferred.

## Configuration

`config/settings.json` holds corpus, model, retrieval, and generation defaults.
Its relative paths resolve against the settings file's own directory
(`config/`), not the working directory, so the system can be launched from
anywhere. `config/feature_overrides.json` holds optional recurring
measure-range overrides.

| Variable | Overrides |
| --- | --- |
| `SOPRANO_QA_RAG_DATASET_ROOT` | Dataset/corpus repository root |
| `SOPRANO_QA_MODEL_PATH` | Local answer GGUF checkpoint |
| `SOPRANO_QA_EMBEDDING_MODEL_PATH` | Local embedding GGUF checkpoint |
| `SOPRANO_QA_SETTINGS` | Alternative settings file, e.g. an evaluation condition in `config/experiments/` |

Two details worth knowing before debugging a path problem. Settings resolve at
import time, so these variables must be set before `soprano_qa.service` is
first imported. And the legacy `SOPRANO_QA_DATASET_ROOT` is ignored by the
service facade while `scripts/build_corpus.py` still honours it, so setting
only the legacy name gives a builder and a service that disagree about which
dataset they use, with no error.

Changes to the dataset's review files invalidate the corpus fingerprint, so the
next query rebuilds and reloads the local artifact automatically.

Web evidence is selected by two settings. `web_export_files` lists the
answer-eligible `database/exports` partitions; an empty list disables that
lineage. `web_coaching` controls the dataset's `web_coaching/` lineage:

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Build `web_coaching` records (claims from web pages and voice-lesson videos) |
| `generation_supplement` | `4` | Up to this many web_coaching records appended after expert evidence (no-range questions only) |
| `generation_supplement_gate` | `"dense_margin"` | Append only when the best web_coaching claim's dense similarity beats every main-index candidate by the margin; `"always"` appends regardless |
| `generation_supplement_margin` | `0.05` | Margin for the `dense_margin` gate |

`score_context.lyrics` (default `true`) supplies the sung text of the selected
bars, from the dataset's `sheet_music/lyrics/`, to selected-range questions.

The four conditions compared in the web_coaching evaluation are kept in
`config/experiments/`; their corpora are written to the gitignored
`data/experiments/`.

## How answers are produced

The corpus combines expert measure annotations with answer-eligible web
database records and `web_coaching` claims, and holds 946 records across the
five pieces (122 expert, 101 web database, 723 web_coaching). Retrieval fuses
BM25 and Qwen3 dense ranks in memory, with corpus vectors cached locally. Piece
identity, evidence eligibility, and measure scope are deterministic
constraints.

Expert evidence stays authoritative. web_coaching claims are ranked in a
separate index (BM25 over the claims plus the same dense model), so the main
index and its calibrated thresholds see exactly the expert and web-database
records. For a question without a selected range, up to four web_coaching
claims are appended after the expert-led selection only when the best claim
is more similar to the question than every main-index candidate by 0.05
(dense cosine). web_coaching records are built only after the dataset's own
`web_coaching/tools/validate.py` reports no errors, which includes checking
every claim's excerpt verbatim against its stored page or transcript.

For a selected range, the sung text of those bars (model-transcribed,
labelled unverified in the prompt) is added as score context. It may only
identify the words sung there; when no evidence grounds the bars and
retrieval did not deliberately abstain, the model may answer from it alone
(`answer_basis: "score_context"`).

Corpus construction is fail-closed: every knowledge unit must have
`rewrite_status: ready` and a finalized measure status. The reviewed
`knowledge_units.answer` text is the authoritative expert input, while raw
source answers, legacy range hints, and editorial notes stay in the dataset
repository and are never serialized into `data/corpus.json`.

For every grounded request the complete selected curated answers are frozen
into one prompt, and Qwen3-14B composes one formal-polite response. There is
exactly one model call, with no second call and no verbatim, extractive,
alternate-model, or internal-knowledge fallback. Where evidence is genuinely
ambiguous the service fails closed with `unavailable` / `no_corpus_evidence`
rather than choosing a candidate that happens to rank slightly higher.

English questions (`language="en"`) add one call before retrieval: Qwen3-14B
translates the question into Korean, and that translation drives the
Korean-calibrated retrieval and gating, while measure mentions are still read
from the English question. The same frozen evidence bundle is then answered
with an English variant of the system prompt that keeps every grounding,
scope, and certainty rule, translates the Korean evidence faithfully, and keeps
lyric words and foreign terms in their original language.

See [docs/rag-llm-pipeline-method.md](docs/rag-llm-pipeline-method.md) for the
full retrieval, measure-routing, prompt-construction, and generation design.

## Evaluation

```bash
conda run -n soprano-qa python evaluation/run_qualitative.py
conda run -n soprano-qa python evaluation/run_synthesized_questions.py
conda run -n soprano-qa python evaluation/run_web_coaching_questions.py \
  --output evaluation/web_coaching_question_results.json
```

The third runner answers the 58 questions in the dataset's
`web_coaching/evaluation/questions/`, each annotated with the web_coaching
claims that answer it, and reports gold-claim recall over the generated
evidence by question type.

These write `evaluation/qualitative.json` (105 original-question cases) and
`evaluation/synthesized_question_results.json` (355 cases: 306 paraphrases and
49 derived from finalized knowledge units without an original question),
covering all five pieces. No arguments are needed in the sibling layout; pass
`--system-root` or `--pipeline-dataset-root` to evaluate an experimental
worktree, and `--limit N` to run a bounded number of incomplete cases. Both
artifacts are resumable, so an interrupted run continues rather than restarting.

A question with several valid annotated ranges is evaluated at one
deterministic representative range, selected reproducibly from the sorted,
non-excluded confirmed ranges using a versioned SHA-256 hash of the source ID.
Both runners share this policy, and changing it invalidates resume
fingerprints.

Inspect the results with the read-only viewer at <http://127.0.0.1:8766/>:

```bash
conda run -n soprano-qa python evaluation/server.py
```

See [evaluation/README.md](evaluation/README.md) for the complete workflow.

## Status

The committed artifacts are a baseline rather than a promoted result, and the
current experiment is not yet merged. Approach history, known failure classes,
and the experimental snapshot registry are maintained in
[docs/pipeline-evolution-log.md](docs/pipeline-evolution-log.md). Update that
handoff before starting and after evaluating every materially different
pipeline version.

## Related repositories

- [soprano-qa-dataset](https://github.com/minigb/soprano-qa-dataset) — scores,
  audio, expert annotations, and the curated knowledge units this system
  retrieves. Required.
- [soprano-qa-demo](https://github.com/minigb/soprano-qa-demo) — the
  presentation web application. See its
  [demo setup](https://github.com/minigb/soprano-qa-demo#setup).
