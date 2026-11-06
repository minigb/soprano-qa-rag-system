# Soprano QA RAG System

Local, measure-aware hybrid retrieval and answer generation for five soprano
works. The experimental `llm.pipeline="single_pass"` uses the existing BM25 + embedding
retriever, then one Qwen3-14B completion. There is no model-based input admission,
KU reselection, output review, quality rewrite, or source-text substitution.
The generation prompt handles service scope and untrusted instructions, asks for
faithful use of relevant passages, and permits general music knowledge when useful.
Unrelated requests should receive a friendly explanation of the assistant's scope.
This prompt is guidance, not a guarantee that every answer is correct or attack-proof.

An optional [Vertex Gemini writer](docs/gemini-vertex.md) is available through
`config/settings.gemini.json`. It preserves the same retrieval and prompt while
using `gemini-3.8-flash` for answer generation. The local configurations remain
available; English retrieval-query translation still uses Qwen3-14B.

Implementation: `codex/helpful-rag-responses`, in
`/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`.
The pre-injection original remains in the `rag-pre-injection` worktree. The
model remains Qwen3-14B-Q6_K. The candidate prompt and one-call writer are in
[`soprano_qa/single_pass.py`](soprano_qa/single_pass.py); the library and CLI
both enter through `response_policy.compose_music_answer`. The historical
reviewed path remains available with `llm.pipeline="reviewed"` for comparison.
This checkout does not switch the old Lab backend to the new implementation.

Input shape validation and inexpensive source-text screening remain. No model
is used for these checks. Classification `topic` metadata stays in retrieval
but is omitted from generation inputs. Unsafe retrieved material is excluded
before the one writing call; it does not trigger a second answer-generation
attempt. The answer itself is not evaluated or rewritten before delivery.

This repository is the RAG/LLM layer. It reads reviewed expert content from the
sibling dataset repository and is consumed by the sibling demo application.

It is a Python library with a CLI, not a deployable service: HTTP transport,
process management, and authentication are the caller's responsibility.
Answer inference runs locally, and two GGUF checkpoints totalling about 12.8 GB must
be on disk before a grounded request can be served.

The validated default configuration is preserved. The single-pass candidate
is opt-in through `config/settings.single-pass.json`: its speed improved, but
expert meaning errors increased in offline regression checks. The following
single-pass behavior describes that candidate, not a replacement of the tested
reviewed default. See [the timing and quality report](evaluation/single_pass_2026-11-04/report.md).

To try the candidate with the CLI, add
`--settings config/settings.single-pass.json`. For the Python API, set
`SOPRANO_QA_SETTINGS` to that file's absolute path **before** importing the service.
The previous default and its evaluations are documented in
[evaluation/helpful_responses_2026-11-04/README.md](evaluation/helpful_responses_2026-11-04/README.md).

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

After input screening, required checkpoints are validated and loaded before
corpus or retrieval work begins; a missing or incompatible model stops the
request with an error rather than silently changing the pipeline. Known
control attacks are rejected before model loading. `retrieval.n_gpu_layers` and
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
evidence text (`answer_language: "ko"`). English adds a query-translation call.
Korean generation makes one writing call; English adds one query translation,
then one writing call. Both use the same 14B checkpoint. No source-translation
catalog or separate relevance/quality model runs in the candidate path.

`ask()` returns a `dict`. The fields to branch on:

| Field | Values |
| --- | --- |
| `answer` | Generated prose (not quality-reviewed) (Korean, or English for a generated `language="en"` answer) |
| `language` / `answer_language` | Requested language / language of `answer` |
| `retrieval_query` | The text used for retrieval, or `null` when recovery excludes unsafe material |
| `generation_mode` | `llm_single_pass`, `response_error`; `retrieval_only` / `unavailable` for the extractive path. The historical reviewed path has additional modes |
| `answer_basis` | Retrieved evidence, score context, general music knowledge (possibly combined), or `service_scope` / `service_status` |
| `unavailable_reason` | Machine-readable reason for domain guidance, service errors or unavailable extractive results; `null` for a delivered music answer |
| `generation_validation_warning` | `null` or `unsafe_source_excluded` in the candidate path; legacy quality-warning values only in the historical reviewed path |

`llm_single_pass` identifies generated text, not a quality or scope verdict.
`answer_basis` describes the supplied context; it does not verify each generated
claim. Scope explanations are generated by the same writer. Empty model text
returns `response_error` technical guidance without another generation attempt.
The `evidence` bundle retains retrieval provenance and is not a replacement answer.
Record IDs and evidence labels are not intended for end-user display.

`model_status()` and `corpus_stats()` report checkpoint and corpus state.
`corpus_stats()` initializes the retrieval index, so prefer it at startup.

`GenerationBackendUnavailable`, `DenseRetrievalUnavailable`, and invalid-argument
`ValueError` exceptions still propagate; the first two indicate a misconfigured
installation. Unsafe retrieved material can lead to a one-call general music
answer; invalid inputs remain explicit service errors. Ordinary answer content
never enters the historical deterministic or model-based output finalizers.

Each local-model completion holds a process-wide lock, so concurrent requests
may wait. Loaded models are cached for the process lifetime; a long-lived
process avoids loading them for every request. Timing reports separate cold
loading from warmed requests and include English query translation.

## Configuration

`llm.pipeline="single_pass"` selects the one-call writer. Historical controls
such as `source_preservation`, `synthesis_prompt`, `synthesis_reasoning` and
`evidence_translation` belong to the reviewed path; they do not add steps to
single-pass generation. The model, context length, sampling and token limits
still use the common `llm` settings.

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
identify the words sung there, not establish pitches, rhythm or dynamics.
The single-pass response metadata describes the supplied retrieval context;
it does not judge whether a generated claim used lyrics or background knowledge.

Corpus construction is fail-closed: every knowledge unit must have
`rewrite_status: ready` and a finalized measure status. The reviewed
`knowledge_units.answer` text is the authoritative expert input, while raw
source answers, legacy range hints, and editorial notes stay in the dataset
repository and are never serialized into `data/corpus.json`.

The existing retriever, source priority, measure filters and supplement logic
select the context. Their BM25 and dense scores are unchanged by the single-pass
update. No 14B model reranks or selects a second subset. The complete selected
source answer texts reach the writing prompt along with question, work, scope,
source type and available attribution. Nonessential internal metadata is omitted.

Application instructions occupy the system role; the question, passages and
score lyrics are serialized as untrusted JSON data in the user role. The writer
returns Korean or English prose directly, using the initial system's natural-prose
writing instructions adapted for general music knowledge and task scope. It
should preserve qualifiers, alternatives,
notation and attribution, and distinguish conflicting sources. When context is
insufficient, it can use general music knowledge without inventing facts about
a specific score. There is no online check of whether it succeeded, no content
retry, and no BERTScore-based response choice.

English questions add one Korean translation before retrieval. The original
English question remains the writing task; only the search query is translated.
The output writer translates relevant source content itself in the same call.

The previous reviewed implementations and their limitations are recorded in
[evaluation/helpful_responses_2026-11-04/README.md](evaluation/helpful_responses_2026-11-04/README.md)
and the frozen E37 experiments. Their multi-call checks, structured citation
rendering, original-text substitution and translation catalog do not describe
the single-pass candidate's unreviewed delivery. The single-pass writer uses the preserved
initial writing instructions in `soprano_qa/single_pass_prompt.py`, with no
constrained JSON draft or historical source renderer. Development evaluation is separate from serving: fixed-setting
BERTScore and offline Claude checks never block or replace a user's answer.

## Evaluation

The [local answer-review demo](sites/answer-review/README.md) displays all 460
paired cases with searchable questions, piece and quality filters, full KU
evidence, original Claude assessments, and browser-local review notes. Start
it with `python -m http.server 8766 --bind 127.0.0.1 --directory sites/answer-review/dist`
and open <http://localhost:8766/>. It displays the saved evaluation answers.

The [expanded 460-case comparison](evaluation/expanded_460_2026-11-02/report.md)
reruns the existing Korean questions on E37 and the pre-security original, with
fresh Claude Opus 5.5 judgments. Material source-meaning changes fall from 27 to
0, but naturalness falls from 3.50 to 2.89 out of 4. Expert original-validation
BERTScore F1 remains 0.972987 (62/62 answers). Overall answer quality has not
been preserved: repeated quotations and weak sentence connections need work.
The report documents two overlength answers, the native corpus difference,
and the [complete answer comparison](evaluation/expanded_460_2026-11-02/review/review.html).

The [2026-11-02 follow-up comparison](evaluation/ku_background_2026-10-31/followup-report.md)
covers 290 Korean/English normal and adversarial requests per system. Expert
validation BERTScore F1 is 0.972987 (62/62 answers), with 0/58 observed injection
successes, no judged material meaning drift, and 0/216 normal false refusals.
The cases were reused during refinement; literal KU quotations raise reference
overlap, and the score is not factual accuracy. The report compares the original
system and E25 using independent Claude judgments, direct meaning checks and
paired latency. The [earlier report](evaluation/ku_background_2026-10-31/report.md)
retains the initial-security and strict-baseline comparisons.

The [2026-10-25 bilingual security comparison](evaluation/security_comparison_2026-10-25/report.md)
contains 400 measured requests comparing the original and hardened systems on
Korean/English attacks and answerable music questions. It reports attack success,
false refusals, blinded answer quality and warm latency, including a repeatable
remaining bypass and the hardening's usability costs.
Those findings concern the earlier frozen snapshots.

The [2026-10-31 quality refinement](evaluation/quality_refinement_2026-10-31/report.md)
compares the current secured baseline with the refined pipeline on 120 bilingual
cases (240 requests). False refusals fall from 6/50 to 1/50 and direct attack
success from 3/36 to 0/36; effective rubric quality rises from 78.0 to 93.4/100.
Two refined answers still contain material support errors. The report includes
language breakdowns, latency, regressions, review limitations and a separate
audit of the static demo samples. This authored suite is not a security guarantee.

The subsequent [meaning-fidelity audit](evaluation/meaning_fidelity_2026-10-31/report.md)
found ten new material regressions on targeted historical and demo questions,
despite unchanged returned evidence. Material errors increased from 14 to 19
across its 39 music requests. The refinement has not established meaning
preservation; see the [side-by-side answers](evaluation/meaning_fidelity_2026-10-31/review.html)
and exported [Korean](evaluation/meaning_fidelity_2026-10-31/prompts/answer-ko.txt)/[English](evaluation/meaning_fidelity_2026-10-31/prompts/answer-en.txt)
system prompts. This difficult regression set is not a production accuracy estimate.

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

### Direct Sumi Jo feedback and source priority

The default runtime orders relevant, scope-eligible knowledge units as Sumi Jo,
then experts, then web evidence. See [source-priority documentation](docs/source-priority.md)
and the [retrieval development pilot](evaluation/source_priority/README.md).
Sumi Jo feedback currently covers Nella fantasia and has no confirmed measure
locations; original recordings and transcripts remain in the dataset repository.
