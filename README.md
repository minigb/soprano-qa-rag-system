# Soprano QA RAG System

Local, measure-aware hybrid retrieval and grounded answer delivery for five
soprano works. Finalized expert answers are returned directly. The configured
local answer model is called at most once, and only when the leading evidence
is authoritative, answer-eligible web material. This repository is the
RAG/LLM layer and uses the sibling dataset and demo repositories described
below.

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

The default answer checkpoint needs approximately 5 GB. Every generated
request validates and loads that checkpoint and its `llama-cpp-python` backend
before corpus or retrieval work begins. Retrieval-only queries can omit it by
passing `--no-generate`. The embedding checkpoint is mandatory when
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

Run the current five-piece canonical qualitative inference and optional
Codex CLI answer-fidelity review with:

```bash
conda run -n soprano-qa python evaluation/run_qualitative.py
conda run -n soprano-qa python evaluation/judge_qualitative.py \
  --cohort measure_scoped_question_no_range
```

The inference runner writes `evaluation/qualitative.json`; the judge writes a
separate `evaluation/qualitative_judged.json` and never changes the inference
artifact. The canonical benchmark contains 79 questions and 164 inference
cases. The configured judgment cohort covers 41 no-range semantic cases
across all five pieces: 38 semantic shadows of measure-scoped questions and
three exact reported regressions. It does not judge every canonical or
synthesized answer.

The current inference artifact completed all 164 cases. Of these, 160 use the
`expert_verbatim` / `authoritative_expert` path. Four selected-range causal
questions deliberately return unavailable because those ranges contain only
a procedural sibling unit, not reviewed causal authority. In the 41-case
no-range cohort, a required expert unit is Hit@1 and Hit@3 in all 41 cases,
with MRR 1.000000.

The offline judge invokes only the authenticated Codex CLI from the
`soprano-qa` Conda environment and requires its structured JSON result to
match the script's schema. The script resolves and fingerprints the configured
model, reasoning effort, and service tier, then passes them explicitly to an
isolated `codex exec` process; the current selection is `gpt-5.6-sol` with
`ultra` reasoning and the `priority` service tier. A dedicated authenticated
provider alias disables WebSocket transport and sets HTTP-request and
SSE-stream retry counts to zero. The per-case timeout defaults to 600 seconds.
Missing CLI authentication, an unavailable model, a timeout, a failed command,
or an invalid response aborts the run. Codex CLI is the sole judge backend,
and the script makes no automatic retry. The current judge contract is schema
2.1/protocol v9.
Its verdicts are automated triage for human inspection, not independent human
evidence. Canonical ranges used for scope auditing are judge metadata, not a
requirement to expose unrequested measure numbers in the displayed answer.
The current completed judgment artifact passes all 41 selected cases across
the five pieces, with no recorded semantic-drift category.

To test wording robustness, run the hybrid RAG+LLM pipeline over three Korean
reformulations of all 79 active questions across the five pieces. In addition
to the normal source/range contexts, the benchmark asks 38 semantically
self-contained measure-scoped questions without a selected range:

```bash
conda run -n soprano-qa python \
  evaluation/run_synthesized_questions.py \
  --dataset-root ../soprano-qa-dataset \
  --output evaluation/synthesized_question_results.json
```

The completed result retains 237 unique variant formulations and contains 161
canonical formulation contexts, producing 483 inference cases. Of those, 114
are the no-range semantic cohort. No semantic LLM judge was run for the
synthesized artifact; retrieval metrics and generation modes do not establish
answer accuracy. The current run completed all 483 cases: 346 returned a
finalized expert answer directly and 137 deliberately returned unavailable.
The abstentions comprise 124 cases with insufficient authoritative evidence,
ten with missing range-scoped causal authority, and three with ambiguous dense
grounding. Overall Hit@1 is 454/483 (93.9959%), Hit@6 is 470/483 (97.3085%),
and MRR is 0.955487; all three variants hit within six results in 154/161
formulation contexts (95.6522%). The no-range cohort achieves Hit@1 in
109/114 cases (95.6140%),
Hit@3 in 114/114, and MRR 0.978070 across all five pieces. All required model
preflights passed. Of the 114 no-range cases, 104 return finalized expert
answers and ten abstain under the conservative answer-admissibility gate;
unavailable cases are not substituted with another answer path. Inspect the
human expected/reference answer beside each generated answer with the
read-only viewer:

```bash
conda run -n soprano-qa python evaluation/server.py
```

Open <http://127.0.0.1:8766/>. The viewer loads both complete artifacts,
groups the 164 original results and 483 synthesized results into 164 matching
inference scenarios, and keeps the two question families visibly separate.
All 647 generated outputs are available; the Question type filter can isolate
either family. See
[evaluation/README.md](evaluation/README.md) for the complete workflow and
[the synthesized evaluation report](evaluation/synthesized_question_comparison.md)
for current retrieval and generation-path metrics.

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

The reviewed `knowledge_units.answer` text is the authoritative expert answer.
Original annotator questions remain retrieval aliases, while ordinary raw
source answers, legacy range hints, and editorial notes remain only in the
dataset repository and are not serialized into `data/corpus.json`. The corpus
is generated reproducibly with `python scripts/build_corpus.py`; its strict
release schema rejects unexpected editorial fields instead of relying on the
answer prompt to ignore them. Confirmed knowledge-unit ranges may route a
merged multi-range unit, but they never substitute old source prose for its
curated answer. When finalized expert evidence directly answers a query, the
service returns each selected curated answer intact, with deterministic
formal-polite ending normalization, and does not let the answer model
paraphrase or merge
its claims. Structured output labels this intentional path as
`expert_verbatim` / `authoritative_expert`. A
semantically relevant record with a confirmed non-overlapping range is kept
as lower-ranked `other_range_context_only`: it remains inspectable and can
help detect a range mismatch, but it cannot ground, broaden, or receive an
automatic citation in the selected-range answer. Strongest range-applicable
alias matches remain primary, and up to two confirmed other-range alias
matches may follow as explicitly secondary context.

If range-overlapping dense-only candidates are too close to distinguish
safely, the service fails closed with `unavailable` / `no_corpus_evidence`
rather than choosing one because it happens to rank slightly higher. Free-form
synthesis is allowed only when the leading primary record is directly
supported, answer-eligible web evidence. That path makes exactly one local
answer-model call against the frozen evidence bundle. Backend errors, context
errors, and rejected drafts propagate as errors; raw web prose is never copied
into the answer.

Runtime retrieval accepts only the release corpus schema and rejects expert
records containing provisional review fields. Pending-review compatibility is
not part of the answer path.
An unseen code/UI-style CamelCase token or other unsupported output causes the
sole generated draft to be rejected.
Changes to the review files invalidate the corpus fingerprint, so the next
query rebuilds and reloads the local artifact automatically. No data copy into
the demo repository is required.

See
[docs/rag-llm-pipeline-method.md](docs/rag-llm-pipeline-method.md) for the
retrieval, measure-routing, prompt construction, and generation design.

To run the web application, continue with the
[demo setup](https://github.com/minigb/soprano-qa-demo#setup).
