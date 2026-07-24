# Soprano QA Local RAG System

Standalone, local RAG system for five soprano works. A query always identifies
the piece and supplies a question. Add an inclusive measure range for a local
question; omit the range when the question concerns the whole song.

See [Measure-aware RAG pipeline](docs/measure-aware-rag.md) for the complete
request, corpus, range-routing, database-evidence, ranking, and generation flow.

The system combines two separately provenanced evidence lineages from:

```text
../soprano-qa-dataset-database-collect
```

The default is resolved relative to this repository through
`config/settings.json`, so it does not depend on the shell's working
directory.

- 110 consolidated human-expert annotation units, including 71 with one or
  more exact measure ranges and 39 with no specific measure.
- 81 `research_open` and 20 `research_conditional` web-database chunks for
  sourced background and general performance context.

The web release currently has no edition-qualified measure chunks. Therefore,
only overlapping human annotations can ground a measure-specific claim. Web
evidence may appear as clearly labeled global context, but it is never treated
as evidence about a requested bar.

`catalog-only.jsonl` is deliberately excluded from answer generation, and the
empty `demo-local.jsonl` partition is not ingested. Expert `source_ids` remain
separate from web `web_source_ids` and `claim_ids`. Conditional licenses,
governing jurisdictions, terms URLs, permissions, attributions, and source URLs
are retained in the generated corpus, model context, and deterministic evidence
notices.

## Setup

Use the existing conda environment:

```bash
conda run -n soprano-qa python -m pip install -r requirements.txt \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

The extra index gives `llama-cpp-python` access to CUDA wheels. If installation
falls back to CPU, generation still works more slowly. Corpus building,
retrieval-only queries, and the tests use only the Python standard library.

## Local model

The default checkpoint is:

```text
Qwen/Qwen3-8B-GGUF / Qwen3-8B-Q4_K_M.gguf
```

Download it if `models/Qwen3-8B-Q4_K_M.gguf` is not already present:

```bash
conda run -n soprano-qa python scripts/download_model.py
```

## Build the combined corpus

```bash
conda run -n soprano-qa python scripts/build_corpus.py
```

This reads the curated expert files and answer-eligible web export partitions,
then writes:

```text
data/corpus.json
data/corpus_stats.json
```

The default generated snapshot contains 211 records, of which 210 are
retrievable. One expert source item whose text is only a request for web
research remains present for lineage and stable IDs but is explicitly marked
non-retrievable.

To use another compatible dataset checkout, set
`SOPRANO_QA_RAG_DATASET_ROOT` or edit `config/settings.json`. Corpus statistics
store the resolved dataset root and a SHA-256 fingerprint of both expert
consolidation/spellcheck phases, every canonical web source/claim/chunk,
facts-only release-audit, selected web-export, and override file. The builder
also verifies that spellcheck preserved expert source grouping, order, topic,
measure scope, and question provenance. Web ingestion recomputes effective
asset rights from status, permissions, and jurisdiction, then verifies claim/source/asset
lineage and license manifests. Statistics also store the exact generated
corpus SHA-256. A query automatically rebuilds when either identity changes, so
an override cannot
silently reuse a corpus from another checkout and a stale or corrupt corpus
cannot be blessed by current stats. Corpus and stats files are individually
published with atomic replacements.

Relative `SOPRANO_QA_RAG_DATASET_ROOT` and `SOPRANO_QA_MODEL_PATH` overrides
are resolved from this repository, not from the shell's working directory.

## Ask a measure-specific question

Retrieval only:

```bash
conda run -n soprano-qa python scripts/ask.py \
  --piece die-forelle \
  --measures 28-30 \
  --question "28마디부터 분위기 변화를 어떻게 표현해야 하나요?" \
  --no-generate
```

Full local generation:

```bash
conda run -n soprano-qa python scripts/ask.py \
  --piece die-forelle \
  --measures 28-30 \
  --question "28마디부터 분위기 변화를 어떻게 표현해야 하나요?"
```

Ranges are inclusive. Single and disjoint ranges are accepted, for example
`--measures 12` and `--measures "12, 35-38"`. Hyphens and en dashes are both
accepted. Measures must be positive and ranges must be ascending.
If the question itself identifies a bar location, it must also be covered by
`--measures`; contradictory ranges are rejected before retrieval. Likewise, an
explicitly locative form such as `28마디의`, `제28마디`, `첫 마디`,
`28마디부터 30까지`, `28, 30마디`, `bar 28`, or `bars 28-30` requires
`--measures`, because omitting the flag declares a whole-song question. Opening
counts such as `첫 4마디` mean measures 1-4. Relative endings such as
`마지막 마디` require the caller to resolve them with an explicit numeric
range. Cardinal descriptions such as `4마디 프레이즈` and total-length
questions such as `총 67마디인가요?` remain valid whole-song questions.

For a ranged query, non-overlapping local records are excluded. Overlapping
expert evidence ranks before general context, even when a general chunk has a
larger lexical score.

## Ask a whole-song question

Do not pass `--measures`:

```bash
conda run -n soprano-qa python scripts/ask.py \
  --piece die-forelle \
  --question "이 작품의 정식 표제와 작품 번호는 무엇인가요?"
```

With no range, general expert and web evidence ranks before isolated measure
examples. A local example may still be retrieved when it is textually useful;
the answer prompt requires its measures to be named and prohibits generalizing
it to the entire song.

Available piece slugs are:

```text
una-voce-poco-fa
la-capinera
die-forelle
nella-fantasia
in-flowery-clouds
```

Add `--json` for machine-readable query, ranking, scope-relation, provenance,
rights notices, and answer fields. In normal output, an `Evidence notices`
section deterministically prints each retrieved web chunk's usage class,
generated-text license, inherited license and attribution conditions, and
source URLs, including territorial limits and governing terms links. This does
not rely on the language model remembering license terms. Project-authored
Korean summaries also receive a deterministic Soprano QA project credit and CC
BY 4.0 link. If a requested retrieval depth exceeds the model context, the
generation path retries with fewer whole ranked records so citations, notices,
and the actual model context stay aligned. Add
`--rebuild-corpus` to force a rebuild before a query.

## Retrieval behavior

The dependency-free index uses BM25 over Korean-aware word and character
n-grams. Piece selection is a hard filter rather than part of the lexical
score. Controlled web topics receive Korean search labels so questions about
identity, creators, editions, form, text, performance, recordings, or rights
can retrieve the externally collected evidence even though its schema topics
are English identifiers. Character n-grams improve ranking within a word, but
eligibility requires each full normalized query concept; n-gram fragments never
satisfy the relevance gate. For compound questions, the selector finds a
bounded, minimal set of chunks whose union covers every concept while preserving
overlapping-measure or whole-song scope priority. This permits complementary
evidence such as one pronunciation chunk plus one form chunk without allowing a
supported word to mask an unsupported subject. Source titles and institution
names may influence ranking only after semantic topic/question/answer text
qualifies a record; they cannot establish relevance by themselves.

If no evidence has meaningful textual overlap, retrieval returns no chunks.
Generation then uses a separate, explicitly ungrounded internal-knowledge
prompt. The response contains no corpus citations or evidence notices and is
identified by `answer_basis: internal_knowledge`. A grounded model can also
signal that loose lexical matches do not answer the core question, which
routes through the same fallback.

For supported questions, the model cites short labels such as `[E1]`; the
application validates and deterministically maps them to exact corpus IDs such
as `[sqa-0058]` and `[webchunk-cecff2bace03ab67e32d]`, repairs an unambiguous
mistyped opaque ID, and removes unknown citations. If the local model omits
labels entirely, the answer is preserved and receives a deterministic
`제공된 검색 근거` footer listing the exact records supplied as context.

## Service API

Applications in sibling repositories can use the reusable service facade:

```python
from soprano_qa.service import ask

result = ask(
    piece_id="die-forelle",
    question="피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?",
    measure_range=(2, 5),
    generate=True,
)
```

The service owns corpus refresh, retrieval, local-model serialization,
citation validation, internal-knowledge fallback, and answer provenance. A web
application should adapt this result rather than copy the pipeline
implementation.

## Recurring features

For recurring musical features, do not widen `measure_range` to the whole
piece. Add explicit occurrence ranges to:

```text
config/feature_overrides.json
```

Use `config/feature_overrides.example.json` as the template, then rebuild the
corpus. Overrides apply only to expert records; they cannot turn global web
evidence into measure evidence. `match_source_ids` must be a non-empty unique
list and must resolve to exactly one corpus record; use `id` when a source unit
was split across records. Unknown fields and selector-only no-op overrides are
rejected. Explicit scopes must agree with their ranges: `global` has no range,
`local` has one, `multi_range` has more than one, and `recurring` has at least
one explicit valid range.

## Validation

```bash
python3 -m unittest discover -v
python3 -m py_compile soprano_qa/*.py scripts/*.py tests/*.py
```

The integration tests rebuild into a temporary directory and verify corpus
counts, all 135 expert source links, phase-2/phase-3 structural lineage,
same-piece/topic lineage, rights partition filtering and territorial notices,
input/output fingerprints, corrupt-corpus recovery, strict integer and natural
language range contracts, canonical web claim/source/asset lineage, placeholder
exclusion, inclusive and disjoint measure matching, compound evidence coverage,
whole-song routing, Korean inflection handling, citation sanitization,
internal-knowledge fallback, current model-download compatibility, and safe
retrieval rejection for unsupported or out-of-domain questions.
