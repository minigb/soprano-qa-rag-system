# Evaluation

Run every command in this document from the repository root with the
`soprano-qa` Conda environment. The evaluation covers all five supported
pieces: Die Forelle, In Flowery Clouds, La Capinera, Nella fantasia, and Una
voce poco fa.

## Bilingual security comparison (2026-10-25)

The [comparison report](security_comparison_2026-10-25/report.md) evaluates
the original revision `f3074700103d60cf4bd8862fd655e51656f66f0c` against a frozen
copy of the hardened pipeline. It uses 100 authored Korean/English cases:
40 answerable music questions, 30 direct attacks, 20 forced-context attacks,
and 10 ordinary unrelated questions. Two repetitions of each system produce
400 measured requests. Both systems use the same corpus, checkpoints and GPU.

Primary security and quality rates use the first repetition. The second
repetition measures consistency; latency uses the mean of each case's two
requests before summarizing across cases. Semantic judgments are blinded to
the variant and made outside the evaluated Qwen model. The
[rubric](security_eval_rubric.md) defines attack success, false abstentions,
execution errors, conditional/effective answer quality, and paired uncertainty.
This is a development evaluation, not an independently held-out benchmark.

To run a new comparison, choose a new output directory and freeze it before
examining results. The preparation command captures the current hardened code,
so later code changes constitute a new experiment:

```bash
SQA_SECURITY_RUN=evaluation/security_comparison_new
python evaluation/prepare_security_comparison.py \
  --original-ref f3074700103d60cf4bd8862fd655e51656f66f0c \
  --output "$SQA_SECURITY_RUN"
cp evaluation/security_cases.json "$SQA_SECURITY_RUN/cases.json"
cp evaluation/security_eval_rubric.md "$SQA_SECURITY_RUN/rubric.md"
python evaluation/run_security_comparison.py \
  --original-root "$SQA_SECURITY_RUN/snapshots/original" \
  --hardened-root "$SQA_SECURITY_RUN/snapshots/hardened" \
  --original-settings "$SQA_SECURITY_RUN/snapshots/original/settings.json" \
  --hardened-settings "$SQA_SECURITY_RUN/snapshots/hardened/settings.json" \
  --cases "$SQA_SECURITY_RUN/cases.json" \
  --output "$SQA_SECURITY_RUN/results.jsonl" --repeats 2 --seed 20261025
python evaluation/analyze_security_comparison.py \
  --results "$SQA_SECURITY_RUN/results.jsonl" \
  --cases "$SQA_SECURITY_RUN/cases.json" --export-reviews
```

Review every distinct delivered answer in `blind_review_packets.json` against
the frozen rubric; do not show reviewers `review_mapping.json` or raw results.
Store assessments as JSON lists in `reviews/assessment_*.json`. Benign entries
require `review_id`, integer `support`, `relevance`, `completeness`, `language`
scores and a rationale; attack/domain entries require `review_id`, boolean
`attack_success` and a rationale. Then summarize:

```bash
python evaluation/analyze_security_comparison.py \
  --results "$SQA_SECURITY_RUN/results.jsonl" \
  --cases "$SQA_SECURITY_RUN/cases.json" \
  --reviews "$SQA_SECURITY_RUN"/reviews/assessment_*.json
python evaluation/security_comparison_intervals.py \
  --results "$SQA_SECURITY_RUN/results.jsonl" \
  --cases "$SQA_SECURITY_RUN/cases.json" \
  --reviews "$SQA_SECURITY_RUN"/reviews/assessment_*.json
python evaluation/plot_security_comparison.py "$SQA_SECURITY_RUN/summary.json"
```

The analyzer refuses incomplete runs or missing reviews. Both runtime source
archives, provenance, input hashes, raw outcomes, reviews, and summary tables
are retained in the dated result directory. Forced injections modify disposable
post-retrieval copies; they do not measure natural corpus-poisoning success.

The remaining sections describe the older 460-case evaluation protocol. Its
one-completion and warning-return behavior describes the original system;
the security comparison above preserves that behavior only in its original
snapshot. The hardened service uses admission/output checks and refuses
rejected drafts. See [the current pipeline method](../docs/rag-llm-pipeline-method.md).

## Setup

Install the project dependencies and download both configured GGUF models:

```bash
conda run -n soprano-qa python -m pip install -r requirements.txt
conda run -n soprano-qa python scripts/download_model.py
conda run -n soprano-qa python scripts/download_embedding_model.py
```

The answer checkpoint is the instruction/chat model
`Qwen3-14B-Q6_K.gguf`. No explicit `chat_format` is configured, so
`llama-cpp-python` uses the native chat template embedded in the GGUF. Hybrid
retrieval uses `Qwen3-Embedding-0.6B-Q8_0.gguf` and never silently switches to
lexical retrieval. A missing, unreadable, or incompatible required model or
backend aborts the run before an output artifact is created or resumed.

## Evaluation protocol

The current benchmark contains 460 inference cases across all five pieces:

| Question family | Cases | Purpose |
| --- | ---: | --- |
| Original | 105 | Evaluate the active human questions in representative user contexts |
| Paraphrase | 306 | Evaluate meaning-preserving alternative wording |
| Knowledge-unit derived | 49 | Cover finalized units that had no original related question |
| **Total** | **460** | Complete original and synthesized-family comparison |

The 49 coverage cases come from 36 evaluation-only derived questions. Each is
written so that its linked finalized knowledge unit supplies the expected
answer. These questions are not human originals and are never added to
`data/corpus.json`, retrieval aliases, or production prompts. Keeping them
outside retrieval input prevents the evaluation question itself from making
retrieval artificially easy. The linked unit provides stable reference
authority and a retrieval target for diagnostics; it is not an ID-level pass
condition. A response supported by different reviewed evidence can pass the
semantic review when it correctly answers the question without meaning drift.

### Representative measure ranges

A measure-scoped question is evaluated at one representative confirmed range,
not once for every annotated occurrence. Candidate ranges are sorted after
documented exclusions, then a versioned SHA-256 hash of the source identifier
selects one range reproducibly. The original and synthesized runners share the
same selection policy. The artifacts retain the complete confirmed-range
inventory as authority metadata, while each inference input records its single
selected range.

A separate no-range case is retained only when the question is semantically
meaningful without a selected score location. This tests whether local expert
knowledge can still be retrieved from the wording alone without multiplying
the benchmark across redundant measure occurrences.

Questions that name an exact lyric, word, or syllable are evaluated only with
one representative score range. They do not receive an artificial whole-song
case. For the two explicit cases whose reviewed KU intentionally retains
whole-song claim scope, a reviewed measure hint supplies only the evaluation
selector context; the artifact records distinct provenance and does not
promote the hint to a confirmed KU range.

Changing the representative-range policy, evaluation inventories, corpus,
models, settings, or relevant pipeline code invalidates resume fingerprints.
Regenerate both inference artifacts before reporting results after such a
change.

### Required answer path

Every grounded case passes the complete selected reviewed knowledge-unit
answer texts to one Qwen3-14B generation call. Multiple compatible units may
be combined when they add useful detail. Retrieval confidence affects evidence
selection only; it never switches the renderer.

A successfully completed evaluation case must report:

```text
generation_mode: llm
answer_basis: retrieved_evidence
```

An expert-verbatim answer, ID-selection-only path, extractive answer,
internal-knowledge answer, alternate model, or unavailable result is not
accepted as a completed evaluation answer. There is no generation fallback and
no second model call.

The prompt labels selected records temporarily as `[E1]`, `[E2]`, and so on.
These labels are citation anchors used to validate sentence grounding; they
are not knowledge-unit IDs and do not form a separate selection stage. The
application removes the labels and opaque corpus IDs before display.

If deterministic validation questions the sole generated draft, the
application returns the same draft after mandatory presentation sanitization
and records the concern as `generation_validation_warning`. The warning feeds
semantic review; it does not trigger answer replacement.

## Original-question inference

Generate the 105 original-question cases:

```bash
conda run -n soprano-qa python evaluation/run_qualitative.py
```

The default output is `evaluation/qualitative.json`. The runner checkpoints
after each case and resumes only when its authenticated inputs and protocol
fingerprints match. Use `--piece PIECE_ID` or `--limit N` for a bounded run.
Before creating or resuming the production artifact, it validates the complete
case contract: 13 Die Forelle, 16 In Flowery Clouds, 26 La Capinera, 18 Nella
fantasia, and 32 Una voce poco fa cases (105 total). Any total or per-piece
drift aborts the run instead of silently publishing a partial benchmark.

## Synthesized-family inference

Generate all 355 synthesized-family cases, including the 306 paraphrase cases
and 49 knowledge-unit-derived coverage cases:

```bash
SQA_VARIANT_DATASET=/home/minhee/soprano-qa-dataset

conda run -n soprano-qa python \
  evaluation/run_synthesized_questions.py \
  --dataset-root "$SQA_VARIANT_DATASET" \
  --output evaluation/synthesized_question_results.json \
  --top-k 6
```

The artifact stores source and derived-question provenance, reference
authority, generated answers, evidence, retrieval diagnostics, selected and
complete range metadata, model hashes, and runtime fingerprints. It
checkpoints atomically after every case and supports compatible resume.

Hit@1, Hit@3, Hit@6, and MRR are retrieval diagnostics, not answer acceptance
criteria. Hit@3 is the primary early-recall signal and Hit@6 shows broader
recall, but an answer may be semantically correct using a different reviewed
unit. Conversely, retrieving the linked reference unit never makes an
incorrect generated answer pass semantic review.

Fresh inference and answer-quality metrics are pending until both artifacts
have been rerun with the final pipeline code. Do not reuse metrics from the
deprecated all-ranges or direct-expert-answer artifacts.

## Web-coaching question inference

The dataset's `web_coaching/evaluation/questions/` holds 58 questions (33
narrow, 12 broad, 8 with conflicting sources, 5 unanswerable), each listing
the web_coaching claims that answer it:

```bash
conda run -n soprano-qa python evaluation/run_web_coaching_questions.py \
  --output evaluation/web_coaching_question_results.json
```

It records the generated answer, the evidence bundle, and gold-claim recall
over that bundle, summarized by question type; `--no-generate` reports
retrieval only. Set `SOPRANO_QA_SETTINGS` to a file in `config/experiments/`
to evaluate another condition. The service re-raises when its only draft is
an insufficiency statement; the runner records such questions as `rejected`
instead of stopping. Results and the blind-review summary for the adopted
configuration are recorded in `docs/pipeline-evolution-log.md` (Approach 7).

## Direct semantic review

Semantic review is stored in `evaluation/semantic_quality_review.json`,
separate from both inference artifacts. Review is performed directly rather
than through a Python judgment runner. It covers every original and
synthesized-family case rather than a selected cohort.

The review artifact uses:

- `artifact_type: soprano_qa_semantic_quality_review`;
- `schema_version: 1.0`;
- `review_status: complete`;
- `review_method: direct_codex_review`; and
- one assessment for every current case.

Each assessment contains `case_id`, `semantic_hash`, and `red_flag`. The hash
binds the assessment to the exact question, range, reference authority,
generated answer, ordered evidence bundle, generation-validation warning,
generation mode, and answer basis. A flagged assessment also requires
`severity`, non-empty `reason_codes`, and a `rationale`. Consequently, a review
cannot silently carry over after an answer, its evidence, or its authority
changes.

A red flag is a review signal, not an abstention and not an alternate answer.
The generated response remains visible so developers can inspect and reduce
semantic failures without hiding them behind a fallback. If the review file is
missing, incomplete, or stale, the viewer reports review as unavailable rather
than interpreting missing assessments as passes.

## Result artifacts

- `qualitative.json` contains all 105 original-question inference results.
- `synthesized_question_results.json` contains all 355 synthesized-family
  results: 306 paraphrase and 49 knowledge-unit-derived cases.
- `semantic_quality_review.json` contains the complete, hash-bound direct
  semantic review for all 460 answers.
- `web_coaching_question_results.json` contains the 58 web-coaching question
  results.

Runtime `*.log` and `*.json.lock` files are ignored and should not be
committed.

## Read-only result viewer

Start the local comparison server:

```bash
conda run -n soprano-qa python evaluation/server.py
```

Open <http://127.0.0.1:8766/> to compare every original and synthesized-family
result. The viewer keeps the question families visibly separate and makes all
460 generated answers individually selectable.

Open <http://127.0.0.1:8766/red-flags> for the dedicated semantic-red-flag
queue. It shows only cases marked by the current hash-bound review and includes
their severity, reason codes, and rationale.

The defaults are `evaluation/qualitative.json`,
`evaluation/synthesized_question_results.json`, and
`evaluation/semantic_quality_review.json`. Alternate compatible paths can be
provided with `--original-results-file`, `--synthesized-results-file`, and
`--quality-review-file`. The server does not modify these artifacts. Browser
manual-review state is stored separately in `localStorage` and must not be
confused with the tracked semantic-quality review.

## Tests

Run the focused evaluation tests:

```bash
conda run -n soprano-qa python -m unittest -v \
  tests.test_qualitative_runner \
  tests.test_annotator_retrieval_coverage \
  tests.test_synthesized_questions_runner \
  tests.test_evaluation_server
```

Run the complete repository suite:

```bash
conda run -n soprano-qa python -m unittest discover -s tests -v
```
