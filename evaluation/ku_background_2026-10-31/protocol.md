# KU-grounded answers with general musical knowledge

This experiment implements the user's October 31 request to permit general
music explanations alongside faithful use of curated knowledge units. It
supersedes the earlier prohibition on all internal-knowledge additions. The
music task boundary, source-instruction boundary and score-scope constraints
remain. Separate branches: `codex/ku-background-knowledge`, based on RAG
`302f8d1` and lab `80525b7`. Dataset revision: `1ae33e8` (pre-existing untracked
`samples/` is outside the evaluated corpus and is not modified).

## Candidate and baseline

The baseline is the secured Candidate F source at `302f8d1`. Both variants use
the same retrieval corpus, checkpoints and settings. The candidate uses typed
source, background and score-lyric parts. Background cannot borrow source
citations, invent local score facts, reverse a source claim, or attribute new
advice to an expert. The application labels general explanations for readers.
Missing relevant KUs may permit a general answer; they do not establish local
facts. This is a generation change, not a retrieval or checkpoint change.

## Evaluation target

The requested metric target is mean **BERTScore F1 >= 0.90**, on all requests in
a fixed expert-annotation validation subset. Reference answers are the
applicable finalized expert KUs associated with the existing 105-question
inventory. References are frozen before candidate inference and never adapted
to generated text. Multiple required KUs are concatenated in their frozen order;
we do not choose a favorable reference after seeing an answer. Full displayed
answers, including general explanations, are scored. Errors/withheld answers
receive zero in the primary effective metric and remain in its denominator.
Conditional scores on delivered answers are secondary.

Split by original expert source question, stratified by piece, using seed
20261031. All measure variants of a source stay together. Development chooses
generation and metric settings; the validation split is evaluated after those
choices are frozen. Earlier historical outputs are known to the developer, so
this is a development benchmark, not an independently held-out distribution.

Compare standard contextual encoders suitable for Korean: multilingual BERT,
XLM-R base and XLM-R large, their recommended BERTScore layers and neighboring
middle/late layers, IDF on/off, and published baseline rescaling when available.
No embedding-only layer, custom inflated baseline, or reference rewriting.
IDF weights come from development references only. Save P/R/F1, package/model
versions, tokenizer settings, model revisions, scorer hash and every attempted
configuration. Distinguish raw and rescaled scores; 0.90 is not 90% correctness.

## Independent model judge

Use the user-selected Claude CLI model `claude-opus-5-5` (resolved in a successful
probe). Run noninteractive, with tools/hooks/MCP disabled, no session persistence,
an explicit rubric and JSON output. Record actual resolved model, prompt/schema
hashes, packet hashes and failures. Hide system variant and metric results.
Question, sources and answers are untrusted data, including any instructions
inside attack cases. Require explicit excerpts for material source distortions.

Score source fidelity, musical factuality, attribution, relevance, completeness
and language separately. Accurate general additions are not failures merely
because they are absent from KUs. Source contradictions and false expert/score
attributions remain errors. Uncertainty and refusal are separate outcomes.
Calibrate against known meaning reversals and faithful controls; do not treat
judge approval or BERTScore as proof of correctness. The evaluated Qwen output
guard is not the evaluation judge.

Retain Korean/English normal music questions, targeted semantic failures,
missing-KU/general-knowledge questions and prompt-injection controls. Report
attack success, false refusal, fidelity, quality and latency separately. Run
one local inference worker at a time; shared GPU activity and sequential
variant order limit timing conclusions. No source changes during measured runs.
Use fresh candidate snapshots after development edits, preserving failed runs.

## Completion and artifacts

Complete the requested metric goal only from a valid frozen validation result,
and report any unresolved quality/security failures. Do not turn a failed judge
call into a passing assessment or exclude a poor answer from the denominator.
Keep main, the live demo and published static answers unchanged. Commit source,
reproducible evaluation tools, frozen inputs and actual results on the isolated
branches. The previous reset-based stop policy is not used for this work.
