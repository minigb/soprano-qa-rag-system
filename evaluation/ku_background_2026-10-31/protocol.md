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
the same retrieval corpus, checkpoints and shared inference settings. Candidate
algorithm flags and any additional resources are recorded separately below.
The candidate uses typed
source, background and score-lyric parts. Background cannot borrow source
citations, invent local score facts, reverse a source claim, or attribute new
advice to an expert. The application labels general explanations for readers.
Missing relevant KUs may permit a general answer; they do not establish local
facts. This is a generation change, not a retrieval or checkpoint change.

Also include the current main code at `29fe3e2` as a separately named baseline
under the same evaluation model, corpus and settings. It already has the initial
security implementation and Sumi Jo source ranking; Candidate F is the subsequent
strict refinement. The genuinely pre-security version was `f307470`, evaluated
in the separate October 25 report. Include a fresh run of it on this same
290-request suite as a fourth system, so the original comparison is available
without pooling older measurements. Do not call either current comparison
baseline unprotected, pool earlier experiments, or reuse historical latencies.

The final-comparison candidate E19 (`93f3552`) additionally uses complete Korean
source quotation and 129 reviewed English source translations. A translation is
selected only by exact original-text hash; new/changed sources use guarded model
translation with a separate 1536-token limit. The translations are machine
authored and separately model reviewed, not human-expert English references.
This offline source preparation is an extra resource excluded from request
latency. The four-way comparison evaluates the complete engineering package;
it does not isolate a security prompt's effect under equal source-preparation
cost. All versions use the original, unchanged Korean corpus and expert targets.

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

The user clarified on October 31 and again on November 1 that 0.90 is a
sufficient floor, not a maximization target. High literal overlap cannot justify
irrelevant quotations, unnatural or repetitive prose, changed meanings, excessive
refusals, or an unexamined latency cost. Overall quality determines acceptance.

Split by original expert source question, stratified by piece, using seed
20261031. All measure variants of a source stay together. Development chooses
generation and metric settings; the validation split is evaluated after those
choices are frozen. Earlier historical outputs are known to the developer, so
this is a development benchmark, not an independently held-out distribution.

After the failed E11 validation, E12 and later refinements used those observed
failures. All cases, references and metric settings remain frozen, but the later
validation is reused internal evaluation. A new independent expert test set is
required for an unbiased post-refinement generalization estimate.

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
