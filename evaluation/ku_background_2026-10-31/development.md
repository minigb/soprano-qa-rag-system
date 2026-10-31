# Development record

All choices below preceded the final validation comparison. The development
cohort has 43 expert-reference requests, ten answerable general-music requests,
and two questions about unavailable local facts. Refusals remain in the expert
BERTScore denominator with a score of zero. Complete answers are scored.

The 18 encoder/layer/IDF combinations in `bertscore-development-b/` were tried
once on development B. XLM-R large, layer 18, no IDF, raw F1 and the slow tokenizer
were selected there and kept for every later run. No validation output was used
to choose the metric. The raw scales of different encoders are not calibrated
accuracy percentages.

| Run | Expert answers / 43 | Effective expert F1 | General answers / 10 | Change or finding |
| --- | ---: | ---: | ---: | --- |
| B | 38 | 0.838086 | 4 | Initial typed source/background answers; too many refusals. |
| C | 37 | 0.823354 | 3 | Separating task admission alone did not fix the refusals. |
| D | 41 | 0.909693 | 6 | Fixed valid-output rejections, bounded output-review retry and scoped source wording. |
| E | 41 | 0.912229 | 6 | Select passages that answer the question before composition. |
| E2 | 41 | 0.912229 | 6 | Correct note-value versus pitch-interval query translation. |
| E3 | 41 | 0.912330 | 9 | Require a general explanation or explicit limitation when no primary passage is selected. More answers exposed general factual errors. |
| E5 | 41 | 0.913376 | 10 | Checked bilingual terminology and a dedicated general explanation prompt. Remaining unrelated-source attribution and conditional-scope issues. |
| E7 | 40 | 0.889482 | 9 | Joint intent/label selection regressed: an invalid mixed result refused a diction question; retrieved context still contaminated a general fermata answer. Rejected. |
| E8 | 42 | 0.905950 | 10 | Question-only routing passed the numeric target but discarded relevant sources for short score/performance questions. Wrong grace-note and small-note explanations were found by inspecting answers. Rejected before validation. |
| E9 | 42 | 0.934070 | 8 | Removed question-only routing; the three confirmed notation/definition errors returned to source-grounded explanations. Two general questions remain withheld. Independent review is saved in `judge-e9/`. |

A was an earlier 24-request pilot, not the full development cohort. E4 compared
generation modes in isolated diagnostics, not a complete service run; E6's scope
experiment was incorporated into E7 without a separate measured cohort. Their
outputs are not substituted into any later run. See
`diagnostics/general-generation/README.md`. Thinking/sampling did not reliably
fix the observed terminology errors and were not adopted.

E8 separates a question-only general/evidence routing decision from passage
selection, limits general explanations to one paragraph, and corrects a false
whole-piece generalization detection on an explicitly conditional phrase. The
scope fix uses the actual previously rejected answer as a regression case;
semantic review is still required and no source text is rewritten. E8 source
is frozen at `654a86d68176bef7b22fb3d4b3f9307e9e68b59e` in
`provenance/e8-final/`; the exact policies are in `prompts-e8-final/`.

Despite passing the development metric, E8 was rejected on meaning and factual
quality. Its prepared `comparison-e8-not-run.config.json` was never executed.
E9 removes the destructive question-only routing pass, retaining actual-passage
relevance selection, concise general explanations and the conditional-scope
fix. It does not special-case evaluation questions or copy reference answers.

Independent Claude judgments vary. Earlier B/C used the first rubric; D onward
include actual lyric context in the second rubric. Development judgments were
used to diagnose problems, not pooled as independent trials. In particular,
`source_fidelity=not_applicable` on a general answer does **not** mean it is
factually correct. The final summarizer reports substantial factual errors
separately, and the final comparison judges deduplicated packets jointly.

The diagnostic `bertscore-fidelity-controls.json` shows that three of four
previously annotated material meaning errors still exceed raw F1 0.90. These
known historical controls are outside the target cohort and were not used to
retune the metric. BERTScore cannot establish preservation of negation,
conditions, attribution or practical meaning by itself.

## Snapshot metadata caveat

The reusable snapshot preparer retains an older `protocol` object describing
two repetitions and paired order in each `provenance.json`. Those fields do not
describe this experiment. The actual protocol is `protocol.md`, the final
comparison configuration, and each run's manifest: one measured repetition per
system, sequential systems, one untimed warm-up, identical frozen inputs and
settings, and one GPU generation worker. Source/model/corpus hashes in the
snapshot provenance remain applicable. The runtime directory called
`presecurity` contains **main `29fe3e2` with initial security**; the actual
pre-security baseline is `f307470` in `baseline-pre-security`.
