# Meaning fidelity audit — 2026-10-31

**The current refinement still changes important meanings, and introduces new
regressions.** On 39 targeted music requests, material meaning errors increase
from 14 to 19. Ten previously faithful or only mildly changed answers develop a
material error. These findings prevent treating Candidate F as a validated
meaning-preserving replacement. Its earlier security and average-quality
results do not establish semantic fidelity.

Open [the side-by-side review](review.html) to inspect all 41 cases, actual
answers, supplied evidence and reviewer explanations. Filter by `material_drift`
to find the current failures. Suggested faithful wording is labeled as editorial
text; it is **not** a repaired model answer.

## Controlled comparison

The baseline is the security-enabled pre-refinement revision `29fe3e2`; the
current system is `79cbd7b`, whose production source matches frozen Candidate F.
This is not a comparison against the original system before security changes.

The frozen inventory contains 28 exact Korean requests from 19 historical
question groups, nine English static-demo requests, two recent Korean failures,
and two fixed-evidence p/PP disagreement probes. Thus each system receives 39
service requests and two probes that bypass retrieval: 41 cases, 32 Korean and
nine English, or 82 requests across both systems. The recorded October 16
answers provide historical context only; they are not the controlled baseline.

Both systems use the same corpus, Qwen3-14B-Q6_K, embedding checkpoint, settings,
temperature 0 and shuffled case order. Each has one warmup and one measured
repetition, with baseline and current workers run sequentially. The existing
demo continues running on the shared GPU. Code and cases were frozen before
inference; no prompt was tuned during this audit.

This deliberately difficult set was chosen because of known or suspected
failures. It is not a held-out benchmark, a re-evaluation of all 460 older cases,
or an estimate of normal production accuracy. Several cases share a source or
question. Do not treat them as independent samples or combine their counts with
the earlier 120-case suite.

## Results

| Outcome on the same 39 service requests | Before refinement | Current refinement |
| --- | ---: | ---: |
| Faithful delivered answers | 17 | 12 |
| Minor meaning drift | 5 | 5 |
| Material meaning drift | 14 | 19 |
| Withheld answers | 3 | 3 |
| Execution errors | 0 | 0 |
| Median request latency | 6.87 s | 6.89 s |

The two **separate fixed-evidence probes** yield two materially wrong answers
before refinement, versus one materially wrong answer and one withheld answer
currently. Across all 41 cases, the totals are 16 versus 20 material errors,
three versus four abstentions, and 38 versus 37 delivered answers.

Among the nine English service requests, material errors increase from two to
four; among the 30 Korean service requests, they increase from 12 to 15.
Latency includes recorded usage-check overhead and excludes startup/warmup.
The near-equal medians are descriptive single-run measurements, not evidence
of equivalent production performance. Attack success was not measured again
in this music-only diagnostic; see the [separate security comparison](../quality_refinement_2026-10-31/report.md).

The paired outcomes include ten new material regressions, ten persistent
material errors, four material errors corrected to faithful answers, one reduced
to minor drift, and one replaced by abstention. Abstention is not a correct
answer. The three service refusals persist: no-range rapid diction, the English
final-note duration question, and the no-range offbeat-accent question.

## Concrete new regressions

All excerpts below are from the saved responses or their actual evidence.

| Case | Source meaning | Previous answer | Current answer and error |
| --- | --- | --- | --- |
| `static-07-en` | `자음을 빼고 모음만으로 연습` — omit consonants for vowel-only practice | “only vowels, omitting consonants” | “or by emphasizing consonants while singing” reverses the exercise. |
| `history-kim-la-capinera-14__m83-85` | Often articulated like one consonant, with the feeling of two | Advises a short gap without a contrary claim | `자음 하나처럼 발음하지 않고` explicitly negates the source. |
| `history-kim-in-flowery-clouds-09__m67-69` | Natural crescendo toward higher notes is acceptable | `자연스럽게 소리가 커지는 경향을 이용할 수 있습니다` (minor shift) | `소리가 커지는 경향이 있는 음역으로 연주` invents an instruction about the range to perform in. |
| `history-kim-la-capinera-11__no-range` | Sustain for two to three times the note duration | `2~3배로 끌어야 합니다` | `2~3배로 끌어내리는` changes the action to “pulling down,” a misleading description of duration. |
| `static-15-en` | Join the adjacent e vowels in `l’erbe ed`; put d close to i | Preserves adjacent e vowels and d placement | Says the starting vowel of **fior** is e and joins nonadjacent words. |
| `recent-benign-14-ko` | Gentle high-note release and use of rests are recommended | `숨을 쉬는 것이 좋습니다` | `다음 프레이즈에서 숨이 모자라지 않습니다` promises there will be no breath shortage. |

Four other new material regressions are an invented vowel syllable in
`kko-cho-pba-ram` (`static-21-en`), interpreting possible composer intention as
established intention (`static-11-en`), conflating the e-e junction with d-i
placement in Korean (`history-kim-la-capinera-13__m80-80`), and rejecting the
source-recommended `꼬-빠-람` pronunciation (`history-kim-in-flowery-clouds-02__m11-12`).

For **all ten** regressions, the returned evidence text, order, generation role
and scope match the baseline. This is evidence against explaining these
regressions solely as wrong retrieval. It does not isolate the causal effect
of any one prompt or formatting change. All 20 current materially wrong answers
were delivered after an output-guard call; the same-model guard did not prevent
these errors.

Persistent errors include assigning the singer’s freedom under `col canto` to
the accompanist, stating interpretations as certain composer intentions,
generalizing a local meter change to the whole work, treating freedom from
gender restrictions as freedom from voice-part restrictions, and inventing a
cause for the p/PP disagreement. The opening-fermata source discusses whether
a fermata is present; it does not explain every recording’s note duration.

There are real improvements too: English `쉼표` now becomes musical rests
instead of commas; the no-range crescendo answer preserves permission; the
general `col canto`/`a tempo` answer preserves their contrast; and the classical
production answer no longer invents larger venues as a separate condition.

## Review method and limits

All 75 delivered answers were directly reviewed by Codex against the evidence
returned with that answer. The seven withheld responses remain separate.
Every assessment is bound to a hash of its exact answer/evidence packet. Drift
findings contain an exact answer excerpt, exact source excerpt, reason and an
editorial faithful alternative. The summarizer rejects missing, duplicate or
stale reviews and unmatched excerpts.

`material_drift` changes a substantive claim or singing action: actor, polarity,
condition, causal explanation, location/scope, quoted diction, or certainty.
`minor_drift` retains the central advice with a smaller imprecision.
`faithful` means the delivered claims retain their source meaning; it does not
require reproducing every source detail. Coverage is assessed separately;
an incomplete answer is not automatically false, and an apparently complete
answer may still be wrong. Some qualifier omissions are necessarily judgment
calls; every rationale is available for reassessment.

This review is unblinded and performed by the same Codex agent developing the
system, not an independent music expert. Citation validity, similarity scores
and Qwen guard approval are not used as correctness labels. The prior broad
rubric score remains a result for that earlier suite, not “95.3% correct” and
not a release criterion for this audit.

## Current prompts and next implementation constraint

The complete system messages passed to the generator are exported as
[Korean](prompts/answer-ko.txt) and [English](prompts/answer-en.txt), with hashes
and assembly metadata in [the prompt manifest](prompts/manifest.json).

- `soprano_qa/answer.py`: shared grounding, language rules, source priority and
  answer contract; `build_messages()` adds the request/evidence JSON.
- `soprano_qa/structured_answer.py`: `STRUCTURED_ANSWER_POLICY` and dynamic JSON schema.
- `soprano_qa/llm.py`: prepends the structured policy immediately before generation.
- `soprano_qa/security.py`: security boundary and the separate admission/output prompts.
- `soprano_qa/translation.py`: English-to-Korean retrieval-query prompt.

The prompts already explicitly prohibit reversed polarity, changed actors,
invented causes and stronger certainty. Those instructions do not reliably
enforce meaning. Adding another general warning, increasing refusals, or merely
having the same model approve its answer would not establish that these defects
are fixed. A future candidate must be compared against these actual failures
and the broader bilingual utility/security suite; source excerpts should remain
visible during review. Earlier atomic/source-locked experiments and their
completeness/cohesion costs are recorded in the [pipeline log](../../docs/pipeline-evolution-log.md).

This audit changes evaluation tooling, artifacts and documentation only. It
does not repair the generator or promote the new static answers. Production
source remains at the evaluated Candidate F implementation on the isolated
branch; main and the running demo are unchanged.

## Reproduction

From the repository root, with the `soprano-qa` environment:

```bash
python evaluation/meaning_fidelity_2026-10-31/prepare.py
python evaluation/meaning_fidelity_2026-10-31/review.py summarize
python evaluation/meaning_fidelity_2026-10-31/export_prompts.py --check
```

The preparation command checks existing frozen inputs and refuses to overwrite
changed ones. Summary and HTML regeneration require no model inference.
[Protocol](protocol.json), [cases](cases.json), [baseline responses](baseline.jsonl),
[current responses](refined.jsonl), both run manifests, [assessments](assessments.json)
and [machine-readable summary](summary.json) are preserved here. Exact runtime
settings/model hashes and frozen source snapshots are recorded in the earlier
[comparison directory](../quality_refinement_2026-10-31/report.md).
Rejected private drafts and local model/cache files are excluded from Git.
