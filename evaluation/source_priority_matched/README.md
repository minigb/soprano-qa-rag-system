# Matched-candidate source-priority evaluation — 2026-10-25

**This evaluation does not establish an ordinary-question answer-quality gain
from source-priority ranking.** It helps filter the controlled low-authority
contradictions out of a small context budget, but does not resolve conflicts
when both sides remain visible. A separate real-source check exposes unsupported
explanations of a genuine p/PP discrepancy under both ranking orders.

## Controlled comparison

The primary run has 20 cases and 52 condition outputs: 11 natural answerable
questions across five works, three negative/scope controls, and six synthetic
conflicts tested at two budgets. Both conditions receive the **identical frozen
candidate pool and cosine scores**. Only ranking changes:

- Relevance: descending answer-only cosine, then ID for ties.
- Priority: Sumi Jo → expert → web, preserving relevance order within a tier.

Natural pools use a common, source-independent cosine threshold of 0.43 and
at most six same-piece candidates. They are truncated to three context records.
Conflicts use three fixed records, tested with two records and all three.
There is no threshold tuning after observing these results. This is a comparison
conditional on shared candidates, **not a full production-retriever comparison**.

The generation model, 768-token limit, temperature zero, production prompt,
structured citations, security guards, and output validation are identical.
Both prompts retain the same authority/conflict policy; the baseline removes
authority **ranking**, not source metadata or generation instructions. The
schedule is randomized. Six identical prompts are reused, giving 46 unique
primary prompts; cached outputs are not independent replications.

See [the pre-inference protocol](PROTOCOL.md), [frozen pools](frozen.json),
[runtime/model hashes](runtime_manifest.json), and [raw outputs](responses.jsonl).
All corpus text is unchanged. Fabricated counterclaims exist only in these
evaluation artifacts and are never indexed in the runtime corpus.

## Primary results

“Fully satisfactory” requires grounding, relevance, and completeness all to
score 2/2. The 0–6 sum weights those dimensions equally. Rejected answers count
as abstentions: they can be safe without answering an answerable question.

| Stratum | Measure | Relevance | Priority |
| --- | --- | ---: | ---: |
| Natural questions, n=11 | Accepted generated answer | 9/11 | 7/11 |
| Natural questions | Fully satisfactory | 3/11 | 4/11 |
| Natural questions | Mean quality /6 | 4.09 | 3.91 |
| Natural questions | Reference unit retained | 10/11 | 10/11 |
| Negative/scope controls, n=3 | Appropriate abstention | 3/3 | 3/3 |
| Synthetic conflicts, top 2, n=6 | Accepted generated answer | 0/6 | 5/6 |
| Synthetic conflicts, top 2 | Fully satisfactory | 0/6 | 4/6 |
| Synthetic conflicts, all 3, n=6 | Accepted generated answer | 0/6 | 0/6 |

Natural-question paired quality difference (priority minus relevance):
**−0.18/6**, with exploratory 95% cluster-bootstrap interval **[−1.00, +0.55]**.
The fully-satisfactory difference is +9.1 percentage points, interval
[−36.4, +45.5]. These results are too uncertain to support superiority.

For capped synthetic conflicts, the quality difference is +2.33/6, interval
[+1.33, +3.00]. The mechanism matters: priority excludes the synthetic
contradiction in every capped case. Relevance includes it in every case, and the
guarded pipeline abstains. With all three records visible, **both orders abstain
on all six cases**. No returned answer endorses a synthetic counterclaim, but
none explains an exposed synthetic conflict either. Filtering out conflict is
not evidence of successful conflict resolution.

The selected natural-question sets differ in 6/11 pairs; set or order differs
in 8/11. In the all-three conflict comparison, all six sets are identical and
only their order differs. There are no backend execution failures: 29 primary
outputs are validation/guard rejections (including reused outputs), comprising
19 admission rejections, four output-guard rejections, and six local-scope
overgeneralization rejections. Raw logs initially label the latter exception
`execution_error`; the analyzer explicitly classifies `GeneratedAnswerRejected`
as validation rejection and retains the original trace.

[Machine-readable summary and paired intervals](summary.json).

## Post-hoc checks

**Synthetic-label sensitivity:** twelve additional all-three outputs replace
the synthetic record's displayed provenance with neutral “Source C” labels.
Claims, candidates, scores, and other settings stay fixed. Both rankings still
abstain on all six cases. The explicit fabricated-source label therefore does
not, by itself, explain the main conflict result. These remain synthetic stress
tests; this does not prove how all natural disagreements will behave.
[Sensitivity results](neutral_marker_sensitivity/summary.json).

**Observed p/PP disagreement:** Sumi Jo's unchanged opening transcript says PP;
expert unit `nella-fantasia-ku-002` says p. We tested two new questions against
only those two original records, reversing their ranking with the same scores.
All four returned answers invent an explanation for the discrepancy, such as
interpretation differences, notation not reflecting intent, or volume balancing.
The sources establish none of those as the cause of their disagreement. Both
orders recognize the discrepancy, but fail to preserve uncertainty about its
cause. We did not decide which score marking is correct or alter either source.
These are two formulations of **one** source disagreement, so no uncertainty
interval is reported. [Observed-disagreement results](observed_disagreement/summary.json).

In total there are 68 condition outputs, including the post-hoc checks—not 68
independent questions or 68 successful answers.

## Answer review and limitations

Forty unique primary answer/support packets were reviewed with condition labels,
cosine scores, and evidence order hidden. Citations are canonicalized to source
IDs. The reviewer is **the coding assistant that authored the experiment**, not
an independent human or singer. The assistant had already seen the design and
candidate inventory; “condition-masked” must not be read as independent blinding.
Review rationales remain available for reassessment:
[rubric](RUBRIC.md), [review packets](review_packets.json), [ratings](reviews.json),
and [condition key](review_key.json).

The cases are small, manually authored development sets. The bootstrap resamples
source/target clusters (10,000 draws), not individual completions. Its intervals
describe this exploratory set and are not population guarantees. Zero-width
intervals for all-tied outcomes do not establish equivalence. No model or
threshold was tuned from the evaluation results.

Synthetic contradictions deliberately reuse question terms and receive high
relevance scores. They are stress tests, not a representative frequency sample
of web errors. The echo fixture also contains an unrelated, unchanged expert
unit about `vuò`; that distractor is kept visible in the frozen pool. Its
presence limits that fixture's interpretation, though accepted output did not
use its unrelated claim.
Excluding that fixture does not reverse the pattern: capped-context acceptance
is 4/5 for priority versus 0/5 for relevance, and all-three acceptance is 0/5
under either ordering.

Failures include wrongly citing the ending-note Sumi Jo unit for word splitting,
misstating an Italian apostrophe/elision example, losing source hedges, and
unhelpful refusals despite answer-bearing evidence. Some affect both conditions.
The reference-unit proxy cannot capture all of these errors; semantic review
is essential.

## Consequences

Keep authority as a provenance/selection preference, but do not claim that
unconditional source priority is a demonstrated quality improvement. The next
change should explicitly preserve source disagreements and unknown causes,
align answer and guard policies about source authority, and prevent a weakly
related high-authority passage from displacing a closer answer. The admission
guard currently has no explicit Sumi Jo → expert → web precedence rule; this is
a plausible interaction to investigate, not a proven causal explanation.
Validate such changes on independently authored questions with independent
musical review. This task changes evaluation artifacts only, not production
ranking, guards, or source text.

## Reproduction

Use the `soprano-qa` Conda environment and the existing local model files.

```bash
python -m evaluation.source_priority_matched.run prepare
python -m evaluation.source_priority_matched.run generate
python -m evaluation.source_priority_matched.analyze packets
# Independently review packets into reviews.json using RUBRIC.md, then:
python -m evaluation.source_priority_matched.analyze summarize
python -m unittest tests.test_source_priority_matched
```

Generation checkpoints every output and refuses to combine different contracts.
Do not rerun `prepare` over an existing run: archive/copy the directory first,
because preparation creates a new timestamped contract. Identical prompt
outputs are reused; failed outputs are not silently retried. The post-hoc
scripts are `sensitivity.py` and `observed_conflict.py`; their results have
separate contracts and summaries. Usage was checked between runs, and each
generator also checks every ten minutes and before the reported reset time.
No reset was observed and no notification service was installed.
