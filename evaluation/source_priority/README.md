# Sumi Jo source-priority development pilot

Run `python -m evaluation.source_priority.run` from the repository root; add
`--hybrid` in the `soprano-qa` environment to use the configured local embedding
model. No answer-generation model is called. Questions and expected IDs are
read only by the evaluation runner, never by indexing or retrieval.

The 14 manually authored development cases comprise ten Sumi Jo answer targets,
two unrelated questions, one wrong-piece query, and one unreviewed-range query.
They are a small engineering pilot, not the curated expert benchmark or an
independent held-out test. The two unrelated controls evaluate whether Sumi Jo
is spuriously retrieved, not whether the entire existing system abstains.

| Retrieval mode | Intended Sumi Jo unit first | In top 6 | Sumi Jo false positives on controls |
| --- | ---: | ---: | ---: |
| Lexical | 4/10 | 4/10 | 0/4 |
| Hybrid, answer-only semantic fallback | 8/10 | 8/10 | 0/4 |

The expert/web-only baseline has 0/10 Sumi Jo recall by construction. Its saved
answers-to-retrieval mapping helps inspect fallback evidence; that difference
must not be presented as a causal estimate of ranking quality. The hybrid
thresholds were inspected on this development set, so the improvement needs
confirmation on independently authored questions.

The remaining hybrid misses concern large-hall singing (similarity below the
0.60 threshold) and rubato (near-tied semantic answers). Keep these misses
visible instead of injecting evaluation formulations as retrieval aliases.

`generation-smoke.json` records one separate end-to-end local Qwen3 generation
for the microphone/timbre question. It completed with `generation_mode: llm`
and `answer_basis: retrieved_evidence`, using Sumi Jo first, then expert units,
then web coaching. This verifies integration and accepted output formatting;
it is not a blinded answer-quality or conflict-resolution evaluation.

Next research steps: independently label relevance across all three tiers;
compare authority ordering against a common relevance reranker using the same
candidate pool; evaluate source-conflict handling and citation fidelity with
blinded answer review; and repeat range tests after Sumi Jo's locations receive
score review. The current tests verify ordering, provenance, filtering,
confidence/margin gates, fallback, and structured prompt attribution.
