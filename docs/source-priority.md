# Source-prioritized retrieval

Relevant knowledge units are ordered **Sumi Jo → experts → crawled web**.
Both `web_coaching` and `web_database` belong to the web tier. Relevance and
measure eligibility are checked before authority ordering. An unrelated
high-authority passage cannot replace a directly answering lower-tier source.
Lower tiers supplement gaps; generation instructions preserve attribution and
disagreements rather than silently blending contradictory advice.

The default configuration enables `sumijo.enabled` and
`retrieval.source_priority`. The corpus contains seven `sumijo_feedback`
records, 122 expert units, 723 web-coaching units, and 101 web-database records.
Sumi Jo records carry their own `sumijo_source` object with source ID, speaker,
artifact paths, hashes, and attribution basis. Service evidence has
`kind: sumijo`; it is not labeled as web or expert annotation.

The existing expert/web retriever and its statistics remain separate from the
Sumi Jo index. Direct feedback uses the existing lexical/concept gate. In
hybrid mode it also embeds **answers only**, using the existing local model.
A semantic fallback requires cosine similarity at least 0.60, a lead of at
least 0.04 over the next Sumi Jo answer, and the existing answer-constraint
checks. These configurable thresholds are conservative development settings,
not independently calibrated relevance probabilities. Original transcript
questions are lexical aliases; experiment formulations are never indexed.

The final generation bundle is capped at `top_k`, including web supplements,
when source priority is enabled. Ranking is stable within each tier. Existing
causal-authority hard rejections remain fail-closed. The structured generation
payload retains Sumi Jo provenance, and both Korean and English system prompts
state the authority and conflict policy.

All current Sumi Jo units have unconfirmed measure locations. They are eligible
for relevant no-range questions, but excluded from range-selected retrieval.
Their source questions and source excerpts still preserve all original measure
references. Only Nella fantasia currently has direct Sumi Jo feedback.

```bash
python scripts/build_corpus.py
python -m evaluation.source_priority.run
# Run in the soprano-qa environment with the configured embedding model:
python -m evaluation.source_priority.run --hybrid
python -m unittest tests.test_source_priority tests.test_combined_corpus
```

The corpus schema is version 9; startup detects stale inputs and rebuilds the
runtime corpus. To run the historical expert/web ablation, set
`sumijo.enabled: false` and `retrieval.source_priority: false` in a separate
settings file. Disable both to avoid loading unused Sumi Jo records.

See `evaluation/source_priority/` for the development pilot and saved results.
It evaluates retrieval, not generated-answer fidelity or conflict resolution.
