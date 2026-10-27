# Matched-candidate evaluation protocol (frozen before inference)

Question: Does authority ordering improve answers relative to relevance-only
ordering when both receive exactly the same candidate pool?

Twenty new development cases: 11 natural answerable questions spanning all five
works, three negative/scope controls, and six explicitly synthetic conflicts.
Seven natural questions target direct Sumi Jo feedback. These are newly authored
but not an independent external benchmark; the author knows the corpus.

For natural cases, score same-piece answers with the configured Qwen3 embedding
model, without questions, aliases, author names, or provenance in the embedding.
Freeze the top six above cosine 0.43. Range-selected cases require confirmed
overlap. This common gate and pool are identical for both conditions. This tests
ranking conditional on a common pool, not the production retriever's recall.

For conflicts, freeze a controlled three-record pool: an unchanged authoritative
corpus answer, another unchanged same-piece expert answer, and a fabricated
lower-tier contradiction labeled synthetic in its provenance. These are stress
tests, not observed real-world source disagreements. No invented claim enters
the dataset or runtime corpus. Keep original higher-authority wording intact.

Conditions: relevance sorts by answer cosine, with ID tie-breaking; priority
sorts the same scores by Sumi Jo, expert, web, then relevance within each tier.
Natural cases use top 3. Conflicts use top 2 (selection effect) and all 3
(order-only effect). This produces 52 paired-condition outputs over 26
case/budget pairs. Candidate hashes must match within each pair. Report whether
the selected sets and orders actually differ, including null-treatment pairs.

Use the same production generation prompt, including authority/conflict policy,
Qwen3-14B-Q6_K model, temperature zero, token budget, structured citations,
security guards, and output validation in both arms. Thus the comparator is
relevance-only *ranking*, not removal of source metadata or authority policy.
Run a seeded randomized schedule. Cache identical prompts rather than treating
identical deterministic completions as independent observations. Persist every
failure and abstention; never silently regenerate unsuccessful answers.

Evaluation: report reference-unit availability/retention, mean selected cosine,
and output acceptance. Review anonymized answers with evidence IDs canonicalized
and condition/order hidden. Grade grounding, relevance, and completeness 0–2;
identify unsupported claims, incorrect source attribution, and unsafe adoption
of the fabricated contradiction. Grade explicit conflict acknowledgment only
when both opposed claims were actually exposed. An appropriate abstention is a
success on negative controls but incomplete on an answerable case.

Reviews are performed by the coding assistant, not independent humans or expert
singers, and are labeled accordingly. Preserve explanations and review packets
for human reassessment. No regex/keyword match is treated as semantic quality.
Use paired comparisons, with 95% percentile bootstrap intervals resampling
target/source clusters (10,000 draws, seed 20261025); related paraphrases and
budget variants are not independent samples. Report natural/control/conflict
strata separately. All results are exploratory; do not claim statistical
superiority from an underpowered development set or tune thresholds afterward.

Monitor Codex account usage at least every ten minutes between generation calls;
stop at a reported/observed reset, preserving checkpoints. No notifier is used.
