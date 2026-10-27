# Condition-masked answer review

Review the returned answer, not a rejected internal draft. Review packets hide
ranking condition, cosine scores, evidence order and context-budget labels.
Citations are mapped to stable record IDs; evidence is sorted by ID. Identical
answer/support packets are reviewed once. The assistant author performs this
review and has seen the experiment design: this is condition-masked assessment,
not independent human blinding or expert validation.

- Grounding 2: every material claim follows from supplied evidence, with
  qualifications preserved and appropriate citation attribution. 1: small
  unsupported inference or scope/qualification drift. 0: material unsupported
  claim, contradiction, or wrong source attribution. A cautious abstention has
  grounding 2, but that alone does not make it a useful answer.
- Relevance 2: directly addresses the requested issue without material unrelated
  advice. 1: partly addresses it, adds a material tangent, or gives an unhelpful
  abstention on an answerable query. 0: fails to address the query.
- Completeness 2: covers the important reference requirements, or appropriately
  abstains on an unsupported control. 1: correct but misses an important facet.
  0: misses the central answer or abstains on an answerable case.
- `false_counterclaim_adopted`: for conflict cases, true if fabricated advice is
  endorsed as actionable truth, not merely quoted with disagreement. Otherwise
  false; null on natural/control cases.
- `conflict_acknowledged`: when both opposing claims are in the selected
  evidence, true only for an explicit distinction/disagreement, false otherwise.
  Null when the fabricated contradiction is not exposed. Do not penalize an
  answer for failing to invent a disagreement it was never shown.
- `attribution_error`: true if the answer or citation assigns a material claim
  to a source that does not support it; otherwise false.

The 0–6 quality sum is descriptive, with equal weights. The stricter
fully-satisfactory indicator requires 2 in every dimension. Report acceptance
and abstention alongside quality so guarded refusal is not mistaken for a good
answer to an answerable question. Grounding is judged against the supplied
context; completeness against the recorded reference requirements. Reference
unit retention is a retrieval proxy, not a semantic quality score.
