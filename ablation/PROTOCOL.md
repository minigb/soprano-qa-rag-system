# Component attribution experiment, 2026-11-03 KST

No baseline/current production code, settings, source corpus, or reference answers are edited.
The generator is fixed to Qwen3-14B-Q6_K, temperature 0, top_p .8, seed 42, 768 output tokens.
Contexts are frozen recorded pre-security retrieval results in their original order, not newly
retrieved for each condition. The two p/PP fixtures use the same existing two-source pool in
every condition. Missing historical contexts remain explicitly empty; failures are not dropped.
English questions use their frozen retrieval result, so translation/retrieval latency is excluded.

The cohort is the complete initial bilingual100, all expert105, all background24, two existing
p/PP diagnostic questions, and two new Korean/English kimchi-recipe boundary requests: 233.
This is a reused internal diagnostic set, not an independent holdout or a representative population.

The primary design is the full 2×2×2 factorial:

- G: current static/admission/output checks and strict quality-validation withholding on/off.
  G never modifies the writer prompt. Both potential drafts are retained even if admission would
  deny them. These shadow computations are offline only and not released by G-on conditions.
- S: current additional KU selection on/off, after the same frozen retrieval result. The original
  writer prompt is unchanged except for the selected context. Selection calls/results are recorded.
- Q: substitute the complete KUs cited by that exact draft, versus return generated prose.
  Korean insertion is literal. English uses exact-hash historical source translations where available,
  otherwise the same14B model translates the complete source, with that extra call disclosed.
  No cited source means Q has no eligible source to substitute: retain the original response and flag
  Q as inapplicable. Q uses a shared citation choice to isolate rendering, not reproduce E37's
  separate selection-only generation. No claim of an exact E37 production replay is made.

The original prompt and baseline finalizer (including warning-bearing delivery) are retained.
The same exact draft is shared across G/Q comparisons. Only S changes source content before writing.
Guard decisions are evaluated against each actual rendered alternative. Results are paired
counterfactuals, not independent repeated generation samples. Report conditional substitution effects
and all-case results, separately by language. Quote translation effects remain a limitation.

Two additional policy conditions use S=0/Q=0:

- warning: preserve useful drafts with a visible caution when evidence/quality validation fails,
  but retain a separate instruction/domain-only security check. Nonmusic tasks receive a polite,
  explicit explanation of the soprano assistant's scope, not the generic no-evidence message.
- helpful: the same warning policy; when the draft is absent, unhelpful, or unsafe, attempt one
  music-focused answer allowing accurate general knowledge and explicit local uncertainty.
  Unknown score facts are not invented. A missing exact score fact can be accompanied by useful
  general guidance. Unsafe/nonmusic task completion is never authorized by a quality warning.

The policy comparison is separate from the three-factor effects: domain phrasing, security-only
review, and fallback behavior are documented rather than attributed to a single prompt sentence.
Always record refusals, limited responses, errors, selected/dropped KUs, guard outcomes, displayed
warnings, all actual calls, and latency excluding retrieval. Do not count polite domain explanations
as music false refusals. Score available meaningful music content, not mere nonempty strings.

Claude Opus5.5 CLI judges blinded exact output/evidence packets using the existing meaning,
factuality, naturalness, relevance, completeness and attack-success rubric. Exact duplicate packets
share one verdict; never choose a favorable re-judgment. Report every failed call and refusal.
Retain generation drafts/diagnostics privately. Do not send private reasoning to the evaluator.
Capture full visible answer examples for newly introduced/resolved semantic errors and lost content.
No metric tuning or adoption to production occurs in this experiment.
