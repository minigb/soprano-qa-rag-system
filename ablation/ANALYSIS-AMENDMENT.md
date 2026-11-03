# Diagnostic separation of G's two existing decisions

Added after observing the first 28 completed cases, before reading Claude judgments.
The initial G bundle included both static/admission/output security checks and strict
handling of baseline deterministic citation/scope validation. Early diagnostics showed
that missing per-sentence citations could dominate G's withholding count even when the
output guard allowed the answer. Calling this entire effect “the security classifier”
would conflate distinct causes.

Two additional S=0/Q=0 conditions are therefore derived from recorded decisions and the
same saved draft, without new generation, new judging, retuning, or replacing any result:

- `security_only`: withhold on static/admission/output guard rejection, but retain the
  baseline's warning-bearing delivery when deterministic citation/scope validation fails.
- `validation_only`: withhold on deterministic validation failure, without applying the
  static/admission/output guards.

Each derived visible response is exactly one of the existing g0s0q0 or g1s0q0 responses,
with identical question/evidence/reference metadata. Reuse that exact packet's first
Claude judgment. These are paired counterfactual decision rules, not fresh model runs.
Keep the original 10 conditions and all measured results unchanged. Report the original
G contrast as the combined gate/strict-validation effect, never as the isolated guard.
The primary one-component comparisons use security_only, g0s1q0, and g0s0q1 versus g0s0q0;
validation_only directly addresses the user's warning-versus-withholding requirement.

The production current system also changes structured drafting. That change is deliberately
held out here to retain the original writer; these rates are not deployment estimates for
the complete current production system.

## Post-run completeness correction, before final aggregation

After all 233 generations, code review found that the frozen G runner called the
static input/source screens and semantic output gate but omitted the separate
`screen_generated_answer` pattern screen. `audit_static_output.py` applies that
existing deterministic function to the saved displayed answers, without new LLM
calls or changed wording. It rejects 20 G-off rendered alternatives. All were
already withheld in their corresponding G-on condition, so the original ten
conditions and the factorial effects do not change.

The previously planned security-only derivation, however, would release one of
these cases: `direct-09-ko` ends with an injected marker and the semantic gate
incorrectly allowed it. The full static output screen rejects that marker.
The final `security_only` derivation therefore includes this existing check and
uses the already judged G-on refusal for that one case. No normal-question
decision changes. The pre-correction difference and all rejected outputs remain
in `static-output-audit.json`; this is a completeness correction to the isolated
security decision, not a newly tuned guard or favorable re-generation.
