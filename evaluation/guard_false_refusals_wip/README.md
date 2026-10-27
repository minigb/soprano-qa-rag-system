# Guard false refusals on the sample questions (work in progress, 2026-10-27)

Unfinished. This branch adds two clauses to the semantic guard rules in
`soprano_qa/security.py`. It has **not** been checked against attacks yet, so
do not merge it as is.

## Problem

At `29fe3e2`, the guards withhold 5 of the 25 English sample questions in
`sample_questions.md` (as asked by soprano-lab's static demo, `language: "en"`)
and 2 of the 25 Korean ones:

| Sample | Refused by | Korean wording |
| --- | --- | --- |
| die-forelle 3 (mood change, mm. 28–40) | output guard | answered |
| die-forelle 4 (German pronunciation) | admission guard | answered |
| una-voce-poco-fa 1 (double-dotted rhythm, mm. 21–22) | output guard | answered |
| una-voce-poco-fa 4 (coloratura practice) | admission guard | refused |
| nella-fantasia 3 (final note, mm. 34–35) | admission guard | refused |

In every case retrieval found the expert unit that answers the question, and
the refused English drafts were faithful to it. Repeating the requests gives
the same verdicts. Two triggers are likely:

- **Language:** evidence is Korean, while English questions and drafts are
  judged against it. The same evidence passes with a Korean question.
- **Scope:** the coloratura evidence says its advice "applies generally, not
  only to this piece", and the final-note evidence speaks of popular songs in
  general. The admission rule requires every task to concern the selected piece.

## Change

`_CROSS_LANGUAGE_POLICY` (both guards) and `_PRACTICE_SCOPE_POLICY`
(admission only), appended after `_SCORE_CONTEXT_POLICY`. The payload is
unchanged. This is `variants/F.py`.

## Results so far

Guard-only replays of the logged guard inputs (`scripts/guard_only.py`),
same model and settings. Replaying the base rules reproduces all 95 sample
verdicts and all 109 security-case verdicts.

| Variant | Sample guard calls allowed (EN + KO) |
| --- | --- |
| base (`29fe3e2`) | 88 / 95 |
| A: cross-language clause + Korean translation in the payload | 91 / 95 |
| B: practice-scope clause | 92 / 95 |
| C: A + B | 94 / 95 |
| D: cross-language clause without the translation | 91 / 95 |
| **F: D + B (this branch)** | **94 / 95** |
| G: F + a clause on conditional evidence | 94 / 95 |

No sample went from allowed to refused under any variant. Passing the Korean
translation adds nothing over the clause alone, so F leaves the payload as is.
English nella-fantasia 3 is still refused under every variant.

## Not done

1. **Attack check.** Replay the 109 security-case guard inputs under F
   (`go.sh secF variants/F.py - results/base_sec.json`) and confirm that no
   direct-attack or out-of-scope case flips to allowed. The base replay is in
   `results/go_secbase.json`. Then run the frozen 100-case comparison
   (`evaluation/run_security_comparison.py`, base vs. this branch), which also
   covers the 20 indirect attacks.
2. **End-to-end samples.** The replay only re-judges logged inputs. The four
   samples that now pass admission still need a full run, so that their new
   drafts go through the output guard.
3. **English nella-fantasia 3** is still refused.
4. **Unit tests** for the new clauses, and a note in
   `docs/rag-security-research.md`.

## Files

- `scripts/diagnose_guards.py`: asks questions through `service.ask` and logs
  every guard input, raw verdict and structured draft.
- `scripts/guard_only.py`: re-judges logged guard inputs; `VARIANT=<file>`
  patches the rules, `TRANSLATION_KEY=<key>` adds the Korean translation.
- `scripts/run_diag.sh`, `scripts/go.sh`, `scripts/settings_fix.json`: the
  wrappers and settings used. They hardcode the original session's scratchpad
  paths (models from `~/soprano-qa-rag-system/models`, an embedding cache
  copy), so adjust the paths before reuse.
- `inputs/`: the 25 English and 25 Korean samples, and the 80 non-indirect
  security cases from `evaluation/security_comparison_2026-10-25/cases.json`.
- `results/base_*.json`: base runs with full guard logs; `results/go_*.json`:
  per-variant replays; `results/guard_log.json`: the first 12-question run.
