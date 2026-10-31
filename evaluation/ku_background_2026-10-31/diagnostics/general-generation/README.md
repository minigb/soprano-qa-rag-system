# General-generation development diagnostics

These are isolated empty-evidence probes, **not public-service benchmarks** and
not part of the expert BERTScore target or validation split. The four questions
come from the already frozen development cohort. No output was substituted for
a measured service response.

`general-modes.jsonl` compares greedy constrained output, sampled constrained
output and unconstrained thinking followed by strict parsing. Every attempted
output is retained. The thinking private text is omitted; its hash, total token
usage, final answer, prompt hash and parsing result are recorded. No mode fixed
all Korean terminology errors. The sampled settings follow the original Qwen3
recommendations in [Qwen's documentation](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html#transformers).

`terminology-modes.jsonl` adds Italian/English term names, without definitions.
This improved portamento but retained notation errors. `empty-bundle-diagnostic.jsonl`
shows that requiring an explanation or limitation avoided empty arrays on an
answerable note-value question; it did not establish factual correctness.

The two archived scripts used the production source at `20f46ab` (E4), before
the E5 terminology aid. Run from an E4 checkout with the same model file and
copy `general-diagnostic-cases.json` into the indicated runtime directory.
They import that checkout's prompts; running against later source is a new
experiment. E4 source hashes are preserved in `../../provenance/e4/`.
