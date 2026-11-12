# Vertex Gemini answer backend

The optional `config/settings.gemini.json` replaces only the single-pass answer
writer with `gemini-3.8-flash` over Google Cloud Vertex AI. It uses the existing J
writing prompt and hybrid BM25 + Qwen embedding retrieval. English search-query
translation still uses Qwen3-14B, so changing the answer model does not also change
the retrieval model. Korean requests need only the local embedding checkpoint.

The local candidate (`settings.single-pass.json`), reviewed default
(`settings.json`), pre-security original and Lab backend are preserved. There is
no online LLM admission, KU reselection, output reviewer or quality retry in this
path. Existing inexpensive source screening remains. No search tools or Google
grounding are enabled in the Gemini request.

## Run

Sign in with Application Default Credentials for a Google Cloud project that
has Vertex AI enabled, and set `GOOGLE_CLOUD_PROJECT`. `GOOGLE_CLOUD_LOCATION`
defaults to `global`. Use the same Python environment as the rest of Soprano QA
so the local embedding backend remains available.

```bash
cd soprano-qa-rag-system
python -m pip install -r requirements-vertex.txt
python scripts/ask.py \
  --settings config/settings.gemini.json \
  --piece una-voce-poco-fa --question 'col canto에서는 누가 누구를 따라가나요?'
```

For the library, set `SOPRANO_QA_SETTINGS` to the absolute Gemini configuration
path **before importing** `soprano_qa.service`. The service's `model` and
`generation_metadata` fields identify the remote writer, requested/resolved
model, finish reason, token usage and API wall time. English translation's local
model is identified separately. The CLI uses Korean questions.

Only ADC credentials are used, not an AI Studio Developer API key. Credentials
remain outside this checkout and must never be committed. No billing settings,
account upgrades, cloud buckets, notifications or hosted services are created.
Standard API calls use the configured Google project's billing/credits.

## Generation and measurement

Gemini uses `thinking_level=LOW` with a 2048-token output budget, including any
provider thinking. The local candidate uses `/no_think` and 768 output tokens;
tokenization and reasoning behavior differ. The exact system/user text is kept
unchanged for the controlled comparison, including the Qwen `/no_think` suffix.
Gemini's API reasoning setting is authoritative for its behavior.

The common configuration still supplies temperature 0, top_p .8 and seed42.
**Google documents that Gemini 3.8 Flash ignores temperature/top_p/top_k.** Their presence
does not make this a matched-decoding or deterministic comparison; seeds are
also not comparable across providers. This is a provider/model swap with fixed
task context, not an isolation of model size or quantization.

The provider returns only final text parts, never thought parts. Empty output,
provider blocks, token-limit truncation and network errors remain technical
failures, with metadata available for diagnosis; they are not counted as safe
refusals or silently replaced with a local answer. There are no automatic API
retries. This finish-status handling does not evaluate the answer's meaning.

## Sources and comparison

- [Official Gemini 3.8 Flash configuration guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Frozen writer evaluation](../evaluation/gemini_flash_writer_2026-11-07/README.md)

The comparison keeps all 109 expert/regression cases and the 74-case bilingual
challenge inventory. One historical English query-translation failure has no
baseline writer input, so only 73 challenge cases can be paired. Offline Claude
judging and the frozen BERTScore configuration are separate from serving.

The active comparison uses only this one Gemini model and the frozen Qwen3-14B
candidate J. Prior Argon partial captures are archived; they are not resumed or
substituted for Flash results. The original comparison used account3 after an unsuccessful account4 setup.
The user subsequently configured and authorized the new account4 credentials.
The seven missing answers were recovered there with the same Flash model; see the
[recovery supplement](../evaluation/gemini_flash_recovery_2026-11-07/report.md).
Shared YouTube API work continues unchanged, so observed API delays and 429s
cannot be attributed to model speed alone.

The [original first-attempt comparison](../evaluation/gemini_flash_writer_2026-11-07/report.md)
found better source fidelity on returned answers, but six missing expert responses
and slower API timings under shared load. First-attempt full105 BERTScore0.897379
did not meet the0.90 target; returned99 answers scored0.951765. The default
configuration remains local/reviewed, and this Gemini backend is opt-in.

After the newly authorized account4 retry, all105 expert answers are available
and fixed BERTScore is0.950992. The original first-attempt score and failures
remain unchanged. No new latency benchmark was run.
