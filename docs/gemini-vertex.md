# Vertex Gemini answer model

`llm.provider: "vertex_gemini"` makes `gemini-3.8-flash` on Google Cloud Vertex
AI the answer model, in place of the local Qwen3-14B. Nothing else changes: the
same writing prompt, the same hybrid BM25 + Qwen embedding retrieval, and the
same source priority. The English search-query translation also goes to Gemini,
so a host needs only the local embedding checkpoint. The soprano-lab web demo
runs this way because its host has no GPU.

There is no model-based admission, evidence reselection, output review or
quality retry. The rule-based question and source screens run as they do for
the local model. No search tools or Google grounding are enabled in the request.

## Run

Sign in with Application Default Credentials for a Google Cloud project that
has Vertex AI enabled, and set `GOOGLE_CLOUD_PROJECT`. `GOOGLE_CLOUD_LOCATION`
defaults to `global`. Use the same Python environment as the rest of Soprano QA
so the local embedding backend remains available.

```bash
cd soprano-qa-rag-system
python -m pip install -r requirements-vertex.txt
SOPRANO_QA_LLM_PROVIDER=vertex_gemini python scripts/ask.py \
  --piece una-voce-poco-fa --question 'col canto에서는 누가 누구를 따라가나요?'
```

The environment variable works the same way for the library and the demo:
set it before `soprano_qa.service` is imported. The service's `model` and
`generation_metadata` fields identify the provider, the requested and resolved
model, finish reason, token usage and API wall time.

Only ADC credentials are used, not an AI Studio Developer API key. Credentials
remain outside this checkout and must never be committed. Standard API calls
use the configured Google project's billing or credits.

## Options

The defaults are `thinking_level=LOW`, a 2048-token output budget that includes
any provider thinking, and a 180 s request timeout. An `llm.vertex_gemini`
object in the settings overrides `model`, `thinking_level`, `max_output_tokens`
and `timeout_ms`. The local model uses `/no_think` and 768 output tokens;
tokenization and reasoning behavior differ. The system and user text is sent
unchanged, including the Qwen `/no_think` suffix; Gemini's API reasoning
setting is authoritative for its behavior.

The common settings still supply temperature 0, top_p 0.8 and seed 42.
**Google documents that Gemini 3.8 Flash ignores temperature, top_p and top_k.**
Seeds are not comparable across providers either.

The provider returns only final text parts, never thought parts. Empty output,
provider blocks, token-limit truncation, API errors and timeouts raise
`VertexGenerationError`, a `soprano_qa.llm.GenerationFailed`, with metadata for
diagnosis. They are not counted as refusals or replaced with a local answer, and
there are no automatic retries. This finish-status handling does not evaluate
the answer's meaning.

## Sources and comparison

- [Official Gemini 3.8 Flash configuration guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Frozen writer evaluation](../evaluation/gemini_flash_writer_2026-11-07/README.md)

The comparison kept all 109 expert and regression cases and the 74-case
bilingual challenge inventory. One historical English query-translation failure
has no baseline writer input, so only 73 challenge cases could be paired.
Offline Claude judging and the frozen BERTScore configuration are separate from
serving. At that time English query translation still used the local model.

The [first-attempt comparison](../evaluation/gemini_flash_writer_2026-11-07/report.md)
found better source fidelity on returned answers, but six missing expert
responses and slower API timings under shared load. Its full 105-answer
BERTScore of 0.897379 missed the 0.90 target; the 99 returned answers scored
0.951765. After a retry on newly authorized credentials, all 105 expert answers
were available and the fixed BERTScore was 0.950992; see the
[recovery supplement](../evaluation/gemini_flash_recovery_2026-11-07/report.md).
No new latency benchmark was run.
