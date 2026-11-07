# Local answer comparison demo

The default screen compares the **pre-security original system** on the left
with the latest **Gemini3.8Flash** generated answers on the right. It contains
the 109 music cases for which both results exist: 105 expert-based cases and
4 additional regression cases. The remaining 351 music questions have not been
run with Gemini. All three historical 460-case versions remain in the version
selector, with their existing browser-notes keys.

Answers are exported verbatim from
`evaluation/gemini_flash_recovery_2026-11-07/combined/expert.jsonl`.
Six initial API errors were recovered with one retry of the same input; those
answers have a separate disclosure and an “API 오류 후 재시도” filter.
No source text is substituted for a generated answer. The original answer,
retrieved evidence, Claude judgment and BERTScore for each case stay unchanged.
The right column uses the saved Claude and BERTScore results for that exact
answer. Expert-case BERTScore is 0.950992 with the frozen metric setting.

Summary cards cover the displayed 109 pairs (BERTScore: 105 expert cases).
This is a whole-system result comparison: the left column is the initial
system, not the more recent Qwen3-14B candidate J used in the controlled model
comparison. Original and current evaluation references are shown separately
because some references changed. The recorded timings also have different
scopes: original full response vs Gemini generation call only. They do not
constitute a controlled latency comparison. Retry timings include only the
successful call, not the first failed call or waiting time.

The “이전 거절 5건” filter selects the five requests withheld in the November 3
capture. This static tool does not call a model when opened, and it does not
switch the Lab application's RAG backend.

From this directory:

```bash
python -m http.server 8766 --bind 127.0.0.1 --directory dist
```

Open <http://localhost:8766/?evaluation=gemini-20261107>. An existing URL with
an older `evaluation=` parameter still opens that older version; select the
latest Gemini entry or use this URL. To share a particular case locally, use its
“링크 복사” button. The hostname, port and browser must match to reuse local
notes. Notes and bookmarks stay in browser localStorage and can be exported
as JSON; they are not sent to the server. Keep the server running while using
the demo. The user requested localhost-only serving; do not deploy this Site.

The same static files are mirrored to the Lab worktree at
`demo/static/answer-review/`, available through the Lab demo server at
`/static/answer-review/index.html`.

## Data and validation

The implementation and demo are on branch `codex/helpful-rag-responses`
at `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`.
From that checkout:

```bash
python sites/answer-review/export_gemini.py
python sites/answer-review/verify_gemini.py
node --check sites/answer-review/dist/app.js
```

The independent verifier checks all 218 full answers, the unchanged left
column and historical datasets, exact reference and evidence texts, score
mappings, retry markers and source hashes. Exporting does not regenerate or
rescore any answers. `verify_data.py`, `verify_latest.py` and `verify_helpful.py`
still validate their respective historical datasets, independent of the new
default. Running the older exporters changes the default to their own version;
do not run them to refresh this Gemini view.

The exporter reads the frozen evaluation. It projects only fields shown in
the UI: full question/answer/source text, measurements and judgment summaries.
It does not include private traces, local model paths or CLI response metadata.
Unscorable BERTScore values are null, not a misleading computed zero.

Browser checks use an existing Playwright installation and running server:

```bash
PLAYWRIGHT_MODULE=/absolute/path/to/node_modules/playwright \
  node tests/browser.cjs
```

Optional environment variables: `REVIEW_BASE_URL`, `REVIEW_BROWSER_PATH`,
`REVIEW_BROWSER_LIBS`, `REVIEW_SCREENSHOT_DIR`. The test uses an isolated browser
context and does not modify user notes. It covers exact displayed answers,
filter counts, overlength score labels, notes/export persistence, deep links,
no-result recovery and mobile layouts. Browser setup is not required to use
the demo.

For browser QA without starting any HTTP server, set `REVIEW_STATIC_DIR` to the
absolute `dist` path. Playwright intercepts an in-memory test origin and reads
only the local static files; this does not bind a port or start the demo.
