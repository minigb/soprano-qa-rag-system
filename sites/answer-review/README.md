# Local answer comparison demo

The default screen compares unchanged initial-system answers with outputs from
Qwen3-14B on `codex/helpful-rag-responses`. The current policy provides usable
music answers with a qualification when quality checks remain unresolved and
uses general music knowledge when source-based generation cannot answer.
The displayed text is the actual service response, including warnings or errors.
No source text is substituted for a generated answer.

All 460 frozen questions remain. The left column retains its original answers,
Claude judgments and BERTScore values. The right column is captured with actual
hybrid retrieval in `evaluation/helpful_responses_2026-11-04/full-final/`.
Changed question/answer/evidence packets receive new Claude Opus 5.5 assessments;
identical complete packets reuse the first prior judgment under the same rubric
and model. BERTScore is recomputed on full displayed answers with the existing
fixed setting. No score for a different answer is transferred.

The “이전 거절 5건” filter selects the five requests withheld in the November 3
capture. E37 and the November 3 outputs remain separate historical versions,
with their original browser-notes keys. This static tool does not call a model
when opened, and it does not switch the Lab application's RAG backend.

From this directory:

```bash
python -m http.server 8766 --bind 127.0.0.1 --directory dist
```

Open <http://localhost:8766/>. To share a particular case locally, use its
“링크 복사” button. The hostname, port and browser must match to reuse local
notes. Notes and bookmarks stay in browser localStorage and can be exported
as JSON; they are not sent to the server. Keep the server running while using
the demo. The user requested localhost-only serving; do not deploy this Site.

The same static files are mirrored to the Lab worktree at
`demo/static/answer-review/`, available through the Lab demo server at
`/static/answer-review/index.html`.

## Data and validation

```bash
python export_data.py
python verify_data.py
node --check dist/app.js
```

Those first two commands reproduce/verify the preserved historical E37 data.
The latest implementation and capture are on branch `codex/helpful-rag-responses`
at `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`.
From that checkout, run `python sites/answer-review/export_helpful.py` and
`python sites/answer-review/verify_helpful.py` after generation and evaluation.
The independent verifier checks all 920 full answers, the unchanged left
column, source text, score mappings and frozen implementation hashes.

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
