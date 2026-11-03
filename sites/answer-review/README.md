# Local answer comparison demo

The default screen compares the unchanged initial-system answers with fresh
outputs from the current Qwen3-14B semantic-generation work in progress.
The right-hand output is the real service response, including refusals and
failures, without editing or substituting source text. It is not a claim that
the candidate is ready for deployment. `dist/evaluations.json` preserves E37 as
a separate historical version and keeps the old notes key unchanged.

All 460 frozen questions remain. The left column retains its original answers,
Claude judgments and BERTScore values. The latest right column is generated in
`evaluation/latest_demo_2026-11-03/` using the unchanged current working-tree source
snapshot and actual retrieval. Its Claude judgments and fixed-setting BERTScore
are newly computed; no old score is attached to a changed answer. The UI also
supports explicitly unevaluated outputs without treating them as zero or failures.
This static review tool does not call a model when opened in a browser.

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
The latest capture/export lives on branch `codex/latest-answer-demo` at
`/home/minhee/.codex/worktrees/rag-latest-answer-demo/soprano-qa-rag-system`.
From that checkout, use `python sites/answer-review/export_latest.py` and
`python sites/answer-review/verify_latest.py` after generation and evaluation.
The independent verifier checks every full answer, unchanged left column,
source text, score mapping and production-source hash.

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
