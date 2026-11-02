# Local answer comparison demo

All 460 frozen questions and 920 full answers from `expanded_460_2026-11-02`,
with the original Claude judgments, BERTScore values, retrieved KUs and expert
references. This is a static review tool; it does not generate new answers or
call a model. No dependencies or external assets are required to serve it.

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
