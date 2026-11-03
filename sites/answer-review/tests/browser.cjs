const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
  const launch = { headless: true, args: ["--no-sandbox"] };
  if (process.env.REVIEW_BROWSER_PATH) launch.executablePath = process.env.REVIEW_BROWSER_PATH;
  if (process.env.REVIEW_BROWSER_LIBS) launch.env = { ...process.env, LD_LIBRARY_PATH: process.env.REVIEW_BROWSER_LIBS };
  const browser = await chromium.launch(launch);
  const errors = [], checks = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
    const staticDir = process.env.REVIEW_STATIC_DIR && path.resolve(process.env.REVIEW_STATIC_DIR);
    if (staticDir) await context.route("http://answer-review.test/**", async (route) => {
      const pathname = decodeURIComponent(new URL(route.request().url()).pathname);
      const file = path.resolve(staticDir, "." + (pathname.endsWith("/") ? pathname + "index.html" : pathname));
      if (!file.startsWith(staticDir + path.sep) || !fs.existsSync(file)) return route.fulfill({ status: 404, body: "Not found" });
      const type = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" }[path.extname(file)];
      return route.fulfill({ status: 200, contentType: type, body: fs.readFileSync(file) });
    });
    const page = await context.newPage(); page.on("pageerror", (e) => errors.push(e.message));
    const url = process.env.REVIEW_BASE_URL || (staticDir ? "http://answer-review.test/" : "http://127.0.0.1:8766/");
    const response = await page.goto(url); assert.equal(response.status(), 200);
    await page.waitForSelector(".answer-text");
    const readJSON = async (name) => staticDir ? JSON.parse(fs.readFileSync(path.join(staticDir, name), "utf8"))
      : (await context.request.get(new URL(name, url).href)).json();
    const catalog = await readJSON("evaluations.json");
    const active = catalog.evaluations.find((e) => e.id === (new URL(url).searchParams.get("evaluation") || catalog.default));
    const data = await readJSON(active.data);
    assert.equal(new URL(page.url()).searchParams.get("evaluation"), active.id);
    assert.equal(await page.locator(".case-item").count(), 460);
    const selected = new URLSearchParams(new URL(page.url()).hash.slice(1)).get("case");
    const c = data.cases.find((row) => row.id === selected);
    for (const version of ["original", "current"]) assert.equal(await page.locator(`.answer-text[data-version=${version}]`).textContent(), c.versions[version].answer);
    checks.push("460 cases and full original/current text");
    const assessed = (v) => v.judge.available !== false && Number.isFinite(v.judge.naturalness);
    const paired = (c) => assessed(c.versions.original) && assessed(c.versions.current);
    const predicates = {
      declined: (c) => paired(c) && c.versions.current.judge.naturalness < c.versions.original.judge.naturalness,
      resolved: (c) => paired(c) && c.versions.original.judge.source_fidelity === "material_drift" && c.versions.current.judge.source_fidelity !== "material_drift"
        && !c.versions.current.judge.refusal && c.versions.current.generation_mode !== "unavailable"
        && (!c.versions.current.status || c.versions.current.status === "response"),
      remaining: (c) => assessed(c.versions.current) && (c.versions.current.judge.source_fidelity !== "faithful" || c.versions.current.judge.factual_correctness <= 2),
      unavailable: (c) => c.versions.current.status && c.versions.current.status !== 'response' || c.versions.current.generation_mode === 'unavailable' || Boolean(c.versions.current.judge.refusal) || !c.versions.current.answer.trim(),
      unreviewed: (c) => !assessed(c.versions.current),
      overlength: (c) => Object.values(c.versions).some((v) => !v.bertscore.scored && v.bertscore.reason !== "not_evaluated"),
    };
    for (const [filter, predicate] of Object.entries(predicates)) {
      await page.locator(`[data-filter="${filter}"]`).click();
      assert.equal(await page.locator(".case-item").count(), data.cases.filter(predicate).length);
    }
    if (data.cases.some(predicates.overlength)) {
      assert.equal(await page.locator(".unscorable").first().textContent(), "계산 불가");
      const overlong = new URLSearchParams(new URL(page.url()).hash.slice(1)).get("case");
      assert.equal(await page.locator('.answer-text[data-version="current"]').textContent(), data.cases.find((row) => row.id === overlong).versions.current.answer);
    }
    checks.push("paired filters and untruncated overlength answers");
    if (data.assessment_status === "not_evaluated") {
      await page.locator("#clear-filters").click();
      assert.equal(await page.locator('.current .fidelity').textContent(), "미평가");
      assert.equal(await page.locator('.current .unscorable').count(), 0);
      assert.equal(await page.locator('.current .metric strong').allTextContents().then((x) => x.every((v) => v === "미평가")), true);
      assert.equal((await page.locator('#overview').textContent()).includes('NaN'), false);
      assert.equal((await page.locator('#overview').textContent()).includes('null'), false);
      checks.push("unrated latest outputs never treated as zero, improved, declined or unscorable");
    }
    await page.locator("#clear-filters").click();
    await page.locator("#piece").selectOption("nella-fantasia"); assert.equal(await page.locator(".case-item").count(), 75);
    await page.locator("#clear-filters").click(); await page.locator("#family").selectOption("original"); assert.equal(await page.locator(".case-item").count(), 105);
    await page.locator("#clear-filters").click();
    await page.locator("#search").fill("expert-kim-nella-fantasia-03__no-range"); assert.equal(await page.locator(".case-item").count(), 1);
    await page.locator("#search").fill("nonexistent_case_zzzz"); assert.equal(await page.locator(".case-item").count(), 0);
    await page.getByRole("button", { name: "모든 문항 보기", exact: true }).click(); assert.equal(await page.locator(".case-item").count(), 460);
    checks.push("piece/family/search filters and empty-state recovery");
    await page.locator("#show-judgments").uncheck(); assert.equal(await page.locator(".current .judge-box").isVisible(), false);
    await page.locator("#show-judgments").check();
    await page.locator(".current .evidence summary").click(); assert.equal(await page.locator(".current .source-text").first().isVisible(), true);
    await page.locator(".references summary").click(); assert.equal(await page.locator(".references .source-text").first().isVisible(), true);
    checks.push("judgment toggle and full source panels");
    const note = "테스트 메모 <img src=x onerror=alert(1)> & 의미 조건 확인";
    await page.locator("#bookmark").click(); await page.locator("#review-note").fill(note); await page.locator("#verdict").selectOption("meaning");
    const deepLink = page.url(); await page.reload(); await page.waitForSelector("#review-note");
    assert.equal(await page.locator("#review-note").inputValue(), note);
    assert.equal(await page.locator("#verdict").inputValue(), "meaning");
    assert.equal(await page.locator("#bookmark").getAttribute("aria-pressed"), "true");
    const downloadPromise = page.waitForEvent("download"); await page.locator("#export-notes").click();
    const download = await downloadPromise; const exported = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
    assert.equal(exported.notes.length, 1); assert.equal(exported.notes[0].text, note);
    await page.locator('[data-filter="marked"]').click(); assert.equal(await page.locator(".case-item").count(), 1);
    checks.push("local notes/bookmark persistence and JSON export");
    if (catalog.evaluations.length > 1) {
      const other = catalog.evaluations.find((e) => e.id !== active.id);
      await page.locator('#evaluation').selectOption(other.id);
      await page.waitForFunction(({id, label}) => document.querySelector('#evaluation')?.value === id && document.querySelector('.current .answer-label h3')?.textContent === label, {id:other.id, label:other.answerLabel});
      assert.equal(await page.locator('#review-note').inputValue(), '');
      const historical = await readJSON(other.data);
      const id = new URLSearchParams(new URL(page.url()).hash.slice(1)).get('case');
      assert.equal(await page.locator('.answer-text[data-version=current]').textContent(), historical.cases.find((x) => x.id === id).versions.current.answer);
      await page.locator('#evaluation').selectOption(active.id);
      await page.waitForFunction(({id, label}) => document.querySelector('#evaluation')?.value === id && document.querySelector('.current .answer-label h3')?.textContent === label && document.querySelector('#review-note')?.value !== '', {id:active.id, label:active.answerLabel});
      assert.equal(await page.locator('#review-note').inputValue(), note);
      checks.push('historical version remains accessible and version notes stay isolated');
    }
    await page.goto(deepLink); await page.waitForSelector(".answer-text"); assert.equal(page.url(), deepLink);
    await page.locator("#next").click(); const nextLink = page.url(); assert.notEqual(nextLink, deepLink);
    await page.goBack(); await page.waitForSelector(".answer-text"); assert.equal(page.url(), deepLink);
    checks.push("case deep links, next and browser back");
    await page.locator("#method-button").click(); assert.equal(await page.locator("#method-dialog").isVisible(), true);
    await page.keyboard.press("Escape"); assert.equal(await page.locator("#method-dialog").isVisible(), false);
    checks.push("method dialog keyboard close");
    for (const width of [1440, 1100, 800, 390]) {
      await page.setViewportSize({ width, height: 950 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `horizontal overflow at ${width}`);
    }
    if (process.env.REVIEW_SCREENSHOT_DIR) {
      fs.mkdirSync(process.env.REVIEW_SCREENSHOT_DIR, { recursive: true });
      await page.screenshot({ path: path.join(process.env.REVIEW_SCREENSHOT_DIR, "mobile.png"), fullPage: true });
      await page.setViewportSize({ width: 1440, height: 1050 });
      await page.screenshot({ path: path.join(process.env.REVIEW_SCREENSHOT_DIR, "desktop.png"), fullPage: true });
    }
    checks.push("responsive layouts at four viewport widths");
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ status: "passed", checks, browser_errors: errors }));
  } finally { await browser.close(); }
})().catch((e) => { console.error(e); process.exit(1); });
