"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, className = "", text = null) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
};
const FAMILY = { original: "전문가 원질문", paraphrase: "표현 변형", ku_derived: "KU 파생 질문" };
const FIDELITY = { faithful: "의미 보존", minor_drift: "경미한 의미 변경", material_drift: "중대한 의미 변경", uncertain: "판정 불확실", not_applicable: "해당 없음" };
let storageKey;
const FEATURED = "expert-kim-in-flowery-clouds-02__m11-12";
const state = { evaluation: null, data: null, selected: null, filtered: [], focus: "all", notes: {}, storageOK: true };
const assessed = (v) => v.judge.available !== false && Number.isFinite(v.judge.naturalness);
const pairedAssessment = (c) => assessed(c.versions.original) && assessed(c.versions.current);
const unscorable = (v) => !v.bertscore.scored && v.bertscore.reason !== "not_evaluated";
const numberText = (n, digits) => Number.isFinite(n) ? (digits == null ? String(n) : n.toFixed(digits)) : "미평가";
const filters = [
  ["all", "전체 문항", () => true],
  ["previous-refusals", "이전 거절 5건", (c) => c.versions.current.previous_refusal === true],
  ["recovered", "API 오류 후 재시도", (c) => c.versions.current.technical_retry === true],
  ["declined", "자연스러움 하락", (c) => pairedAssessment(c) && c.versions.current.judge.naturalness < c.versions.original.judge.naturalness],
  ["resolved", "중대한 의미 변경 해소", (c) => pairedAssessment(c) && c.versions.original.judge.source_fidelity === "material_drift" && c.versions.current.judge.source_fidelity !== "material_drift" && usefulVersion(c.versions.current)],
  ["unavailable", "답변 미제공", (c) => !usefulVersion(c.versions.current)],
  ["remaining", "선택 버전 의미·사실 검토", (c) => assessed(c.versions.current) && (c.versions.current.judge.source_fidelity !== "faithful" || c.versions.current.judge.factual_correctness <= 2)],
  ["unreviewed", "최신 답변 미평가", (c) => !assessed(c.versions.current)],
  ["overlength", "BERTScore 계산 불가", (c) => unscorable(c.versions.current) || unscorable(c.versions.original)],
  ["marked", "내 검토 기록", (c) => hasNote(c.id)],
];

function usefulVersion(v) {
  return (!v.status || v.status === "response") && !["unavailable", "response_error", "domain_guidance"].includes(v.generation_mode)
    && !v.judge.refusal && Boolean(v.answer.trim());
}

function hasNote(id) {
  const n = state.notes[id];
  return Boolean(n && (n.bookmarked || n.text || n.verdict));
}
function noteFor(id) { return state.notes[id] || { bookmarked: false, text: "", verdict: "" }; }
function pieceFor(id) { return state.data.pieces.find((p) => p.id === id); }
function rangeText(range) { return range ? `${range[0]}–${range[1]}마디` : "마디 선택 없음"; }
function rangesText(ranges) { return ranges?.length ? ranges.map((r) => `${r[0]}–${r[1]}`).join(", ") + "마디" : "곡 전체 / 구간 지정 없음"; }
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $("toast").hidden = true; }, 3200);
}
function saveNote(id, patch) {
  state.notes[id] = { ...noteFor(id), ...patch, updated_at: new Date().toISOString() };
  if (!hasNote(id)) delete state.notes[id];
  try { localStorage.setItem(storageKey, JSON.stringify(state.notes)); state.storageOK = true; }
  catch { state.storageOK = false; }
  const status = $("note-status");
  if (status) status.textContent = state.storageOK ? "이 브라우저에 저장됨" : "브라우저 저장 실패 · 내보내기로 보관하세요";
  $("export-notes").disabled = !Object.keys(state.notes).some(hasNote);
  renderFilterButtons();
  renderList();
}
function renderOverview() {
  const old = state.data.summary.original, now = state.data.summary.current;
  const items = [
    ["중대한 출처 의미 변경", numberText(old.all.material_drifts), numberText(now.all.material_drifts), "건", `표시된 ${state.data.cases.length}문항 · 원본 → ${state.evaluation.shortLabel}`, false],
    ["자연스러움 평균", numberText(old.all.naturalness_mean_effective_out_of_4, 2), numberText(now.all.naturalness_mean_effective_out_of_4, 2), "/ 4", state.data.naturalness_comparison ? `양쪽 답변 제공 문항 중 ${state.data.naturalness_comparison.losses}건 하락` : "최신 답변은 아직 Claude 미평가", true],
    ["전문가 원질문 BERTScore", numberText(old.original.bertscore_effective_f1, 4), numberText(now.original.bertscore_effective_f1, 4), "F1", state.evaluation.metricNote, false],
    [state.evaluation.latencyTitle || "응답 지연 중앙값", old.all.latency_all.median_seconds.toFixed(2), now.all.latency_all.median_seconds.toFixed(2), "초", state.evaluation.latencyNote || "버전별 1회 실행 · 생성 당시 기록", true, state.evaluation.latencyComparable === false ? "/" : "→"],
  ];
  $("overview").replaceChildren(...items.map(([title, before, after, unit, note, warn, separator = "→"]) => {
    const box = el("article", `stat${warn ? " warn" : ""}`), values = el("div", "stat-values");
    values.append(el("span", "stat-old", String(before)), el("span", "stat-arrow", separator), el("strong", "stat-new", String(after)), el("span", "stat-unit", unit));
    box.append(el("h2", "stat-title", title), values, el("p", "stat-note", note));
    return box;
  }));
}
function renderFilterButtons() {
  $("focus-filters").replaceChildren(...filters.map(([key, label, predicate]) => {
    const button = el("button", "filter-chip", label);
    button.dataset.filter = key;
    button.setAttribute("aria-pressed", String(key === state.focus));
    button.append(el("span", "count", String(state.data.cases.filter(predicate).length)));
    button.addEventListener("click", () => { state.focus = key; updateResults(); });
    return button;
  }));
}
function updateResults() {
  const query = $("search").value.trim().toLocaleLowerCase();
  const piece = $("piece").value, family = $("family").value;
  const predicate = filters.find(([key]) => key === state.focus)[2];
  state.filtered = state.data.cases.filter((c) =>
    (!piece || c.piece_id === piece) && (!family || c.family === family) && predicate(c) &&
    (!query || c.searchText.includes(query)));
  const order = $("sort").value;
  if (order === "naturalness") state.filtered.sort((a, b) => (a.versions.current.judge.naturalness ?? Infinity) - (b.versions.current.judge.naturalness ?? Infinity) || a.index - b.index);
  if (order === "decline") state.filtered.sort((a, b) => (pairedAssessment(a) ? a.versions.current.judge.naturalness - a.versions.original.judge.naturalness : Infinity) - (pairedAssessment(b) ? b.versions.current.judge.naturalness - b.versions.original.judge.naturalness : Infinity) || a.index - b.index);
  if (order === "bert") state.filtered.sort((a, b) => (a.versions.current.bertscore.f1 ?? Infinity) - (b.versions.current.bertscore.f1 ?? Infinity) || a.index - b.index);
  if (!state.filtered.some((c) => c.id === state.selected)) state.selected = state.filtered[0]?.id || null;
  renderFilterButtons(); renderList(); renderComparison(); updateURL(false);
}
function renderList() {
  $("result-count").textContent = `${state.filtered.length} / ${state.data.cases.length}문항`;
  $("case-list").replaceChildren(...state.filtered.map((c) => {
    const button = el("button", "case-item");
    button.dataset.caseId = c.id;
    button.setAttribute("aria-current", String(state.selected === c.id));
    const piece = el("div", "case-piece"), bottom = el("div", "case-bottom");
    piece.append(el("span", "", pieceFor(c.piece_id).title), el("span", "", noteFor(c.id).bookmarked ? "★" : String(c.index).padStart(3, "0")));
    bottom.append(el("span", "", rangeText(c.measure_range)));
    const current = c.versions.current.judge, original = c.versions.original.judge;
    if (!assessed(c.versions.current)) bottom.append(el("span", "mini-badge judgment-content", "미평가"));
    else if (current.source_fidelity !== "faithful" || current.factual_correctness <= 2) bottom.append(el("span", "mini-badge danger judgment-content", "선택 버전 검토"));
    else if (original.source_fidelity === "material_drift") bottom.append(el("span", "mini-badge judgment-content", "의미 변경 해소"));
    else if (current.naturalness < original.naturalness) bottom.append(el("span", "mini-badge warn judgment-content", "자연스러움 ↓"));
    button.append(piece, el("div", "case-question", c.question), bottom);
    button.addEventListener("click", () => selectCase(c.id));
    return button;
  }));
  if (!state.filtered.length) $("case-list").append(el("p", "empty", "조건에 맞는 문항이 없습니다."));
}
function updateURL(push) {
  const url = new URL(location.href);
  if (state.evaluation) url.searchParams.set("evaluation", state.evaluation.id);
  url.hash = state.selected ? `case=${encodeURIComponent(state.selected)}` : "";
  if (url.href !== location.href) history[push ? "pushState" : "replaceState"]({}, "", url);
}
function selectCase(id, push = true) {
  state.selected = id; renderList(); renderComparison(); updateURL(push);
  const active = $("case-list").querySelector('[aria-current="true"]');
  if (active) $("case-list").scrollTop = Math.max(0, active.offsetTop - $("case-list").offsetTop - 60);
}
function makeSource(source, reference = false) {
  const box = el("div", "source-block"), label = el("div", "source-label");
  label.append(el("strong", "", source.id || source.knowledge_unit_id || "KU"), el("span", "", rangesText(source.measure_ranges)));
  if (reference) label.append(el("span", "pill", source.reference_role === "supporting" ? "보조 맥락" : "참조 답변"));
  else if (source.evidence_type) label.append(el("span", "pill", source.evidence_type === "expert_annotation" ? "전문가 KU" : source.evidence_type === "sumijo_feedback" ? "조수미 피드백" : source.evidence_type));
  box.append(label, el("p", "source-text", source.text));
  return box;
}
function makeMetric(label, value, unit, unscorable = false) {
  const metric = el("div", "metric"), strong = el("strong", unscorable ? "unscorable" : "", value);
  if (unit) strong.append(el("small", "", ` ${unit}`));
  metric.append(el("span", "", label), strong); return metric;
}
function makeAnswer(c, version) {
  const v = c.versions[version], current = version === "current";
  const card = el("section", `answer-card ${version}`);
  const head = el("div", "answer-head"), title = el("div", "answer-label");
  title.append(el("span", "version-dot"), el("h3", "", current ? state.evaluation.answerLabel : "원본 시스템"));
  head.append(title, el("small", "", current ? state.evaluation.answerSubtitle : "보안 적용 전"));
  const body = el("div", "answer-body"), answer = el("p", "answer-text", v.answer), meta = el("div", "answer-meta");
  answer.dataset.version = version;
  const timeLabel = state.evaluation.latencyLabels?.[version];
  meta.append(el("span", "", "저장된 답변 전문"), el("span", "", `${[...v.answer].length}자 · ${timeLabel ? timeLabel + " " : ""}${v.latency_seconds.toFixed(2)}초`));
  body.append(answer, meta);
  if (v.technical_retry) body.append(el("p", "secondary-note retry-note", "최초 API 호출 오류 후 같은 입력으로 1회 재시도한 답변입니다. 표시 시간은 성공한 재시도 호출만 포함합니다."));
  if (v.status && v.status !== "response") body.append(el("p", "secondary-note", "실행 오류로 생성된 답변이 없습니다."));
  else if (!usefulVersion(v)) body.append(el("p", "secondary-note", "이 음악 질문에 답변을 제공하지 못했습니다."));
  const judge = el("div", "judge-box judgment-content"), jt = el("div", "judge-title");
  jt.append(el("span", "", "CLAUDE 평가"), el("span", `fidelity ${v.judge.source_fidelity || "pending"}`, assessed(v) ? FIDELITY[v.judge.source_fidelity] || v.judge.source_fidelity : "미평가"));
  const metrics = el("div", "metric-row");
  metrics.append(makeMetric("자연스러움", numberText(v.judge.naturalness), assessed(v) ? "/ 4" : ""), makeMetric("사실성", numberText(v.judge.factual_correctness), assessed(v) ? "/ 4" : ""), makeMetric("BERTScore F1", v.bertscore.scored ? v.bertscore.f1.toFixed(4) : v.bertscore.reason === "not_evaluated" ? "미평가" : "계산 불가", "", unscorable(v)));
  judge.append(jt, metrics);
  if (v.bertscore.reason === "not_evaluated") judge.append(el("p", "judge-rationale", "이 답변의 Claude·BERTScore 평가는 아직 실시하지 않았습니다. 이전 버전의 점수를 적용하지 않았습니다."));
  else if (!v.bertscore.scored) judge.append(el("p", "judge-rationale", v.bertscore.reason === "encoder_length_limit"
    ? `${v.bertscore.candidate_tokens}토큰으로 512토큰 제한을 넘었습니다. 답변은 자르지 않았고 전체 집계에서만 0점 기여로 포함했습니다.`
    : "답변 거절 또는 실행 실패로 BERTScore를 계산하지 못했습니다. 전체 평균에는 0점 기여로 포함했습니다."));
  for (const [label, text] of assessed(v) ? [["자연스러움", v.judge.naturalness_rationale], ["의미·내용", v.judge.rationale]] : []) {
    const p = el("p", "judge-rationale"); p.append(el("strong", "", label + " · "), document.createTextNode(text || "판정 설명 없음")); judge.append(p);
  }
  if (v.secondary_note) judge.append(el("div", "secondary-note", "추가 대조 메모 · " + v.secondary_note));
  const evidence = el("details", "evidence"), content = el("div", "evidence-content");
  evidence.append(el("summary", "", `검색된 근거 KU ${v.evidence.length}개 보기`));
  content.append(...v.evidence.map((e) => makeSource(e))); evidence.append(content);
  card.append(head, body, judge, evidence); return card;
}
function renderComparison() {
  const index = state.filtered.findIndex((c) => c.id === state.selected), c = state.filtered[index];
  $("position").textContent = c ? `${index + 1} / ${state.filtered.length} · 문항 비교` : "문항 비교";
  $("prev").disabled = index <= 0; $("next").disabled = index < 0 || index >= state.filtered.length - 1;
  if (!c) {
    const empty = el("div", "empty", "검색어나 필터를 바꿔 문항을 찾아보세요.");
    empty.append(el("br")); const reset = el("button", "outline", "모든 문항 보기"); reset.addEventListener("click", clearFilters); empty.append(reset);
    $("comparison").replaceChildren(empty); return;
  }
  const question = el("div", "question-block"), metadata = el("div", "question-meta"), idRow = el("div", "question-id-row"), actions = el("div", "case-actions");
  metadata.append(el("span", "pill piece", pieceFor(c.piece_id).title), el("span", "pill", rangeText(c.measure_range)), el("span", "pill", FAMILY[c.family]));
  const copy = el("button", "outline", "링크 복사"); copy.addEventListener("click", copyLink);
  const bookmark = el("button", `outline${noteFor(c.id).bookmarked ? " marked" : ""}`, noteFor(c.id).bookmarked ? "★" : "☆");
  bookmark.id = "bookmark"; bookmark.setAttribute("aria-label", "이 문항 북마크"); bookmark.setAttribute("aria-pressed", String(Boolean(noteFor(c.id).bookmarked)));
  bookmark.addEventListener("click", () => {
    saveNote(c.id, { bookmarked: !noteFor(c.id).bookmarked });
    bookmark.textContent = noteFor(c.id).bookmarked ? "★" : "☆";
    bookmark.classList.toggle("marked", Boolean(noteFor(c.id).bookmarked));
    bookmark.setAttribute("aria-pressed", String(Boolean(noteFor(c.id).bookmarked)));
    if (state.focus === "marked") updateResults();
  });
  actions.append(copy, bookmark); idRow.append(el("span", "case-id", c.id), actions);
  const heading = el("h2", "", c.question); heading.id = "selected-question";
  question.append(metadata, heading, idRow);
  const pair = el("div", "pair"); pair.append(makeAnswer(c, "original"), makeAnswer(c, "current"));
  const references = el("details", "references"), refContent = el("div");
  if (c.evaluation_references) {
    references.append(el("summary", "", "버전별 평가 참조 · 원본 / " + state.evaluation.shortLabel));
    refContent.append(el("p", "reference-hint", "평가 참조가 버전별로 달라질 수 있습니다. 아래는 각 평가에 쓰인 참조이며, 실제 검색 근거는 위의 각 답변 아래에 있습니다."));
    const referencePair = el("div", "pair");
    for (const version of ["original", "current"]) {
      const ref = c.evaluation_references[version], section = el("section", "reference-version");
      section.dataset.version = version;
      section.append(el("h3", "", version === "original" ? "원본 평가 참조" : state.evaluation.shortLabel + " 평가 참조"), ...ref.sources.map((r) => makeSource(r, true)));
      const bert = el("details", "bert-reference-details"), text = el("p", "source-text bert-reference", ref.bertscore_text);
      text.dataset.version = version;
      bert.append(el("summary", "", "실제 BERTScore 참조문 보기"), text); section.append(bert); referencePair.append(section);
    }
    refContent.append(referencePair);
  } else {
    references.append(el("summary", "", `공통 전문가 참조 · ${c.references.length}개 KU`));
    refContent.append(el("p", "reference-hint", "평가에 사용한 전문가 KU입니다. 시스템별로 실제 검색된 근거는 각 답변 아래에서 확인할 수 있습니다. 보조 맥락을 모두 포함해야 정답인 것은 아닙니다."), ...c.references.map((r) => makeSource(r, true)));
  }
  references.append(refContent);
  $("comparison").replaceChildren(question, pair, references, makeNotes(c));
}
function makeNotes(c) {
  const box = el("section", "notes"), heading = el("div", "notes-heading"), controls = el("div", "note-controls");
  heading.append(el("h3", "", "나의 검토"), el("span", "", "이 브라우저에만 저장"));
  const label = el("label", "", "검토 의견"); label.htmlFor = "verdict";
  const verdict = el("select"); verdict.id = "verdict";
  [["", "의견 선택"], ["naturalness", "자연스러움 개선 필요"], ["meaning", "의미 확인 필요"], ["both", "의미·자연스러움 모두 검토"], ["source", "원자료 확인 필요"], ["good", "현재 답변이 적절함"]].forEach(([value, title]) => { const option = el("option", "", title); option.value = value; verdict.append(option); });
  verdict.value = noteFor(c.id).verdict;
  verdict.addEventListener("change", () => saveNote(c.id, { verdict: verdict.value }));
  controls.append(label, verdict);
  const input = el("textarea"); input.id = "review-note"; input.setAttribute("aria-label", "문항 검토 메모");
  input.placeholder = "어떤 문장이나 의미가 달라졌나요? 확인할 내용을 적어두세요."; input.value = noteFor(c.id).text;
  input.addEventListener("input", () => saveNote(c.id, { text: input.value }));
  const footer = el("div", "note-footer"), status = el("span", "", state.storageOK ? (hasNote(c.id) ? "이 브라우저에 저장됨" : "입력하면 자동 저장됩니다") : "브라우저 저장 불가 · 내보내기로 보관하세요");
  status.id = "note-status"; footer.append(status, el("span", "", "서버에 전송되지 않습니다 · Alt + ← / → 로 문항 이동"));
  box.append(heading, controls, input, footer); return box;
}
async function copyLink() {
  try { await navigator.clipboard.writeText(location.href); toast("이 문항의 링크를 복사했습니다."); }
  catch { toast("주소창의 URL을 복사하면 같은 문항을 열 수 있습니다."); }
}
function move(delta) {
  const index = state.filtered.findIndex((c) => c.id === state.selected);
  const next = state.filtered[index + delta]; if (next) selectCase(next.id);
}
function clearFilters() {
  $("search").value = ""; $("piece").value = ""; $("family").value = ""; $("sort").value = "source"; state.focus = "all"; updateResults();
}
function exportNotes() {
  const notes = Object.entries(state.notes).filter(([id]) => hasNote(id)).map(([id, note]) => {
    const c = state.data.cases.find((row) => row.id === id);
    return { case_id: id, question: c?.question, piece_id: c?.piece_id, ...note };
  });
  const blob = new Blob([JSON.stringify({ schema_version: 1, evaluation_id: state.data.evaluation_id, exported_at: new Date().toISOString(), notes }, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob), a = el("a"); a.href = url; a.download = `soprano-review-${new Date().toISOString().slice(0, 10)}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast(`${notes.length}개 문항의 검토 기록을 내보냈습니다.`);
}
function readHash() { return new URLSearchParams(location.hash.slice(1)).get("case"); }
function restoreNotes() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    if (!saved || typeof saved !== "object" || Array.isArray(saved)) return;
    const allowed = new Set(state.data.cases.map((c) => c.id));
    for (const [id, note] of Object.entries(saved)) {
      if (!allowed.has(id) || !note || typeof note !== "object") continue;
      state.notes[id] = { bookmarked: note.bookmarked === true, text: typeof note.text === "string" ? note.text : "", verdict: typeof note.verdict === "string" ? note.verdict : "", updated_at: typeof note.updated_at === "string" ? note.updated_at : null };
    }
  } catch { state.storageOK = false; }
}
async function init() {
  try {
    const catalogResponse = await fetch("evaluations.json", { cache: "no-cache" });
    if (!catalogResponse.ok) throw new Error("평가 버전 목록을 불러올 수 없습니다.");
    const catalog = await catalogResponse.json();
    const requested = new URL(location.href).searchParams.get("evaluation") || catalog.default;
    state.evaluation = catalog.evaluations.find((e) => e.id === requested);
    if (!state.evaluation) throw new Error("알 수 없는 평가 버전입니다.");
    storageKey = state.evaluation.storageKey;
    for (const entry of catalog.evaluations) { const option = el("option", "", entry.label); option.value = entry.id; $("evaluation").append(option); }
    $("evaluation").value = state.evaluation.id;
    $("evaluation").addEventListener("change", () => {
      const target = new URL(location.href); target.searchParams.set("evaluation", $("evaluation").value); location.href = target.href;
    });
    $("evaluation-notice").textContent = state.evaluation.note;
    $("method-intro").textContent = state.evaluation.methodIntro;
    $("method-bert").textContent = state.evaluation.methodBert;
    $("method-scope").textContent = state.evaluation.methodScope;
    if (state.evaluation.methodInventory) $("method-inventory").textContent = state.evaluation.methodInventory;
    $("snapshot").textContent = state.evaluation.date + " 답변 기록";
    const response = await fetch(state.evaluation.data, { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    const expectedCount = state.evaluation.caseCount ?? 460;
    if (state.data.cases.length !== expectedCount || new Set(state.data.cases.map((c) => c.id)).size !== expectedCount) throw new Error(`${expectedCount}개 문항을 확인할 수 없습니다.`);
    $("total-count").textContent = String(state.data.cases.length);
    $("coverage-footer").textContent = state.evaluation.footer || `전체 ${state.data.cases.length}문항 · 저장된 답변 ${state.data.cases.length * 2}개 · 모든 문항 한국어`;
    for (const c of state.data.cases) c.searchText = [c.id, c.question, pieceFor(c.piece_id).title, pieceFor(c.piece_id).subtitle, c.versions.original.answer, c.versions.current.answer].join(" ").toLocaleLowerCase();
    restoreNotes();
    for (const p of state.data.pieces) { const option = el("option", "", p.title); option.value = p.id; $("piece").append(option); }
    const requestedCase = readHash();
    state.selected = requestedCase || state.evaluation.featuredCase || FEATURED;
    renderOverview(); updateResults();
    if (requestedCase && !state.data.cases.some((c) => c.id === requestedCase)) toast("이 문항은 선택한 버전의 결과가 없어 첫 문항을 표시합니다.");
    $("export-notes").disabled = !Object.keys(state.notes).some(hasNote);
    $("export-notes").addEventListener("click", exportNotes);
    $("search").addEventListener("input", updateResults);
    for (const id of ["piece", "family", "sort"]) $(id).addEventListener("change", updateResults);
    $("clear-filters").addEventListener("click", clearFilters);
    $("prev").addEventListener("click", () => move(-1)); $("next").addEventListener("click", () => move(1));
    $("show-judgments").addEventListener("change", () => document.body.classList.toggle("hide-judgments", !$("show-judgments").checked));
    window.addEventListener("keydown", (event) => {
      if (!event.altKey || !["ArrowLeft", "ArrowRight"].includes(event.key) || event.target.matches("input, textarea, select") || $("method-dialog").open) return;
      event.preventDefault(); move(event.key === "ArrowLeft" ? -1 : 1);
    });
    window.addEventListener("popstate", () => { state.selected = readHash() || state.evaluation.featuredCase || FEATURED; clearFilters(); });
  } catch (error) {
    $("comparison").replaceChildren(el("p", "empty", `평가 데이터를 불러오지 못했습니다. localhost 서버에서 열고 새로고침해 주세요. (${error.message})`));
    $("result-count").textContent = "불러오기 실패";
  }
}
$("method-button").addEventListener("click", () => $("method-dialog").showModal());
$("close-method").addEventListener("click", () => $("method-dialog").close());
$("method-dialog").addEventListener("click", (event) => { if (event.target === $("method-dialog")) { const r = event.target.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close(); } });
init();
