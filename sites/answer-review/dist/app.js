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
const STORAGE_KEY = "soprano-answer-review:expanded_460_2026-11-02:v1";
const FEATURED = "expert-kim-in-flowery-clouds-02__m11-12";
const state = { data: null, selected: null, filtered: [], focus: "all", notes: {}, storageOK: true };
const filters = [
  ["all", "전체 문항", () => true],
  ["declined", "자연스러움 하락", (c) => c.versions.current.judge.naturalness < c.versions.original.judge.naturalness],
  ["resolved", "중대한 의미 변경 해소", (c) => c.versions.original.judge.source_fidelity === "material_drift" && c.versions.current.judge.source_fidelity !== "material_drift"],
  ["remaining", "현재 의미·사실 검토", (c) => c.versions.current.judge.source_fidelity !== "faithful" || c.versions.current.judge.factual_correctness <= 2],
  ["overlength", "BERTScore 계산 불가", (c) => !c.versions.current.bertscore.scored || !c.versions.original.bertscore.scored],
  ["marked", "내 검토 기록", (c) => hasNote(c.id)],
];

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
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.notes)); state.storageOK = true; }
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
    ["중대한 출처 의미 변경", old.all.material_drifts, now.all.material_drifts, "건", "전체 460문항 · 원본 → 현재", false],
    ["자연스러움 평균", old.all.naturalness_mean_useful_out_of_4.toFixed(2), now.all.naturalness_mean_useful_out_of_4.toFixed(2), "/ 4", `${state.data.naturalness_comparison.losses}문항에서 하락 · Claude 평가`, true],
    ["전문가 원질문 BERTScore", old.original.bertscore_effective_f1.toFixed(4), now.original.bertscore_effective_f1.toFixed(4), "F1", "105문항 평균 · 기준 0.90 충족", false],
    ["응답 지연 중앙값", old.all.latency_all.median_seconds.toFixed(2), now.all.latency_all.median_seconds.toFixed(2), "초", "버전별 1회 실행 · 생성 당시 기록", true],
  ];
  $("overview").replaceChildren(...items.map(([title, before, after, unit, note, warn]) => {
    const box = el("article", `stat${warn ? " warn" : ""}`), values = el("div", "stat-values");
    values.append(el("span", "stat-old", String(before)), el("span", "stat-arrow", "→"), el("strong", "stat-new", String(after)), el("span", "stat-unit", unit));
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
  if (order === "naturalness") state.filtered.sort((a, b) => a.versions.current.judge.naturalness - b.versions.current.judge.naturalness || a.index - b.index);
  if (order === "decline") state.filtered.sort((a, b) => (a.versions.current.judge.naturalness - a.versions.original.judge.naturalness) - (b.versions.current.judge.naturalness - b.versions.original.judge.naturalness) || a.index - b.index);
  if (order === "bert") state.filtered.sort((a, b) => (a.versions.current.bertscore.f1 ?? -1) - (b.versions.current.bertscore.f1 ?? -1) || a.index - b.index);
  if (!state.filtered.some((c) => c.id === state.selected)) state.selected = state.filtered[0]?.id || null;
  renderFilterButtons(); renderList(); renderComparison(); updateURL(false);
}
function renderList() {
  $("result-count").textContent = `${state.filtered.length} / 460문항`;
  $("case-list").replaceChildren(...state.filtered.map((c) => {
    const button = el("button", "case-item");
    button.dataset.caseId = c.id;
    button.setAttribute("aria-current", String(state.selected === c.id));
    const piece = el("div", "case-piece"), bottom = el("div", "case-bottom");
    piece.append(el("span", "", pieceFor(c.piece_id).title), el("span", "", noteFor(c.id).bookmarked ? "★" : String(c.index).padStart(3, "0")));
    bottom.append(el("span", "", rangeText(c.measure_range)));
    const current = c.versions.current.judge, original = c.versions.original.judge;
    if (current.source_fidelity !== "faithful" || current.factual_correctness <= 2) bottom.append(el("span", "mini-badge danger judgment-content", "현재 답변 검토"));
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
  title.append(el("span", "version-dot"), el("h3", "", current ? "현재 시스템" : "원본 시스템"));
  head.append(title, el("small", "", current ? "E37" : "보안 적용 전"));
  const body = el("div", "answer-body"), answer = el("p", "answer-text", v.answer), meta = el("div", "answer-meta");
  answer.dataset.version = version;
  meta.append(el("span", "", "저장된 답변 전문"), el("span", "", `${[...v.answer].length}자 · ${v.latency_seconds.toFixed(2)}초`));
  body.append(answer, meta);
  const judge = el("div", "judge-box judgment-content"), jt = el("div", "judge-title");
  jt.append(el("span", "", "CLAUDE 평가"), el("span", `fidelity ${v.judge.source_fidelity}`, FIDELITY[v.judge.source_fidelity] || v.judge.source_fidelity));
  const metrics = el("div", "metric-row");
  metrics.append(makeMetric("자연스러움", String(v.judge.naturalness), "/ 4"), makeMetric("사실성", String(v.judge.factual_correctness), "/ 4"), makeMetric("BERTScore F1", v.bertscore.scored ? v.bertscore.f1.toFixed(4) : "계산 불가", "", !v.bertscore.scored));
  judge.append(jt, metrics);
  if (!v.bertscore.scored) judge.append(el("p", "judge-rationale", `${v.bertscore.candidate_tokens}토큰으로 512토큰 제한을 넘었습니다. 답변은 자르지 않았고 전체 집계에서만 0점 기여로 포함했습니다.`));
  for (const [label, text] of [["자연스러움", v.judge.naturalness_rationale], ["의미·내용", v.judge.rationale]]) {
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
  references.append(el("summary", "", `공통 전문가 참조 · ${c.references.length}개 KU`));
  refContent.append(el("p", "reference-hint", "평가에 사용한 전문가 KU입니다. 시스템별로 실제 검색된 근거는 각 답변 아래에서 확인할 수 있습니다. 보조 맥락을 모두 포함해야 정답인 것은 아닙니다."), ...c.references.map((r) => makeSource(r, true)));
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
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
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
    const response = await fetch("data.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    if (state.data.cases.length !== 460) throw new Error("460개 문항을 확인할 수 없습니다.");
    for (const c of state.data.cases) c.searchText = [c.id, c.question, pieceFor(c.piece_id).title, pieceFor(c.piece_id).subtitle, c.versions.original.answer, c.versions.current.answer].join(" ").toLocaleLowerCase();
    restoreNotes();
    for (const p of state.data.pieces) { const option = el("option", "", p.title); option.value = p.id; $("piece").append(option); }
    state.selected = readHash() || FEATURED;
    renderOverview(); updateResults();
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
    window.addEventListener("popstate", () => { state.selected = readHash() || FEATURED; clearFilters(); });
  } catch (error) {
    $("comparison").replaceChildren(el("p", "empty", `평가 데이터를 불러오지 못했습니다. localhost 서버에서 열고 새로고침해 주세요. (${error.message})`));
    $("result-count").textContent = "불러오기 실패";
  }
}
$("method-button").addEventListener("click", () => $("method-dialog").showModal());
$("close-method").addEventListener("click", () => $("method-dialog").close());
$("method-dialog").addEventListener("click", (event) => { if (event.target === $("method-dialog")) { const r = event.target.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close(); } });
init();
