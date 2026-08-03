"use strict";

const ReviewState = window.SopranoReviewState;

const PIECE_LABELS = {
  "die-forelle": "Die Forelle",
  "in-flowery-clouds": "꽃구름 속에",
  "la-capinera": "La Capinera",
  "nella-fantasia": "Nella Fantasia",
  "una-voce-poco-fa": "Una voce poco fa",
};

const STATUS_LABELS = {
  retrievable: "Retrievable",
  measure_review_pending: "Measure pending",
  rewrite_review_pending: "Rewrite pending",
  excluded_unanswerable: "Excluded",
};

const RATING_LABELS = {
  pass: "Pass",
  fail: "Fail",
  uncertain: "Uncertain",
  not_applicable: "N/A",
};

const ISSUE_LABELS = {
  awkward_paraphrase: "Awkward paraphrase",
  range_questionable: "Range questionable",
  retrieval_miss: "Retrieval miss",
  irrelevant_evidence: "Irrelevant evidence",
  unsupported_answer: "Unsupported answer",
  factual_error: "Factual error",
  citation_error: "Citation error",
  needs_expert_review: "Needs expert review",
};

const SEMANTIC_LABELS = {
  reliable: "Reliable RAG+LLM",
  pass: "Semantic pass only",
  review: "Human review",
  fail: "Semantic fail",
  pending: "Judge pending",
};

const SEMANTIC_TONES = {
  reliable: "success",
  pass: "info",
  review: "warning",
  fail: "danger",
  pending: "",
};

const state = {
  payload: null,
  draft: null,
  fingerprint: null,
  storageKey: null,
  selectedSourceId: null,
  selectedCaseBySource: new Map(),
  pieceId: "all",
  query: "",
  annotator: "all",
  reviewStatus: "all",
  range: "all",
  basis: "all",
  generationMode: "all",
  runStatus: "all",
  variant: "all",
  semantic: "all",
  completion: "all",
  manualReviewEnabled: false,
};

const elements = {
  viewerEyebrow: document.getElementById("viewer-eyebrow"),
  viewerTitle: document.getElementById("viewer-title"),
  viewerSubtitle: document.getElementById("viewer-subtitle"),
  findingBanner: document.getElementById("finding-banner"),
  findingTitle: document.getElementById("finding-title"),
  findingCopy: document.getElementById("finding-copy"),
  findingBadge: document.getElementById("finding-badge"),
  summaryGrid: document.getElementById("summary-grid"),
  progressCopy: document.getElementById("progress-copy"),
  progressLabel: document.getElementById("progress-label"),
  progressTrack: document.getElementById("progress-track"),
  progressFill: document.getElementById("progress-fill"),
  pieceTabs: document.getElementById("piece-tabs"),
  search: document.getElementById("search"),
  annotatorFilter: document.getElementById("annotator-filter"),
  statusFilter: document.getElementById("status-filter"),
  rangeFilter: document.getElementById("range-filter"),
  basisFilter: document.getElementById("basis-filter"),
  generationModeFilter: document.getElementById("generation-mode-filter"),
  runStatusFilter: document.getElementById("run-status-filter"),
  variantFilter: document.getElementById("variant-filter"),
  semanticFilter: document.getElementById("semantic-filter"),
  completionFilter: document.getElementById("completion-filter"),
  resetFilters: document.getElementById("reset-filters"),
  resultCount: document.getElementById("result-count"),
  previous: document.getElementById("previous-question"),
  next: document.getElementById("next-question"),
  nextUnreviewed: document.getElementById("next-unreviewed"),
  queueList: document.getElementById("queue-list"),
  detail: document.getElementById("review-detail"),
  exportReview: document.getElementById("export-review"),
  importReview: document.getElementById("import-review"),
  importReviewLabel: document.getElementById("import-review-label"),
  clearReview: document.getElementById("clear-review"),
  saveStatus: document.getElementById("save-status"),
  announcement: document.getElementById("announcement"),
};

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function appendBadge(container, text, tone = "") {
  container.appendChild(createElement("span", `badge ${tone}`.trim(), text));
}

function formatRange(range) {
  if (!Array.isArray(range)) return "No range";
  return range[0] === range[1]
    ? `m. ${range[0]}`
    : `mm. ${range[0]}–${range[1]}`;
}

function formatRanges(ranges) {
  if (!Array.isArray(ranges) || ranges.length === 0) return "None";
  return ranges.map(formatRange).join(", ");
}

function evidenceRanges(value) {
  if (!Array.isArray(value) || value.length === 0) return [];
  if (
    value.length === 2
    && Number.isInteger(value[0])
    && Number.isInteger(value[1])
  ) {
    return [value];
  }
  return value.filter(range => Array.isArray(range) && range.length === 2);
}

function objectOrEmpty(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : {};
}

function isSemanticPayload(payload) {
  const supported = [
    "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "8.1",
  ];
  return supported.includes(payload?.schema_version);
}

function isSynthesizedPayload(payload = state.payload) {
  if (
    payload?.artifact_type
    === "soprano_qa_synthesized_hybrid_rag_llm_evaluation"
  ) {
    return true;
  }
  return listOrEmpty(payload?.results).some(result => (
    listOrEmpty(result.synthesized_question_variants).length > 0
    || listOrEmpty(result.inference_runs).some(
      run => Boolean(objectOrEmpty(run.inference_input).synthesized_variant_id),
    )
  ));
}

function semanticRunProgress() {
  return ReviewState.semanticRunProgress(state.payload);
}

function manualReviewEnabledForPayload(payload) {
  if (!isSemanticPayload(payload)) return true;
  return ReviewState.semanticRunProgress(payload).judge_complete;
}

function retrievalOutput(run) {
  return objectOrEmpty(run?.retrieval_probe);
}

function generatedOutput(run) {
  return objectOrEmpty(run?.generated_answer);
}

function semanticAggregate(run) {
  return objectOrEmpty(objectOrEmpty(run?.semantic_evaluation).aggregate);
}

function listOrEmpty(value) {
  return Array.isArray(value) ? value : [];
}

function runVariantId(run) {
  return objectOrEmpty(run?.inference_input).synthesized_variant_id || "";
}

function variantSlot(variantId) {
  const match = String(variantId || "").match(/-(syn-\d+)$/);
  return match ? match[1] : String(variantId || "");
}

function runQuestion(run) {
  return objectOrEmpty(run?.inference_input).question || "";
}

function executionStatus(run) {
  if (run?.error) return "error";
  const declared = String(run?.status || "").toLocaleLowerCase();
  if (["failed", "error"].includes(declared)) return "error";
  if (["completed", "complete", "success", "succeeded"].includes(declared)) {
    return "completed";
  }
  if (generatedOutput(run).answer) return "completed";
  return "pending";
}

function runMatchesFilters(run) {
  const generated = generatedOutput(run);
  if (
    state.basis !== "all"
    && state.basis !== "other"
    && generated.answer_basis !== state.basis
  ) {
    return false;
  }
  if (
    state.basis === "other"
    && ["retrieved_evidence", "internal_knowledge"].includes(
      generated.answer_basis,
    )
  ) {
    return false;
  }
  if (
    state.generationMode !== "all"
    && generated.generation_mode !== state.generationMode
  ) {
    return false;
  }
  if (
    state.runStatus !== "all"
    && executionStatus(run) !== state.runStatus
  ) {
    return false;
  }
  return state.variant === "all"
    || variantSlot(runVariantId(run)) === state.variant;
}

function runMatchesQuery(result, run) {
  const query = state.query.toLocaleLowerCase();
  if (!query) return true;
  const reference = objectOrEmpty(result.authoritative_reference);
  const general = [
    result.source_id,
    result.piece_id,
    result.annotator,
    result.original_question,
    result.paraphrased_question,
    ...listOrEmpty(result.knowledge_unit_ids),
    ...listOrEmpty(result.expected_retrieval_eligible_knowledge_unit_ids),
    reference.source_answer,
    ...listOrEmpty(reference.linked_knowledge_units).flatMap(unit => [
      unit.knowledge_unit_id,
      unit.answer,
    ]),
  ]
    .filter(Boolean)
    .join(" ")
    .toLocaleLowerCase();
  if (general.includes(query)) return true;

  const matchingVariantIds = new Set(
    listOrEmpty(result.synthesized_question_variants)
      .filter(variant => flattenSearchValues(variant)
        .join(" ")
        .toLocaleLowerCase()
        .includes(query))
      .map(variant => variant.variant_id),
  );
  if (matchingVariantIds.has(runVariantId(run))) return true;
  return flattenSearchValues(run)
    .join(" ")
    .toLocaleLowerCase()
    .includes(query);
}

function flattenSearchValues(value) {
  if (value === null || value === undefined) return [];
  if (Array.isArray(value)) {
    return value.flatMap(item => flattenSearchValues(item));
  }
  if (typeof value === "object") {
    return Object.values(value).flatMap(item => flattenSearchValues(item));
  }
  return [String(value)];
}

function semanticStatus(result) {
  return ReviewState.semanticQuestionStatus(result);
}

function resultCases(result, { applyRunFilters = true } = {}) {
  const questionReview = state.draft.questions[result.source_id];
  return result.inference_runs
    .map((run, index) => {
      const key = ReviewState.caseKey(result, run, index);
      return {
        key,
        index,
        run,
        review: questionReview.cases[key],
      };
    })
    .filter(item => (
      !applyRunFilters
      || (runMatchesFilters(item.run) && runMatchesQuery(result, item.run))
    ));
}

function ratingNeedsNote(value) {
  return value === "fail" || value === "uncertain";
}

function reviewNeedsNote(review, fields) {
  return fields.some(field => ratingNeedsNote(review[field]))
    && !review.notes.trim();
}

function questionReview(result) {
  return state.draft.questions[result.source_id];
}

function questionComplete(result) {
  const review = questionReview(result);
  return ReviewState.isQuestionComplete(
    review,
    resultCases(result, { applyRunFilters: false }).map(item => item.review),
  );
}

function questionFlagged(result) {
  const review = questionReview(result);
  return ReviewState.isFlagged(
    review,
    resultCases(result, { applyRunFilters: false }).map(item => item.review),
  );
}

function generatedBasis(result) {
  return new Set(
    result.inference_runs
      .map(run => generatedOutput(run).answer_basis)
      .filter(Boolean),
  );
}

function rangeKind(result) {
  const ranges = new Set(
    result.inference_runs
      .map(run => objectOrEmpty(run.inference_input).measure_range)
      .filter(Array.isArray)
      .map(range => `${range[0]}-${range[1]}`),
  );
  if (ranges.size === 0) return "none";
  return ranges.size === 1 ? "single" : "multiple";
}

function filteredResults() {
  return state.payload.results.filter(result => {
    if (state.pieceId !== "all" && result.piece_id !== state.pieceId) {
      return false;
    }
    if (state.annotator !== "all" && result.annotator !== state.annotator) {
      return false;
    }
    if (
      state.reviewStatus !== "all"
      && result.review_status !== state.reviewStatus
    ) {
      return false;
    }
    if (state.range !== "all" && rangeKind(result) !== state.range) {
      return false;
    }
    if (
      !result.inference_runs.some(
        run => runMatchesFilters(run) && runMatchesQuery(result, run),
      )
    ) {
      return false;
    }
    if (
      state.semantic !== "all"
      && semanticStatus(result) !== state.semantic
    ) {
      return false;
    }

    const complete = questionComplete(result);
    const flagged = questionFlagged(result);
    if (state.completion === "complete" && !complete) return false;
    if (state.completion === "incomplete" && complete) return false;
    if (state.completion === "flagged" && !flagged) return false;
    return true;
  });
}

function persistDraft(message = "이 브라우저에 자동 저장됨") {
  if (!state.manualReviewEnabled) {
    elements.saveStatus.textContent =
      "Judge 실행 중에는 수동 semantic-fidelity 검토가 잠겨 있다.";
    return;
  }
  try {
    localStorage.setItem(state.storageKey, JSON.stringify(state.draft));
    elements.saveStatus.textContent = message;
  } catch (error) {
    elements.saveStatus.textContent = `로컬 저장 실패: ${error}`;
  }
}

function readStoredDraft() {
  const base = ReviewState.createDraft(state.payload);
  if (!state.manualReviewEnabled) return base;
  let raw;
  try {
    raw = localStorage.getItem(state.storageKey);
  } catch (error) {
    elements.saveStatus.textContent =
      `브라우저 저장소를 사용할 수 없어 저장 없이 시작함: ${error.message}`;
    return base;
  }
  if (!raw) return base;
  try {
    return ReviewState.normalizeDraft(state.payload, JSON.parse(raw));
  } catch (error) {
    elements.saveStatus.textContent =
      `이전 로컬 검토를 읽지 못해 새로 시작함: ${error.message}`;
    return base;
  }
}

function summaryCard(value, label, tone = "") {
  const card = createElement(
    "div",
    `summary-card ${tone ? `semantic-${tone}` : ""}`.trim(),
  );
  card.append(
    createElement("strong", "", String(value)),
    createElement("span", "", label),
  );
  return card;
}

function countRating(field, value) {
  let count = 0;
  Object.values(state.draft.questions).forEach(question => {
    if (field === "paraphrase_natural") {
      if (question.paraphrase_natural === value) count += 1;
      return;
    }
    Object.values(question.cases).forEach(review => {
      if (review[field] === value) count += 1;
    });
  });
  return count;
}

function synthesizedRunStats() {
  const stats = {
    total: 0,
    completed: 0,
    pending: 0,
    error: 0,
    llm: 0,
    extractive: 0,
    other_mode: 0,
    variants: new Set(),
  };
  state.payload.results.forEach(result => {
    result.inference_runs.forEach(run => {
      stats.total += 1;
      stats[executionStatus(run)] += 1;
      const mode = generatedOutput(run).generation_mode;
      if (mode === "llm") stats.llm += 1;
      else if (mode && mode.includes("extract")) stats.extractive += 1;
      else if (mode) stats.other_mode += 1;
      const variantId = runVariantId(run);
      if (variantId) stats.variants.add(variantId);
    });
  });
  return stats;
}

function renderSummary() {
  const progress = ReviewState.computeProgress(state.payload, state.draft);
  const summary = state.payload.summary;
  const semantic = ReviewState.semanticStatusCounts(state.payload);
  const answerQuality = objectOrEmpty(summary.answer_quality_metric);
  const isSemanticRun = isSemanticPayload(state.payload);
  const isSynthesisRun = isSynthesizedPayload() && !isSemanticRun;
  const judgeProgress = isSemanticRun ? semanticRunProgress() : null;
  const synthesis = isSynthesisRun ? synthesizedRunStats() : null;
  const cards = isSemanticRun
    ? [
      [
        `${judgeProgress.judged_cases} / ${judgeProgress.total_cases}`,
        "Judged cases",
        judgeProgress.judge_complete ? "pass" : "pending",
      ],
      [
        answerQuality.passed_cases ?? 0,
        "Expert-claim fidelity passes",
        "pass",
      ],
      [semantic.cases.reliable, "Reliable grounded RAG+LLM", "reliable"],
      [semantic.cases.review, "Judge review", "review"],
      [semantic.cases.fail, "Judge fail", "fail"],
      [semantic.cases.pending, "Judge pending", "pending"],
    ]
    : isSynthesisRun
      ? [
        [state.payload.results.length, "Human source questions"],
        [synthesis.variants.size, "Synthesized formulations"],
        [synthesis.completed, `Completed cases / ${synthesis.total}`, "pass"],
        [synthesis.llm, "LLM answers", "reliable"],
        [synthesis.extractive, "Extractive safeguards", "review"],
        [
          synthesis.error,
          "Inference errors",
          synthesis.error ? "fail" : "pass",
        ],
      ]
    : [
      [
        progress.completed_questions,
        `Questions reviewed / ${progress.total_questions}`,
      ],
      [progress.completed_cases, `Cases reviewed / ${progress.total_cases}`],
      [countRating("paraphrase_natural", "pass"), "Paraphrase passes"],
      [countRating("answer_acceptable", "pass"), "Answer passes"],
      [
        summary.questions_with_any_expert_evidence ?? 0,
        `Expert-evidence hits / ${summary.retrieval_eligible_questions ?? 0}`,
      ],
      [progress.flagged_questions, "Flagged questions"],
    ];
  elements.summaryGrid.replaceChildren(
    ...cards.map(([value, label, tone]) => summaryCard(value, label, tone)),
  );
  const completionPercent = isSemanticRun
    ? judgeProgress.completion_percent
    : isSynthesisRun
      ? (synthesis.total ? (synthesis.completed / synthesis.total) * 100 : 0)
      : progress.completion_percent;
  elements.progressLabel.textContent = isSemanticRun
    ? "Judge 진행률"
    : isSynthesisRun
      ? "Inference 진행률"
      : "수동 검토 진행률";
  elements.progressCopy.textContent = isSemanticRun
    ? `${judgeProgress.judged_cases} / ${judgeProgress.total_cases} cases judged`
    : isSynthesisRun
      ? `${synthesis.completed} / ${synthesis.total} cases completed`
      : `${progress.completed_questions} / ${progress.total_questions} questions`;
  elements.progressTrack.setAttribute(
    "aria-valuenow",
    String(completionPercent),
  );
  elements.progressFill.style.width = `${completionPercent}%`;
}

function renderFinding() {
  const summary = state.payload.summary;
  if (isSemanticPayload(state.payload)) {
    const primary = objectOrEmpty(summary.primary_metric);
    const quality = objectOrEmpty(summary.answer_quality_metric);
    const progress = semanticRunProgress();
    const judged = progress.judged_cases;
    const passed = quality.passed_cases
      ?? primary.passed_cases
      ?? ReviewState.semanticStatusCounts(state.payload).cases.pass;
    const conservativePassed = primary.passed_cases
      ?? ReviewState.semanticStatusCounts(state.payload).cases.reliable;
    const total = progress.total_cases;
    const rate = progress.judge_complete && total
      ? passed / total
      : null;
    elements.findingBanner.hidden = false;
    elements.findingBadge.classList.remove("success", "warning");
    if (!judged) {
      elements.findingTitle.textContent =
        "생성 답변에 대한 의미 기반 판정이 아직 없다.";
      elements.findingCopy.textContent =
        `전체 ${total}개 추론 사례 중 judge 완료 사례가 없다. 검색 결과는 `
        + "답변 신뢰도와 분리된 진단으로 상세 화면에서 확인할 수 있다.";
      elements.findingBadge.textContent = `0 / ${total} judged`;
      elements.findingBadge.classList.add("warning");
      return;
    }
    if (!progress.judge_complete) {
      elements.findingTitle.textContent =
        "의미 기반 judge 실행이 아직 진행 중이다.";
      elements.findingCopy.textContent =
        `${total}개 추론 사례 중 ${judged}개만 판정되었다. 지금까지 `
        + `${passed}개가 전문가 원문 의미 충실도 기준을 통과했지만, `
        + "완료 전에는 통과율을 표시하지 않는다. "
        + "수동 semantic-fidelity 검토는 스냅샷이 안정될 때까지 잠겨 있다.";
      elements.findingBadge.textContent = `${judged} / ${total} judged · in progress`;
      elements.findingBadge.classList.add("warning");
      return;
    }
    elements.findingTitle.textContent =
      "원문 전문가 답변과 비교한 RAG+LLM 의미 충실도 결과";
    elements.findingCopy.textContent =
      `${total}개 사례 중 ${passed}개가 원문 전문가 주장에 대한 정확성·`
      + `완전성·안전성 기준을 통과했다. 이 중 미해결 curation 상태까지 `
      + `${conservativePassed}개가 자동 신뢰 판정을 받았다. 정확한 KU ID `
      + "적중은 별도 검색 진단이며 의미 충실도 통과 조건이 아니다.";
    const percent = `${(rate * 100).toFixed(1)}%`;
    elements.findingBadge.textContent = `${passed} / ${total} · ${percent}`;
    if (rate >= 0.8) {
      elements.findingBadge.classList.add("success");
    } else if (rate >= 0.5) {
      elements.findingBadge.classList.add("warning");
    }
    return;
  }
  if (isSynthesizedPayload()) {
    const stats = synthesizedRunStats();
    elements.findingBanner.hidden = false;
    elements.findingBadge.classList.remove("success", "warning");
    elements.findingTitle.textContent =
      "Human expected answer and synthesized-question output comparison";
    elements.findingCopy.textContent =
      `${stats.variants.size}개 합성 질문의 ${stats.total}개 범위별 추론 사례 중 `
      + `${stats.completed}개가 완료되었다. 각 사례에서 인간 주석 기반 `
      + "expected/reference "
      + "answer와 생성 답변을 좌우로 비교할 수 있다.";
    elements.findingBadge.textContent =
      `${stats.completed} / ${stats.total} complete`;
    elements.findingBadge.classList.add(
      stats.error || stats.pending ? "warning" : "success",
    );
    return;
  }
  const runCount = summary.inference_runs_completed;
  const expectedHits = summary.questions_with_expected_expert_evidence;
  const expertHits = summary.questions_with_any_expert_evidence;
  const eligible = summary.retrieval_eligible_questions;
  elements.findingBanner.hidden = false;
  elements.findingBadge.classList.remove("warning");
  elements.findingBadge.classList.toggle("success", expertHits > 0);
  if (expertHits === 0) {
    elements.findingTitle.textContent =
      "이 실행에서는 전문가 annotation 근거가 검색되지 않았다.";
    elements.findingCopy.textContent =
      `검색 가능한 질문 ${eligible}개 중 기대한 전문가 근거를 찾은 질문은 `
      + `${expectedHits}개다. 전체 ${runCount}개 추론 중 근거 기반 RAG는 `
      + `${summary.rag_llm_runs}개다.`;
    elements.findingBadge.textContent = `0 / ${eligible} expert hits`;
    return;
  }
  elements.findingTitle.textContent = "전문가 annotation 근거를 찾은 질문이 있다.";
  elements.findingCopy.textContent =
    `검색 가능한 질문 ${eligible}개 중 ${expertHits}개에서 전문가 근거를 찾았고, `
    + `${expectedHits}개에서 기대한 전문가 근거를 찾았다.`;
  elements.findingBadge.textContent =
    `${expertHits} / ${eligible} expert hits`;
}

function renderPieceTabs() {
  const pieces = Object.keys(PIECE_LABELS);
  const options = [["all", "All pieces"], ...pieces.map(
    pieceId => [pieceId, PIECE_LABELS[pieceId]],
  )];
  elements.pieceTabs.replaceChildren();
  options.forEach(([pieceId, label]) => {
    const count = pieceId === "all"
      ? state.payload.results.length
      : state.payload.results.filter(result => result.piece_id === pieceId).length;
    const button = createElement(
      "button",
      "piece-tab",
      `${label} · ${count}`,
    );
    button.type = "button";
    button.dataset.pieceId = pieceId;
    if (pieceId === state.pieceId) {
      button.setAttribute("aria-current", "page");
    }
    button.addEventListener("click", () => {
      state.pieceId = pieceId;
      state.selectedSourceId = null;
      render();
      Array.from(elements.pieceTabs.querySelectorAll(".piece-tab"))
        .find(item => item.dataset.pieceId === pieceId)
        ?.focus();
    });
    elements.pieceTabs.appendChild(button);
  });
}

function statusTone(status) {
  if (status === "retrievable") return "success";
  if (status === "rewrite_review_pending") return "danger";
  return "warning";
}

function selectedResult(items) {
  if (!state.selectedSourceId) return null;
  return items.find(result => result.source_id === state.selectedSourceId) || null;
}

function ensureSelection(items) {
  if (!selectedResult(items)) {
    state.selectedSourceId = items[0]?.source_id || null;
  }
  if (!state.selectedSourceId) return;
  const result = items.find(item => item.source_id === state.selectedSourceId);
  if (!result) return;
  const cases = resultCases(result);
  const selectedCase = state.selectedCaseBySource.get(result.source_id);
  if (!cases.some(item => item.key === selectedCase)) {
    state.selectedCaseBySource.set(result.source_id, cases[0]?.key || null);
  }
}

function queueReviewLabel(result) {
  if (!state.manualReviewEnabled) return ["Manual locked", "warning"];
  if (questionFlagged(result)) return ["Flagged", "danger"];
  if (questionComplete(result)) return ["Reviewed", "success"];
  return ["Incomplete", "warning"];
}

function renderQueue(items) {
  elements.resultCount.textContent = `${items.length} / ${state.payload.results.length}`;
  elements.queueList.replaceChildren();
  if (items.length === 0) {
    elements.queueList.appendChild(
      createElement("div", "empty-state", "조건에 맞는 질문이 없다."),
    );
    return;
  }

  items.forEach(result => {
    const matchingRun = result.inference_runs.find(
      run => runMatchesFilters(run) && runMatchesQuery(result, run),
    )
      || result.inference_runs[0];
    const button = createElement("button", "queue-item");
    button.type = "button";
    button.setAttribute("role", "option");
    button.setAttribute(
      "aria-selected",
      String(result.source_id === state.selectedSourceId),
    );
    button.tabIndex = result.source_id === state.selectedSourceId ? 0 : -1;
    button.dataset.sourceId = result.source_id;

    const idLine = createElement("span", "queue-id");
    idLine.append(
      createElement("span", "", result.source_id),
      createElement(
        "span",
        "",
        formatRanges(listOrEmpty(result.inference_measure_ranges)),
      ),
    );
    button.append(
      idLine,
      createElement(
        "span",
        "queue-preview",
        runQuestion(matchingRun) || result.paraphrased_question,
      ),
    );
    const badges = createElement("span", "badge-row");
    appendBadge(
      badges,
      STATUS_LABELS[result.review_status] || result.review_status,
      statusTone(result.review_status),
    );
    if (isSemanticPayload(state.payload)) {
      const semantic = semanticStatus(result);
      appendBadge(
        badges,
        SEMANTIC_LABELS[semantic],
        SEMANTIC_TONES[semantic],
      );
    }
    const [reviewLabel, reviewTone] = queueReviewLabel(result);
    appendBadge(badges, reviewLabel, reviewTone);
    if (generatedBasis(result).has("internal_knowledge")) {
      appendBadge(badges, "Internal", "plum");
    }
    if (isSynthesizedPayload()) {
      const generated = generatedOutput(matchingRun);
      const variantId = runVariantId(matchingRun);
      if (variantId) appendBadge(badges, variantSlot(variantId), "info");
      if (generated.generation_mode) {
        appendBadge(
          badges,
          generated.generation_mode,
          generated.generation_mode === "llm" ? "success" : "warning",
        );
      }
      const status = executionStatus(matchingRun);
      appendBadge(
        badges,
        status,
        status === "completed"
          ? "success"
          : status === "error" ? "danger" : "warning",
      );
    }
    button.appendChild(badges);
    button.addEventListener("click", () => {
      state.selectedSourceId = result.source_id;
      render();
      elements.queueList
        .querySelector('[aria-selected="true"]')
        ?.focus();
    });
    elements.queueList.appendChild(button);
  });
}

function addMetaBox(container, label, value) {
  const box = createElement("div", "meta-box");
  box.append(
    createElement("strong", "", label),
    createElement("span", "", value),
  );
  container.appendChild(box);
}

function ratingSelect(
  label,
  value,
  onChange,
  {
    allowNotApplicable = true,
    help = "",
    focusKey = "",
  } = {},
) {
  const field = createElement("label", "review-field");
  field.appendChild(createElement("span", "field-label", label));
  const select = document.createElement("select");
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Unreviewed";
  select.appendChild(blank);
  ReviewState.RATING_VALUES.forEach(rating => {
    if (!allowNotApplicable && rating === "not_applicable") return;
    const option = document.createElement("option");
    option.value = rating;
    option.textContent = RATING_LABELS[rating];
    select.appendChild(option);
  });
  select.value = value || "";
  if (focusKey) select.dataset.reviewFocus = focusKey;
  select.addEventListener("change", event => {
    onChange(event.target.value || null, focusKey);
  });
  field.append(
    select,
    createElement(
      "span",
      `field-help ${help ? "required" : ""}`.trim(),
      help,
    ),
  );
  return field;
}

function notesField(label, review, onInput, required) {
  const field = createElement("label", "review-field");
  field.appendChild(createElement("span", "field-label", label));
  const textarea = document.createElement("textarea");
  textarea.value = review.notes;
  textarea.placeholder = required
    ? "Fail 또는 uncertain 사유를 기록한다."
    : "선택 사항";
  textarea.addEventListener("input", event => {
    onInput(event.target.value, false);
  });
  field.append(
    textarea,
    createElement(
      "span",
      `field-help ${required ? "required" : ""}`.trim(),
      required ? "Fail/uncertain에는 노트가 필요하다." : "",
    ),
  );
  return field;
}

function issueTagControls(
  review,
  onChange,
  allowedTags = ReviewState.ISSUE_TAGS,
  focusPrefix = "",
) {
  const wrapper = createElement("div", "issue-tags");
  allowedTags.forEach(tag => {
    const label = createElement("label", "issue-tag");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = review.issue_tags.includes(tag);
    const focusKey = focusPrefix ? `${focusPrefix}:${tag}` : "";
    if (focusKey) input.dataset.reviewFocus = focusKey;
    input.addEventListener("change", () => {
      const tags = new Set(review.issue_tags);
      if (input.checked) tags.add(tag);
      else tags.delete(tag);
      onChange(
        ReviewState.ISSUE_TAGS.filter(item => tags.has(item)),
        focusKey,
      );
    });
    label.append(
      input,
      createElement("span", "", ISSUE_LABELS[tag] || tag),
    );
    wrapper.appendChild(label);
  });
  return wrapper;
}

function restoreReviewFocus(focusKey) {
  if (!focusKey) return;
  const target = Array.from(
    document.querySelectorAll("[data-review-focus]"),
  ).find(element => element.dataset.reviewFocus === focusKey);
  target?.focus();
}

function updateReview(mutator, message, focusKey = "") {
  if (!state.manualReviewEnabled) return;
  mutator();
  persistDraft(message);
  render();
  restoreReviewFocus(focusKey);
}

function renderManualReviewGuard(kind) {
  const card = createElement("section", "review-card review-locked");
  card.append(
    createElement("h3", "", `${kind} manual review locked`),
    createElement(
      "p",
      "review-subtitle",
      "Judge가 모든 사례를 끝내고 run 상태가 complete가 된 뒤 수동 "
      + "semantic-fidelity 평가를 시작할 수 있다. 실행 중인 체크포인트에는 "
      + "브라우저 검토 상태를 저장하지 않는다.",
    ),
  );
  return card;
}

function renderQuestionReview(result) {
  if (!state.manualReviewEnabled) {
    return renderManualReviewGuard("Question wording");
  }
  const review = questionReview(result);
  const card = createElement("section", "review-card");
  card.append(
    createElement("h3", "", "Question wording review"),
    createElement(
      "p",
      "review-subtitle",
      "패러프레이즈의 자연스러움과 원문 의미 보존을 평가한다.",
    ),
  );
  const grid = createElement("div", "review-grid question-review-grid");
  const required = reviewNeedsNote(review, ["paraphrase_natural"]);
  grid.append(
    ratingSelect(
      "Paraphrase naturalness",
      review.paraphrase_natural,
      (value, focusKey) => updateReview(() => {
        review.paraphrase_natural = value;
      }, "질문 평가 저장됨", focusKey),
      {
        allowNotApplicable: false,
        help: required ? "노트가 필요하다." : "",
        focusKey: `question:${result.source_id}:paraphrase`,
      },
    ),
    notesField(
      "Question notes",
      review,
      (value, rerender) => {
        review.notes = value;
        persistDraft("질문 노트 저장됨");
        if (rerender) render();
      },
      required,
    ),
  );
  card.appendChild(grid);
  card.appendChild(
    issueTagControls(
      review,
      (tags, focusKey) => updateReview(() => {
        review.issue_tags = tags;
      }, "질문 태그 저장됨", focusKey),
      ["awkward_paraphrase", "needs_expert_review"],
      `question:${result.source_id}:issue`,
    ),
  );
  return card;
}

function renderCaseReview(caseItem) {
  if (!state.manualReviewEnabled) {
    return renderManualReviewGuard("Semantic-fidelity");
  }
  const review = caseItem.review;
  const card = createElement("section", "review-card");
  card.append(
    createElement("h3", "", "Manual semantic-fidelity verdict"),
    createElement(
      "p",
      "review-subtitle",
      isSynthesizedPayload() && !isSemanticPayload(state.payload)
        ? "이 사례의 인간 expected/reference answer와 생성 답변 사이의 "
          + "의미 충실도를 평가한다. 검색/KU 적중은 보조 진단으로 "
          + "따로 기록한다."
        : "이 사례의 정확한 judge 권위와 생성 답변 사이의 의미 충실도를 "
          + "평가한다. 검색/KU 적중은 보조 진단으로 따로 기록한다.",
    ),
  );
  const grid = createElement("div", "review-grid");
  const required = reviewNeedsNote(
    review,
    ["range_correct", "evidence_relevant", "answer_acceptable"],
  );
  grid.append(
    ratingSelect(
      "Range correctness",
      review.range_correct,
      (value, focusKey) => updateReview(() => {
        review.range_correct = value;
      }, "범위 평가 저장됨", focusKey),
      {
        help: required ? "노트가 필요할 수 있다." : "",
        focusKey: `case:${caseItem.key}:range`,
      },
    ),
    ratingSelect(
      "Evidence relevance",
      review.evidence_relevant,
      (value, focusKey) => updateReview(() => {
        review.evidence_relevant = value;
      }, "근거 평가 저장됨", focusKey),
      {
        help: required ? "노트가 필요할 수 있다." : "",
        focusKey: `case:${caseItem.key}:evidence`,
      },
    ),
    ratingSelect(
      "Semantic fidelity",
      review.answer_acceptable,
      (value, focusKey) => updateReview(() => {
        review.answer_acceptable = value;
      }, "답변 평가 저장됨", focusKey),
      {
        help: required ? "노트가 필요할 수 있다." : "",
        focusKey: `case:${caseItem.key}:answer`,
      },
    ),
  );
  card.appendChild(grid);
  const notes = notesField(
      "Semantic-fidelity notes",
    review,
    (value, rerender) => {
      review.notes = value;
      persistDraft("추론 노트 저장됨");
      if (rerender) render();
    },
    required,
  );
  notes.style.marginTop = "9px";
  card.appendChild(notes);
  card.appendChild(
    issueTagControls(
      review,
      (tags, focusKey) => updateReview(() => {
        review.issue_tags = tags;
      }, "추론 태그 저장됨", focusKey),
      [
        "range_questionable",
        "retrieval_miss",
        "irrelevant_evidence",
        "unsupported_answer",
        "factual_error",
        "citation_error",
        "needs_expert_review",
      ],
      `case:${caseItem.key}:issue`,
    ),
  );
  return card;
}

function evidenceCard(evidence) {
  const card = createElement("article", "evidence-card");
  const identifier = evidence.id || evidence.evidence_id || "unknown evidence";
  card.append(
    createElement(
      "span",
      "evidence-id",
      `${identifier} · ${evidence.kind || evidence.evidence_type || "unknown"}`,
    ),
    createElement("p", "", evidence.text || evidence.answer || ""),
  );
  const badges = createElement("div", "badge-row");
  if (evidence.scope_match) appendBadge(badges, evidence.scope_match, "info");
  if (listOrEmpty(evidence.source_ids).length) {
    appendBadge(
      badges,
      `sources: ${evidence.source_ids.join(", ")}`,
      "info",
    );
  }
  const ranges = evidenceRanges(
    evidence.measure_ranges || evidence.measure_range,
  );
  if (ranges.length) {
    appendBadge(badges, formatRanges(ranges), "accent");
  }
  if (badges.childElementCount) card.appendChild(badges);
  return card;
}

function evidenceList(evidence) {
  const list = createElement("div", "evidence-list");
  if (!Array.isArray(evidence) || evidence.length === 0) {
    list.appendChild(
      createElement("div", "no-evidence", "검색된 근거가 없다."),
    );
    return list;
  }
  evidence.forEach(item => list.appendChild(evidenceCard(item)));
  return list;
}

function referenceReviewRecord(run) {
  const semantic = objectOrEmpty(run.semantic_evaluation);
  const direct = objectOrEmpty(semantic.reference_review);
  if (Object.keys(direct).length) return direct;
  return objectOrEmpty(semanticAggregate(run).reference_review);
}

function renderReferenceReviewNotice(run) {
  const review = referenceReviewRecord(run);
  if (!Object.keys(review).length) return null;
  const notice = createElement(
    "aside",
    `reference-review-notice ${review.required ? "required" : ""}`.trim(),
  );
  const title = review.required
    ? "Reference review required"
    : "Reference curation gate cleared";
  notice.appendChild(createElement("strong", "", title));
  if (review.required) {
    const reasons = listOrEmpty(review.reasons);
    if (reasons.length) {
      const list = document.createElement("ul");
      reasons.forEach(reason => {
        list.appendChild(createElement("li", "", String(reason)));
      });
      notice.appendChild(list);
    } else {
      notice.appendChild(
        createElement(
          "p",
          "",
          "Reference review is required, but this checkpoint does not include a reason.",
        ),
      );
    }
  }
  if (review.policy) {
    notice.appendChild(createElement("p", "", review.policy));
  }
  return notice;
}

function fallbackHumanReference(result) {
  const reference = objectOrEmpty(result?.authoritative_reference);
  const sourceAnswer = reference.source_answer || reference.answer;
  if (sourceAnswer) {
    return {
      source: "authoritative_reference.source_answer",
      items: [{ reference_id: "HUMAN", text: sourceAnswer }],
    };
  }
  const items = listOrEmpty(reference.linked_knowledge_units)
    .filter(unit => unit?.answer)
    .map((unit, index) => ({
      reference_id: unit.knowledge_unit_id || `KU-${index + 1}`,
      text: unit.answer,
    }));
  return {
    source: items.length
      ? "authoritative_reference.linked_knowledge_units"
      : null,
    items,
  };
}

function renderCaseAuthorityComparison(run, result) {
  const authority = ReviewState.caseReferenceAuthority(run);
  const isSynthesisRun = isSynthesizedPayload()
    && !isSemanticPayload(state.payload);
  const comparisonReference = authority.items.length || !isSynthesisRun
    ? authority
    : fallbackHumanReference(result);
  const generated = generatedOutput(run);
  const section = createElement("section", "authority-comparison");
  const heading = createElement("div", "comparison-heading");
  const title = createElement("div");
  title.append(
    createElement(
      "p",
      "eyebrow",
      isSynthesisRun ? "Expected vs generated" : "Primary qualitative audit",
    ),
    createElement(
      "h3",
      "",
      isSynthesisRun
        ? "Human expected answer vs generated answer"
        : "Exact case authority vs generated answer",
    ),
  );
  const badges = createElement("div", "badge-row");
  appendBadge(
    badges,
    formatRange(objectOrEmpty(run.inference_input).measure_range),
    "accent",
  );
  if (comparisonReference.items.length) {
    appendBadge(
      badges,
      isSynthesisRun
        ? `${comparisonReference.items.length} human reference item(s)`
        : `${comparisonReference.items.length} exact R item(s)`,
      "info",
    );
  } else {
    appendBadge(badges, "Exact R items pending", "warning");
  }
  heading.append(title, badges);
  section.appendChild(heading);

  const grid = createElement("div", "comparison-grid");
  const referenceColumn = createElement("article", "comparison-column authority");
  referenceColumn.appendChild(
    createElement(
      "h4",
      "",
      isSynthesisRun
        ? "Human expected/reference answer"
        : "Exact per-case judge authority",
    ),
  );
  if (comparisonReference.items.length) {
    referenceColumn.appendChild(
      createElement(
        "p",
        "comparison-source",
        `${isSynthesisRun ? "Reference source" : "Validated source"}: `
          + comparisonReference.source,
      ),
    );
    const list = createElement("div", "authority-item-list");
    comparisonReference.items.forEach(item => {
      const card = createElement("article", "authority-item");
      const itemHeading = createElement("div", "authority-item-heading");
      itemHeading.appendChild(
        createElement("strong", "", item.reference_id),
      );
      if (item.status) {
        appendBadge(
          itemHeading,
          item.status,
          assessmentTone(item.status),
        );
      }
      if (item.scope) appendBadge(itemHeading, item.scope, "info");
      if (item.knowledge_unit_id) {
        appendBadge(itemHeading, item.knowledge_unit_id, "accent");
      }
      card.append(
        itemHeading,
        createElement("p", "", item.text),
      );
      if (listOrEmpty(item.flags).length) {
        card.appendChild(
          createElement(
            "p",
            "comparison-source",
            `Reference flags: ${item.flags.join(", ")}`,
          ),
        );
      }
      list.appendChild(card);
    });
    referenceColumn.appendChild(list);
  } else {
    referenceColumn.appendChild(
      createElement(
        "div",
        "no-evidence neutral",
        "이 체크포인트에는 검증된 exact R item이 아직 없다. 아래의 global "
        + "original annotation을 이 사례의 범위별 judge 권위로 대신 사용하면 안 된다.",
      ),
    );
  }

  const candidateColumn = createElement("article", "comparison-column candidate");
  candidateColumn.appendChild(createElement("h4", "", "Generated RAG+LLM answer"));
  const candidateBadges = createElement("div", "badge-row");
  if (generated.generation_mode) {
    appendBadge(candidateBadges, generated.generation_mode, "info");
  }
  if (generated.answer_basis) {
    appendBadge(
      candidateBadges,
      generated.answer_basis,
      generated.answer_basis === "retrieved_evidence" ? "success" : "warning",
    );
  }
  const status = executionStatus(run);
  appendBadge(
    candidateBadges,
    status,
    status === "completed"
      ? "success"
      : status === "error" ? "danger" : "warning",
  );
  if (candidateBadges.childElementCount) {
    candidateColumn.appendChild(candidateBadges);
  }
  candidateColumn.appendChild(
    createElement(
      "p",
      "candidate-answer",
      generated.answer || "아직 생성된 답변이 없다.",
    ),
  );
  if (generated.generation_fallback_reason) {
    candidateColumn.appendChild(
      createElement(
        "div",
        "notice",
        `Generation note: ${generated.generation_fallback_reason}`,
      ),
    );
  }
  grid.append(referenceColumn, candidateColumn);
  section.appendChild(grid);
  const caseAuthority = objectOrEmpty(
    objectOrEmpty(run.inference_input).case_reference_authority,
  );
  if (caseAuthority.manual_review_required) {
    section.appendChild(
      createElement(
        "div",
        "notice comparison-review-warning",
        caseAuthority.manual_review_reason
          ? `Human reference review note: ${caseAuthority.manual_review_reason}`
          : "This case uses range-applicable supporting reference material "
            + "and needs human review.",
      ),
    );
  }
  const reviewNotice = renderReferenceReviewNotice(run);
  if (reviewNotice) section.appendChild(reviewNotice);
  return section;
}

function prettyDiagnosticLabel(key) {
  return String(key).replaceAll("_", " ");
}

function renderDiagnostics(title, diagnostics) {
  const section = createElement("section", "diagnostics-panel");
  section.appendChild(createElement("h4", "", title));
  const values = Object.entries(objectOrEmpty(diagnostics));
  if (!values.length) {
    section.appendChild(
      createElement("div", "no-evidence neutral", "진단 데이터가 아직 없다."),
    );
    return section;
  }
  const grid = createElement("div", "diagnostics-grid");
  values.forEach(([key, value]) => {
    const item = createElement("div", "diagnostic-item");
    let text;
    let tone = "";
    if (typeof value === "boolean") {
      text = value ? "Yes" : "No";
      tone = value ? "success" : "danger";
    } else if (Array.isArray(value)) {
      text = value.join(", ") || "None";
    } else if (value && typeof value === "object") {
      text = JSON.stringify(value);
    } else {
      text = value ?? "—";
    }
    item.append(
      createElement("span", "diagnostic-label", prettyDiagnosticLabel(key)),
      createElement("strong", `diagnostic-value ${tone}`.trim(), String(text)),
    );
    grid.appendChild(item);
  });
  section.appendChild(grid);
  return section;
}

function appendAssessmentList(container, label, values, tone = "") {
  if (!Array.isArray(values) || values.length === 0) return;
  const section = createElement(
    "div",
    `assessment-list ${tone}`.trim(),
  );
  section.appendChild(createElement("strong", "", label));
  const list = document.createElement("ul");
  values.forEach(value => {
    let text = value;
    if (value && typeof value === "object") {
      const description = value.reference_text
        || value.candidate_text
        || value.excluded_text
        || value.text
        || value.reference_point
        || value.reference_quote
        || value.description
        || value.reason
        || value.assertion_id
        || "";
      const identifier = value.reference_id
        || value.candidate_id
        || value.excluded_id
        || value.assertion_id
        || value.evidence_id
        || "";
      const classification = value.status || value.relation || "";
      const severity = value.severity || "";
      const labelParts = [identifier, classification].filter(Boolean);
      const itemLabel = labelParts.length
        ? `${labelParts.join(" · ")}: `
        : "";
      const quote = value.candidate_quote || "";
      const linkedCandidates = listOrEmpty(value.candidate_ids);
      const linkedReferences = listOrEmpty(value.reference_ids);
      const candidateTexts = listOrEmpty(value.candidate_texts);
      const missingAnchors = listOrEmpty(value.missing_anchors);
      const requiredAnchors = listOrEmpty(value.required_anchors);
      const novelAnchors = listOrEmpty(value.novel_anchors);
      const bestEntailment = value.best_entailment;
      const entailmentThreshold = value.threshold;
      const scopeAuthority = value.scope_authority || "";
      const curatorScope = value.curator_scope || "";
      const guardScope = value.guard_scope || "";
      const evidenceIds = listOrEmpty(value.evidence_ids);
      const details = [];
      if (quote) details.push(`candidate: “${quote}”`);
      if (linkedCandidates.length) {
        details.push(`candidate IDs: ${linkedCandidates.join(", ")}`);
      }
      if (linkedReferences.length) {
        details.push(`reference IDs: ${linkedReferences.join(", ")}`);
      }
      if (candidateTexts.length) {
        details.push(
          `candidate text: ${candidateTexts
            .map(item => `“${item}”`)
            .join("; ")}`,
        );
      }
      if (requiredAnchors.length) {
        details.push(`required literal anchors: ${requiredAnchors.join(", ")}`);
      }
      if (missingAnchors.length) {
        details.push(`missing literal anchors: ${missingAnchors.join(", ")}`);
      }
      if (novelAnchors.length) {
        details.push(`novel numeric anchors: ${novelAnchors.join(", ")}`);
      }
      if (typeof bestEntailment === "number") {
        details.push(`best entailment: ${bestEntailment.toFixed(4)}`);
      }
      if (typeof entailmentThreshold === "number") {
        details.push(
          `required entailment: ${entailmentThreshold.toFixed(4)}`,
        );
      }
      if (scopeAuthority) {
        details.push(`scope authority: ${scopeAuthority}`);
      }
      if (curatorScope) {
        details.push(`curator scope: ${curatorScope}`);
      }
      if (guardScope) {
        details.push(`guard scope: ${guardScope}`);
      }
      if (evidenceIds.length) {
        details.push(`retrieved expert IDs: ${evidenceIds.join(", ")}`);
      }
      if (severity) {
        details.push(`severity: ${severity}`);
      }
      if (value.support_requires_review === true) {
        details.push("supplemental support requires human review");
      }
      text = `${itemLabel}${description}`;
      if (details.length) text += ` · ${details.join(" · ")}`;
    }
    list.appendChild(createElement("li", "", String(text)));
  });
  section.appendChild(list);
  container.appendChild(section);
}

function assessmentTone(status) {
  if (
    ["contradicted", "unsupported", "mixed", "asserted", "fail"]
      .includes(status)
  ) {
    return "danger";
  }
  if (
    [
      "missing",
      "covered_guard_rejected",
      "uncertain",
      "human_review",
    ].includes(status)
  ) {
    return "warning";
  }
  if (["covered", "supported", "negated", "absent", "pass"].includes(status)) {
    return "success";
  }
  return "";
}

function appendExhaustiveAssessments(
  container,
  label,
  values,
  statusOrder,
) {
  if (!Array.isArray(values) || values.length === 0) return;
  const statuses = [
    ...statusOrder,
    ...values
      .map(value => value?.status || value?.relation || "unclassified")
      .filter(status => !statusOrder.includes(status)),
  ];
  [...new Set(statuses)].forEach(status => {
    const matching = values.filter(
      value => (value?.status || value?.relation || "unclassified") === status,
    );
    appendAssessmentList(
      container,
      `${label} · ${prettyDiagnosticLabel(status)} (${matching.length})`,
      matching,
      assessmentTone(status),
    );
  });
}

function renderJudgeFrame(frameName, frameValue) {
  const frame = objectOrEmpty(frameValue);
  const assessment = Object.keys(objectOrEmpty(frame.assessment)).length
    ? objectOrEmpty(frame.assessment)
    : frame;
  const computed = objectOrEmpty(frame.computed);
  const card = createElement("article", "judge-frame");
  const heading = createElement("div", "judge-frame-heading");
  heading.appendChild(
    createElement("h4", "", prettyDiagnosticLabel(frameName)),
  );
  const badges = createElement("div", "badge-row");
  appendBadge(badges, frame.status || "pending");
  if (assessment.relationship) {
    appendBadge(badges, assessment.relationship, "info");
  }
  const frameScore = computed.weighted_score ?? frame.score;
  if (typeof frameScore === "number") {
    appendBadge(
      badges,
      `${frameScore.toFixed(1)} / 100`,
      computed.pass ? "success" : "danger",
    );
  }
  if (typeof assessment.confidence === "number") {
    appendBadge(badges, `confidence ${assessment.confidence.toFixed(2)}`);
  }
  heading.appendChild(badges);
  card.appendChild(heading);
  const scores = objectOrEmpty(assessment.scores);
  if (Object.keys(scores).length) {
    const scoreGrid = createElement("div", "judge-score-grid");
    Object.entries(scores).forEach(([name, value]) => {
      if (value === null || value === undefined) return;
      const score = createElement("div", "judge-score");
      score.append(
        createElement("span", "", prettyDiagnosticLabel(name)),
        createElement(
          "strong",
          value >= 3 ? "success-text" : "danger-text",
          `${value} / 4`,
        ),
      );
      scoreGrid.appendChild(score);
    });
    card.appendChild(scoreGrid);
  }
  if (assessment.rationale) {
    card.append(
      createElement("strong", "assessment-label", "Judge rationale"),
      createElement("p", "judge-rationale", assessment.rationale),
    );
  }
  const guardedReferenceIds = new Set(
    [
      ...listOrEmpty(assessment.literal_anchor_coverage_warnings),
      ...listOrEmpty(assessment.atomic_claim_entailment_warnings),
    ]
      .map(item => item?.reference_id)
      .filter(Boolean),
  );
  const displayedReferenceAssessments = listOrEmpty(
    assessment.reference_assessments,
  ).map(item => (
    guardedReferenceIds.has(item?.reference_id)
      ? { ...item, status: "covered_guard_rejected" }
      : item
  ));
  const targetReferenceAssessments = displayedReferenceAssessments.filter(
    item => item?.scope_authority !== "retrieved_expert_factuality_only",
  );
  const supplementalReferenceAssessments = displayedReferenceAssessments.filter(
    item => item?.scope_authority === "retrieved_expert_factuality_only",
  );
  appendExhaustiveAssessments(
    card,
    "Target R claims — completeness",
    targetReferenceAssessments,
    [
      "contradicted",
      "missing",
      "covered_guard_rejected",
      "covered",
      "not_required",
    ],
  );
  appendExhaustiveAssessments(
    card,
    "Retrieved S claims — factuality only",
    supplementalReferenceAssessments,
    [
      "contradicted",
      "covered_guard_rejected",
      "covered",
      "missing",
      "not_required",
    ],
  );
  appendExhaustiveAssessments(
    card,
    "Candidate segments (exhaustive)",
    assessment.candidate_assessments,
    ["contradicted", "mixed", "unsupported", "supported", "irrelevant"],
  );
  appendAssessmentList(
    card,
    "Material omissions",
    assessment.missing_material_points,
    "warning",
  );
  appendAssessmentList(
    card,
    "Contradictions",
    assessment.contradictions,
    "danger",
  );
  appendAssessmentList(
    card,
    "Unsupported material claims",
    assessment.unsupported_material_claims,
    "danger",
  );
  appendAssessmentList(
    card,
    "Critical errors",
    assessment.critical_error_types,
    "danger",
  );
  appendAssessmentList(
    card,
    "Unanchored critical-error claims (manual review)",
    assessment.unanchored_critical_error_warnings,
    "warning",
  );
  appendAssessmentList(
    card,
    "Relationship labels returned as critical errors (manual review)",
    assessment.misplaced_relationship_error_warnings,
    "warning",
  );
  appendAssessmentList(
    card,
    "Covered reference points",
    assessment.covered_reference_points,
    "success",
  );
  appendAssessmentList(
    card,
    "Frame failure reasons",
    computed.failure_reasons || frame.reasons,
    "warning",
  );
  appendAssessmentList(
    card,
    "Literal pronunciation-anchor guard (raw coverage not accepted)",
    assessment.literal_anchor_coverage_warnings,
    "warning",
  );
  appendAssessmentList(
    card,
    "Required-claim NLI guard (atomic or S-substitution coverage not corroborated)",
    assessment.atomic_claim_entailment_warnings,
    "warning",
  );
  appendAssessmentList(
    card,
    "Novel numeric-fact guard",
    assessment.novel_numeric_fact_warnings,
    "danger",
  );
  appendAssessmentList(
    card,
    "Deterministic consistency corrections",
    assessment.deterministic_normalizations,
    "warning",
  );
  return card;
}

function normalizeRangeClaimItems(value, prefix) {
  return listOrEmpty(value)
    .map((item, index) => {
      if (typeof item === "string" && item.trim()) {
        return {
          reference_id: `${prefix}${String(index + 1).padStart(3, "0")}`,
          text: item.trim(),
        };
      }
      if (!item || typeof item !== "object") return null;
      const text = item.text
        || item.reference_text
        || item.excluded_text
        || item.claim;
      if (typeof text !== "string" || !text.trim()) return null;
      return {
        reference_id: item.reference_id
          || item.excluded_id
          || `${prefix}${String(index + 1).padStart(3, "0")}`,
        text: text.trim(),
      };
    })
    .filter(Boolean);
}

function renderRangeClaimInventory(allowed, excluded) {
  if (!allowed.length && !excluded.length) return null;
  const section = createElement("section", "range-claim-inventory");
  section.appendChild(
    createElement("h4", "", "Reviewed atomic range contrast"),
  );
  const grid = createElement("div", "range-claim-grid");
  [
    ["A · selected-range claims", allowed, "allowed"],
    ["X · other-range claims", excluded, "excluded"],
  ].forEach(([label, items, tone]) => {
    const column = createElement("div", `range-claim-column ${tone}`);
    column.appendChild(createElement("strong", "", label));
    if (!items.length) {
      column.appendChild(createElement("p", "", "None"));
    } else {
      items.forEach(item => {
        const claim = createElement("article", "range-claim");
        claim.append(
          createElement("strong", "", item.reference_id),
          createElement("p", "", item.text),
        );
        column.appendChild(claim);
      });
    }
    grid.appendChild(column);
  });
  section.appendChild(grid);
  return section;
}

function formatProbability(value) {
  return typeof value === "number"
    ? `${(value * 100).toFixed(2)}%`
    : null;
}

function formatSigned(value) {
  if (typeof value !== "number") return null;
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
}

function renderHybridRangeAssessment(item) {
  const card = createElement("article", "hybrid-range-assessment");
  const heading = createElement("div", "hybrid-range-heading");
  const id = item.excluded_id || item.reference_id || "X?";
  heading.appendChild(createElement("strong", "", id));
  const badges = createElement("div", "badge-row");
  const finalRelation = item.relation || item.status;
  const rawLlmRelation = item.llm_relation || item.raw_llm_relation;
  if (finalRelation) {
    appendBadge(
      badges,
      `final: ${finalRelation}`,
      assessmentTone(finalRelation),
    );
  }
  if (rawLlmRelation) {
    appendBadge(badges, `raw LLM: ${rawLlmRelation}`, "info");
  }
  if (item.candidate_link_provenance) {
    appendBadge(
      badges,
      `links: ${item.candidate_link_provenance}`,
      "info",
    );
  }
  if (item.llm_duplicate_conflict) {
    appendBadge(badges, "conflicting raw X rows", "warning");
  }
  heading.appendChild(badges);
  card.appendChild(heading);
  const text = item.excluded_text || item.reference_text || item.text;
  if (text) card.appendChild(createElement("p", "hybrid-claim-text", text));

  const modelScores = objectOrEmpty(item.model_scores);
  const nli = Object.keys(objectOrEmpty(modelScores.nli)).length
    ? objectOrEmpty(modelScores.nli)
    : objectOrEmpty(item.nli_probabilities || item.nli);
  const embedding = Object.keys(objectOrEmpty(modelScores.embedding)).length
    ? objectOrEmpty(modelScores.embedding)
    : objectOrEmpty(item.embedding_scores || item.embedding);
  const metricValues = [
    ["NLI entailment", formatProbability(nli.entailment)],
    ["NLI neutral", formatProbability(nli.neutral)],
    ["NLI contradiction", formatProbability(nli.contradiction)],
    [
      "Embedding contrast delta",
      formatSigned(
        embedding.contrast_delta
          ?? item.embedding_delta
          ?? item.embedding_contrast_delta,
      ),
    ],
    [
      "Candidate ↔ X",
      formatSigned(embedding.candidate_vs_excluded),
    ],
    [
      `Candidate ↔ best A${embedding.best_allowed_id
        ? ` (${embedding.best_allowed_id})`
        : ""}`,
      formatSigned(embedding.candidate_vs_best_allowed),
    ],
  ].filter(([, value]) => value !== null);
  if (metricValues.length) {
    const metrics = createElement("div", "hybrid-metric-grid");
    metricValues.forEach(([label, value]) => {
      const metric = createElement("div", "hybrid-metric");
      metric.append(
        createElement("span", "", label),
        createElement("strong", "", value),
      );
      metrics.appendChild(metric);
    });
    card.appendChild(metrics);
  }

  const allowedSimilarities = listOrEmpty(embedding.allowed_similarities);
  if (allowedSimilarities.length) {
    appendAssessmentList(
      card,
      "Embedding similarity to each A claim",
      allowedSimilarities.map(value => {
        const score = typeof value.cosine_similarity === "number"
          ? value.cosine_similarity.toFixed(4)
          : "—";
        return `${value.allowed_id || "A?"}: ${score}`;
      }),
    );
  }
  const candidateNli = listOrEmpty(modelScores.candidate_nli);
  if (candidateNli.length) {
    appendAssessmentList(
      card,
      "Clause-level NLI localization",
      candidateNli.map(value => (
        `${value.candidate_id || "C?"}: `
        + `entail ${formatProbability(value.entailment) || "—"}, `
        + `neutral ${formatProbability(value.neutral) || "—"}, `
        + `contradict ${formatProbability(value.contradiction) || "—"}`
      )),
    );
  }
  appendAssessmentList(
    card,
    "Decision reasons",
    item.decision_reasons || item.reasons,
    finalRelation === "asserted" ? "danger" : "warning",
  );
  appendAssessmentList(
    card,
    "Raw range-judge observations",
    item.llm_raw_observations,
    item.llm_duplicate_conflict ? "warning" : "",
  );
  const candidateTexts = listOrEmpty(
    item.candidate_texts || item.llm_candidate_texts,
  );
  appendAssessmentList(
    card,
    "Candidate segments linked by range judge",
    candidateTexts,
  );
  return card;
}

function renderRangeScopeEvaluation(semantic, aggregate, run) {
  const current = objectOrEmpty(semantic.range_scope);
  const currentAssessment = objectOrEmpty(current.assessment);
  const aggregateAssessment = objectOrEmpty(
    aggregate.range_scope_evaluation,
  );
  const legacy = objectOrEmpty(
    semantic.range_scope_guard || aggregate.range_scope_guard,
  );
  const assessment = Object.keys(currentAssessment).length
    ? currentAssessment
    : Object.keys(aggregateAssessment).length
      ? aggregateAssessment
      : legacy;
  const requirement = objectOrEmpty(current.requirement);
  if (
    !Object.keys(current).length
    && !Object.keys(assessment).length
    && !Object.keys(requirement).length
  ) {
    return null;
  }

  const card = createElement("article", "judge-frame");
  const heading = createElement("div", "judge-frame-heading");
  heading.appendChild(
    createElement("h4", "", "Range-scope NLI (independent contrast check)"),
  );
  const badges = createElement("div", "badge-row");
  const executionStatus = current.status || "complete";
  const decisionStatus = assessment.status
    || (
      executionStatus === "complete"
        ? "pending"
        : executionStatus
    );
  appendBadge(badges, `frame: ${executionStatus}`);
  appendBadge(
    badges,
    `decision: ${decisionStatus}`,
    assessmentTone(decisionStatus),
  );
  if (typeof assessment.confidence === "number") {
    appendBadge(
      badges,
      `confidence ${assessment.confidence.toFixed(2)}`,
    );
  }
  if (assessment.wrong_measure_application === true) {
    appendBadge(badges, "Wrong-range claim asserted", "danger");
  }
  heading.appendChild(badges);
  card.appendChild(heading);

  const input = objectOrEmpty(run?.inference_input);
  const allowedClaims = normalizeRangeClaimItems(
    input.allowed_range_contrast_claims
      || requirement.allowed_range_contrast_items,
    "A",
  );
  const excludedClaims = normalizeRangeClaimItems(
    requirement.excluded_reference_items?.length
      ? requirement.excluded_reference_items
      : input.excluded_range_contrast_claims,
    "X",
  );
  const inventory = renderRangeClaimInventory(allowedClaims, excludedClaims);
  if (inventory) card.appendChild(inventory);

  const guardPolicy = objectOrEmpty(
    assessment.guard_policy
      || current.guard_policy
      || aggregateAssessment.guard_policy,
  );
  const guardThresholds = objectOrEmpty(
    guardPolicy.thresholds
      || assessment.guard_thresholds
      || current.guard_thresholds,
  );
  const thresholdDiagnostics = {
    protocol_version: guardPolicy.version || guardPolicy.protocol,
    nli_threshold:
      guardPolicy.nli_threshold
      ?? guardThresholds.nli_entailment_or_contradiction
      ?? guardThresholds.nli_threshold,
    embedding_delta_threshold:
      guardPolicy.embedding_delta_threshold
      ?? guardThresholds.embedding_contrast_delta
      ?? guardThresholds.embedding_delta_threshold,
  };
  if (
    Object.values(thresholdDiagnostics).some(
      value => value !== undefined && value !== null,
    )
  ) {
    card.appendChild(
      renderDiagnostics(
        "Hybrid guard thresholds",
        Object.fromEntries(
          Object.entries(thresholdDiagnostics).filter(
            ([, value]) => value !== undefined && value !== null,
          ),
        ),
      ),
    );
  }

  if (assessment.rationale) {
    card.append(
      createElement("strong", "assessment-label", "Range judge rationale"),
      createElement("p", "judge-rationale", assessment.rationale),
    );
  } else if (decisionStatus === "not_applicable") {
    card.appendChild(
      createElement(
        "div",
        "no-evidence neutral",
        "이 사례에는 비교할 다른 범위 또는 분리 단위의 명시적 주장이 없다.",
      ),
    );
  } else if (executionStatus !== "complete") {
    card.appendChild(
      createElement(
        "div",
        "no-evidence neutral",
        "다른 범위의 주장을 현재 답변에 잘못 적용했는지 판정하는 별도 NLI가 아직 완료되지 않았다.",
      ),
    );
  }

  const excludedAssessments = listOrEmpty(
    assessment.excluded_claim_assessments,
  );
  if (excludedAssessments.length) {
    const assessmentSection = createElement(
      "section",
      "hybrid-range-assessment-list",
    );
    assessmentSection.appendChild(
      createElement("h4", "", "Hybrid X-claim decisions"),
    );
    excludedAssessments.forEach(item => {
      assessmentSection.appendChild(renderHybridRangeAssessment(item));
    });
    card.appendChild(assessmentSection);
  }
  if (
    !excludedAssessments.length
    && listOrEmpty(requirement.excluded_reference_items).length
  ) {
    appendAssessmentList(
      card,
      "Excluded claims awaiting range-scope classification",
      requirement.excluded_reference_items,
      "warning",
    );
  }

  // Preserve schema 2–4 deterministic range-guard diagnostics.
  appendAssessmentList(
    card,
    "Positive other-range assertions",
    assessment.matched_forbidden_assertions,
    assessmentTone(decisionStatus),
  );
  appendAssessmentList(
    card,
    "Case-range-only concepts found in candidate",
    assessment.candidate_allowed_only_matches,
    "success",
  );
  appendAssessmentList(
    card,
    "Range deterministic parser/consistency normalizations",
    assessment.deterministic_normalizations,
    "warning",
  );
  if (requirement.reason) {
    appendAssessmentList(
      card,
      "Range-scope requirement",
      [requirement.reason],
    );
  }
  return card;
}

function renderSemanticEvaluation(run) {
  const semantic = objectOrEmpty(run.semantic_evaluation);
  const aggregate = semanticAggregate(run);
  const section = createElement("section", "semantic-panel");
  const heading = createElement("div", "semantic-heading");
  const title = createElement("div");
  title.append(
    createElement("p", "eyebrow", "Expert-claim fidelity evaluation"),
    createElement("h3", "", "Generated answer compared with original expert claims"),
  );
  const status = ReviewState.semanticCaseStatus(run);
  const badges = createElement("div", "badge-row");
  appendBadge(
    badges,
    SEMANTIC_LABELS[status],
    SEMANTIC_TONES[status],
  );
  if (aggregate.answer_quality_status) {
    appendBadge(
      badges,
      `answer quality: ${aggregate.answer_quality_status}`,
      assessmentTone(aggregate.answer_quality_status),
    );
  }
  const aggregateScore = aggregate.weighted_score ?? aggregate.score;
  if (typeof aggregateScore === "number") {
    appendBadge(badges, `${aggregateScore.toFixed(1)} / 100`, "info");
  }
  if (aggregate.disagreement === true) {
    appendBadge(badges, "Frame disagreement", "warning");
  }
  if (semantic.reference_authority) {
    appendBadge(
      badges,
      `authority: ${prettyDiagnosticLabel(semantic.reference_authority)}`,
      "info",
    );
  }
  heading.append(title, badges);
  section.appendChild(heading);
  if (!Object.keys(aggregate).length) {
    section.appendChild(
      createElement(
        "div",
        "no-evidence neutral",
        "두 판정 프레임의 결과가 아직 완성되지 않았다.",
      ),
    );
  } else {
    appendAssessmentList(
      section,
      "Reliability failure reasons",
      aggregate.reliability_failure_reasons || aggregate.reasons,
      "danger",
    );
    appendAssessmentList(
      section,
      "Judge disagreement reasons",
      aggregate.disagreement_reasons,
      "warning",
    );
    appendAssessmentList(
      section,
      "Weak-confidence frames",
      aggregate.weak_confidence_frames,
      "warning",
    );
  }
  const supplementalValidation = objectOrEmpty(
    semantic.supplemental_evidence_validation
      || aggregate.supplemental_evidence_validation,
  );
  if (Object.keys(supplementalValidation).length) {
    appendAssessmentList(
      section,
      "Authenticated retrieved expert evidence",
      listOrEmpty(supplementalValidation.accepted_evidence)
        .map(item => item?.evidence_id)
        .filter(Boolean),
      "success",
    );
    appendAssessmentList(
      section,
      "Rejected supplemental evidence",
      supplementalValidation.rejected_evidence,
      "warning",
    );
    if (supplementalValidation.auto_pass_eligible === false) {
      section.appendChild(
        createElement(
          "div",
          "notice",
          "Supplemental-evidence integrity prevents automatic pass.",
        ),
      );
    }
  }
  const reviewDisclosureValidation = objectOrEmpty(
    semantic.review_disclosure_validation
      || aggregate.review_disclosure_validation,
  );
  if (Object.keys(reviewDisclosureValidation).length) {
    const displayFlag = value => (
      typeof value === "boolean" ? (value ? "Yes" : "No") : "—"
    );
    section.appendChild(
      renderDiagnostics(
        "Review-disclosure authentication audit",
        {
          status: reviewDisclosureValidation.status,
          evidence_ids: listOrEmpty(reviewDisclosureValidation.evidence_ids),
          expected_text: reviewDisclosureValidation.expected_text,
          stripped_from_semantic_candidate: displayFlag(
            reviewDisclosureValidation.stripped_from_semantic_candidate,
          ),
          auto_pass_eligible: displayFlag(
            reviewDisclosureValidation.auto_pass_eligible,
          ),
          failure_reason: reviewDisclosureValidation.failure_reason,
          semantic_candidate_empty: displayFlag(
            reviewDisclosureValidation.semantic_candidate_empty,
          ),
          additional_disclosure_remains: displayFlag(
            reviewDisclosureValidation.additional_disclosure_remains,
          ),
        },
      ),
    );
    section.appendChild(
      createElement(
        "div",
        "notice",
        "An authenticated review disclosure is evaluator metadata, not a "
          + "musical claim, and is excluded from musical-answer judging.",
      ),
    );
    if (reviewDisclosureValidation.auto_pass_eligible === false) {
      section.appendChild(
        createElement(
          "div",
          "notice",
          "Review-disclosure authentication prevents automatic pass.",
        ),
      );
    }
  }
  const frames = objectOrEmpty(semantic.frames);
  const frameGrid = createElement("div", "judge-frame-grid");
  ["claim_alignment", "contradiction_first"].forEach(name => {
    frameGrid.appendChild(renderJudgeFrame(name, frames[name]));
  });
  section.appendChild(frameGrid);
  const rangeScope = renderRangeScopeEvaluation(semantic, aggregate, run);
  if (rangeScope) section.appendChild(rangeScope);
  return section;
}

function renderPhaseErrors(errors) {
  const entries = Object.entries(objectOrEmpty(errors));
  if (!entries.length) return null;
  const notice = createElement("section", "phase-errors");
  notice.appendChild(createElement("h4", "", "Pipeline phase errors"));
  entries.forEach(([phase, error]) => {
    const value = objectOrEmpty(error);
    notice.appendChild(
      createElement(
        "p",
        "",
        `${phase}: ${value.type || "Error"} · ${value.message || "Unknown error"}`,
      ),
    );
  });
  return notice;
}

function renderPipeline(caseItem, result) {
  const run = caseItem.run;
  const retrieval = retrievalOutput(run);
  const generated = generatedOutput(run);
  const details = document.createElement("details");
  details.className = "secondary-details retrieval-details";
  details.appendChild(
    createElement(
      "summary",
      "",
      "Secondary retrieval, KU, evidence, and grounding diagnostics",
    ),
  );
  const input = objectOrEmpty(run.inference_input);
  const kuMetadata = createElement("div", "metadata-strip secondary-metadata");
  addMetaBox(
    kuMetadata,
    "Range source KUs",
    listOrEmpty(input.range_source_knowledge_unit_ids).join(", ") || "None",
  );
  addMetaBox(
    kuMetadata,
    "Expected KUs in this case",
    (
      input.expected_retrieval_eligible_knowledge_unit_ids
      || listOrEmpty(result.expected_retrieval_eligible_knowledge_unit_ids)
    ).join(", ") || "None",
  );
  addMetaBox(
    kuMetadata,
    "Range-applicable KUs",
    listOrEmpty(input.range_applicable_knowledge_unit_ids).join(", ")
      || "None",
  );
  addMetaBox(
    kuMetadata,
    "All linked KUs",
    listOrEmpty(result.knowledge_unit_ids).join(", ") || "None",
  );
  details.appendChild(kuMetadata);
  const grid = createElement("div", "pipeline-grid");

  const retrievalColumn = createElement("section", "pipeline-column");
  retrievalColumn.appendChild(
    createElement("h4", "", "Retrieval diagnostics (secondary)"),
  );
  const retrievalEvidence = listOrEmpty(retrieval.evidence);
  const retrievalStatus = createElement(
    "div",
    `pipeline-status ${retrievalEvidence.length ? "success" : "danger"}`,
  );
  retrievalStatus.append(
    createElement(
      "strong",
      "",
      retrieval.answer_basis || "Retrieval pending",
    ),
    createElement(
      "span",
      "",
      `${retrievalEvidence.length} evidence item(s) · ${retrieval.scope || "—"}`,
    ),
  );
  retrievalColumn.append(
    retrievalStatus,
    renderDiagnostics(
      "Expected-source / KU diagnostics",
      retrieval.diagnostics
        || objectOrEmpty(run.diagnostics).retrieval
        || run.diagnostics,
    ),
    evidenceList(retrievalEvidence),
  );

  const generationColumn = createElement("section", "pipeline-column");
  generationColumn.appendChild(
    createElement("h4", "", "Generation grounding diagnostics (secondary)"),
  );
  const generatedEvidence = listOrEmpty(generated.evidence);
  const grounded = generated.rag_llm_succeeded === true
    || (
      generated.generation_mode === "llm"
      && generated.answer_basis === "retrieved_evidence"
      && generatedEvidence.length > 0
    );
  const expertGrounded = listOrEmpty(generated.expert_evidence_ids).length > 0;
  let groundingLabel = generated.answer
    ? "No retrieved grounding"
    : "Generation pending";
  if (expertGrounded) groundingLabel = "Expert annotation-grounded";
  else if (grounded) groundingLabel = "Retrieved evidence-grounded";
  const generationStatus = createElement(
    "div",
    `pipeline-status ${grounded ? "success" : "danger"}`,
  );
  generationStatus.append(
    createElement(
      "strong",
      "",
      groundingLabel,
    ),
    createElement(
      "span",
      "",
      `${generated.generation_mode || "—"} · ${generated.answer_basis || "—"}`,
    ),
  );
  generationColumn.appendChild(generationStatus);
  generationColumn.appendChild(
    renderDiagnostics(
      "Generation grounding diagnostics",
      generated.diagnostics || objectOrEmpty(run.diagnostics).generation,
    ),
  );
  generationColumn.appendChild(createElement("h4", "", "Supplied evidence"));
  generationColumn.appendChild(evidenceList(generatedEvidence));

  const expected = (
    run.inference_input.expected_retrieval_eligible_knowledge_unit_ids
    || listOrEmpty(result.expected_retrieval_eligible_knowledge_unit_ids)
  );
  if (
    expected.length
    && generated.answer
    && !generated.expected_expert_evidence_retrieved
  ) {
    const missing = createElement("div", "notice");
    missing.textContent =
      `Expected annotation miss: ${expected.join(", ")}`;
    generationColumn.appendChild(missing);
  }

  grid.append(retrievalColumn, generationColumn);
  details.appendChild(grid);
  return details;
}

function renderCasePanel(result, caseItem, tabId, panelId) {
  const panel = createElement("section", "case-panel");
  panel.setAttribute("role", "tabpanel");
  panel.id = panelId;
  panel.setAttribute("aria-labelledby", tabId);
  const input = caseItem.run.inference_input;
  const heading = createElement("div", "case-heading");
  const headingCopy = createElement("div");
  headingCopy.append(
    createElement("h3", "", formatRange(input.measure_range)),
    createElement(
      "p",
      "",
      input.measure_range_provenance
        || "No measure range applied to this inference",
    ),
  );
  const caseMetadata = createElement("div", "metadata-strip");
  addMetaBox(
    caseMetadata,
    "Case ID",
    caseItem.run.case_id || caseItem.key,
  );
  addMetaBox(
    caseMetadata,
    "Selected inference range",
    formatRange(input.measure_range),
  );
  addMetaBox(
    caseMetadata,
    "Range context applied",
    input.measure_range_applied ? "Yes" : "No",
  );
  if (runVariantId(caseItem.run)) {
    addMetaBox(
      caseMetadata,
      "Synthesized variant",
      runVariantId(caseItem.run),
    );
  }
  addMetaBox(caseMetadata, "Run status", executionStatus(caseItem.run));
  const badges = createElement("div", "badge-row");
  const generated = generatedOutput(caseItem.run);
  const caseSemanticStatus = ReviewState.semanticCaseStatus(caseItem.run);
  appendBadge(
    badges,
    generated.answer_basis || "Generation pending",
    generated.rag_llm_succeeded ? "success" : "plum",
  );
  if (isSemanticPayload(state.payload)) {
    appendBadge(
      badges,
      SEMANTIC_LABELS[caseSemanticStatus],
      SEMANTIC_TONES[caseSemanticStatus],
    );
  }
  if (state.manualReviewEnabled) {
    appendBadge(
      badges,
      ReviewState.isCaseComplete(caseItem.review) ? "Reviewed" : "Incomplete",
      ReviewState.isCaseComplete(caseItem.review) ? "success" : "warning",
    );
  } else {
    appendBadge(badges, "Manual semantic fidelity locked", "warning");
  }
  heading.append(headingCopy, badges);
  panel.append(heading, caseMetadata);
  if (runQuestion(caseItem.run)) {
    const question = createElement("article", "case-question-card");
    question.append(
      createElement(
        "p",
        "content-label",
        runVariantId(caseItem.run)
          ? "Selected synthesized inference question"
          : "Inference question",
      ),
      createElement("p", "question-text", runQuestion(caseItem.run)),
    );
    panel.appendChild(question);
  }
  panel.appendChild(renderCaseAuthorityComparison(caseItem.run, result));
  if (caseItem.run.error) {
    const runError = objectOrEmpty(caseItem.run.error);
    panel.appendChild(
      createElement(
        "div",
        "notice run-error-notice",
        `${runError.type || "InferenceError"}: `
          + (runError.message || String(caseItem.run.error)),
      ),
    );
  }
  if (
    isSemanticPayload(state.payload)
    || caseItem.run.semantic_evaluation
  ) {
    panel.appendChild(renderSemanticEvaluation(caseItem.run));
  }
  panel.append(
    renderCaseReview(caseItem),
    renderPipeline(caseItem, result),
  );
  const phaseErrors = renderPhaseErrors(caseItem.run.phase_errors);
  if (phaseErrors) panel.appendChild(phaseErrors);
  const raw = document.createElement("details");
  raw.className = "raw-details";
  raw.append(
    createElement("summary", "", "이 추론의 원시 JSON 보기"),
    createElement(
      "pre",
      "",
      JSON.stringify(
        {
          case_id: caseItem.run.case_id,
          status: caseItem.run.status,
          inference_input: caseItem.run.inference_input,
          retrieval_probe: caseItem.run.retrieval_probe,
          generated_answer: caseItem.run.generated_answer,
          semantic_evaluation: caseItem.run.semantic_evaluation,
          phase_errors: caseItem.run.phase_errors,
          error: caseItem.run.error,
        },
        null,
        2,
      ),
    ),
  );
  panel.appendChild(raw);
  return panel;
}

function renderAuthoritativeReference(result) {
  const reference = objectOrEmpty(result.authoritative_reference);
  const sourceAnswer = reference.source_answer || reference.answer;
  if (!sourceAnswer) return null;
  const section = createElement("section", "reference-panel");
  const heading = createElement("div", "reference-heading");
  const copy = createElement("div");
  copy.append(
    createElement("p", "eyebrow", "Source lineage · global, not case-scoped"),
    createElement("h3", "", "Original annotation (global source answer)"),
  );
  const badges = createElement("div", "badge-row");
  if (reference.curation_status) {
    appendBadge(badges, reference.curation_status, "info");
  }
  const sourceHintRanges = evidenceRanges(
    reference.source_legacy_measure_range_hints,
  );
  if (sourceHintRanges.length) {
    appendBadge(
      badges,
      `legacy source hint: ${formatRanges(sourceHintRanges)}`,
      "warning",
    );
  }
  heading.append(copy, badges);
  section.append(
    heading,
    createElement("p", "reference-answer", sourceAnswer),
    createElement(
      "div",
      "notice source-authority-notice",
      "이 원문 annotation은 출처 계보를 보여준다. 선택 범위에 따라 judge가 "
      + "사용한 exact per-case R items와 다를 수 있으므로, 생성 답변 평가는 "
      + "각 inference case의 primary comparison을 기준으로 한다.",
    ),
  );
  if (reference.curation_notes) {
    section.appendChild(
      createElement(
        "div",
        "notice",
        `Source curation note: ${reference.curation_notes}`,
      ),
    );
  }

  const units = listOrEmpty(reference.linked_knowledge_units);
  if (units.length) {
    const details = document.createElement("details");
    details.className = "secondary-details linked-unit-details";
    details.appendChild(
      createElement(
        "summary",
        "",
        `Secondary linked-KU lineage diagnostics · ${units.length}`,
      ),
    );
    const unitList = createElement("div", "linked-unit-list");
    units.forEach(unit => {
      const card = createElement("article", "linked-unit-card");
      const unitHeading = createElement("div", "linked-unit-heading");
      unitHeading.appendChild(
        createElement("strong", "", unit.knowledge_unit_id || "Unknown KU"),
      );
      const unitBadges = createElement("div", "badge-row");
      if (unit.rewrite_status) appendBadge(unitBadges, unit.rewrite_status);
      if (unit.measure_status) {
        appendBadge(
          unitBadges,
          unit.measure_status,
          unit.measure_status === "specific" ? "accent" : "warning",
        );
      }
      const ranges = evidenceRanges(unit.measure_ranges);
      if (ranges.length) appendBadge(unitBadges, formatRanges(ranges), "accent");
      unitHeading.appendChild(unitBadges);
      card.append(
        unitHeading,
        createElement("p", "", unit.answer || ""),
      );
      if (unit.rewrite_notes || unit.measure_notes) {
        card.appendChild(
          createElement(
            "p",
            "unit-notes",
            [unit.rewrite_notes, unit.measure_notes].filter(Boolean).join(" · "),
          ),
        );
      }
      const contributors = listOrEmpty(unit.contributing_source_answers);
      if (contributors.length) {
        const details = document.createElement("details");
        details.className = "contributor-details";
        details.appendChild(
          createElement(
            "summary",
            "",
            `Contributing source answers · ${contributors.length}`,
          ),
        );
        contributors.forEach(contributor => {
          const block = createElement("div", "contributor-answer");
          block.append(
            createElement(
              "strong",
              "",
              contributor.source_id || "Unknown source",
            ),
          );
          const hintRanges = evidenceRanges(
            contributor.legacy_measure_range_hints,
          );
          if (hintRanges.length) {
            const hintBadges = createElement("div", "badge-row");
            appendBadge(
              hintBadges,
              `legacy source hint: ${formatRanges(hintRanges)}`,
              "warning",
            );
            block.appendChild(hintBadges);
          }
          if (contributor.question) {
            block.appendChild(
              createElement(
                "p",
                "unit-notes",
                `Original source question: ${contributor.question}`,
              ),
            );
          }
          block.appendChild(
            createElement("p", "", contributor.answer || ""),
          );
          details.appendChild(block);
        });
        card.appendChild(details);
      }
      unitList.appendChild(card);
    });
    details.appendChild(unitList);
    section.appendChild(details);
  }
  return section;
}

function renderDetail(result) {
  elements.detail.replaceChildren();
  if (!result) {
    elements.detail.appendChild(
      createElement("div", "empty-state", "검토할 질문을 선택한다."),
    );
    return;
  }

  const heading = createElement("div", "detail-heading");
  const copy = createElement("div");
  copy.append(
    createElement("p", "eyebrow", PIECE_LABELS[result.piece_id] || result.piece_id),
    createElement("h2", "", result.source_id),
    createElement(
      "p",
      "detail-path",
      `${isSemanticPayload(state.payload)
        ? "semantic evaluation snapshot"
        : isSynthesizedPayload()
          ? "synthesized hybrid RAG+LLM snapshot"
          : "legacy manual-check snapshot"} · ${result.annotator}`,
    ),
  );
  const badges = createElement("div", "badge-row");
  appendBadge(
    badges,
    STATUS_LABELS[result.review_status] || result.review_status,
    statusTone(result.review_status),
  );
  if (isSemanticPayload(state.payload)) {
    const semantic = semanticStatus(result);
    appendBadge(
      badges,
      SEMANTIC_LABELS[semantic],
      SEMANTIC_TONES[semantic],
    );
  }
  const [reviewLabel, reviewTone] = queueReviewLabel(result);
  appendBadge(badges, reviewLabel, reviewTone);
  if (state.manualReviewEnabled && questionFlagged(result)) {
    appendBadge(badges, "Manual flag", "danger");
  }
  heading.append(copy, badges);
  elements.detail.appendChild(heading);

  const questions = createElement("div", "question-grid");
  const original = createElement("article", "content-card");
  original.append(
    createElement("p", "content-label", "Original annotator question"),
    createElement("p", "question-text", result.original_question),
  );
  const paraphrase = createElement("article", "content-card highlight");
  paraphrase.append(
    createElement("p", "content-label", "Paraphrased inference question"),
    createElement("p", "question-text", result.paraphrased_question),
  );
  questions.append(original, paraphrase);
  elements.detail.appendChild(questions);

  const metadata = createElement("div", "metadata-strip");
  addMetaBox(metadata, "Piece", PIECE_LABELS[result.piece_id] || result.piece_id);
  addMetaBox(metadata, "Applied ranges", formatRanges(result.inference_measure_ranges));
  addMetaBox(
    metadata,
    "Annotator",
    result.annotator || "—",
  );
  addMetaBox(
    metadata,
    "Inference cases",
    String(result.inference_runs.length),
  );
  elements.detail.appendChild(metadata);
  elements.detail.appendChild(
    createElement(
      "div",
      "notice",
      isSynthesizedPayload()
        ? "합성 질문 표현만 달라지며, 작품과 마디 범위는 원래 인간 주석 "
          + "평가 "
          + "사례의 구조화된 context를 그대로 사용한다."
        : "적용 범위는 연결된 specific knowledge unit의 확정 마디 범위에서 "
          + "가져온 evaluation query context다.",
    ),
  );
  const synthesisComparison = isSynthesizedPayload()
    && !isSemanticPayload(state.payload);
  const reference = renderAuthoritativeReference(result);
  if (!synthesisComparison) {
    if (reference) elements.detail.appendChild(reference);
    elements.detail.appendChild(renderQuestionReview(result));
  }

  const cases = resultCases(result);
  if (!cases.length) {
    elements.detail.appendChild(
      createElement("div", "empty-state", "이 질문에는 추론 사례가 없다."),
    );
    return;
  }
  const selectedKey = state.selectedCaseBySource.get(result.source_id)
    || cases[0].key;
  const selected = cases.find(item => item.key === selectedKey) || cases[0];
  const panelId = `case-panel-${result.source_id}`;
  const tabs = createElement("div", "case-tabs");
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Inference cases");
  cases.forEach((item, visibleIndex) => {
    const tabId = `case-tab-${result.source_id}-${item.index}`;
    const button = createElement("button", "case-tab");
    button.type = "button";
    button.id = tabId;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(item.key === selected.key));
    button.setAttribute("aria-controls", panelId);
    button.tabIndex = item.key === selected.key ? 0 : -1;
    button.append(
      document.createTextNode(
        [
          variantSlot(runVariantId(item.run)),
          formatRange(item.run.inference_input.measure_range),
        ].filter(Boolean).join(" · "),
      ),
      createElement(
        "span",
        "tab-status",
        state.manualReviewEnabled
          ? (ReviewState.isCaseComplete(item.review) ? "✓" : "•")
          : (ReviewState.semanticCaseStatus(item.run) === "pending" ? "…" : "✓"),
      ),
    );
    const activate = focus => {
      state.selectedCaseBySource.set(result.source_id, item.key);
      renderDetail(result);
      writeLocation();
      if (focus) document.getElementById(tabId)?.focus();
    };
    button.addEventListener("click", () => {
      activate(true);
    });
    button.addEventListener("keydown", event => {
      let targetIndex = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        targetIndex = (visibleIndex + 1) % cases.length;
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        targetIndex = (visibleIndex - 1 + cases.length) % cases.length;
      } else if (event.key === "Home") {
        targetIndex = 0;
      } else if (event.key === "End") {
        targetIndex = cases.length - 1;
      }
      if (targetIndex === null) return;
      event.preventDefault();
      const target = cases[targetIndex];
      state.selectedCaseBySource.set(result.source_id, target.key);
      renderDetail(result);
      writeLocation();
      document
        .getElementById(`case-tab-${result.source_id}-${target.index}`)
        ?.focus();
    });
    tabs.appendChild(button);
  });
  const selectedTabId = `case-tab-${result.source_id}-${selected.index}`;
  elements.detail.append(
    tabs,
    renderCasePanel(result, selected, selectedTabId, panelId),
  );
  if (synthesisComparison) {
    elements.detail.appendChild(renderQuestionReview(result));
    if (reference) elements.detail.appendChild(reference);
  }
}

function updateNavigation(items) {
  const index = items.findIndex(
    result => result.source_id === state.selectedSourceId,
  );
  elements.previous.disabled = index <= 0;
  elements.next.disabled = index < 0 || index >= items.length - 1;
  elements.nextUnreviewed.disabled = !state.manualReviewEnabled
    || !items.some(result => !questionComplete(result));
}

function moveSelection(offset, focusQueue = false) {
  const items = filteredResults();
  const index = items.findIndex(
    result => result.source_id === state.selectedSourceId,
  );
  const target = items[index + offset];
  if (!target) return;
  state.selectedSourceId = target.source_id;
  render();
  const selected = elements.queueList.querySelector('[aria-selected="true"]');
  selected?.scrollIntoView({ block: "nearest" });
  if (focusQueue) selected?.focus();
}

function moveToNextUnreviewed() {
  if (!state.manualReviewEnabled) return;
  const items = filteredResults();
  if (!items.length) return;
  const current = items.findIndex(
    result => result.source_id === state.selectedSourceId,
  );
  for (let offset = 1; offset <= items.length; offset += 1) {
    const candidate = items[(current + offset + items.length) % items.length];
    if (!questionComplete(candidate)) {
      state.selectedSourceId = candidate.source_id;
      render();
      return;
    }
  }
}

function writeLocation() {
  if (!state.selectedSourceId) return;
  const params = new URLSearchParams();
  params.set("source", state.selectedSourceId);
  const caseKey = state.selectedCaseBySource.get(state.selectedSourceId);
  if (caseKey) params.set("case", caseKey);
  history.replaceState(null, "", `#${params.toString()}`);
}

function readLocation() {
  const params = new URLSearchParams(location.hash.slice(1));
  const sourceId = params.get("source");
  const caseKey = params.get("case");
  if (state.payload.results.some(result => result.source_id === sourceId)) {
    state.selectedSourceId = sourceId;
    if (caseKey) state.selectedCaseBySource.set(sourceId, caseKey);
  }
}

function render() {
  if (!state.payload || !state.draft) return;
  renderSummary();
  renderFinding();
  renderPieceTabs();
  const items = filteredResults();
  ensureSelection(items);
  renderQueue(items);
  renderDetail(selectedResult(items));
  updateNavigation(items);
  writeLocation();
}

function resetFilters() {
  state.query = "";
  state.annotator = "all";
  state.reviewStatus = "all";
  state.range = "all";
  state.basis = "all";
  state.generationMode = "all";
  state.runStatus = "all";
  state.variant = "all";
  state.semantic = "all";
  state.completion = "all";
  state.pieceId = "all";
  elements.search.value = "";
  elements.annotatorFilter.value = "all";
  elements.statusFilter.value = "all";
  elements.rangeFilter.value = "all";
  elements.basisFilter.value = "all";
  elements.generationModeFilter.value = "all";
  elements.runStatusFilter.value = "all";
  elements.variantFilter.value = "all";
  elements.semanticFilter.value = "all";
  elements.completionFilter.value = "all";
  state.selectedSourceId = null;
  render();
}

function downloadJson(value, filename) {
  const blob = new Blob(
    [JSON.stringify(value, null, 2) + "\n"],
    { type: "application/json;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportReview() {
  if (!state.manualReviewEnabled) return;
  const review = ReviewState.exportReview(state.payload, state.draft);
  downloadJson(
    review,
    `soprano-qa-qualitative-review-${state.fingerprint}.json`,
  );
  elements.saveStatus.textContent = "검토 JSON을 내보냄";
}

async function importReview(event) {
  if (!state.manualReviewEnabled) {
    event.target.value = "";
    return;
  }
  const [file] = event.target.files;
  event.target.value = "";
  if (!file) return;
  try {
    const candidate = JSON.parse(await file.text());
    const normalized = ReviewState.normalizeDraft(state.payload, candidate);
    if (!window.confirm("현재 로컬 검토를 불러온 파일로 바꿀까?")) return;
    state.draft = normalized;
    persistDraft("검토 JSON을 불러오고 저장함");
    render();
  } catch (error) {
    elements.saveStatus.textContent = `불러오기 실패: ${error.message}`;
  }
}

function clearReview() {
  if (!state.manualReviewEnabled) return;
  if (!window.confirm("이 실행에 대해 브라우저에 저장한 검토를 모두 지울까?")) {
    return;
  }
  try {
    localStorage.removeItem(state.storageKey);
  } catch (error) {
    elements.saveStatus.textContent =
      `브라우저 저장소를 지울 수 없음: ${error.message}`;
  }
  state.draft = ReviewState.createDraft(state.payload);
  persistDraft("새 검토 상태로 초기화함");
  render();
}

function wireControls() {
  elements.search.addEventListener("input", event => {
    state.query = event.target.value.trim();
    state.selectedSourceId = null;
    render();
  });
  elements.annotatorFilter.addEventListener("change", event => {
    state.annotator = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.statusFilter.addEventListener("change", event => {
    state.reviewStatus = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.rangeFilter.addEventListener("change", event => {
    state.range = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.basisFilter.addEventListener("change", event => {
    state.basis = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.generationModeFilter.addEventListener("change", event => {
    state.generationMode = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.runStatusFilter.addEventListener("change", event => {
    state.runStatus = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.variantFilter.addEventListener("change", event => {
    state.variant = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.semanticFilter.addEventListener("change", event => {
    state.semantic = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.completionFilter.addEventListener("change", event => {
    state.completion = event.target.value;
    state.selectedSourceId = null;
    render();
  });
  elements.resetFilters.addEventListener("click", resetFilters);
  elements.previous.addEventListener("click", () => moveSelection(-1));
  elements.next.addEventListener("click", () => moveSelection(1));
  elements.nextUnreviewed.addEventListener("click", moveToNextUnreviewed);
  elements.exportReview.addEventListener("click", exportReview);
  elements.importReview.addEventListener("change", importReview);
  elements.clearReview.addEventListener("click", clearReview);
  elements.queueList.addEventListener("keydown", event => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveSelection(1, true);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveSelection(-1, true);
    }
  });
}

function appendFilterOptions(select, values) {
  values.forEach(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function configurePayloadControls() {
  const runs = state.payload.results.flatMap(result => result.inference_runs);
  const generationModes = new Set();
  const variants = new Set();
  state.payload.results.forEach(result => {
    listOrEmpty(result.synthesized_question_variants).forEach(variant => {
      if (variant?.variant_id) variants.add(variantSlot(variant.variant_id));
    });
  });
  runs.forEach(run => {
    const mode = generatedOutput(run).generation_mode;
    if (mode) generationModes.add(mode);
    const variantId = runVariantId(run);
    if (variantId) variants.add(variantSlot(variantId));
  });
  appendFilterOptions(
    elements.generationModeFilter,
    [...generationModes].sort(),
  );
  appendFilterOptions(elements.variantFilter, [...variants].sort());

  const synthesized = isSynthesizedPayload();
  elements.generationModeFilter.closest(".filter").hidden =
    generationModes.size === 0;
  elements.runStatusFilter.closest(".filter").hidden = runs.length === 0;
  elements.variantFilter.closest(".filter").hidden = !synthesized;
  elements.semanticFilter.closest(".filter").hidden =
    !isSemanticPayload(state.payload);

  if (synthesized && !isSemanticPayload(state.payload)) {
    document.title = "Soprano QA · Expected vs generated";
    elements.viewerEyebrow.textContent =
      "Soprano QA · five-piece synthesized evaluation";
    elements.viewerTitle.textContent = "Expected answer vs generated answer";
    elements.viewerSubtitle.textContent =
      "5개 작품의 인간 주석 기반 expected/reference answer와, 동일한 의미를 "
      + "다른 표현으로 물은 합성 질문에 대한 hybrid RAG+LLM 답변을 "
      + "좌우로 비교한다. 생성 모드·실행 상태·합성 질문 필터로 "
      + "사례를 줄일 수 있다.";
  }
}

function configureManualReviewControls() {
  const disabled = !state.manualReviewEnabled;
  elements.exportReview.disabled = disabled;
  elements.importReview.disabled = disabled;
  elements.clearReview.disabled = disabled;
  elements.completionFilter.disabled = disabled;
  elements.importReviewLabel.classList.toggle("disabled", disabled);
  elements.importReviewLabel.setAttribute("aria-disabled", String(disabled));
  const explanation = disabled
    ? "Judge 실행이 완료된 안정된 스냅샷에서만 사용할 수 있다."
    : "";
  [
    elements.exportReview,
    elements.importReview,
    elements.importReviewLabel,
    elements.clearReview,
    elements.completionFilter,
  ].forEach(element => {
    element.title = explanation;
  });
  if (disabled) {
    state.completion = "all";
    elements.completionFilter.value = "all";
    elements.saveStatus.textContent =
      "실행 중 스냅샷: 수동 semantic-fidelity 상태 저장 비활성화";
  }
}

function validatePayload(payload) {
  if (
    !payload
    || ![
      "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "8.1",
    ].includes(
      payload.schema_version,
    )
    || !Array.isArray(payload.results)
    || payload.results.length === 0
  ) {
    throw new Error("지원하지 않는 evaluation result 형식이다.");
  }
  if (
    payload.results.some(
      result => (
        !Array.isArray(result.inference_runs)
        || result.inference_runs.length === 0
      ),
    )
  ) {
    throw new Error("Inference run 배열이 없는 결과가 있다.");
  }
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

async function loadPayload() {
  return fetchJson("/api/results");
}

function showLoadError(error) {
  elements.detail.replaceChildren();
  const box = createElement("div", "error-state");
  box.append(
    createElement("strong", "", "검토 결과를 불러오지 못했다."),
    document.createElement("br"),
    document.createTextNode(String(error)),
    document.createElement("br"),
    document.createElement("br"),
    document.createTextNode(
      "저장소 루트에서 `python3 evaluation/server.py`를 실행한다. 다른 "
      + "스냅샷은 `--results-file PATH`로 명시할 수 있다.",
    ),
  );
  elements.detail.appendChild(box);
  elements.saveStatus.textContent = "Load failed";
}

async function initialize() {
  if (!ReviewState) {
    showLoadError(new Error("review-state.js가 로드되지 않았다."));
    return;
  }
  try {
    state.payload = await loadPayload();
    validatePayload(state.payload);
    state.manualReviewEnabled = manualReviewEnabledForPayload(state.payload);
    state.fingerprint = ReviewState.fingerprintRun(state.payload);
    state.storageKey = ReviewState.storageKey(state.fingerprint);
    state.draft = readStoredDraft();
    readLocation();
    wireControls();
    configurePayloadControls();
    configureManualReviewControls();
    if (state.manualReviewEnabled) {
      persistDraft("이 실행의 로컬 semantic-fidelity 검토 준비됨");
    }
    render();
  } catch (error) {
    showLoadError(error);
  }
}

initialize();
