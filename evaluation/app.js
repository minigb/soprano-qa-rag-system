"use strict";

const ReviewState = window.SopranoReviewState;
const PAGE_MODE = document.body.dataset.viewerMode || "comparison";
const RED_FLAGS_PAGE = PAGE_MODE === "red-flags";

const PIECE_LABELS = {
  "die-forelle": "Die Forelle",
  "in-flowery-clouds": "꽃구름 속에",
  "la-capinera": "La Capinera",
  "nella-fantasia": "Nella Fantasia",
  "una-voce-poco-fa": "Una voce poco fa",
};

const STATUS_LABELS = {
  retrievable: "Retrievable",
  approved: "Approved",
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

const QUESTION_KIND_LABELS = {
  human_original: "Human original",
  source_question_paraphrase: "Synthesized paraphrase",
  knowledge_unit_derived: "KU-derived question",
};

const QUESTION_KIND_TONES = {
  human_original: "accent",
  source_question_paraphrase: "info",
  knowledge_unit_derived: "plum",
};

const QUESTION_KIND_ORDER = [
  "human_original",
  "source_question_paraphrase",
  "knowledge_unit_derived",
];

const QUALITY_REVIEW_LABELS = {
  flagged: "Semantic red flag",
  pass: "Direct review pass",
  not_available: "Quality review unavailable",
};

const QUALITY_REVIEW_TONES = {
  flagged: "danger",
  pass: "success",
  not_available: "warning",
};

function defaultQualityFilter() {
  return RED_FLAGS_PAGE ? "flagged" : "all";
}

const state = {
  payload: null,
  draft: null,
  fingerprint: null,
  storageKey: null,
  selectedSourceId: null,
  selectedCaseBySource: new Map(),
  pieceId: "all",
  questionOrigin: "all",
  query: "",
  annotator: "all",
  reviewStatus: "all",
  range: "all",
  runStatus: "all",
  variant: "all",
  semantic: "all",
  quality: defaultQualityFilter(),
  completion: "all",
  manualReviewEnabled: false,
};

const elements = {
  viewerEyebrow: document.getElementById("viewer-eyebrow"),
  viewerTitle: document.getElementById("viewer-title"),
  viewerSubtitle: document.getElementById("viewer-subtitle"),
  comparisonPageLink: document.getElementById("comparison-page-link"),
  redFlagsPageLink: document.getElementById("red-flags-page-link"),
  qualityReviewBanner: document.getElementById("quality-review-banner"),
  pieceTabs: document.getElementById("piece-tabs"),
  search: document.getElementById("search"),
  questionOriginFilter: document.getElementById("question-origin-filter"),
  annotatorFilter: document.getElementById("annotator-filter"),
  statusFilter: document.getElementById("status-filter"),
  rangeFilter: document.getElementById("range-filter"),
  runStatusFilter: document.getElementById("run-status-filter"),
  variantFilter: document.getElementById("variant-filter"),
  semanticFilter: document.getElementById("semantic-filter"),
  qualityReviewFilter: document.getElementById("quality-review-filter"),
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

function sanitizeDisplayedAnswer(value) {
  const answer = String(value || "");
  const footerIndex = answer.search(
    /제공된[\s*_~`]*검색[\s*_~`]*근거[\s*_~`]*[:：]?/i,
  );
  const withoutFooter = footerIndex >= 0
    ? answer.slice(0, footerIndex)
    : answer;
  return withoutFooter
    .replace(
      /[\[【]\s*(?:(?:출처|근거|source|citation)\s*[:：-]?\s*)?(?:(?:E(?:vidence)?\s*(?:(?:no\.?|number|번호)\s*)?(?:[:：#._–—-]\s*)?\d+)|(?:(?:[a-z0-9]+-)+ku[-_a-z0-9*]*|sqa[-_a-z0-9*]+|web(?:chunk|src|source|claim)[-_a-z0-9*]+))(?:\s*[,;，、]\s*(?:(?:E(?:vidence)?\s*(?:(?:no\.?|number|번호)\s*)?(?:[:：#._–—-]\s*)?\d+)|(?:(?:[a-z0-9]+-)+ku[-_a-z0-9*]*|sqa[-_a-z0-9*]+|web(?:chunk|src|source|claim)[-_a-z0-9*]+)))*\s*[\]】]/gi,
      "",
    )
    .replace(
      /(^|[^a-z0-9_-])(?:(?:[a-z0-9]+-)+ku-?\d{3}|sqa-\d+|webchunk-[a-z0-9*-]+)(?![a-z0-9_-])/gi,
      "$1",
    )
    .replace(
      /[\[【(（]\s*(?:(?:[a-z0-9]+[\s*_~`–—-]+)+ku(?:[\s*_~`–—-]+[a-z0-9*]+)?|sqa(?:[\s*_~`–—-]+[a-z0-9*]+)+|web[\s*_~`–—-]*(?:chunk|src|source|claim)(?:[\s*_~`–—-]+[a-z0-9*]+)+)\s*[\]】)）]/gi,
      "",
    )
    .replace(
      /(^|[^a-z0-9])(?:(?:[a-z0-9]+[\s*_~`–—-]+)+ku(?:[\s*_~`–—-]+[a-z0-9*]+)?|sqa(?:[\s*_~`–—-]+[a-z0-9*]+)+|web[\s*_~`–—-]*(?:chunk|src|source|claim)(?:[\s*_~`–—-]+[a-z0-9*]+)+)(?=$|[^a-z0-9])/gi,
      "$1",
    )
    .replace(/[ \t]+([,.;:!?，。])/g, "$1")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
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

function isComparisonPayload(payload = state.payload) {
  return payload?.artifact_type
    === "soprano_qa_original_synthesized_answer_comparison";
}

function runQuestionOrigin(run) {
  const declared = String(run?.viewer_question_origin || "");
  if (["original", "synthesized"].includes(declared)) return declared;
  return runVariantId(run) ? "synthesized" : "original";
}

function runQuestionKind(run) {
  const declared = String(run?.viewer_question_kind || "");
  if (QUESTION_KIND_ORDER.includes(declared)) return declared;
  const input = objectOrEmpty(run?.inference_input);
  if (input.question_provenance === "knowledge_unit_derived") {
    return "knowledge_unit_derived";
  }
  return runQuestionOrigin(run) === "original"
    ? "human_original"
    : "source_question_paraphrase";
}

function displaySourceId(result) {
  return result?.source_evaluation_id || result?.source_id || "";
}

function questionKindLabel(kind) {
  return QUESTION_KIND_LABELS[kind] || kind || "Unknown";
}

function questionKindClass(kind) {
  return String(kind || "unknown").replaceAll("_", "-");
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

function qualityReview(run) {
  return objectOrEmpty(run?.quality_review);
}

function qualityReviewCategory(run) {
  const review = qualityReview(run);
  if (review.review_status !== "complete") return "not_available";
  return review.red_flag === true ? "flagged" : "pass";
}

function appendQualityReviewBadge(container, run) {
  const category = qualityReviewCategory(run);
  appendBadge(
    container,
    QUALITY_REVIEW_LABELS[category],
    QUALITY_REVIEW_TONES[category],
  );
}

function qualityReviewSummary(runs) {
  const reviewedRuns = listOrEmpty(runs);
  const redFlagCount = reviewedRuns.filter(
    run => qualityReviewCategory(run) === "flagged",
  ).length;
  const passCount = reviewedRuns.filter(
    run => qualityReviewCategory(run) === "pass",
  ).length;
  let category = "not_available";
  if (redFlagCount > 0) category = "flagged";
  else if (reviewedRuns.length > 0 && passCount === reviewedRuns.length) {
    category = "pass";
  }
  return {
    category,
    redFlagCount,
    runCount: reviewedRuns.length,
  };
}

function appendScenarioQualityReviewBadge(container, runs) {
  const summary = qualityReviewSummary(runs);
  if (summary.category === "flagged") {
    appendBadge(
      container,
      `${summary.redFlagCount} semantic red flag${
        summary.redFlagCount === 1 ? "" : "s"
      }`,
      "danger",
    );
    return;
  }
  if (summary.category === "pass") {
    appendBadge(container, "All visible runs passed direct review", "success");
    return;
  }
  appendBadge(container, QUALITY_REVIEW_LABELS.not_available, "warning");
}

function isBenchmarkRagLlmAnswer(output) {
  return output?.generation_mode === "llm"
    && output?.answer_basis === "retrieved_evidence"
    && listOrEmpty(output?.evidence).length > 0;
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
  const kind = runQuestionKind(run);
  if (state.questionOrigin === "original" && kind !== "human_original") {
    return false;
  }
  if (
    state.questionOrigin === "synthesized"
    && runQuestionOrigin(run) !== "synthesized"
  ) {
    return false;
  }
  if (
    ["source_question_paraphrase", "knowledge_unit_derived"].includes(
      state.questionOrigin,
    )
    && kind !== state.questionOrigin
  ) {
    return false;
  }
  if (
    state.runStatus !== "all"
    && executionStatus(run) !== state.runStatus
  ) {
    return false;
  }
  if (
    state.quality !== "all"
    && qualityReviewCategory(run) !== state.quality
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
    result.source_evaluation_id,
    result.viewer_scenario_id,
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
  if (["retrievable", "approved"].includes(status)) return "success";
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
  const matchingCaseCount = items.reduce(
    (total, result) => total + resultCases(result).length,
    0,
  );
  const totalCaseCount = state.payload.results.reduce(
    (total, result) => total + result.inference_runs.length,
    0,
  );
  elements.resultCount.textContent =
    `${items.length} / ${state.payload.results.length} scenarios · `
    + `${matchingCaseCount} / ${totalCaseCount} results`;
  elements.queueList.replaceChildren();
  if (items.length === 0) {
    const qualityReviewComplete =
      objectOrEmpty(state.payload.quality_review).review_status === "complete";
    let message = "조건에 맞는 질문이 없다.";
    if (RED_FLAGS_PAGE && !qualityReviewComplete) {
      message = "No direct semantic review is available, so red flags cannot be shown.";
    } else if (RED_FLAGS_PAGE) {
      message = "No semantic red flags match the current filters.";
    }
    elements.queueList.appendChild(
      createElement("div", "empty-state", message),
    );
    return;
  }

  items.forEach(result => {
    const matchingRuns = result.inference_runs.filter(
      run => runMatchesFilters(run) && runMatchesQuery(result, run),
    );
    const matchingRun = matchingRuns[0] || result.inference_runs[0];
    const scenarioQuality = qualityReviewSummary(matchingRuns);
    const button = createElement("button", "queue-item");
    button.classList.toggle(
      "has-quality-red-flag",
      scenarioQuality.redFlagCount > 0,
    );
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
      createElement("span", "", displaySourceId(result)),
      createElement(
        "span",
        "",
        formatRange(objectOrEmpty(matchingRun.inference_input).measure_range),
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
    const questionKinds = new Set(
      result.inference_runs.map(runQuestionKind),
    );
    QUESTION_KIND_ORDER.forEach(kind => {
      const count = result.inference_runs.filter(
        run => runQuestionKind(run) === kind,
      ).length;
      if (!questionKinds.has(kind)) return;
      appendBadge(
        badges,
        `${questionKindLabel(kind)} · ${count}`,
        QUESTION_KIND_TONES[kind],
      );
    });
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
    appendScenarioQualityReviewBadge(badges, matchingRuns);
    if (isSynthesizedPayload()) {
      const kind = runQuestionKind(matchingRun);
      appendBadge(
        badges,
        `Showing ${questionKindLabel(kind)}`,
        QUESTION_KIND_TONES[kind],
      );
      if (result.viewer_synthesized_only) {
        appendBadge(badges, "Synthesized only", "plum");
      }
      const variantId = runVariantId(matchingRun);
      if (variantId) appendBadge(badges, variantSlot(variantId), "info");
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
      runQuestionOrigin(caseItem.run) === "synthesized"
        && !isSemanticPayload(state.payload)
        ? "이 사례의 인간 expected/reference answer와 파이프라인 답변 사이의 "
          + "의미 충실도를 평가한다. 검색/KU 적중은 보조 진단으로 "
          + "따로 기록한다."
        : "이 사례의 확정된 expected/reference answer와 파이프라인 답변 사이의 "
          + "의미 충실도를 평가한다. 검색/KU 적중은 보조 진단으로 따로 기록한다.",
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

function finalizedKnowledgeUnits(reference, applicableIds = []) {
  const applicable = new Set(listOrEmpty(applicableIds));
  return listOrEmpty(reference.linked_knowledge_units).filter(unit => (
    unit?.answer
    && unit.rewrite_status === "ready"
    && ["specific", "whole_piece", "unspecified"].includes(
      unit.measure_status,
    )
    && (!applicable.size || applicable.has(unit.knowledge_unit_id))
  ));
}

function renderCaseAuthorityComparison(run, result) {
  const authority = ReviewState.caseReferenceAuthority(run);
  const isSynthesisRun = runQuestionOrigin(run) === "synthesized";
  const comparisonReference = authority;
  const generated = generatedOutput(run);
  const section = createElement("section", "authority-comparison");
  const heading = createElement("div", "comparison-heading");
  const title = createElement("div");
  title.append(
    createElement(
      "p",
      "eyebrow",
      isSynthesisRun
        ? "Expected vs pipeline answer"
        : "Original expected vs pipeline answer",
    ),
    createElement(
      "h3",
      "",
      isSynthesisRun
        ? "Finalized expected answer vs pipeline answer"
        : "Finalized expected answer vs pipeline answer",
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
        ? `${comparisonReference.items.length} finalized KU reference item(s)`
        : `${comparisonReference.items.length} finalized reference item(s)`,
      "info",
    );
  } else {
    appendBadge(badges, "Finalized reference pending", "warning");
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
        ? "Finalized knowledge-unit reference answer"
        : "Finalized per-case reference answer",
    ),
  );
  if (comparisonReference.items.length) {
    referenceColumn.appendChild(
      createElement(
        "p",
        "comparison-source",
        `Reference source: ${comparisonReference.source}`,
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
        "이 체크포인트에는 범위에 적용되는 최종 knowledge-unit reference가 없다.",
      ),
    );
  }

  const candidateColumn = createElement("article", "comparison-column candidate");
  candidateColumn.appendChild(createElement("h4", "", "Pipeline answer"));
  const candidateBadges = createElement("div", "badge-row");
  if (generated.generation_mode) {
    appendBadge(
      candidateBadges,
      generated.generation_mode,
      "success",
    );
  }
  if (generated.answer_basis) {
    appendBadge(
      candidateBadges,
      generated.answer_basis,
      "success",
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
      sanitizeDisplayedAnswer(generated.answer)
        || "아직 파이프라인 답변이 없습니다.",
    ),
  );
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
    createElement("h3", "", "Pipeline answer compared with original expert claims"),
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
    createElement("h4", "", "Answer grounding diagnostics (secondary)"),
  );
  const generatedEvidence = listOrEmpty(generated.evidence);
  const grounded = isBenchmarkRagLlmAnswer(generated);
  const groundingLabel = grounded
    ? "Retrieved evidence-grounded RAG+LLM answer"
    : "Invalid benchmark answer";
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
      "Answer grounding diagnostics",
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

function renderQualityAssessment(run) {
  const review = qualityReview(run);
  const category = qualityReviewCategory(run);
  const banner = createElement(
    "section",
    `case-quality-assessment ${QUALITY_REVIEW_TONES[category]}`,
  );
  const copy = createElement("div");
  if (category === "not_available") {
    copy.append(
      createElement("strong", "", "Direct semantic review is not available"),
      createElement(
        "p",
        "",
        "This run has not been classified as either passing or red-flagged. "
          + "Do not interpret the absence of a flag as a semantic pass.",
      ),
    );
  } else if (category === "flagged") {
    copy.append(
      createElement(
        "strong",
        "",
        `Semantic red flag · ${review.severity || "severity unavailable"}`,
      ),
      createElement("p", "", review.rationale || "No rationale supplied."),
    );
    const codes = createElement("div", "badge-row");
    listOrEmpty(review.reason_codes).forEach(code => {
      appendBadge(codes, code, "danger");
    });
    copy.appendChild(codes);
  } else {
    copy.append(
      createElement("strong", "", "Direct semantic review passed"),
      createElement(
        "p",
        "",
        "The reviewed answer preserved the intended meaning of its finalized "
          + "reference for this exact question and range context.",
      ),
    );
  }
  const hash = review.semantic_hash
    ? createElement(
      "code",
      "quality-hash",
      `judgment ${String(review.semantic_hash).slice(0, 12)}`,
    )
    : null;
  banner.appendChild(copy);
  if (hash) banner.appendChild(hash);
  return banner;
}

function renderCasePanel(result, caseItem, tabId, panelId) {
  const panel = createElement("section", "case-panel");
  panel.setAttribute("role", "tabpanel");
  panel.id = panelId;
  panel.setAttribute("aria-labelledby", tabId);
  const input = caseItem.run.inference_input;
  const questionOrigin = runQuestionOrigin(caseItem.run);
  const questionKind = runQuestionKind(caseItem.run);
  panel.classList.add(questionKindClass(questionKind));
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
    "Question family",
    questionKindLabel(questionKind),
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
  const caseSemanticStatus = ReviewState.semanticCaseStatus(caseItem.run);
  appendBadge(
    badges,
    questionKindLabel(questionKind),
    QUESTION_KIND_TONES[questionKind],
  );
  appendBadge(
    badges,
    "RAG+LLM · retrieved evidence",
    "success",
  );
  appendQualityReviewBadge(badges, caseItem.run);
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
  panel.append(heading, renderQualityAssessment(caseItem.run), caseMetadata);
  if (runQuestion(caseItem.run)) {
    const question = createElement("article", "case-question-card");
    question.append(
      createElement(
        "p",
        "content-label",
        questionKind === "knowledge_unit_derived"
          ? "KU-derived evaluation question · no human original"
          : questionOrigin === "synthesized"
            ? `Synthesized paraphrase · ${variantSlot(runVariantId(caseItem.run))}`
            : "Original human evaluation question",
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
          quality_review: caseItem.run.quality_review,
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

function comparisonCaseGroups(cases) {
  return QUESTION_KIND_ORDER.map(kind => ({
    kind,
    cases: cases.filter(item => runQuestionKind(item.run) === kind),
  }));
}

function questionKindDescription(kind) {
  if (kind === "human_original") return "Human-authored evaluation wording";
  if (kind === "source_question_paraphrase") {
    return "Meaning-preserving paraphrases of the human question";
  }
  return "Evaluation-only wording derived from a reviewed knowledge unit";
}

function questionKindCardLabel(caseItem) {
  const kind = runQuestionKind(caseItem.run);
  if (kind === "human_original") return "Original";
  if (kind === "knowledge_unit_derived") return "KU-derived";
  return variantSlot(runVariantId(caseItem.run));
}

function caseElementId(prefix, result, caseItem) {
  const scenario = String(result.viewer_scenario_id || result.source_id)
    .replace(/[^a-zA-Z0-9_-]/g, "-");
  return `${prefix}-${scenario}-${caseItem.index}`;
}

function selectComparisonCase(result, caseItem, focusId = "") {
  state.selectedCaseBySource.set(result.source_id, caseItem.key);
  renderDetail(result);
  writeLocation();
  if (focusId) document.getElementById(focusId)?.focus();
}

function renderQuestionWordingComparison(result, cases, selected) {
  const section = createElement("section", "question-comparison-panel");
  const heading = createElement("div", "question-comparison-heading");
  const copy = createElement("div");
  copy.append(
    createElement("p", "eyebrow", "Questions sent to the pipeline"),
    createElement(
      "h3",
      "",
      result.viewer_synthesized_only
        ? "KU-derived evaluation question"
        : "Original and synthesized wording",
    ),
    createElement(
      "p",
      "section-description",
      "Each card is a separate completed inference result for the same "
        + "piece, scenario type, measure range, and finalized answer authority.",
    ),
  );
  const counts = createElement("div", "badge-row");
  comparisonCaseGroups(cases).forEach(group => {
    if (!group.cases.length) return;
    appendBadge(
      counts,
      `${questionKindLabel(group.kind)} · ${group.cases.length}`,
      QUESTION_KIND_TONES[group.kind],
    );
  });
  heading.append(copy, counts);
  section.appendChild(heading);

  const families = createElement("div", "question-family-list");
  comparisonCaseGroups(cases).forEach(group => {
    if (!group.cases.length) return;
    const family = createElement(
      "section",
      `question-family-group ${questionKindClass(group.kind)}`,
    );
    const familyHeading = createElement("div", "question-family-heading");
    familyHeading.append(
      createElement(
        "strong",
        "",
        `${questionKindLabel(group.kind)} questions`,
      ),
      createElement(
        "span",
        "",
        questionKindDescription(group.kind),
      ),
    );
    family.appendChild(familyHeading);
    const cards = createElement("div", "question-family-cards");
    group.cases.forEach(caseItem => {
      const selectedCase = caseItem.key === selected.key;
      const cardId = caseElementId("question-card", result, caseItem);
      const card = createElement(
        "button",
        `question-origin-card ${questionKindClass(group.kind)}${
          selectedCase ? " selected" : ""
        }`,
      );
      const cardQuality = qualityReviewCategory(caseItem.run);
      card.classList.toggle("has-quality-red-flag", cardQuality === "flagged");
      card.type = "button";
      card.id = cardId;
      card.setAttribute("aria-pressed", String(selectedCase));
      const cardHeading = createElement("span", "question-card-heading");
      cardHeading.append(
        createElement(
          "strong",
          "",
          questionKindCardLabel(caseItem),
        ),
        createElement(
          "span",
          "",
          formatRange(objectOrEmpty(caseItem.run.inference_input).measure_range),
        ),
      );
      const cardStatus = createElement("span", "question-card-footer");
      cardStatus.appendChild(
        createElement(
          "span",
          "question-card-status",
          selectedCase ? "Selected result" : "Open result",
        ),
      );
      if (cardQuality === "flagged") {
        cardStatus.appendChild(
          createElement("span", "question-card-quality danger", "Semantic red flag"),
        );
      }
      card.append(
        cardHeading,
        createElement("span", "question-card-text", runQuestion(caseItem.run)),
        cardStatus,
      );
      card.addEventListener("click", () => {
        selectComparisonCase(result, caseItem, cardId);
      });
      cards.appendChild(card);
    });
    family.appendChild(cards);
    families.appendChild(family);
  });
  section.appendChild(families);

  const sourceMetadata = document.createElement("details");
  sourceMetadata.className = "source-question-metadata";
  sourceMetadata.appendChild(
    createElement(
      "summary",
      "",
      result.viewer_synthesized_only
        ? "Show KU-derived lineage"
        : "Show source question metadata",
    ),
  );
  const metadataGrid = createElement("div", "question-grid");
  const rawQuestion = createElement("article", "content-card");
  const curatedQuestion = createElement("article", "content-card highlight");
  if (result.viewer_synthesized_only) {
    rawQuestion.append(
      createElement("p", "content-label", "Question origin"),
      createElement(
        "p",
        "question-text",
        "Derived from a finalized reviewed knowledge unit; no human original question.",
      ),
    );
    curatedQuestion.append(
      createElement("p", "content-label", "Lineage source annotations"),
      createElement(
        "p",
        "question-text",
        listOrEmpty(result.source_annotation_ids).join(", ") || "—",
      ),
    );
  } else {
    rawQuestion.append(
      createElement("p", "content-label", "Raw annotator question"),
      createElement("p", "question-text", result.original_question || "—"),
    );
    curatedQuestion.append(
      createElement("p", "content-label", "Curated evaluation wording"),
      createElement("p", "question-text", result.paraphrased_question || "—"),
    );
  }
  metadataGrid.append(rawQuestion, curatedQuestion);
  sourceMetadata.append(
    createElement(
      "p",
      "source-metadata-note",
      result.viewer_synthesized_only
        ? "This evaluation-only question is not presented as a human annotation."
        : "These fields describe the source annotation. The evaluated prompt for "
          + "each result is the wording shown in the cards above.",
    ),
    metadataGrid,
  );
  section.appendChild(sourceMetadata);
  return section;
}

function renderAuthoritativeReference(result) {
  const reference = objectOrEmpty(result.authoritative_reference);
  const units = finalizedKnowledgeUnits(reference);
  if (!units.length) return null;
  const section = createElement("section", "reference-panel");
  const heading = createElement("div", "reference-heading");
  const copy = createElement("div");
  copy.append(
    createElement("p", "eyebrow", "Finalized reference authority"),
    createElement("h3", "", "Finalized knowledge-unit answers"),
  );
  const badges = createElement("div", "badge-row");
  appendBadge(badges, `${units.length} finalized unit(s)`, "success");
  heading.append(copy, badges);
  section.appendChild(heading);
  const unitList = createElement("div", "linked-unit-list");
  units.forEach(unit => {
    const card = createElement("article", "linked-unit-card");
    const unitHeading = createElement("div", "linked-unit-heading");
    unitHeading.appendChild(
      createElement("strong", "", unit.knowledge_unit_id || "Unknown KU"),
    );
    const unitBadges = createElement("div", "badge-row");
    appendBadge(unitBadges, "finalized", "success");
    if (unit.measure_status) {
      appendBadge(
        unitBadges,
        unit.measure_status,
        unit.measure_status === "specific" ? "accent" : "info",
      );
    }
    const ranges = evidenceRanges(unit.measure_ranges);
    if (ranges.length) appendBadge(unitBadges, formatRanges(ranges), "accent");
    unitHeading.appendChild(unitBadges);
    card.append(
      unitHeading,
      createElement("p", "", unit.answer),
    );
    unitList.appendChild(card);
  });
  section.appendChild(unitList);
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

  const cases = resultCases(result);
  if (!cases.length) {
    elements.detail.appendChild(
      createElement("div", "empty-state", "이 질문에는 필터와 일치하는 추론 사례가 없다."),
    );
    return;
  }
  const selectedKey = state.selectedCaseBySource.get(result.source_id)
    || cases[0].key;
  const selected = cases.find(item => item.key === selectedKey) || cases[0];
  const selectedKind = runQuestionKind(selected.run);

  const heading = createElement("div", "detail-heading");
  const copy = createElement("div");
  copy.append(
    createElement("p", "eyebrow", PIECE_LABELS[result.piece_id] || result.piece_id),
    createElement("h2", "", displaySourceId(result)),
    createElement(
      "p",
      "detail-path",
      `scenario ${result.viewer_scenario_id || result.source_id} · ${
        result.annotator || questionKindLabel(selectedKind)
      }`,
    ),
  );
  const badges = createElement("div", "badge-row");
  appendBadge(
    badges,
    `Selected: ${questionKindLabel(selectedKind)}`,
    QUESTION_KIND_TONES[selectedKind],
  );
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
  appendScenarioQualityReviewBadge(
    badges,
    cases.map(caseItem => caseItem.run),
  );
  if (state.manualReviewEnabled && questionFlagged(result)) {
    appendBadge(badges, "Manual flag", "danger");
  }
  heading.append(copy, badges);
  elements.detail.appendChild(heading);
  elements.detail.appendChild(
    renderQuestionWordingComparison(result, cases, selected),
  );

  const metadata = createElement("div", "metadata-strip");
  addMetaBox(metadata, "Piece", PIECE_LABELS[result.piece_id] || result.piece_id);
  addMetaBox(
    metadata,
    "Scenario range",
    formatRange(objectOrEmpty(selected.run.inference_input).measure_range),
  );
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
      result.viewer_synthesized_only
        ? "This is a synthesized-only evaluation question derived from a "
          + "finalized reviewed knowledge unit. It has no human original and "
          + "is reviewed as its own question-and-range scenario."
        : result.inference_runs.some(
          run => runQuestionKind(run) === "source_question_paraphrase",
        )
          ? "The human original and its three synthesized paraphrases share "
            + "the same scenario context and finalized answer authority."
          : "This documented reported-regression formulation is an "
            + "original-only evaluation scenario and has no synthesized "
            + "paraphrase counterpart.",
    ),
  );
  const reference = renderAuthoritativeReference(result);
  const panelId = `case-panel-${result.source_id}`;
  const tabGroups = createElement("div", "case-tab-groups");
  tabGroups.setAttribute("role", "tablist");
  tabGroups.setAttribute("aria-label", "Evaluation inference results");
  comparisonCaseGroups(cases).forEach(group => {
    if (!group.cases.length) return;
    const groupElement = createElement(
      "div",
      `case-tab-group ${questionKindClass(group.kind)}`,
    );
    groupElement.setAttribute("role", "presentation");
    groupElement.appendChild(
      createElement(
        "span",
        "case-tab-group-label",
        `${questionKindLabel(group.kind)} · ${group.cases.length}`,
      ),
    );
    const tabs = createElement("div", "case-tabs");
    tabs.setAttribute("role", "presentation");
    group.cases.forEach(item => {
      const visibleIndex = cases.findIndex(candidate => candidate.key === item.key);
      const tabId = caseElementId("case-tab", result, item);
      const button = createElement("button", "case-tab");
      const tabQuality = qualityReviewCategory(item.run);
      button.classList.toggle("has-quality-red-flag", tabQuality === "flagged");
      button.type = "button";
      button.id = tabId;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(item.key === selected.key));
      button.setAttribute("aria-controls", panelId);
      button.tabIndex = item.key === selected.key ? 0 : -1;
      button.append(
        document.createTextNode(
          [
            questionKindCardLabel(item),
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
      if (tabQuality === "flagged") {
        const redFlag = createElement("span", "tab-quality-flag", "!");
        redFlag.setAttribute("aria-label", "Semantic red flag");
        redFlag.title = "Semantic red flag";
        button.appendChild(redFlag);
      }
      button.addEventListener("click", () => {
        selectComparisonCase(result, item, tabId);
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
        selectComparisonCase(
          result,
          target,
          caseElementId("case-tab", result, target),
        );
      });
      tabs.appendChild(button);
    });
    groupElement.appendChild(tabs);
    tabGroups.appendChild(groupElement);
  });
  const selectedTabId = caseElementId("case-tab", result, selected);
  elements.detail.append(
    tabGroups,
    renderCasePanel(result, selected, selectedTabId, panelId),
  );
  elements.detail.appendChild(renderQuestionReview(result));
  if (reference) elements.detail.appendChild(reference);
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
  state.questionOrigin = "all";
  state.annotator = "all";
  state.reviewStatus = "all";
  state.range = "all";
  state.runStatus = "all";
  state.variant = "all";
  state.semantic = "all";
  state.quality = defaultQualityFilter();
  state.completion = "all";
  state.pieceId = "all";
  elements.search.value = "";
  elements.questionOriginFilter.value = "all";
  elements.annotatorFilter.value = "all";
  elements.statusFilter.value = "all";
  elements.rangeFilter.value = "all";
  elements.runStatusFilter.value = "all";
  elements.variantFilter.value = "all";
  elements.semanticFilter.value = "all";
  elements.qualityReviewFilter.value = state.quality;
  elements.completionFilter.value = "all";
  syncQuestionOriginControls();
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
  elements.questionOriginFilter.addEventListener("change", event => {
    state.questionOrigin = event.target.value;
    if (["original", "knowledge_unit_derived"].includes(state.questionOrigin)) {
      state.variant = "all";
      elements.variantFilter.value = "all";
    }
    syncQuestionOriginControls();
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
  elements.qualityReviewFilter.addEventListener("change", event => {
    state.quality = event.target.value;
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

function syncQuestionOriginControls() {
  const supportsVariants = isSynthesizedPayload();
  const variantsUnavailable = ["original", "knowledge_unit_derived"].includes(
    state.questionOrigin,
  );
  elements.variantFilter.closest(".filter").hidden =
    !supportsVariants || variantsUnavailable;
  elements.variantFilter.disabled = !supportsVariants || variantsUnavailable;
}

function renderQualityReviewStatus() {
  const review = objectOrEmpty(state.payload.quality_review);
  const complete = review.review_status === "complete";
  elements.qualityReviewBanner.replaceChildren();
  elements.qualityReviewBanner.hidden = !RED_FLAGS_PAGE;
  if (!RED_FLAGS_PAGE) return;
  elements.qualityReviewBanner.className = complete
    ? "quality-review-banner complete"
    : "quality-review-banner unavailable";
  const copy = createElement("div");
  if (complete) {
    copy.append(
      createElement("strong", "", "Direct semantic quality review is current"),
      createElement(
        "p",
        "",
        `${review.assessment_count} exact inference runs were reviewed; `
          + `${review.red_flag_count} currently carry a semantic red flag. `
          + "Each assessment is bound to its exact question, reference authority, "
          + "range, and generated answer.",
      ),
    );
  } else {
    copy.append(
      createElement("strong", "", "Direct semantic quality review is not available"),
      createElement(
        "p",
        "",
        "No pass or red-flag conclusion is available for these runs. An absent "
          + "review is never presented as zero semantic problems.",
      ),
    );
  }
  const pageLink = createElement(
    "a",
    "button compact",
    RED_FLAGS_PAGE ? "View all results" : "Open red-flag review",
  );
  pageLink.href = RED_FLAGS_PAGE ? "/" : "/red-flags";
  elements.qualityReviewBanner.append(copy, pageLink);
}

function configurePayloadControls() {
  const runs = state.payload.results.flatMap(result => result.inference_runs);
  const variants = new Set();
  state.payload.results.forEach(result => {
    listOrEmpty(result.synthesized_question_variants).forEach(variant => {
      if (variant?.variant_id) variants.add(variantSlot(variant.variant_id));
    });
  });
  runs.forEach(run => {
    const variantId = runVariantId(run);
    if (variantId) variants.add(variantSlot(variantId));
  });
  appendFilterOptions(elements.variantFilter, [...variants].sort());
  elements.qualityReviewFilter.value = state.quality;
  renderQualityReviewStatus();

  elements.comparisonPageLink.toggleAttribute("data-active", !RED_FLAGS_PAGE);
  elements.redFlagsPageLink.toggleAttribute("data-active", RED_FLAGS_PAGE);
  if (RED_FLAGS_PAGE) {
    elements.redFlagsPageLink.setAttribute("aria-current", "page");
    elements.comparisonPageLink.removeAttribute("aria-current");
  } else {
    elements.comparisonPageLink.setAttribute("aria-current", "page");
    elements.redFlagsPageLink.removeAttribute("aria-current");
  }

  const synthesized = isSynthesizedPayload();
  elements.runStatusFilter.closest(".filter").hidden = runs.length === 0;
  syncQuestionOriginControls();
  elements.semanticFilter.closest(".filter").hidden =
    !isSemanticPayload(state.payload);

  if (isComparisonPayload()) {
    const summary = objectOrEmpty(state.payload.summary);
    if (RED_FLAGS_PAGE) {
      document.title = "Soprano QA · Semantic Red Flags";
      elements.viewerEyebrow.textContent =
        "Soprano QA · direct semantic quality review";
      elements.viewerTitle.textContent = "Semantic red-flag review";
      elements.viewerSubtitle.textContent =
        "Review generated answers whose meaning may be incorrect, incomplete, "
          + "or unsafe for presentation. This page starts with the direct quality "
          + "filter set to red flags and labels human originals, synthesized "
          + "paraphrases, and KU-derived questions separately.";
      return;
    }
    document.title = "Soprano QA · Original and Synthesized Comparison";
    elements.viewerEyebrow.textContent =
      "Soprano QA · all five-piece results";
    elements.viewerTitle.textContent =
      "Original and synthesized answer comparison";
    elements.viewerSubtitle.textContent =
      `${summary.original_case_count} human-original results, `
      + `${summary.synthesized_paraphrase_case_count} synthesized paraphrase `
      + `results, and ${summary.knowledge_unit_derived_case_count} KU-derived `
      + `results are grouped into ${summary.scenario_count} scenarios. All `
      + `${summary.total_case_count} completed outputs across five pieces are `
      + "available; use Question family to isolate each kind.";
    return;
  }

  if (synthesized && !isSemanticPayload(state.payload)) {
    document.title = "Soprano QA · Expected vs pipeline answer";
    elements.viewerEyebrow.textContent =
      "Soprano QA · five-piece synthesized evaluation";
    elements.viewerTitle.textContent = "Expected answer vs pipeline answer";
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
    || payload.artifact_type
      !== "soprano_qa_original_synthesized_answer_comparison"
    || payload.schema_version !== "1.0"
    || !Array.isArray(payload.results)
    || payload.results.length === 0
  ) {
    throw new Error("The server did not return the current comparison artifact.");
  }
  if (
    payload.results.some(
      result => (
        !Array.isArray(result.inference_runs)
        || result.inference_runs.length === 0
      ),
    )
  ) {
    throw new Error("A comparison scenario has no inference results.");
  }
  const scenarioIds = new Set(payload.results.map(result => result.source_id));
  if (scenarioIds.size !== payload.results.length) {
    throw new Error("Comparison scenario IDs must be unique.");
  }
  const runs = payload.results.flatMap(result => result.inference_runs);
  if (
    runs.some(run => !["original", "synthesized"].includes(
      run.viewer_question_origin,
    ))
  ) {
    throw new Error("An inference result has no valid question type.");
  }
  const summary = objectOrEmpty(payload.summary);
  const originalCount = runs.filter(
    run => run.viewer_question_origin === "original",
  ).length;
  const synthesizedCount = runs.filter(
    run => run.viewer_question_origin === "synthesized",
  ).length;
  if (
    summary.scenario_count !== payload.results.length
    || summary.original_case_count !== originalCount
    || summary.synthesized_case_count !== synthesizedCount
    || summary.total_case_count !== runs.length
  ) {
    throw new Error("Comparison summary counts do not match its inference results.");
  }
  const quality = objectOrEmpty(payload.quality_review);
  if (!["complete", "not_available"].includes(quality.review_status)) {
    throw new Error("Comparison quality-review status is invalid.");
  }
  if (quality.review_status === "not_available") {
    if (
      Object.hasOwn(quality, "red_flag_count")
      || runs.some(run => qualityReview(run).review_status !== "not_available")
    ) {
      throw new Error("Unavailable quality review cannot report pass/flag results.");
    }
  } else {
    const attached = runs.map(run => qualityReview(run));
    const redFlagCount = attached.filter(review => review.red_flag === true).length;
    if (
      attached.some(review => (
        review.review_status !== "complete"
        || typeof review.red_flag !== "boolean"
      ))
      || quality.assessment_count !== runs.length
      || quality.red_flag_count !== redFlagCount
    ) {
      throw new Error("Complete quality-review coverage does not match the runs.");
    }
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
      "From the repository root, run `conda run -n soprano-qa python "
      + "evaluation/server.py`. Both the complete original and synthesized "
      + "evaluation artifacts are required.",
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
