"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ReviewState = require("./review-state.js");

function run({ range = null, evidence = false, manualReview = {} } = {}) {
  return {
    inference_input: {
      question: "질문",
      measure_range: range,
      measure_range_applied: range !== null,
    },
    retrieval_probe: {
      evidence: evidence ? [{ evidence_id: "expert-piece-ku-001" }] : [],
      expert_evidence_ids: evidence ? ["expert-piece-ku-001"] : [],
    },
    generated_answer: {
      answer: "답변",
      evidence: [],
      expert_evidence_ids: [],
    },
    manual_review: {
      range_correct: null,
      evidence_relevant: null,
      answer_acceptable: null,
      notes: "",
      ...manualReview,
    },
  };
}

function samplePayload() {
  return {
    schema_version: "1.0",
    run: {
      created_at: "2026-07-30T00:00:00Z",
      generator: "test",
    },
    summary: {
      selected_questions: 2,
    },
    results: [
      {
        source_id: "kim-piece-a-01",
        piece_id: "piece-a",
        original_question: "원문 질문",
        paraphrased_question: "자연스러운 질문",
        manual_review: {
          paraphrase_natural: null,
          notes: "",
        },
        inference_runs: [
          run(),
          run({ range: [3, 4], evidence: true }),
          run({ range: [8, 8], evidence: true }),
        ],
      },
      {
        source_id: "yeon-piece-b-02",
        piece_id: "piece-b",
        original_question: "두 번째 질문",
        paraphrased_question: "두 번째 자연스러운 질문",
        manual_review: {
          paraphrase_natural: null,
          notes: "",
        },
        inference_runs: [run({ range: [10, 12] })],
      },
    ],
  };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function reverseObjectKeys(value) {
  if (Array.isArray(value)) {
    return value.map(reverseObjectKeys);
  }
  if (value !== null && typeof value === "object") {
    return Object.keys(value)
      .reverse()
      .reduce((output, key) => {
        output[key] = reverseObjectKeys(value[key]);
        return output;
      }, {});
  }
  return value;
}

test("exports the required API to CommonJS and a browser global", () => {
  const expectedFunctions = [
    "fingerprintRun",
    "storageKey",
    "caseKey",
    "semanticCaseStatus",
    "semanticQuestionStatus",
    "semanticStatusCounts",
    "semanticRunProgress",
    "caseReferenceAuthority",
    "createDraft",
    "normalizeDraft",
    "isCaseComplete",
    "isQuestionComplete",
    "isFlagged",
    "computeProgress",
    "exportReview",
  ];
  assert.deepEqual(ReviewState.RATING_VALUES, [
    "pass",
    "fail",
    "uncertain",
    "not_applicable",
  ]);
  assert.ok(Object.isFrozen(ReviewState.RATING_VALUES));
  assert.ok(Object.isFrozen(ReviewState.ISSUE_TAGS));
  expectedFunctions.forEach((name) => {
    assert.equal(typeof ReviewState[name], "function");
  });

  const source = fs.readFileSync(
    path.join(__dirname, "review-state.js"),
    "utf8",
  );
  const browser = {};
  vm.runInNewContext(source, browser, { filename: "review-state.js" });
  assert.equal(typeof browser.SopranoReviewState.createDraft, "function");
  assert.equal(browser.module, undefined);
});

test("semantic status helpers distinguish reliability from semantic pass", () => {
  const payload = samplePayload();
  const runs = payload.results[0].inference_runs;
  runs[0].semantic_evaluation = {
    aggregate: {
      status: "pass",
      reliable_rag_llm_pass: true,
    },
  };
  runs[1].semantic_evaluation = {
    aggregate: {
      status: "pass",
      reliable_rag_llm_pass: false,
    },
  };
  runs[2].semantic_evaluation = {
    aggregate: {
      status: "human_review",
      reliable_rag_llm_pass: false,
    },
  };
  payload.results[1].inference_runs[0].semantic_evaluation = {
    aggregate: {
      status: "fail",
      reliable_rag_llm_pass: false,
    },
  };

  assert.equal(ReviewState.semanticCaseStatus(runs[0]), "reliable");
  assert.equal(ReviewState.semanticCaseStatus(runs[1]), "pass");
  assert.equal(ReviewState.semanticCaseStatus(runs[2]), "review");
  assert.equal(
    ReviewState.semanticQuestionStatus(payload.results[0]),
    "review",
  );
  assert.equal(
    ReviewState.semanticQuestionStatus(payload.results[1]),
    "fail",
  );
  assert.deepEqual(ReviewState.semanticStatusCounts(payload), {
    cases: {
      reliable: 1,
      pass: 1,
      review: 1,
      fail: 1,
      pending: 0,
    },
    questions: {
      reliable: 0,
      pass: 0,
      review: 1,
      fail: 1,
      pending: 0,
    },
  });
});

test("legacy and incomplete snapshots remain semantic-status pending", () => {
  const payload = samplePayload();
  payload.results[0].inference_runs[0].semantic_evaluation = {
    aggregate: {
      status: "fail",
      reliable_rag_llm_pass: false,
    },
  };
  assert.equal(
    ReviewState.semanticQuestionStatus(payload.results[0]),
    "pending",
  );
  assert.deepEqual(ReviewState.semanticStatusCounts(payload).cases, {
    reliable: 0,
    pass: 0,
    review: 0,
    fail: 1,
    pending: 3,
  });
});

test("semantic run progress requires both all judgments and a complete run", () => {
  const payload = samplePayload();
  payload.schema_version = "5.0";
  payload.run = { status: "in_progress" };
  payload.results.forEach((result) => {
    result.inference_runs.forEach((item) => {
      item.semantic_evaluation = {
        aggregate: {
          status: "pass",
          reliable_rag_llm_pass: true,
        },
      };
    });
  });

  assert.deepEqual(ReviewState.semanticRunProgress(payload), {
    total_cases: 4,
    judged_cases: 4,
    pending_cases: 0,
    completion_percent: 100,
    declared_complete: false,
    judge_complete: false,
  });

  payload.run.status = "complete";
  assert.equal(
    ReviewState.semanticRunProgress(payload).judge_complete,
    true,
  );
  payload.results[1].inference_runs[0].semantic_evaluation = null;
  assert.deepEqual(ReviewState.semanticRunProgress(payload), {
    total_cases: 4,
    judged_cases: 3,
    pending_cases: 1,
    completion_percent: 75,
    declared_complete: true,
    judge_complete: false,
  });
});

test("case authority prefers persisted exact R items and falls back to validated frame items", () => {
  const item = run();
  item.semantic_evaluation = {
    authoritative_reference_items: [
      { reference_id: "R001", text: "정확한 사례 권위" },
    ],
    frames: {
      claim_alignment: {
        assessment: {
          reference_assessments: [
            {
              reference_id: "R001",
              reference_text: "프레임의 권위",
              status: "covered",
            },
          ],
        },
      },
    },
  };
  assert.deepEqual(ReviewState.caseReferenceAuthority(item), {
    source: "semantic_evaluation.authoritative_reference_items",
    items: [{ reference_id: "R001", text: "정확한 사례 권위" }],
  });

  delete item.semantic_evaluation.authoritative_reference_items;
  assert.deepEqual(ReviewState.caseReferenceAuthority(item), {
    source:
      "semantic_evaluation.frames.claim_alignment.assessment.reference_assessments",
    items: [
      {
        reference_id: "R001",
        text: "프레임의 권위",
        status: "covered",
      },
    ],
  });
  item.semantic_evaluation = { frames: {} };
  assert.deepEqual(ReviewState.caseReferenceAuthority(item), {
    source: null,
    items: [],
  });
});

test("run fingerprints are deterministic and ignore embedded manual reviews", () => {
  const payload = samplePayload();
  const fingerprint = ReviewState.fingerprintRun(payload);

  assert.match(fingerprint, /^v1-[0-9a-f]{16}$/);
  assert.equal(
    ReviewState.fingerprintRun(reverseObjectKeys(payload)),
    fingerprint,
  );

  const editedReview = clone(payload);
  editedReview.results[0].manual_review.paraphrase_natural = true;
  editedReview.results[0].manual_review.notes = "검토 완료";
  editedReview.results[0].inference_runs[0].manual_review.answer_acceptable =
    false;
  assert.equal(ReviewState.fingerprintRun(editedReview), fingerprint);

  const editedRun = clone(payload);
  editedRun.results[0].inference_runs[0].generated_answer.answer =
    "다른 기계 출력";
  assert.notEqual(ReviewState.fingerprintRun(editedRun), fingerprint);
  assert.equal(
    ReviewState.storageKey(fingerprint),
    `soprano-review-state:v1:${fingerprint}`,
  );
  assert.throws(() => ReviewState.storageKey(""), /non-empty string/);
});

test("case keys preserve no-range and distinct multi-range identities", () => {
  const payload = samplePayload();
  const result = payload.results[0];

  assert.equal(
    ReviewState.caseKey(result, result.inference_runs[0], 0),
    "kim-piece-a-01::range:none",
  );
  assert.equal(
    ReviewState.caseKey(result, result.inference_runs[1], 1),
    "kim-piece-a-01::range:3-4",
  );
  assert.equal(
    ReviewState.caseKey(result, result.inference_runs[2], 2),
    "kim-piece-a-01::range:8-8",
  );

  const duplicate = {
    source_id: "kim-piece-a-09",
    inference_runs: [run({ range: [3, 4] }), run({ range: [3, 4] })],
  };
  assert.equal(
    ReviewState.caseKey(duplicate, duplicate.inference_runs[0], 0),
    "kim-piece-a-09::range:3-4::occurrence:1",
  );
  assert.equal(
    ReviewState.caseKey(duplicate, duplicate.inference_runs[1], 1),
    "kim-piece-a-09::range:3-4::occurrence:2",
  );
});

test("draft initialization applies no-range and no-evidence defaults", () => {
  const payload = samplePayload();
  const draft = ReviewState.createDraft(payload);
  const first = draft.questions["kim-piece-a-01"];
  const noRange = first.cases["kim-piece-a-01::range:none"];
  const rangedWithEvidence = first.cases["kim-piece-a-01::range:3-4"];
  const rangedWithoutEvidence =
    draft.questions["yeon-piece-b-02"].cases[
      "yeon-piece-b-02::range:10-12"
    ];

  assert.equal(draft.schema_version, "1.0");
  assert.equal(
    draft.source_run_fingerprint,
    ReviewState.fingerprintRun(payload),
  );
  assert.equal(first.paraphrase_natural, null);
  assert.deepEqual(first.issue_tags, []);
  assert.equal(noRange.range_correct, "not_applicable");
  assert.equal(noRange.evidence_relevant, "not_applicable");
  assert.equal(noRange.answer_acceptable, null);
  assert.equal(rangedWithEvidence.range_correct, null);
  assert.equal(rangedWithEvidence.evidence_relevant, null);
  assert.equal(rangedWithoutEvidence.range_correct, null);
  assert.equal(rangedWithoutEvidence.evidence_relevant, "not_applicable");
});

test("draft initialization migrates legacy boolean ratings", () => {
  const payload = samplePayload();
  payload.results[0].manual_review.paraphrase_natural = true;
  payload.results[0].inference_runs[1].manual_review = {
    range_correct: false,
    evidence_relevant: true,
    answer_acceptable: false,
    notes: "실패 사유",
  };

  const draft = ReviewState.createDraft(payload);
  const review =
    draft.questions["kim-piece-a-01"].cases[
      "kim-piece-a-01::range:3-4"
    ];
  assert.equal(
    draft.questions["kim-piece-a-01"].paraphrase_natural,
    "pass",
  );
  assert.equal(review.range_correct, "fail");
  assert.equal(review.evidence_relevant, "pass");
  assert.equal(review.answer_acceptable, "fail");
});

test("normalization fills partial imports and canonicalizes issue tags", () => {
  const payload = samplePayload();
  const base = ReviewState.createDraft(payload);
  const candidate = {
    schema_version: "1.0",
    source_run_fingerprint: base.source_run_fingerprint,
    questions: {
      "kim-piece-a-01": {
        paraphrase_natural: true,
        notes: "질문 검토",
        issue_tags: [
          "factual_error",
          "awkward_paraphrase",
          "factual_error",
        ],
        cases: {
          "kim-piece-a-01::range:none": {
            range_correct: null,
            answer_acceptable: true,
          },
          "kim-piece-a-01::range:3-4": {
            range_correct: false,
            evidence_relevant: true,
            answer_acceptable: "uncertain",
            notes: "범위와 답변을 재검토해야 함",
            issue_tags: ["retrieval_miss", "range_questionable"],
          },
        },
      },
    },
  };

  const normalized = ReviewState.normalizeDraft(payload, candidate);
  const question = normalized.questions["kim-piece-a-01"];
  assert.equal(question.paraphrase_natural, "pass");
  assert.deepEqual(question.issue_tags, [
    "awkward_paraphrase",
    "factual_error",
  ]);
  assert.equal(
    question.cases["kim-piece-a-01::range:none"].range_correct,
    "not_applicable",
  );
  assert.equal(
    question.cases["kim-piece-a-01::range:none"].evidence_relevant,
    "not_applicable",
  );
  assert.equal(
    question.cases["kim-piece-a-01::range:none"].answer_acceptable,
    "pass",
  );
  assert.deepEqual(
    question.cases["kim-piece-a-01::range:3-4"].issue_tags,
    ["range_questionable", "retrieval_miss"],
  );
  assert.equal(
    question.cases["kim-piece-a-01::range:8-8"].answer_acceptable,
    null,
  );
  assert.ok(normalized.questions["yeon-piece-b-02"]);
});

test("normalization rejects stale, unknown, and malformed import data", () => {
  const payload = samplePayload();
  const base = ReviewState.createDraft(payload);

  const stale = clone(base);
  stale.source_run_fingerprint = "v1-0000000000000000";
  assert.throws(
    () => ReviewState.normalizeDraft(payload, stale),
    /different evaluation run/,
  );

  const unknownSource = clone(base);
  unknownSource.questions["unknown-source"] = {};
  assert.throws(
    () => ReviewState.normalizeDraft(payload, unknownSource),
    /unknown source_id/,
  );

  const unknownCase = clone(base);
  unknownCase.questions["kim-piece-a-01"].cases["unknown-case"] = {};
  assert.throws(
    () => ReviewState.normalizeDraft(payload, unknownCase),
    /unknown case key/,
  );

  const invalidRating = clone(base);
  invalidRating.questions["kim-piece-a-01"].paraphrase_natural = "maybe";
  assert.throws(
    () => ReviewState.normalizeDraft(payload, invalidRating),
    /must be null, a boolean, or one of/,
  );

  const invalidTag = clone(base);
  invalidTag.questions["kim-piece-a-01"].issue_tags = ["invented_tag"];
  assert.throws(
    () => ReviewState.normalizeDraft(payload, invalidTag),
    /not a recognized issue tag/,
  );

  const extraField = clone(base);
  extraField.results = [];
  assert.throws(
    () => ReviewState.normalizeDraft(payload, extraField),
    /unknown field: results/,
  );
});

test("completion requires every rating and notes for fail or uncertain", () => {
  const completeCase = {
    range_correct: "not_applicable",
    evidence_relevant: "pass",
    answer_acceptable: "pass",
    notes: "",
    issue_tags: [],
  };
  assert.equal(ReviewState.isCaseComplete(completeCase), true);

  const failedCase = { ...completeCase, answer_acceptable: "fail" };
  assert.equal(ReviewState.isCaseComplete(failedCase), false);
  failedCase.notes = "근거가 답을 뒷받침하지 않음";
  assert.equal(ReviewState.isCaseComplete(failedCase), true);

  const uncertainCase = {
    ...completeCase,
    evidence_relevant: "uncertain",
    notes: " ",
  };
  assert.equal(ReviewState.isCaseComplete(uncertainCase), false);
  uncertainCase.notes = "전문가 확인 필요";
  assert.equal(ReviewState.isCaseComplete(uncertainCase), true);

  const question = {
    paraphrase_natural: "pass",
    notes: "",
    issue_tags: [],
    cases: { one: completeCase },
  };
  assert.equal(ReviewState.isQuestionComplete(question), true);
  assert.equal(ReviewState.isQuestionComplete(question, []), false);

  question.paraphrase_natural = "fail";
  assert.equal(ReviewState.isQuestionComplete(question), false);
  question.notes = "표현이 부자연스러움";
  assert.equal(ReviewState.isQuestionComplete(question), true);
});

test("flagging covers explicit issue tags and fail or uncertain ratings", () => {
  const cleanCase = {
    range_correct: "pass",
    evidence_relevant: "pass",
    answer_acceptable: "pass",
    notes: "정보 메모만 있음",
    issue_tags: [],
  };
  const question = {
    paraphrase_natural: "pass",
    notes: "",
    issue_tags: [],
    cases: { one: cleanCase },
  };
  assert.equal(ReviewState.isFlagged(question), false);

  cleanCase.evidence_relevant = "uncertain";
  assert.equal(ReviewState.isFlagged(question), true);
  cleanCase.evidence_relevant = "pass";
  question.issue_tags = ["awkward_paraphrase"];
  assert.equal(ReviewState.isFlagged(question), true);
});

test("progress reports question, case, piece, and flagged counts", () => {
  const payload = samplePayload();
  const draft = ReviewState.createDraft(payload);
  const initial = ReviewState.computeProgress(payload, draft);

  assert.equal(initial.total_questions, 2);
  assert.equal(initial.completed_questions, 0);
  assert.equal(initial.incomplete_questions, 2);
  assert.equal(initial.total_cases, 4);
  assert.equal(initial.completed_cases, 0);
  assert.equal(initial.completion_percent, 0);
  assert.equal(initial.all_complete, false);
  assert.equal(initial.by_piece["piece-a"].total_questions, 1);
  assert.equal(initial.by_piece["piece-a"].total_cases, 3);

  Object.values(draft.questions).forEach((question) => {
    question.paraphrase_natural = "pass";
    Object.values(question.cases).forEach((caseReview) => {
      if (caseReview.range_correct === null) {
        caseReview.range_correct = "pass";
      }
      if (caseReview.evidence_relevant === null) {
        caseReview.evidence_relevant = "pass";
      }
      caseReview.answer_acceptable = "pass";
    });
  });
  const flagged =
    draft.questions["kim-piece-a-01"].cases[
      "kim-piece-a-01::range:3-4"
    ];
  flagged.answer_acceptable = "fail";
  flagged.notes = "답변이 허용 기준을 충족하지 않음";

  const complete = ReviewState.computeProgress(payload, draft);
  assert.equal(complete.completed_questions, 2);
  assert.equal(complete.completed_cases, 4);
  assert.equal(complete.flagged_questions, 1);
  assert.equal(complete.flagged_cases, 1);
  assert.equal(complete.completion_percent, 100);
  assert.equal(complete.all_complete, true);
  assert.equal(complete.by_piece["piece-a"].flagged_questions, 1);
  assert.equal(complete.by_piece["piece-b"].flagged_questions, 0);
});

test("review export contains only normalized review state and is detached", () => {
  const payload = samplePayload();
  const draft = ReviewState.createDraft(payload);
  draft.questions["kim-piece-a-01"].paraphrase_natural = "pass";

  const exported = ReviewState.exportReview(payload, draft);
  assert.deepEqual(Object.keys(exported), [
    "schema_version",
    "source_run_fingerprint",
    "questions",
  ]);
  assert.equal(exported.results, undefined);
  assert.equal(JSON.stringify(exported).includes("generated_answer"), false);
  assert.equal(JSON.stringify(exported).includes("original_question"), false);

  exported.questions["kim-piece-a-01"].issue_tags.push("factual_error");
  assert.deepEqual(draft.questions["kim-piece-a-01"].issue_tags, []);
});

test("the real selected snapshot initializes every review case", () => {
  const resultsPath = path.join(__dirname, "manual_check_results.json");
  if (!fs.existsSync(resultsPath)) {
    return;
  }
  const payload = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
  const draft = ReviewState.createDraft(payload);
  const questions = Object.values(draft.questions);
  const cases = questions.flatMap((question) => Object.values(question.cases));
  const progress = ReviewState.computeProgress(payload, draft);
  const expectedNoEvidence = payload.results
    .flatMap((result) => result.inference_runs)
    .filter((run) => {
      const outputs = [run.retrieval_probe, run.generated_answer];
      return outputs.every((output) => (
        (!Array.isArray(output.evidence) || output.evidence.length === 0)
        && (
          !Array.isArray(output.expert_evidence_ids)
          || output.expert_evidence_ids.length === 0
        )
      ));
    }).length;

  assert.equal(questions.length, 50);
  assert.equal(cases.length, 62);
  assert.equal(
    cases.filter(
      (caseReview) => caseReview.range_correct === "not_applicable",
    ).length,
    payload.summary.no_range_questions,
  );
  assert.equal(
    cases.filter(
      (caseReview) => caseReview.evidence_relevant === "not_applicable",
    ).length,
    expectedNoEvidence,
  );
  assert.equal(progress.total_questions, 50);
  assert.equal(progress.total_cases, 62);
  assert.equal(progress.completed_questions, 0);
  assert.equal(progress.flagged_questions, 0);
});
