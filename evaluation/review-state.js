(function (root, factory) {
  "use strict";

  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.SopranoReviewState = api;
  }
})(
  typeof globalThis !== "undefined" ? globalThis : this,
  function () {
    "use strict";

    var DRAFT_SCHEMA_VERSION = "1.0";
    var STORAGE_KEY_PREFIX = "soprano-review-state:v1:";
    var RATING_VALUES = Object.freeze([
      "pass",
      "fail",
      "uncertain",
      "not_applicable",
    ]);
    var ISSUE_TAGS = Object.freeze([
      "awkward_paraphrase",
      "range_questionable",
      "retrieval_miss",
      "irrelevant_evidence",
      "unsupported_answer",
      "factual_error",
      "citation_error",
      "needs_expert_review",
    ]);

    var RATING_SET = Object.create(null);
    var ISSUE_TAG_ORDER = Object.create(null);
    RATING_VALUES.forEach(function (value) {
      RATING_SET[value] = true;
    });
    ISSUE_TAGS.forEach(function (value, index) {
      ISSUE_TAG_ORDER[value] = index;
    });

    function isRecord(value) {
      return value !== null && typeof value === "object" && !Array.isArray(value);
    }

    function own(object, key) {
      return Object.prototype.hasOwnProperty.call(object, key);
    }

    function fail(message) {
      throw new TypeError(message);
    }

    function requireRecord(value, path) {
      if (!isRecord(value)) {
        fail(path + " must be an object.");
      }
      return value;
    }

    function requireNonemptyString(value, path) {
      if (typeof value !== "string" || value.trim() === "") {
        fail(path + " must be a non-empty string.");
      }
      return value;
    }

    function assertAllowedKeys(value, allowed, path) {
      Object.keys(value).forEach(function (key) {
        if (allowed.indexOf(key) === -1) {
          fail(path + " contains an unknown field: " + key + ".");
        }
      });
    }

    function normalizeText(value, fallback, path) {
      if (value === undefined) {
        return fallback;
      }
      if (typeof value !== "string") {
        fail(path + " must be a string.");
      }
      return value;
    }

    function normalizeRating(value, fallback, path) {
      if (value === undefined) {
        return fallback;
      }
      if (value === null) {
        return fallback === "not_applicable" ? fallback : null;
      }
      if (value === true) {
        return "pass";
      }
      if (value === false) {
        return "fail";
      }
      if (typeof value !== "string" || !RATING_SET[value]) {
        fail(
          path +
            " must be null, a boolean, or one of: " +
            RATING_VALUES.join(", ") +
            ".",
        );
      }
      return value;
    }

    function normalizeIssueTags(value, fallback, path) {
      if (value === undefined) {
        return fallback.slice();
      }
      if (!Array.isArray(value)) {
        fail(path + " must be an array.");
      }

      var seen = Object.create(null);
      value.forEach(function (tag, index) {
        if (typeof tag !== "string" || !own(ISSUE_TAG_ORDER, tag)) {
          fail(path + "[" + index + "] is not a recognized issue tag.");
        }
        seen[tag] = true;
      });
      return ISSUE_TAGS.filter(function (tag) {
        return seen[tag] === true;
      });
    }

    function canonicalJson(value, stack) {
      if (value === null) {
        return "null";
      }
      if (typeof value === "string" || typeof value === "boolean") {
        return JSON.stringify(value);
      }
      if (typeof value === "number") {
        if (!Number.isFinite(value)) {
          fail("The run payload contains a non-finite number.");
        }
        return JSON.stringify(value);
      }
      if (typeof value !== "object") {
        fail("The run payload must contain only JSON-compatible values.");
      }
      if (stack.indexOf(value) !== -1) {
        fail("The run payload must not contain circular references.");
      }

      stack.push(value);
      var serialized;
      if (Array.isArray(value)) {
        serialized =
          "[" +
          value
            .map(function (item) {
              return canonicalJson(item, stack);
            })
            .join(",") +
          "]";
      } else {
        serialized =
          "{" +
          Object.keys(value)
            .filter(function (key) {
              return key !== "manual_review";
            })
            .sort()
            .map(function (key) {
              return JSON.stringify(key) + ":" + canonicalJson(value[key], stack);
            })
            .join(",") +
          "}";
      }
      stack.pop();
      return serialized;
    }

    function fnv1a(text, seed) {
      var hash = seed >>> 0;
      for (var index = 0; index < text.length; index += 1) {
        hash ^= text.charCodeAt(index);
        hash = Math.imul(hash, 0x01000193) >>> 0;
      }
      return hash >>> 0;
    }

    function hex32(value) {
      return (value >>> 0).toString(16).padStart(8, "0");
    }

    function fingerprintRun(payload) {
      requireRecord(payload, "payload");
      var canonical = canonicalJson(payload, []);
      var first = fnv1a(canonical, 0x811c9dc5);
      var second = fnv1a(canonical, (0x9e3779b9 ^ canonical.length) >>> 0);
      return "v1-" + hex32(first) + hex32(second);
    }

    function storageKey(fingerprint) {
      requireNonemptyString(fingerprint, "fingerprint");
      return STORAGE_KEY_PREFIX + fingerprint;
    }

    function readMeasureRange(run, path) {
      requireRecord(run, path);
      var input = run.inference_input;
      if (input === undefined || input === null) {
        return null;
      }
      requireRecord(input, path + ".inference_input");
      var range = input.measure_range;
      if (range === undefined || range === null) {
        return null;
      }
      if (
        !Array.isArray(range) ||
        range.length !== 2 ||
        !Number.isInteger(range[0]) ||
        !Number.isInteger(range[1]) ||
        range[0] < 1 ||
        range[1] < range[0]
      ) {
        fail(
          path +
            ".inference_input.measure_range must be null or a positive inclusive [start, end] pair.",
        );
      }
      return [range[0], range[1]];
    }

    function rangeToken(run, path) {
      var range = readMeasureRange(run, path);
      return range === null ? "none" : range[0] + "-" + range[1];
    }

    function caseKey(result, run, index) {
      requireRecord(result, "result");
      var sourceId = requireNonemptyString(result.source_id, "result.source_id");
      if (!Number.isInteger(index) || index < 0) {
        fail("index must be a non-negative integer.");
      }

      var token = rangeToken(run, "run");
      var key = sourceId + "::range:" + token;
      var runs = Array.isArray(result.inference_runs)
        ? result.inference_runs
        : [];
      var matchingIndexes = [];
      runs.forEach(function (candidate, candidateIndex) {
        if (
          rangeToken(
            candidate,
            "result.inference_runs[" + candidateIndex + "]",
          ) === token
        ) {
          matchingIndexes.push(candidateIndex);
        }
      });
      if (matchingIndexes.length > 1) {
        var occurrence = matchingIndexes.indexOf(index);
        if (occurrence === -1) {
          occurrence = index;
        }
        key += "::occurrence:" + (occurrence + 1);
      }
      return key;
    }

    function validatePayload(payload) {
      requireRecord(payload, "payload");
      if (!Array.isArray(payload.results)) {
        fail("payload.results must be an array.");
      }
      var seen = Object.create(null);
      payload.results.forEach(function (result, resultIndex) {
        var path = "payload.results[" + resultIndex + "]";
        requireRecord(result, path);
        var sourceId = requireNonemptyString(result.source_id, path + ".source_id");
        requireNonemptyString(result.piece_id, path + ".piece_id");
        if (own(seen, sourceId)) {
          fail("payload.results contains duplicate source_id: " + sourceId + ".");
        }
        seen[sourceId] = true;
        if (!Array.isArray(result.inference_runs)) {
          fail(path + ".inference_runs must be an array.");
        }
      });
      return payload;
    }

    function hasEvidence(run) {
      return ["retrieval_probe", "generated_answer"].some(function (field) {
        var output = run[field];
        if (!isRecord(output)) {
          return false;
        }
        return ["evidence", "expert_evidence_ids"].some(function (listField) {
          return Array.isArray(output[listField]) && output[listField].length > 0;
        });
      });
    }

    function semanticCaseStatus(run) {
      if (!isRecord(run)) {
        return "pending";
      }
      var semantic = run.semantic_evaluation;
      if (!isRecord(semantic) || !isRecord(semantic.aggregate)) {
        return "pending";
      }
      var aggregate = semantic.aggregate;
      if (aggregate.status === "fail") {
        return "fail";
      }
      if (
        aggregate.status === "human_review"
        || aggregate.status === "review"
      ) {
        return "review";
      }
      if (
        aggregate.status === "pass" &&
        aggregate.reliable_rag_llm_pass === true
      ) {
        return "reliable";
      }
      if (aggregate.status === "pass") {
        return "pass";
      }
      return "pending";
    }

    function semanticQuestionStatus(result) {
      if (!isRecord(result) || !Array.isArray(result.inference_runs)) {
        return "pending";
      }
      var statuses = result.inference_runs.map(semanticCaseStatus);
      if (statuses.length === 0) {
        return "pending";
      }
      if (statuses.indexOf("pending") !== -1) {
        return "pending";
      }
      if (statuses.indexOf("fail") !== -1) {
        return "fail";
      }
      if (statuses.indexOf("review") !== -1) {
        return "review";
      }
      if (
        statuses.every(function (status) {
          return status === "reliable";
        })
      ) {
        return "reliable";
      }
      return "pass";
    }

    function semanticStatusCounts(payload) {
      validatePayload(payload);
      var cases = {
        reliable: 0,
        pass: 0,
        review: 0,
        fail: 0,
        pending: 0,
      };
      var questions = {
        reliable: 0,
        pass: 0,
        review: 0,
        fail: 0,
        pending: 0,
      };
      payload.results.forEach(function (result) {
        questions[semanticQuestionStatus(result)] += 1;
        result.inference_runs.forEach(function (run) {
          cases[semanticCaseStatus(run)] += 1;
        });
      });
      return {
        cases: cases,
        questions: questions,
      };
    }

    function semanticRunProgress(payload) {
      validatePayload(payload);
      var counts = semanticStatusCounts(payload);
      var totalCases = Object.keys(counts.cases).reduce(function (total, key) {
        return total + counts.cases[key];
      }, 0);
      var judgedCases = totalCases - counts.cases.pending;
      var run = isRecord(payload.run) ? payload.run : {};
      var phases = isRecord(payload.summary) && isRecord(payload.summary.phases)
        ? payload.summary.phases
        : {};
      var judgePhase = isRecord(phases.judge) ? phases.judge : {};
      var declaredComplete =
        run.status === "complete" || judgePhase.status === "complete";
      var judgeComplete =
        totalCases > 0 && judgedCases === totalCases && declaredComplete;
      return {
        total_cases: totalCases,
        judged_cases: judgedCases,
        pending_cases: counts.cases.pending,
        completion_percent: roundedPercent(judgedCases, totalCases),
        declared_complete: declaredComplete,
        judge_complete: judgeComplete,
      };
    }

    function normalizeReferenceItems(items) {
      if (!Array.isArray(items)) {
        return [];
      }
      var seen = Object.create(null);
      var normalized = [];
      items.forEach(function (item, index) {
        if (!isRecord(item)) {
          return;
        }
        var text = item.text || item.reference_text;
        if (typeof text !== "string" || text.trim() === "") {
          return;
        }
        var id = item.reference_id;
        if (typeof id !== "string" || id.trim() === "") {
          id = "R" + String(index + 1).padStart(3, "0");
        }
        var key = id + "\u0000" + text.trim();
        if (own(seen, key)) {
          return;
        }
        seen[key] = true;
        var value = {
          reference_id: id,
          text: text.trim(),
        };
        if (typeof item.status === "string" && item.status.trim() !== "") {
          value.status = item.status;
        }
        ["scope", "scope_authority", "knowledge_unit_id"].forEach(function (key) {
          if (typeof item[key] === "string" && item[key].trim() !== "") {
            value[key] = item[key];
          }
        });
        if (Array.isArray(item.flags)) {
          value.flags = item.flags.slice();
        }
        if (Array.isArray(item.candidate_ids)) {
          value.candidate_ids = item.candidate_ids.slice();
        }
        normalized.push(value);
      });
      return normalized;
    }

    function caseReferenceAuthority(run) {
      if (!isRecord(run)) {
        return { source: null, items: [] };
      }
      var semantic = isRecord(run.semantic_evaluation)
        ? run.semantic_evaluation
        : {};
      var packet = isRecord(semantic.reference_packet)
        ? semantic.reference_packet
        : {};
      var authority = isRecord(semantic.reference_authority)
        ? semantic.reference_authority
        : {};
      var input = isRecord(run.inference_input) ? run.inference_input : {};
      var caseReferenceAuthority = isRecord(input.case_reference_authority)
        ? input.case_reference_authority
        : {};
      var directSources = [
        [
          caseReferenceAuthority.source
            ? "inference_input.case_reference_authority:" +
              caseReferenceAuthority.source
            : "inference_input.case_reference_authority.items",
          caseReferenceAuthority.items,
        ],
        [
          "authoritative_reference_items",
          run.authoritative_reference_items,
        ],
        [
          "semantic_evaluation.authoritative_reference_items",
          semantic.authoritative_reference_items,
        ],
        [
          "semantic_evaluation.reference_items",
          semantic.reference_items,
        ],
        [
          "semantic_evaluation.reference_authority.items",
          authority.items,
        ],
        [
          "semantic_evaluation.reference_packet.authoritative_reference_items",
          packet.authoritative_reference_items,
        ],
        [
          "inference_input.authoritative_reference_items",
          input.authoritative_reference_items,
        ],
      ];
      for (var sourceIndex = 0; sourceIndex < directSources.length; sourceIndex += 1) {
        var directItems = normalizeReferenceItems(
          Array.isArray(directSources[sourceIndex][1])
            ? directSources[sourceIndex][1].filter(function (item) {
                return (
                  !isRecord(item) ||
                  item.scope_authority !== "retrieved_expert_factuality_only"
                );
              })
            : directSources[sourceIndex][1],
        );
        if (directItems.length > 0) {
          return {
            source: directSources[sourceIndex][0],
            items: directItems,
          };
        }
      }

      var frames = isRecord(semantic.frames) ? semantic.frames : {};
      var frameNames = ["claim_alignment", "contradiction_first"];
      for (var frameIndex = 0; frameIndex < frameNames.length; frameIndex += 1) {
        var frameName = frameNames[frameIndex];
        var frame = isRecord(frames[frameName]) ? frames[frameName] : {};
        var assessment = isRecord(frame.assessment) ? frame.assessment : {};
        var assessedItems = normalizeReferenceItems(
          Array.isArray(assessment.reference_assessments)
            ? assessment.reference_assessments.filter(function (item) {
                return (
                  !isRecord(item) ||
                  item.scope_authority !== "retrieved_expert_factuality_only"
                );
              })
            : assessment.reference_assessments,
        );
        if (assessedItems.length > 0) {
          return {
            source:
              "semantic_evaluation.frames." +
              frameName +
              ".assessment.reference_assessments",
            items: assessedItems,
          };
        }
      }
      return { source: null, items: [] };
    }

    function embeddedQuestionReview(result, path) {
      var review = result.manual_review;
      if (review === undefined || review === null) {
        review = {};
      }
      requireRecord(review, path);
      assertAllowedKeys(
        review,
        ["paraphrase_natural", "notes", "issue_tags"],
        path,
      );
      return {
        paraphrase_natural: normalizeRating(
          review.paraphrase_natural,
          null,
          path + ".paraphrase_natural",
        ),
        notes: normalizeText(review.notes, "", path + ".notes"),
        issue_tags: normalizeIssueTags(
          review.issue_tags,
          [],
          path + ".issue_tags",
        ),
      };
    }

    function embeddedCaseReview(run, path) {
      var review = run.manual_review;
      if (review === undefined || review === null) {
        review = {};
      }
      requireRecord(review, path);
      assertAllowedKeys(
        review,
        [
          "range_correct",
          "evidence_relevant",
          "answer_acceptable",
          "notes",
          "issue_tags",
        ],
        path,
      );
      var noRange = readMeasureRange(run, path.replace(/\.manual_review$/, "")) === null;
      var noEvidence = !hasEvidence(run);
      return {
        range_correct: normalizeRating(
          review.range_correct,
          noRange ? "not_applicable" : null,
          path + ".range_correct",
        ),
        evidence_relevant: normalizeRating(
          review.evidence_relevant,
          noEvidence ? "not_applicable" : null,
          path + ".evidence_relevant",
        ),
        answer_acceptable: normalizeRating(
          review.answer_acceptable,
          null,
          path + ".answer_acceptable",
        ),
        notes: normalizeText(review.notes, "", path + ".notes"),
        issue_tags: normalizeIssueTags(
          review.issue_tags,
          [],
          path + ".issue_tags",
        ),
      };
    }

    function createDraft(payload) {
      validatePayload(payload);
      var draft = {
        schema_version: DRAFT_SCHEMA_VERSION,
        source_run_fingerprint: fingerprintRun(payload),
        questions: {},
      };

      payload.results.forEach(function (result, resultIndex) {
        var sourceId = result.source_id;
        var questionPath =
          "payload.results[" + resultIndex + "].manual_review";
        var question = embeddedQuestionReview(result, questionPath);
        question.cases = {};

        result.inference_runs.forEach(function (run, runIndex) {
          var key = caseKey(result, run, runIndex);
          if (own(question.cases, key)) {
            fail(
              "Could not create a unique case key for " +
                sourceId +
                " at inference run " +
                runIndex +
                ".",
            );
          }
          question.cases[key] = embeddedCaseReview(
            run,
            "payload.results[" +
              resultIndex +
              "].inference_runs[" +
              runIndex +
              "].manual_review",
          );
        });
        draft.questions[sourceId] = question;
      });
      return draft;
    }

    function normalizeQuestion(candidate, base, path) {
      requireRecord(candidate, path);
      assertAllowedKeys(
        candidate,
        ["paraphrase_natural", "notes", "issue_tags", "cases"],
        path,
      );
      var normalized = {
        paraphrase_natural: normalizeRating(
          candidate.paraphrase_natural,
          base.paraphrase_natural,
          path + ".paraphrase_natural",
        ),
        notes: normalizeText(candidate.notes, base.notes, path + ".notes"),
        issue_tags: normalizeIssueTags(
          candidate.issue_tags,
          base.issue_tags,
          path + ".issue_tags",
        ),
        cases: {},
      };

      var candidateCases = candidate.cases;
      if (candidateCases === undefined) {
        candidateCases = {};
      }
      requireRecord(candidateCases, path + ".cases");
      Object.keys(candidateCases).forEach(function (key) {
        if (!own(base.cases, key)) {
          fail(path + ".cases contains an unknown case key: " + key + ".");
        }
      });

      Object.keys(base.cases).forEach(function (key) {
        var baseCase = base.cases[key];
        var candidateCase = own(candidateCases, key) ? candidateCases[key] : {};
        var casePath = path + ".cases[" + JSON.stringify(key) + "]";
        requireRecord(candidateCase, casePath);
        assertAllowedKeys(
          candidateCase,
          [
            "range_correct",
            "evidence_relevant",
            "answer_acceptable",
            "notes",
            "issue_tags",
          ],
          casePath,
        );
        normalized.cases[key] = {
          range_correct: normalizeRating(
            candidateCase.range_correct,
            baseCase.range_correct,
            casePath + ".range_correct",
          ),
          evidence_relevant: normalizeRating(
            candidateCase.evidence_relevant,
            baseCase.evidence_relevant,
            casePath + ".evidence_relevant",
          ),
          answer_acceptable: normalizeRating(
            candidateCase.answer_acceptable,
            baseCase.answer_acceptable,
            casePath + ".answer_acceptable",
          ),
          notes: normalizeText(
            candidateCase.notes,
            baseCase.notes,
            casePath + ".notes",
          ),
          issue_tags: normalizeIssueTags(
            candidateCase.issue_tags,
            baseCase.issue_tags,
            casePath + ".issue_tags",
          ),
        };
      });
      return normalized;
    }

    function normalizeDraft(payload, candidate) {
      var base = createDraft(payload);
      requireRecord(candidate, "candidate");
      assertAllowedKeys(
        candidate,
        ["schema_version", "source_run_fingerprint", "questions"],
        "candidate",
      );
      if (candidate.schema_version !== DRAFT_SCHEMA_VERSION) {
        fail(
          "candidate.schema_version must equal " + DRAFT_SCHEMA_VERSION + ".",
        );
      }
      if (candidate.source_run_fingerprint !== base.source_run_fingerprint) {
        fail("The imported review belongs to a different evaluation run.");
      }

      var candidateQuestions = candidate.questions;
      if (candidateQuestions === undefined) {
        candidateQuestions = {};
      }
      requireRecord(candidateQuestions, "candidate.questions");
      Object.keys(candidateQuestions).forEach(function (sourceId) {
        if (!own(base.questions, sourceId)) {
          fail(
            "candidate.questions contains an unknown source_id: " +
              sourceId +
              ".",
          );
        }
      });

      var normalized = {
        schema_version: DRAFT_SCHEMA_VERSION,
        source_run_fingerprint: base.source_run_fingerprint,
        questions: {},
      };
      Object.keys(base.questions).forEach(function (sourceId) {
        normalized.questions[sourceId] = normalizeQuestion(
          own(candidateQuestions, sourceId)
            ? candidateQuestions[sourceId]
            : {},
          base.questions[sourceId],
          "candidate.questions[" + JSON.stringify(sourceId) + "]",
        );
      });
      return normalized;
    }

    function validCompleteRating(value) {
      return typeof value === "string" && RATING_SET[value] === true;
    }

    function reviewNeedsNote(review, fields) {
      var needsNote = fields.some(function (field) {
        return review[field] === "fail" || review[field] === "uncertain";
      });
      return (
        !needsNote ||
        (typeof review.notes === "string" && review.notes.trim() !== "")
      );
    }

    function isCaseComplete(review) {
      if (!isRecord(review)) {
        return false;
      }
      var fields = [
        "range_correct",
        "evidence_relevant",
        "answer_acceptable",
      ];
      return (
        fields.every(function (field) {
          return validCompleteRating(review[field]);
        }) && reviewNeedsNote(review, fields)
      );
    }

    function caseReviewList(questionReview, caseReviews) {
      var value = caseReviews;
      if (value === undefined && isRecord(questionReview)) {
        value = questionReview.cases;
      }
      if (Array.isArray(value)) {
        return value;
      }
      if (isRecord(value)) {
        return Object.keys(value).map(function (key) {
          return value[key];
        });
      }
      return [];
    }

    function isQuestionComplete(questionReview, caseReviews) {
      if (!isRecord(questionReview)) {
        return false;
      }
      var cases = caseReviewList(questionReview, caseReviews);
      return (
        validCompleteRating(questionReview.paraphrase_natural) &&
        reviewNeedsNote(questionReview, ["paraphrase_natural"]) &&
        cases.length > 0 &&
        cases.every(isCaseComplete)
      );
    }

    function reviewHasFlag(review, fields) {
      if (!isRecord(review)) {
        return false;
      }
      if (
        Array.isArray(review.issue_tags) &&
        review.issue_tags.length > 0
      ) {
        return true;
      }
      return fields.some(function (field) {
        return review[field] === "fail" || review[field] === "uncertain";
      });
    }

    function isFlagged(questionReview, caseReviews) {
      if (!isRecord(questionReview)) {
        return false;
      }
      if (reviewHasFlag(questionReview, ["paraphrase_natural"])) {
        return true;
      }
      return caseReviewList(questionReview, caseReviews).some(function (review) {
        return reviewHasFlag(review, [
          "range_correct",
          "evidence_relevant",
          "answer_acceptable",
        ]);
      });
    }

    function roundedPercent(completed, total) {
      if (total === 0) {
        return 0;
      }
      return Math.round((completed / total) * 1000) / 10;
    }

    function emptyProgressBucket() {
      return {
        total_questions: 0,
        completed_questions: 0,
        incomplete_questions: 0,
        flagged_questions: 0,
        total_cases: 0,
        completed_cases: 0,
        incomplete_cases: 0,
        flagged_cases: 0,
      };
    }

    function computeProgress(payload, draft) {
      validatePayload(payload);
      var normalized =
        draft === undefined ? createDraft(payload) : normalizeDraft(payload, draft);
      var progress = emptyProgressBucket();
      progress.source_run_fingerprint = normalized.source_run_fingerprint;
      progress.by_piece = {};

      payload.results.forEach(function (result) {
        var pieceId = result.piece_id;
        if (!own(progress.by_piece, pieceId)) {
          progress.by_piece[pieceId] = emptyProgressBucket();
        }
        var piece = progress.by_piece[pieceId];
        var question = normalized.questions[result.source_id];
        var caseReviews = Object.keys(question.cases).map(function (key) {
          return question.cases[key];
        });

        progress.total_questions += 1;
        piece.total_questions += 1;
        if (isQuestionComplete(question, caseReviews)) {
          progress.completed_questions += 1;
          piece.completed_questions += 1;
        }
        if (isFlagged(question, caseReviews)) {
          progress.flagged_questions += 1;
          piece.flagged_questions += 1;
        }

        caseReviews.forEach(function (review) {
          progress.total_cases += 1;
          piece.total_cases += 1;
          if (isCaseComplete(review)) {
            progress.completed_cases += 1;
            piece.completed_cases += 1;
          }
          if (
            reviewHasFlag(review, [
              "range_correct",
              "evidence_relevant",
              "answer_acceptable",
            ])
          ) {
            progress.flagged_cases += 1;
            piece.flagged_cases += 1;
          }
        });
      });

      progress.incomplete_questions =
        progress.total_questions - progress.completed_questions;
      progress.incomplete_cases =
        progress.total_cases - progress.completed_cases;
      progress.completion_percent = roundedPercent(
        progress.completed_questions,
        progress.total_questions,
      );
      progress.all_complete =
        progress.total_questions > 0 &&
        progress.completed_questions === progress.total_questions;

      Object.keys(progress.by_piece).forEach(function (pieceId) {
        var piece = progress.by_piece[pieceId];
        piece.incomplete_questions =
          piece.total_questions - piece.completed_questions;
        piece.incomplete_cases = piece.total_cases - piece.completed_cases;
      });
      return progress;
    }

    function exportReview(payload, draft) {
      var normalized = normalizeDraft(payload, draft);
      return JSON.parse(JSON.stringify(normalized));
    }

    return Object.freeze({
      RATING_VALUES: RATING_VALUES,
      ISSUE_TAGS: ISSUE_TAGS,
      fingerprintRun: fingerprintRun,
      storageKey: storageKey,
      caseKey: caseKey,
      semanticCaseStatus: semanticCaseStatus,
      semanticQuestionStatus: semanticQuestionStatus,
      semanticStatusCounts: semanticStatusCounts,
      semanticRunProgress: semanticRunProgress,
      caseReferenceAuthority: caseReferenceAuthority,
      createDraft: createDraft,
      normalizeDraft: normalizeDraft,
      isCaseComplete: isCaseComplete,
      isQuestionComplete: isQuestionComplete,
      isFlagged: isFlagged,
      computeProgress: computeProgress,
      exportReview: exportReview,
    });
  },
);
