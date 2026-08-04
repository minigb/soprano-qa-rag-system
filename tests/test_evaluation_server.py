"""Focused tests for the local evaluation comparison viewer."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest

from evaluation import server as viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
REAL_ORIGINAL_RESULTS = EVALUATION_ROOT / "qualitative.json"
REAL_SYNTHESIZED_RESULTS = (
    EVALUATION_ROOT / "synthesized_question_results.json"
)

SOURCE_ID = "kim-in-flowery-clouds-07"
PIECE_ID = "in-flowery-clouds"
KNOWLEDGE_UNIT_ID = "in-flowery-clouds-ku-008"
ORIGINAL_CASE_ID = f"{SOURCE_ID}__native-no-range-01"
REFERENCE_TEXT = (
    "작곡가는 박자와 조성을 바꾸어 분위기를 전환합니다."
)


def _case_reference_authority() -> dict[str, object]:
    return {
        "source": "applicable_finalized_knowledge_units",
        "items": [
            {
                "reference_id": "R001",
                "text": REFERENCE_TEXT,
                "scope": "curated_knowledge_unit_answer",
                "flags": [],
                "scope_authority": "finalized_knowledge_unit",
                "reference_role": "required_any",
                "measure_status": "whole_piece",
                "measure_ranges": [],
                "source_sentence_index": None,
                "source_claim_index": None,
                "knowledge_unit_id": KNOWLEDGE_UNIT_ID,
            }
        ],
        "manual_review_required": False,
        "manual_review_reason": None,
    }


def _pipeline_result(answer: str = REFERENCE_TEXT) -> dict[str, object]:
    return {
        "piece_id": PIECE_ID,
        "scope": "whole_piece",
        "measure_range": None,
        "answer": answer,
        "generation_mode": "expert_verbatim",
        "answer_basis": "authoritative_expert",
        "unavailable_reason": None,
        "has_primary_grounding": True,
        "has_selected_range_grounding": False,
        "has_confirmed_local_examples": False,
        "pipeline": "soprano_qa",
        "model": {
            "path": "/models/Qwen3-8B.gguf",
            "checkpoint_exists": True,
            "backend": "llama-cpp-python",
        },
        "retrieval": {
            "configured_mode": "hybrid",
            "active_mode": "hybrid",
            "dense_available": True,
            "last_dense_error": None,
            "last_search_mode": "hybrid",
            "query_route": "hybrid",
            "dense_attempted": True,
            "dense_contributed": True,
        },
        "evidence": [
            {
                "id": KNOWLEDGE_UNIT_ID,
                "kind": "expert",
            }
        ],
        "evidence_notices": [],
    }


def _shared_question_metadata() -> dict[str, object]:
    return {
        "source_id": SOURCE_ID,
        "piece_id": PIECE_ID,
        "annotator": "kim",
        "inventory_schema_version": "9.0",
        "original_question": "박자가 바뀌는 이유는 무엇인가요?",
        "paraphrased_question": "이 곡에서 박자가 바뀌는 이유는?",
        "review_status": "retrievable",
    }


def _original_payload() -> dict[str, object]:
    question = {
        **_shared_question_metadata(),
        "knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
        "retrieval_eligible_knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
        "measure_range_hints": [],
        "inference_measure_ranges": [],
        "excluded_inference_measure_ranges": [],
        "inference_scope": "whole_piece",
        "reference_material": {
            "linked_knowledge_units": [
                {
                    "knowledge_unit_id": KNOWLEDGE_UNIT_ID,
                    "answer": REFERENCE_TEXT,
                    "rewrite_status": "ready",
                    "measure_status": "whole_piece",
                    "measure_ranges": [],
                }
            ]
        },
        "inference_runs": [
            {
                "case_id": ORIGINAL_CASE_ID,
                "status": "completed",
                "inference_input": {
                    "piece_id": PIECE_ID,
                    "question": "이 곡에서 박자가 바뀌는 이유는?",
                    "measure_range": None,
                    "measure_range_applied": False,
                    "case_kind": "native_no_range",
                    "question_provenance": "inventory_paraphrase",
                    "case_reference_authority": (
                        _case_reference_authority()
                    ),
                },
                "expected_retrieval": {
                    "linked_knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
                    "retrieval_eligible_knowledge_unit_ids": [
                        KNOWLEDGE_UNIT_ID
                    ],
                    "range_applicable_knowledge_unit_ids": [
                        KNOWLEDGE_UNIT_ID
                    ],
                },
                "pipeline_result": _pipeline_result(),
                "retrieval_diagnostics": {
                    "semantic_target_first_rank": 1,
                },
                "error": None,
            }
        ],
    }
    return {
        "artifact_type": viewer.QUALITATIVE_ARTIFACT_TYPE,
        "schema_version": viewer.QUALITATIVE_SCHEMA_VERSION,
        "run": {
            "status": "completed",
            "generate": True,
            "target_pieces": [PIECE_ID],
        },
        "questions": [question],
        "summary": {"case_count": 1},
    }


def _synthesized_payload() -> dict[str, object]:
    variants = [
        {
            "variant_id": f"{SOURCE_ID}-syn-{index:02d}",
            "variant_index": index,
            "question": f"합성 질문 {index}: 박자 변화의 이유는?",
        }
        for index in range(1, 4)
    ]
    runs = []
    for variant in variants:
        index = int(variant["variant_index"])
        runs.append(
            {
                "case_id": f"{SOURCE_ID}-syn-{index:02d}__native-no-range-01",
                "status": "completed",
                "inference_input": {
                    "piece_id": PIECE_ID,
                    "source_id": SOURCE_ID,
                    "question": variant["question"],
                    "synthesized_variant_id": variant["variant_id"],
                    "variant_index": index,
                    "measure_range": None,
                    "measure_range_applied": False,
                    "case_kind": "native_no_range",
                    "question_provenance": "synthesized_variant",
                    "expected_knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
                    "expected_retrieval_eligible_knowledge_unit_ids": [
                        KNOWLEDGE_UNIT_ID
                    ],
                    "range_applicable_knowledge_unit_ids": [
                        KNOWLEDGE_UNIT_ID
                    ],
                    "case_reference_authority": (
                        _case_reference_authority()
                    ),
                },
                "retrieval_probe": {
                    "piece_id": PIECE_ID,
                    "evidence": [
                        {"id": KNOWLEDGE_UNIT_ID, "kind": "expert"}
                    ],
                },
                "generated_answer": _pipeline_result(
                    f"합성 질문 {index}에 대한 근거 답변입니다."
                ),
                "error": None,
            }
        )

    return {
        "artifact_type": viewer.SYNTHESIZED_ARTIFACT_TYPE,
        "schema_version": viewer.SYNTHESIZED_SCHEMA_VERSION,
        "run": {
            "status": "complete",
            "generate": True,
            "target_pieces": [PIECE_ID],
        },
        "results": [
            {
                **_shared_question_metadata(),
                "knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
                "expected_retrieval_eligible_knowledge_unit_ids": [
                    KNOWLEDGE_UNIT_ID
                ],
                "measure_range_hints": [],
                "inference_measure_ranges": [],
                "excluded_inference_measure_ranges": [],
                "inference_scope": "whole_piece",
                "authoritative_reference": {
                    "linked_knowledge_units": [
                        {
                            "knowledge_unit_id": KNOWLEDGE_UNIT_ID,
                            "answer": REFERENCE_TEXT,
                            "rewrite_status": "ready",
                            "measure_status": "whole_piece",
                            "measure_ranges": [],
                        }
                    ]
                },
                "synthesized_question_variants": variants,
                "inference_runs": runs,
            }
        ],
        "summary": {"case_count": 3},
    }


def _quality_review_payload(
    original_payload: dict[str, object],
    synthesized_payload: dict[str, object],
    *,
    flagged_case_ids: set[str] | None = None,
) -> dict[str, object]:
    comparison = viewer.build_comparison_payload(
        original_payload,
        synthesized_payload,
    )
    flagged_case_ids = flagged_case_ids or set()
    assessments = []
    for result in comparison["results"]:
        for run in result["inference_runs"]:
            assessment: dict[str, object] = {
                "case_id": run["case_id"],
                "semantic_hash": viewer.semantic_review_case_hash(result, run),
                "red_flag": run["case_id"] in flagged_case_ids,
            }
            if assessment["red_flag"]:
                assessment.update(
                    {
                        "severity": "high",
                        "reason_codes": ["meaning_changed"],
                        "rationale": (
                            "The generated answer changes the finalized "
                            "reference meaning."
                        ),
                    }
                )
            assessments.append(assessment)
    return {
        "artifact_type": viewer.QUALITY_REVIEW_ARTIFACT_TYPE,
        "schema_version": viewer.QUALITY_REVIEW_SCHEMA_VERSION,
        "review_status": "complete",
        "review_method": viewer.QUALITY_REVIEW_METHOD,
        "assessments": assessments,
    }


class EvaluationViewerServerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.original_results_file = root / "original.json"
        self.synthesized_results_file = root / "synthesized.json"
        self.quality_review_file = root / "semantic-quality-review.json"
        self.original_payload = _original_payload()
        self.synthesized_payload = _synthesized_payload()
        self._write_fixtures()

        self.server = viewer.create_server(
            port=0,
            evaluation_root=EVALUATION_ROOT,
            original_results_file=self.original_results_file,
            synthesized_results_file=self.synthesized_results_file,
            quality_review_file=self.quality_review_file,
        )
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _write_fixtures(self) -> None:
        self.original_results_file.write_text(
            json.dumps(self.original_payload, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        self.synthesized_results_file.write_text(
            json.dumps(
                self.synthesized_payload,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)

    def request(self, path: str) -> tuple[int, dict[str, str], bytes]:
        port = self.server.server_address[1]
        connection = HTTPConnection("127.0.0.1", port, timeout=2)
        self.addCleanup(connection.close)
        connection.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def test_api_combines_every_original_and_synthesized_run_without_mutation(
        self,
    ) -> None:
        original_before = self.original_results_file.read_bytes()
        synthesized_before = self.synthesized_results_file.read_bytes()

        status, headers, body = self.request("/api/results")
        comparison = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(
            comparison["artifact_type"],
            viewer.COMPARISON_ARTIFACT_TYPE,
        )
        self.assertEqual(
            comparison["schema_version"],
            viewer.COMPARISON_SCHEMA_VERSION,
        )
        self.assertEqual(
            comparison["quality_review"],
            {"review_status": "not_available"},
        )
        self.assertEqual(
            comparison["summary"],
            {
                "question_group_count": 1,
                "scenario_count": 1,
                "paired_scenario_count": 1,
                "original_only_scenario_count": 0,
                "original_case_count": 1,
                "synthesized_case_count": 3,
                "total_case_count": 4,
                "per_piece": {
                    PIECE_ID: {
                        "scenarios": 1,
                        "original": 1,
                        "synthesized": 3,
                        "total": 4,
                    }
                },
            },
        )
        self.assertEqual(len(comparison["results"]), 1)
        result = comparison["results"][0]
        self.assertEqual(result["source_id"], ORIGINAL_CASE_ID)
        self.assertEqual(result["viewer_scenario_id"], ORIGINAL_CASE_ID)
        self.assertEqual(result["source_evaluation_id"], SOURCE_ID)
        self.assertEqual(
            result["viewer_comparison"],
            {"original_case_count": 1, "synthesized_case_count": 3},
        )
        runs = result["inference_runs"]
        self.assertTrue(
            all(
                run["quality_review"]
                == {"review_status": "not_available"}
                for run in runs
            )
        )
        self.assertEqual(
            [run["viewer_question_origin"] for run in runs],
            ["original", "synthesized", "synthesized", "synthesized"],
        )
        self.assertEqual(
            [
                run["inference_input"].get("variant_index")
                for run in runs[1:]
            ],
            [1, 2, 3],
        )
        self.assertEqual(
            runs[0]["generated_answer"]["answer"],
            REFERENCE_TEXT,
        )
        self.assertEqual(
            comparison["source_artifacts"]["original"]["case_count"],
            1,
        )
        self.assertEqual(
            comparison["source_artifacts"]["synthesized"]["case_count"],
            3,
        )
        self.assertEqual(
            self.original_results_file.read_bytes(),
            original_before,
        )
        self.assertEqual(
            self.synthesized_results_file.read_bytes(),
            synthesized_before,
        )

    def test_normalizes_canonical_results_without_mutating_the_source(self) -> None:
        source = _original_payload()
        source_before = deepcopy(source)

        normalized = viewer.normalize_results_payload(source)

        self.assertEqual(source, source_before)
        self.assertEqual(
            normalized["viewer_projection"]["source_collection"],
            "questions",
        )
        self.assertEqual(len(normalized["results"]), 1)
        result = normalized["results"][0]
        self.assertEqual(
            result["authoritative_reference"]["linked_knowledge_units"][0][
                "knowledge_unit_id"
            ],
            KNOWLEDGE_UNIT_ID,
        )
        run = result["inference_runs"][0]
        self.assertEqual(
            run["inference_input"][
                "expected_retrieval_eligible_knowledge_unit_ids"
            ],
            [KNOWLEDGE_UNIT_ID],
        )
        generated = run["generated_answer"]
        self.assertFalse(generated["rag_llm_succeeded"])
        self.assertTrue(generated["authoritative_expert_succeeded"])
        self.assertTrue(generated["grounded_answer_succeeded"])
        self.assertEqual(
            run["retrieval_probe"]["evidence"][0]["id"],
            KNOWLEDGE_UNIT_ID,
        )

    def test_complete_quality_review_is_attached_to_every_exact_run(self) -> None:
        flagged_case_id = (
            f"{SOURCE_ID}-syn-02__native-no-range-01"
        )
        review = _quality_review_payload(
            self.original_payload,
            self.synthesized_payload,
            flagged_case_ids={flagged_case_id},
        )
        self.quality_review_file.write_text(
            json.dumps(review, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        status, _, body = self.request("/api/results")
        comparison = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(
            comparison["quality_review"],
            {
                "artifact_type": viewer.QUALITY_REVIEW_ARTIFACT_TYPE,
                "schema_version": viewer.QUALITY_REVIEW_SCHEMA_VERSION,
                "review_status": "complete",
                "review_method": viewer.QUALITY_REVIEW_METHOD,
                "assessment_count": 4,
                "red_flag_count": 1,
            },
        )
        runs = comparison["results"][0]["inference_runs"]
        attached = {run["case_id"]: run["quality_review"] for run in runs}
        self.assertTrue(attached[flagged_case_id]["red_flag"])
        self.assertEqual(attached[flagged_case_id]["severity"], "high")
        self.assertEqual(
            attached[flagged_case_id]["reason_codes"],
            ["meaning_changed"],
        )
        passing = attached[ORIGINAL_CASE_ID]
        self.assertFalse(passing["red_flag"])
        self.assertNotIn("severity", passing)
        self.assertNotIn("reason_codes", passing)
        self.assertNotIn("rationale", passing)

    def test_quality_review_fails_closed_for_partial_duplicate_stale_or_invalid_assessments(
        self,
    ) -> None:
        valid = _quality_review_payload(
            self.original_payload,
            self.synthesized_payload,
        )
        invalid_reviews = {}

        partial = deepcopy(valid)
        partial["assessments"].pop()
        invalid_reviews["partial"] = partial

        duplicate = deepcopy(valid)
        duplicate["assessments"].append(
            deepcopy(duplicate["assessments"][0])
        )
        invalid_reviews["duplicate"] = duplicate

        stale = deepcopy(valid)
        stale["assessments"][0]["semantic_hash"] = "0" * 64
        invalid_reviews["stale"] = stale

        pass_with_flag_metadata = deepcopy(valid)
        pass_with_flag_metadata["assessments"][0]["rationale"] = (
            "Passing runs cannot carry flag-only metadata."
        )
        invalid_reviews["pass_with_flag_metadata"] = pass_with_flag_metadata

        red_flag_without_metadata = deepcopy(valid)
        red_flag_without_metadata["assessments"][0]["red_flag"] = True
        invalid_reviews["red_flag_without_metadata"] = red_flag_without_metadata

        for label, payload in invalid_reviews.items():
            with self.subTest(label=label):
                self.quality_review_file.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                status, _, body = self.request("/api/results")
                error = json.loads(body)
                self.assertEqual(status, 500)
                self.assertEqual(error["code"], "results_invalid")
                self.assertIn("quality review", error["error"])

    def test_semantic_hash_covers_every_judgment_relevant_field(self) -> None:
        comparison = viewer.build_comparison_payload(
            self.original_payload,
            self.synthesized_payload,
        )
        baseline_result = comparison["results"][0]
        baseline_run = baseline_result["inference_runs"][0]
        baseline_hash = viewer.semantic_review_case_hash(
            baseline_result,
            baseline_run,
        )

        def changed_hash(
            mutate: object,
        ) -> str:
            result = deepcopy(baseline_result)
            run = result["inference_runs"][0]
            mutate(result, run)
            return viewer.semantic_review_case_hash(result, run)

        mutations = {
            "piece": lambda result, run: result.__setitem__(
                "piece_id", "another-piece"
            ),
            "source evaluation id": lambda result, run: result.__setitem__(
                "source_evaluation_id", "another-source"
            ),
            "origin": lambda result, run: run.__setitem__(
                "viewer_question_origin", "synthesized"
            ),
            "case id": lambda result, run: run.__setitem__(
                "case_id", "another-case"
            ),
            "question": lambda result, run: run["inference_input"].__setitem__(
                "question", "다른 질문"
            ),
            "case kind": lambda result, run: run["inference_input"].__setitem__(
                "case_kind", "another_kind"
            ),
            "range": lambda result, run: run["inference_input"].__setitem__(
                "measure_range", [1, 2]
            ),
            "authority": lambda result, run: run["inference_input"][
                "case_reference_authority"
            ].__setitem__("manual_review_required", True),
            "answer": lambda result, run: run["generated_answer"].__setitem__(
                "answer", "다른 답변"
            ),
            "mode": lambda result, run: run["generated_answer"].__setitem__(
                "generation_mode", "llm"
            ),
            "basis": lambda result, run: run["generated_answer"].__setitem__(
                "answer_basis", "retrieved_evidence"
            ),
            "scope": lambda result, run: run["generated_answer"].__setitem__(
                "scope", "selected_range"
            ),
            "unavailable reason": lambda result, run: run[
                "generated_answer"
            ].__setitem__("unavailable_reason", "changed"),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                self.assertNotEqual(changed_hash(mutate), baseline_hash)

    def test_fails_closed_when_either_required_artifact_is_missing(self) -> None:
        for path in (
            self.original_results_file,
            self.synthesized_results_file,
        ):
            with self.subTest(missing=path.name):
                contents = path.read_bytes()
                path.unlink()
                try:
                    status, _, body = self.request("/api/results")
                finally:
                    path.write_bytes(contents)

                error = json.loads(body)
                self.assertEqual(status, 404)
                self.assertEqual(error["code"], "results_not_found")
                self.assertIn("Both original and synthesized", error["error"])

    def test_fails_closed_when_either_artifact_is_invalid(self) -> None:
        for path, valid_payload in (
            (self.original_results_file, self.original_payload),
            (self.synthesized_results_file, self.synthesized_payload),
        ):
            with self.subTest(invalid=path.name):
                invalid = deepcopy(valid_payload)
                invalid["run"]["fallback_used"] = True
                path.write_text(json.dumps(invalid), encoding="utf-8")
                try:
                    status, _, body = self.request("/api/results")
                finally:
                    self._write_fixtures()

                error = json.loads(body)
                self.assertEqual(status, 500)
                self.assertEqual(error["code"], "results_invalid")
                self.assertIn("removed runtime fields", error["error"])

    def test_fails_closed_for_empty_complete_artifacts(self) -> None:
        self.original_payload["questions"] = []
        self.synthesized_payload["results"] = []
        self._write_fixtures()

        status, _, body = self.request("/api/results")
        error = json.loads(body)

        self.assertEqual(status, 500)
        self.assertEqual(error["code"], "results_invalid")
        self.assertIn("complete evaluation question set", error["error"])

    def test_served_viewer_exposes_all_case_comparison_controls(self) -> None:
        status, _, body = self.request("/")
        html = body.decode("utf-8")

        self.assertEqual(status, 200)
        for marker in (
            'id="question-origin-filter"',
            'id="generation-mode-filter"',
            'id="run-status-filter"',
            'id="variant-filter"',
            'id="quality-review-filter"',
            'id="quality-review-banner"',
            'id="red-flags-page-link"',
        ):
            self.assertIn(marker, html)
        self.assertIn('data-viewer-mode="comparison"', html)
        self.assertIn("164개 원본 질문 추론", html)
        self.assertIn("483개 합성 질문 추론", html)
        self.assertNotIn('class="comparison-overview"', html)
        self.assertNotIn('id="finding-banner"', html)
        self.assertNotIn('class="summary-section"', html)
        self.assertNotIn('id="progress-track"', html)

    def test_red_flags_page_reuses_viewer_in_flagged_mode(self) -> None:
        status, _, body = self.request("/red-flags")
        html = body.decode("utf-8")

        self.assertEqual(status, 200)
        self.assertIn('data-viewer-mode="red-flags"', html)
        self.assertNotIn('data-viewer-mode="comparison"', html)
        self.assertIn('id="quality-review-filter"', html)
        self.assertIn('id="comparison-page-link"', html)

        status, _, body = self.request("/red-flags.html")
        self.assertEqual(status, 200)
        self.assertIn(
            'data-viewer-mode="red-flags"',
            body.decode("utf-8"),
        )

    def test_assets_require_and_separate_the_comparison_artifact(self) -> None:
        app = (EVALUATION_ROOT / "app.js").read_text(encoding="utf-8")
        review_state = (EVALUATION_ROOT / "review-state.js").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"soprano_qa_original_synthesized_answer_comparison"',
            app,
        )
        self.assertIn('payload.schema_version !== "1.0"', app)
        self.assertIn("viewer_question_origin", app)
        self.assertIn("questionOriginFilter", app)
        self.assertIn("renderQuestionWordingComparison", app)
        self.assertIn("comparisonCaseGroups", app)
        self.assertIn("case-tab-groups", app)
        self.assertIn('PAGE_MODE === "red-flags"', app)
        self.assertIn('return RED_FLAGS_PAGE ? "flagged" : "all"', app)
        self.assertIn("qualityReviewCategory", app)
        self.assertIn("renderQualityAssessment", app)
        self.assertIn("renderQualityReviewStatus", app)
        self.assertIn("case_reference_authority", review_state)
        self.assertNotIn("renderFinding", app)
        self.assertNotIn("renderSummary", app)

    def test_viewer_treats_authoritative_expert_as_a_successful_answer_path(
        self,
    ) -> None:
        app = (EVALUATION_ROOT / "app.js").read_text(encoding="utf-8")
        html = (EVALUATION_ROOT / "index.html").read_text(encoding="utf-8")

        helper_start = app.index("function isAuthoritativeExpertAnswer")
        helper_end = app.index("function semanticAggregate", helper_start)
        helpers = app[helper_start:helper_end]
        self.assertIn('generation_mode === "expert_verbatim"', helpers)
        self.assertIn('answer_basis === "authoritative_expert"', helpers)
        self.assertIn("grounded_answer_succeeded", helpers)
        self.assertIn("answerPathTone", helpers)
        self.assertIn(
            '<option value="authoritative_expert">Authoritative expert</option>',
            html,
        )
        self.assertIn("Pipeline answer", app)
        self.assertNotIn("Generated RAG+LLM answer", app)

    def test_server_does_not_classify_extractive_output_as_success(self) -> None:
        metadata = viewer._answer_path_success_metadata(
            {
                "generation_mode": "extractive",
                "answer_basis": "retrieval_extractive",
                "evidence": [{"id": "legacy-evidence"}],
            }
        )

        self.assertFalse(metadata["rag_llm_succeeded"])
        self.assertFalse(metadata["authoritative_expert_succeeded"])
        self.assertFalse(metadata["grounded_answer_succeeded"])

    def test_viewer_uses_finalized_units_without_raw_editorial_lineage(self) -> None:
        app = (EVALUATION_ROOT / "app.js").read_text(encoding="utf-8")
        comparison_start = app.index("function renderCaseAuthorityComparison")
        comparison_end = app.index(
            "function prettyDiagnosticLabel",
            comparison_start,
        )
        comparison = app[comparison_start:comparison_end]
        reference_start = app.index("function renderAuthoritativeReference")
        reference_end = app.index("function renderDetail", reference_start)
        reference = app[reference_start:reference_end]

        self.assertIn("ReviewState.caseReferenceAuthority", comparison)
        self.assertNotIn("fallbackHumanReference", app)
        self.assertNotIn("source_answer", comparison)
        self.assertIn("Finalized knowledge-unit answers", reference)
        for hidden_field in (
            "source_answer",
            "curation_notes",
            "rewrite_notes",
            "measure_notes",
            "contributing_source_answers",
        ):
            self.assertNotIn(hidden_field, reference)

    def test_viewer_sanitizes_legacy_answers_before_candidate_rendering(
        self,
    ) -> None:
        app = (EVALUATION_ROOT / "app.js").read_text(encoding="utf-8")
        sanitizer_start = app.index("function sanitizeDisplayedAnswer")
        sanitizer_end = app.index("function appendBadge", sanitizer_start)
        sanitizer = app[sanitizer_start:sanitizer_end]
        candidate_start = app.index(
            'const candidateColumn = createElement("article"'
        )
        candidate_end = app.index(
            "grid.append(referenceColumn, candidateColumn)",
            candidate_start,
        )
        candidate_rendering = app[candidate_start:candidate_end]

        self.assertLess(sanitizer_start, candidate_start)
        self.assertIn(r"제공된[\s*_~`]*검색", sanitizer)
        self.assertIn("webchunk", sanitizer)
        self.assertIn(
            "sanitizeDisplayedAnswer(generated.answer)",
            candidate_rendering,
        )
        self.assertNotIn("generated.answer ||", candidate_rendering)

    def test_real_artifacts_include_all_647_results_across_five_pieces(
        self,
    ) -> None:
        original = json.loads(REAL_ORIGINAL_RESULTS.read_text(encoding="utf-8"))
        synthesized = json.loads(
            REAL_SYNTHESIZED_RESULTS.read_text(encoding="utf-8")
        )

        comparison = viewer.build_comparison_payload(original, synthesized)
        results = comparison["results"]
        runs = [run for result in results for run in result["inference_runs"]]
        summary = comparison["summary"]

        self.assertEqual(summary["question_group_count"], 79)
        self.assertEqual(summary["scenario_count"], 164)
        self.assertEqual(summary["paired_scenario_count"], 161)
        self.assertEqual(summary["original_only_scenario_count"], 3)
        self.assertEqual(summary["original_case_count"], 164)
        self.assertEqual(summary["synthesized_case_count"], 483)
        self.assertEqual(summary["total_case_count"], 647)
        self.assertEqual(len(results), 164)
        self.assertEqual(len(runs), 647)
        self.assertEqual(
            Counter(map(len, (r["inference_runs"] for r in results))),
            {4: 161, 1: 3},
        )
        self.assertEqual(
            Counter(run["viewer_question_origin"] for run in runs),
            {"original": 164, "synthesized": 483},
        )
        source_original_ids = {
            run["case_id"]
            for question in original["questions"]
            for run in question["inference_runs"]
        }
        source_synthesized_ids = {
            run["case_id"]
            for result in synthesized["results"]
            for run in result["inference_runs"]
        }
        self.assertEqual(
            {
                run["case_id"]
                for run in runs
                if run["viewer_question_origin"] == "original"
            },
            source_original_ids,
        )
        self.assertEqual(
            {
                run["case_id"]
                for run in runs
                if run["viewer_question_origin"] == "synthesized"
            },
            source_synthesized_ids,
        )
        self.assertEqual(len({run["case_id"] for run in runs}), 647)
        expected_pieces = {
            "die-forelle",
            "in-flowery-clouds",
            "la-capinera",
            "nella-fantasia",
            "una-voce-poco-fa",
        }
        self.assertEqual({result["piece_id"] for result in results}, expected_pieces)
        self.assertEqual(set(summary["per_piece"]), expected_pieces)
        self.assertEqual(
            {result["source_evaluation_id"] for result in results},
            {question["source_id"] for question in original["questions"]},
        )
        self.assertEqual(
            len({result["viewer_scenario_id"] for result in results}),
            164,
        )


if __name__ == "__main__":
    unittest.main()
