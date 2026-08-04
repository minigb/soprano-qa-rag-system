"""Focused tests for the local evaluation comparison viewer."""

from __future__ import annotations

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
SOURCE_ID = "kim-in-flowery-clouds-07"
PIECE_ID = "in-flowery-clouds"
KNOWLEDGE_UNIT_ID = "in-flowery-clouds-ku-008"
ORIGINAL_CASE_ID = f"{SOURCE_ID}__native-no-range-01"
REFERENCE_TEXT = (
    "작곡가는 박자와 조성을 바꾸어 분위기를 전환합니다."
)
DERIVED_SOURCE_ID = f"{KNOWLEDGE_UNIT_ID}-derived-q01"
DERIVED_QUESTION = "박자와 조성 변화는 분위기에 어떤 영향을 주나요?"


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
        "generation_mode": "llm",
        "answer_basis": "retrieved_evidence",
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
                "question_group_id": SOURCE_ID,
                "question_provenance": viewer.SOURCE_VARIANT_PROVENANCE,
                "synthesized_question_kind": viewer.SOURCE_PARAPHRASE_KIND,
                "source_annotation_ids": [SOURCE_ID],
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


def _derived_result() -> dict[str, object]:
    runs = []
    for measure_range, suffix in ((None, "no-range"), ([47, 48], "m47-48")):
        generated = _pipeline_result(
            f"KU-derived {suffix} 질문에 대한 근거 답변입니다."
        )
        generated["scope"] = (
            "whole_piece" if measure_range is None else "range"
        )
        generated["measure_range"] = deepcopy(measure_range)
        generated["has_selected_range_grounding"] = (
            None if measure_range is None else True
        )
        runs.append(
            {
                "case_id": f"{DERIVED_SOURCE_ID}__{suffix}",
                "status": "completed",
                "inference_input": {
                    "piece_id": PIECE_ID,
                    "source_id": DERIVED_SOURCE_ID,
                    "lineage_source_ids": [SOURCE_ID],
                    "evaluation_question_id": DERIVED_SOURCE_ID,
                    "question": DERIVED_QUESTION,
                    "synthesized_variant_id": None,
                    "variant_index": None,
                    "transformations": [],
                    "synthesized_question_kind": viewer.DERIVED_QUESTION_KIND,
                    "measure_range": deepcopy(measure_range),
                    "measure_range_applied": measure_range is not None,
                    "case_kind": (
                        "knowledge_unit_derived_no_range"
                        if measure_range is None
                        else "knowledge_unit_derived_representative_range"
                    ),
                    "question_provenance": viewer.DERIVED_QUESTION_PROVENANCE,
                    "expected_knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
                    "expected_retrieval_eligible_knowledge_unit_ids": [
                        KNOWLEDGE_UNIT_ID
                    ],
                    "range_applicable_knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
                    "case_reference_authority": _case_reference_authority(),
                },
                "retrieval_probe": {
                    "piece_id": PIECE_ID,
                    "evidence": [
                        {"id": KNOWLEDGE_UNIT_ID, "kind": "expert"}
                    ],
                },
                "generated_answer": generated,
                "error": None,
            }
        )
    return {
        "source_id": DERIVED_SOURCE_ID,
        "question_group_id": DERIVED_SOURCE_ID,
        "question_provenance": viewer.DERIVED_QUESTION_PROVENANCE,
        "synthesized_question_kind": viewer.DERIVED_QUESTION_KIND,
        "source_annotation_ids": [SOURCE_ID],
        "piece_id": PIECE_ID,
        "annotator": None,
        "inventory_schema_version": None,
        "original_question": None,
        "paraphrased_question": DERIVED_QUESTION,
        "review_status": "approved",
        "knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
        "expected_retrieval_eligible_knowledge_unit_ids": [KNOWLEDGE_UNIT_ID],
        "measure_range_hints": [],
        "inference_measure_ranges": [[47, 48]],
        "representative_inference_measure_range": [47, 48],
        "excluded_inference_measure_ranges": [],
        "inference_scope": "no_range_and_representative_range",
        "authoritative_reference": {
            "linked_knowledge_units": [
                {
                    "knowledge_unit_id": KNOWLEDGE_UNIT_ID,
                    "answer": REFERENCE_TEXT,
                    "rewrite_status": "ready",
                    "measure_status": "specific",
                    "measure_ranges": [[47, 48]],
                }
            ]
        },
        "synthesized_question_variants": [],
        "derived_knowledge_unit_question": {
            "knowledge_unit_id": KNOWLEDGE_UNIT_ID,
            "question": DERIVED_QUESTION,
        },
        "inference_runs": runs,
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
                "human_source_question_group_count": 1,
                "paired_human_question_group_count": 1,
                "original_only_question_group_count": 0,
                "knowledge_unit_derived_question_group_count": 0,
                "scenario_count": 1,
                "paired_scenario_count": 1,
                "original_only_scenario_count": 0,
                "synthesized_only_scenario_count": 0,
                "original_case_count": 1,
                "synthesized_paraphrase_case_count": 3,
                "knowledge_unit_derived_case_count": 0,
                "synthesized_case_count": 3,
                "total_case_count": 4,
                "per_piece": {
                    PIECE_ID: {
                        "scenarios": 1,
                        "original": 1,
                        "synthesized": 3,
                        "synthesized_paraphrase": 3,
                        "knowledge_unit_derived": 0,
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
        self.assertEqual(result["viewer_scenario_kind"], "human_source")
        self.assertFalse(result["viewer_synthesized_only"])
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
            [run["viewer_question_kind"] for run in runs],
            [
                "human_original",
                "source_question_paraphrase",
                "source_question_paraphrase",
                "source_question_paraphrase",
            ],
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

    def test_original_only_projection_contains_all_120_cases_with_dynamic_counts(
        self,
    ) -> None:
        original = _original_payload()
        template = original["questions"][0]
        questions = []
        for index in range(120):
            question = deepcopy(template)
            source_id = f"{SOURCE_ID}-preview-{index + 1:03d}"
            case_id = f"{source_id}__native-no-range-01"
            question["source_id"] = source_id
            question["original_question"] = f"원문 질문 {index + 1}"
            question["paraphrased_question"] = f"평가 질문 {index + 1}"
            run = question["inference_runs"][0]
            run["case_id"] = case_id
            run["inference_input"]["question"] = (
                question["paraphrased_question"]
            )
            questions.append(question)
        original["questions"] = questions
        # Source summary values are descriptive only; viewer counts come from
        # the validated runs that it actually projects.
        original["summary"]["case_count"] = 1
        original_before = deepcopy(original)

        comparison = viewer.build_original_only_payload(original)

        self.assertEqual(original, original_before)
        self.assertEqual(len(comparison["results"]), 120)
        self.assertEqual(
            set(comparison["source_artifacts"]),
            {viewer.ORIGINAL_QUESTION_ORIGIN},
        )
        self.assertEqual(
            comparison["source_artifacts"]["original"]["case_count"],
            120,
        )
        self.assertEqual(
            comparison["viewer_projection"]["projection"],
            "all_original_cases_by_scenario",
        )
        self.assertTrue(comparison["viewer_projection"]["original_only"])
        self.assertEqual(
            comparison["summary"],
            {
                "question_group_count": 120,
                "human_source_question_group_count": 120,
                "paired_human_question_group_count": 0,
                "original_only_question_group_count": 120,
                "knowledge_unit_derived_question_group_count": 0,
                "scenario_count": 120,
                "paired_scenario_count": 0,
                "original_only_scenario_count": 120,
                "synthesized_only_scenario_count": 0,
                "original_case_count": 120,
                "synthesized_paraphrase_case_count": 0,
                "knowledge_unit_derived_case_count": 0,
                "synthesized_case_count": 0,
                "total_case_count": 120,
                "per_piece": {
                    PIECE_ID: {
                        "scenarios": 120,
                        "original": 120,
                        "synthesized": 0,
                        "synthesized_paraphrase": 0,
                        "knowledge_unit_derived": 0,
                        "total": 120,
                    }
                },
            },
        )
        runs = [
            result["inference_runs"][0]
            for result in comparison["results"]
        ]
        self.assertEqual(
            {run["viewer_question_origin"] for run in runs},
            {viewer.ORIGINAL_QUESTION_ORIGIN},
        )
        self.assertEqual(
            {run["viewer_question_kind"] for run in runs},
            {viewer.ORIGINAL_QUESTION_KIND},
        )
        self.assertTrue(
            all(
                run["quality_review"]
                == {"review_status": viewer.QUALITY_REVIEW_NOT_AVAILABLE}
                for run in runs
            )
        )

    def test_original_only_server_does_not_open_other_artifacts(self) -> None:
        missing_synthesized = (
            self.original_results_file.parent / "never-created-synthesized.json"
        )
        missing_review = (
            self.original_results_file.parent / "never-created-review.json"
        )
        preview_server = viewer.create_server(
            port=0,
            evaluation_root=EVALUATION_ROOT,
            original_results_file=self.original_results_file,
            synthesized_results_file=missing_synthesized,
            quality_review_file=missing_review,
            original_only=True,
        )
        preview_thread = threading.Thread(
            target=preview_server.serve_forever,
            daemon=True,
        )
        preview_thread.start()
        try:
            port = preview_server.server_address[1]
            connection = HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                connection.request(
                    "GET",
                    "/api/results",
                    headers={"Host": f"127.0.0.1:{port}"},
                )
                response = connection.getresponse()
                body = response.read()
            finally:
                connection.close()
        finally:
            preview_server.shutdown()
            preview_thread.join(timeout=2)
            preview_server.server_close()

        comparison = json.loads(body)
        self.assertEqual(response.status, 200)
        self.assertEqual(comparison["summary"]["original_case_count"], 1)
        self.assertEqual(comparison["summary"]["synthesized_case_count"], 0)
        self.assertFalse(missing_synthesized.exists())
        self.assertFalse(missing_review.exists())

    def test_original_only_cli_validates_only_the_original_artifact(self) -> None:
        missing_synthesized = (
            self.original_results_file.parent / "missing-synthesized.json"
        )
        missing_review = self.original_results_file.parent / "missing-review.json"
        arguments = viewer._parse_arguments(
            [
                "--original-only",
                "--original-results-file",
                str(self.original_results_file),
                "--synthesized-results-file",
                str(missing_synthesized),
                "--quality-review-file",
                str(missing_review),
            ]
        )

        self.assertTrue(arguments.original_only)
        self.assertEqual(arguments.original_results_file, self.original_results_file)
        with self.assertRaises(SystemExit):
            viewer._parse_arguments(
                [
                    "--original-only",
                    "--original-results-file",
                    str(self.original_results_file.parent / "missing-original.json"),
                    "--synthesized-results-file",
                    str(missing_synthesized),
                    "--quality-review-file",
                    str(missing_review),
                ]
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
        self.assertEqual(generated["generation_mode"], "llm")
        self.assertEqual(generated["answer_basis"], "retrieved_evidence")
        for removed_field in (
            "rag_llm_succeeded",
            "authoritative_expert_succeeded",
            "grounded_answer_succeeded",
        ):
            self.assertNotIn(removed_field, generated)
        self.assertEqual(
            run["retrieval_probe"]["evidence"][0]["id"],
            KNOWLEDGE_UNIT_ID,
        )

    def test_keeps_ku_derived_runs_as_distinct_synthesized_only_scenarios(
        self,
    ) -> None:
        self.synthesized_payload["results"].append(_derived_result())
        self._write_fixtures()

        status, _, body = self.request("/api/results")
        comparison = json.loads(body)

        self.assertEqual(status, 200)
        summary = comparison["summary"]
        self.assertEqual(summary["question_group_count"], 2)
        self.assertEqual(
            summary["knowledge_unit_derived_question_group_count"], 1
        )
        self.assertEqual(summary["scenario_count"], 3)
        self.assertEqual(summary["paired_scenario_count"], 1)
        self.assertEqual(summary["synthesized_only_scenario_count"], 2)
        self.assertEqual(summary["original_case_count"], 1)
        self.assertEqual(summary["synthesized_paraphrase_case_count"], 3)
        self.assertEqual(summary["knowledge_unit_derived_case_count"], 2)
        self.assertEqual(summary["synthesized_case_count"], 5)
        self.assertEqual(summary["total_case_count"], 6)

        derived = [
            result
            for result in comparison["results"]
            if result["viewer_scenario_kind"] == "knowledge_unit_derived"
        ]
        self.assertEqual(len(derived), 2)
        self.assertEqual(
            {
                tuple(result["inference_runs"][0]["inference_input"][
                    "measure_range"
                ] or [])
                for result in derived
            },
            {(), (47, 48)},
        )
        for result in derived:
            self.assertTrue(result["viewer_synthesized_only"])
            self.assertEqual(result["source_evaluation_id"], DERIVED_SOURCE_ID)
            self.assertEqual(
                result["viewer_comparison"],
                {"original_case_count": 0, "synthesized_case_count": 1},
            )
            self.assertEqual(len(result["inference_runs"]), 1)
            run = result["inference_runs"][0]
            self.assertEqual(run["viewer_question_origin"], "synthesized")
            self.assertEqual(
                run["viewer_question_kind"], "knowledge_unit_derived"
            )

    def test_human_source_scenarios_require_all_three_exact_paraphrases(
        self,
    ) -> None:
        missing_variant = deepcopy(self.synthesized_payload)
        missing_variant["results"][0]["inference_runs"].pop()
        with self.assertRaisesRegex(
            viewer.EvaluationArtifactError,
            "variants 1, 2, and 3 exactly once",
        ):
            viewer.build_comparison_payload(
                self.original_payload,
                missing_variant,
            )

        repeated_index = deepcopy(self.synthesized_payload)
        repeated_index["results"][0]["inference_runs"][2][
            "inference_input"
        ]["variant_index"] = 2
        with self.assertRaisesRegex(
            viewer.EvaluationArtifactError,
            "variants 1, 2, and 3 exactly once",
        ):
            viewer.build_comparison_payload(
                self.original_payload,
                repeated_index,
            )

    def test_keeps_documented_reported_regression_as_original_only(self) -> None:
        original = deepcopy(self.original_payload)
        reported = deepcopy(original["questions"][0]["inference_runs"][0])
        reported["case_id"] = f"{SOURCE_ID}__reported-no-range-01"
        reported["inference_input"]["case_kind"] = "reported_regression"
        reported["inference_input"][
            "question_provenance"
        ] = viewer.REPORTED_REGRESSION_PROVENANCE
        reported["inference_input"]["question"] = (
            "박자가 계속 달라지는 까닭이 궁금합니다."
        )
        original["questions"][0]["inference_runs"].append(reported)

        comparison = viewer.build_comparison_payload(
            original,
            self.synthesized_payload,
        )

        self.assertEqual(comparison["summary"]["scenario_count"], 2)
        self.assertEqual(comparison["summary"]["paired_scenario_count"], 1)
        self.assertEqual(
            comparison["summary"]["original_only_scenario_count"], 1
        )
        self.assertEqual(comparison["summary"]["original_case_count"], 2)
        self.assertEqual(comparison["summary"]["total_case_count"], 5)

    def test_allows_only_explicitly_excluded_human_groups_to_be_original_only(
        self,
    ) -> None:
        synthesized = deepcopy(self.synthesized_payload)
        synthesized["results"] = [_derived_result()]

        with self.assertRaisesRegex(
            viewer.EvaluationArtifactError,
            "must be excluded_unanswerable",
        ):
            viewer.build_comparison_payload(self.original_payload, synthesized)

        original = deepcopy(self.original_payload)
        original["questions"][0]["review_status"] = "excluded_unanswerable"
        comparison = viewer.build_comparison_payload(original, synthesized)

        self.assertEqual(
            comparison["summary"]["original_only_question_group_count"], 1
        )
        self.assertEqual(comparison["summary"]["original_only_scenario_count"], 1)
        self.assertEqual(
            comparison["summary"]["synthesized_only_scenario_count"], 2
        )

    def test_accepts_optional_generation_validation_warning(self) -> None:
        warning = "Generated answer may overgeneralize local evidence"
        self.original_payload["questions"][0]["inference_runs"][0][
            "pipeline_result"
        ]["generation_validation_warning"] = warning
        self.synthesized_payload["results"][0]["inference_runs"][0][
            "generated_answer"
        ]["generation_validation_warning"] = warning
        self._write_fixtures()

        status, _, body = self.request("/api/results")
        comparison = json.loads(body)

        self.assertEqual(status, 200)
        runs = comparison["results"][0]["inference_runs"]
        self.assertEqual(
            runs[0]["generated_answer"]["generation_validation_warning"],
            warning,
        )
        self.assertEqual(
            runs[1]["generated_answer"]["generation_validation_warning"],
            warning,
        )

    def test_complete_quality_review_is_attached_to_every_exact_run(self) -> None:
        self.synthesized_payload["results"].append(_derived_result())
        self._write_fixtures()
        flagged_case_id = f"{DERIVED_SOURCE_ID}__m47-48"
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
                "assessment_count": 6,
                "red_flag_count": 1,
            },
        )
        runs = [
            run
            for result in comparison["results"]
            for run in result["inference_runs"]
        ]
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
            "question kind": lambda result, run: run.__setitem__(
                "viewer_question_kind", "knowledge_unit_derived"
            ),
            "question provenance": lambda result, run: run[
                "inference_input"
            ].__setitem__("question_provenance", "knowledge_unit_derived"),
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
            "answer evidence": lambda result, run: run["generated_answer"][
                "evidence"
            ][0].__setitem__("text", "다른 근거"),
            "generation validation warning": lambda result, run: run[
                "generated_answer"
            ].__setitem__(
                "generation_validation_warning",
                "Generated answer may overgeneralize local evidence",
            ),
            "mode": lambda result, run: run["generated_answer"].__setitem__(
                "generation_mode", "expert_verbatim"
            ),
            "basis": lambda result, run: run["generated_answer"].__setitem__(
                "answer_basis", "authoritative_expert"
            ),
            "scope": lambda result, run: run["generated_answer"].__setitem__(
                "scope", "selected_range"
            ),
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
            'id="run-status-filter"',
            'id="variant-filter"',
            'id="quality-review-filter"',
            'id="quality-review-banner"',
            'id="red-flags-page-link"',
        ):
            self.assertIn(marker, html)
        self.assertIn('data-viewer-mode="comparison"', html)
        self.assertRegex(
            html,
            r'id="quality-review-banner"\s+aria-live="polite"\s+hidden',
        )
        self.assertIn("검토된 지식 단위에서", html)
        self.assertIn('value="source_question_paraphrase"', html)
        self.assertIn('value="knowledge_unit_derived"', html)
        self.assertNotIn("120개", html)
        self.assertNotIn("406개", html)
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
        self.assertIn("viewer_question_kind", app)
        self.assertIn("viewer_synthesized_only", app)
        self.assertIn("runQuestionKind", app)
        self.assertIn("knowledge_unit_derived_case_count", app)
        self.assertIn("synthesized_paraphrase_case_count", app)
        self.assertIn("KU-derived evaluation question", app)
        self.assertIn("questionOriginFilter", app)
        self.assertIn("renderQuestionWordingComparison", app)
        self.assertIn("comparisonCaseGroups", app)
        self.assertIn("case-tab-groups", app)
        self.assertIn('PAGE_MODE === "red-flags"', app)
        self.assertIn('return RED_FLAGS_PAGE ? "flagged" : "all"', app)
        self.assertIn("qualityReviewCategory", app)
        self.assertIn("renderQualityAssessment", app)
        self.assertIn("renderQualityReviewStatus", app)
        self.assertIn("qualityReviewSummary", app)
        self.assertIn("appendScenarioQualityReviewBadge", app)
        self.assertIn("has-quality-red-flag", app)
        self.assertIn("tab-quality-flag", app)
        self.assertIn(
            "No semantic red flags match the current filters.",
            app,
        )
        self.assertIn(
            "elements.qualityReviewBanner.hidden = !RED_FLAGS_PAGE",
            app,
        )
        self.assertIn("case_reference_authority", review_state)
        self.assertNotIn("renderFinding", app)
        self.assertNotIn("renderSummary", app)

    def test_viewer_exposes_only_the_current_rag_llm_answer_path(self) -> None:
        app = (EVALUATION_ROOT / "app.js").read_text(encoding="utf-8")
        html = (EVALUATION_ROOT / "index.html").read_text(encoding="utf-8")

        helper_start = app.index("function isBenchmarkRagLlmAnswer")
        helper_end = app.index("function semanticAggregate", helper_start)
        helpers = app[helper_start:helper_end]
        self.assertIn('generation_mode === "llm"', helpers)
        self.assertIn('answer_basis === "retrieved_evidence"', helpers)
        self.assertNotIn("expert_verbatim", app)
        self.assertNotIn("authoritative_expert", app)
        self.assertNotIn("no_corpus_evidence", app)
        self.assertNotIn('id="basis-filter"', html)
        self.assertNotIn('id="generation-mode-filter"', html)
        self.assertIn("Pipeline answer", app)
        self.assertNotIn("Generated RAG+LLM answer", app)

    def test_server_rejects_deprecated_or_unsafe_answer_artifacts(self) -> None:
        corruptions = {
            "expert verbatim": lambda output: output.update(
                generation_mode="expert_verbatim",
                answer_basis="authoritative_expert",
            ),
            "unavailable": lambda output: output.update(
                generation_mode="unavailable",
                answer_basis="no_corpus_evidence",
                unavailable_reason="no evidence",
            ),
            "extractive": lambda output: output.update(
                generation_mode="extractive",
                answer_basis="retrieval_extractive",
            ),
            "empty evidence": lambda output: output.__setitem__("evidence", []),
            "natural-language insufficiency refusal": lambda output: (
                output.__setitem__(
                    "answer",
                    "현재 제공된 정보만으로는 정확히 답변하기 어렵습니다.",
                )
            ),
            "opaque knowledge-unit id": lambda output: output.__setitem__(
                "answer",
                "박자와 조성을 정확히 표현해야 합니다. "
                "[in-flowery-clouds-ku-008]",
            ),
            "short internal evidence label": lambda output: (
                output.__setitem__(
                    "answer",
                    "박자와 조성을 정확히 표현해야 합니다. [E1]",
                )
            ),
            "evidence footer": lambda output: output.__setitem__(
                "answer",
                "박자와 조성을 정확히 표현해야 합니다.\n"
                "제공된 검색 근거: [E1]",
            ),
            "pipeline label": lambda output: output.__setitem__(
                "answer",
                "범위 안내: 이 내용은 국소 예시입니다.\n"
                "박자와 조성을 정확히 표현해야 합니다.",
            ),
        }

        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                output = _pipeline_result()
                corrupt(output)
                with self.assertRaises(viewer.EvaluationArtifactError):
                    viewer._validate_pipeline_result(
                        output,
                        label="test output",
                        allowed_fields=viewer.PIPELINE_RESULT_FIELDS,
                    )

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

if __name__ == "__main__":
    unittest.main()
