"""Focused tests for the lightweight qualitative runner."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import run_qualitative as qualitative


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def checkpoint_record(path: Path, *, backend: str) -> dict:
    return {
        "path": str(path.resolve()),
        "checkpoint_exists": True,
        "kind": "file",
        "backend": backend,
        "sha256": "a" * 64,
    }


def clean_generated_response(
    *,
    generation_path: Path,
    embedding_path: Path,
    measure_range: list[int] | None,
    query_route: str = "hybrid",
) -> dict:
    lexical_route = query_route == "lexical"
    return {
        "piece_id": "die-forelle",
        "scope": "range" if measure_range is not None else "whole_piece",
        "measure_range": measure_range,
        "answer": "검색된 전문가 근거를 바탕으로 생성한 답변입니다.",
        "generation_mode": "llm",
        "answer_basis": "retrieved_evidence",
        "unavailable_reason": None,
        "has_primary_grounding": True,
        "has_selected_range_grounding": (
            True if measure_range is not None else None
        ),
        "has_confirmed_local_examples": measure_range is None,
        "pipeline": "soprano_qa",
        "model": {
            "path": str(generation_path.resolve()),
            "checkpoint_exists": True,
            "backend": "llama-cpp-python",
        },
        "retrieval": {
            "configured_mode": "hybrid",
            "active_mode": "hybrid",
            "dense_available": True,
            "embedding_model": str(embedding_path.resolve()),
            "embedding_dimension": 1024,
            "last_dense_error": None,
            "last_search_mode": (
                "lexical_route" if lexical_route else "hybrid"
            ),
            "query_route": query_route,
            "route_reason": (
                "short_exact_lexical_query"
                if lexical_route
                else "semantic_query"
            ),
            "ambiguous_candidates": [],
            "dense_attempted": not lexical_route,
            "dense_contributed": not lexical_route,
        },
        "evidence": [
            {
                "id": "die-forelle-ku-001",
                "kind": "expert",
                "source_ids": ["kim-die-forelle-01"],
            }
        ],
    }


def production_contract_questions(
    case_counts: dict[str, int] | None = None,
) -> list[dict]:
    counts = (
        qualitative.EXPECTED_PRODUCTION_CASE_COUNTS
        if case_counts is None
        else case_counts
    )
    return [
        {
            "source_id": f"{piece_id}-production-fixture",
            "piece_id": piece_id,
            "inference_runs": [
                {
                    "case_id": f"{piece_id}-case-{index:03d}",
                    "inference_input": {"piece_id": piece_id},
                }
                for index in range(case_count)
            ],
        }
        for piece_id, case_count in counts.items()
    ]


class QualitativeRunnerTests(unittest.TestCase):
    def test_production_case_contract_accepts_canonical_five_piece_counts(
        self,
    ) -> None:
        questions = production_contract_questions()

        qualitative.validate_production_case_contract(questions)

        self.assertEqual(
            sum(
                len(question["inference_runs"])
                for question in questions
            ),
            qualitative.EXPECTED_PRODUCTION_CASE_COUNT,
        )

    def test_production_case_contract_rejects_total_and_piece_drift(
        self,
    ) -> None:
        missing_case_counts = dict(
            qualitative.EXPECTED_PRODUCTION_CASE_COUNTS
        )
        missing_case_counts["die-forelle"] -= 1
        shifted_case_counts = dict(
            qualitative.EXPECTED_PRODUCTION_CASE_COUNTS
        )
        shifted_case_counts["die-forelle"] -= 1
        shifted_case_counts["in-flowery-clouds"] += 1

        for counts, expected_error in (
            (missing_case_counts, "Expected 105 production cases"),
            (shifted_case_counts, "case counts by piece drifted"),
        ):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(
                    qualitative.QualitativeEvaluationInputError,
                    expected_error,
                ):
                    qualitative.validate_production_case_contract(
                        production_contract_questions(counts)
                    )

    def build_dataset(self, root: Path) -> None:
        piece_id = "die-forelle"
        sources = [
            {
                "source_id": "kim-die-forelle-01",
                "annotator": "kim",
                "source_text": "질문 1\n답변 1",
                "question": "원래 질문 1?",
                "answer": "범위가 확인된 전문가 답변이다.",
                "legacy_measure_ranges": [],
                "curation_status": "included",
                "curation_notes": "",
            },
            {
                "source_id": "kim-die-forelle-02",
                "annotator": "kim",
                "source_text": "질문 2\n답변 2",
                "question": "원래 질문 2?",
                "answer": "원문에는 검토 필요함 표시와 충돌이 남아 있다.",
                "legacy_measure_ranges": [],
                "curation_status": "included",
                "curation_notes": "",
            },
            {
                "source_id": "kim-die-forelle-03",
                "annotator": "kim",
                "source_text": "질문 3",
                "question": "원래 질문 3?",
                "answer": "",
                "legacy_measure_ranges": [],
                "curation_status": "excluded_unanswerable",
                "curation_notes": "전문가 답변이 없음",
            },
        ]
        units = [
            {
                "knowledge_unit_id": "die-forelle-ku-001",
                "source_ids": ["kim-die-forelle-01"],
                "answer": "범위가 확인된 전문가 답변이다.",
                "rewrite_status": "ready",
                "rewrite_notes": "",
                "measure_range_hints": [],
                "measure_status": "specific",
                "measure_ranges": [[2, 5], [10, 12]],
                "measure_notes": "",
            },
            {
                "knowledge_unit_id": "die-forelle-ku-002",
                "source_ids": ["kim-die-forelle-02"],
                "answer": "최종 정리된 지식 단위 답변이다.",
                "rewrite_status": "ready",
                "rewrite_notes": "obsolete 검토 필요함 메모",
                "measure_range_hints": [],
                "measure_status": "whole_piece",
                "measure_ranges": [],
                "measure_notes": "",
            },
        ]
        review = {
            "schema_version": "1.0",
            "piece_id": piece_id,
            "source_files": [],
            "source_annotations": sources,
            "knowledge_units": units,
        }
        questions = [
            {
                "source_id": "kim-die-forelle-01",
                "annotator": "kim",
                "original_question": "원래 질문 1?",
                "paraphrased_question": "첫 번째 질문은 어떻게 처리할까?",
                "knowledge_unit_ids": ["die-forelle-ku-001"],
                "retrieval_eligible_knowledge_unit_ids": [
                    "die-forelle-ku-001"
                ],
                "measure_range_hints": [],
                "inference_measure_ranges": [[2, 5], [10, 12]],
                "inference_scope": "measure_range",
                "review_status": "retrievable",
            },
            {
                "source_id": "kim-die-forelle-02",
                "annotator": "kim",
                "original_question": "원래 질문 2?",
                "paraphrased_question": "두 번째 질문은 어떻게 처리할까?",
                "knowledge_unit_ids": ["die-forelle-ku-002"],
                "retrieval_eligible_knowledge_unit_ids": [
                    "die-forelle-ku-002"
                ],
                "measure_range_hints": [],
                "inference_measure_ranges": [],
                "inference_scope": "no_range",
                "review_status": "retrievable",
            },
            {
                "source_id": "kim-die-forelle-03",
                "annotator": "kim",
                "original_question": "원래 질문 3?",
                "paraphrased_question": "세 번째 질문은 무엇일까?",
                "knowledge_unit_ids": [],
                "retrieval_eligible_knowledge_unit_ids": [],
                "measure_range_hints": [],
                "inference_measure_ranges": [],
                "inference_scope": "no_range",
                "review_status": "excluded_unanswerable",
            },
        ]
        inventory = {
            "schema_version": "1.1",
            "piece_id": piece_id,
            "questions": questions,
        }
        write_json(
            root / "expert_curation" / "review" / f"{piece_id}.json",
            review,
        )
        write_json(
            root
            / "expert_curation"
            / "evaluation_questions"
            / f"{piece_id}.json",
            inventory,
        )

    def test_loads_one_representative_confirmed_range_per_question(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_dataset(root)
            questions, paths = qualitative.load_qualitative_questions(
                root,
                piece_ids=("die-forelle",),
                enforce_no_range_semantic_policy=False,
            )

        self.assertEqual(len(paths), 2)
        self.assertEqual(
            [question["source_id"] for question in questions],
            ["kim-die-forelle-01", "kim-die-forelle-02"],
        )
        self.assertEqual(
            [
                case["case_id"]
                for question in questions
                for case in question["inference_runs"]
            ],
            [
                "kim-die-forelle-01__m10-12",
                "kim-die-forelle-02__no-range",
            ],
        )
        self.assertEqual(
            questions[0]["inference_measure_ranges"],
            [[2, 5], [10, 12]],
        )
        self.assertEqual(
            questions[0]["representative_inference_measure_range"],
            [10, 12],
        )
        ranged_input = questions[0]["inference_runs"][0]["inference_input"]
        self.assertEqual(
            ranged_input["measure_range_selection_policy"],
            qualitative.REPRESENTATIVE_RANGE_POLICY,
        )
        self.assertEqual(
            ranged_input["measure_range_provenance"],
            qualitative.REPRESENTATIVE_RANGE_PROVENANCE,
        )
        first_authority = questions[0]["inference_runs"][0][
            "inference_input"
        ]["case_reference_authority"]
        second_authority = questions[1]["inference_runs"][0][
            "inference_input"
        ]["case_reference_authority"]
        self.assertEqual(
            first_authority["source"],
            "applicable_finalized_knowledge_units",
        )
        self.assertEqual(
            first_authority["items"][0]["text"],
            "범위가 확인된 전문가 답변이다.",
        )
        self.assertEqual(
            {
                "reference_role": first_authority["items"][0][
                    "reference_role"
                ],
                "measure_status": first_authority["items"][0][
                    "measure_status"
                ],
                "measure_ranges": first_authority["items"][0][
                    "measure_ranges"
                ],
            },
            {
                "reference_role": "required_any",
                "measure_status": "specific",
                "measure_ranges": [[2, 5], [10, 12]],
            },
        )
        self.assertEqual(
            second_authority["items"][0]["text"],
            "최종 정리된 지식 단위 답변이다.",
        )
        serialized_authority = json.dumps(
            second_authority,
            ensure_ascii=False,
        )
        self.assertNotIn("원문에는 검토 필요함", serialized_authority)
        self.assertNotIn("obsolete 검토 필요함 메모", serialized_authority)
        self.assertNotIn("rewrite_notes", serialized_authority)
        self.assertNotIn("measure_notes", serialized_authority)

    def test_rejects_nonfinal_or_empty_case_authority(self):
        for defect in ("nonfinal", "empty"):
            with (
                self.subTest(defect=defect),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                self.build_dataset(root)
                review_path = (
                    root
                    / "expert_curation"
                    / "review"
                    / "die-forelle.json"
                )
                review = qualitative.load_json(review_path)
                unit = review["knowledge_units"][1]
                if defect == "nonfinal":
                    unit["rewrite_status"] = "needs_review"
                    inventory_path = (
                        root
                        / "expert_curation"
                        / "evaluation_questions"
                        / "die-forelle.json"
                    )
                    inventory = qualitative.load_json(inventory_path)
                    inventory["questions"][1][
                        "retrieval_eligible_knowledge_unit_ids"
                    ] = []
                    write_json(inventory_path, inventory)
                    expected = "is not a finalized knowledge unit"
                else:
                    unit["answer"] = " "
                    expected = "has no answer"
                write_json(review_path, review)

                with self.assertRaisesRegex(
                    qualitative.QualitativeEvaluationInputError,
                    expected,
                ):
                    qualitative.load_qualitative_questions(
                        root,
                        piece_ids=("die-forelle",),
                        enforce_no_range_semantic_policy=False,
                    )

    def test_active_dataset_has_all_no_range_semantic_cases(self):
        questions, _ = qualitative.load_qualitative_questions(
            Path("/home/minhee/soprano-qa-dataset")
        )
        cases = [
            case
            for question in questions
            for case in question["inference_runs"]
        ]
        excluded_question = next(
            question
            for question in questions
            if question["source_id"] == "kim-la-capinera-01"
        )

        self.assertEqual(len(questions), 79)
        qualitative.validate_production_case_contract(questions)
        self.assertEqual(
            len(cases),
            qualitative.EXPECTED_PRODUCTION_CASE_COUNT,
        )
        self.assertEqual(
            Counter(
                case["inference_input"]["piece_id"] for case in cases
            ),
            qualitative.EXPECTED_PRODUCTION_CASE_COUNTS,
        )
        cohort_cases = [
            case
            for case in cases
            if case["inference_input"].get("evaluation_cohort")
            == qualitative.NO_RANGE_COHORT_NAME
        ]

        expected_piece_counts = Counter(
            question["piece_id"] for question in questions
        )
        expected_piece_counts.update(
            policy["piece_id"]
            for policy in qualitative.NO_RANGE_SEMANTIC_CASES.values()
        )
        for source_id, formulations in (
            qualitative.REPORTED_REGRESSION_FORMULATIONS.items()
        ):
            expected_piece_counts.update(
                [
                    qualitative.NO_RANGE_SEMANTIC_CASES[source_id][
                        "piece_id"
                    ]
                ]
                * len(formulations)
            )
        self.assertEqual(len(cases), sum(expected_piece_counts.values()))
        self.assertEqual(
            Counter(
                case["inference_input"]["piece_id"] for case in cases
            ),
            expected_piece_counts,
        )
        self.assertEqual(len(cohort_cases), 26)
        self.assertEqual(
            Counter(
                case["inference_input"]["piece_id"]
                for case in cohort_cases
            ),
            {
                "die-forelle": 3,
                "in-flowery-clouds": 6,
                "la-capinera": 6,
                "nella-fantasia": 4,
                "una-voce-poco-fa": 7,
            },
        )
        by_source = {question["source_id"]: question for question in questions}
        for source_id in qualitative.INTRINSIC_RANGE_ONLY_SOURCES:
            with self.subTest(source_id=source_id):
                runs = by_source[source_id]["inference_runs"]
                self.assertEqual(len(runs), 1)
                self.assertIsNotNone(
                    runs[0]["inference_input"]["measure_range"]
                )
                self.assertEqual(
                    runs[0]["inference_input"]["case_kind"],
                    (
                        qualitative.REVIEWED_HINT_CONTEXT_CASE_KIND
                        if source_id
                        in qualitative.REVIEWED_HINT_CONTEXT_RANGE_SOURCES
                        else "confirmed_measure_range"
                    ),
                )
        lyric_run = by_source["kim-la-capinera-13"]["inference_runs"][0]
        self.assertEqual(lyric_run["inference_input"]["measure_range"], [80, 80])
        hint_context = by_source["kim-una-voce-poco-fa-08"]
        self.assertEqual(hint_context["inference_measure_ranges"], [])
        self.assertEqual(
            hint_context["evaluation_context_measure_ranges"],
            [[32, 32]],
        )
        self.assertEqual(hint_context["inference_scope"], "no_range")
        hint_run = hint_context["inference_runs"][0]
        self.assertEqual(hint_run["inference_input"]["measure_range"], [32, 32])
        self.assertEqual(
            hint_run["inference_input"]["measure_range_provenance"],
            qualitative.REVIEWED_HINT_CONTEXT_RANGE_PROVENANCE,
        )
        self.assertEqual(
            hint_run["inference_input"]["case_reference_authority"]["items"][0][
                "measure_status"
            ],
            "whole_piece",
        )
        self.assertEqual(
            sum(
                case["inference_input"]["case_kind"]
                == qualitative.REPORTED_CASE_KIND
                for case in cohort_cases
            ),
            3,
        )
        reported = {
            case["case_id"]: case
            for case in cohort_cases
            if case["inference_input"]["case_kind"]
            == qualitative.REPORTED_CASE_KIND
        }
        self.assertEqual(
            reported[
                "kim-in-flowery-clouds-01__reported-no-range-01"
            ]["inference_input"]["question"],
            "악보의 페르마타 표시가 음원에서 들리는 음 길이와 "
            "일치하지 않을 때는 어떻게 해야 할까?",
        )
        meter_case = reported[
            "kim-in-flowery-clouds-07__reported-no-range-01"
        ]
        self.assertEqual(
            meter_case["inference_input"]["question"],
            "이 곡에서 박자가 여러번 바뀌는 이유는 무엇인가?",
        )
        self.assertEqual(
            meter_case["inference_input"][
                "semantic_required_any_knowledge_unit_ids"
            ],
            ["in-flowery-clouds-ku-008"],
        )
        self.assertEqual(
            reported[
                "kim-una-voce-poco-fa-11__reported-no-range-01"
            ]["inference_input"]["semantic_required_any_knowledge_unit_ids"],
            ["una-voce-poco-fa-ku-016"],
        )
        required_and_supporting = next(
            case
            for case in cohort_cases
            if case["case_id"] == "kim-die-forelle-02__no-range"
        )
        authority_items = required_and_supporting["inference_input"][
            "case_reference_authority"
        ]["items"]
        self.assertEqual(
            [
                (
                    item["knowledge_unit_id"],
                    item["reference_role"],
                    item["measure_status"],
                    item["measure_ranges"],
                )
                for item in authority_items
            ],
            [
                (
                    "die-forelle-ku-002",
                    "required_any",
                    "specific",
                    [[2, 27], [41, 54]],
                ),
                (
                    "die-forelle-ku-003",
                    "supporting",
                    "whole_piece",
                    [],
                ),
            ],
        )
        self.assertEqual(
            len({case["case_id"] for case in cases}),
            len(cases),
        )
        # The authoritative range metadata remains untouched for auditability.
        self.assertIn(
            [78, 81], excluded_question["inference_measure_ranges"]
        )
        self.assertEqual(
            excluded_question["excluded_inference_measure_ranges"][0][
                "measure_range"
            ],
            [78, 81],
        )
        self.assertNotIn(
            "kim-la-capinera-01__m78-81",
            [case["case_id"] for case in cases],
        )

    def test_diagnostics_report_expected_unit_and_source_ranks(self):
        case = {
            "expected_retrieval": {
                "source_id": "kim-die-forelle-01",
                "linked_knowledge_unit_ids": ["die-forelle-ku-001"],
                "retrieval_eligible_knowledge_unit_ids": [
                    "die-forelle-ku-001"
                ],
                "range_applicable_knowledge_unit_ids": [
                    "die-forelle-ku-001"
                ],
                "range_applicable_retrieval_eligible_knowledge_unit_ids": [
                    "die-forelle-ku-001"
                ],
                "semantic_required_any_knowledge_unit_ids": [
                    "die-forelle-ku-001"
                ],
                "semantic_supporting_knowledge_unit_ids": [],
            }
        }
        result = {
            "evidence": [
                {"id": "web-1", "kind": "web", "source_ids": []},
                {
                    "id": "die-forelle-ku-001",
                    "kind": "expert",
                    "source_ids": ["kim-die-forelle-01"],
                },
                {
                    "id": "die-forelle-ku-099",
                    "kind": "expert",
                    "source_ids": ["kim-die-forelle-99"],
                },
            ]
        }

        diagnostics = qualitative.build_retrieval_diagnostics(case, result)

        self.assertTrue(diagnostics["source_retrieved"])
        self.assertEqual(diagnostics["source_first_rank"], 2)
        self.assertEqual(
            diagnostics["expected_knowledge_unit_ranks"],
            {"die-forelle-ku-001": 2},
        )
        self.assertTrue(
            diagnostics["all_expected_knowledge_units_retrieved"]
        )
        self.assertEqual(
            diagnostics["unexpected_expert_knowledge_unit_ids"],
            ["die-forelle-ku-099"],
        )
        self.assertTrue(diagnostics["any_semantic_target_retrieved"])
        self.assertEqual(diagnostics["semantic_target_first_rank"], 2)

    def test_no_range_semantic_rollup_reports_hit_at_3_without_changing_hit_at_k(
        self,
    ):
        cases = []
        for index, first_rank in enumerate((1, 3, 4, None), start=1):
            cases.append(
                {
                    "case_id": f"semantic-case-{index}",
                    "status": "completed",
                    "inference_input": {
                        "piece_id": "die-forelle",
                        "evaluation_cohort": qualitative.NO_RANGE_COHORT_NAME,
                    },
                    "pipeline_result": {
                        "generation_mode": "llm",
                        "answer_basis": "retrieved_evidence",
                    },
                    "retrieval_diagnostics": {
                        "source_retrieved": first_rank is not None,
                        "retrieval_target_available": False,
                        "semantic_target_available": True,
                        "semantic_target_first_rank": first_rank,
                    },
                }
            )
        snapshot = {
            "run": {},
            "questions": [
                {
                    "piece_id": "die-forelle",
                    "inference_runs": cases,
                }
            ],
        }

        qualitative.refresh_summary(snapshot)

        cohort = snapshot["summary"]["evaluation_cohorts"][
            qualitative.NO_RANGE_COHORT_NAME
        ]
        self.assertEqual(cohort["semantic_target_cases_completed"], 4)
        self.assertEqual(cohort["semantic_target_hit_at_1_cases"], 1)
        self.assertEqual(cohort["semantic_target_hit_at_3_cases"], 2)
        self.assertEqual(cohort["semantic_target_hit_at_3_rate"], 0.5)
        # Hit@k remains the existing "retrieved anywhere" metric: rank 4 is
        # still a hit when the configured retrieval depth is larger than 3.
        self.assertEqual(cohort["semantic_target_hit_at_k_cases"], 3)
        self.assertEqual(
            cohort["semantic_target_mean_reciprocal_rank"],
            0.395833,
        )

    def test_limit_is_resumable_and_completed_cases_are_not_repeated(self):
        generation_path = Path("/models/answer.gguf")
        embedding_path = Path("/models/embedding.gguf")
        generation_model = checkpoint_record(
            generation_path,
            backend="llama-cpp-python",
        )
        retrieval_embedding_model = checkpoint_record(
            embedding_path,
            backend="llama-cpp-embedding",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_dataset(root)
            questions, _ = qualitative.load_qualitative_questions(
                root,
                piece_ids=("die-forelle",),
                enforce_no_range_semantic_policy=False,
            )
        snapshot = qualitative.new_snapshot(
            questions=questions,
            dataset_root=Path("/dataset"),
            input_files=[],
            input_fingerprint="fingerprint",
            top_k=6,
            retrieval_embedding_model=retrieval_embedding_model,
            generation_model=generation_model,
        )
        self.assertTrue(snapshot["run"]["generate"])
        calls = []
        checkpoints = []

        def ask_fn(**kwargs):
            calls.append(kwargs)
            return clean_generated_response(
                generation_path=generation_path,
                embedding_path=embedding_path,
                measure_range=(
                    list(kwargs["measure_range"])
                    if kwargs["measure_range"] is not None
                    else None
                ),
            )

        for _ in range(2):
            qualitative.run_selected_cases(
                snapshot,
                ask_fn=ask_fn,
                piece_ids=("die-forelle",),
                limit=1,
                top_k=6,
                retrieval_embedding_model=retrieval_embedding_model,
                generation_model=generation_model,
                checkpoint=lambda: checkpoints.append(True),
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(calls[0]["measure_range"], (10, 12))
        self.assertIsNone(calls[1]["measure_range"])
        self.assertTrue(calls[0]["generate"])
        self.assertNotIn("allow_internal_knowledge", calls[0])
        first_case = questions[0]["inference_runs"][0]
        self.assertEqual(first_case["status"], "completed")
        self.assertEqual(first_case["attempt_count"], 1)
        qualitative.validate_resume_snapshot(
            snapshot,
            input_fingerprint="fingerprint",
            top_k=6,
            retrieval_embedding_model=retrieval_embedding_model,
            generation_model=generation_model,
        )
        corrupted = deepcopy(snapshot)
        corrupted["questions"][0]["inference_runs"][0][
            "pipeline_result"
        ]["model"]["path"] = "/models/wrong.gguf"
        with self.assertRaisesRegex(
            qualitative.QualitativeEvaluationInputError,
            "generation model path",
        ):
            qualitative.validate_resume_snapshot(
                corrupted,
                input_fingerprint="fingerprint",
                top_k=6,
                retrieval_embedding_model=retrieval_embedding_model,
                generation_model=generation_model,
            )

    def test_response_contract_accepts_hybrid_router_lexical_route(self):
        generation_path = Path("/models/answer.gguf")
        embedding_path = Path("/models/embedding.gguf")
        generation_model = checkpoint_record(
            generation_path,
            backend="llama-cpp-python",
        )
        retrieval_embedding_model = checkpoint_record(
            embedding_path,
            backend="llama-cpp-embedding",
        )
        case = {
            "inference_input": {
                "piece_id": "die-forelle",
                "measure_range": None,
            }
        }
        response = clean_generated_response(
            generation_path=generation_path,
            embedding_path=embedding_path,
            measure_range=None,
            query_route="lexical",
        )

        validated = qualitative.validate_case_response(
            case,
            response,
            retrieval_embedding_model=retrieval_embedding_model,
            generation_model=generation_model,
        )

        self.assertEqual(validated["retrieval"]["active_mode"], "hybrid")
        self.assertEqual(validated["retrieval"]["query_route"], "lexical")
        self.assertFalse(validated["retrieval"]["dense_attempted"])

    def test_response_contract_rejects_stale_or_mismatched_metadata(self):
        generation_path = Path("/models/answer.gguf")
        embedding_path = Path("/models/embedding.gguf")
        generation_model = checkpoint_record(
            generation_path,
            backend="llama-cpp-python",
        )
        retrieval_embedding_model = checkpoint_record(
            embedding_path,
            backend="llama-cpp-embedding",
        )
        case = {
            "inference_input": {
                "piece_id": "die-forelle",
                "measure_range": None,
            }
        }
        base = clean_generated_response(
            generation_path=generation_path,
            embedding_path=embedding_path,
            measure_range=None,
        )
        corruptions = {
            "deprecated expert-verbatim path": lambda item: item.update(
                generation_mode="expert_verbatim",
                answer_basis="authoritative_expert",
            ),
            "unavailable path": lambda item: item.update(
                generation_mode="unavailable",
                answer_basis="no_corpus_evidence",
                unavailable_reason="no evidence",
            ),
            "empty generated evidence": lambda item: item.__setitem__(
                "evidence", []
            ),
            "natural-language refusal": lambda item: item.__setitem__(
                "answer",
                "현재 제공된 정보만으로는 정확히 답변하기 어렵습니다.",
            ),
            "removed generation field": lambda item: item.update(
                context_limited=False
            ),
            "removed retrieval field": lambda item: item[
                "retrieval"
            ].update(fallback_used=False),
            "configured lexical mode": lambda item: item[
                "retrieval"
            ].update(configured_mode="lexical"),
            "wrong generation model": lambda item: item["model"].update(
                path="/models/other.gguf"
            ),
            "wrong embedding model": lambda item: item[
                "retrieval"
            ].update(embedding_model="/models/other.gguf"),
        }

        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                response = deepcopy(base)
                corrupt(response)
                with self.assertRaises(
                    qualitative.QualitativeEvaluationInputError
                ):
                    qualitative.validate_case_response(
                        case,
                        response,
                        retrieval_embedding_model=(
                            retrieval_embedding_model
                        ),
                        generation_model=generation_model,
                    )

    def test_case_failure_checkpoints_once_and_aborts(self):
        generation_model = checkpoint_record(
            Path("/models/answer.gguf"),
            backend="llama-cpp-python",
        )
        retrieval_embedding_model = checkpoint_record(
            Path("/models/embedding.gguf"),
            backend="llama-cpp-embedding",
        )
        snapshot = {
            "questions": [
                {
                    "piece_id": "die-forelle",
                    "inference_runs": [
                        {
                            "case_id": "first",
                            "status": "pending",
                            "attempt_count": 0,
                            "last_attempt_at": None,
                            "inference_input": {
                                "piece_id": "die-forelle",
                                "question": "첫 질문",
                                "measure_range": None,
                            },
                            "pipeline_result": None,
                            "retrieval_diagnostics": None,
                            "error": None,
                        },
                        {
                            "case_id": "second",
                            "status": "pending",
                            "attempt_count": 0,
                            "last_attempt_at": None,
                            "inference_input": {
                                "piece_id": "die-forelle",
                                "question": "둘째 질문",
                                "measure_range": None,
                            },
                            "pipeline_result": None,
                            "retrieval_diagnostics": None,
                            "error": None,
                        },
                    ],
                }
            ]
        }
        checkpoints = []

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            qualitative.run_selected_cases(
                snapshot,
                ask_fn=lambda **_: (_ for _ in ()).throw(
                    RuntimeError("generation failed")
                ),
                piece_ids=("die-forelle",),
                limit=None,
                top_k=6,
                retrieval_embedding_model=retrieval_embedding_model,
                generation_model=generation_model,
                checkpoint=lambda: checkpoints.append(True),
            )

        first, second = snapshot["questions"][0]["inference_runs"]
        self.assertEqual(checkpoints, [True])
        self.assertEqual(first["status"], "error")
        self.assertEqual(first["attempt_count"], 1)
        self.assertEqual(first["error"]["type"], "RuntimeError")
        self.assertEqual(second["status"], "pending")
        self.assertEqual(second["attempt_count"], 0)

    def test_resume_rejects_a_different_input_fingerprint(self):
        retrieval_embedding_model = checkpoint_record(
            Path("/models/embedding.gguf"),
            backend="llama-cpp-embedding",
        )
        generation_model = checkpoint_record(
            Path("/models/answer.gguf"),
            backend="llama-cpp-python",
        )
        snapshot = qualitative.new_snapshot(
            questions=[],
            dataset_root=Path("/dataset"),
            input_files=[],
            input_fingerprint="original",
            top_k=6,
            retrieval_embedding_model=retrieval_embedding_model,
            generation_model=generation_model,
        )

        qualitative.validate_resume_snapshot(
            snapshot,
            input_fingerprint="original",
            top_k=6,
            retrieval_embedding_model=retrieval_embedding_model,
            generation_model=generation_model,
        )
        with self.assertRaisesRegex(
            qualitative.QualitativeEvaluationInputError,
            "choose a new --output path",
        ):
            qualitative.validate_resume_snapshot(
                snapshot,
                input_fingerprint="changed",
                top_k=6,
                retrieval_embedding_model=retrieval_embedding_model,
                generation_model=generation_model,
            )

    def test_fingerprint_binds_both_model_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding_path = root / "embedding.gguf"
            generation_path = root / "answer.gguf"
            missing_embedding = qualitative.embedding_model_record(
                embedding_path
            )
            missing_generation = qualitative.generation_model_record(
                generation_path
            )
            missing_fingerprint = qualitative.build_input_fingerprint(
                input_files=[],
                top_k=6,
                retrieval_embedding_model=missing_embedding,
                generation_model=missing_generation,
            )

            embedding_path.write_bytes(b"embedding-checkpoint")
            present_embedding = qualitative.embedding_model_record(
                embedding_path
            )
            embedding_fingerprint = qualitative.build_input_fingerprint(
                input_files=[],
                top_k=6,
                retrieval_embedding_model=present_embedding,
                generation_model=missing_generation,
            )
            generation_path.write_bytes(b"generation-checkpoint")
            present_generation = qualitative.generation_model_record(
                generation_path
            )
            complete_fingerprint = qualitative.build_input_fingerprint(
                input_files=[],
                top_k=6,
                retrieval_embedding_model=present_embedding,
                generation_model=present_generation,
            )

        self.assertFalse(missing_embedding["checkpoint_exists"])
        self.assertFalse(missing_generation["checkpoint_exists"])
        self.assertTrue(present_embedding["checkpoint_exists"])
        self.assertTrue(present_generation["checkpoint_exists"])
        self.assertNotEqual(missing_fingerprint, embedding_fingerprint)
        self.assertNotEqual(embedding_fingerprint, complete_fingerprint)

    def test_missing_retrieval_model_fails_before_corpus_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            settings = {
                "dataset_root": str(root),
                "embedding_model_path": str(root / "missing.gguf"),
                "retrieval": {"mode": "hybrid"},
            }
            arguments = SimpleNamespace(
                dataset_root=root,
                output=output,
                top_k=6,
            )
            with (
                patch.dict(qualitative.os.environ, {}, clear=False),
                patch.object(
                    qualitative,
                    "parse_arguments",
                    return_value=arguments,
                ),
                patch.object(
                    qualitative,
                    "load_settings",
                    return_value=settings,
                ),
                patch.object(
                    qualitative,
                    "validate_retrieval_requirements",
                    side_effect=RuntimeError("embedding checkpoint missing"),
                ) as validate,
                patch("soprano_qa.answer.ensure_corpus") as ensure_corpus,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "embedding checkpoint missing",
                ):
                    qualitative.main([])

            validate.assert_called_once_with(settings)
            ensure_corpus.assert_not_called()
            self.assertFalse(output.exists())

    def test_runner_requires_hybrid_configuration_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            settings = {
                "dataset_root": str(root),
                "retrieval": {"mode": "lexical"},
            }
            arguments = SimpleNamespace(
                dataset_root=root,
                output=output,
                top_k=6,
            )
            with (
                patch.dict(qualitative.os.environ, {}, clear=False),
                patch.object(
                    qualitative,
                    "parse_arguments",
                    return_value=arguments,
                ),
                patch.object(
                    qualitative,
                    "load_settings",
                    return_value=settings,
                ),
                patch.object(
                    qualitative,
                    "validate_retrieval_requirements",
                ) as validate_retrieval,
                patch.object(
                    qualitative,
                    "validate_generation_requirements",
                ) as validate_generation,
                patch("soprano_qa.answer.ensure_corpus") as ensure_corpus,
            ):
                with self.assertRaisesRegex(
                    qualitative.QualitativeEvaluationInputError,
                    "requires retrieval.mode='hybrid'",
                ):
                    qualitative.main([])

            validate_retrieval.assert_not_called()
            validate_generation.assert_not_called()
            ensure_corpus.assert_not_called()
            self.assertFalse(output.exists())

    def test_missing_generation_model_fails_before_corpus_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            settings = {
                "dataset_root": str(root),
                "embedding_model_path": str(root / "embedding.gguf"),
                "model_path": str(root / "missing-answer.gguf"),
                "llm": {},
                "retrieval": {"mode": "hybrid"},
            }
            arguments = SimpleNamespace(
                dataset_root=root,
                output=output,
                top_k=6,
            )
            with (
                patch.dict(qualitative.os.environ, {}, clear=False),
                patch.object(
                    qualitative,
                    "parse_arguments",
                    return_value=arguments,
                ),
                patch.object(
                    qualitative,
                    "load_settings",
                    return_value=settings,
                ),
                patch.object(
                    qualitative,
                    "validate_retrieval_requirements",
                ) as validate_retrieval,
                patch.object(
                    qualitative,
                    "validate_generation_requirements",
                    side_effect=RuntimeError("answer checkpoint missing"),
                ) as validate_generation,
                patch("soprano_qa.answer.ensure_corpus") as ensure_corpus,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "answer checkpoint missing",
                ):
                    qualitative.main([])

            validate_retrieval.assert_called_once_with(settings)
            validate_generation.assert_called_once_with(
                settings["model_path"],
                settings["llm"],
            )
            ensure_corpus.assert_not_called()
            self.assertFalse(output.exists())

    def test_generate_flag_is_removed(self):
        with self.assertRaises(SystemExit):
            qualitative.parse_arguments(["--generate"])

    def test_generation_preflight_precedes_existing_output_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            output.write_text("existing-output", encoding="utf-8")
            settings = {
                "dataset_root": str(root),
                "embedding_model_path": str(root / "embedding.gguf"),
                "model_path": str(root / "missing-answer.gguf"),
                "llm": {},
                "retrieval": {"mode": "hybrid"},
            }
            arguments = SimpleNamespace(
                dataset_root=root,
                output=output,
                top_k=6,
            )
            with (
                patch.dict(qualitative.os.environ, {}, clear=False),
                patch.object(
                    qualitative,
                    "parse_arguments",
                    return_value=arguments,
                ),
                patch.object(
                    qualitative,
                    "load_settings",
                    return_value=settings,
                ),
                patch.object(
                    qualitative,
                    "validate_retrieval_requirements",
                ),
                patch.object(
                    qualitative,
                    "validate_generation_requirements",
                    side_effect=RuntimeError("answer checkpoint missing"),
                ),
                patch.object(qualitative, "load_json") as load_output,
                patch("soprano_qa.answer.ensure_corpus") as ensure_corpus,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "answer checkpoint missing",
                ):
                    qualitative.main([])

            load_output.assert_not_called()
            ensure_corpus.assert_not_called()
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "existing-output",
            )

if __name__ == "__main__":
    unittest.main()
