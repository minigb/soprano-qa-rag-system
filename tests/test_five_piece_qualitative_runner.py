"""Focused tests for the lightweight five-piece qualitative runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import run_five_piece_qualitative as qualitative


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class FivePieceQualitativeRunnerTests(unittest.TestCase):
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
                "answer": "검토 표시가 있는 전문가 답변이다.",
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
                "answer": "검토 표시가 있는 전문가 답변이다.",
                "rewrite_status": "needs_review",
                "rewrite_notes": "세부 표현 확인 필요",
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
                "retrieval_eligible_knowledge_unit_ids": [],
                "measure_range_hints": [],
                "inference_measure_ranges": [],
                "inference_scope": "no_range",
                "review_status": "rewrite_review_pending",
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

    def test_loads_answerable_questions_and_expands_each_confirmed_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_dataset(root)
            questions, paths = qualitative.load_qualitative_questions(
                root,
                piece_ids=("die-forelle",),
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
                "kim-die-forelle-01__m2-5",
                "kim-die-forelle-01__m10-12",
                "kim-die-forelle-02__no-range",
            ],
        )
        self.assertEqual(
            questions[0]["inference_measure_ranges"],
            [[2, 5], [10, 12]],
        )
        self.assertEqual(
            questions[1]["reference_material"]["linked_knowledge_units"][0][
                "rewrite_status"
            ],
            "needs_review",
        )

    def test_active_dataset_has_79_questions_and_124_valid_cases(self):
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
        self.assertEqual(len(cases), 124)
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

    def test_limit_is_resumable_and_completed_cases_are_not_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_dataset(root)
            questions, _ = qualitative.load_qualitative_questions(
                root,
                piece_ids=("die-forelle",),
            )
        snapshot = qualitative.new_snapshot(
            questions=questions,
            dataset_root=Path("/dataset"),
            input_files=[],
            input_fingerprint="fingerprint",
            generate=True,
            top_k=6,
        )
        calls = []
        checkpoints = []

        def ask_fn(**kwargs):
            calls.append(kwargs)
            return {
                "answer": "근거 기반 답변",
                "generation_mode": "llm",
                "answer_basis": "retrieved_evidence",
                "evidence": [
                    {
                        "id": "die-forelle-ku-001",
                        "kind": "expert",
                        "source_ids": ["kim-die-forelle-01"],
                    }
                ],
            }

        for _ in range(2):
            qualitative.run_selected_cases(
                snapshot,
                ask_fn=ask_fn,
                piece_ids=("die-forelle",),
                limit=1,
                generate=True,
                top_k=6,
                checkpoint=lambda: checkpoints.append(True),
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(calls[0]["measure_range"], (2, 5))
        self.assertEqual(calls[1]["measure_range"], (10, 12))
        self.assertTrue(calls[0]["generate"])
        self.assertFalse(calls[0]["allow_internal_knowledge"])
        first_case = questions[0]["inference_runs"][0]
        self.assertEqual(first_case["status"], "completed")
        self.assertEqual(first_case["attempt_count"], 1)

    def test_resume_rejects_a_different_input_fingerprint(self):
        snapshot = {
            "artifact_type": qualitative.ARTIFACT_TYPE,
            "schema_version": qualitative.SCHEMA_VERSION,
            "run": {"input_fingerprint": "original"},
        }

        qualitative.validate_resume_snapshot(
            snapshot,
            input_fingerprint="original",
        )
        with self.assertRaisesRegex(
            qualitative.QualitativeEvaluationInputError,
            "choose a new --output path",
        ):
            qualitative.validate_resume_snapshot(
                snapshot,
                input_fingerprint="changed",
            )

    def test_fingerprint_binds_optional_embedding_checkpoint_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "embedding.gguf"
            missing = qualitative.embedding_model_record(checkpoint)
            missing_fingerprint = qualitative.build_input_fingerprint(
                input_files=[],
                generate=True,
                top_k=6,
                retrieval_embedding_model=missing,
            )

            checkpoint.write_bytes(b"embedding-checkpoint")
            present = qualitative.embedding_model_record(checkpoint)
            present_fingerprint = qualitative.build_input_fingerprint(
                input_files=[],
                generate=True,
                top_k=6,
                retrieval_embedding_model=present,
            )

        self.assertFalse(missing["checkpoint_exists"])
        self.assertIsNone(missing["sha256"])
        self.assertTrue(present["checkpoint_exists"])
        self.assertEqual(present["kind"], "file")
        self.assertNotEqual(missing_fingerprint, present_fingerprint)

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
                generate=False,
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

    def test_explicit_lexical_mode_does_not_require_embedding_model(self):
        qualitative.validate_retrieval_requirements(
            {
                "embedding_model_path": "/definitely/missing.gguf",
                "retrieval": {"mode": "lexical"},
            }
        )


if __name__ == "__main__":
    unittest.main()
