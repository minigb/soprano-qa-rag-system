"""Focused tests for the five-piece qualitative answer-fidelity judge."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from evaluation import judge_five_piece_qualitative as judge


def completed_source_artifact() -> dict:
    return {
        "artifact_type": judge.SOURCE_ARTIFACT_TYPE,
        "schema_version": "1.0",
        "run": {"status": "completed", "generate": True},
        "questions": [
            {
                "source_id": "kim-die-forelle-01",
                "piece_id": "die-forelle",
                "original_question": "원래 질문은 무엇인가?",
                "paraphrased_question": "이 부분은 어떻게 표현할까?",
                "reference_material": {
                    "source_answer": "가볍고 유연하게 노래해야 한다.",
                    "linked_knowledge_units": [
                        {
                            "knowledge_unit_id": "die-forelle-ku-001",
                            "source_ids": ["kim-die-forelle-01"],
                            "answer": "고음이 과도하게 강조되지 않게 유연하게 노래한다.",
                            "rewrite_status": "ready",
                            "rewrite_notes": "",
                            "measure_status": "specific",
                            "measure_ranges": [[2, 5]],
                            "measure_notes": "",
                        }
                    ],
                },
                "inference_runs": [
                    {
                        "case_id": "kim-die-forelle-01__m2-5",
                        "status": "completed",
                        "inference_input": {
                            "piece_id": "die-forelle",
                            "question": "이 부분은 어떻게 표현할까?",
                            "measure_range": [2, 5],
                        },
                        "expected_retrieval": {
                            "source_id": "kim-die-forelle-01",
                            "linked_knowledge_unit_ids": [
                                "die-forelle-ku-001"
                            ],
                            "range_applicable_knowledge_unit_ids": [
                                "die-forelle-ku-001"
                            ],
                        },
                        "pipeline_result": {
                            "answer": "2~5마디에서는 고음을 과장하지 않고 유연하게 노래한다.",
                            "evidence": [
                                {
                                    "id": "retrieval-secret-id",
                                    "kind": "expert",
                                    "text": "추가 전문가 근거는 프레이즈를 유연하게 잇는다.",
                                    "measure_ranges": [[2, 5]],
                                    "scope_match": "overlaps_query_range",
                                }
                            ],
                        },
                        "retrieval_diagnostics": {
                            "any_expected_knowledge_unit_retrieved": False,
                        },
                    }
                ],
            },
            {
                "source_id": "kim-la-capinera-01",
                "piece_id": "la-capinera",
                "original_question": "두 번째 원래 질문은 무엇인가?",
                "paraphrased_question": "프레이즈를 어떻게 연결할까?",
                "reference_material": {
                    "source_answer": "호흡을 유지해 프레이즈를 연결한다.",
                    "linked_knowledge_units": [
                        {
                            "knowledge_unit_id": "la-capinera-ku-001",
                            "source_ids": ["kim-la-capinera-01"],
                            "answer": "호흡을 유지해 긴 프레이즈를 연결한다.",
                            "rewrite_status": "ready",
                            "rewrite_notes": "",
                            "measure_status": "whole_piece",
                            "measure_ranges": [],
                            "measure_notes": "",
                        }
                    ],
                },
                "inference_runs": [
                    {
                        "case_id": "kim-la-capinera-01__no-range",
                        "status": "completed",
                        "inference_input": {
                            "piece_id": "la-capinera",
                            "question": "프레이즈를 어떻게 연결할까?",
                            "measure_range": None,
                        },
                        "expected_retrieval": {
                            "source_id": "kim-la-capinera-01",
                            "linked_knowledge_unit_ids": [
                                "la-capinera-ku-001"
                            ],
                            "range_applicable_knowledge_unit_ids": [
                                "la-capinera-ku-001"
                            ],
                        },
                        "pipeline_result": {
                            "answer": "호흡을 유지하면서 긴 프레이즈를 연결한다.",
                            "evidence": [],
                        },
                        "retrieval_diagnostics": {
                            "source_retrieved": False,
                        },
                    }
                ],
            },
        ],
    }


def new_snapshot(source: dict | None = None) -> dict:
    source = source or completed_source_artifact()
    model = {
        "path": "/models/Qwen3-4B",
        "backend": "transformers",
        "kind": "directory",
        "checkpoint_exists": True,
        "sha256": "judge-model",
    }
    return judge.new_snapshot(
        source=source,
        source_path=Path("/evaluation/qualitative.json"),
        source_sha256="source-artifact",
        judge_model=model,
        judge_max_tokens=512,
        input_fingerprint="fingerprint",
        implementation_sha256="implementation",
    )


class FivePieceQualitativeJudgeTests(unittest.TestCase):
    def test_prompt_uses_expert_references_but_not_retrieval_exactness(self):
        snapshot = new_snapshot()
        case = snapshot["cases"][0]

        serialized_case = json.dumps(case, ensure_ascii=False)
        messages = judge.build_judge_messages(case)
        serialized_messages = json.dumps(messages, ensure_ascii=False)

        self.assertIn("고음이 과도하게 강조되지 않게", serialized_messages)
        self.assertIn("die-forelle-ku-001", serialized_messages)
        self.assertIn("2~5마디에서는", serialized_messages)
        self.assertEqual(
            json.loads(messages[1]["content"])["selected_measure_range"],
            [2, 5],
        )
        self.assertIsNone(
            json.loads(messages[1]["content"])["exact_annotator_answer"]
        )
        no_range_messages = judge.build_judge_messages(snapshot["cases"][1])
        self.assertIn(
            "호흡을 유지해 프레이즈를 연결한다",
            no_range_messages[1]["content"],
        )
        self.assertNotIn("retrieval-secret-id", serialized_case)
        self.assertNotIn("retrieval-secret-id", serialized_messages)
        self.assertNotIn("retrieval_diagnostics", serialized_messages)
        self.assertIn("추가 전문가 근거", serialized_messages)
        self.assertIn("검색 성공 여부", messages[0]["content"])

    def test_response_validation_accepts_three_verdicts_and_is_strict(self):
        for verdict in judge.VERDICTS:
            with self.subTest(verdict=verdict):
                validated = judge.validate_judge_response(
                    json.dumps(
                        {
                            "verdict": verdict,
                            "reason": " 핵심 내용이 전문가 답변과 일치한다. ",
                        },
                        ensure_ascii=False,
                    )
                )
                self.assertEqual(validated["verdict"], verdict)
                self.assertEqual(
                    validated["reason"],
                    "핵심 내용이 전문가 답변과 일치한다.",
                )

        invalid_responses = (
            "```json\n{\"verdict\":\"pass\",\"reason\":\"좋다\"}\n```",
            '{"verdict":"automatic_pass","reason":"좋다"}',
            '{"verdict":"pass","reason":"좋다","score":1}',
            "not json",
        )
        for raw in invalid_responses:
            with self.subTest(raw=raw), self.assertRaises(
                judge.QualitativeJudgeResponseError
            ):
                judge.validate_judge_response(raw)

    def test_selected_range_separates_applicable_and_other_linked_units(self):
        snapshot = new_snapshot()
        case = snapshot["cases"][0]
        other = deepcopy(case["reference"]["linked_knowledge_units"][0])
        other["knowledge_unit_id"] = "die-forelle-ku-other"
        other["answer"] = "다른 구간에서만 적용되는 조언이다."
        other["measure_ranges"] = [[40, 42]]
        case["reference"]["linked_knowledge_units"].append(other)

        payload = json.loads(judge.build_judge_messages(case)[1]["content"])

        self.assertEqual(
            [
                unit["knowledge_unit_id"]
                for unit in payload[
                    "range_applicable_linked_knowledge_units"
                ]
            ],
            ["die-forelle-ku-001"],
        )
        self.assertEqual(payload["other_range_linked_knowledge_units"], [])
        self.assertEqual(payload["withheld_other_range_reference_count"], 1)

    def test_runner_retries_checkpoints_and_resumes_without_rejudging(self):
        snapshot = new_snapshot()
        checkpoints = []
        raw_responses = iter(
            (
                "malformed",
                '{"verdict":"pass","reason":"핵심 조언이 충실하다."}',
            )
        )

        def checkpoint() -> None:
            judge.refresh_summary(snapshot)
            checkpoints.append(deepcopy(snapshot["summary"]))

        selected = judge.run_judgments(
            snapshot,
            judge_fn=lambda _messages: next(raw_responses),
            checkpoint=checkpoint,
            max_attempts=3,
            limit=1,
        )

        self.assertEqual(selected, 1)
        first = snapshot["cases"][0]["judgment"]
        second = snapshot["cases"][1]["judgment"]
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["verdict"], "pass")
        self.assertEqual(first["attempt_count"], 2)
        self.assertFalse(first["attempts"][0]["valid"])
        self.assertTrue(first["attempts"][1]["valid"])
        self.assertEqual(second["status"], "pending")
        self.assertGreaterEqual(len(checkpoints), 4)
        self.assertIsNone(snapshot["summary"]["final_case_pass_rate"])

        calls = []

        def second_judge(messages):
            calls.append(messages)
            return (
                '{"verdict":"review","reason":'
                '"세부 범위는 전문가 확인이 더 필요하다."}'
            )

        selected = judge.run_judgments(
            snapshot,
            judge_fn=second_judge,
            checkpoint=checkpoint,
            max_attempts=3,
        )
        judge.refresh_summary(snapshot)

        self.assertEqual(selected, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["attempt_count"], 2)
        self.assertEqual(snapshot["run"]["status"], "completed")
        self.assertEqual(
            snapshot["summary"]["verdicts"],
            {"pass": 1, "review": 1},
        )
        self.assertEqual(snapshot["summary"]["final_case_pass_rate"], 0.5)
        self.assertEqual(
            snapshot["summary"]["by_piece"]["die-forelle"][
                "final_case_pass_rate"
            ],
            1.0,
        )
        self.assertEqual(
            snapshot["summary"]["by_piece"]["la-capinera"][
                "final_case_pass_rate"
            ],
            0.0,
        )

    def test_minimum_rate_gate_runs_only_after_every_case_is_complete(self):
        snapshot = new_snapshot()
        first = snapshot["cases"][0]["judgment"]
        first.update(
            {
                "status": "completed",
                "verdict": "fail",
                "reason": "핵심 내용이 반대다.",
                "completed_at": judge.utc_now(),
            }
        )
        judge.refresh_summary(snapshot)

        self.assertEqual(snapshot["run"]["status"], "in_progress")
        self.assertEqual(
            judge.quality_gate_exit_code(
                snapshot,
                minimum_pass_rate=0.90,
            ),
            0,
        )

        second = snapshot["cases"][1]["judgment"]
        second.update(
            {
                "status": "completed",
                "verdict": "pass",
                "reason": "핵심 내용이 일치한다.",
                "completed_at": judge.utc_now(),
            }
        )
        judge.refresh_summary(snapshot)

        self.assertEqual(snapshot["run"]["status"], "completed")
        self.assertEqual(snapshot["summary"]["final_case_pass_rate"], 0.5)
        self.assertEqual(
            judge.quality_gate_exit_code(
                snapshot,
                minimum_pass_rate=0.90,
            ),
            2,
        )
        self.assertEqual(
            judge.quality_gate_exit_code(
                snapshot,
                minimum_pass_rate=0.50,
            ),
            0,
        )

    def test_unfinished_inference_is_unavailable_and_never_enters_gate(self):
        source = completed_source_artifact()
        source["run"]["status"] = "in_progress"
        unavailable = source["questions"][1]["inference_runs"][0]
        unavailable["status"] = "pending"
        unavailable["pipeline_result"] = None
        snapshot = new_snapshot(source)
        calls = []

        judge.run_judgments(
            snapshot,
            judge_fn=lambda messages: calls.append(messages)
            or '{"verdict":"pass","reason":"핵심 내용이 일치한다."}',
            checkpoint=lambda: judge.refresh_summary(snapshot),
            max_attempts=1,
        )
        judge.refresh_summary(snapshot)

        self.assertEqual(len(calls), 1)
        self.assertEqual(snapshot["run"]["status"], "input_incomplete")
        self.assertEqual(snapshot["summary"]["unavailable_case_count"], 1)
        self.assertIsNone(snapshot["summary"]["final_case_pass_rate"])
        self.assertFalse(
            snapshot["summary"]["minimum_pass_rate_gate_eligible"]
        )
        self.assertEqual(
            judge.quality_gate_exit_code(
                snapshot,
                minimum_pass_rate=0.90,
            ),
            0,
        )

    def test_resume_rejects_changed_inputs_or_model_configuration(self):
        snapshot = new_snapshot()

        judge.validate_resume_snapshot(
            snapshot,
            input_fingerprint="fingerprint",
        )
        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "choose a new --output path",
        ):
            judge.validate_resume_snapshot(
                snapshot,
                input_fingerprint="changed",
            )

    def test_source_hash_ignores_run_timestamps_but_detects_answer_changes(self):
        original = completed_source_artifact()
        timestamp_only = deepcopy(original)
        timestamp_only["run"]["updated_at"] = "later"
        timestamp_only["run"]["finished_at"] = "later"

        self.assertEqual(
            judge.source_judgment_input_sha256(original),
            judge.source_judgment_input_sha256(timestamp_only),
        )

        changed_answer = deepcopy(original)
        changed_answer["questions"][0]["inference_runs"][0][
            "pipeline_result"
        ]["answer"] = "서로 다른 답변"
        self.assertNotEqual(
            judge.source_judgment_input_sha256(original),
            judge.source_judgment_input_sha256(changed_answer),
        )

    def test_legacy_resume_migrates_only_identical_judge_packets(self):
        source = completed_source_artifact()
        snapshot = new_snapshot(source)
        model = snapshot["run"]["judge_model"]

        self.assertTrue(
            judge.can_migrate_semantically_identical_resume(
                snapshot,
                source=source,
                judge_model=model,
                judge_max_tokens=512,
            )
        )

        changed_source = deepcopy(source)
        changed_source["questions"][0]["inference_runs"][0][
            "pipeline_result"
        ]["answer"] = "서로 다른 답변"
        self.assertFalse(
            judge.can_migrate_semantically_identical_resume(
                snapshot,
                source=changed_source,
                judge_model=model,
                judge_max_tokens=512,
            )
        )

    def test_default_minimum_pass_rate_is_ninety_percent(self):
        arguments = judge.parse_arguments([])
        self.assertEqual(arguments.minimum_pass_rate, 0.90)


if __name__ == "__main__":
    unittest.main()
