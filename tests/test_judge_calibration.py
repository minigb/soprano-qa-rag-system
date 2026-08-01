"""Focused tests for the schema-8 local judge calibration contract."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evaluation import calibrate_judge as calibration
from evaluation import run_question_evaluation as evaluator
from tests.test_question_evaluation import (
    TEST_RANGE_GUARD,
    assessment_payload,
    atomic_coverage_signals,
    hybrid_range_signals,
)


DATASET_ROOT = Path("/home/minhee/soprano-qa-dataset")
JUDGE_MAX_TOKENS = 1536
EXPECTED_VERDICTS = {
    "good_retrieved_expert_supplement": "pass",
    "supplement_only_cannot_replace_target": "fail",
    "good_second_beat_accent": "pass",
    "minor_omission_lerbe_e_melisma": "human_review",
    "contradictory_fermata_mor": "fail",
    "unsupported_mandatory_training_regimen": "fail",
    "cautious_unresolved_meter_change": "human_review",
    "overconfident_fermata_original": "fail",
    "overconfident_unresolved_meter_simultaneity": "human_review",
    "good_early_echo_contrast": "human_review",
    "good_early_shared_vocabulary_paraphrase": "human_review",
    "good_early_negated_correction": "human_review",
    "wrong_later_dynamics_projected_to_early_echo": "fail",
    "wrong_later_dynamics_paraphrased_at_early_echo": "human_review",
    "good_later_rapid_alternation": "pass",
    "wrong_early_echo_projected_to_later_dynamics": "fail",
    "unrelated_same_composer_background_only": "fail",
    "good_second_beat_semantic_paraphrase": "human_review",
    "wrong_early_unseen_dense_alternation": "human_review",
    "wrong_early_mixed_final_assertion": "fail",
    "wrong_early_double_negation": "fail",
    "good_early_natural_negation": "human_review",
    "wrong_later_unseen_separated_blocks": "human_review",
    "good_later_rapid_with_two_parts_contrast": "pass",
    "uncertain_early_range_projection": "human_review",
}
CONTRADICTION_CONTROLS = {
    "contradictory_fermata_mor",
    "overconfident_fermata_original",
    "overconfident_unresolved_meter_simultaneity",
}
UNSUPPORTED_CONTROLS = {
    "unsupported_mandatory_training_regimen",
}
UNRELATED_CONTROLS = {
    "unrelated_same_composer_background_only",
}
WRONG_RANGE_CONTROLS = {
    "wrong_later_dynamics_projected_to_early_echo",
    "wrong_later_dynamics_paraphrased_at_early_echo",
    "wrong_early_echo_projected_to_later_dynamics",
    "wrong_early_unseen_dense_alternation",
    "wrong_early_mixed_final_assertion",
    "wrong_later_unseen_separated_blocks",
}
NEGATED_RANGE_CONTROLS = {
    "good_early_negated_correction",
    "good_early_natural_negation",
}
UNCERTAIN_RANGE_CONTROLS = {
    "wrong_early_double_negation",
    "uncertain_early_range_projection",
}


def semantic_mode(control_id: str) -> str:
    if control_id in {
        "minor_omission_lerbe_e_melisma",
        "good_second_beat_semantic_paraphrase",
    }:
        return "minor_omission"
    if control_id in CONTRADICTION_CONTROLS:
        return "contradiction"
    if control_id in UNSUPPORTED_CONTROLS:
        return "unsupported"
    if control_id in UNRELATED_CONTROLS:
        return "unrelated"
    return "equivalent"


def semantic_raw(control: dict, frame: str) -> str:
    packet = control["reference_packet"]
    references = evaluator._all_semantic_reference_items(packet)
    payload = assessment_payload(
        frame,
        reference_items=references,
        candidate_items=evaluator._candidate_answer_items(packet),
        mode=semantic_mode(control["control_id"]),
        has_range=packet["selected_measure_range"] is not None,
    )
    references_by_id = {
        item["reference_id"]: item
        for item in references
    }
    for assessment in payload["reference_assessments"]:
        reference = references_by_id[assessment["reference_id"]]
        if (
            assessment["status"] == "not_required"
            and reference.get("scope")
            in {"direct_required", "mixed_or_ambiguous"}
        ):
            assessment["status"] = "missing"
    return json.dumps(payload, ensure_ascii=False)


def validated_semantic_state(control: dict, frame: str) -> dict:
    packet = control["reference_packet"]
    raw = semantic_raw(control, frame)
    coverage_signals = atomic_coverage_signals(
        packet,
        overrides=(
            {
                "R002": {
                    "entailment": 0.001,
                    "neutral": 0.998,
                    "contradiction": 0.001,
                }
            }
            if control["control_id"]
            == "supplement_only_cannot_replace_target"
            else None
        ),
    )
    validated = evaluator.validate_judge_assessment(
        raw,
        expected_frame=frame,
        has_measure_range=packet["selected_measure_range"] is not None,
        candidate_answer=packet["candidate_answer"],
        authoritative_reference_items=(
            evaluator._all_semantic_reference_items(packet)
        ),
        candidate_answer_items=evaluator._candidate_answer_items(packet),
        allowed_context_texts=(
            [packet["question"]]
            + evaluator._authoritative_question_contexts(packet)
            + evaluator._supplemental_question_contexts(packet)
        ),
        atomic_coverage_signals=coverage_signals,
        atomic_coverage_nli_threshold=0.8,
    )
    return {
        "status": "complete",
        "assessment": validated,
        "computed": evaluator.compute_frame_result(validated),
        "attempts": [
            {
                "raw_response": raw,
                "atomic_coverage_signals": coverage_signals,
            }
        ],
    }


def range_relation(control_id: str) -> str:
    if control_id in WRONG_RANGE_CONTROLS:
        return "asserted"
    if control_id in NEGATED_RANGE_CONTROLS:
        return "negated"
    if control_id in UNCERTAIN_RANGE_CONTROLS:
        return "uncertain"
    return "absent"


def range_raw(control: dict, *, relation: str | None = None) -> str:
    packet = control["reference_packet"]
    relation = relation or range_relation(control["control_id"])
    excluded = evaluator._excluded_reference_items(packet)
    candidates = evaluator._candidate_answer_items(packet)
    assessments = []
    for index, item in enumerate(excluded):
        active = index == 0 and relation != "absent"
        assessments.append(
            {
                "excluded_id": item["reference_id"],
                "relation": relation if index == 0 else "absent",
                "candidate_ids": (
                    [candidates[0]["candidate_id"]] if active else []
                ),
            }
        )
    return json.dumps(
        {
            "frame": evaluator.RANGE_FRAME_NAME,
            "excluded_claim_assessments": assessments,
            "confidence": 0.95,
            "rationale": "선택 범위 밖 주장의 적용 여부를 판별한다.",
        },
        ensure_ascii=False,
    )


def attach_range_state(
    control: dict,
    *,
    relation: str | None = None,
) -> None:
    packet = control["reference_packet"]
    requirement = evaluator.range_scope_requirement(packet)
    if not requirement["applicable"]:
        control["range_scope"] = {
            **requirement,
            "attempts": [],
            "assessment": {
                "status": "not_applicable",
                "applicable": False,
                "wrong_measure_application": False,
            },
        }
        return
    raw = range_raw(control, relation=relation)
    signals = hybrid_range_signals(packet)
    assessment = evaluator.validate_range_judge_assessment(
        raw,
        packet=packet,
        hybrid_signals=signals,
        nli_threshold=0.8,
        embedding_delta_threshold=0.05,
    )
    control["range_scope"] = {
        "status": "complete",
        "attempts": [
            {
                "raw_response": raw,
                "hybrid_signals": signals,
            }
        ],
        "requirement": requirement,
        "assessment": assessment,
    }


def complete_control(
    control: dict,
    *,
    forced_range_relation: str | None = None,
) -> dict:
    for frame in evaluator.FRAME_NAMES:
        control["frames"][frame] = validated_semantic_state(control, frame)
    attach_range_state(
        control,
        relation=forced_range_relation,
    )
    control["aggregate"] = evaluator.aggregate_judge_frames(
        control["frames"],
        generated_answer=control["generated_answer"],
        range_scope_evaluation=control["range_scope"]["assessment"],
        reference_review=control["reference_review"],
        supplemental_evidence_validation=control[
            "reference_packet"
        ]["supplemental_evidence_validation"],
        review_disclosure_validation=control[
            "reference_packet"
        ]["review_disclosure_validation"],
    )
    control["expectation_evaluation"] = (
        calibration.evaluate_control_expectation(control)
    )
    return control


def complete_artifact(expected: dict) -> dict:
    controls = [
        complete_control(deepcopy(control))
        for control in expected["controls"]
    ]
    artifact = calibration.new_snapshot(
        controls=controls,
        dataset_root=DATASET_ROOT,
        input_files=deepcopy(expected["input_files"]),
        input_fingerprint=expected["input_fingerprint"],
        judge_model=deepcopy(expected["judge_model"]),
        range_guard=deepcopy(expected["range_guard"]),
        max_tokens=JUDGE_MAX_TOKENS,
    )
    artifact["run"]["runtime_versions"] = deepcopy(
        expected["runtime_versions"]
    )
    calibration.refresh_summary(artifact)
    return artifact


class ControlConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controls, cls.paths = calibration.build_controls(DATASET_ROOT)

    def test_controls_are_the_exact_25_case_contract(self) -> None:
        self.assertEqual(calibration.EXPECTED_CONTROL_COUNT, 25)
        self.assertEqual(
            evaluator.CALIBRATION_EXPECTED_CONTROL_COUNT,
            25,
        )
        self.assertEqual(len(self.controls), 25)
        self.assertEqual(
            [item["control_id"] for item in self.controls],
            list(EXPECTED_VERDICTS),
        )
        self.assertEqual(
            {
                item["control_id"]: item["expected"]["verdict"]
                for item in self.controls
            },
            EXPECTED_VERDICTS,
        )

    def test_controls_preserve_verbatim_dataset_authority(self) -> None:
        review_answers = {}
        for path in self.paths:
            if path.parent.name != "review":
                continue
            for source in evaluator.load_json(path)["source_annotations"]:
                review_answers[source["source_id"]] = source["answer"]

        for control in self.controls:
            with self.subTest(control=control["control_id"]):
                packet = control["reference_packet"]
                if packet[
                    "authoritative_original_expert_answer_withheld_reason"
                ]:
                    self.assertIsNone(
                        packet["authoritative_original_expert_answer"]
                    )
                else:
                    self.assertEqual(
                        packet["authoritative_original_expert_answer"],
                        review_answers[control["source_id"]],
                    )
                self.assertTrue(
                    evaluator._authoritative_reference_items(packet)
                )
                self.assertTrue(evaluator._candidate_answer_items(packet))
                self.assertEqual(
                    packet["candidate_answer"],
                    control["candidate_answer"],
                )

    def test_warned_controls_use_authenticated_production_disclosures(
        self,
    ) -> None:
        warned_ids = {
            "cautious_unresolved_meter_change",
            "overconfident_fermata_original",
            "overconfident_unresolved_meter_simultaneity",
        }
        observed = set()
        for control in self.controls:
            audit = control["reference_packet"][
                "review_disclosure_validation"
            ]
            if not audit["expected_text"]:
                self.assertEqual(audit["status"], "none_required")
                continue
            observed.add(control["control_id"])
            with self.subTest(control=control["control_id"]):
                self.assertEqual(
                    audit["status"],
                    "authenticated_stripped",
                )
                self.assertTrue(audit["auto_pass_eligible"])
                self.assertTrue(
                    control["generated_answer"]["answer"].startswith(
                        audit["expected_text"] + "\n\n"
                    )
                )
                self.assertNotEqual(
                    control["generated_answer"]["answer"],
                    control["candidate_answer"],
                )
                self.assertEqual(
                    control["reference_packet"]["candidate_answer"],
                    control["candidate_answer"],
                )
        self.assertEqual(observed, warned_ids)

    def test_invalid_reprise_controls_are_not_reintroduced(self) -> None:
        self.assertNotIn(
            "wrong_non_applicable_diction_at_reprise",
            EXPECTED_VERDICTS,
        )
        self.assertNotIn(
            "wrong_non_applicable_diction_paraphrased_at_reprise",
            EXPECTED_VERDICTS,
        )
        self.assertFalse(any(
            control["source_id"] == "kim-la-capinera-01"
            and control["selected_measure_range"] == [78, 81]
            for control in self.controls
        ))

    def test_double_negation_is_never_calibrated_as_an_automatic_pass(
        self,
    ) -> None:
        control = next(
            item
            for item in self.controls
            if item["control_id"] == "wrong_early_double_negation"
        )
        self.assertEqual(control["expected"]["verdict"], "fail")
        self.assertEqual(
            control["expected"]["allowed_verdicts"],
            ["fail", "human_review"],
        )
        self.assertEqual(
            control["expected"]["required_signal"],
            "range_uncertainty",
        )
        self.assertNotIn(
            "pass",
            control["expected"]["allowed_verdicts"],
        )

    def test_fermata_contradiction_allows_conservative_failure_label(
        self,
    ) -> None:
        control = next(
            item
            for item in self.controls
            if item["control_id"] == "contradictory_fermata_mor"
        )
        expected = control["expected"]
        self.assertEqual(
            expected.get("allowed_verdicts", [expected["verdict"]]),
            ["fail"],
        )
        self.assertIn(
            "materially_unreliable",
            expected["allowed_relationships"],
        )
        self.assertEqual(expected["required_signal"], "contradiction")
        self.assertEqual(expected["required_signal_frames"], "any")

    def test_unresolved_conflict_accepts_only_review_safe_warning_signal(
        self,
    ) -> None:
        control = next(
            item
            for item in self.controls
            if item["control_id"]
            == "overconfident_unresolved_meter_simultaneity"
        )
        expected = control["expected"]
        self.assertEqual(
            expected["required_signal"],
            "contradiction_or_review_warning",
        )
        self.assertNotIn("pass", expected["allowed_verdicts"])

    def test_good_negations_require_explicit_range_nli_negation(
        self,
    ) -> None:
        for control_id in NEGATED_RANGE_CONTROLS:
            with self.subTest(control=control_id):
                control = next(
                    item
                    for item in self.controls
                    if item["control_id"] == control_id
                )
                self.assertEqual(
                    control["expected"]["required_signal"],
                    "range_negation",
                )
                self.assertEqual(
                    control["expected"]["allowed_relationships"],
                    list(evaluator.RELATIONSHIPS),
                )

    def test_range_controls_use_separate_nli_requirement(self) -> None:
        range_controls = {
            control["control_id"]
            for control in self.controls
            if evaluator.range_scope_requirement(
                control["reference_packet"]
            )["applicable"]
        }
        self.assertEqual(
            range_controls,
            {
                "good_early_echo_contrast",
                "good_early_shared_vocabulary_paraphrase",
                "good_early_negated_correction",
                "wrong_later_dynamics_projected_to_early_echo",
                "wrong_later_dynamics_paraphrased_at_early_echo",
                "good_later_rapid_alternation",
                "wrong_early_echo_projected_to_later_dynamics",
                "wrong_early_unseen_dense_alternation",
                "wrong_early_mixed_final_assertion",
                "wrong_early_double_negation",
                "good_early_natural_negation",
                "wrong_later_unseen_separated_blocks",
                "good_later_rapid_with_two_parts_contrast",
                "uncertain_early_range_projection",
            },
        )


class ExpectationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controls, _ = calibration.build_controls(DATASET_ROOT)

    def control(self, control_id: str) -> dict:
        return deepcopy(next(
            item
            for item in self.controls
            if item["control_id"] == control_id
        ))

    def test_synthetic_results_cover_every_expected_verdict_and_signal(
        self,
    ) -> None:
        for source in self.controls:
            with self.subTest(control=source["control_id"]):
                control = complete_control(deepcopy(source))
                result = control["expectation_evaluation"]
                self.assertTrue(result["match"], result)
                self.assertIn(
                    result["actual_verdict"],
                    control["expected"].get(
                        "allowed_verdicts",
                        [control["expected"]["verdict"]],
                    ),
                )

    def test_unanchored_contradiction_is_review_signal_not_strict_signal(
        self,
    ) -> None:
        payload = assessment_payload("claim_alignment")
        payload["critical_error_types"] = ["contradicts_expert"]
        assessment = evaluator.validate_judge_assessment(
            json.dumps(payload, ensure_ascii=False),
            expected_frame="claim_alignment",
            has_measure_range=True,
            candidate_answer="후보 답변",
            authoritative_reference_items=(
                evaluator._authoritative_reference_items(
                    self.control(
                        "overconfident_unresolved_meter_simultaneity"
                    )["reference_packet"]
                )
            ),
            candidate_answer_items=[
                {"candidate_id": "C001", "text": "후보 답변"},
                {"candidate_id": "C002", "text": "다른 배경이다."},
            ],
        )
        self.assertFalse(
            calibration._assessment_has_required_signal(
                assessment,
                "contradiction",
            )
        )
        self.assertTrue(
            calibration._assessment_has_required_signal(
                assessment,
                "contradiction_or_review_warning",
            )
        )
        self.assertFalse(
            evaluator.compute_frame_result(assessment)["pass"]
        )

    def test_wrong_range_must_be_signaled_by_range_scope_nli(self) -> None:
        control = self.control(
            "wrong_later_dynamics_projected_to_early_echo"
        )
        all_absent = complete_control(
            deepcopy(control),
            forced_range_relation="absent",
        )
        self.assertEqual(
            all_absent["aggregate"]["status"],
            "human_review",
        )
        self.assertFalse(
            all_absent["expectation_evaluation"]["match"]
        )
        self.assertFalse(
            all_absent["expectation_evaluation"][
                "range_scope_signal_match"
            ]
        )

        asserted = complete_control(
            control,
            forced_range_relation="asserted",
        )
        self.assertEqual(asserted["aggregate"]["status"], "fail")
        self.assertTrue(asserted["expectation_evaluation"]["match"])
        self.assertTrue(
            asserted["expectation_evaluation"][
                "range_scope_signal_match"
            ]
        )

    def test_negated_and_uncertain_range_relations_are_distinct(self) -> None:
        no_negation_signal = complete_control(
            self.control("good_early_negated_correction"),
            forced_range_relation="absent",
        )
        self.assertEqual(
            no_negation_signal["aggregate"]["status"],
            "human_review",
        )
        self.assertFalse(
            no_negation_signal["expectation_evaluation"]["match"]
        )
        self.assertFalse(
            no_negation_signal["expectation_evaluation"][
                "range_scope_signal_match"
            ]
        )

        negated = complete_control(
            self.control("good_early_negated_correction"),
            forced_range_relation="negated",
        )
        self.assertEqual(
            negated["aggregate"]["status"],
            "human_review",
        )
        self.assertTrue(all(
            not state["computed"]["pass"]
            for state in negated["frames"].values()
        ))
        self.assertTrue(all(
            "no_covered_direct_required_reference"
            in state["computed"]["failure_reasons"]
            for state in negated["frames"].values()
        ))
        self.assertEqual(
            negated["range_scope"]["assessment"]["status"],
            "pass",
        )
        self.assertTrue(
            negated["expectation_evaluation"][
                "range_scope_signal_match"
            ]
        )
        negated["range_scope"]["assessment"]["status"] = "human_review"
        unlocalized_signal = calibration.evaluate_control_expectation(
            negated
        )
        self.assertTrue(unlocalized_signal["range_scope_signal_match"])
        self.assertTrue(unlocalized_signal["match"])

        uncertain = complete_control(
            self.control("uncertain_early_range_projection"),
            forced_range_relation="uncertain",
        )
        self.assertEqual(uncertain["aggregate"]["status"], "human_review")
        self.assertTrue(uncertain["expectation_evaluation"]["match"])
        self.assertTrue(
            uncertain["expectation_evaluation"][
                "range_scope_signal_match"
            ]
        )

        double_negation = complete_control(
            self.control("wrong_early_double_negation"),
            forced_range_relation="uncertain",
        )
        self.assertEqual(
            double_negation["aggregate"]["status"],
            "human_review",
        )
        self.assertNotEqual(
            double_negation["aggregate"]["status"],
            "pass",
        )
        self.assertTrue(
            double_negation["expectation_evaluation"]["match"]
        )
        self.assertTrue(
            double_negation["expectation_evaluation"][
                "range_scope_signal_match"
            ]
        )

    def test_unrelated_background_fails_without_inventing_hard_error(
        self,
    ) -> None:
        control = complete_control(
            self.control("unrelated_same_composer_background_only")
        )
        self.assertEqual(control["aggregate"]["status"], "fail")
        self.assertTrue(control["expectation_evaluation"]["match"])
        for state in control["frames"].values():
            self.assertEqual(
                state["assessment"]["critical_error_types"],
                [],
            )
            self.assertEqual(
                state["assessment"]["relationship"],
                "unrelated",
            )

    def test_partial_diction_answer_rejects_transferred_consonant_coverage(
        self,
    ) -> None:
        control = complete_control(
            self.control("minor_omission_lerbe_e_melisma")
        )
        self.assertTrue(control["expectation_evaluation"]["match"])

        for state in control["frames"].values():
            assessments = state["assessment"]
            consonant_reference = next(
                item
                for item in assessments["reference_assessments"]
                if "d발음" in item["reference_text"]
            )
            candidate = assessments["candidate_assessments"][0]
            consonant_reference.update(
                status="covered",
                candidate_ids=[candidate["candidate_id"]],
            )
            candidate["reference_ids"].append(
                consonant_reference["reference_id"]
            )
            state["computed"] = evaluator.compute_frame_result(
                assessments
            )

        control["aggregate"] = evaluator.aggregate_judge_frames(
            control["frames"],
            generated_answer=control["generated_answer"],
            range_scope_evaluation=control["range_scope"]["assessment"],
            reference_review=control["reference_review"],
            supplemental_evidence_validation=control[
                "reference_packet"
            ]["supplemental_evidence_validation"],
            review_disclosure_validation=control[
                "reference_packet"
            ]["review_disclosure_validation"],
        )
        result = calibration.evaluate_control_expectation(control)
        self.assertFalse(result["match"])
        self.assertFalse(
            result["checks"]["forbidden_reference_points_not_claimed"]
        )
        self.assertEqual(
            set(result["forbidden_coverage_matches"]),
            set(evaluator.FRAME_NAMES),
        )
        self.assertTrue(all(
            result["forbidden_coverage_matches"][frame]
            for frame in evaluator.FRAME_NAMES
        ))

    def test_literal_guard_rejects_raw_false_coverage_for_automatic_pass(
        self,
    ) -> None:
        control = self.control("minor_omission_lerbe_e_melisma")
        packet = control["reference_packet"]
        references = evaluator._authoritative_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)
        for frame in evaluator.FRAME_NAMES:
            raw = json.dumps(
                assessment_payload(
                    frame,
                    reference_items=references,
                    candidate_items=candidates,
                ),
                ensure_ascii=False,
            )
            validated = evaluator.validate_judge_assessment(
                raw,
                expected_frame=frame,
                has_measure_range=True,
                candidate_answer=packet["candidate_answer"],
                authoritative_reference_items=references,
                candidate_answer_items=candidates,
                allowed_context_texts=(
                    [packet["question"]]
                    + evaluator._authoritative_question_contexts(packet)
                ),
            )
            self.assertTrue(
                validated["literal_anchor_coverage_warnings"]
            )
            control["frames"][frame] = {
                "status": "complete",
                "attempts": [{"raw_response": raw}],
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }
        attach_range_state(control)
        control["aggregate"] = evaluator.aggregate_judge_frames(
            control["frames"],
            generated_answer=control["generated_answer"],
            range_scope_evaluation=control["range_scope"]["assessment"],
            reference_review=control["reference_review"],
            supplemental_evidence_validation=control[
                "reference_packet"
            ]["supplemental_evidence_validation"],
            review_disclosure_validation=control[
                "reference_packet"
            ]["review_disclosure_validation"],
        )

        result = calibration.evaluate_control_expectation(control)

        self.assertEqual(control["aggregate"]["status"], "human_review")
        self.assertTrue(result["match"])
        self.assertTrue(all(
            result["raw_forbidden_coverage_matches"][frame]
            for frame in evaluator.FRAME_NAMES
        ))
        self.assertTrue(all(
            result["literal_anchor_guard_rejections"][frame]
            for frame in evaluator.FRAME_NAMES
        ))
        self.assertTrue(all(
            not result["forbidden_coverage_matches"][frame]
            for frame in evaluator.FRAME_NAMES
        ))


class CalibrationRunTests(unittest.TestCase):
    def test_runner_uses_both_semantic_frames_and_checkpoints(self) -> None:
        controls, _ = calibration.build_controls(DATASET_ROOT)
        control = deepcopy(controls[0])
        snapshot = calibration.new_snapshot(
            controls=[control],
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            judge_model={"path": "Qwen3-4B"},
            range_guard=TEST_RANGE_GUARD,
            max_tokens=512,
        )
        calls = []
        checkpoints = 0

        def judge_fn(messages):
            system = messages[0]["content"]
            frame = (
                "claim_alignment"
                if "Frame: claim_alignment" in system
                else "contradiction_first"
            )
            calls.append(frame)
            return semantic_raw(control, frame)

        def checkpoint():
            nonlocal checkpoints
            checkpoints += 1

        calibration.run_calibration(
            snapshot,
            judge_fn=judge_fn,
            range_signal_fn=hybrid_range_signals,
            coverage_signal_fn=atomic_coverage_signals,
            range_nli_threshold=0.8,
            range_embedding_delta_threshold=0.05,
            checkpoint=checkpoint,
            max_attempts=2,
        )
        self.assertEqual(calls, list(evaluator.FRAME_NAMES))
        self.assertGreaterEqual(checkpoints, 3)
        self.assertTrue(
            snapshot["controls"][0]["expectation_evaluation"]["match"]
        )

    def test_runner_can_execute_a_resumable_control_subset(self) -> None:
        controls, _ = calibration.build_controls(DATASET_ROOT)
        selected = deepcopy(controls[0])
        skipped = deepcopy(controls[1])
        snapshot = calibration.new_snapshot(
            controls=[selected, skipped],
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            judge_model={"path": "Qwen3-4B"},
            range_guard=TEST_RANGE_GUARD,
            max_tokens=512,
        )

        def judge_fn(messages):
            frame = (
                "claim_alignment"
                if "Frame: claim_alignment" in messages[0]["content"]
                else "contradiction_first"
            )
            return semantic_raw(selected, frame)

        calibration.run_calibration(
            snapshot,
            judge_fn=judge_fn,
            range_signal_fn=hybrid_range_signals,
            coverage_signal_fn=atomic_coverage_signals,
            range_nli_threshold=0.8,
            range_embedding_delta_threshold=0.05,
            checkpoint=lambda: None,
            max_attempts=2,
            selected_control_ids={selected["control_id"]},
        )
        self.assertIsNotNone(
            snapshot["controls"][0]["expectation_evaluation"]
        )
        self.assertIsNone(
            snapshot["controls"][1]["expectation_evaluation"]
        )
        self.assertEqual(snapshot["summary"]["completed_controls"], 1)

    def test_runner_executes_range_scope_as_a_third_nli_frame(self) -> None:
        controls, _ = calibration.build_controls(DATASET_ROOT)
        control = deepcopy(next(
            item
            for item in controls
            if item["control_id"] == "good_early_negated_correction"
        ))
        snapshot = calibration.new_snapshot(
            controls=[control],
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            judge_model={"path": "Qwen3-4B"},
            range_guard=TEST_RANGE_GUARD,
            max_tokens=512,
        )
        calls = []

        def judge_fn(messages):
            system = messages[0]["content"]
            if "checking whether" in system:
                calls.append("range_scope")
                return range_raw(control, relation="negated")
            frame = (
                "claim_alignment"
                if "Frame: claim_alignment" in system
                else "contradiction_first"
            )
            calls.append(frame)
            return semantic_raw(control, frame)

        calibration.run_calibration(
            snapshot,
            judge_fn=judge_fn,
            range_signal_fn=hybrid_range_signals,
            coverage_signal_fn=atomic_coverage_signals,
            range_nli_threshold=0.8,
            range_embedding_delta_threshold=0.05,
            checkpoint=lambda: None,
            max_attempts=2,
        )
        self.assertEqual(
            calls,
            [*evaluator.FRAME_NAMES, evaluator.RANGE_FRAME_NAME],
        )
        self.assertEqual(
            snapshot["controls"][0]["range_scope"]["assessment"][
                "excluded_claim_assessments"
            ][0]["relation"],
            "negated",
        )
        self.assertTrue(
            snapshot["controls"][0]["expectation_evaluation"]["match"]
        )

    def test_atomic_snapshot_round_trip_keeps_all_25_controls(self) -> None:
        controls, _ = calibration.build_controls(DATASET_ROOT)
        snapshot = calibration.new_snapshot(
            controls=controls,
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            judge_model={"path": "Qwen3-4B"},
            range_guard=TEST_RANGE_GUARD,
            max_tokens=512,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            evaluator.atomic_write_json(output, snapshot)
            loaded = evaluator.load_json(output)
        self.assertEqual(loaded["schema_version"], calibration.SCHEMA_VERSION)
        self.assertEqual(
            len(loaded["controls"]),
            calibration.EXPECTED_CONTROL_COUNT,
        )


class CalibrationArtifactReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.judge_model_path = calibration.default_qwen3_4b_path()
        cls.expected = calibration.expected_calibration_inputs(
            dataset_root=DATASET_ROOT,
            judge_model_path=cls.judge_model_path,
            range_embedding_model_path=(
                evaluator.DEFAULT_RANGE_EMBEDDING_MODEL_PATH
            ),
            range_nli_model_path=evaluator.DEFAULT_RANGE_NLI_MODEL_PATH,
            range_nli_threshold=0.8,
            range_embedding_delta_threshold=0.05,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        cls.artifact = complete_artifact(cls.expected)
        mismatches = [
            control["control_id"]
            for control in cls.artifact["controls"]
            if not control["expectation_evaluation"]["match"]
        ]
        if mismatches:
            raise AssertionError(
                f"synthetic calibration controls did not match: {mismatches}"
            )

    def validate(
        self,
        artifact: dict,
        *,
        model: dict | None = None,
        reconstruct_inputs: bool = False,
    ) -> dict:
        reconstruction = (
            nullcontext()
            if reconstruct_inputs
            else mock.patch.object(
                calibration,
                "expected_calibration_inputs",
                return_value=deepcopy(self.expected),
            )
        )
        with reconstruction:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "calibration.json"
                evaluator.atomic_write_json(output, artifact)
                return evaluator.validate_judge_calibration(
                    output,
                    judge_model=model or self.expected["judge_model"],
                    range_guard=self.expected["range_guard"],
                    dataset_root=DATASET_ROOT,
                    judge_max_tokens=JUDGE_MAX_TOKENS,
                )

    def assert_rejected(self, artifact: dict) -> None:
        with self.assertRaises(evaluator.EvaluationInputError):
            self.validate(artifact)

    def test_complete_artifact_is_reconstructed_from_raw_model_outputs(
        self,
    ) -> None:
        record = self.validate(
            deepcopy(self.artifact),
            reconstruct_inputs=True,
        )
        self.assertTrue(record["quality_gate_passed"])
        self.assertEqual(record["total_controls"], 25)
        self.assertEqual(record["matched_controls"], 25)

    def test_forged_contract_state_raw_output_and_summary_are_rejected(
        self,
    ) -> None:
        mutations = {}

        contract = deepcopy(self.artifact)
        contract["controls"][0]["candidate_answer"] += " 위조"
        mutations["contract"] = contract

        assessment = deepcopy(self.artifact)
        assessment["controls"][0]["frames"]["claim_alignment"][
            "assessment"
        ]["relationship"] = "unrelated"
        mutations["assessment"] = assessment

        raw_output = deepcopy(self.artifact)
        raw_output["controls"][0]["frames"]["claim_alignment"][
            "attempts"
        ][-1]["raw_response"] = "{}"
        mutations["raw_output"] = raw_output

        aggregate = deepcopy(self.artifact)
        aggregate["controls"][0]["aggregate"]["status"] = "fail"
        mutations["aggregate"] = aggregate

        range_output = deepcopy(self.artifact)
        range_control = next(
            item
            for item in range_output["controls"]
            if item["control_id"]
            == "wrong_later_dynamics_projected_to_early_echo"
        )
        range_control["range_scope"]["assessment"][
            "wrong_measure_application"
        ] = False
        mutations["range_output"] = range_output

        summary = deepcopy(self.artifact)
        summary["summary"]["matched_controls"] = 22
        mutations["summary"] = summary

        manifest = deepcopy(self.artifact)
        manifest["run"]["input_files"][0]["sha256"] = "forged"
        mutations["input_manifest"] = manifest

        truncated = deepcopy(self.artifact)
        truncated["controls"] = truncated["controls"][:1]
        mutations["truncated_controls"] = truncated

        for name, artifact in mutations.items():
            with self.subTest(forgery=name):
                self.assert_rejected(artifact)

    def test_forged_evaluation_model_identity_is_rejected(self) -> None:
        forged_model = deepcopy(self.expected["judge_model"])
        forged_model["sha256"] = "forged"
        with self.assertRaises(evaluator.EvaluationInputError):
            self.validate(
                deepcopy(self.artifact),
                model=forged_model,
            )


if __name__ == "__main__":
    unittest.main()
