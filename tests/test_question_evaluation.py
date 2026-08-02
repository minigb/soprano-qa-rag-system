"""Focused tests for the resumable schema-8 reliability evaluator."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation import run_question_evaluation as evaluator


DATASET_ROOT = Path("/home/minhee/soprano-qa-dataset")
SYNTHESIZED_DATASET_ROOT = Path(
    "/home/minhee/soprano-qa-dataset-evaluation-set-synthesized"
)
DENSE_SYSTEM_ROOT = Path(
    "/home/minhee/soprano-qa-rag-system-dense-retrieval"
)
TEST_RANGE_GUARD = {
    "embedding_model": {"path": "embedding", "sha256": "embedding"},
    "nli_model": {"path": "nli", "sha256": "nli"},
    "thresholds": {
        "nli_entailment_or_contradiction": 0.8,
        "embedding_contrast_delta": 0.05,
    },
}
TEST_REFERENCE_ITEMS = [
    {"reference_id": "R001", "text": "핵심 조언을 보존한다."},
    {"reference_id": "R002", "text": "보조 지침도 설명한다."},
    {"reference_id": "R003", "text": "페르마타는 음가보다 길게 늘인다."},
]
SCOPED_REFERENCE_ITEMS = [
    {
        "reference_id": "R001",
        "text": "핵심 조언을 보존한다.",
        "scope": "direct_required",
        "flags": [],
        "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
    },
    {
        "reference_id": "R002",
        "text": "보조 배경을 설명한다.",
        "scope": "optional_background",
        "flags": [],
        "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
    },
]
TEST_CANDIDATE_ITEMS = [
    {"candidate_id": "C001", "text": "핵심 조언을 보존한다."},
    {"candidate_id": "C002", "text": "질문과 무관한 배경이다."},
]


def hybrid_range_signals(
    packet: dict,
    *,
    embedding_delta: float = -0.1,
    entailment: float = 0.05,
    contradiction: float = 0.05,
    candidate_nli: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Return internally consistent synthetic local-model range signals."""

    allowed = evaluator._allowed_range_contrast_items(packet)
    excluded = evaluator._excluded_reference_items(packet)
    candidates = evaluator._candidate_answer_items(packet)
    neutral = 1.0 - entailment - contradiction
    allowed_similarity = 0.6
    excluded_similarity = allowed_similarity + embedding_delta
    return {
        "excluded_claim_signals": [
            {
                "excluded_id": item["reference_id"],
                "embedding": {
                    "candidate_vs_excluded": excluded_similarity,
                    "allowed_similarities": [
                        {
                            "allowed_id": allowed_item["reference_id"],
                            "cosine_similarity": allowed_similarity,
                        }
                        for allowed_item in allowed
                    ],
                    "best_allowed_id": allowed[0]["reference_id"],
                    "candidate_vs_best_allowed": allowed_similarity,
                    "contrast_delta": embedding_delta,
                },
                "nli": {
                    "entailment": entailment,
                    "neutral": neutral,
                    "contradiction": contradiction,
                },
                "candidate_nli": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        **(
                            candidate_nli.get(candidate["candidate_id"])
                            if candidate_nli
                            and candidate["candidate_id"] in candidate_nli
                            else {
                                "entailment": 0.05,
                                "neutral": 0.90,
                                "contradiction": 0.05,
                            }
                        ),
                    }
                    for candidate in candidates
                ],
            }
            for item in excluded
        ]
    }


def atomic_coverage_signals(
    packet: dict,
    *,
    entailment: float = 0.95,
    contradiction: float = 0.01,
    overrides: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Return exhaustive synthetic candidate-to-guarded-reference NLI."""

    candidates = evaluator._candidate_answer_items(packet)
    references = evaluator._coverage_guard_reference_items(
        evaluator._all_semantic_reference_items(packet)
    )

    def scores(reference_id: str) -> dict[str, float]:
        if overrides and reference_id in overrides:
            return dict(overrides[reference_id])
        return {
            "entailment": entailment,
            "neutral": 1.0 - entailment - contradiction,
            "contradiction": contradiction,
        }

    return {
        "atomic_claim_signals": [
            {
                "reference_id": reference["reference_id"],
                "full_answer_nli": scores(reference["reference_id"]),
                "candidate_nli": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        **scores(reference["reference_id"]),
                    }
                    for candidate in candidates
                ],
                "contextualized_candidate_nli": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "context_text": (
                            evaluator
                            ._candidate_with_question_subject_context(
                                packet["question"],
                                candidate["text"],
                            )
                        ),
                        **scores(reference["reference_id"]),
                    }
                    for candidate in candidates
                ],
            }
            for reference in references
        ]
    }


def assessment_payload(
    frame: str,
    *,
    reference_items: list[dict] | None = None,
    candidate_items: list[dict] | None = None,
    mode: str = "equivalent",
    has_range: bool = True,
    confidence: float = 0.9,
) -> dict:
    """Return an exhaustive raw judge response for a controlled relation."""

    references = reference_items or TEST_REFERENCE_ITEMS
    candidates = candidate_items or TEST_CANDIDATE_ITEMS
    reference_ids = [item["reference_id"] for item in references]
    candidate_ids = [item["candidate_id"] for item in candidates]
    if not reference_ids or not candidate_ids:
        raise AssertionError("fixtures require at least one R and one C item")
    first_reference = reference_ids[0]
    first_candidate = candidate_ids[0]

    reference_assessments = [
        {
            "reference_id": reference_id,
            "status": "not_required",
            "candidate_ids": [],
        }
        for reference_id in reference_ids
    ]
    candidate_assessments = [
        {
            "candidate_id": candidate_id,
            "status": "irrelevant",
            "reference_ids": [],
        }
        for candidate_id in candidate_ids
    ]
    scores = {
        "core_correctness": 4,
        "core_coverage": 4,
        "factual_safety": 4,
        "question_relevance": 4,
        "range_consistency": 4 if has_range else None,
    }
    relationship = "equivalent"
    critical_errors: list[str] = []
    rationale = "질문에 필요한 전문가 핵심을 충실히 보존한다."

    if mode == "equivalent":
        for reference in reference_assessments:
            reference.update(
                status="covered",
                candidate_ids=candidate_ids,
            )
        for candidate in candidate_assessments:
            candidate.update(
                status="supported",
                reference_ids=reference_ids,
            )
    elif mode == "minor_omission":
        reference_assessments[0].update(
            status="covered",
            candidate_ids=[first_candidate],
        )
        candidate_assessments[0].update(
            status="supported",
            reference_ids=[first_reference],
        )
        if len(reference_assessments) > 1:
            reference_assessments[1]["status"] = "missing"
        relationship = "minor_omission"
        scores["core_coverage"] = 3
        rationale = "보조 지침 하나를 생략했지만 핵심은 보존한다."
    elif mode == "contradiction":
        reference_assessments[0].update(
            status="contradicted",
            candidate_ids=candidate_ids,
        )
        for candidate in candidate_assessments:
            candidate.update(
                status="contradicted",
                reference_ids=[first_reference],
            )
        relationship = "contradictory"
        scores["core_correctness"] = 1
        scores["factual_safety"] = 1
        critical_errors = ["contradicts_expert"]
        rationale = "후보가 전문가의 핵심을 정면으로 뒤집는다."
    elif mode == "unsupported":
        reference_assessments[0].update(
            status="covered",
            candidate_ids=[first_candidate],
        )
        candidate_assessments[0].update(
            status="mixed",
            reference_ids=[first_reference],
        )
        relationship = "materially_unreliable"
        scores["factual_safety"] = 1
        critical_errors = ["ungrounded_as_fact"]
        rationale = "지원되는 핵심에 근거 없는 물질적 주장을 덧붙인다."
    elif mode == "unrelated":
        for reference in reference_assessments:
            reference["status"] = "missing"
        relationship = "unrelated"
        scores.update(
            core_correctness=1,
            core_coverage=0,
            question_relevance=0,
        )
        rationale = "같은 작곡가의 배경만 말하고 질문에는 답하지 않는다."
    elif mode == "supplement_only":
        supplemental_ids = {
            item["reference_id"]
            for item in references
            if item.get("scope_authority")
            == evaluator.RETRIEVED_EXPERT_SCOPE_AUTHORITY
        }
        if not supplemental_ids:
            raise AssertionError(
                "supplement_only fixture requires at least one S item"
            )
        for item in reference_assessments:
            if item["reference_id"] in supplemental_ids:
                item.update(
                    status="covered",
                    candidate_ids=candidate_ids,
                )
            else:
                source = next(
                    reference
                    for reference in references
                    if reference["reference_id"]
                    == item["reference_id"]
                )
                item["status"] = (
                    "not_required"
                    if source.get("scope") == "optional_background"
                    else "missing"
                )
        for item in candidate_assessments:
            item.update(
                status="supported",
                reference_ids=sorted(supplemental_ids),
            )
        relationship = "materially_incomplete"
        scores["core_coverage"] = 1
        rationale = "검색된 보충 근거는 맞지만 필수 원문 답변이 빠졌다."
    else:
        raise AssertionError(f"unknown fixture mode: {mode}")

    return {
        "frame": frame,
        "relationship": relationship,
        "scores": scores,
        "reference_assessments": reference_assessments,
        "candidate_assessments": candidate_assessments,
        "critical_error_types": critical_errors,
        "confidence": confidence,
        "rationale": rationale,
    }


def validate_payload(
    payload: dict,
    *,
    reference_items: list[dict] | None = None,
    candidate_items: list[dict] | None = None,
    has_range: bool = True,
    atomic_signals: dict | None = None,
    atomic_nli_threshold: float | None = None,
) -> dict:
    references = reference_items or TEST_REFERENCE_ITEMS
    candidates = candidate_items or TEST_CANDIDATE_ITEMS
    return evaluator.validate_judge_assessment(
        json.dumps(payload, ensure_ascii=False),
        expected_frame=payload["frame"],
        has_measure_range=has_range,
        candidate_answer=" ".join(item["text"] for item in candidates),
        authoritative_reference_items=references,
        candidate_answer_items=candidates,
        atomic_coverage_signals=atomic_signals,
        atomic_coverage_nli_threshold=atomic_nli_threshold,
    )


def frame_states(
    *,
    mode: str = "equivalent",
    confidence: float = 0.9,
) -> dict[str, dict]:
    states = {}
    for frame in evaluator.FRAME_NAMES:
        validated = validate_payload(
            assessment_payload(
                frame,
                mode=mode,
                confidence=confidence,
            )
        )
        states[frame] = {
            "assessment": validated,
            "computed": evaluator.compute_frame_result(validated),
        }
    return states


def grounded_answer() -> dict:
    return {
        "generation_mode": "llm",
        "answer_basis": "retrieved_evidence",
        "evidence": [{"id": "unit", "kind": "expert"}],
        "diagnostics": {"target_source_grounded": True},
    }


class QuestionInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions, cls.paths = evaluator.load_evaluation_questions(
            DATASET_ROOT
        )
        cls.expert_catalog = (
            evaluator.load_trusted_expert_evidence_catalog(
                evaluator.PROJECT_ROOT / "data" / "corpus.json"
            )
        )

    @staticmethod
    def trusted_evidence(
        record: dict,
        selected_range: list[int] | None,
    ) -> dict:
        from soprano_qa.answer import expert_prompt_factuality_material
        from soprano_qa.retrieval import (
            measure_scope_match,
            scope_evidence_role,
        )

        selected_ranges = (
            [selected_range]
            if selected_range is not None
            else []
        )
        scope_match = measure_scope_match(record, selected_ranges)
        return {
            "id": record["id"],
            "kind": "expert",
            "evidence_type": "expert_annotation",
            "piece_id": record["piece"],
            "source_ids": list(record["source_ids"]),
            "measure_ranges": deepcopy(record.get("measure_range", [])),
            "scope_match": scope_match,
            "generation_role": scope_evidence_role(scope_match),
            "selected_range_claim_authority": (
                scope_match == "overlaps_query_range"
                if selected_range is not None
                else None
            ),
            "in_requested_scope": (
                scope_match == "overlaps_query_range"
                if selected_range is not None
                else scope_match
                in {
                    "general_evidence",
                    "local_example",
                    "unscoped_pending_review",
                    "unspecified_scope",
                }
            ),
            "text": record["answer"],
            "rewrite_status": record["rewrite_status"],
            "rewrite_notes": record.get("rewrite_notes", ""),
            "measure_status": record["measure_status"],
            "measure_notes": record.get("measure_notes", ""),
            "retrieval_review_warning": record.get(
                "retrieval_review_warning",
                "",
            ),
            "generation_expert_authority": (
                expert_prompt_factuality_material(
                    record,
                    [selected_range]
                    if selected_range is not None
                    else [],
                )
            ),
        }

    def supplemental_packet(self, answer: str) -> dict:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-die-forelle-01"
        )
        case = deepcopy(question["inference_runs"][0])
        record = self.expert_catalog["die-forelle-ku-001"]
        case["generated_answer"] = {
            "piece_id": question["piece_id"],
            "measure_range": None,
            "answer": answer,
            "generation_mode": "llm",
            "answer_basis": "retrieved_evidence",
            "evidence": [self.trusted_evidence(record, None)],
            "diagnostics": {"target_source_grounded": True},
        }
        return evaluator._reference_packet(
            question,
            case,
            trusted_expert_catalog=self.expert_catalog,
        )

    def warned_supplemental_packet(
        self,
        answer: str,
        *,
        mutate_evidence: object | None = None,
    ) -> tuple[dict, str]:
        from soprano_qa.answer import (
            build_review_disclosure_from_records,
        )

        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-in-flowery-clouds-07"
        )
        case = deepcopy(question["inference_runs"][0])
        record = self.expert_catalog["in-flowery-clouds-ku-008"]
        evidence = self.trusted_evidence(
            record,
            case["inference_input"]["measure_range"],
        )
        if mutate_evidence is not None:
            mutate_evidence(evidence)
        disclosure = build_review_disclosure_from_records([record])
        case["generated_answer"] = {
            "piece_id": question["piece_id"],
            "measure_range": deepcopy(
                case["inference_input"]["measure_range"]
            ),
            "answer": answer,
            "generation_mode": "llm",
            "answer_basis": "retrieved_evidence",
            "evidence": [evidence],
            "diagnostics": {"target_source_grounded": True},
        }
        return (
            evaluator._reference_packet(
                question,
                case,
                trusted_expert_catalog=self.expert_catalog,
            ),
            disclosure,
        )

    def test_inventory_is_exactly_40_questions_and_51_valid_cases(self) -> None:
        self.assertEqual(len(self.questions), 40)
        self.assertEqual(
            sum(len(item["inference_runs"]) for item in self.questions),
            evaluator.EXPECTED_CASE_COUNT,
        )
        self.assertEqual(evaluator.EXPECTED_CASE_COUNT, 51)
        self.assertEqual(
            {item["piece_id"] for item in self.questions},
            set(evaluator.TARGET_PIECES),
        )
        self.assertNotIn(
            "nella-fantasia",
            {item["piece_id"] for item in self.questions},
        )
        self.assertNotIn(
            "una-voce-poco-fa",
            {item["piece_id"] for item in self.questions},
        )

    def test_only_the_explicitly_invalid_reprise_pairing_is_excluded(
        self,
    ) -> None:
        raw_path = (
            DATASET_ROOT
            / "expert_curation"
            / "evaluation_questions"
            / "la-capinera.json"
        )
        raw = evaluator.load_json(raw_path)
        source = next(
            item
            for item in raw["questions"]
            if item["source_id"] == "kim-la-capinera-01"
        )
        self.assertEqual(
            source["inference_measure_ranges"],
            [[11, 14], [78, 81]],
        )
        self.assertEqual(
            set(evaluator.EXCLUDED_EVALUATION_CASES),
            {("kim-la-capinera-01", (78, 81))},
        )
        loaded = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-la-capinera-01"
        )
        self.assertEqual(
            [
                run["inference_input"]["measure_range"]
                for run in loaded["inference_runs"]
            ],
            [[11, 14]],
        )
        raw_case_count = 0
        for path in self.paths:
            if path.parent.name != "evaluation_questions":
                continue
            raw_case_count += sum(
                len(item["inference_measure_ranges"])
                if item["inference_measure_ranges"]
                else 1
                for item in evaluator.load_json(path)["questions"]
            )
        self.assertEqual(raw_case_count, 52)

    def test_authoritative_answer_is_verbatim_source_answer(self) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-die-forelle-01"
        )
        review = evaluator.load_json(
            DATASET_ROOT
            / "expert_curation"
            / "review"
            / "die-forelle.json"
        )
        source = next(
            item
            for item in review["source_annotations"]
            if item["source_id"] == question["source_id"]
        )
        self.assertEqual(
            question["authoritative_reference"]["source_answer"],
            source["answer"],
        )
        self.assertEqual(
            question["authoritative_reference"][
                "source_legacy_measure_range_hints"
            ],
            source["legacy_measure_ranges"],
        )
        scope = question["reference_claim_scope"]
        self.assertEqual(
            scope,
            question["authoritative_reference"]["reference_claim_scope"],
        )
        self.assertEqual(
            scope["source_answer_sha256"],
            hashlib.sha256(
                source["answer"].encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(
            [item["text"] for item in scope["sentence_items"]],
            evaluator._source_answer_sentences(source["answer"]),
        )

    def test_curated_scope_is_carried_to_reference_items(self) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-die-forelle-02"
        )
        case = deepcopy(question["inference_runs"][0])
        case["generated_answer"] = {
            "answer": "두 번째 박의 악센트는 송어의 움직임을 표현한다.",
            "evidence": [],
        }
        packet = evaluator._reference_packet(question, case)
        references = evaluator._authoritative_reference_items(packet)
        self.assertEqual(
            [item["scope"] for item in references],
            [
                "direct_required",
                "direct_required",
                "direct_required",
                "optional_background",
                "optional_background",
            ],
        )
        self.assertTrue(all(
            item["scope_authority"]
            == evaluator.QUESTION_SCOPE_AUTHORITY
            for item in references
        ))
        self.assertEqual(
            [item["text"] for item in references],
            [
                "반주부 높은음자리에서 보이는 두번째 박자의 악센트는 "
                "송어가 뛰어노는 것을 표현한 것이라고 볼 수 있다.",
                "두번째 박자의 악센트는 높은음자리표와 낮은음자리표를 "
                "이동하면서 반복되어서 보여지고,",
                "흐르는 물에서 뛰어오르는 송어의 느낌을 잘 나타낸다.",
                "이와같이 슈베르트는 반주부에 곡의 분위기와 배경, "
                "상황등을 잘 표현해놓았으며, 이것이 슈베르트를 가곡의 "
                "왕으로 부르게 되는 이유 중 하나이기도 하다.",
                "이 전 작곡가들과 달리 슈베르트는 시를 피아노 반주로 "
                "표현했던 첫번째 작곡가이고, 반주부의 격을 성악가와 "
                "동등한 위치로 높여주었던 작곡가이다.",
            ],
        )
        self.assertEqual(
            [item["source_claim_index"] for item in references],
            [None, 1, 2, None, None],
        )

    def test_other_source_fallback_is_supporting_only(self) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-la-capinera-17"
        )
        case = deepcopy(next(
            item
            for item in question["inference_runs"]
            if item["inference_input"]["measure_range"] == [51, 58]
        ))
        case["generated_answer"] = {
            "answer": "두 블록의 대비가 메아리 효과를 만든다.",
            "evidence": [],
        }
        packet = evaluator._reference_packet(question, case)
        references = evaluator._authoritative_reference_items(packet)
        self.assertTrue(references)
        self.assertTrue(all(
            item["scope"] == "optional_background"
            and item["scope_authority"]
            == evaluator.SUPPORTING_SCOPE_AUTHORITY
            for item in references
        ))
        review = evaluator.reference_review_requirement(packet)
        self.assertTrue(review["required"])
        self.assertIn(
            "question_level_reference_scope_not_applicable_to_case",
            review["reasons"],
        )
        self.assertIn(
            "no_applicable_direct_required_reference",
            review["reasons"],
        )

    def test_unresolved_curation_note_is_linkable_optional_authority(
        self,
    ) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-in-flowery-clouds-07"
        )
        case = deepcopy(question["inference_runs"][0])
        case["generated_answer"] = {
            "answer": (
                "자료마다 조성 변화 시점 설명이 달라 정확한 시점은 "
                "단정하기 어렵다."
            ),
            "evidence": [],
        }
        packet = evaluator._reference_packet(question, case)
        references = evaluator._authoritative_reference_items(packet)
        review_item = next(
            item
            for item in references
            if item["scope_authority"]
            == evaluator.CURATION_REVIEW_SCOPE_AUTHORITY
        )
        self.assertEqual(review_item["scope"], "optional_background")
        self.assertEqual(
            review_item["knowledge_unit_id"],
            "in-flowery-clouds-ku-008",
        )
        self.assertIn("정확한 전환 지점을 확인", review_item["text"])
        self.assertTrue(
            evaluator.reference_review_requirement(packet)["required"]
        )

    def test_support_only_reference_without_hard_error_requires_review(
        self,
    ) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-la-capinera-17"
        )
        case = deepcopy(question["inference_runs"][0])
        case["generated_answer"] = {
            "answer": "두 구간의 대비가 새소리의 메아리를 표현한다.",
            "evidence": [],
        }
        packet = evaluator._reference_packet(question, case)
        review = evaluator.reference_review_requirement(packet)
        references = evaluator._authoritative_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(
                frame,
                reference_items=references,
                candidate_items=candidates,
            )
            payload["relationship"] = "materially_incomplete"
            validated = validate_payload(
                payload,
                reference_items=references,
                candidate_items=candidates,
            )
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }

        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            reference_review=review,
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertEqual(
            aggregate["answer_quality_status"],
            "human_review",
        )
        self.assertTrue(aggregate["pipeline_is_grounded_rag_llm"])
        self.assertFalse(aggregate["reliable_rag_llm_pass"])
        self.assertIn(
            "support_only_reference_requires_review",
            aggregate["disagreement_reasons"],
        )

    def test_atomic_required_claim_blocks_partial_sentence_coverage(
        self,
    ) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-die-forelle-05"
        )
        case = deepcopy(question["inference_runs"][0])
        case["generated_answer"] = {
            "answer": "변형된 유절가곡으로 볼 수 있다.",
            "evidence": [],
        }
        packet = evaluator._reference_packet(question, case)
        review = evaluator.reference_review_requirement(packet)
        self.assertFalse(review["required"])
        references = evaluator._authoritative_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)
        required_atom_id = next(
            item["reference_id"]
            for item in references
            if item.get("source_sentence_index") == 3
            and item.get("source_claim_index") == 1
        )
        self.assertEqual(
            review["conditional_mixed_reference_ids"],
            [],
        )

        complete_states = {}
        partial_states = {}
        for frame in evaluator.FRAME_NAMES:
            complete = validate_payload(
                assessment_payload(
                    frame,
                    reference_items=references,
                    candidate_items=candidates,
                ),
                reference_items=references,
                candidate_items=candidates,
            )
            complete_states[frame] = {
                "assessment": complete,
                "computed": evaluator.compute_frame_result(complete),
            }

            partial_payload = assessment_payload(
                frame,
                reference_items=references,
                candidate_items=candidates,
            )
            required_atom = next(
                item
                for item in partial_payload["reference_assessments"]
                if item["reference_id"] == required_atom_id
            )
            required_atom.update(status="missing", candidate_ids=[])
            for candidate in partial_payload["candidate_assessments"]:
                candidate["reference_ids"] = [
                    reference_id
                    for reference_id in candidate["reference_ids"]
                    if reference_id != required_atom_id
                ]
            partial_payload["relationship"] = "minor_omission"
            partial_payload["scores"]["core_coverage"] = 3
            partial_payload["rationale"] = (
                "혼합 범위의 일부 내용을 생략했다."
            )
            partial = validate_payload(
                partial_payload,
                reference_items=references,
                candidate_items=candidates,
            )
            partial_states[frame] = {
                "assessment": partial,
                "computed": evaluator.compute_frame_result(partial),
            }

        aggregate = evaluator.aggregate_judge_frames(
            complete_states,
            generated_answer=grounded_answer(),
            reference_review=review,
        )
        self.assertEqual(aggregate["status"], "pass")
        self.assertFalse(
            aggregate["reference_review"][
                "conditional_mixed_fully_covered"
            ]
        )

        aggregate = evaluator.aggregate_judge_frames(
            partial_states,
            generated_answer=grounded_answer(),
            reference_review=review,
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertIn(
            "missing_authoritative_reference",
            partial_states["claim_alignment"]["computed"][
                "failure_reasons"
            ],
        )

    def test_questions_do_not_contain_measure_locators(self) -> None:
        for question in self.questions:
            with self.subTest(source_id=question["source_id"]):
                self.assertIsNone(
                    evaluator.ABSOLUTE_MEASURE_RE.search(
                        question["paraphrased_question"]
                    )
                )
                for case in question["inference_runs"]:
                    self.assertEqual(
                        case["inference_input"]["question"],
                        question["paraphrased_question"],
                    )

    def test_merged_unit_filters_other_range_in_both_directions(self) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-la-capinera-17"
        )
        expectations = {
            (51, 58): {
                "included": ["yeon-la-capinera-07"],
                "excluded": ["kim-la-capinera-17"],
                "original_available": False,
            },
            (117, 124): {
                "included": ["kim-la-capinera-17"],
                "excluded": ["yeon-la-capinera-07"],
                "original_available": True,
            },
        }
        for selected_range, expected in expectations.items():
            with self.subTest(selected_range=selected_range):
                case = deepcopy(next(
                    item
                    for item in question["inference_runs"]
                    if tuple(item["inference_input"]["measure_range"])
                    == selected_range
                ))
                case["generated_answer"] = {
                    "answer": "범위별 대비를 설명한다.",
                    "evidence": [],
                }
                packet = evaluator._reference_packet(question, case)
                unit = packet[
                    "range_applicable_linked_knowledge_units"
                ][0]
                self.assertEqual(unit["answer"], "")
                self.assertEqual(
                    [
                        item["source_id"]
                        for item in packet[
                            "range_scoped_contributing_source_answers"
                        ]
                    ],
                    expected["included"],
                )
                self.assertEqual(
                    [
                        item["source_id"]
                        for item in packet[
                            "excluded_other_range_source_answers"
                        ]
                    ],
                    expected["excluded"],
                )
                self.assertEqual(
                    packet["authoritative_original_expert_answer"]
                    is not None,
                    expected["original_available"],
                )
                self.assertTrue(packet["range_disambiguation_applied"])
                self.assertEqual(
                    evaluator.range_scope_requirement(packet)["status"],
                    "judge_required",
                )

    def test_semantic_packet_is_positive_only_and_evidence_blind(self) -> None:
        question = next(
            item
            for item in self.questions
            if item["source_id"] == "kim-la-capinera-17"
        )
        case = deepcopy(next(
            item
            for item in question["inference_runs"]
            if item["inference_input"]["measure_range"] == [51, 58]
        ))
        case["generated_answer"] = {
            "answer": "ff와 pp의 두 구간 대비를 살린다.",
            "evidence": [{"text": "검색 근거 비밀 문장"}],
        }
        packet = evaluator._reference_packet(question, case)
        user_message = evaluator.build_judge_messages(
            "claim_alignment",
            packet,
        )[1]["content"]
        self.assertIn("authoritative_reference_items", user_message)
        self.assertIn("candidate_answer_items", user_message)
        self.assertNotIn("빠르게 번갈아서 등장할까", user_message)
        self.assertNotIn("검색 근거 비밀 문장", user_message)
        self.assertNotIn(
            "excluded_other_range_source_answers",
            user_message,
        )

    def test_retrieved_expert_supplement_supports_factuality_only(
        self,
    ) -> None:
        packet = self.supplemental_packet(
            "《Die Forelle》는 여러 판본이 있어 전주가 있거나 없을 "
            "수 있고, 둘 다 가능하지만 일반적으로 전주가 있는 악보를 "
            "많이 사용한다. 전주 유무가 다르면 원본과 비교해 보는 "
            "것이 좋다."
        )
        validation = packet["supplemental_evidence_validation"]
        self.assertTrue(validation["auto_pass_eligible"])
        self.assertEqual(
            [item["evidence_id"] for item in validation["accepted_evidence"]],
            ["die-forelle-ku-001"],
        )
        supplements = evaluator._supplemental_factuality_items(packet)
        self.assertTrue(
            any("원본과 비교" in item["text"] for item in supplements)
        )
        semantic_packet = evaluator._semantic_reference_packet(packet)
        self.assertEqual(
            semantic_packet["authoritative_reference_items"],
            evaluator._authoritative_reference_items(packet),
        )
        self.assertEqual(
            semantic_packet["supplemental_factuality_items"],
            supplements,
        )

    def test_supplement_only_coverage_cannot_replace_required_r_claim(
        self,
    ) -> None:
        packet = self.supplemental_packet(
            "전주 유무가 다르면 원본과 비교해 보는 것이 좋다."
        )
        references = evaluator._all_semantic_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)
        frames = {}
        for frame in evaluator.FRAME_NAMES:
            assessment = evaluator.validate_judge_assessment(
                json.dumps(
                    assessment_payload(
                        frame,
                        reference_items=references,
                        candidate_items=candidates,
                        mode="supplement_only",
                        has_range=False,
                    ),
                    ensure_ascii=False,
                ),
                expected_frame=frame,
                has_measure_range=False,
                candidate_answer=packet["candidate_answer"],
                authoritative_reference_items=references,
                candidate_answer_items=candidates,
                allowed_context_texts=[packet["question"]],
                atomic_coverage_signals=atomic_coverage_signals(
                    packet
                ),
                atomic_coverage_nli_threshold=0.8,
            )
            computed = evaluator.compute_frame_result(assessment)
            self.assertFalse(computed["pass"])
            self.assertIn(
                "missing_authoritative_reference",
                computed["failure_reasons"],
            )
            frames[frame] = {
                "assessment": assessment,
                "computed": computed,
            }
        aggregate = evaluator.aggregate_judge_frames(
            frames,
            generated_answer={
                "generation_mode": "llm",
                "answer_basis": "retrieved_evidence",
                "evidence": [{"kind": "expert"}],
                "diagnostics": {"target_source_grounded": True},
            },
            supplemental_evidence_validation=packet[
                "supplemental_evidence_validation"
            ],
        )
        self.assertNotEqual(aggregate["answer_quality_status"], "pass")

    def test_supplemental_substitution_nli_blocks_false_r_coverage(
        self,
    ) -> None:
        packet = self.supplemental_packet(
            "악보의 원전도 함께 대조하는 편이 낫다."
        )
        references = evaluator._all_semantic_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)
        signals = atomic_coverage_signals(
            packet,
            overrides={
                "R002": {
                    "entailment": 0.001,
                    "neutral": 0.998,
                    "contradiction": 0.001,
                }
            },
        )
        payload = assessment_payload(
            "claim_alignment",
            reference_items=references,
            candidate_items=candidates,
            mode="equivalent",
            has_range=False,
        )
        for reference in payload["reference_assessments"]:
            if reference["reference_id"] == "R002":
                reference.update(
                    status="covered",
                    candidate_ids=["C001"],
                )
            else:
                reference.update(
                    status="not_required",
                    candidate_ids=[],
                )
        payload["candidate_assessments"][0].update(
            status="supported",
            reference_ids=["R002"],
        )
        validated = evaluator.validate_judge_assessment(
            json.dumps(payload, ensure_ascii=False),
            expected_frame="claim_alignment",
            has_measure_range=False,
            candidate_answer=packet["candidate_answer"],
            authoritative_reference_items=references,
            candidate_answer_items=candidates,
            allowed_context_texts=[packet["question"]],
            atomic_coverage_signals=signals,
            atomic_coverage_nli_threshold=0.8,
        )
        warnings = validated["atomic_claim_entailment_warnings"]
        self.assertEqual(
            [item["reference_id"] for item in warnings],
            ["R002"],
        )
        self.assertEqual(
            warnings[0]["guard_scope"],
            "supplemental_substitution_required_claim",
        )
        computed = evaluator.compute_frame_result(validated)
        self.assertFalse(computed["pass"])
        self.assertIn(
            "unverified_atomic_claim_entailment",
            computed["failure_reasons"],
        )

    def test_supplemental_evidence_is_authenticated_and_range_safe(
        self,
    ) -> None:
        packet = self.supplemental_packet("전주 판본을 비교한다.")
        record = self.expert_catalog["die-forelle-ku-001"]
        evidence = self.trusted_evidence(record, None)
        evidence["text"] = "위조된 전문가 답변"
        invalid = evaluator._validated_retrieved_expert_evidence(
            {
                "piece_id": "die-forelle",
                "measure_range": None,
                "evidence": [evidence],
            },
            piece_id="die-forelle",
            selected_measure_range=None,
            trusted_expert_catalog=self.expert_catalog,
        )
        self.assertFalse(invalid["auto_pass_eligible"])
        self.assertEqual(invalid["accepted_evidence"], [])

        whole_piece = evaluator._validated_retrieved_expert_evidence(
            {
                "piece_id": "die-forelle",
                "measure_range": [2, 27],
                "evidence": [
                    self.trusted_evidence(record, [2, 27])
                ],
            },
            piece_id="die-forelle",
            selected_measure_range=[2, 27],
            trusted_expert_catalog=self.expert_catalog,
        )
        self.assertEqual(
            [item["evidence_id"] for item in whole_piece["accepted_evidence"]],
            ["die-forelle-ku-001"],
        )

        specific = next(
            value
            for value in self.expert_catalog.values()
            if value["measure_status"] == "specific"
        )
        wrong_range = evaluator._validated_retrieved_expert_evidence(
            {
                "piece_id": specific["piece"],
                "measure_range": [999, 999],
                "evidence": [
                    self.trusted_evidence(specific, [999, 999])
                ],
            },
            piece_id=specific["piece"],
            selected_measure_range=[999, 999],
            trusted_expert_catalog=self.expert_catalog,
        )
        self.assertTrue(wrong_range["auto_pass_eligible"])
        self.assertEqual(wrong_range["accepted_evidence"], [])
        self.assertEqual(
            [
                item["evidence_id"]
                for item in wrong_range["accepted_secondary_context"]
            ],
            [specific["id"]],
        )

    def test_confirmed_no_range_local_example_is_measure_bound_support(
        self,
    ) -> None:
        record = self.expert_catalog["la-capinera-ku-021"]
        evidence = self.trusted_evidence(record, None)
        validation = evaluator._validated_retrieved_expert_evidence(
            {
                "piece_id": "la-capinera",
                "measure_range": None,
                "evidence": [evidence],
            },
            piece_id="la-capinera",
            selected_measure_range=None,
            trusted_expert_catalog=self.expert_catalog,
        )

        self.assertTrue(validation["auto_pass_eligible"])
        self.assertEqual(validation["rejected_evidence"], [])
        accepted = validation["accepted_evidence"][0]
        self.assertEqual(accepted["scope_match"], "local_example")
        self.assertEqual(accepted["generation_role"], "local_example")
        self.assertEqual(
            accepted["measure_ranges"],
            record["measure_range"],
        )
        self.assertFalse(accepted["whole_piece_claim_authority"])

        packet = {
            "piece_id": "la-capinera",
            "question": "이 노래를 부를 때 무엇에 유의해야 할까?",
            "selected_measure_range": None,
            "required_claim_scope": "test",
            "candidate_answer": "",
            "retrieved_expert_factuality_evidence": (
                validation["accepted_evidence"]
            ),
        }
        supplements = evaluator._supplemental_factuality_items(packet)
        self.assertTrue(supplements)
        self.assertTrue(
            all(
                item["generation_role"] == "local_example"
                and item["applicable_measure_ranges"]
                == record["measure_range"]
                and not item["whole_piece_claim_authority"]
                for item in supplements
            )
        )
        packet_with_same_primary = {
            **packet,
            "range_applicable_linked_knowledge_units": [
                {
                    "answer": record["answer"],
                    "contributing_source_answers": [],
                }
            ],
        }
        self.assertTrue(
            any(
                item["text"] in record["answer"]
                and item["generation_role"] == "local_example"
                for item in evaluator._supplemental_factuality_items(
                    packet_with_same_primary
                )
            )
        )
        judge_system = evaluator.build_judge_messages(
            "claim_alignment",
            packet,
        )[0]["content"]
        self.assertIn(
            "supports only a candidate claim explicitly limited",
            judge_system,
        )
        self.assertIn('"곡 전반", "자주", or "많이"', judge_system)

        forged = deepcopy(evidence)
        forged["measure_ranges"] = [[1, 1]]
        rejected = evaluator._validated_retrieved_expert_evidence(
            {
                "piece_id": "la-capinera",
                "measure_range": None,
                "evidence": [forged],
            },
            piece_id="la-capinera",
            selected_measure_range=None,
            trusted_expert_catalog=self.expert_catalog,
        )
        self.assertFalse(rejected["auto_pass_eligible"])
        self.assertIn(
            "measure_ranges",
            rejected["rejected_evidence"][0]["reason"],
        )

    def test_supplemental_authentication_requires_production_fields(
        self,
    ) -> None:
        record = self.expert_catalog["in-flowery-clouds-ku-008"]
        selected_range = [39, 49]
        evidence = self.trusted_evidence(record, selected_range)
        generated = {
            "piece_id": record["piece"],
            "measure_range": selected_range,
            "evidence": [evidence],
        }

        for missing_field, expected_reason in (
            ("piece_id", "generated_answer_piece_missing"),
            (
                "measure_range",
                "generated_answer_measure_range_missing",
            ),
        ):
            with self.subTest(missing_field=missing_field):
                incomplete = deepcopy(generated)
                incomplete.pop(missing_field)
                validation = (
                    evaluator._validated_retrieved_expert_evidence(
                        incomplete,
                        piece_id=record["piece"],
                        selected_measure_range=selected_range,
                        trusted_expert_catalog=self.expert_catalog,
                    )
                )
                self.assertFalse(validation["auto_pass_eligible"])
                self.assertEqual(
                    validation["rejected_evidence"][0]["reason"],
                    expected_reason,
                )

        for authority_mutation in ("missing", "mismatched"):
            with self.subTest(authority_mutation=authority_mutation):
                invalid_evidence = deepcopy(evidence)
                if authority_mutation == "missing":
                    invalid_evidence.pop("generation_expert_authority")
                else:
                    invalid_evidence[
                        "generation_expert_authority"
                    ]["answer_texts"] = ["위조된 프롬프트 근거"]
                validation = (
                    evaluator._validated_retrieved_expert_evidence(
                        {
                            **generated,
                            "evidence": [invalid_evidence],
                        },
                        piece_id=record["piece"],
                        selected_measure_range=selected_range,
                        trusted_expert_catalog=self.expert_catalog,
                    )
                )
                self.assertFalse(validation["auto_pass_eligible"])
                self.assertEqual(validation["accepted_evidence"], [])
                self.assertEqual(
                    validation["rejected_evidence"][0]["reason"],
                    "generation_expert_authority_mismatch",
                )

    def test_authenticated_review_disclosure_is_not_semantic_content(
        self,
    ) -> None:
        body = (
            "박자와 반주 리듬의 변화가 분위기를 전환한다. "
            "조성 변화 시점은 단정하기 어렵다."
        )
        _, disclosure = self.warned_supplemental_packet("")
        packet, _ = self.warned_supplemental_packet(
            disclosure + "\n\n" + body
        )
        audit = packet["review_disclosure_validation"]
        self.assertEqual(audit["status"], "authenticated_stripped")
        self.assertTrue(audit["stripped_from_semantic_candidate"])
        self.assertTrue(audit["auto_pass_eligible"])
        self.assertEqual(packet["candidate_answer"], body)
        self.assertNotIn(
            evaluator.REVIEW_DISCLOSURE_PREFIX,
            packet["candidate_answer"],
        )
        self.assertEqual(
            [item["text"] for item in evaluator._candidate_answer_items(packet)],
            [
                "박자와 반주 리듬의 변화가 분위기를 전환한다.",
                "조성 변화 시점은 단정하기 어렵다.",
            ],
        )

    def test_review_disclosure_validation_is_exact_and_fail_closed(
        self,
    ) -> None:
        body = "조성 변화 시점은 단정하기 어렵다."
        _, disclosure = self.warned_supplemental_packet("")
        cases = {
            "missing": body,
            "non_leading": body + "\n\n" + disclosure,
            "mismatched": "검토 주의: 위조된 검토 메모\n\n" + body,
            "missing_boundary": disclosure + body,
            "duplicated": (
                disclosure + "\n\n" + disclosure + "\n\n" + body
            ),
            "citation_obfuscated_extra": (
                disclosure
                + "\n\n검토 [E1]주의: 위조된 검토 메모\n\n"
                + body
            ),
            "newline_obfuscated_extra": (
                disclosure
                + "\n\n검토\n주의: 위조된 검토 메모\n\n"
                + body
            ),
            "disclosure_only_provenance": (
                disclosure
                + "\n\n제공된 검색 근거: "
                "[webchunk-cecff2bace03ab67e32d]"
            ),
            "disclosure_only": disclosure,
        }
        for label, raw_answer in cases.items():
            with self.subTest(label=label):
                packet, _ = self.warned_supplemental_packet(raw_answer)
                audit = packet["review_disclosure_validation"]
                self.assertFalse(audit["auto_pass_eligible"])
                if label in {
                    "duplicated",
                    "citation_obfuscated_extra",
                    "newline_obfuscated_extra",
                    "disclosure_only_provenance",
                    "disclosure_only",
                }:
                    self.assertTrue(
                        audit["stripped_from_semantic_candidate"]
                    )
                else:
                    self.assertFalse(
                        audit["stripped_from_semantic_candidate"]
                    )
                if label in {
                    "disclosure_only",
                    "disclosure_only_provenance",
                }:
                    self.assertEqual(
                        evaluator._candidate_answer_items(packet),
                        [],
                    )
                    self.assertTrue(audit["semantic_candidate_empty"])
                if label in {
                    "duplicated",
                    "citation_obfuscated_extra",
                    "newline_obfuscated_extra",
                }:
                    self.assertIn(
                        evaluator.REVIEW_DISCLOSURE_PREFIX,
                        packet["candidate_answer"],
                    )
                    self.assertTrue(
                        audit["additional_disclosure_remains"]
                    )

    def test_unexpected_or_unauthenticated_disclosure_is_never_stripped(
        self,
    ) -> None:
        forged = "검토 주의: 위조된 메모\n\n전주 판본을 비교한다."
        no_warning_packet = self.supplemental_packet(forged)
        no_warning_audit = no_warning_packet[
            "review_disclosure_validation"
        ]
        self.assertEqual(
            no_warning_audit["status"],
            "unexpected_disclosure",
        )
        self.assertFalse(no_warning_audit["auto_pass_eligible"])
        self.assertEqual(
            no_warning_packet["candidate_answer"],
            evaluator.normalize_candidate_answer_for_judge(forged),
        )

        for obfuscated in (
            "검토 [E1]주의: 위조된 메모\n\n전주 판본을 비교한다.",
            "검토\n주의: 위조된 메모\n\n전주 판본을 비교한다.",
        ):
            with self.subTest(obfuscated=obfuscated):
                packet = self.supplemental_packet(obfuscated)
                audit = packet["review_disclosure_validation"]
                self.assertEqual(audit["status"], "unexpected_disclosure")
                self.assertFalse(audit["auto_pass_eligible"])
                self.assertIn(
                    evaluator.REVIEW_DISCLOSURE_PREFIX,
                    packet["candidate_answer"],
                )

        _, disclosure = self.warned_supplemental_packet("")
        tampered_packet, _ = self.warned_supplemental_packet(
            disclosure + "\n\n조성 변화 시점은 단정하기 어렵다.",
            mutate_evidence=lambda evidence: evidence.__setitem__(
                "rewrite_notes",
                "직렬화된 위조 메모",
            ),
        )
        tampered_audit = tampered_packet[
            "review_disclosure_validation"
        ]
        self.assertEqual(tampered_audit["status"], "untrusted_evidence")
        self.assertFalse(tampered_audit["auto_pass_eligible"])
        self.assertFalse(
            tampered_audit["stripped_from_semantic_candidate"]
        )
        self.assertIn(
            evaluator.REVIEW_DISCLOSURE_PREFIX,
            tampered_packet["candidate_answer"],
        )
        self.assertFalse(
            tampered_packet["supplemental_evidence_validation"][
                "auto_pass_eligible"
            ]
        )

    def test_invalid_review_disclosure_blocks_rag_llm_auto_pass(
        self,
    ) -> None:
        audit = {
            "status": "invalid",
            "expected_text": "검토 주의: 신뢰된 메모",
            "evidence_ids": ["unit"],
            "stripped_from_semantic_candidate": False,
            "auto_pass_eligible": False,
            "failure_reason": "trusted_review_disclosure_missing",
            "semantic_candidate_empty": False,
            "additional_disclosure_remains": False,
        }
        aggregate = evaluator.aggregate_judge_frames(
            frame_states(),
            generated_answer=grounded_answer(),
            review_disclosure_validation=audit,
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertEqual(
            aggregate["answer_quality_status"],
            "human_review",
        )
        self.assertFalse(aggregate["pipeline_is_grounded_rag_llm"])
        self.assertIn(
            "review_disclosure_authentication_requires_review",
            aggregate["disagreement_reasons"],
        )


class SynthesizedQuestionInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions, cls.paths = evaluator.load_evaluation_questions(
            SYNTHESIZED_DATASET_ROOT,
            question_set="synthesized",
        )

    def test_synthesized_set_keeps_40_records_and_expands_153_cases(
        self,
    ) -> None:
        self.assertEqual(len(self.questions), 40)
        cases = [
            case
            for question in self.questions
            for case in question["inference_runs"]
        ]
        self.assertEqual(
            len(cases),
            evaluator.EXPECTED_SYNTHESIZED_CASE_COUNT,
        )
        self.assertEqual(len({case["case_id"] for case in cases}), 153)
        self.assertEqual(
            [path.parent.name for path in self.paths].count(
                "evaluation_question_variants"
            ),
            3,
        )
        for question in self.questions:
            variants = question["synthesized_question_variants"]
            self.assertEqual(len(variants), 3)
            by_id = {item["variant_id"]: item for item in variants}
            self.assertEqual(
                {
                    case["inference_input"]["synthesized_variant_id"]
                    for case in question["inference_runs"]
                },
                set(by_id),
            )
            for case in question["inference_runs"]:
                variant_id = case["inference_input"][
                    "synthesized_variant_id"
                ]
                self.assertTrue(case["case_id"].startswith(variant_id + "__"))
                self.assertEqual(
                    case["inference_input"]["question"],
                    by_id[variant_id]["question"],
                )

    def test_variant_shards_are_bound_and_form_checked(self) -> None:
        real_load = evaluator.load_json

        def run_mutation(mutator, expected_error: str) -> None:
            def mutating_load(path):
                payload = real_load(path)
                if (
                    path.parent.name == "evaluation_question_variants"
                    and path.name == "die-forelle.json"
                ):
                    payload = deepcopy(payload)
                    mutator(payload)
                return payload

            with patch.object(
                evaluator,
                "load_json",
                side_effect=mutating_load,
            ):
                with self.assertRaisesRegex(
                    evaluator.EvaluationInputError,
                    expected_error,
                ):
                    evaluator.load_evaluation_questions(
                        SYNTHESIZED_DATASET_ROOT,
                        question_set="synthesized",
                    )

        cases = (
            (
                lambda value: value["questions"][0].__setitem__(
                    "base_paraphrased_question",
                    "바뀐 기준 질문?",
                ),
                "base paraphrased question drift",
            ),
            (
                lambda value: value["questions"][0]["variants"][0]
                .__setitem__("variant_id", "unstable-id"),
                "expected stable ID",
            ),
            (
                lambda value: value["questions"][0]["variants"][0]
                .__setitem__("question", "2마디에서는 어떻게 해야 할까?"),
                "measure locator",
            ),
            (
                lambda value: value["questions"][0]["variants"][0]
                .__setitem__("question", "이 경우에는 어떻게 해야 하나요?"),
                "non-honorific",
            ),
            (
                lambda value: value["questions"][0]["variants"][1]
                .__setitem__(
                    "question",
                    value["questions"][0]["variants"][0]["question"],
                ),
                "duplicate synthesized question",
            ),
        )
        for mutator, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                run_mutation(mutator, expected_error)

    def test_snapshot_records_benchmark_and_pipeline_provenance(self) -> None:
        snapshot = evaluator.new_snapshot(
            questions=deepcopy(self.questions),
            dataset_root=SYNTHESIZED_DATASET_ROOT,
            input_files=[],
            input_fingerprint="synthesized-test",
            top_k=6,
            generator_model={"path": "generator"},
            judge_model={"path": "judge"},
            range_guard=TEST_RANGE_GUARD,
            question_set="synthesized",
            system_root=DENSE_SYSTEM_ROOT,
            benchmark_dataset_root=SYNTHESIZED_DATASET_ROOT,
            pipeline_dataset_root=DATASET_ROOT,
        )
        run = snapshot["run"]
        self.assertEqual(run["question_set"], "synthesized")
        self.assertEqual(run["system_root"], str(DENSE_SYSTEM_ROOT))
        self.assertEqual(
            run["benchmark_dataset_root"],
            str(SYNTHESIZED_DATASET_ROOT),
        )
        self.assertEqual(run["pipeline_dataset_root"], str(DATASET_ROOT))


class RangeScopeNliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        questions, _ = evaluator.load_evaluation_questions(DATASET_ROOT)
        cls.question = next(
            item
            for item in questions
            if item["source_id"] == "kim-la-capinera-17"
        )
        cls.case_template = next(
            item
            for item in cls.question["inference_runs"]
            if item["inference_input"]["measure_range"] == [51, 58]
        )

    def packet(self, answer: str) -> dict:
        case = deepcopy(self.case_template)
        case["generated_answer"] = {"answer": answer, "evidence": []}
        return evaluator._reference_packet(self.question, case)

    def test_reviewed_range_contrast_claims_override_shared_effect_text(
        self,
    ) -> None:
        packet = self.packet(
            "ff와 pp를 두 덩어리로 대비해 메아리를 표현한다."
        )
        excluded = evaluator._excluded_reference_items(packet)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(
            excluded[0]["text"],
            "ff와 pp가 빠르게 번갈아 등장한다.",
        )
        self.assertEqual(
            evaluator._allowed_range_contrast_items(packet),
            [
                {
                    "reference_id": "A001",
                    "text": "ff와 pp가 두 블록으로 나뉘어 대비된다.",
                }
            ],
        )
        system_prompt = evaluator.build_range_judge_messages(packet)[0][
            "content"
        ]
        self.assertIn("human-reviewed atomic", system_prompt)
        self.assertIn("Assertion does not require", system_prompt)

    @staticmethod
    def raw_assessment(
        packet: dict,
        relation: str,
        *,
        confidence: float = 0.95,
    ) -> str:
        candidate_ids = [
            item["candidate_id"]
            for item in evaluator._candidate_answer_items(packet)
        ]
        excluded_ids = [
            item["reference_id"]
            for item in evaluator._excluded_reference_items(packet)
        ]
        values = []
        for index, excluded_id in enumerate(excluded_ids):
            active = index == 0 and relation != "absent"
            values.append(
                {
                    "excluded_id": excluded_id,
                    "relation": relation if index == 0 else "absent",
                    "candidate_ids": candidate_ids if active else [],
                }
            )
        return json.dumps(
            {
                "frame": "range_scope",
                "excluded_claim_assessments": values,
                "confidence": confidence,
                "rationale": "다른 범위 주장의 긍정·부정 양상을 판별한다.",
            },
            ensure_ascii=False,
        )

    def test_single_apostrophe_escape_repair_is_audited_for_range_frame(
        self,
    ) -> None:
        packet = self.packet(
            "ff와 pp를 두 덩어리로 대비해 메아리를 표현한다."
        )
        payload = json.loads(self.raw_assessment(packet, "absent"))
        payload["rationale"] = "표기 'vuò(vuo')'를 언급한다."
        valid_raw = json.dumps(payload, ensure_ascii=False)
        malformed_raw = valid_raw.replace(
            "vuo')",
            "vuo\\')",
            1,
        )
        frame_state = {"status": "pending", "attempts": []}

        completed = evaluator._run_one_range_frame(
            packet=packet,
            judge_fn=lambda _messages: malformed_raw,
            range_signal_fn=hybrid_range_signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
            frame_state=frame_state,
            checkpoint=lambda: None,
            max_attempts=1,
        )

        self.assertTrue(completed)
        self.assertEqual(
            frame_state["assessment"]["rationale"],
            payload["rationale"],
        )
        self.assertEqual(
            frame_state["assessment"]["deterministic_normalizations"],
            [evaluator.JSON_APOSTROPHE_ESCAPE_NORMALIZATION],
        )
        self.assertEqual(len(frame_state["attempts"]), 1)
        attempt = frame_state["attempts"][0]
        self.assertEqual(
            attempt["raw_response"],
            malformed_raw,
        )
        self.assertEqual(
            attempt["parser_normalizations"],
            [
                evaluator.JSON_APOSTROPHE_ESCAPE_NORMALIZATION
            ],
        )
        self.assertNotIn("error", attempt)

    def test_asserted_negated_uncertain_and_double_negation_fixtures(
        self,
    ) -> None:
        fixtures = (
            (
                "ff와 pp가 빠르게 번갈아 나온다.",
                "asserted",
                "fail",
                True,
            ),
            (
                "ff와 pp가 빠르게 번갈아 나오는 것은 아니다.",
                "negated",
                "pass",
                False,
            ),
            (
                "ff와 pp가 빠르게 번갈아 나오는지는 확실하지 않다.",
                "uncertain",
                "human_review",
                False,
            ),
            (
                "ff와 pp가 빠르게 번갈아 나오지 않는 것은 아니다.",
                "uncertain",
                "human_review",
                False,
            ),
        )
        for answer, relation, status, wrong_range in fixtures:
            with self.subTest(answer=answer, relation=relation):
                packet = self.packet(answer)
                validated = evaluator.validate_range_judge_assessment(
                    self.raw_assessment(packet, relation),
                    packet=packet,
                    hybrid_signals=hybrid_range_signals(packet),
                    nli_threshold=0.8,
                    embedding_delta_threshold=0.05,
                )
                self.assertEqual(validated["status"], status)
                self.assertEqual(
                    validated["wrong_measure_application"],
                    wrong_range,
                )
                if "않는 것은 아니다" in answer:
                    self.assertNotEqual(validated["status"], "pass")
                self.assertEqual(
                    validated["excluded_claim_assessments"][0][
                        "candidate_texts"
                    ],
                    [answer],
                )

    def test_range_scope_requires_every_x_id_exactly_once(self) -> None:
        packet = self.packet("ff와 pp가 빠르게 번갈아 나온다.")
        raw = json.loads(self.raw_assessment(packet, "asserted"))
        malformed = []

        missing = deepcopy(raw)
        missing["excluded_claim_assessments"].pop()
        malformed.append(missing)

        duplicate = deepcopy(raw)
        duplicate["excluded_claim_assessments"].append(
            deepcopy(duplicate["excluded_claim_assessments"][0])
        )

        unknown_candidate = deepcopy(raw)
        unknown_candidate["excluded_claim_assessments"][0][
            "candidate_ids"
        ] = ["C999"]
        malformed.append(unknown_candidate)

        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(evaluator.JudgeOutputError):
                    evaluator.validate_range_judge_assessment(
                        json.dumps(value, ensure_ascii=False),
                        packet=packet,
                        hybrid_signals=hybrid_range_signals(packet),
                        nli_threshold=0.8,
                        embedding_delta_threshold=0.05,
                    )

        deduplicated = evaluator.validate_range_judge_assessment(
            json.dumps(duplicate, ensure_ascii=False),
            packet=packet,
            hybrid_signals=hybrid_range_signals(packet),
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
        )
        self.assertEqual(deduplicated["status"], "fail")
        self.assertIn(
            "X001:exact_duplicate_observations_deduplicated",
            deduplicated["deterministic_normalizations"],
        )

        absent_with_link = deepcopy(raw)
        absent_with_link["excluded_claim_assessments"][0][
            "relation"
        ] = "absent"
        with self.assertRaises(evaluator.JudgeOutputError):
            evaluator.validate_range_judge_assessment(
                json.dumps(absent_with_link, ensure_ascii=False),
                packet=packet,
                hybrid_signals=hybrid_range_signals(packet),
                nli_threshold=0.8,
                embedding_delta_threshold=0.05,
            )

    def test_conflicting_duplicate_x_uses_clause_nli_but_stays_review(
        self,
    ) -> None:
        packet = self.packet(
            "ff와 pp가 빠르게 번갈아 나오는 대목은 아니다. "
            "ff와 pp는 두 블록으로 대비해 메아리를 표현한다."
        )
        raw = {
            "frame": evaluator.RANGE_FRAME_NAME,
            "excluded_claim_assessments": [
                {
                    "excluded_id": "X001",
                    "relation": "negated",
                    "candidate_ids": ["C001"],
                },
                {
                    "excluded_id": "X001",
                    "relation": "asserted",
                    "candidate_ids": ["C002"],
                },
            ],
            "confidence": 1.0,
            "rationale": "첫 문장은 부정하고 둘째 문장은 다른 구조를 설명한다.",
        }
        signals = hybrid_range_signals(
            packet,
            entailment=0.0004,
            contradiction=0.9992,
            candidate_nli={
                "C001": {
                    "entailment": 0.0005,
                    "neutral": 0.0003,
                    "contradiction": 0.9992,
                },
                "C002": {
                    "entailment": 0.0006,
                    "neutral": 0.9990,
                    "contradiction": 0.0004,
                },
            },
        )

        validated = evaluator.validate_range_judge_assessment(
            json.dumps(raw, ensure_ascii=False),
            packet=packet,
            hybrid_signals=signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
        )
        item = validated["excluded_claim_assessments"][0]
        self.assertEqual(item["relation"], "negated")
        self.assertEqual(item["candidate_ids"], ["C001"])
        self.assertTrue(item["llm_duplicate_conflict"])
        self.assertEqual(validated["status"], "human_review")
        self.assertIn(
            "X001:conflicting_duplicate_observations:"
            "clause_nli_contradiction",
            validated["deterministic_normalizations"],
        )

    def test_clause_entailment_dominates_conflicting_duplicate_x(self) -> None:
        packet = self.packet(
            "ff와 pp가 빠르게 번갈아 나오는 대목은 아니다. "
            "하지만 실제로는 빠르게 번갈아 등장한다."
        )
        raw = {
            "frame": evaluator.RANGE_FRAME_NAME,
            "excluded_claim_assessments": [
                {
                    "excluded_id": "X001",
                    "relation": "negated",
                    "candidate_ids": ["C001"],
                },
                {
                    "excluded_id": "X001",
                    "relation": "asserted",
                    "candidate_ids": ["C002"],
                },
            ],
            "confidence": 1.0,
            "rationale": "두 절이 서로 충돌한다.",
        }
        signals = hybrid_range_signals(
            packet,
            entailment=0.95,
            contradiction=0.01,
            candidate_nli={
                "C001": {
                    "entailment": 0.0005,
                    "neutral": 0.0003,
                    "contradiction": 0.9992,
                },
                "C002": {
                    "entailment": 0.9990,
                    "neutral": 0.0006,
                    "contradiction": 0.0004,
                },
            },
        )
        validated = evaluator.validate_range_judge_assessment(
            json.dumps(raw, ensure_ascii=False),
            packet=packet,
            hybrid_signals=signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
        )
        item = validated["excluded_claim_assessments"][0]
        self.assertEqual(item["relation"], "asserted")
        self.assertEqual(item["candidate_ids"], ["C002"])
        self.assertEqual(validated["status"], "fail")
        self.assertTrue(validated["wrong_measure_application"])

    def test_range_prompt_is_a_separate_semantic_nli_contract(self) -> None:
        packet = self.packet(
            "ff와 pp가 빠르게 번갈아 나오지 않는 것은 아니다."
        )
        system = evaluator.build_range_judge_messages(packet)[0]["content"]
        self.assertIn("compare the C text directly", system)
        self.assertIn("ordinary negation", system)
        self.assertIn("double negation", system)
        self.assertIn("asserted|negated|uncertain|absent", system)

    def test_hybrid_guard_decision_table_preserves_all_three_signals(
        self,
    ) -> None:
        packet = self.packet("강약을 두 블록으로 대비한다.")
        fixtures = (
            (
                "absent",
                hybrid_range_signals(packet, entailment=0.85),
                "asserted",
                "nli_entailment_at_or_above_threshold",
            ),
            (
                "absent",
                hybrid_range_signals(packet, embedding_delta=0.06),
                "uncertain",
                "embedding_contrast_risk_at_or_above_threshold",
            ),
            (
                "negated",
                hybrid_range_signals(
                    packet,
                    contradiction=0.85,
                ),
                "negated",
                "nli_contradiction_at_or_above_threshold",
            ),
            (
                "uncertain",
                hybrid_range_signals(packet, embedding_delta=-0.2),
                "uncertain",
                "llm_uncertain",
            ),
        )
        for raw_relation, signals, expected, reason in fixtures:
            with self.subTest(
                raw_relation=raw_relation,
                expected=expected,
            ):
                assessment = evaluator.validate_range_judge_assessment(
                    self.raw_assessment(packet, raw_relation),
                    packet=packet,
                    hybrid_signals=signals,
                    nli_threshold=0.8,
                    embedding_delta_threshold=0.05,
                )
                claim = assessment["excluded_claim_assessments"][0]
                self.assertEqual(claim["llm_relation"], raw_relation)
                self.assertEqual(claim["relation"], expected)
                self.assertIn(reason, claim["decision_reasons"])
                self.assertEqual(
                    claim["model_scores"],
                    {
                        "embedding": signals[
                            "excluded_claim_signals"
                        ][0]["embedding"],
                        "nli": signals["excluded_claim_signals"][0]["nli"],
                        "candidate_nli": signals[
                            "excluded_claim_signals"
                        ][0]["candidate_nli"],
                    },
                )

    def test_unlocalized_hybrid_negation_can_never_reconcile_to_pass(
        self,
    ) -> None:
        packet = self.packet(
            "빠른 교대는 아니라고 본다. 매일 30분 훈련해야 한다."
        )
        raw = json.loads(self.raw_assessment(packet, "absent"))
        signals = hybrid_range_signals(
            packet,
            contradiction=0.9,
            embedding_delta=-0.2,
        )
        range_assessment = evaluator.validate_range_judge_assessment(
            json.dumps(raw, ensure_ascii=False),
            packet=packet,
            hybrid_signals=signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
        )
        claim = range_assessment["excluded_claim_assessments"][0]
        self.assertEqual(claim["relation"], "negated")
        self.assertEqual(range_assessment["status"], "human_review")
        self.assertEqual(
            claim["candidate_link_provenance"],
            "synthesized_full_answer_unlocalized",
        )

        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(frame)
            payload["candidate_assessments"][0].update(
                status="unsupported",
                reference_ids=[],
            )
            payload["relationship"] = "materially_unreliable"
            payload["scores"]["factual_safety"] = 1
            payload["critical_error_types"] = [
                "wrong_measure_application"
            ]
            validated = validate_payload(payload)
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=range_assessment,
        )
        self.assertNotEqual(aggregate["status"], "pass")
        self.assertNotIn(
            "uncertain_other_range_mentions_reconciled_for_review",
            aggregate["disagreement_reasons"],
        )

    def test_clause_nli_localizes_clear_negation_to_exact_candidate(
        self,
    ) -> None:
        packet = self.packet(
            "ff와 pp가 빠르게 번갈아 나오는 대목은 아니다. "
            "두 강약은 두 블록으로 대비된다."
        )
        signals = hybrid_range_signals(
            packet,
            contradiction=0.9,
            embedding_delta=-0.2,
            candidate_nli={
                "C001": {
                    "entailment": 0.01,
                    "neutral": 0.04,
                    "contradiction": 0.95,
                },
                "C002": {
                    "entailment": 0.01,
                    "neutral": 0.98,
                    "contradiction": 0.01,
                },
            },
        )
        assessment = evaluator.validate_range_judge_assessment(
            self.raw_assessment(packet, "absent"),
            packet=packet,
            hybrid_signals=signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
        )
        claim = assessment["excluded_claim_assessments"][0]
        self.assertEqual(claim["relation"], "negated")
        self.assertEqual(claim["candidate_ids"], ["C001"])
        self.assertEqual(
            claim["candidate_link_provenance"],
            "clause_nli_localized",
        )
        self.assertEqual(assessment["status"], "pass")

    def test_clause_entailment_overrides_full_answer_negation(self) -> None:
        packet = self.packet(
            "ff와 pp가 빠르게 번갈아 나온다. "
            "하지만 그렇게 들리지 않을 수도 있다."
        )
        signals = hybrid_range_signals(
            packet,
            contradiction=0.9,
            embedding_delta=-0.2,
            candidate_nli={
                "C001": {
                    "entailment": 0.95,
                    "neutral": 0.04,
                    "contradiction": 0.01,
                },
                "C002": {
                    "entailment": 0.01,
                    "neutral": 0.98,
                    "contradiction": 0.01,
                },
            },
        )
        assessment = evaluator.validate_range_judge_assessment(
            self.raw_assessment(packet, "negated"),
            packet=packet,
            hybrid_signals=signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
        )
        claim = assessment["excluded_claim_assessments"][0]
        self.assertEqual(claim["relation"], "asserted")
        self.assertEqual(claim["candidate_ids"], ["C001"])
        self.assertEqual(
            claim["candidate_link_provenance"],
            "clause_nli_localized",
        )
        self.assertEqual(assessment["status"], "fail")


class RuntimeSelectionTests(unittest.TestCase):
    def tearDown(self) -> None:
        evaluator._activate_system_root(evaluator.PROJECT_ROOT)

    def test_cli_defaults_output_from_question_set(self) -> None:
        canonical = evaluator.parse_arguments([])
        synthesized = evaluator.parse_arguments(
            ["--question-set", "synthesized"]
        )
        self.assertEqual(canonical.output, evaluator.DEFAULT_OUTPUT)
        self.assertEqual(
            synthesized.output,
            evaluator.DEFAULT_SYNTHESIZED_OUTPUT,
        )

    def test_system_root_is_activated_before_settings_import(self) -> None:
        arguments = evaluator.parse_arguments(
            [
                "--system-root",
                str(DENSE_SYSTEM_ROOT),
                "--question-set",
                "synthesized",
                "--dataset-root",
                str(SYNTHESIZED_DATASET_ROOT),
                "--pipeline-dataset-root",
                str(DATASET_ROOT),
            ]
        )
        self.assertEqual(arguments.system_root, DENSE_SYSTEM_ROOT)
        self.assertEqual(arguments.dataset_root, SYNTHESIZED_DATASET_ROOT)
        self.assertEqual(arguments.pipeline_dataset_root, DATASET_ROOT)
        self.assertEqual(Path(sys.path[0]), DENSE_SYSTEM_ROOT)
        settings_module = evaluator._import_system_module(
            "soprano_qa.settings",
            DENSE_SYSTEM_ROOT,
        )
        self.assertTrue(
            Path(settings_module.__file__).resolve().is_relative_to(
                DENSE_SYSTEM_ROOT
            )
        )

    def test_dense_system_files_and_checkpoint_are_fingerprinted(self) -> None:
        settings = evaluator._load_runtime_settings(DENSE_SYSTEM_ROOT)
        paths = {
            path.resolve()
            for path in evaluator._system_fingerprint_paths(
                DENSE_SYSTEM_ROOT,
                settings,
            )
        }
        self.assertIn(
            DENSE_SYSTEM_ROOT / "soprano_qa" / "dense.py",
            paths,
        )
        self.assertIn(
            DENSE_SYSTEM_ROOT / "config" / "settings.json",
            paths,
        )
        self.assertIn(Path(settings["corpus_path"]).resolve(), paths)
        self.assertIn(Path(settings["stats_path"]).resolve(), paths)
        self.assertIn(
            Path(settings["embedding_model_path"]).resolve(),
            paths,
        )
        self.assertIn(
            Path(settings["embedding_cache_path"]).resolve(),
            paths,
        )
        self.assertIn(DENSE_SYSTEM_ROOT / "requirements.txt", paths)

    def test_full_evaluator_refuses_a_corpus_rebuild_after_fingerprinting(
        self,
    ) -> None:
        settings = {"dataset_root": str(DATASET_ROOT)}
        self.assertEqual(
            evaluator.validate_pipeline_dataset_root(settings, DATASET_ROOT),
            DATASET_ROOT,
        )
        with self.assertRaisesRegex(
            evaluator.EvaluationInputError,
            "will not rebuild authenticated inputs",
        ):
            evaluator.validate_pipeline_dataset_root(
                settings,
                SYNTHESIZED_DATASET_ROOT,
            )

    def test_full_evaluator_requires_a_current_authenticated_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.json"
            stats = root / "stats.json"
            corpus.write_text("[]\n", encoding="utf-8")
            settings = {
                "dataset_root": str(root),
                "corpus_path": str(corpus),
                "stats_path": str(stats),
                "web_export_files": ["research-open.jsonl"],
            }
            stats_payload = {
                "corpus_schema_version": 6,
                "dataset_root": str(root.resolve()),
                "input_fingerprint": "current-inputs",
                "corpus_sha256": evaluator.file_sha256(corpus),
                "web_export_files": ["research-open.jsonl"],
            }
            stats.write_text(
                json.dumps(stats_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            validated = evaluator.validate_prebuilt_corpus(
                settings,
                corpus_input_fingerprint_fn=lambda _: "current-inputs",
            )
            self.assertEqual(validated, stats_payload)

            corpus.write_text("[{}]\n", encoding="utf-8")
            with self.assertRaisesRegex(
                evaluator.EvaluationInputError,
                "corpus_sha256",
            ):
                evaluator.validate_prebuilt_corpus(
                    settings,
                    corpus_input_fingerprint_fn=lambda _: "current-inputs",
                )

    def test_full_evaluator_reauthenticates_embedding_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.json"
            stats = root / "stats.json"
            embedding_model = root / "embedding.gguf"
            embedding_cache = root / "embedding-cache.json"
            corpus.write_text("[]\n", encoding="utf-8")
            embedding_model.write_bytes(b"model-v1")
            embedding_cache.write_text("{}\n", encoding="utf-8")
            settings = {
                "dataset_root": str(root),
                "corpus_path": str(corpus),
                "stats_path": str(stats),
                "embedding_model_path": str(embedding_model),
                "embedding_cache_path": str(embedding_cache),
                "web_export_files": ["research-open.jsonl"],
            }
            stats.write_text(
                json.dumps(
                    {
                        "corpus_schema_version": 6,
                        "dataset_root": str(root.resolve()),
                        "input_fingerprint": "current-inputs",
                        "corpus_sha256": evaluator.file_sha256(corpus),
                        "web_export_files": ["research-open.jsonl"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expected = evaluator.authenticate_pipeline_state(
                settings,
                corpus_input_fingerprint_fn=lambda _: "current-inputs",
            )
            embedding_cache.write_text('{"changed": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                evaluator.EvaluationInputError,
                "embedding_cache",
            ):
                evaluator.reauthenticate_pipeline_state(
                    settings,
                    corpus_input_fingerprint_fn=lambda _: "current-inputs",
                    expected_state=expected,
                )


class PipelinePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        questions, _ = evaluator.load_evaluation_questions(DATASET_ROOT)
        cls.range_question = next(
            item
            for item in questions
            if item["source_id"] == "kim-die-forelle-02"
        )

    def make_snapshot(self) -> dict:
        return evaluator.new_snapshot(
            questions=[deepcopy(self.range_question)],
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            top_k=6,
            generator_model={"path": "generator"},
            judge_model={"path": "judge"},
            range_guard=TEST_RANGE_GUARD,
        )

    def test_input_fingerprint_binds_generator_runtime_settings(
        self,
    ) -> None:
        common = {
            "input_files": [],
            "top_k": 6,
            "generator_model_sha256": "generator",
            "judge_model_sha256": "judge",
            "judge_backend": "transformers",
            "range_guard": TEST_RANGE_GUARD,
        }
        first = evaluator.build_input_fingerprint(
            **common,
            generator_runtime_settings={
                "llm": {"max_tokens": 768, "temperature": 0.0},
            },
        )
        second = evaluator.build_input_fingerprint(
            **common,
            generator_runtime_settings={
                "llm": {"max_tokens": 1024, "temperature": 0.0},
            },
        )
        self.assertNotEqual(first, second)

    def test_input_fingerprint_binds_required_mode_and_pipeline_state(
        self,
    ) -> None:
        common = {
            "input_files": [],
            "top_k": 6,
            "generator_runtime_settings": {},
            "generator_model_sha256": "generator",
            "judge_model_sha256": "judge",
            "judge_backend": "transformers",
            "range_guard": TEST_RANGE_GUARD,
        }
        lexical = evaluator.build_input_fingerprint(
            **common,
            required_retrieval_mode="lexical",
            pipeline_state={"embedding_cache": "cache-v1"},
        )
        hybrid = evaluator.build_input_fingerprint(
            **common,
            required_retrieval_mode="hybrid",
            pipeline_state={"embedding_cache": "cache-v1"},
        )
        changed_cache = evaluator.build_input_fingerprint(
            **common,
            required_retrieval_mode="lexical",
            pipeline_state={"embedding_cache": "cache-v2"},
        )
        self.assertEqual(len({lexical, hybrid, changed_cache}), 3)

    def test_input_fingerprint_and_resume_bind_judge_retry_policy(
        self,
    ) -> None:
        common = {
            "input_files": [],
            "top_k": 6,
            "generator_runtime_settings": {"llm": {"max_tokens": 768}},
            "generator_model_sha256": "generator",
            "judge_model_sha256": "judge",
            "judge_backend": "transformers",
            "range_guard": TEST_RANGE_GUARD,
        }
        first = evaluator.build_input_fingerprint(
            **common,
            judge_max_attempts=1,
        )
        second = evaluator.build_input_fingerprint(
            **common,
            judge_max_attempts=4,
        )
        self.assertNotEqual(first, second)

        snapshot = self.make_snapshot()
        with self.assertRaises(evaluator.EvaluationInputError):
            evaluator.validate_resume_snapshot(
                snapshot,
                input_fingerprint="test",
                judge_max_attempts=1,
            )

    def test_input_fingerprint_binds_question_and_system_roots(self) -> None:
        common = {
            "input_files": [],
            "top_k": 6,
            "generator_runtime_settings": {"llm": {"max_tokens": 768}},
            "generator_model_sha256": "generator",
            "judge_model_sha256": "judge",
            "judge_backend": "transformers",
            "range_guard": TEST_RANGE_GUARD,
            "benchmark_dataset_root": SYNTHESIZED_DATASET_ROOT,
            "pipeline_dataset_root": DATASET_ROOT,
        }
        canonical = evaluator.build_input_fingerprint(
            **common,
            question_set="canonical",
            system_root=evaluator.PROJECT_ROOT,
        )
        synthesized = evaluator.build_input_fingerprint(
            **common,
            question_set="synthesized",
            system_root=evaluator.PROJECT_ROOT,
        )
        alternate_system = evaluator.build_input_fingerprint(
            **common,
            question_set="canonical",
            system_root=DENSE_SYSTEM_ROOT,
        )
        alternate_pipeline = evaluator.build_input_fingerprint(
            **{
                **common,
                "pipeline_dataset_root": SYNTHESIZED_DATASET_ROOT,
            },
            question_set="canonical",
            system_root=evaluator.PROJECT_ROOT,
        )
        self.assertEqual(
            len({canonical, synthesized, alternate_system, alternate_pipeline}),
            4,
        )

    @staticmethod
    def pipeline_result(*, generate: bool) -> dict:
        return {
            "scope": "range",
            "answer": "송어의 움직임을 표현한다.",
            "generation_mode": "llm" if generate else "extractive",
            "answer_basis": (
                "retrieved_evidence"
                if generate
                else "retrieval_extractive"
            ),
            "generation_fallback_reason": None,
            "context_limited": False,
            "evidence": [
                {
                    "id": "die-forelle-ku-002",
                    "kind": "expert",
                    "evidence_type": "expert_annotation",
                    "source_ids": ["kim-die-forelle-02"],
                    "text": "송어의 움직임을 표현한다.",
                    "scope_match": "overlaps_query_range",
                    "in_requested_scope": True,
                }
            ],
            "evidence_notices": [],
        }

    def test_range_is_metadata_and_question_is_not_rewritten(self) -> None:
        snapshot = self.make_snapshot()
        case = snapshot["results"][0]["inference_runs"][0]
        calls = []

        def ask_fn(**kwargs):
            calls.append(kwargs)
            return self.pipeline_result(generate=kwargs["generate"])

        evaluator.run_pipeline_phase(
            snapshot,
            phase="retrieval",
            ask_fn=ask_fn,
            checkpoint=lambda: None,
            top_k=6,
            selected_case_ids={case["case_id"]},
        )
        self.assertEqual(calls[0]["measure_range"], (2, 27))
        self.assertFalse(calls[0]["allow_internal_knowledge"])
        self.assertEqual(
            calls[0]["question"],
            "피아노 파트에서 두 번째 박의 악센트는 무엇을 표현할까?",
        )
        self.assertNotIn("마디", calls[0]["question"])
        diagnostics = case["retrieval_probe"]["diagnostics"]
        self.assertTrue(diagnostics["target_source_grounded"])
        self.assertTrue(
            diagnostics["any_expected_knowledge_unit_retrieved"]
        )

    def test_completed_case_is_skipped_and_failed_case_retries(self) -> None:
        snapshot = self.make_snapshot()
        case = snapshot["results"][0]["inference_runs"][0]
        calls = 0

        def ask_fn(**kwargs):
            nonlocal calls
            calls += 1
            return self.pipeline_result(generate=False)

        arguments = {
            "snapshot": snapshot,
            "phase": "retrieval",
            "ask_fn": ask_fn,
            "checkpoint": lambda: None,
            "top_k": 6,
            "selected_case_ids": {case["case_id"]},
        }
        evaluator.run_pipeline_phase(**arguments)
        evaluator.run_pipeline_phase(**arguments)
        self.assertEqual(calls, 1)

        evaluator._invalidate_case_from_phase(case, "retrieval")
        evaluator.run_pipeline_phase(
            **{
                **arguments,
                "ask_fn": lambda **kwargs: (_ for _ in ()).throw(
                    RuntimeError("temporary failure")
                ),
            }
        )
        self.assertIsNone(case["retrieval_probe"])
        self.assertIn("retrieval", case["phase_errors"])
        evaluator.run_pipeline_phase(**arguments)
        self.assertIsNotNone(case["retrieval_probe"])
        self.assertNotIn("retrieval", case["phase_errors"])

    def test_strict_hybrid_mode_accepts_intentional_lexical_route(self) -> None:
        snapshot = self.make_snapshot()
        case = snapshot["results"][0]["inference_runs"][0]

        def ask_fn(**kwargs):
            result = self.pipeline_result(generate=False)
            result["retrieval"] = {
                "configured_mode": "hybrid",
                "active_mode": "hybrid",
                "dense_available": True,
                "fallback_reason": None,
                "last_dense_error": None,
                "last_search_mode": "lexical_route",
                "query_route": "lexical",
                "route_reason": "content_free_query",
                "dense_attempted": False,
                "dense_contributed": False,
                "fallback_used": False,
            }
            return result

        evaluator.run_pipeline_phase(
            snapshot,
            phase="retrieval",
            ask_fn=ask_fn,
            checkpoint=lambda: None,
            top_k=6,
            selected_case_ids={case["case_id"]},
            required_retrieval_mode="hybrid",
        )
        validation = case["retrieval_probe"]["retrieval_validation"]
        self.assertEqual(validation["last_search_mode"], "lexical_route")
        self.assertFalse(validation["fallback_used"])

    def test_strict_hybrid_mode_fails_closed_on_dense_fallback(self) -> None:
        snapshot = self.make_snapshot()
        case = snapshot["results"][0]["inference_runs"][0]

        def ask_fn(**kwargs):
            result = self.pipeline_result(generate=False)
            result["retrieval"] = {
                "configured_mode": "hybrid",
                "active_mode": "hybrid",
                "dense_available": True,
                "fallback_reason": None,
                "last_dense_error": "embedding failed",
                "last_search_mode": "lexical_fallback",
                "query_route": "lexical",
                "route_reason": "dense_query_error",
                "dense_attempted": True,
                "dense_contributed": False,
                "fallback_used": True,
            }
            return result

        with self.assertRaises(evaluator.RetrievalModeMismatchError):
            evaluator.run_pipeline_phase(
                snapshot,
                phase="retrieval",
                ask_fn=ask_fn,
                checkpoint=lambda: None,
                top_k=6,
                selected_case_ids={case["case_id"]},
                required_retrieval_mode="hybrid",
            )
        self.assertIsNone(case["retrieval_probe"])
        self.assertEqual(
            case["phase_errors"]["retrieval"]["type"],
            "RetrievalModeMismatchError",
        )


class JudgeProtocolTests(unittest.TestCase):
    def test_prompt_enforces_curated_scope_and_no_support_transfer(
        self,
    ) -> None:
        prompt = evaluator.JUDGE_SYSTEM_COMMON
        normalized_prompt = " ".join(prompt.split())
        self.assertIn(
            "Each R item may include a curator-approved scope",
            normalized_prompt,
        )
        self.assertIn(
            "direct_required: hard completeness",
            normalized_prompt,
        )
        self.assertIn(
            "optional_background: omission alone is allowed",
            normalized_prompt,
        )
        self.assertIn(
            "It remains factual authority",
            normalized_prompt,
        )
        self.assertIn(
            "Never transfer support between R claims",
            normalized_prompt,
        )
        self.assertIn(
            "Classify EVERY R and S ID exactly once",
            normalized_prompt,
        )
        self.assertIn(
            "supported by linked R or S items",
            normalized_prompt,
        )
        self.assertIn(
            "a candidate that only recommends comparison covers S but "
            "leaves R missing",
            normalized_prompt,
        )
        self.assertIn(
            "Never invent candidate wording in the rationale",
            normalized_prompt,
        )
        self.assertIn(
            '"reference_id": "S001"',
            prompt,
        )
        self.assertIn(
            "classify every S factuality item",
            evaluator.FRAME_INSTRUCTIONS["claim_alignment"],
        )
        self.assertIn(
            "classify every R and every S item",
            evaluator.FRAME_INSTRUCTIONS["contradiction_first"],
        )
        self.assertIn(
            "if all other R items were hidden",
            normalized_prompt,
        )
        self.assertIn(
            "a candidate that only says to connect the vowels covers the "
            "first R item, leaves the consonant R item missing",
            normalized_prompt,
        )
        self.assertIn(
            "s-substitution-nli",
            evaluator.JUDGE_PROTOCOL_VERSION,
        )

    def test_candidate_normalization_removes_only_provenance_labels(
        self,
    ) -> None:
        normalized = evaluator.normalize_candidate_answer_for_judge(
            "[die-forelle-ku-002] [sqa-0001] "
            "[webchunk-cecff2bace03ab67e32d] "
            "악센트는 송어를 표현한다.\n\n"
            "제공된 검색 근거: [die-forelle-ku-002] "
            "[webchunk-cecff2bace03ab67e32d]"
        )
        self.assertEqual(normalized, "악센트는 송어를 표현한다.")

    def test_atomic_nli_guard_blocks_sibling_claim_transfer(self) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": (
                    "두번째 박자의 악센트는 높은음자리표와 "
                    "낮은음자리표를 이동하면서 반복된다."
                ),
                "scope": "direct_required",
                "flags": ["multi_claim"],
                "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
                "source_sentence_index": 2,
                "source_claim_index": 1,
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": (
                    "두 번째 박의 강조는 물 위로 솟구쳐 노는 "
                    "송어의 생동감을 그린다."
                ),
            }
        ]
        neutral = {
            "entailment": 0.0006,
            "neutral": 0.9990,
            "contradiction": 0.0004,
        }
        signals = {
            "atomic_claim_signals": [
                {
                    "reference_id": "R001",
                    "full_answer_nli": neutral,
                    "candidate_nli": [
                        {"candidate_id": "C001", **neutral}
                    ],
                    "contextualized_candidate_nli": [
                        {
                            "candidate_id": "C001",
                            "context_text": candidates[0]["text"],
                            **neutral,
                        }
                    ],
                }
            ]
        }
        frames = {}
        for frame in evaluator.FRAME_NAMES:
            validated = validate_payload(
                assessment_payload(
                    frame,
                    reference_items=references,
                    candidate_items=candidates,
                ),
                reference_items=references,
                candidate_items=candidates,
                atomic_signals=signals,
                atomic_nli_threshold=0.8,
            )
            self.assertEqual(
                validated["atomic_claim_entailment_warnings"][0][
                    "reference_id"
                ],
                "R001",
            )
            computed = evaluator.compute_frame_result(validated)
            self.assertFalse(computed["pass"])
            self.assertIn(
                "unverified_atomic_claim_entailment",
                computed["failure_reasons"],
            )
            frames[frame] = {
                "assessment": validated,
                "computed": computed,
            }
        aggregate = evaluator.aggregate_judge_frames(
            frames,
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "human_review")

    def test_atomic_nli_guard_accepts_semantic_mechanism_paraphrase(
        self,
    ) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": (
                    "두번째 박자의 악센트는 높은음자리표와 "
                    "낮은음자리표를 이동하면서 반복된다."
                ),
                "scope": "direct_required",
                "flags": ["multi_claim"],
                "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
                "source_sentence_index": 2,
                "source_claim_index": 1,
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": (
                    "두 번째 박의 악센트가 높은 성부와 낮은 성부 "
                    "사이를 오가며 되풀이된다."
                ),
            }
        ]
        entailed = {
            "entailment": 0.9981,
            "neutral": 0.0015,
            "contradiction": 0.0004,
        }
        signals = {
            "atomic_claim_signals": [
                {
                    "reference_id": "R001",
                    "full_answer_nli": entailed,
                    "candidate_nli": [
                        {"candidate_id": "C001", **entailed}
                    ],
                    "contextualized_candidate_nli": [
                        {
                            "candidate_id": "C001",
                            "context_text": candidates[0]["text"],
                            **entailed,
                        }
                    ],
                }
            ]
        }
        validated = validate_payload(
            assessment_payload(
                "claim_alignment",
                reference_items=references,
                candidate_items=candidates,
            ),
            reference_items=references,
            candidate_items=candidates,
            atomic_signals=signals,
            atomic_nli_threshold=0.8,
        )
        self.assertEqual(
            validated["atomic_claim_entailment_warnings"],
            [],
        )
        self.assertTrue(evaluator.compute_frame_result(validated)["pass"])

    def test_atomic_nli_guard_uses_question_subject_for_referent(
        self,
    ) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": (
                    "두번째 박자의 악센트는 높은음자리표와 "
                    "낮은음자리표를 이동하면서 반복된다."
                ),
                "scope": "direct_required",
                "flags": ["multi_claim"],
                "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
                "source_sentence_index": 2,
                "source_claim_index": 1,
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": (
                    "악센트가 위아래 성부를 오가며 반복되는 움직임도 "
                    "그 이미지를 강화한다."
                ),
            }
        ]
        neutral = {
            "entailment": 0.0124,
            "neutral": 0.9873,
            "contradiction": 0.0003,
        }
        contextualized = {
            "entailment": 0.9012,
            "neutral": 0.0982,
            "contradiction": 0.0006,
        }
        context_text = (
            "피아노 파트에서 두 번째 박의 악센트에 대해 말하면, "
            + candidates[0]["text"]
        )
        signals = {
            "atomic_claim_signals": [
                {
                    "reference_id": "R001",
                    "full_answer_nli": neutral,
                    "candidate_nli": [
                        {"candidate_id": "C001", **neutral}
                    ],
                    "contextualized_candidate_nli": [
                        {
                            "candidate_id": "C001",
                            "context_text": context_text,
                            **contextualized,
                        }
                    ],
                }
            ]
        }
        validated = evaluator.validate_judge_assessment(
            json.dumps(
                assessment_payload(
                    "claim_alignment",
                    reference_items=references,
                    candidate_items=candidates,
                ),
                ensure_ascii=False,
            ),
            expected_frame="claim_alignment",
            has_measure_range=True,
            candidate_answer=candidates[0]["text"],
            authoritative_reference_items=references,
            candidate_answer_items=candidates,
            allowed_context_texts=[
                "피아노 파트에서 두 번째 박의 악센트는 무엇을 표현할까?"
            ],
            atomic_coverage_signals=signals,
            atomic_coverage_nli_threshold=0.8,
        )
        self.assertEqual(
            evaluator._candidate_with_question_subject_context(
                "피아노 파트에서 두 번째 박의 악센트는 무엇을 표현할까?",
                candidates[0]["text"],
            ),
            context_text,
        )
        self.assertEqual(
            validated["atomic_claim_entailment_warnings"],
            [],
        )
        self.assertTrue(evaluator.compute_frame_result(validated)["pass"])

    def test_literal_pronunciation_anchor_blocks_false_coverage(self) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": (
                    "ed에서 d발음은 가능하면 다음 단어인 i에 가깝게 "
                    "발음한다."
                ),
                "scope": "direct_required",
                "flags": [],
                "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": "이어지는 e 모음들을 부드럽게 연음한다.",
            }
        ]
        frames = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(
                frame,
                reference_items=references,
                candidate_items=candidates,
            )
            validated = validate_payload(
                payload,
                reference_items=references,
                candidate_items=candidates,
            )
            self.assertEqual(
                validated["literal_anchor_coverage_warnings"][0][
                    "missing_anchors"
                ],
                ["d"],
            )
            computed = evaluator.compute_frame_result(validated)
            self.assertFalse(computed["pass"])
            self.assertIn(
                "unverified_literal_anchor_coverage",
                computed["failure_reasons"],
            )
            frames[frame] = {
                "assessment": validated,
                "computed": computed,
            }

        aggregate = evaluator.aggregate_judge_frames(
            frames,
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertFalse(aggregate["reliable_rag_llm_pass"])

    def test_standalone_pronunciation_anchor_clears_guard(self) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": "ed에서 d 발음은 다음 i에 가깝게 발음한다.",
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": "끝의 d를 다음 i 쪽으로 붙여 발음한다.",
            }
        ]
        validated = validate_payload(
            assessment_payload(
                "claim_alignment",
                reference_items=references,
                candidate_items=candidates,
            ),
            reference_items=references,
            candidate_items=candidates,
        )
        self.assertEqual(
            validated["literal_anchor_coverage_warnings"],
            [],
        )
        self.assertTrue(evaluator.compute_frame_result(validated)["pass"])

    def test_unlinked_or_embedded_anchor_does_not_clear_guard(self) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": "l발음과 r 발음을 각각 명확하게 한다.",
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": "l’erbe의 l를 명확하게 한다.",
            },
            {
                "candidate_id": "C002",
                "text": "관련 없는 문장에서 r를 언급한다.",
            },
        ]
        payload = assessment_payload(
            "claim_alignment",
            reference_items=references,
            candidate_items=candidates,
        )
        payload["reference_assessments"][0]["candidate_ids"] = ["C001"]
        payload["candidate_assessments"][0]["reference_ids"] = ["R001"]
        payload["candidate_assessments"][1].update(
            status="irrelevant",
            reference_ids=[],
        )
        validated = validate_payload(
            payload,
            reference_items=references,
            candidate_items=candidates,
        )
        warning = validated["literal_anchor_coverage_warnings"][0]
        self.assertEqual(warning["required_anchors"], ["l", "r"])
        self.assertEqual(warning["missing_anchors"], ["r"])

    def test_novel_numeric_prescription_is_normalized_to_material_error(
        self,
    ) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": "느리게 연습하고 음정 없이 발음만 연습한다.",
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": (
                    "느리게 연습하되 립트릴을 매일 30분씩 반드시 "
                    "병행한다."
                ),
            }
        ]
        validated = validate_payload(
            assessment_payload(
                "contradiction_first",
                reference_items=references,
                candidate_items=candidates,
                has_range=False,
            ),
            reference_items=references,
            candidate_items=candidates,
            has_range=False,
        )
        self.assertEqual(
            validated["novel_numeric_fact_warnings"][0]["novel_anchors"],
            ["30분"],
        )
        self.assertEqual(
            validated["candidate_assessments"][0]["status"],
            "mixed",
        )
        self.assertEqual(
            validated["relationship"],
            "materially_unreliable",
        )
        self.assertIn(
            "invents_specific_fact",
            validated["critical_error_types"],
        )
        self.assertFalse(evaluator.compute_frame_result(validated)["pass"])

    def test_numeric_fact_repeated_from_question_context_is_allowed(
        self,
    ) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": "제시된 시간을 지켜 연습한다.",
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": "질문에 제시된 대로 30분 연습한다.",
            }
        ]
        payload = assessment_payload(
            "claim_alignment",
            reference_items=references,
            candidate_items=candidates,
            has_range=False,
        )
        validated = evaluator.validate_judge_assessment(
            json.dumps(payload, ensure_ascii=False),
            expected_frame="claim_alignment",
            has_measure_range=False,
            candidate_answer=candidates[0]["text"],
            authoritative_reference_items=references,
            candidate_answer_items=candidates,
            allowed_context_texts=["30분 연습은 어떻게 해야 할까?"],
        )
        self.assertEqual(validated["novel_numeric_fact_warnings"], [])
        self.assertTrue(evaluator.compute_frame_result(validated)["pass"])

    def test_exhaustive_schema_accepts_json_independent_of_key_order(
        self,
    ) -> None:
        payload = assessment_payload("claim_alignment")
        payload = dict(reversed(list(payload.items())))
        validated = validate_payload(payload)
        self.assertEqual(validated["relationship"], "equivalent")
        self.assertEqual(
            {item["reference_id"] for item in validated[
                "reference_assessments"
            ]},
            {"R001", "R002", "R003"},
        )
        self.assertEqual(
            {item["candidate_id"] for item in validated[
                "candidate_assessments"
            ]},
            {"C001", "C002"},
        )

    def test_every_reference_and_candidate_id_is_required_exactly_once(
        self,
    ) -> None:
        base = assessment_payload("claim_alignment")
        malformed = []

        missing_reference = deepcopy(base)
        missing_reference["reference_assessments"].pop()
        malformed.append(missing_reference)

        duplicate_reference = deepcopy(base)
        duplicate_reference["reference_assessments"][-1][
            "reference_id"
        ] = "R001"
        malformed.append(duplicate_reference)

        missing_candidate = deepcopy(base)
        missing_candidate["candidate_assessments"].pop()
        malformed.append(missing_candidate)

        unknown_candidate = deepcopy(base)
        unknown_candidate["candidate_assessments"][-1][
            "candidate_id"
        ] = "C999"
        malformed.append(unknown_candidate)

        for payload in malformed:
            with self.subTest(payload=payload):
                with self.assertRaises(evaluator.JudgeOutputError):
                    validate_payload(payload)

    def test_redundant_reference_links_prune_but_bad_links_block(
        self,
    ) -> None:
        one_way_reference = assessment_payload("claim_alignment")
        one_way_reference["candidate_assessments"][0][
            "reference_ids"
        ] = ["R002"]
        validated = validate_payload(one_way_reference)
        self.assertEqual(validated["semantic_link_warnings"], [])
        self.assertIn(
            "R001:C001:one_way_redundant_reference_link_removed",
            validated["deterministic_normalizations"],
        )
        self.assertIn(
            "R003:C001:one_way_redundant_reference_link_removed",
            validated["deterministic_normalizations"],
        )
        self.assertTrue(evaluator.compute_frame_result(validated)["pass"])

        one_way_candidate = assessment_payload("claim_alignment")
        one_way_candidate["reference_assessments"][0][
            "candidate_ids"
        ] = ["C001"]
        validated = validate_payload(one_way_candidate)
        self.assertTrue(validated["semantic_link_warnings"])
        self.assertFalse(evaluator.compute_frame_result(validated)["pass"])

        unsupported_with_link = assessment_payload("claim_alignment")
        unsupported_with_link["candidate_assessments"][0][
            "status"
        ] = "unsupported"
        validated = validate_payload(unsupported_with_link)
        self.assertFalse(evaluator.compute_frame_result(validated)["pass"])

    def test_redundant_one_way_link_is_not_eligible_for_automatic_pass(
        self,
    ) -> None:
        payload = assessment_payload(
            "claim_alignment",
            reference_items=TEST_REFERENCE_ITEMS[:2],
            candidate_items=TEST_CANDIDATE_ITEMS,
        )
        payload["reference_assessments"] = [
            {
                "reference_id": "R001",
                "status": "covered",
                "candidate_ids": ["C001"],
            },
            {
                "reference_id": "R002",
                "status": "covered",
                "candidate_ids": ["C002"],
            },
        ]
        payload["candidate_assessments"] = [
            {
                "candidate_id": "C001",
                "status": "supported",
                "reference_ids": ["R001", "R002"],
            },
            {
                "candidate_id": "C002",
                "status": "supported",
                "reference_ids": ["R002"],
            },
        ]
        validated = validate_payload(
            payload,
            reference_items=TEST_REFERENCE_ITEMS[:2],
            candidate_items=TEST_CANDIDATE_ITEMS,
        )
        self.assertTrue(validated["semantic_link_warnings"])
        self.assertEqual(validated["semantic_link_orphans"], [])
        self.assertFalse(evaluator.compute_frame_result(validated)["pass"])

    def test_one_way_redundant_reference_link_is_safely_pruned(
        self,
    ) -> None:
        payload = assessment_payload(
            "claim_alignment",
            reference_items=SCOPED_REFERENCE_ITEMS,
            candidate_items=TEST_CANDIDATE_ITEMS,
        )
        payload["candidate_assessments"][1]["reference_ids"] = ["R001"]
        validated = validate_payload(
            payload,
            reference_items=SCOPED_REFERENCE_ITEMS,
            candidate_items=TEST_CANDIDATE_ITEMS,
        )
        optional = next(
            item
            for item in validated["reference_assessments"]
            if item["reference_id"] == "R002"
        )
        self.assertEqual(optional["candidate_ids"], ["C001"])
        self.assertIn(
            "R002:C002:one_way_redundant_reference_link_removed",
            validated["deterministic_normalizations"],
        )
        self.assertEqual(validated["semantic_link_warnings"], [])
        self.assertTrue(evaluator.compute_frame_result(validated)["pass"])

        direct_payload = assessment_payload(
            "claim_alignment",
            reference_items=SCOPED_REFERENCE_ITEMS,
            candidate_items=TEST_CANDIDATE_ITEMS,
        )
        direct_payload["candidate_assessments"][1]["reference_ids"] = [
            "R002"
        ]
        direct_validated = validate_payload(
            direct_payload,
            reference_items=SCOPED_REFERENCE_ITEMS,
            candidate_items=TEST_CANDIDATE_ITEMS,
        )
        direct = next(
            item
            for item in direct_validated["reference_assessments"]
            if item["reference_id"] == "R001"
        )
        self.assertEqual(direct["candidate_ids"], ["C001"])
        self.assertIn(
            "R001:C002:one_way_redundant_reference_link_removed",
            direct_validated["deterministic_normalizations"],
        )
        self.assertEqual(direct_validated["semantic_link_warnings"], [])
        self.assertTrue(
            evaluator.compute_frame_result(direct_validated)["pass"]
        )

    def test_link_bookkeeping_only_failure_requires_review_not_failure(
        self,
    ) -> None:
        states = frame_states()
        payload = assessment_payload("claim_alignment")
        payload["reference_assessments"][1]["candidate_ids"] = ["C002"]
        validated = validate_payload(payload)
        self.assertIn(
            "semantic_link_inconsistency",
            evaluator.compute_frame_result(validated)["failure_reasons"],
        )
        states["claim_alignment"] = {
            "assessment": validated,
            "computed": evaluator.compute_frame_result(validated),
        }

        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertFalse(aggregate["reliable_rag_llm_pass"])

    def test_strict_json_and_range_score_contract(self) -> None:
        payload = assessment_payload(
            "claim_alignment",
            has_range=False,
        )
        raw = "```json\n" + json.dumps(payload) + "\n```"
        with self.assertRaises(evaluator.JudgeOutputError):
            evaluator.validate_judge_assessment(
                raw,
                expected_frame="claim_alignment",
                has_measure_range=False,
                candidate_answer="후보",
                authoritative_reference_items=TEST_REFERENCE_ITEMS,
                candidate_answer_items=TEST_CANDIDATE_ITEMS,
            )

        payload["scores"]["range_consistency"] = 4
        with self.assertRaises(evaluator.JudgeOutputError):
            validate_payload(payload, has_range=False)

    def test_strict_json_repairs_one_apostrophe_escape_only(self) -> None:
        payload = assessment_payload("claim_alignment")
        payload["rationale"] = "표기 'vuò(vuo')'를 설명한다."
        valid_raw = json.dumps(payload, ensure_ascii=False)

        def validate(raw: str) -> dict:
            return evaluator.validate_judge_assessment(
                raw,
                expected_frame="claim_alignment",
                has_measure_range=True,
                candidate_answer="후보",
                authoritative_reference_items=TEST_REFERENCE_ITEMS,
                candidate_answer_items=TEST_CANDIDATE_ITEMS,
            )

        unchanged = validate(valid_raw)
        self.assertEqual(unchanged["rationale"], payload["rationale"])
        self.assertNotIn(
            evaluator.JSON_APOSTROPHE_ESCAPE_NORMALIZATION,
            unchanged["deterministic_normalizations"],
        )

        literal_backslash_payload = deepcopy(payload)
        literal_backslash_payload["rationale"] = (
            "유효하게 인코딩된 백슬래시와 아포스트로피 \\' 표기다."
        )
        literal_backslash = validate(json.dumps(
            literal_backslash_payload,
            ensure_ascii=False,
        ))
        self.assertEqual(
            literal_backslash["rationale"],
            literal_backslash_payload["rationale"],
        )
        self.assertNotIn(
            evaluator.JSON_APOSTROPHE_ESCAPE_NORMALIZATION,
            literal_backslash["deterministic_normalizations"],
        )

        malformed_raw = valid_raw.replace(
            "vuo')",
            "vuo\\')",
            1,
        )
        repaired = validate(malformed_raw)
        self.assertEqual(repaired["rationale"], payload["rationale"])
        self.assertIn(
            evaluator.JSON_APOSTROPHE_ESCAPE_NORMALIZATION,
            repaired["deterministic_normalizations"],
        )

        invalid_responses = {
            "arbitrary invalid escape": valid_raw.replace(
                "vuo')",
                "vuo\\q')",
                1,
            ),
            "markdown fence": f"```json\n{valid_raw}\n```",
            "trailing text after eligible repair": (
                malformed_raw + "\ntrailing"
            ),
            "second apostrophe escape": malformed_raw.replace(
                "설명한다.",
                "설명\\'한다.",
                1,
            ),
        }
        for label, raw in invalid_responses.items():
            with self.subTest(label=label):
                with self.assertRaises(evaluator.JudgeOutputError):
                    validate(raw)

    def test_conservative_repairs_accept_common_bounded_schema_errors(
        self,
    ) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": "배경 설명이다.",
                "scope": "optional_background",
                "flags": [],
                "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
            }
        ]
        candidates = [{"candidate_id": "C001", "text": "다른 답변이다."}]
        payload = assessment_payload(
            "claim_alignment",
            reference_items=references,
            candidate_items=candidates,
        )
        payload["scores"].update(
            core_correctness=2,
            core_coverage=1,
            question_relevance=1,
            range_consistency=None,
        )
        payload["relationship"] = "unrelated"
        payload["reference_assessments"][0].update(
            status="optional_background",
            candidate_ids=[],
        )
        payload["candidate_assessments"][0].update(
            status="irrelevant",
            reference_ids=[],
        )
        validated = validate_payload(
            payload,
            reference_items=references,
            candidate_items=candidates,
        )
        self.assertEqual(validated["scores"]["range_consistency"], 2)
        self.assertEqual(
            validated["reference_assessments"][0]["status"],
            "not_required",
        )
        self.assertIn(
            "null_range_consistency_normalized_to_uncertain",
            validated["deterministic_normalizations"],
        )
        self.assertIn(
            "R001:optional_scope_label_normalized_to_not_required",
            validated["deterministic_normalizations"],
        )
        self.assertFalse(evaluator.compute_frame_result(validated)["pass"])

    def test_missing_optional_s_row_is_reconstructed_from_candidate_links(
        self,
    ) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": "전주가 있는 악보를 일반적으로 많이 사용한다.",
                "scope": "direct_required",
                "flags": [],
                "scope_authority": evaluator.QUESTION_SCOPE_AUTHORITY,
            },
            {
                "reference_id": "S001",
                "text": "원본 악보와 비교해 보는 것이 좋다.",
                "scope": "optional_background",
                "flags": [],
                "scope_authority": (
                    evaluator.RETRIEVED_EXPERT_SCOPE_AUTHORITY
                ),
                "evidence_ids": ["die-forelle-ku-001"],
                "source_ids": ["yeon-die-forelle-09"],
                "support_requires_review": False,
                "retrieval_review_warnings": [],
            },
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": (
                    "전주가 있는 악보를 많이 쓰며 원본과 비교하면 좋다."
                ),
            }
        ]
        payload = assessment_payload(
            "claim_alignment",
            reference_items=references,
            candidate_items=candidates,
        )
        payload["reference_assessments"] = [
            payload["reference_assessments"][0]
        ]
        validated = validate_payload(
            payload,
            reference_items=references,
            candidate_items=candidates,
        )
        supplemental = validated["reference_assessments"][1]
        self.assertEqual(supplemental["reference_id"], "S001")
        self.assertEqual(supplemental["status"], "covered")
        self.assertEqual(supplemental["candidate_ids"], ["C001"])
        self.assertIn(
            "S001:missing_supplemental_assessment_reconstructed_from_"
            "candidate_links",
            validated["deterministic_normalizations"],
        )
        self.assertTrue(evaluator.compute_frame_result(validated)["pass"])

        unasserted = assessment_payload(
            "claim_alignment",
            reference_items=references,
            candidate_items=candidates,
        )
        unasserted["reference_assessments"] = [
            unasserted["reference_assessments"][0]
        ]
        unasserted["candidate_assessments"][0]["reference_ids"] = ["R001"]
        validated_unasserted = validate_payload(
            unasserted,
            reference_items=references,
            candidate_items=candidates,
        )
        self.assertEqual(
            validated_unasserted["reference_assessments"][1]["status"],
            "not_required",
        )

        missing_r = assessment_payload(
            "claim_alignment",
            reference_items=references,
            candidate_items=candidates,
        )
        missing_r["reference_assessments"] = []
        with self.assertRaisesRegex(
            evaluator.JudgeOutputError,
            "missing IDs=.*R001",
        ):
            validate_payload(
                missing_r,
                reference_items=references,
                candidate_items=candidates,
            )

    def test_exhaustive_material_errors_force_failure(self) -> None:
        for mode, reason in (
            ("contradiction", "reported_contradiction"),
            ("unsupported", "reported_unsupported_material_claim"),
        ):
            with self.subTest(mode=mode):
                validated = validate_payload(
                    assessment_payload(
                        "claim_alignment",
                        mode=mode,
                    )
                )
                result = evaluator.compute_frame_result(validated)
                self.assertFalse(result["pass"])
                self.assertIn(reason, result["failure_reasons"])
                self.assertTrue(validated["critical_error_types"])

    def test_unanchored_critical_error_routes_to_review_not_frame_error(
        self,
    ) -> None:
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(frame)
            payload["critical_error_types"] = ["invents_specific_fact"]
            validated = validate_payload(payload)
            self.assertEqual(validated["critical_error_types"], [])
            self.assertEqual(
                validated["unanchored_critical_error_warnings"],
                ["invents_specific_fact"],
            )
            result = evaluator.compute_frame_result(validated)
            self.assertFalse(result["pass"])
            self.assertIn(
                "unanchored_critical_error_requires_review",
                result["failure_reasons"],
            )
            states[frame] = {
                "assessment": validated,
                "computed": result,
            }
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertEqual(
            aggregate["answer_quality_status"],
            "human_review",
        )
        self.assertIn(
            "judge_error_warning_requires_review",
            aggregate["disagreement_reasons"],
        )

    def test_misplaced_relationship_and_unsubstantiated_low_scores_review(
        self,
    ) -> None:
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(
                frame,
                reference_items=SCOPED_REFERENCE_ITEMS,
            )
            payload["relationship"] = "materially_incomplete"
            payload["scores"]["core_correctness"] = 3
            payload["scores"]["core_coverage"] = 2
            payload["critical_error_types"] = ["minor_omission"]
            payload["rationale"] = (
                "필수 R은 모두 covered로 분류했지만 선택적 배경의 "
                "세부사항을 생략했다고 평가한다."
            )
            validated = validate_payload(
                payload,
                reference_items=SCOPED_REFERENCE_ITEMS,
            )
            self.assertEqual(validated["critical_error_types"], [])
            self.assertEqual(
                validated["misplaced_relationship_error_warnings"],
                ["minor_omission"],
            )
            result = evaluator.compute_frame_result(validated)
            self.assertFalse(result["pass"])
            self.assertIn(
                "misplaced_relationship_error_label_requires_review",
                result["failure_reasons"],
            )
            states[frame] = {
                "assessment": validated,
                "computed": result,
            }

        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertEqual(
            aggregate["answer_quality_status"],
            "human_review",
        )

    def test_misplaced_relationship_repairs_unanchored_material_relation(
        self,
    ) -> None:
        for misplaced_label in evaluator.RELATIONSHIPS:
            with self.subTest(misplaced_label=misplaced_label):
                payload = assessment_payload("claim_alignment")
                payload["relationship"] = "materially_unreliable"
                payload["scores"]["factual_safety"] = 2
                payload["critical_error_types"] = [misplaced_label]
                validated = validate_payload(payload)
                self.assertEqual(
                    validated["relationship"],
                    "materially_incomplete",
                )
                self.assertEqual(validated["critical_error_types"], [])
                self.assertEqual(
                    validated[
                        "misplaced_relationship_error_warnings"
                    ],
                    [misplaced_label],
                )

        payload = assessment_payload("claim_alignment")
        payload["critical_error_types"] = ["typo_or_new_taxon"]
        with self.assertRaisesRegex(
            evaluator.JudgeOutputError,
            "unknown critical_error_types",
        ):
            validate_payload(payload)

    def test_misplaced_relationship_never_softens_material_error(
        self,
    ) -> None:
        for mode, expected_error in (
            ("unsupported", "ungrounded_as_fact"),
            ("contradiction", "contradicts_expert"),
        ):
            with self.subTest(mode=mode):
                states = {}
                for frame in evaluator.FRAME_NAMES:
                    payload = assessment_payload(frame, mode=mode)
                    payload["critical_error_types"].append(
                        "minor_omission"
                    )
                    validated = validate_payload(payload)
                    self.assertIn(
                        expected_error,
                        validated["critical_error_types"],
                    )
                    self.assertEqual(
                        validated["misplaced_relationship_error_warnings"],
                        ["minor_omission"],
                    )
                    states[frame] = {
                        "assessment": validated,
                        "computed": (
                            evaluator.compute_frame_result(validated)
                        ),
                    }
                aggregate = evaluator.aggregate_judge_frames(
                    states,
                    generated_answer=grounded_answer(),
                )
                self.assertEqual(aggregate["status"], "fail")

    def test_missing_and_model_declared_optional_references_fail_closed(
        self,
    ) -> None:
        missing = assessment_payload("claim_alignment")
        missing["relationship"] = "minor_omission"
        for reference in missing["reference_assessments"][1:]:
            reference.update(status="missing", candidate_ids=[])
        for candidate in missing["candidate_assessments"]:
            candidate["reference_ids"] = ["R001"]
        missing_result = evaluator.compute_frame_result(
            validate_payload(missing)
        )
        self.assertFalse(missing_result["pass"])
        self.assertIn(
            "missing_authoritative_reference",
            missing_result["failure_reasons"],
        )

        not_required = assessment_payload("claim_alignment")
        for reference in not_required["reference_assessments"][1:]:
            reference.update(status="not_required", candidate_ids=[])
        for candidate in not_required["candidate_assessments"]:
            candidate["reference_ids"] = ["R001"]
        optional_result = evaluator.compute_frame_result(
            validate_payload(not_required)
        )
        self.assertFalse(optional_result["pass"])
        self.assertIn(
            "unapproved_not_required_reference",
            optional_result["failure_reasons"],
        )

    def test_curated_optional_omission_does_not_block_completeness(
        self,
    ) -> None:
        candidates = [TEST_CANDIDATE_ITEMS[0]]
        for optional_status in ("missing", "not_required"):
            with self.subTest(optional_status=optional_status):
                payload = assessment_payload(
                    "claim_alignment",
                    reference_items=SCOPED_REFERENCE_ITEMS,
                    candidate_items=candidates,
                )
                payload["reference_assessments"][0] = {
                    "reference_id": "R001",
                    "status": "covered",
                    "candidate_ids": ["C001"],
                }
                payload["reference_assessments"][1] = {
                    "reference_id": "R002",
                    "status": optional_status,
                    "candidate_ids": [],
                }
                payload["candidate_assessments"][0] = {
                    "candidate_id": "C001",
                    "status": "supported",
                    "reference_ids": ["R001"],
                }
                validated = validate_payload(
                    payload,
                    reference_items=SCOPED_REFERENCE_ITEMS,
                    candidate_items=candidates,
                )
                computed = evaluator.compute_frame_result(validated)
                self.assertTrue(computed["pass"])
                self.assertNotIn(
                    "missing_authoritative_reference",
                    computed["failure_reasons"],
                )
                self.assertNotIn(
                    "unapproved_not_required_reference",
                    computed["failure_reasons"],
                )

    def test_direct_required_cannot_be_declared_not_required(self) -> None:
        candidates = [TEST_CANDIDATE_ITEMS[0]]
        payload = assessment_payload(
            "claim_alignment",
            reference_items=SCOPED_REFERENCE_ITEMS,
            candidate_items=candidates,
        )
        payload["reference_assessments"][0] = {
            "reference_id": "R001",
            "status": "not_required",
            "candidate_ids": [],
        }
        payload["reference_assessments"][1] = {
            "reference_id": "R002",
            "status": "covered",
            "candidate_ids": ["C001"],
        }
        payload["candidate_assessments"][0] = {
            "candidate_id": "C001",
            "status": "supported",
            "reference_ids": ["R002"],
        }
        with self.assertRaisesRegex(
            evaluator.JudgeOutputError,
            "direct_required cannot be not_required",
        ):
            validate_payload(
                payload,
                reference_items=SCOPED_REFERENCE_ITEMS,
                candidate_items=candidates,
            )

    def test_optional_background_still_enforces_factuality(self) -> None:
        candidates = [
            TEST_CANDIDATE_ITEMS[0],
            {
                "candidate_id": "C002",
                "text": "보조 배경은 사실이 아니다.",
            },
        ]
        payload = assessment_payload(
            "claim_alignment",
            reference_items=SCOPED_REFERENCE_ITEMS,
            candidate_items=candidates,
        )
        payload["relationship"] = "contradictory"
        payload["scores"]["core_correctness"] = 2
        payload["scores"]["factual_safety"] = 2
        payload["reference_assessments"] = [
            {
                "reference_id": "R001",
                "status": "covered",
                "candidate_ids": ["C001"],
            },
            {
                "reference_id": "R002",
                "status": "contradicted",
                "candidate_ids": ["C002"],
            },
        ]
        payload["candidate_assessments"] = [
            {
                "candidate_id": "C001",
                "status": "supported",
                "reference_ids": ["R001"],
            },
            {
                "candidate_id": "C002",
                "status": "contradicted",
                "reference_ids": ["R002"],
            },
        ]
        payload["critical_error_types"] = ["contradicts_expert"]
        validated = validate_payload(
            payload,
            reference_items=SCOPED_REFERENCE_ITEMS,
            candidate_items=candidates,
        )
        computed = evaluator.compute_frame_result(validated)
        self.assertFalse(computed["pass"])
        self.assertIn(
            "reported_contradiction",
            computed["failure_reasons"],
        )

    def test_unrelated_same_composer_background_is_a_failure(self) -> None:
        states = frame_states(mode="unrelated")
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "fail")
        self.assertFalse(aggregate["reliable_rag_llm_pass"])
        for state in states.values():
            self.assertFalse(state["computed"]["pass"])
            self.assertIn(
                "material_relationship",
                state["computed"]["failure_reasons"],
            )
            self.assertTrue(all(
                item["status"] == "missing"
                for item in state["assessment"][
                    "reference_assessments"
                ]
            ))

    def test_weak_confidence_reviews_but_does_not_soften_failures(
        self,
    ) -> None:
        aggregate = evaluator.aggregate_judge_frames(
            frame_states(confidence=0.5),
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "human_review")

        aggregate = evaluator.aggregate_judge_frames(
            frame_states(mode="contradiction", confidence=0.5),
            generated_answer=grounded_answer(),
        )
        self.assertEqual(aggregate["status"], "fail")
        self.assertEqual(
            set(aggregate["weak_confidence_frames"]),
            set(evaluator.FRAME_NAMES),
        )

    def test_range_nli_reconciles_only_errors_confined_to_its_c_ids(
        self,
    ) -> None:
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(frame)
            payload["reference_assessments"][0][
                "candidate_ids"
            ] = ["C002"]
            payload["candidate_assessments"][0].update(
                status="unsupported",
                reference_ids=[],
            )
            payload["relationship"] = "materially_unreliable"
            payload["scores"]["factual_safety"] = 1
            payload["critical_error_types"] = [
                "wrong_measure_application"
            ]
            payload["rationale"] = (
                "부정 언급을 근거 없는 주장으로 잘못 분류했다."
            )
            validated = validate_payload(payload)
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }
        self.assertTrue(all(
            not state["computed"]["pass"] for state in states.values()
        ))

        negated_scope = {
            "status": "pass",
            "applicable": True,
            "wrong_measure_application": False,
            "excluded_claim_assessments": [
                {
                    "excluded_id": "X001",
                    "relation": "negated",
                    "candidate_ids": ["C001"],
                }
            ],
        }
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=negated_scope,
        )
        self.assertEqual(aggregate["status"], "pass")
        self.assertIn(
            "negated_other_range_mentions_reconciled",
            aggregate["disagreement_reasons"],
        )

        wrong_candidate_scope = deepcopy(negated_scope)
        wrong_candidate_scope["excluded_claim_assessments"][0][
            "candidate_ids"
        ] = ["C002"]
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=wrong_candidate_scope,
        )
        self.assertEqual(aggregate["status"], "fail")
        self.assertNotIn(
            "negated_other_range_mentions_reconciled",
            aggregate["disagreement_reasons"],
        )

    def test_range_reconciliation_cannot_erase_judge_error_warning(
        self,
    ) -> None:
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(frame)
            payload["reference_assessments"][0][
                "candidate_ids"
            ] = ["C002"]
            payload["candidate_assessments"][0].update(
                status="unsupported",
                reference_ids=[],
            )
            payload["relationship"] = "materially_unreliable"
            payload["scores"]["factual_safety"] = 1
            payload["critical_error_types"] = [
                "wrong_measure_application",
                "minor_omission",
            ]
            payload["rationale"] = (
                "다른 범위에 관한 부정 언급을 오류로 분류하면서 "
                "비핵심 라벨도 critical_error_types에 잘못 넣었다."
            )
            validated = validate_payload(payload)
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }

        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation={
                "status": "pass",
                "applicable": True,
                "wrong_measure_application": False,
                "excluded_claim_assessments": [
                    {
                        "excluded_id": "X001",
                        "relation": "negated",
                        "candidate_ids": ["C001"],
                    }
                ],
            },
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertFalse(aggregate["reliable_rag_llm_pass"])
        self.assertIn(
            "judge_error_warning_requires_review",
            aggregate["disagreement_reasons"],
        )

    def test_range_negation_never_erases_unsafe_mixed_advice(self) -> None:
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(frame)
            payload["candidate_assessments"][0].update(
                status="mixed",
                reference_ids=["R001", "R002", "R003"],
            )
            payload["relationship"] = "materially_unreliable"
            payload["scores"]["factual_safety"] = 1
            payload["critical_error_types"] = ["unsafe_vocal_advice"]
            payload["rationale"] = (
                "빠른 교대를 부정하지만 매일 30분 호흡 훈련을 "
                "의무화하는 근거 없는 조언을 덧붙였다."
            )
            validated = validate_payload(payload)
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }
        negated_scope = {
            "status": "pass",
            "applicable": True,
            "wrong_measure_application": False,
            "excluded_claim_assessments": [
                {
                    "excluded_id": "X001",
                    "relation": "negated",
                    "candidate_ids": ["C001"],
                }
            ],
        }
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=negated_scope,
        )
        self.assertEqual(aggregate["status"], "fail")
        self.assertNotIn(
            "negated_other_range_mentions_reconciled",
            aggregate["disagreement_reasons"],
        )

    def test_uncertain_range_reconciliation_can_only_yield_review(
        self,
    ) -> None:
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(frame)
            payload["reference_assessments"][0][
                "candidate_ids"
            ] = ["C002"]
            payload["candidate_assessments"][0].update(
                status="unsupported",
                reference_ids=[],
            )
            payload["relationship"] = "materially_unreliable"
            payload["scores"]["factual_safety"] = 1
            payload["critical_error_types"] = [
                "wrong_measure_application"
            ]
            payload["rationale"] = "이중 부정의 의미가 불확실하다."
            validated = validate_payload(payload)
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }
        uncertain_scope = {
            "status": "human_review",
            "applicable": True,
            "wrong_measure_application": False,
            "excluded_claim_assessments": [
                {
                    "excluded_id": "X001",
                    "relation": "uncertain",
                    "candidate_ids": ["C001"],
                }
            ],
        }
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=uncertain_scope,
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertFalse(aggregate["reliable_rag_llm_pass"])
        self.assertIn(
            "uncertain_other_range_mentions_reconciled_for_review",
            aggregate["disagreement_reasons"],
        )

    def test_review_only_negation_can_escalate_spurious_safety_fail(
        self,
    ) -> None:
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(frame)
            for reference in payload["reference_assessments"]:
                if reference["status"] == "covered":
                    reference["candidate_ids"] = ["C002"]
            payload["candidate_assessments"][0].update(
                status="unsupported",
                reference_ids=[],
            )
            payload["candidate_assessments"][1].update(
                status="supported",
                reference_ids=[
                    item["reference_id"]
                    for item in payload["reference_assessments"]
                    if item["status"] == "covered"
                ],
            )
            payload["relationship"] = "materially_unreliable"
            payload["scores"]["factual_safety"] = 1
            payload["critical_error_types"] = [
                "invents_specific_fact",
                "unsafe_vocal_advice",
            ]
            payload["rationale"] = (
                "단순 부정절을 근거 없는 조언으로 잘못 분류했다."
            )
            validated = validate_payload(payload)
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }
        negated_review = {
            "status": "human_review",
            "applicable": True,
            "wrong_measure_application": False,
            "excluded_claim_assessments": [
                {
                    "excluded_id": "X001",
                    "relation": "negated",
                    "candidate_ids": ["C001"],
                    "candidate_link_provenance": "clause_nli_localized",
                }
            ],
        }
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=negated_review,
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertIn(
            "uncertain_other_range_mentions_reconciled_for_review",
            aggregate["disagreement_reasons"],
        )

        negated_pass = deepcopy(negated_review)
        negated_pass["status"] = "pass"
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=negated_pass,
        )
        self.assertEqual(aggregate["status"], "fail")

    def test_review_only_negation_reconciles_support_only_reference(
        self,
    ) -> None:
        references = [
            {
                "reference_id": "R001",
                "text": "ff와 pp를 두 블록으로 대비해 메아리를 표현한다.",
                "scope": "optional_background",
                "flags": [],
                "scope_authority": evaluator.SUPPORTING_SCOPE_AUTHORITY,
            }
        ]
        candidates = [
            {
                "candidate_id": "C001",
                "text": "ff와 pp가 빠르게 번갈아 나오는 대목은 아니다.",
            },
            {
                "candidate_id": "C002",
                "text": "ff와 pp는 두 블록으로 대비해 메아리를 표현한다.",
            },
        ]
        states = {}
        for frame in evaluator.FRAME_NAMES:
            payload = assessment_payload(
                frame,
                reference_items=references,
                candidate_items=candidates,
            )
            payload["reference_assessments"][0].update(
                status="not_required",
                candidate_ids=[],
            )
            payload["candidate_assessments"][0].update(
                status="unsupported",
                reference_ids=[],
            )
            payload["candidate_assessments"][1].update(
                status="supported",
                reference_ids=["R001"],
            )
            payload["relationship"] = "materially_unreliable"
            payload["scores"].update(
                core_correctness=2,
                factual_safety=1,
                range_consistency=2,
            )
            payload["critical_error_types"] = ["invents_specific_fact"]
            payload["confidence"] = 0.65
            payload["rationale"] = "부정절을 새 주장으로 잘못 분류했다."
            validated = validate_payload(
                payload,
                reference_items=references,
                candidate_items=candidates,
            )
            states[frame] = {
                "assessment": validated,
                "computed": evaluator.compute_frame_result(validated),
            }
        range_review = {
            "status": "human_review",
            "applicable": True,
            "wrong_measure_application": False,
            "excluded_claim_assessments": [
                {
                    "excluded_id": "X001",
                    "relation": "negated",
                    "candidate_ids": ["C001"],
                    "candidate_link_provenance": "clause_nli_localized",
                }
            ],
        }
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=grounded_answer(),
            range_scope_evaluation=range_review,
            reference_review={
                "required": True,
                "reasons": [
                    "question_level_reference_scope_not_applicable_to_case",
                    "no_applicable_direct_required_reference",
                ],
                "conditional_mixed_reference_ids": [],
            },
        )
        self.assertEqual(aggregate["status"], "human_review")
        self.assertEqual(
            aggregate["answer_quality_status"],
            "human_review",
        )

    def test_semantic_pass_requires_rag_llm_expert_evidence_for_primary(
        self,
    ) -> None:
        states = frame_states()
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer={
                "generation_mode": "llm",
                "answer_basis": "internal_knowledge",
                "evidence": [],
            },
        )
        self.assertEqual(aggregate["status"], "pass")
        self.assertFalse(aggregate["pipeline_is_grounded_rag_llm"])
        self.assertFalse(aggregate["reliable_rag_llm_pass"])

        alternate = grounded_answer()
        alternate["diagnostics"]["target_source_grounded"] = False
        aggregate = evaluator.aggregate_judge_frames(
            states,
            generated_answer=alternate,
        )
        self.assertTrue(aggregate["reliable_rag_llm_pass"])
        self.assertFalse(aggregate["target_source_grounded"])


class JudgeResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        questions, _ = evaluator.load_evaluation_questions(DATASET_ROOT)
        cls.question = next(
            item
            for item in questions
            if item["source_id"] == "kim-die-forelle-02"
        )
        cls.range_contrast_question = next(
            item
            for item in questions
            if item["source_id"] == "kim-la-capinera-17"
        )

    @staticmethod
    def one_case_snapshot(
        question: dict,
        *,
        measure_range: list[int] | None = None,
    ) -> tuple[dict, dict]:
        snapshot = evaluator.new_snapshot(
            questions=[deepcopy(question)],
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            top_k=6,
            generator_model={"path": "generator"},
            judge_model={"path": "judge"},
            range_guard=TEST_RANGE_GUARD,
        )
        cases = snapshot["results"][0]["inference_runs"]
        case = next(
            item
            for item in cases
            if item["inference_input"]["measure_range"] == measure_range
        )
        snapshot["results"][0]["inference_runs"] = [case]
        evaluator.refresh_summary(snapshot)
        return snapshot, case

    def test_semantic_attempt_preserves_repaired_raw_and_audit_marker(
        self,
    ) -> None:
        snapshot = evaluator.new_snapshot(
            questions=[deepcopy(self.question)],
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            top_k=6,
            generator_model={"path": "generator"},
            judge_model={"path": "judge"},
            range_guard=TEST_RANGE_GUARD,
        )
        case = snapshot["results"][0]["inference_runs"][0]
        case["generated_answer"] = {
            "answer": "두 번째 박의 악센트는 송어의 움직임을 표현한다.",
            **grounded_answer(),
        }
        packet = evaluator._reference_packet(
            snapshot["results"][0],
            case,
        )
        payload = assessment_payload(
            "claim_alignment",
            reference_items=evaluator._all_semantic_reference_items(
                packet
            ),
            candidate_items=evaluator._candidate_answer_items(packet),
        )
        payload["rationale"] = "표기 'vuò(vuo')'를 언급한다."
        malformed_raw = json.dumps(
            payload,
            ensure_ascii=False,
        ).replace(
            "vuo')",
            "vuo\\')",
            1,
        )
        frame_state = {"status": "pending", "attempts": []}

        completed = evaluator._run_one_judge_frame(
            frame="claim_alignment",
            packet=packet,
            judge_fn=lambda _messages: malformed_raw,
            coverage_signal_fn=atomic_coverage_signals,
            coverage_nli_threshold=0.8,
            frame_state=frame_state,
            checkpoint=lambda: None,
            max_attempts=1,
        )

        self.assertTrue(completed)
        self.assertEqual(len(frame_state["attempts"]), 1)
        attempt = frame_state["attempts"][0]
        self.assertEqual(attempt["raw_response"], malformed_raw)
        self.assertEqual(
            attempt["parser_normalizations"],
            [evaluator.JSON_APOSTROPHE_ESCAPE_NORMALIZATION],
        )
        self.assertNotIn("error", attempt)
        self.assertIn(
            evaluator.JSON_APOSTROPHE_ESCAPE_NORMALIZATION,
            frame_state["assessment"]["deterministic_normalizations"],
        )

    def test_dual_frames_checkpoint_and_resume(self) -> None:
        snapshot = evaluator.new_snapshot(
            questions=[deepcopy(self.question)],
            dataset_root=DATASET_ROOT,
            input_files=[],
            input_fingerprint="test",
            top_k=6,
            generator_model={"path": "generator"},
            judge_model={"path": "judge"},
            range_guard=TEST_RANGE_GUARD,
        )
        case = snapshot["results"][0]["inference_runs"][0]
        case["generated_answer"] = {
            "answer": "두 번째 박의 악센트는 송어의 움직임을 표현한다.",
            **grounded_answer(),
        }
        packet = evaluator._reference_packet(
            snapshot["results"][0],
            case,
        )
        references = evaluator._authoritative_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)
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
            return json.dumps(
                assessment_payload(
                    frame,
                    reference_items=references,
                    candidate_items=candidates,
                ),
                ensure_ascii=False,
            )

        def checkpoint():
            nonlocal checkpoints
            checkpoints += 1

        arguments = {
            "snapshot": snapshot,
            "judge_fn": judge_fn,
            "range_signal_fn": hybrid_range_signals,
            "coverage_signal_fn": atomic_coverage_signals,
            "nli_threshold": 0.8,
            "embedding_delta_threshold": 0.05,
            "checkpoint": checkpoint,
            "max_attempts": 2,
            "selected_case_ids": {case["case_id"]},
        }
        evaluator.run_judge_phase(**arguments)
        aggregate = case["semantic_evaluation"]["aggregate"]
        self.assertEqual(calls, list(evaluator.FRAME_NAMES))
        self.assertGreaterEqual(checkpoints, 4)
        self.assertEqual(aggregate["status"], "pass")
        self.assertTrue(aggregate["reliable_rag_llm_pass"])

        evaluator.run_judge_phase(**arguments)
        self.assertEqual(calls, list(evaluator.FRAME_NAMES))

    def test_exhausted_semantic_frame_is_terminal_unjudgeable(self) -> None:
        snapshot, case = self.one_case_snapshot(
            self.question,
            measure_range=[2, 27],
        )
        case["generated_answer"] = {
            "answer": "두 번째 박의 악센트는 송어의 움직임을 표현한다.",
            **grounded_answer(),
        }
        case["phase_errors"]["judge"] = {"message": "stale failure"}
        invalid_raws = ["invalid json one", "invalid json two"]
        messages_seen = []

        def judge_fn(messages):
            messages_seen.append(deepcopy(messages))
            return invalid_raws[len(messages_seen) - 1]

        arguments = {
            "snapshot": snapshot,
            "judge_fn": judge_fn,
            "range_signal_fn": hybrid_range_signals,
            "coverage_signal_fn": atomic_coverage_signals,
            "nli_threshold": 0.8,
            "embedding_delta_threshold": 0.05,
            "checkpoint": lambda: None,
            "max_attempts": 2,
            "selected_case_ids": {case["case_id"]},
        }
        evaluator.run_judge_phase(**arguments)

        semantic = case["semantic_evaluation"]
        frame_state = semantic["frames"][evaluator.FRAME_NAMES[0]]
        self.assertEqual(frame_state["status"], "unjudgeable")
        self.assertEqual(
            [item["raw_response"] for item in frame_state["attempts"]],
            invalid_raws,
        )
        self.assertTrue(all("error" in item for item in frame_state["attempts"]))
        diagnostic = frame_state["unjudgeable_diagnostic"]
        self.assertEqual(
            diagnostic,
            {
                "status": "unjudgeable",
                "reason": "structured_output_retries_exhausted",
                "frame": evaluator.FRAME_NAMES[0],
                "attempt_count": 2,
                "max_attempts": 2,
                "last_error": frame_state["attempts"][-1]["error"],
            },
        )
        self.assertNotIn(
            "Your previous response was invalid:",
            messages_seen[0][0]["content"],
        )
        self.assertIn(
            "Your previous response was invalid:",
            messages_seen[1][0]["content"],
        )

        aggregate = semantic["aggregate"]
        self.assertTrue(aggregate["unjudgeable"])
        self.assertEqual(aggregate["status"], "human_review")
        self.assertEqual(
            aggregate["answer_quality_status"],
            "human_review",
        )
        self.assertTrue(aggregate["pipeline_is_grounded_rag_llm"])
        self.assertFalse(aggregate["reliable_rag_llm_pass"])
        self.assertFalse(aggregate["answer_quality_rag_llm_pass"])
        self.assertEqual(
            aggregate["unjudgeable_diagnostic"],
            diagnostic,
        )
        self.assertEqual(semantic["unjudgeable_diagnostic"], diagnostic)
        self.assertNotIn("judge", case["phase_errors"])

        summary = snapshot["summary"]
        self.assertEqual(summary["unjudgeable_cases"], 1)
        self.assertEqual(summary["phases"]["judge"]["status"], "complete")
        self.assertEqual(summary["phases"]["judge"]["error_cases"], 0)
        self.assertEqual(
            summary["phases"]["judge"]["unjudgeable_cases"],
            1,
        )
        self.assertEqual(summary["semantic_judge"]["unjudgeable_cases"], 1)
        self.assertEqual(
            summary["semantic_judge"]["by_piece"]["die-forelle"][
                "unjudgeable_cases"
            ],
            1,
        )
        self.assertEqual(snapshot["run"]["status"], "complete")

        evaluator.run_judge_phase(**arguments)
        self.assertEqual(len(messages_seen), 2)
        self.assertEqual(len(frame_state["attempts"]), 2)

    def test_exhausted_range_frame_is_terminal_unjudgeable(self) -> None:
        snapshot, case = self.one_case_snapshot(
            self.range_contrast_question,
            measure_range=[51, 58],
        )
        case["generated_answer"] = {
            "answer": "ff와 pp를 두 덩어리로 대비해 메아리를 표현한다.",
            **grounded_answer(),
        }
        packet = evaluator._reference_packet(
            snapshot["results"][0],
            case,
        )
        references = evaluator._all_semantic_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)
        invalid_raws = ["invalid range one", "invalid range two"]
        range_messages = []

        def judge_fn(messages):
            system = messages[0]["content"]
            for frame in evaluator.FRAME_NAMES:
                if f"Frame: {frame}" in system:
                    return json.dumps(
                        assessment_payload(
                            frame,
                            reference_items=references,
                            candidate_items=candidates,
                        ),
                        ensure_ascii=False,
                    )
            range_messages.append(deepcopy(messages))
            return invalid_raws[len(range_messages) - 1]

        evaluator.run_judge_phase(
            snapshot,
            judge_fn=judge_fn,
            range_signal_fn=hybrid_range_signals,
            coverage_signal_fn=atomic_coverage_signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
            checkpoint=lambda: None,
            max_attempts=2,
            selected_case_ids={case["case_id"]},
        )

        semantic = case["semantic_evaluation"]
        self.assertTrue(
            all(
                state["status"] == "complete"
                for state in semantic["frames"].values()
            )
        )
        range_state = semantic["range_scope"]
        self.assertEqual(range_state["status"], "unjudgeable")
        self.assertEqual(
            [item["raw_response"] for item in range_state["attempts"]],
            invalid_raws,
        )
        self.assertEqual(
            range_state["unjudgeable_diagnostic"]["frame"],
            "range_scope",
        )
        self.assertEqual(
            range_state["unjudgeable_diagnostic"]["attempt_count"],
            2,
        )
        self.assertNotIn(
            "Your previous response was invalid:",
            range_messages[0][0]["content"],
        )
        self.assertIn(
            "Your previous response was invalid:",
            range_messages[1][0]["content"],
        )
        aggregate = semantic["aggregate"]
        self.assertTrue(aggregate["unjudgeable"])
        self.assertEqual(aggregate["status"], "human_review")
        self.assertEqual(
            aggregate["answer_quality_status"],
            "human_review",
        )
        self.assertFalse(aggregate["reliable_rag_llm_pass"])
        self.assertFalse(aggregate["answer_quality_rag_llm_pass"])
        self.assertNotIn("judge", case["phase_errors"])
        self.assertEqual(
            snapshot["summary"]["semantic_judge"]["by_piece"][
                "la-capinera"
            ]["unjudgeable_cases"],
            1,
        )

    def test_coverage_guard_contract_error_remains_phase_error(self) -> None:
        snapshot, case = self.one_case_snapshot(
            self.question,
            measure_range=[2, 27],
        )
        case["generated_answer"] = {
            "answer": "두 번째 박의 악센트는 송어의 움직임을 표현한다.",
            **grounded_answer(),
        }
        packet = evaluator._reference_packet(
            snapshot["results"][0],
            case,
        )
        references = evaluator._all_semantic_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)

        def judge_fn(messages):
            frame = next(
                frame
                for frame in evaluator.FRAME_NAMES
                if f"Frame: {frame}" in messages[0]["content"]
            )
            return json.dumps(
                assessment_payload(
                    frame,
                    reference_items=references,
                    candidate_items=candidates,
                ),
                ensure_ascii=False,
            )

        evaluator.run_judge_phase(
            snapshot,
            judge_fn=judge_fn,
            range_signal_fn=hybrid_range_signals,
            coverage_signal_fn=lambda _packet: {
                "atomic_claim_signals": "malformed"
            },
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
            checkpoint=lambda: None,
            max_attempts=2,
            selected_case_ids={case["case_id"]},
        )

        semantic = case["semantic_evaluation"]
        frame_state = semantic["frames"][evaluator.FRAME_NAMES[0]]
        self.assertEqual(frame_state["status"], "error")
        self.assertIsNone(semantic["aggregate"])
        self.assertTrue(all(
            attempt["error"]["stage"] == "coverage_signal_validation"
            for attempt in frame_state["attempts"]
        ))
        self.assertEqual(
            case["phase_errors"]["judge"]["stage"],
            "coverage_signal_validation",
        )
        self.assertEqual(snapshot["summary"]["unjudgeable_cases"], 0)
        self.assertEqual(
            snapshot["summary"]["phases"]["judge"]["error_cases"],
            1,
        )

    def test_range_guard_contract_error_remains_phase_error(self) -> None:
        snapshot, case = self.one_case_snapshot(
            self.range_contrast_question,
            measure_range=[51, 58],
        )
        case["generated_answer"] = {
            "answer": "ff와 pp를 두 덩어리로 대비해 메아리를 표현한다.",
            **grounded_answer(),
        }
        packet = evaluator._reference_packet(
            snapshot["results"][0],
            case,
        )
        references = evaluator._all_semantic_reference_items(packet)
        candidates = evaluator._candidate_answer_items(packet)

        def judge_fn(messages):
            for frame in evaluator.FRAME_NAMES:
                if f"Frame: {frame}" in messages[0]["content"]:
                    return json.dumps(
                        assessment_payload(
                            frame,
                            reference_items=references,
                            candidate_items=candidates,
                        ),
                        ensure_ascii=False,
                    )
            return "{}"

        evaluator.run_judge_phase(
            snapshot,
            judge_fn=judge_fn,
            range_signal_fn=lambda _packet: {
                "excluded_claim_signals": "malformed"
            },
            coverage_signal_fn=atomic_coverage_signals,
            nli_threshold=0.8,
            embedding_delta_threshold=0.05,
            checkpoint=lambda: None,
            max_attempts=2,
            selected_case_ids={case["case_id"]},
        )

        semantic = case["semantic_evaluation"]
        self.assertTrue(all(
            state["status"] == "complete"
            for state in semantic["frames"].values()
        ))
        self.assertEqual(semantic["range_scope"]["status"], "error")
        self.assertIsNone(semantic["aggregate"])
        self.assertTrue(all(
            attempt["error"]["stage"] == "range_signal_validation"
            for attempt in semantic["range_scope"]["attempts"]
        ))
        self.assertEqual(
            case["phase_errors"]["judge"]["stage"],
            "range_signal_validation",
        )
        self.assertEqual(snapshot["summary"]["unjudgeable_cases"], 0)


class CalibrationGateShapeTests(unittest.TestCase):
    def test_metadata_only_or_zero_control_artifact_cannot_bypass_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            evaluator.atomic_write_json(
                path,
                {
                    "schema_version": evaluator.CALIBRATION_SCHEMA_VERSION,
                    "run": {"status": "complete"},
                    "summary": {
                        "quality_gate_passed": True,
                        "total_controls": 0,
                        "completed_controls": 0,
                        "matched_controls": 0,
                        "required_matches": 0,
                        "accuracy": 1.0,
                        "mismatched_control_ids": [],
                        "frame_errors": 0,
                    },
                    "controls": [],
                },
            )
            with self.assertRaises(evaluator.EvaluationInputError):
                evaluator.validate_judge_calibration(
                    path,
                    judge_model={
                        "path": str(
                            Path("/definitely/missing/judge-model")
                        ),
                        "sha256": "forged",
                    },
                    range_guard=TEST_RANGE_GUARD,
                    dataset_root=DATASET_ROOT,
                    judge_max_tokens=1536,
                )


class SummaryDiagnosticsTests(unittest.TestCase):
    def test_summary_persists_per_piece_case_and_question_rates(
        self,
    ) -> None:
        statuses = {
            "die-forelle": ("pass", True),
            "in-flowery-clouds": ("human_review", False),
            "la-capinera": ("fail", False),
        }
        questions = []
        for piece_id, (status, reliable) in statuses.items():
            questions.append(
                {
                    "piece_id": piece_id,
                    "inference_runs": [
                        {
                            "case_id": f"{piece_id}__fixture",
                            "retrieval_probe": None,
                            "generated_answer": None,
                            "semantic_evaluation": {
                                "aggregate": {
                                    "status": status,
                                    "reliable_rag_llm_pass": reliable,
                                }
                            },
                            "phase_errors": {},
                        }
                    ],
                }
            )
        snapshot = {
            "run": {
                "status": "in_progress",
                "updated_at": None,
                "finished_at": None,
                "phases": {
                    phase: {
                        "status": "pending",
                        "completed_cases": 0,
                        "error_cases": 0,
                        "finished_at": None,
                    }
                    for phase in ("retrieval", "generate", "judge")
                },
            },
            "results": questions,
            "summary": {},
        }
        evaluator.refresh_summary(snapshot)

        primary = snapshot["summary"]["primary_metric"]["by_piece"]
        semantic = snapshot["summary"]["semantic_judge"]["by_piece"]
        self.assertEqual(
            primary["die-forelle"]["question_pass_rate"],
            1.0,
        )
        self.assertEqual(
            primary["in-flowery-clouds"]["question_pass_rate"],
            0.0,
        )
        self.assertEqual(
            semantic["la-capinera"]["case_statuses"],
            {"fail": 1},
        )
        self.assertEqual(
            semantic["la-capinera"]["question_statuses"],
            {"fail": 1},
        )
        self.assertEqual(
            semantic["la-capinera"]["case_pass_rate"],
            0.0,
        )
        self.assertEqual(
            semantic["la-capinera"]["question_pass_rate"],
            0.0,
        )

    def test_partial_run_never_reports_completed_only_rate_as_headline(
        self,
    ) -> None:
        cases = [
            {
                "case_id": "done",
                "retrieval_probe": None,
                "generated_answer": None,
                "semantic_evaluation": {
                    "aggregate": {
                        "status": "pass",
                        "answer_quality_status": "pass",
                        "reliable_rag_llm_pass": True,
                        "answer_quality_rag_llm_pass": True,
                    }
                },
                "phase_errors": {},
            },
            {
                "case_id": "errored",
                "retrieval_probe": None,
                "generated_answer": None,
                "semantic_evaluation": None,
                "phase_errors": {"judge": {"message": "fixture"}},
            },
        ]
        snapshot = {
            "run": {
                "status": "in_progress",
                "updated_at": None,
                "finished_at": None,
                "phases": {
                    phase: {
                        "status": "pending",
                        "completed_cases": 0,
                        "error_cases": 0,
                        "finished_at": None,
                    }
                    for phase in ("retrieval", "generate", "judge")
                },
            },
            "results": [
                {
                    "piece_id": "die-forelle",
                    "inference_runs": cases,
                }
            ],
            "summary": {},
        }
        evaluator.refresh_summary(snapshot)
        for metric_name in ("primary_metric", "answer_quality_metric"):
            metric = snapshot["summary"][metric_name]
            self.assertIsNone(metric["case_pass_rate"])
            self.assertEqual(metric["completed_case_pass_rate"], 1.0)
            self.assertEqual(metric["end_to_end_case_pass_rate"], 0.5)


class AtomicCheckpointTests(unittest.TestCase):
    def test_atomic_write_produces_complete_json_and_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            evaluator.atomic_write_json(path, {"value": "처음"})
            evaluator.atomic_write_json(path, {"value": "완료"})
            self.assertEqual(
                evaluator.load_json(path),
                {"value": "완료"},
            )
            self.assertEqual(
                [item.name for item in path.parent.iterdir()],
                ["result.json"],
            )


if __name__ == "__main__":
    unittest.main()
