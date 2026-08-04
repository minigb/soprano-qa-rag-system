"""Focused tests for qualitative answer-fidelity judgment."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from evaluation import judge_qualitative as judge


TEST_CANDIDATE_SPAN = "생성 답변의 달라진 핵심 주장"
TEST_REFERENCE_SPAN = "최종 지식 단위의 핵심 주장"
TEST_VALIDATION_CONTEXT = {
    "candidate_answer": TEST_CANDIDATE_SPAN,
    "reference_texts_by_id": {"R001": TEST_REFERENCE_SPAN},
}
TEST_EMBEDDING_MODEL_PATH = "/models/qwen3-embedding.gguf"
TEST_GENERATION_MODEL_PATH = "/models/qwen3-answer.gguf"


def source_model_record(*, generation: bool) -> dict:
    return {
        "path": (
            TEST_GENERATION_MODEL_PATH
            if generation
            else TEST_EMBEDDING_MODEL_PATH
        ),
        "checkpoint_exists": True,
        "kind": "file",
        "backend": (
            "llama-cpp-python" if generation else "llama-cpp-embedding"
        ),
        "sha256": ("a" if generation else "b") * 64,
    }


def current_pipeline_result(
    *,
    piece_id: str,
    measure_range: list[int] | None,
    answer: str,
    evidence: list[dict],
) -> dict:
    return {
        "piece_id": piece_id,
        "scope": "range" if measure_range is not None else "whole_piece",
        "measure_range": deepcopy(measure_range),
        "answer": answer,
        "generation_mode": "expert_verbatim",
        "answer_basis": "authoritative_expert",
        "unavailable_reason": None,
        "has_primary_grounding": True,
        "has_selected_range_grounding": (
            True if measure_range is not None else None
        ),
        "has_confirmed_local_examples": measure_range is not None,
        "pipeline": "soprano_qa",
        "model": {
            "path": TEST_GENERATION_MODEL_PATH,
            "checkpoint_exists": True,
            "backend": "llama-cpp-python",
        },
        "retrieval": {
            "configured_mode": "hybrid",
            "active_mode": "hybrid",
            "dense_available": True,
            "embedding_model": TEST_EMBEDDING_MODEL_PATH,
            "embedding_dimension": 1024,
            "last_dense_error": None,
            "last_search_mode": "hybrid",
            "query_route": "hybrid",
            "route_reason": "dense_result_returned",
            "ambiguous_candidates": [],
            "dense_attempted": True,
            "dense_contributed": True,
        },
        "evidence": deepcopy(evidence),
        "evidence_notices": [],
    }


def judgment_response(
    verdict: str = "pass",
    *,
    drift_type: str = "predicate",
    direct_answer: bool | None = None,
    faithful: bool | None = None,
) -> str:
    if verdict == "pass":
        payload = {
            "verdict": "pass",
            "direct_answer": True,
            "faithful_to_authority": True,
            "drift_types": [],
            "material_differences": [],
            "reason": "질문의 핵심을 최종 지식 단위와 같은 뜻으로 답했습니다.",
        }
    else:
        is_direct = (
            direct_answer
            if direct_answer is not None
            else drift_type != "missing_core_meaning"
        )
        payload = {
            "verdict": verdict,
            "direct_answer": is_direct,
            "faithful_to_authority": (
                faithful if faithful is not None else False
            ),
            "drift_types": [drift_type],
            "material_differences": [
                {
                    "candidate_span": (
                        None
                        if drift_type == "missing_core_meaning"
                        else TEST_CANDIDATE_SPAN
                    ),
                    "reference_id": (
                        None
                        if drift_type in {
                            "unsupported_addition",
                            "presentation_scope",
                        }
                        else "R001"
                    ),
                    "reference_span": (
                        None
                        if drift_type in {
                            "unsupported_addition",
                            "presentation_scope",
                        }
                        else TEST_REFERENCE_SPAN
                    ),
                    "drift_type": drift_type,
                    "explanation": "행위 또는 의미 관계가 달라졌습니다.",
                }
            ],
            "reason": "생성 답변의 핵심 의미 관계가 달라졌습니다.",
        }
    return json.dumps(payload, ensure_ascii=False)


def codex_runtime_fixture() -> dict:
    return {
        "backend": "codex-cli",
        "cli": {
            "path": "/opt/codex/bin/codex",
            "version": "codex-cli 0.test",
            "sha256": "codex-binary",
            "login_status": "Logged in using ChatGPT",
        },
        "model": {
            "slug": "configured-frontier-model",
            "selection_source": "verified_user_config_highest_priority",
            "catalog": {
                "slug": "configured-frontier-model",
                "priority": 1,
            },
            "catalog_entry_sha256": "catalog-entry",
        },
        "settings": {
            "reasoning_effort": "ultra",
            "service_tier": "priority",
            "timeout_seconds": 600,
        },
        "invocation": {
            "strict_config": True,
            "config_overrides": list(judge.CODEX_STATIC_CONFIG_OVERRIDES),
            "disabled_features": list(
                judge.CODEX_REQUIRED_DISABLED_FEATURES
            ),
            "known_deprecated_inert_features": list(
                judge.CODEX_KNOWN_DEPRECATED_INERT_FEATURES
            ),
            "exec_flags": list(judge.CODEX_EXEC_FLAGS),
        },
        "feature_policy": {
            "inventory_sha256": "features",
            "tool_feature_inventory": {
                feature: {"stage": "stable", "enabled": False}
                for feature in judge.CODEX_REQUIRED_DISABLED_FEATURES
            },
            "deprecated_inert_feature_inventory": {
                feature: {
                    "base": {"stage": "deprecated", "enabled": False},
                    "effective": {
                        "stage": "deprecated",
                        "enabled": False,
                    },
                }
                for feature in judge.CODEX_KNOWN_DEPRECATED_INERT_FEATURES
            },
            "all_required_features_effectively_disabled": True,
            "all_known_deprecated_features_inert": True,
        },
        "configuration": {
            "path": "/source/codex/config.toml",
            "routing_state": {"present_keys": [], "sha256": "routing"},
            "exec_ignores_user_config": True,
        },
        "authentication": {
            "source_path": "/source/codex/auth.json",
            "isolated_copy_only": True,
        },
        "selection_sha256": "selection",
        "environment": {
            "python": {
                "conda_environment": "soprano-qa",
                "conda_prefix": "/envs/soprano-qa",
                "python_executable": "/envs/soprano-qa/bin/python",
            },
            "codex_subprocess_policy": deepcopy(
                judge.CODEX_ENVIRONMENT_POLICY
            ),
        },
        "tool_policy": "disabled_and_event_audited",
        "session_persistence": "ephemeral",
    }


def feature_inventory_text(*, effective_disabled: bool) -> str:
    active = [
        f"{feature:<36} stable             "
        f"{'false' if effective_disabled else 'true'}"
        for feature in judge.CODEX_REQUIRED_DISABLED_FEATURES
    ]
    deprecated = [
        f"{feature:<36} deprecated         false"
        for feature in judge.CODEX_KNOWN_DEPRECATED_INERT_FEATURES
    ]
    return "\n".join([*active, *deprecated])


def completed_source_artifact() -> dict:
    return {
        "artifact_type": judge.SOURCE_ARTIFACT_TYPE,
        "schema_version": judge.SOURCE_SCHEMA_VERSION,
        "run": {
            "status": "completed",
            "generate": True,
            "target_pieces": list(judge.PIECE_IDS),
            "retrieval_embedding_model": source_model_record(
                generation=False
            ),
            "generation_model": source_model_record(generation=True),
        },
        "questions": [
            {
                "source_id": "kim-die-forelle-01",
                "piece_id": "die-forelle",
                "original_question": "원래 질문은 무엇인가?",
                "paraphrased_question": "이 부분은 어떻게 표현할까?",
                "reference_material": {
                    "source_answer": "RAW 검토 필요함 원문 답변이다.",
                    "linked_knowledge_units": [
                        {
                            "knowledge_unit_id": "die-forelle-ku-001",
                            "source_ids": ["kim-die-forelle-01"],
                            "answer": "고음이 과도하게 강조되지 않게 유연하게 노래한다.",
                            "rewrite_status": "ready",
                            "rewrite_notes": "RAW rewrite conflict",
                            "measure_status": "specific",
                            "measure_ranges": [[2, 5]],
                            "measure_notes": "RAW measure conflict",
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
                            "semantic_required_any_knowledge_unit_ids": [],
                            "semantic_supporting_knowledge_unit_ids": [],
                            "case_reference_authority": {
                                "source": "applicable_finalized_knowledge_units",
                                "items": [
                                    {
                                        "reference_id": "R001",
                                        "text": "고음이 과도하게 강조되지 않게 유연하게 노래한다.",
                                        "scope": "curated_knowledge_unit_answer",
                                        "flags": [],
                                        "scope_authority": "finalized_knowledge_unit",
                                        "reference_role": "required_any",
                                        "measure_status": "specific",
                                        "measure_ranges": [[2, 5]],
                                        "source_sentence_index": None,
                                        "source_claim_index": None,
                                        "knowledge_unit_id": "die-forelle-ku-001",
                                    }
                                ],
                                "manual_review_required": False,
                                "manual_review_reason": None,
                            },
                        },
                        "expected_retrieval": {
                            "source_id": "kim-die-forelle-01",
                            "linked_knowledge_unit_ids": [
                                "die-forelle-ku-001"
                            ],
                            "range_applicable_knowledge_unit_ids": [
                                "die-forelle-ku-001"
                            ],
                            "semantic_required_any_knowledge_unit_ids": [],
                            "semantic_supporting_knowledge_unit_ids": [],
                        },
                        "pipeline_result": current_pipeline_result(
                            piece_id="die-forelle",
                            measure_range=[2, 5],
                            answer=(
                                "2~5마디에서는 고음을 과장하지 않고 "
                                "유연하게 노래한다."
                            ),
                            evidence=[
                                {
                                    "id": "retrieval-secret-id",
                                    "kind": "expert",
                                    "text": "추가 전문가 근거는 프레이즈를 유연하게 잇는다.",
                                    "measure_ranges": [[2, 5]],
                                    "scope_match": "overlaps_query_range",
                                }
                            ],
                        ),
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
                    "source_answer": "RAW 충돌이 남은 원문 답변이다.",
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
                            "semantic_required_any_knowledge_unit_ids": [],
                            "semantic_supporting_knowledge_unit_ids": [],
                            "case_reference_authority": {
                                "source": "applicable_finalized_knowledge_units",
                                "items": [
                                    {
                                        "reference_id": "R001",
                                        "text": "호흡을 유지해 긴 프레이즈를 연결한다.",
                                        "scope": "curated_knowledge_unit_answer",
                                        "flags": [],
                                        "scope_authority": "finalized_knowledge_unit",
                                        "reference_role": "required_any",
                                        "measure_status": "whole_piece",
                                        "measure_ranges": [],
                                        "source_sentence_index": None,
                                        "source_claim_index": None,
                                        "knowledge_unit_id": "la-capinera-ku-001",
                                    }
                                ],
                                "manual_review_required": False,
                                "manual_review_reason": None,
                            },
                        },
                        "expected_retrieval": {
                            "source_id": "kim-la-capinera-01",
                            "linked_knowledge_unit_ids": [
                                "la-capinera-ku-001"
                            ],
                            "range_applicable_knowledge_unit_ids": [
                                "la-capinera-ku-001"
                            ],
                            "semantic_required_any_knowledge_unit_ids": [],
                            "semantic_supporting_knowledge_unit_ids": [],
                        },
                        "pipeline_result": current_pipeline_result(
                            piece_id="la-capinera",
                            measure_range=None,
                            answer=(
                                "호흡을 유지하면서 긴 프레이즈를 연결한다."
                            ),
                            evidence=[
                                {
                                    "id": "la-capinera-ku-001",
                                    "kind": "expert",
                                }
                            ],
                        ),
                        "retrieval_diagnostics": {
                            "source_retrieved": False,
                        },
                    }
                ],
            },
        ],
    }


def completed_all_piece_cohort_artifact() -> dict:
    source = completed_source_artifact()
    template = source["questions"][0]
    for piece_id in judge.PIECE_IDS:
        question = deepcopy(template)
        source_id = f"cohort-{piece_id}-01"
        unit_id = f"{piece_id}-ku-cohort"
        question.update(
            {
                "source_id": source_id,
                "piece_id": piece_id,
                "original_question": f"{piece_id} 원래 질문은 무엇인가?",
                "paraphrased_question": f"{piece_id} 국소 조언을 어떻게 적용할까?",
            }
        )
        unit = question["reference_material"]["linked_knowledge_units"][0]
        unit.update(
            {
                "knowledge_unit_id": unit_id,
                "source_ids": [source_id],
                "answer": f"{piece_id}의 해당 부분에서 호흡을 유연하게 잇는다.",
                "measure_status": "specific",
                "measure_ranges": [[2, 5]],
            }
        )
        case = question["inference_runs"][0]
        case["case_id"] = f"{source_id}__no-range"
        case["inference_input"].update(
            {
                "piece_id": piece_id,
                "question": question["paraphrased_question"],
                "measure_range": None,
                "evaluation_cohort": (
                    judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
                ),
                "semantic_required_any_knowledge_unit_ids": [unit_id],
                "semantic_supporting_knowledge_unit_ids": [],
            }
        )
        authority_item = case["inference_input"][
            "case_reference_authority"
        ]["items"][0]
        authority_item.update(
            {
                "knowledge_unit_id": unit_id,
                "text": unit["answer"],
                "reference_role": "required_any",
                "measure_status": "specific",
                "measure_ranges": [[2, 5]],
            }
        )
        case["expected_retrieval"].update(
            {
                "source_id": source_id,
                "linked_knowledge_unit_ids": [unit_id],
                "range_applicable_knowledge_unit_ids": [unit_id],
                "semantic_required_any_knowledge_unit_ids": [unit_id],
                "semantic_supporting_knowledge_unit_ids": [],
            }
        )
        case["pipeline_result"]["answer"] = (
            f"{piece_id}의 해당 부분에서는 호흡을 유연하게 이으시면 됩니다."
        )
        case["pipeline_result"].update(
            {
                "piece_id": piece_id,
                "scope": "whole_piece",
                "measure_range": None,
                "has_selected_range_grounding": None,
            }
        )
        source["questions"].append(question)
    return source


def completed_local_required_with_supporting_artifact() -> dict:
    """Build one no-range semantic case with distinct reference roles."""

    source = completed_source_artifact()
    question = source["questions"][0]
    required = question["reference_material"]["linked_knowledge_units"][0]
    supporting = {
        "knowledge_unit_id": "die-forelle-ku-supporting",
        "source_ids": [question["source_id"]],
        "answer": "피아노 반주는 곡의 전반적인 분위기를 형성한다.",
        "rewrite_status": "ready",
        "rewrite_notes": "",
        "measure_status": "whole_piece",
        "measure_ranges": [],
        "measure_notes": "",
    }
    question["reference_material"]["linked_knowledge_units"].append(
        supporting
    )
    case = question["inference_runs"][0]
    case["case_id"] = "kim-die-forelle-01__no-range"
    case["inference_input"].update(
        {
            "question": "고음 구간에서는 어떻게 노래해야 할까?",
            "measure_range": None,
            "evaluation_cohort": (
                judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
            ),
            "semantic_required_any_knowledge_unit_ids": [
                required["knowledge_unit_id"]
            ],
            "semantic_supporting_knowledge_unit_ids": [
                supporting["knowledge_unit_id"]
            ],
        }
    )
    authority = case["inference_input"]["case_reference_authority"]
    authority["items"][0].update(
        {
            "reference_role": "required_any",
            "measure_status": "specific",
            "measure_ranges": [[2, 5]],
        }
    )
    authority["items"].append(
        {
            "reference_id": "R002",
            "text": supporting["answer"],
            "scope": "curated_knowledge_unit_answer",
            "flags": [],
            "scope_authority": "finalized_knowledge_unit",
            "reference_role": "supporting",
            "measure_status": "whole_piece",
            "measure_ranges": [],
            "source_sentence_index": None,
            "source_claim_index": None,
            "knowledge_unit_id": supporting["knowledge_unit_id"],
        }
    )
    case["expected_retrieval"].update(
        {
            "linked_knowledge_unit_ids": [
                required["knowledge_unit_id"],
                supporting["knowledge_unit_id"],
            ],
            "range_applicable_knowledge_unit_ids": [
                required["knowledge_unit_id"],
                supporting["knowledge_unit_id"],
            ],
            "semantic_required_any_knowledge_unit_ids": [
                required["knowledge_unit_id"]
            ],
            "semantic_supporting_knowledge_unit_ids": [
                supporting["knowledge_unit_id"]
            ],
        }
    )
    case["pipeline_result"]["answer"] = (
        "해당 고음 구간에서는 소리를 과장하지 않고 유연하게 "
        "노래하시면 됩니다."
    )
    case["pipeline_result"].update(
        {
            "scope": "whole_piece",
            "measure_range": None,
            "has_selected_range_grounding": None,
        }
    )
    return source


def new_snapshot(
    source: dict | None = None,
    *,
    cohort: str | None = None,
) -> dict:
    source = source or completed_source_artifact()
    runtime = codex_runtime_fixture()
    source_judgment_sha256 = judge.source_judgment_input_sha256(
        source,
        cohort=cohort,
    )
    cases = judge.build_judgment_cases(source, cohort=cohort)
    selection = judge.judgment_selection_record(
        source,
        cohort=cohort,
        cases=cases,
    )
    input_fingerprint = judge.build_input_fingerprint(
        source_judgment_input_sha256=source_judgment_sha256,
        selection=selection,
        judge_runtime=runtime,
        implementation_sha256="implementation",
    )
    return judge.new_snapshot(
        source=source,
        source_path=Path("/evaluation/qualitative.json"),
        source_sha256="source-artifact",
        source_judgment_sha256=source_judgment_sha256,
        judge_runtime=runtime,
        input_fingerprint=input_fingerprint,
        implementation_sha256="implementation",
        cohort=cohort,
    )


def resume_validation_kwargs(
    snapshot: dict,
    *,
    source: dict | None = None,
    cohort: str | None = None,
) -> dict:
    return {
        "source": source or completed_source_artifact(),
        "judge_runtime": snapshot["run"]["judge_runtime"],
        "implementation_sha256": "implementation",
        "cohort": cohort,
    }


class QualitativeJudgeTests(unittest.TestCase):
    def test_prompt_uses_expert_references_but_not_retrieval_exactness(self):
        snapshot = new_snapshot()
        case = snapshot["cases"][0]

        serialized_case = json.dumps(case, ensure_ascii=False)
        messages = judge.build_judge_messages(case)
        serialized_messages = json.dumps(messages, ensure_ascii=False)

        self.assertIn("고음이 과도하게 강조되지 않게", serialized_messages)
        self.assertIn("R001", serialized_messages)
        self.assertNotIn("die-forelle-ku-001", serialized_messages)
        self.assertIn("2~5마디에서는", serialized_messages)
        self.assertEqual(
            json.loads(messages[1]["content"])["selected_measure_range"],
            [2, 5],
        )
        self.assertEqual(
            json.loads(messages[1]["content"])[
                "case_reference_authority"
            ]["source"],
            "applicable_finalized_knowledge_units",
        )
        no_range_messages = judge.build_judge_messages(snapshot["cases"][1])
        self.assertIn(
            "호흡을 유지해 긴 프레이즈를 연결한다",
            no_range_messages[1]["content"],
        )
        self.assertNotIn("RAW 검토 필요함", serialized_case)
        self.assertNotIn("RAW 검토 필요함", serialized_messages)
        self.assertNotIn("RAW 충돌", no_range_messages[1]["content"])
        self.assertNotIn("rewrite_notes", serialized_case)
        self.assertNotIn("measure_notes", serialized_case)
        self.assertNotIn("RAW rewrite conflict", serialized_messages)
        self.assertNotIn("RAW measure conflict", serialized_messages)
        self.assertNotIn("retrieval-secret-id", serialized_case)
        self.assertNotIn("retrieval-secret-id", serialized_messages)
        self.assertNotIn("retrieval_diagnostics", serialized_messages)
        self.assertNotIn("추가 전문가 근거", serialized_case)
        self.assertNotIn("추가 전문가 근거", serialized_messages)
        self.assertIn("검색 성공 여부", messages[0]["content"])
        self.assertIn(
            "selected_measure_range가 null",
            no_range_messages[0]["content"],
        )
        self.assertIn(
            "곡 전체·전 구간·항상·자주·많이",
            no_range_messages[0]["content"],
        )
        self.assertIn("원자 명제", messages[0]["content"])
        self.assertIn("agent_or_role", messages[0]["content"])
        self.assertIn("predicate", messages[0]["content"])
        self.assertIn("causal", messages[0]["content"])
        self.assertIn("modality", messages[0]["content"])
        self.assertIn("negation", messages[0]["content"])
        self.assertIn("scope_or_quantifier", messages[0]["content"])
        self.assertIn("unsupported_addition", messages[0]["content"])
        self.assertIn("missing_core_meaning", messages[0]["content"])

        mutated_case = deepcopy(case)
        mutated_case["case_reference_authority"]["source_answer"] = (
            "RAW source answer"
        )
        mutated_case["case_reference_authority"]["items"][0][
            "rewrite_notes"
        ] = "RAW rewrite note"
        mutated_case["case_reference_authority"]["items"][0][
            "measure_notes"
        ] = "RAW measure note"
        hardened_messages = json.dumps(
            judge.build_judge_messages(mutated_case),
            ensure_ascii=False,
        )
        self.assertNotIn("RAW source answer", hardened_messages)
        self.assertNotIn("RAW rewrite note", hardened_messages)
        self.assertNotIn("RAW measure note", hardened_messages)

    def test_no_range_local_judgment_contract_is_scope_and_role_aware(self):
        snapshot = new_snapshot(
            completed_local_required_with_supporting_artifact()
        )
        case = snapshot["cases"][0]
        messages = judge.build_judge_messages(case)
        payload = json.loads(messages[1]["content"])
        items = payload["case_reference_authority"]["items"]
        contract = payload["judgment_contract"]

        self.assertEqual(
            [
                {
                    "reference_id": item["reference_id"],
                    "reference_role": item["reference_role"],
                    "measure_status": item["measure_status"],
                    "measure_ranges": item["measure_ranges"],
                }
                for item in items
            ],
            [
                {
                    "reference_id": "R001",
                    "reference_role": "required_any",
                    "measure_status": "specific",
                    "measure_ranges": [[2, 5]],
                },
                {
                    "reference_id": "R002",
                    "reference_role": "supporting",
                    "measure_status": "whole_piece",
                    "measure_ranges": [],
                },
            ],
        )
        self.assertEqual(contract["required_any_reference_ids"], ["R001"])
        self.assertEqual(contract["supporting_reference_ids"], ["R002"])
        self.assertEqual(
            contract["supporting_only_without_required"],
            "fail",
        )
        local_contract = contract["no_selected_range_local_scope"]
        self.assertTrue(local_contract["active"])
        self.assertEqual(
            local_contract["canonical_ranges_by_reference_id"],
            {"R001": [[2, 5]]},
        )
        self.assertEqual(
            local_contract["canonical_range_visibility"],
            "judge_only_hidden_from_user",
        )
        self.assertFalse(
            local_contract["candidate_must_name_canonical_range"]
        )
        self.assertEqual(
            local_contract["naturally_localized_direct_required_answer"],
            "pass_eligible",
        )
        self.assertEqual(
            local_contract["unrequested_measure_locator"],
            "fail",
        )
        self.assertEqual(
            local_contract["unrequested_measure_locator_failure_type"],
            "presentation_scope",
        )
        self.assertTrue(
            local_contract[
                "correct_hidden_canonical_range_is_not_an_exception"
            ]
        )
        self.assertEqual(
            local_contract["wrong_or_fabricated_range"],
            "fail",
        )
        self.assertEqual(
            local_contract["whole_piece_or_frequency_overgeneralization"],
            "fail",
        )
        rubric = messages[0]["content"]
        self.assertIn("supporting은 이를 보완할 뿐이며 supporting", rubric)
        self.assertIn("required_any 핵심이 없으면 fail", rubric)
        self.assertIn("specific 지식 단위의 신뢰도나 우선순위를 낮추지", rubric)
        self.assertIn("심사자만 보는 내부 범위 메타데이터", rubric)
        self.assertIn("후보는 숫자 마디를 말할 필요가 없다", rubric)
        self.assertIn("'해당 부분', '이 대목'처럼 자연스럽게", rubric)
        self.assertIn("내부 확정 범위와 정확히", rubric)
        self.assertIn("presentation_scope 실패", rubric)
        self.assertIn("다른 범위나 존재하지 않는", rubric)
        self.assertIn("곡 전체·전 구간·항상·자주·많이", rubric)
        self.assertNotIn(
            "measure_ranges 중 적어도 하나에 명시적으로 한정",
            rubric,
        )
        self.assertEqual(
            payload["generated_answer"],
            "해당 고음 구간에서는 소리를 과장하지 않고 유연하게 "
            "노래하시면 됩니다.",
        )

    def test_hidden_scope_policy_covers_unrequested_and_fabricated_ranges(self):
        snapshot = new_snapshot(
            completed_local_required_with_supporting_artifact()
        )
        base_case = snapshot["cases"][0]
        candidates = {
            "correct_but_unrequested": (
                "2~5마디에서는 고음을 과장하지 않고 유연하게 노래합니다."
            ),
            "fabricated": (
                "99마디에서는 고음을 과장하지 않고 유연하게 노래합니다."
            ),
            "whole_piece_frequency": (
                "이 곡에는 이런 고음이 자주 나오므로 항상 유연하게 노래합니다."
            ),
        }

        for label, candidate in candidates.items():
            with self.subTest(label=label):
                case = deepcopy(base_case)
                case["candidate_answer"] = candidate
                messages = judge.build_judge_messages(case)
                payload = json.loads(messages[1]["content"])
                contract = payload["judgment_contract"][
                    "no_selected_range_local_scope"
                ]

                self.assertIsNone(payload["selected_measure_range"])
                self.assertEqual(
                    contract["canonical_ranges_by_reference_id"],
                    {"R001": [[2, 5]]},
                )
                self.assertEqual(
                    contract["unrequested_measure_locator"],
                    "fail",
                )
                self.assertTrue(
                    contract[
                        "correct_hidden_canonical_range_is_not_an_exception"
                    ]
                )
                self.assertEqual(
                    contract["wrong_or_fabricated_range"],
                    "fail",
                )
                self.assertEqual(
                    contract["whole_piece_or_frequency_overgeneralization"],
                    "fail",
                )
                self.assertEqual(payload["generated_answer"], candidate)

    def test_cohort_snapshot_filters_at_construction_and_records_selection(self):
        source = completed_all_piece_cohort_artifact()
        cohort = judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT

        snapshot = new_snapshot(source, cohort=cohort)

        self.assertEqual(snapshot["schema_version"], "2.1")
        self.assertEqual(
            snapshot["protocol"]["version"],
            "qualitative-answer-fidelity-v9-codex-cli",
        )
        self.assertEqual(
            snapshot["protocol"][
                "no_range_canonical_measure_scope_visibility"
            ],
            "judge_only",
        )
        self.assertFalse(
            snapshot["protocol"][
                "no_range_candidate_measure_locator_required"
            ]
        )
        self.assertEqual(
            snapshot["protocol"]["codex_invocations_per_case"],
            "exactly_one",
        )
        self.assertEqual(
            snapshot["protocol"]["failed_or_interrupted_case_policy"],
            "new_output_required",
        )
        self.assertEqual(len(source["questions"]), 7)
        self.assertEqual(len(snapshot["cases"]), 5)
        self.assertEqual(
            [case["piece_id"] for case in snapshot["cases"]],
            list(judge.PIECE_IDS),
        )
        self.assertTrue(
            all(
                case["evaluation_cohort"] == cohort
                for case in snapshot["cases"]
            )
        )
        self.assertTrue(
            all(case["measure_range"] is None for case in snapshot["cases"])
        )
        self.assertEqual(
            snapshot["run"]["selection"],
            {
                "mode": "cohort",
                "cohort": cohort,
                "source_case_count": 7,
                "selected_case_count": 5,
                "selected_piece_ids": list(judge.PIECE_IDS),
                "required_piece_ids": list(judge.PIECE_IDS),
                "all_required_pieces_present": True,
            },
        )

        attempted = judge.run_judgments(
            snapshot,
            judge_fn=lambda _messages: judgment_response(),
            checkpoint=lambda: judge.refresh_summary(snapshot),
        )
        judge.refresh_summary(snapshot)

        self.assertEqual(attempted, 5)
        self.assertEqual(snapshot["run"]["status"], "completed")
        self.assertEqual(snapshot["summary"]["final_case_pass_rate"], 1.0)

    def test_unfiltered_snapshot_retains_every_source_case(self):
        source = completed_all_piece_cohort_artifact()

        snapshot = new_snapshot(source)

        self.assertEqual(len(snapshot["cases"]), 7)
        self.assertEqual(
            snapshot["run"]["selection"],
            {
                "mode": "all_cases",
                "cohort": None,
                "source_case_count": 7,
                "selected_case_count": 7,
                "selected_piece_ids": list(judge.PIECE_IDS),
                "required_piece_ids": [],
                "all_required_pieces_present": True,
            },
        )

    def test_source_requires_current_schema_generation_and_model_records(self):
        judge.validate_source_artifact(completed_source_artifact())

        defects = []
        stale_schema = completed_source_artifact()
        stale_schema["schema_version"] = "1.1"
        defects.append((stale_schema, "schema is not current"))

        generation_disabled = completed_source_artifact()
        generation_disabled["run"]["generate"] = False
        defects.append((generation_disabled, "mandatory RAG\+LLM generation"))

        incomplete_piece_set = completed_source_artifact()
        incomplete_piece_set["run"]["target_pieces"] = ["die-forelle"]
        defects.append((incomplete_piece_set, "canonical five-piece set"))

        missing_generation_model = completed_source_artifact()
        del missing_generation_model["run"]["generation_model"]
        defects.append((missing_generation_model, "generation_model is missing"))

        wrong_generation_backend = completed_source_artifact()
        wrong_generation_backend["run"]["generation_model"]["backend"] = (
            "qwen-local"
        )
        defects.append(
            (wrong_generation_backend, "backend must be llama-cpp-python")
        )

        missing_embedding_checkpoint = completed_source_artifact()
        missing_embedding_checkpoint["run"]["retrieval_embedding_model"][
            "checkpoint_exists"
        ] = False
        defects.append(
            (missing_embedding_checkpoint, "available model file")
        )

        invalid_model_hash = completed_source_artifact()
        invalid_model_hash["run"]["generation_model"]["sha256"] = "old"
        defects.append((invalid_model_hash, "must be a SHA-256 digest"))

        removed_run_option = completed_source_artifact()
        removed_run_option["run"]["allow_internal_knowledge"] = True
        defects.append((removed_run_option, "removed allow_internal_knowledge"))

        for source, expected in defects:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                judge.QualitativeJudgeInputError,
                expected,
            ):
                judge.validate_source_artifact(source)

    def test_source_completed_responses_require_current_fail_closed_contract(self):
        mutations = {
            "pipeline": lambda response: response.update(
                {"pipeline": "legacy_pipeline"}
            ),
            "answer_path": lambda response: response.update(
                {
                    "generation_mode": "retrieval_only",
                    "answer_basis": "retrieved_evidence",
                }
            ),
            "removed_generation_field": lambda response: response.update(
                {"generation_fallback_reason": None}
            ),
            "removed_retrieval_field": lambda response: response[
                "retrieval"
            ].update({"fallback_used": False}),
            "configured_mode": lambda response: response["retrieval"].update(
                {"configured_mode": "lexical"}
            ),
            "active_mode": lambda response: response["retrieval"].update(
                {"active_mode": "lexical"}
            ),
            "dense_unavailable": lambda response: response[
                "retrieval"
            ].update({"dense_available": False}),
            "dense_error": lambda response: response["retrieval"].update(
                {"last_dense_error": "embedding failed"}
            ),
            "embedding_identity": lambda response: response[
                "retrieval"
            ].update({"embedding_model": "/models/other.gguf"}),
            "invalid_search_mode": lambda response: response[
                "retrieval"
            ].update({"last_search_mode": "lexical"}),
            "route_mismatch": lambda response: response["retrieval"].update(
                {"query_route": "lexical"}
            ),
            "dense_state": lambda response: response["retrieval"].update(
                {"dense_attempted": "yes"}
            ),
            "hybrid_not_attempted": lambda response: response[
                "retrieval"
            ].update({"dense_attempted": False}),
            "missing_route_reason": lambda response: response[
                "retrieval"
            ].update({"route_reason": None}),
            "invalid_ambiguity_diagnostics": lambda response: response[
                "retrieval"
            ].update({"ambiguous_candidates": {}}),
            "legacy_model_status": lambda response: response["model"].update(
                {"llama_cpp_available": True}
            ),
            "generation_backend": lambda response: response["model"].update(
                {"backend": "qwen-local"}
            ),
            "generation_checkpoint": lambda response: response[
                "model"
            ].update({"checkpoint_exists": False}),
            "generation_identity": lambda response: response["model"].update(
                {"path": "/models/other.gguf"}
            ),
        }

        for label, mutate in mutations.items():
            source = completed_source_artifact()
            response = source["questions"][0]["inference_runs"][0][
                "pipeline_result"
            ]
            mutate(response)
            with self.subTest(label=label), self.assertRaises(
                judge.QualitativeJudgeInputError
            ):
                judge.validate_source_artifact(source)

    def test_explicit_lexical_query_route_is_valid_inside_hybrid_retrieval(self):
        source = completed_source_artifact()
        retrieval = source["questions"][0]["inference_runs"][0][
            "pipeline_result"
        ]["retrieval"]
        retrieval.update(
            {
                "last_search_mode": "lexical_route",
                "query_route": "lexical",
                "route_reason": "broad_guidance_query",
                "dense_attempted": False,
                "dense_contributed": False,
            }
        )

        judge.validate_source_artifact(source)

    def test_cohort_validation_checks_completed_cases_outside_selection(self):
        source = completed_all_piece_cohort_artifact()
        unselected_response = source["questions"][1]["inference_runs"][0][
            "pipeline_result"
        ]
        unselected_response["retrieval"]["active_mode"] = "lexical"

        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "active retrieval mode must be hybrid",
        ):
            judge.build_judgment_cases(
                source,
                cohort=judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT,
            )

    def test_cohort_selection_fails_closed_for_invalid_coverage_or_scope(self):
        cohort = judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT

        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "has no cases",
        ):
            judge.build_judgment_cases(
                completed_source_artifact(),
                cohort=cohort,
            )

        missing_piece = completed_all_piece_cohort_artifact()
        missing_piece["questions"] = [
            question
            for question in missing_piece["questions"]
            if not (
                question["piece_id"] == "nella-fantasia"
                and question["inference_runs"][0]["inference_input"].get(
                    "evaluation_cohort"
                )
                == cohort
            )
        ]
        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "must cover all five pieces.*nella-fantasia",
        ):
            judge.build_judgment_cases(missing_piece, cohort=cohort)

        ranged_case = completed_all_piece_cohort_artifact()
        ranged_case["questions"][2]["inference_runs"][0]["inference_input"][
            "measure_range"
        ] = [2, 5]
        ranged_case["questions"][2]["inference_runs"][0][
            "pipeline_result"
        ].update(
            {
                "scope": "range",
                "measure_range": [2, 5],
                "has_selected_range_grounding": True,
            }
        )
        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "must not select a measure range",
        ):
            judge.build_judgment_cases(ranged_case, cohort=cohort)

    def test_rejects_invalid_or_forged_case_authority(self):
        for defect in (
            "missing",
            "empty",
            "nonfinal_label",
            "forged_text",
            "linked_nonfinal",
            "forged_measure_status",
            "forged_measure_ranges",
            "forged_reference_role",
            "inconsistent_role_partition",
        ):
            source = completed_source_artifact()
            inference_input = source["questions"][0]["inference_runs"][0][
                "inference_input"
            ]
            if defect == "missing":
                del inference_input["case_reference_authority"]
                expected = "case_reference_authority is missing"
            elif defect == "empty":
                inference_input["case_reference_authority"]["items"][0][
                    "text"
                ] = " "
                expected = "has no answer"
            elif defect == "nonfinal_label":
                inference_input["case_reference_authority"]["items"][0][
                    "scope_authority"
                ] = "provisional"
                expected = "is not finalized"
            elif defect == "forged_text":
                inference_input["case_reference_authority"]["items"][0][
                    "text"
                ] = "위조된 최종 답변이다."
                expected = "does not match linked knowledge unit"
            elif defect == "linked_nonfinal":
                source["questions"][0]["reference_material"][
                    "linked_knowledge_units"
                ][0]["rewrite_status"] = "needs_review"
                expected = "linked knowledge unit .* is not finalized"
            elif defect == "forged_measure_status":
                inference_input["case_reference_authority"]["items"][0][
                    "measure_status"
                ] = "whole_piece"
                expected = "scope metadata does not match linked knowledge unit"
            elif defect == "forged_measure_ranges":
                inference_input["case_reference_authority"]["items"][0][
                    "measure_ranges"
                ] = [[90, 99]]
                expected = "scope metadata does not match linked knowledge unit"
            elif defect == "forged_reference_role":
                inference_input["case_reference_authority"]["items"][0][
                    "reference_role"
                ] = "supporting"
                expected = "invalid or inconsistent reference role"
            else:
                inference_input[
                    "semantic_required_any_knowledge_unit_ids"
                ] = ["die-forelle-ku-001"]
                source["questions"][0]["inference_runs"][0][
                    "expected_retrieval"
                ]["semantic_required_any_knowledge_unit_ids"] = [
                    "die-forelle-ku-001"
                ]
                expected = "semantic reference roles require"

            with self.subTest(defect=defect), self.assertRaisesRegex(
                judge.QualitativeJudgeInputError,
                expected,
            ):
                judge.build_judgment_cases(source)

    def test_response_validation_accepts_three_verdicts_and_is_strict(self):
        for verdict in judge.VERDICTS:
            with self.subTest(verdict=verdict):
                drift_type = "ambiguity" if verdict == "review" else "predicate"
                validated = judge.validate_judge_response(
                    judgment_response(verdict, drift_type=drift_type),
                    **TEST_VALIDATION_CONTEXT,
                )
                self.assertEqual(validated["verdict"], verdict)
                self.assertIsInstance(validated["direct_answer"], bool)
                self.assertIsInstance(
                    validated["faithful_to_authority"], bool
                )

        invalid_responses = (
            "```json\n" + judgment_response() + "\n```",
            '{"verdict":"automatic_pass","reason":"좋다"}',
            json.dumps(
                {
                    **json.loads(judgment_response()),
                    "score": 1,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    **json.loads(judgment_response()),
                    "reason": "가" * 501,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    **json.loads(judgment_response()),
                    "drift_types": ["predicate"],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    **json.loads(judgment_response("fail")),
                    "material_differences": [],
                },
                ensure_ascii=False,
            ),
            "not json",
        )
        for raw in invalid_responses:
            with self.subTest(raw=raw), self.assertRaises(
                judge.QualitativeJudgeResponseError
            ):
                judge.validate_judge_response(
                    raw,
                    **TEST_VALIDATION_CONTEXT,
                )

    def test_every_semantic_drift_type_is_structured_and_validated(self):
        schema_types = set(
            judge.codex_judge_output_schema()["properties"]["drift_types"][
                "items"
            ]["enum"]
        )
        self.assertEqual(schema_types, set(judge.DRIFT_TYPES))
        for drift_type in judge.DRIFT_TYPES:
            with self.subTest(drift_type=drift_type):
                verdict = "review" if drift_type == "ambiguity" else "fail"
                assessment = judge.validate_judge_response(
                    judgment_response(verdict, drift_type=drift_type),
                    **TEST_VALIDATION_CONTEXT,
                )
                self.assertEqual(assessment["drift_types"], [drift_type])

    def test_material_semantic_drift_cannot_be_review_or_faithful(self):
        for drift_type in judge.FAIL_ONLY_DRIFT_TYPES:
            base = json.loads(
                judgment_response("fail", drift_type=drift_type)
            )
            for mutation in (
                {"verdict": "review", "faithful_to_authority": False},
                {"verdict": "fail", "faithful_to_authority": True},
            ):
                payload = {**base, **mutation}
                with self.subTest(
                    drift_type=drift_type,
                    mutation=mutation,
                ), self.assertRaises(judge.QualitativeJudgeResponseError):
                    judge.validate_judge_response(
                        json.dumps(payload, ensure_ascii=False),
                        **TEST_VALIDATION_CONTEXT,
                    )

        for drift_type in judge.REVIEW_ELIGIBLE_DRIFT_TYPES:
            raw = judgment_response("review", drift_type=drift_type)
            assessment = judge.validate_judge_response(
                raw,
                **TEST_VALIDATION_CONTEXT,
            )
            self.assertFalse(assessment["faithful_to_authority"])
            forged = json.loads(raw)
            forged["faithful_to_authority"] = True
            with self.assertRaises(judge.QualitativeJudgeResponseError):
                judge.validate_judge_response(
                    json.dumps(forged, ensure_ascii=False),
                    **TEST_VALIDATION_CONTEXT,
                )

        with self.assertRaisesRegex(
            judge.QualitativeJudgeResponseError,
            "Fail requires at least one material failure drift type",
        ):
            judge.validate_judge_response(
                judgment_response("fail", drift_type="ambiguity"),
                **TEST_VALIDATION_CONTEXT,
            )

    def test_difference_spans_and_null_rules_are_fail_closed(self):
        valid_question_mismatch = judgment_response(
            "fail",
            drift_type="question_mismatch",
            direct_answer=False,
        )
        judge.validate_judge_response(
            valid_question_mismatch,
            **TEST_VALIDATION_CONTEXT,
        )

        invalid_payloads = []
        forged_candidate = json.loads(judgment_response("fail"))
        forged_candidate["material_differences"][0]["candidate_span"] = (
            "candidate words not supplied"
        )
        invalid_payloads.append(forged_candidate)
        forged_reference = json.loads(judgment_response("fail"))
        forged_reference["material_differences"][0]["reference_span"] = (
            "reference words not supplied"
        )
        invalid_payloads.append(forged_reference)
        addition_with_reference = json.loads(
            judgment_response("fail", drift_type="unsupported_addition")
        )
        addition_with_reference["material_differences"][0].update(
            {
                "reference_id": "R001",
                "reference_span": TEST_REFERENCE_SPAN,
            }
        )
        invalid_payloads.append(addition_with_reference)
        omission_with_candidate = json.loads(
            judgment_response("fail", drift_type="missing_core_meaning")
        )
        omission_with_candidate["material_differences"][0][
            "candidate_span"
        ] = TEST_CANDIDATE_SPAN
        invalid_payloads.append(omission_with_candidate)
        missing_reference_pair = json.loads(judgment_response("fail"))
        missing_reference_pair["material_differences"][0][
            "reference_span"
        ] = None
        invalid_payloads.append(missing_reference_pair)

        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(
                judge.QualitativeJudgeResponseError
            ):
                judge.validate_judge_response(
                    json.dumps(payload, ensure_ascii=False),
                    **TEST_VALIDATION_CONTEXT,
                )

    def test_tool_feature_preflight_rejects_gaps_and_future_features(self):
        inventory = judge._parse_codex_feature_inventory(
            feature_inventory_text(effective_disabled=False)
        )
        effective = judge._parse_codex_feature_inventory(
            feature_inventory_text(effective_disabled=True)
        )
        judge._validate_codex_tool_features(inventory, effective)

        missing = deepcopy(inventory)
        missing.pop("shell_tool")
        unknown = deepcopy(inventory)
        unknown["browser_supertool"] = {
            "stage": "under development",
            "enabled": False,
        }
        unknown_web = deepcopy(effective)
        unknown_web["web_search_future"] = {
            "stage": "experimental",
            "enabled": False,
        }
        ineffective = deepcopy(effective)
        ineffective["computer_use"]["enabled"] = True
        deprecated_missing = deepcopy(inventory)
        deprecated_missing.pop("web_search_cached")
        deprecated_wrong_stage = deepcopy(inventory)
        deprecated_wrong_stage["web_search_request"]["stage"] = "stable"
        deprecated_enabled = deepcopy(effective)
        deprecated_enabled["web_search_cached"]["enabled"] = True
        for original, disabled in (
            (missing, effective),
            (unknown, effective),
            (inventory, unknown_web),
            (inventory, ineffective),
            (deprecated_missing, effective),
            (deprecated_wrong_stage, effective),
            (inventory, deprecated_enabled),
        ):
            with self.subTest(original=original), self.assertRaises(
                judge.QualitativeJudgeBackendError
            ):
                judge._validate_codex_tool_features(original, disabled)

    def test_codex_runtime_requires_soprano_qa_environment(self):
        with (
            mock.patch.dict(
                judge.os.environ,
                {"CONDA_DEFAULT_ENV": "base", "CONDA_PREFIX": "/base"},
            ),
            self.assertRaisesRegex(RuntimeError, "soprano-qa Conda"),
        ):
            judge.codex_runtime_record(timeout_seconds=600)

    def test_codex_runtime_resolves_configured_frontier_model_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "codex"
            executable.write_bytes(b"codex-test-binary")
            config_home = root / "codex-home"
            config_home.mkdir()
            (config_home / "auth.json").write_text("{}", encoding="utf-8")
            (config_home / "config.toml").write_text(
                'model = "frontier-current"\n'
                'model_reasoning_effort = "ultra"\n'
                'service_tier = "priority"\n'
                'model_provider = "openai"\n'
                '[model_providers.private]\n'
                'name = "Private"\n'
                'base_url = "https://provider.invalid"\n'
                'env_key = "FORBIDDEN_PROVIDER_TOKEN"\n',
                encoding="utf-8",
            )
            catalog = {
                "models": [
                    {
                        "slug": "frontier-current",
                        "display_name": "Frontier",
                        "description": "Latest frontier model",
                        "priority": 1,
                        "visibility": "list",
                        "supported_reasoning_levels": [
                            {"effort": "max"},
                            {"effort": "ultra"},
                        ],
                        "service_tiers": [{"id": "priority"}],
                    },
                    {
                        "slug": "balanced",
                        "priority": 2,
                        "visibility": "list",
                        "supported_reasoning_levels": [
                            {"effort": "high"}
                        ],
                        "service_tiers": [],
                    },
                ]
            }
            calls = []

            def completed(arguments, **kwargs):
                arguments = list(arguments)
                calls.append((arguments, kwargs))
                if arguments[-1] == "--version":
                    stdout = "codex-cli 0.test\n"
                elif arguments[-2:] == ["login", "status"]:
                    stdout = "Logged in using ChatGPT\n"
                elif arguments[-2:] == ["debug", "models"]:
                    stdout = json.dumps(catalog)
                elif arguments[-2:] == ["features", "list"]:
                    stdout = feature_inventory_text(
                        effective_disabled="--disable" in arguments
                    )
                elif "exec" in arguments:
                    self.assertEqual(kwargs.get("input_text"), "")
                    self.assertEqual(kwargs.get("expected_returncode"), 1)
                    self.assertEqual(
                        kwargs.get("required_output"),
                        "No prompt provided via stdin.",
                    )
                    return subprocess.CompletedProcess(
                        arguments,
                        1,
                        "Reading prompt from stdin...\n",
                        "No prompt provided via stdin.\n",
                    )
                else:
                    raise AssertionError(arguments)
                return subprocess.CompletedProcess(arguments, 0, stdout, "")

            with (
                mock.patch.dict(
                    judge.os.environ,
                    {
                        "CONDA_DEFAULT_ENV": "soprano-qa",
                        "CONDA_PREFIX": judge.sys.prefix,
                        "CODEX_HOME": str(config_home),
                        "OPENAI_API_KEY": "must-not-be-inherited",
                        "HTTPS_PROXY": "must-not-be-inherited",
                        "FORBIDDEN_PROVIDER_TOKEN": "must-not-be-inherited",
                    },
                ),
                mock.patch.object(judge.shutil, "which", return_value=str(executable)),
                mock.patch.object(
                    judge,
                    "_completed_command",
                    side_effect=completed,
                ),
            ):
                runtime = judge.codex_runtime_record(timeout_seconds=321)

        self.assertEqual(runtime["backend"], "codex-cli")
        self.assertEqual(runtime["model"]["slug"], "frontier-current")
        self.assertEqual(
            runtime["model"]["selection_source"],
            "verified_user_config_highest_priority",
        )
        self.assertEqual(runtime["settings"]["reasoning_effort"], "ultra")
        self.assertEqual(runtime["settings"]["service_tier"], "priority")
        self.assertEqual(runtime["settings"]["timeout_seconds"], 321)
        self.assertEqual(runtime["tool_policy"], "disabled_and_event_audited")
        self.assertEqual(runtime["session_persistence"], "ephemeral")
        self.assertTrue(
            runtime["feature_policy"][
                "all_required_features_effectively_disabled"
            ]
        )
        self.assertTrue(
            runtime["feature_policy"][
                "all_known_deprecated_features_inert"
            ]
        )
        self.assertEqual(
            runtime["invocation"]["known_deprecated_inert_features"],
            list(judge.CODEX_KNOWN_DEPRECATED_INERT_FEATURES),
        )
        self.assertEqual(
            set(
                runtime["feature_policy"][
                    "deprecated_inert_feature_inventory"
                ]
            ),
            set(judge.CODEX_KNOWN_DEPRECATED_INERT_FEATURES),
        )
        self.assertEqual(
            runtime["configuration"]["routing_state"]["present_keys"],
            [
                "model",
                "model_provider",
                "model_providers",
                "model_reasoning_effort",
                "service_tier",
            ],
        )
        expected_environment_keys = {
            *judge.CODEX_MINIMAL_ENVIRONMENT,
            "HOME",
            "CODEX_HOME",
            "TMPDIR",
        }
        for _arguments, keyword_arguments in calls:
            environment = keyword_arguments["environment"]
            self.assertEqual(set(environment), expected_environment_keys)
            self.assertNotIn("OPENAI_API_KEY", environment)
            self.assertNotIn("HTTPS_PROXY", environment)
            self.assertNotIn("FORBIDDEN_PROVIDER_TOKEN", environment)
            disabled_features = [
                _arguments[index + 1]
                for index, argument in enumerate(_arguments[:-1])
                if argument == "--disable"
            ]
            for feature in judge.CODEX_KNOWN_DEPRECATED_INERT_FEATURES:
                self.assertNotIn(feature, disabled_features)

    def test_codex_model_selection_requires_configured_best_model_and_effort(self):
        models = [
            {
                "slug": "older-model",
                "priority": 8,
                "visibility": "list",
                "supported_reasoning_levels": [
                    {"effort": "high"},
                    {"effort": "ultra"},
                ],
                "service_tiers": [{"id": "priority"}],
            },
            {
                "slug": "future-frontier-name",
                "priority": 1,
                "visibility": "list",
                "supported_reasoning_levels": [
                    {"effort": "max"},
                    {"effort": "ultra"},
                ],
                "service_tiers": [{"id": "priority"}],
            },
        ]
        selected, effort, tier, source = judge._select_codex_model(
            models,
            {
                "model": "future-frontier-name",
                "reasoning_effort": "ultra",
                "service_tier": "priority",
            },
        )

        self.assertEqual(selected["slug"], "future-frontier-name")
        self.assertEqual(effort, "ultra")
        self.assertEqual(tier, "priority")
        self.assertEqual(source, "verified_user_config_highest_priority")

        invalid_configurations = (
            (
                {
                    "model": None,
                    "reasoning_effort": "ultra",
                    "service_tier": "priority",
                },
                "explicitly configured",
            ),
            (
                {
                    "model": "missing-model",
                    "reasoning_effort": "ultra",
                    "service_tier": "priority",
                },
                "unavailable",
            ),
            (
                {
                    "model": "older-model",
                    "reasoning_effort": "ultra",
                    "service_tier": "priority",
                },
                "not the unique highest-priority",
            ),
            (
                {
                    "model": "future-frontier-name",
                    "reasoning_effort": None,
                    "service_tier": "priority",
                },
                "effort must be explicitly configured",
            ),
            (
                {
                    "model": "future-frontier-name",
                    "reasoning_effort": "max",
                    "service_tier": "priority",
                },
                "weaker than strongest",
            ),
            (
                {
                    "model": "future-frontier-name",
                    "reasoning_effort": "ultra",
                    "service_tier": None,
                },
                "service tier must be explicitly configured",
            ),
            (
                {
                    "model": "future-frontier-name",
                    "reasoning_effort": "ultra",
                    "service_tier": "unsupported",
                },
                "is not supported",
            ),
        )
        for configured, message in invalid_configurations:
            with self.subTest(configured=configured), self.assertRaisesRegex(
                judge.QualitativeJudgeBackendError,
                message,
            ):
                judge._select_codex_model(
                    models,
                    configured,
                )

    def test_codex_event_stream_rejects_tools_failures_and_malformed_events(self):
        valid = "\n".join(
            json.dumps(event)
            for event in (
                {"type": "thread.started"},
                {"type": "turn.started"},
                {"type": "item.completed", "item": {"type": "reasoning"}},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message"},
                },
                {"type": "turn.completed"},
            )
        )
        judge.validate_codex_event_stream(valid)

        invalid_streams = (
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"type": "command_execution"},
                }
            )
            + "\n"
            + json.dumps({"type": "turn.completed"}),
            json.dumps({"type": "turn.failed"}),
            json.dumps({"type": "turn.started"}),
            "not-json",
        )
        for raw in invalid_streams:
            with self.subTest(raw=raw), self.assertRaises(
                judge.QualitativeJudgeBackendError
            ):
                judge.validate_codex_event_stream(raw)

    def test_codex_event_stream_reports_bounded_sanitized_backend_error(self):
        raw = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "error",
                    "code": "server_error",
                    "message": (
                        "No capacity; Bearer secret-token "
                        "https://example.test/status?access_token=secret "
                        + "x" * 700
                    ),
                },
            }
        )
        with self.assertRaisesRegex(
            judge.QualitativeJudgeBackendError,
            "backend failure",
        ) as caught:
            judge.validate_codex_event_stream(raw)

        diagnostic = str(caught.exception)
        self.assertIn("code=server_error", diagnostic)
        self.assertIn("message=No capacity", diagnostic)
        self.assertIn("Bearer <redacted>", diagnostic)
        self.assertIn("?<redacted-query>", diagnostic)
        self.assertNotIn("secret-token", diagnostic)
        self.assertNotIn("access_token=secret", diagnostic)
        self.assertLessEqual(
            len(diagnostic),
            judge.MAX_CODEX_DIAGNOSTIC_CHARACTERS + 80,
        )

    def test_codex_event_stream_reports_top_level_failure_details(self):
        raw = json.dumps(
            {
                "type": "turn.failed",
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "message": "You've hit your usage limit.",
                },
            }
        )
        with self.assertRaises(judge.QualitativeJudgeBackendError) as caught:
            judge.validate_codex_event_stream(raw)

        diagnostic = str(caught.exception)
        self.assertIn("error.type=rate_limit_error", diagnostic)
        self.assertIn("error.code=rate_limit_exceeded", diagnostic)
        self.assertIn("You've hit your usage limit.", diagnostic)
        self.assertNotIn("forbidden tool", diagnostic)

    def test_codex_event_stream_keeps_tool_attempt_diagnostic_distinct(self):
        raw = json.dumps(
            {
                "type": "item.started",
                "item": {"type": "command_execution"},
            }
        )
        with self.assertRaisesRegex(
            judge.QualitativeJudgeBackendError,
            "forbidden tool or item: 'command_execution'",
        ):
            judge.validate_codex_event_stream(raw)

    def test_invalid_input_is_rejected_before_model_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "invalid.json"
            input_path.write_text(
                '{"artifact_type":"wrong","questions":[]}',
                encoding="utf-8",
            )
            output_path = root / "output.json"
            with (
                mock.patch.object(judge, "codex_runtime_record") as preflight,
                self.assertRaises(judge.QualitativeJudgeInputError),
            ):
                judge.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )

            preflight.assert_not_called()
            self.assertFalse(output_path.exists())

    def test_main_codex_preflight_failure_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "qualitative.json"
            input_path.write_text(
                json.dumps(completed_source_artifact(), ensure_ascii=False),
                encoding="utf-8",
            )
            output_path = root / "qualitative_judged.json"
            with (
                mock.patch.object(
                    judge,
                    "codex_runtime_record",
                    side_effect=judge.QualitativeJudgeBackendError(
                        "Codex unavailable"
                    ),
                ),
                self.assertRaises(judge.QualitativeJudgeBackendError),
            ):
                judge.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertFalse(output_path.exists())
            self.assertFalse(
                output_path.with_suffix(".json.lock").exists()
            )

    def test_codex_judge_uses_isolated_structured_read_only_invocation(self):
        messages = [
            {"role": "system", "content": "rubric"},
            {"role": "user", "content": "case"},
        ]
        runtime = codex_runtime_fixture()
        captured = {}
        with tempfile.TemporaryDirectory() as temporary:
            codex_home = Path(temporary) / "source-codex-home"
            codex_home.mkdir()
            (codex_home / "auth.json").write_text("{}", encoding="utf-8")
            runtime["authentication"]["source_path"] = str(
                codex_home / "auth.json"
            )

            def completed(arguments, **kwargs):
                captured["arguments"] = list(arguments)
                captured.update(kwargs)
                isolated_auth = (
                    Path(kwargs["environment"]["CODEX_HOME"]) / "auth.json"
                )
                captured["auth_is_symlink"] = isolated_auth.is_symlink()
                captured["auth_mode"] = isolated_auth.stat().st_mode & 0o777
                captured["isolated_auth_before"] = isolated_auth.read_text(
                    encoding="utf-8"
                )
                isolated_auth.write_text(
                    '{"refreshed":"temporary-only"}',
                    encoding="utf-8",
                )
                result_index = arguments.index("--output-last-message") + 1
                Path(arguments[result_index]).write_text(
                    judgment_response(),
                    encoding="utf-8",
                )
                events = "\n".join(
                    (
                        json.dumps({"type": "thread.started"}),
                        json.dumps({"type": "turn.started"}),
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {"type": "agent_message"},
                            }
                        ),
                        json.dumps({"type": "turn.completed"}),
                    )
                )
                return subprocess.CompletedProcess(arguments, 0, events, "")

            with (
                mock.patch.dict(
                    judge.os.environ,
                    {
                        "CODEX_HOME": str(codex_home),
                        "OPENAI_API_KEY": "must-not-be-inherited",
                        "HTTPS_PROXY": "must-not-be-inherited",
                        "CUSTOM_MODEL_PROVIDER": "must-not-be-inherited",
                    },
                ),
                mock.patch.object(
                    judge,
                    "_completed_command",
                    side_effect=completed,
                ),
            ):
                raw = judge.build_codex_judge(runtime)(messages)
                captured["source_auth_after"] = (
                    codex_home / "auth.json"
                ).read_text(encoding="utf-8")

        self.assertEqual(
            judge.validate_judge_response(
                raw,
                candidate_answer="candidate",
                reference_texts_by_id={"R001": "reference"},
            )["verdict"],
            "pass",
        )
        arguments = captured["arguments"]
        self.assertIn("--strict-config", arguments)
        config_overrides = [
            arguments[index + 1]
            for index, argument in enumerate(arguments[:-1])
            if argument == "-c"
        ]
        self.assertIn('service_tier="priority"', config_overrides)
        provider = judge.CODEX_JUDGE_PROVIDER_ID
        self.assertIn(f'model_provider="{provider}"', config_overrides)
        self.assertIn(
            f"model_providers.{provider}.requires_openai_auth=true",
            config_overrides,
        )
        self.assertIn(
            f'model_providers.{provider}.wire_api="responses"',
            config_overrides,
        )
        self.assertIn(
            f"model_providers.{provider}.supports_websockets=false",
            config_overrides,
        )
        self.assertIn(
            f"model_providers.{provider}.request_max_retries=0",
            config_overrides,
        )
        self.assertIn(
            f"model_providers.{provider}.stream_max_retries=0",
            config_overrides,
        )
        self.assertNotIn("request_max_retries=0", config_overrides)
        self.assertNotIn("stream_max_retries=0", config_overrides)
        disabled_features = [
            arguments[index + 1]
            for index, argument in enumerate(arguments[:-1])
            if argument == "--disable"
        ]
        self.assertEqual(
            disabled_features,
            list(judge.CODEX_REQUIRED_DISABLED_FEATURES),
        )
        for feature in judge.CODEX_KNOWN_DEPRECATED_INERT_FEATURES:
            self.assertNotIn(feature, disabled_features)
        self.assertIn("--sandbox", arguments)
        self.assertIn("read-only", arguments)
        self.assertIn("--ephemeral", arguments)
        self.assertIn("--ignore-user-config", arguments)
        self.assertIn("--ignore-rules", arguments)
        self.assertIn("--output-schema", arguments)
        self.assertIn("--output-last-message", arguments)
        self.assertIn("--cd", arguments)
        self.assertIn("--json", arguments)
        self.assertEqual(arguments[-1], "-")
        provenance = runtime["invocation"]["exec_flags"]
        self.assertEqual(provenance, list(judge.CODEX_EXEC_FLAGS))
        self.assertIn("--cd", provenance)
        self.assertIn("<temporary-directory>", provenance)
        self.assertIn("--output-schema", provenance)
        self.assertIn("--output-last-message", provenance)
        self.assertEqual(provenance[-1], "-")
        for fictional_flag in (
            "--stdin-prompt",
            "--structured-output",
            "--empty-temporary-cwd",
        ):
            self.assertNotIn(fictional_flag, provenance)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", arguments)
        self.assertIn("<judge_instructions>", captured["input_text"])
        self.assertNotEqual(
            captured["environment"]["CODEX_HOME"],
            str(codex_home),
        )
        self.assertNotEqual(
            captured["environment"]["HOME"],
            str(Path.home()),
        )
        self.assertEqual(
            set(captured["environment"]),
            {
                *judge.CODEX_MINIMAL_ENVIRONMENT,
                "HOME",
                "CODEX_HOME",
                "TMPDIR",
            },
        )
        self.assertNotIn("OPENAI_API_KEY", captured["environment"])
        self.assertNotIn("HTTPS_PROXY", captured["environment"])
        self.assertNotIn("CUSTOM_MODEL_PROVIDER", captured["environment"])
        self.assertFalse(captured["auth_is_symlink"])
        self.assertEqual(captured["auth_mode"], 0o600)
        self.assertEqual(captured["isolated_auth_before"], "{}")
        self.assertEqual(captured["source_auth_after"], "{}")

    def test_codex_judge_rejects_any_other_backend(self):
        invalid_runtimes = (
            {**codex_runtime_fixture(), "backend": "qwen"},
        )
        for runtime in invalid_runtimes:
            with self.subTest(runtime=runtime), self.assertRaisesRegex(
                judge.QualitativeJudgeBackendError,
                "codex-cli as its backend",
            ):
                judge.build_codex_judge(runtime)

    def test_backend_failure_checkpoints_once_and_aborts_all_cases(self):
        snapshot = new_snapshot()
        checkpoints = []

        with self.assertRaisesRegex(
            judge.QualitativeJudgeBackendError,
            "inference unavailable",
        ):
            judge.run_judgments(
                snapshot,
                judge_fn=lambda _messages: (_ for _ in ()).throw(
                    judge.QualitativeJudgeBackendError(
                        "inference unavailable"
                    )
                ),
                checkpoint=lambda: checkpoints.append(True),
            )

        self.assertEqual(
            snapshot["cases"][0]["judgment"]["attempt_count"],
            1,
        )
        self.assertEqual(snapshot["cases"][1]["judgment"]["status"], "pending")
        self.assertEqual(len(checkpoints), 2)

    def test_unexpected_error_checkpoints_once_and_aborts_all_cases(self):
        snapshot = new_snapshot()
        checkpoints = []

        with self.assertRaisesRegex(KeyError, "broken packet"):
            judge.run_judgments(
                snapshot,
                judge_fn=lambda _messages: (_ for _ in ()).throw(
                    KeyError("broken packet")
                ),
                checkpoint=lambda: checkpoints.append(True),
            )

        self.assertEqual(
            snapshot["cases"][0]["judgment"]["attempt_count"],
            1,
        )
        self.assertEqual(snapshot["cases"][1]["judgment"]["status"], "pending")
        self.assertEqual(len(checkpoints), 2)

    def test_case_authority_excludes_other_range_and_editorial_material(self):
        source = completed_source_artifact()
        other = deepcopy(
            source["questions"][0]["reference_material"][
                "linked_knowledge_units"
            ][0]
        )
        other["knowledge_unit_id"] = "die-forelle-ku-other"
        other["answer"] = "다른 구간에서만 적용되는 조언이다."
        other["measure_ranges"] = [[40, 42]]
        other["rewrite_notes"] = "검토 필요함"
        source["questions"][0]["reference_material"][
            "linked_knowledge_units"
        ].append(other)
        snapshot = new_snapshot(source)
        case = snapshot["cases"][0]

        payload = json.loads(judge.build_judge_messages(case)[1]["content"])

        self.assertEqual(
            [
                item["reference_id"]
                for item in payload["case_reference_authority"]["items"]
            ],
            ["R001"],
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("다른 구간에서만", serialized)
        self.assertNotIn("검토 필요함", serialized)

    def test_malformed_output_aborts_once_and_cannot_be_resumed(self):
        snapshot = new_snapshot()
        checkpoints = []

        def checkpoint() -> None:
            judge.refresh_summary(snapshot)
            checkpoints.append(deepcopy(snapshot["summary"]))

        with self.assertRaises(judge.QualitativeJudgeResponseError):
            judge.run_judgments(
                snapshot,
                judge_fn=lambda _messages: "malformed",
                checkpoint=checkpoint,
                limit=1,
            )

        first = snapshot["cases"][0]["judgment"]
        second = snapshot["cases"][1]["judgment"]
        self.assertEqual(first["status"], "error")
        self.assertEqual(first["attempt_count"], 1)
        self.assertFalse(first["attempts"][0]["valid"])
        self.assertEqual(second["status"], "pending")
        self.assertEqual(len(checkpoints), 2)
        self.assertIsNone(snapshot["summary"]["final_case_pass_rate"])

        calls = []
        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "cannot be resumed; choose a new --output path",
        ):
            judge.run_judgments(
                snapshot,
                judge_fn=lambda messages: calls.append(messages)
                or judgment_response(),
                checkpoint=checkpoint,
            )

        self.assertEqual(calls, [])
        self.assertEqual(first["status"], "error")
        self.assertEqual(first["attempt_count"], 1)
        self.assertEqual(len(first["attempts"]), 1)
        self.assertEqual(second["status"], "pending")

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
            or judgment_response(),
            checkpoint=lambda: judge.refresh_summary(snapshot),
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

    def test_resume_rejects_changed_inputs_or_codex_configuration(self):
        source = completed_source_artifact()
        snapshot = new_snapshot(source)

        judge.validate_resume_snapshot(
            snapshot,
            **resume_validation_kwargs(snapshot, source=source),
        )
        changed_source = deepcopy(source)
        changed_source["questions"][0]["inference_runs"][0][
            "pipeline_result"
        ]["answer"] = "fresh source contains a changed candidate"
        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "choose a new --output path",
        ):
            judge.validate_resume_snapshot(
                snapshot,
                **resume_validation_kwargs(
                    snapshot,
                    source=changed_source,
                ),
            )

        stale_schema = deepcopy(snapshot)
        stale_schema["schema_version"] = "1.4"
        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "schema version is not resumable",
        ):
            judge.validate_resume_snapshot(
                stale_schema,
                **resume_validation_kwargs(stale_schema, source=source),
            )

    def test_resume_rejects_forged_or_inconsistent_judgments(self):
        completed = new_snapshot()
        judge.run_judgments(
            completed,
            judge_fn=lambda _messages: judgment_response(),
            checkpoint=lambda: judge.refresh_summary(completed),
            limit=1,
        )
        judge.validate_resume_snapshot(
            completed,
            **resume_validation_kwargs(completed),
        )

        forged_candidate = deepcopy(completed)
        forged_candidate["cases"][0]["candidate_answer"] = (
            "judged candidate was replaced while retaining the pass"
        )

        forged_runtime = deepcopy(completed)
        forged_runtime["run"]["judge_runtime"]["settings"][
            "timeout_seconds"
        ] = 1

        forged_implementation = deepcopy(completed)
        forged_implementation["run"]["implementation_sha256"] = "forged"

        forged_selection = deepcopy(completed)
        forged_selection["run"]["selection"]["selected_case_count"] = 999

        forged_source_hash = deepcopy(completed)
        forged_source_hash["run"]["source_artifact"][
            "judgment_input_sha256"
        ] = "forged"

        zero_attempt_pass = deepcopy(completed)
        zero_judgment = zero_attempt_pass["cases"][0]["judgment"]
        zero_judgment["attempt_count"] = 0
        zero_judgment["attempts"] = []

        count_mismatch = deepcopy(completed)
        count_mismatch["cases"][0]["judgment"]["attempt_count"] = 2

        forged_fields = deepcopy(completed)
        forged_fields["cases"][0]["judgment"]["reason"] = (
            "raw response에 없는 위조 판정"
        )

        forged_raw = deepcopy(completed)
        forged_raw["cases"][0]["judgment"]["attempts"][-1][
            "raw_response"
        ] = "not-json"

        bad_status = deepcopy(completed)
        bad_status["cases"][0]["judgment"]["status"] = "trusted"

        forged_run_status = deepcopy(completed)
        forged_run_status["run"]["status"] = "completed"

        forged_summary = deepcopy(completed)
        forged_summary["summary"]["final_case_pass_rate"] = 1.0

        for snapshot in (
            zero_attempt_pass,
            count_mismatch,
            forged_fields,
            forged_raw,
            bad_status,
            forged_run_status,
            forged_summary,
            forged_candidate,
            forged_runtime,
            forged_implementation,
            forged_selection,
            forged_source_hash,
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(
                judge.QualitativeJudgeInputError
            ):
                judge.validate_resume_snapshot(
                    snapshot,
                    **resume_validation_kwargs(completed),
                )
            self.assertFalse(
                judge.can_migrate_semantically_identical_resume(
                    snapshot,
                    source=completed_source_artifact(),
                    judge_runtime=completed["run"]["judge_runtime"],
                    implementation_sha256="implementation",
                )
            )

    def test_interrupted_attempt_cannot_be_resumed(self):
        snapshot = new_snapshot()
        judgment = snapshot["cases"][0]["judgment"]
        judgment.update(
            {
                "status": "running",
                "attempt_count": 1,
                "attempts": [
                    {
                        "attempt": 1,
                        "started_at": "2026-01-01T00:00:00Z",
                        "completed_at": None,
                        "raw_response": None,
                        "valid": False,
                        "error": None,
                    }
                ],
                "started_at": "2026-01-01T00:00:00Z",
            }
        )
        judge.refresh_summary(snapshot)
        judge.validate_resume_snapshot(
            snapshot,
            **resume_validation_kwargs(snapshot),
        )

        calls = []
        with self.assertRaisesRegex(
            judge.QualitativeJudgeInputError,
            "cannot be resumed; choose a new --output path",
        ):
            judge.run_judgments(
                snapshot,
                judge_fn=lambda messages: calls.append(messages)
                or judgment_response(),
                checkpoint=lambda: judge.refresh_summary(snapshot),
                limit=1,
            )

        self.assertEqual(calls, [])
        self.assertEqual(judgment["status"], "running")
        self.assertEqual(judgment["attempt_count"], 1)
        self.assertIsNone(judgment["attempts"][0]["completed_at"])

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

    def test_cohort_hash_uses_only_selected_packets_and_selection(self):
        cohort = judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
        original = completed_all_piece_cohort_artifact()
        original_hash = judge.source_judgment_input_sha256(
            original,
            cohort=cohort,
        )

        changed_unselected = deepcopy(original)
        changed_unselected["questions"][0]["inference_runs"][0][
            "pipeline_result"
        ]["answer"] = "선택되지 않은 답변의 변경"
        self.assertEqual(
            original_hash,
            judge.source_judgment_input_sha256(
                changed_unselected,
                cohort=cohort,
            ),
        )

        changed_selected = deepcopy(original)
        changed_selected["questions"][2]["inference_runs"][0][
            "pipeline_result"
        ]["answer"] = "선택된 답변의 변경"
        self.assertNotEqual(
            original_hash,
            judge.source_judgment_input_sha256(
                changed_selected,
                cohort=cohort,
            ),
        )

    def test_resume_migration_requires_identical_judge_packets(self):
        source = completed_source_artifact()
        snapshot = new_snapshot(source)
        runtime = snapshot["run"]["judge_runtime"]

        self.assertTrue(
            judge.can_migrate_semantically_identical_resume(
                snapshot,
                source=source,
                judge_runtime=runtime,
                implementation_sha256="implementation",
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
                judge_runtime=runtime,
                implementation_sha256="implementation",
            )
        )
        self.assertFalse(
            judge.can_migrate_semantically_identical_resume(
                snapshot,
                source=source,
                judge_runtime=runtime,
                implementation_sha256="changed-implementation",
            )
        )

    def test_cohort_resume_migration_requires_the_same_selection(self):
        cohort = judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
        source = completed_all_piece_cohort_artifact()
        snapshot = new_snapshot(source, cohort=cohort)
        runtime = snapshot["run"]["judge_runtime"]

        self.assertTrue(
            judge.can_migrate_semantically_identical_resume(
                snapshot,
                source=source,
                judge_runtime=runtime,
                implementation_sha256="implementation",
                cohort=cohort,
            )
        )
        self.assertFalse(
            judge.can_migrate_semantically_identical_resume(
                snapshot,
                source=source,
                judge_runtime=runtime,
                implementation_sha256="implementation",
                cohort=None,
            )
        )

        self.assertFalse(
            judge.can_migrate_semantically_identical_resume(
                snapshot,
                source=source,
                judge_runtime=runtime,
                implementation_sha256="changed-implementation",
                cohort=cohort,
            )
        )

    def test_input_fingerprint_binds_runtime_and_implementation(self):
        snapshot = new_snapshot()
        runtime = snapshot["run"]["judge_runtime"]
        selection = snapshot["run"]["selection"]
        original = judge.build_input_fingerprint(
            source_judgment_input_sha256="source",
            selection=selection,
            judge_runtime=runtime,
            implementation_sha256="implementation",
        )
        changed_runtime = deepcopy(runtime)
        changed_runtime["settings"]["top_p"] = 0.7
        changed_deprecated_policy = deepcopy(runtime)
        changed_deprecated_policy["feature_policy"][
            "deprecated_inert_feature_inventory"
        ]["web_search_cached"]["base"]["enabled"] = True

        self.assertNotEqual(
            original,
            judge.build_input_fingerprint(
                source_judgment_input_sha256="source",
                selection=selection,
                judge_runtime=changed_runtime,
                implementation_sha256="implementation",
            ),
        )
        self.assertNotEqual(
            original,
            judge.build_input_fingerprint(
                source_judgment_input_sha256="source",
                selection=selection,
                judge_runtime=changed_deprecated_policy,
                implementation_sha256="implementation",
            ),
        )
        self.assertNotEqual(
            original,
            judge.build_input_fingerprint(
                source_judgment_input_sha256="source",
                selection=selection,
                judge_runtime=runtime,
                implementation_sha256="changed",
            ),
        )
        changed_selection = deepcopy(selection)
        changed_selection["cohort"] = (
            judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
        )
        self.assertNotEqual(
            original,
            judge.build_input_fingerprint(
                source_judgment_input_sha256="source",
                selection=changed_selection,
                judge_runtime=runtime,
                implementation_sha256="implementation",
            ),
        )

    def test_default_minimum_pass_rate_is_ninety_percent(self):
        arguments = judge.parse_arguments([])
        self.assertEqual(arguments.minimum_pass_rate, 0.90)
        self.assertEqual(arguments.codex_timeout_seconds, 600)
        self.assertIsNone(arguments.cohort)
        self.assertEqual(
            judge.parse_arguments(
                [
                    "--cohort",
                    judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT,
                ]
            ).cohort,
            judge.MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT,
        )
        self.assertFalse(hasattr(arguments, "judge_backend"))
        self.assertFalse(hasattr(arguments, "judge_model_path"))


if __name__ == "__main__":
    unittest.main()
