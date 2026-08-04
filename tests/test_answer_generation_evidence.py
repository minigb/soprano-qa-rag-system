"""Focused tests for grounded-answer evidence selection and safeguards."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from typing import Any, Dict
from unittest import mock

from soprano_qa import answer as answer_cli
from soprano_qa.answer import (
    answer_overgeneralizes_local_examples,
    answer_references_secondary_evidence,
    answer_uses_formal_polite_korean,
    build_extractive_answer,
    build_messages,
    expert_prompt_factuality_material,
    GeneratedAnswerRejected,
    finalize_answer_citations,
    finalize_user_visible_answer,
    has_primary_grounding,
    is_grounded_insufficiency_answer,
    is_underspecified_ranged_query,
    select_generation_evidence,
    suspicious_generation_tokens,
)
from soprano_qa.dense import SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON
from soprano_qa.retrieval import SearchResult


def make_result(
    record_id: str,
    *,
    evidence_type: str = "expert_annotation",
    alias_score: float = 0.0,
    measure_status: str = "specific",
    scope_match: str = "overlaps_query_range",
    source_ids: list[str] | None = None,
    score: float = 1.0,
    text_score: float = 1.0,
    dense_score: float = 0.0,
    dense_content_score: float = 0.0,
    content_concept_coverage: float = 0.0,
    semantic_match_type: str = "strict",
) -> SearchResult:
    record: Dict[str, Any] = {
        "id": record_id,
        "evidence_type": evidence_type,
        "piece": "die-forelle",
        "work": "Die Forelle",
        "topic": "performance",
        "measure_range": [[2, 5]],
        "measure_scope": "local",
        "source_ids": source_ids or ["kim-die-forelle-01"],
        "annotators": ["kim"],
        "measure_status": measure_status,
        "source_answer_context": [],
        "question": "",
        "answer": "전문가 답변",
    }
    return SearchResult(
        record=record,
        score=score,
        text_score=text_score,
        measure_score=0.0,
        piece_score=1.0,
        scope_match=scope_match,
        alias_score=alias_score,
        dense_score=dense_score,
        dense_content_score=dense_content_score,
        content_concept_coverage=content_concept_coverage,
        semantic_match_type=semantic_match_type,
    )


class GenerationEvidenceSelectionTests(unittest.TestCase):
    def test_grounded_insufficiency_detector_recognizes_natural_refusals(
        self,
    ) -> None:
        refusals = (
            "제공된 근거만으로는 이 질문에 답변할 수 없습니다.",
            "관련 정보가 부족해서 정확히 알 수 없습니다.",
            "검색된 자료에 필요한 내용이 없어 판단하기 어렵습니다.",
            "관련 근거를 찾지 못해 판단할 수 없습니다.",
            "I cannot answer because the provided evidence is insufficient.",
            "There is not enough information to determine the answer.",
            "The available context is incomplete, so I cannot know this.",
            "I cannot determine this because that information is not provided.",
            "This question cannot be answered because the evidence is missing.",
        )
        for refusal in refusals:
            with self.subTest(refusal=refusal):
                self.assertTrue(is_grounded_insufficiency_answer(refusal))

    def test_grounded_insufficiency_detector_preserves_non_refusal_advice(
        self,
    ) -> None:
        answers = (
            "숨이 부족할 때는 프레이즈를 나누어 연습하는 것이 좋습니다.",
            "악보의 정보가 부족한 부분은 반주 파트를 확인하십시오.",
            "제공된 정보가 부족한 경우에는 원전 악보를 확인하십시오.",
            "가사가 없어도 음가를 정확히 알 수 있습니다.",
            "If the score lacks information, consult the accompaniment.",
            "When breath is insufficient, shorten the practice phrase.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertFalse(is_grounded_insufficiency_answer(answer))

        self.assertFalse(is_grounded_insufficiency_answer(""))
        self.assertTrue(
            is_grounded_insufficiency_answer("<NO_GROUNDED_ANSWER>")
        )

    def test_keeps_only_maximum_positive_expert_alias_matches_in_input_order(
        self,
    ) -> None:
        weaker_expert = make_result("expert-weaker", alias_score=0.5)
        strongest_first = make_result("expert-strongest-first", alias_score=1.0)
        web_with_same_score = make_result(
            "web-strongest",
            evidence_type="web_database",
            alias_score=1.0,
        )
        unrelated_expert = make_result("expert-unrelated", alias_score=0.0)
        strongest_second = make_result("expert-strongest-second", alias_score=1.0)
        results = [
            weaker_expert,
            strongest_first,
            web_with_same_score,
            unrelated_expert,
            strongest_second,
        ]

        selected = select_generation_evidence(results)

        self.assertEqual(selected, [strongest_first, strongest_second])

    def test_returns_all_results_when_no_positive_expert_alias_exists(
        self,
    ) -> None:
        results = [
            make_result("expert-no-alias", alias_score=0.0),
            make_result(
                "web-positive-alias",
                evidence_type="web_database",
                alias_score=2.0,
            ),
        ]

        self.assertEqual(select_generation_evidence(results), results)

    def test_other_range_alias_is_retained_without_displacing_overlap(
        self,
    ) -> None:
        overlap = make_result(
            "overlap",
            alias_score=0.5,
            scope_match="overlaps_query_range",
        )
        stronger_other_range = make_result(
            "other-range",
            alias_score=1.0,
            scope_match="other_range_context",
        )

        selected = select_generation_evidence(
            [overlap, stronger_other_range]
        )

        self.assertEqual(selected, [overlap, stronger_other_range])
        self.assertTrue(has_primary_grounding(selected))

    def test_no_range_without_alias_supplements_answer_semantic_experts(
        self,
    ) -> None:
        lexical_neighbor = make_result(
            "lexical-neighbor",
            alias_score=0.0,
            dense_score=0.74,
        )
        semantic_target = make_result(
            "semantic-target",
            alias_score=0.0,
            dense_score=0.80,
        )
        semantic_companion = make_result(
            "semantic-companion",
            alias_score=0.0,
            dense_score=0.73,
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [lexical_neighbor]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [
                    semantic_target,
                    lexical_neighbor,
                    semantic_companion,
                ]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="질문과 의미가 맞는 전문가 조언은 무엇인가요?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=3,
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["semantic-target", "lexical-neighbor", "semantic-companion"],
        )

    def test_no_range_semantic_focus_drops_weak_dense_fillers(self) -> None:
        weak_selected = make_result(
            "weak-selected",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.55,
        )
        semantic_target = make_result(
            "semantic-target",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.75,
        )
        unrelated = make_result(
            "unrelated",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.60,
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [weak_selected]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [semantic_target, unrelated]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="의미가 분명한 질문입니다.",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=3,
        )

        self.assertEqual(results, [semantic_target])

    def test_no_range_semantic_focus_drops_unrelated_lexical_expert(
        self,
    ) -> None:
        lexical_target = make_result(
            "lexical-target",
            alias_score=0.0,
            text_score=8.0,
            dense_score=0.50,
            source_ids=["lexical-only-source"],
            semantic_match_type="broad_guidance",
        )
        semantic_target = make_result(
            "semantic-target",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.70,
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [lexical_target]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [semantic_target]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="어휘와 의미가 함께 중요한 질문입니다.",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=3,
        )

        self.assertEqual(results, [semantic_target])

    def test_no_range_semantic_focus_preserves_same_source_lexical_facet(
        self,
    ) -> None:
        lexical_facet = make_result(
            "lexical-facet",
            alias_score=0.0,
            text_score=8.0,
            dense_score=0.40,
            source_ids=["shared-source"],
            semantic_match_type="broad_guidance",
        )
        semantic_leader = make_result(
            "semantic-leader",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.70,
            source_ids=["shared-source"],
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [lexical_facet]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [semantic_leader]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="같은 전문가 답변의 여러 조언을 함께 알려 주세요.",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=3,
        )

        self.assertEqual(results, [semantic_leader, lexical_facet])

    def test_no_range_semantic_focus_prunes_oversized_no_alias_bundle(
        self,
    ) -> None:
        lexical_anchor = make_result(
            "lexical-anchor",
            alias_score=0.0,
            text_score=7.0,
            dense_score=0.60,
        )
        useful_companion = make_result(
            "useful-companion",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.57,
            content_concept_coverage=0.5,
        )
        unrelated = [
            make_result(
                f"unrelated-{index}",
                alias_score=0.0,
                text_score=0.0,
                dense_score=score,
            )
            for index, score in enumerate((0.49, 0.46, 0.44), start=1)
        ]

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [lexical_anchor, useful_companion, *unrelated]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [lexical_anchor, useful_companion, *unrelated]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="여러 후보 중 직접 답하는 내용은 무엇인가요?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [lexical_anchor, useful_companion])

    def test_no_range_semantic_focus_prunes_noncompetitive_dense_neighbor(
        self,
    ) -> None:
        direct_answer = make_result(
            "direct-answer",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.61248,
            dense_content_score=0.605835,
            content_concept_coverage=0.2,
        )
        merely_nearby = make_result(
            "merely-nearby",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.57043,
            dense_content_score=0.567893,
            content_concept_coverage=0.0,
            source_ids=["unrelated-source"],
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [direct_answer, merely_nearby]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [direct_answer, merely_nearby]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="상황에 맞는 행동을 어떻게 이어 가야 하나요?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [direct_answer])

    def test_no_range_direct_lexical_relation_beats_topic_neighbor(
        self,
    ) -> None:
        direct_relation = make_result(
            "direct-relation",
            alias_score=0.0,
            text_score=7.0,
            dense_score=0.49,
            dense_content_score=0.47,
            content_concept_coverage=1.0,
            source_ids=["direct-source"],
            semantic_match_type="strict",
            scope_match="general_evidence",
        )
        topic_neighbor = make_result(
            "topic-neighbor",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.58,
            dense_content_score=0.48,
            content_concept_coverage=0.33,
            source_ids=["other-source"],
            semantic_match_type="dense",
            scope_match="local_example",
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [direct_relation]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [topic_neighbor, direct_relation]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="왜 앞의 박자로 되돌아왔나요?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [direct_relation])

    def test_no_range_general_leader_drops_unsupported_local_example(
        self,
    ) -> None:
        general_rule = make_result(
            "general-rule",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.62,
            dense_content_score=0.57,
            content_concept_coverage=0.0,
            source_ids=["general-source"],
            semantic_match_type="dense",
            scope_match="general_evidence",
        )
        local_example = make_result(
            "local-example",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.60,
            dense_content_score=0.58,
            content_concept_coverage=0.0,
            source_ids=["local-source"],
            semantic_match_type="dense",
            scope_match="local_example",
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [general_rule, local_example]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [general_rule, local_example]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="이 연주 요소는 어떻게 처리해야 하나요?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [general_rule])

    def test_no_range_general_leader_keeps_source_linked_local_detail(
        self,
    ) -> None:
        general_rule = make_result(
            "general-rule",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.62,
            dense_content_score=0.57,
            source_ids=["general-source"],
            semantic_match_type="dense",
            scope_match="general_evidence",
        )
        source_companion = make_result(
            "source-companion",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.60,
            dense_content_score=0.58,
            source_ids=["detail-source"],
            semantic_match_type="dense",
            scope_match="general_evidence",
        )
        local_detail = make_result(
            "local-detail",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.59,
            dense_content_score=0.57,
            source_ids=["detail-source"],
            semantic_match_type="dense",
            scope_match="local_example",
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [general_rule, source_companion, local_detail]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [general_rule, source_companion, local_detail]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="호흡을 어떻게 다루어야 하나요?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(
            results,
            [general_rule, source_companion, local_detail],
        )

    def test_no_range_broad_rescue_focuses_rejected_candidates(self) -> None:
        semantic_target = make_result(
            "semantic-target",
            alias_score=0.0,
            text_score=5.0,
            dense_score=0.72,
            content_concept_coverage=1.0,
        )
        rejected_noise = make_result(
            "rejected-noise",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.54,
            content_concept_coverage=0.0,
        )

        class Index:
            last_answer_candidates = [semantic_target, rejected_noise]

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return []

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [semantic_target, rejected_noise]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="거절 후보 중 직접 답하는 조언은 무엇인가요?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [semantic_target])

    def test_missing_causal_authority_is_not_rescued_for_generation(
        self,
    ) -> None:
        rejected = make_result(
            "noncausal-candidate",
            alias_score=1.0,
            text_score=10.0,
            dense_score=0.75,
            content_concept_coverage=1.0,
        )

        class Index:
            last_answer_candidates = [rejected]
            last_route_reason = SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return []

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                raise AssertionError(
                    "a causal hard rejection must terminate evidence rescue"
                )

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="왜 이 표시가 쓰였을까요?",
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [])

    def test_no_range_weak_expert_drops_unsupported_web_neighbor(
        self,
    ) -> None:
        expert = make_result(
            "expert",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.53,
            content_concept_coverage=0.2,
        )
        web = make_result(
            "web",
            evidence_type="web_database",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.48,
            content_concept_coverage=0.0,
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [expert, web]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [expert]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="전문가 답변이 직접 답하는 질문입니다.",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [expert])

    def test_no_range_alias_authority_skips_semantic_supplement(
        self,
    ) -> None:
        alias_match = make_result("alias-match", alias_score=1.0)

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [alias_match]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                raise AssertionError(
                    "semantic supplementation must not displace an exact alias"
                )

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="원래 주석 질문입니다.",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=3,
        )

        self.assertEqual(results, [alias_match])

    def test_no_range_source_question_authority_skips_semantic_focus(
        self,
    ) -> None:
        causal_authority = make_result(
            "una-voce-poco-fa-ku-004",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.66,
            dense_content_score=0.61,
            semantic_match_type=answer_cli.SOURCE_QUESTION_LED_MATCH_TYPE,
        )
        procedural_neighbor = make_result(
            "una-voce-poco-fa-ku-005",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.74,
            dense_content_score=0.72,
            semantic_match_type="selector_candidate",
            source_ids=["procedural-source"],
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [causal_authority]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [procedural_neighbor, causal_authority]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query=(
                "로시니는 왜 보통의 부점 리듬이 아니라 "
                "겹부점 리듬을 선택했을까?"
            ),
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [causal_authority])

    def test_no_range_focus_preserves_split_units_from_one_source_answer(
        self,
    ) -> None:
        causal = make_result(
            "causal-facet",
            alias_score=0.0,
            dense_score=0.64,
            source_ids=["shared-source"],
        )
        procedural = make_result(
            "procedural-facet",
            alias_score=0.0,
            dense_score=0.72,
            source_ids=["shared-source"],
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [procedural, causal]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [procedural, causal]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="왜 이 리듬을 썼고 어떻게 불러야 하나요?",
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [procedural, causal])

    def test_no_range_semantic_focus_requires_requested_text_slot(
        self,
    ) -> None:
        rhythm_only = make_result(
            "rhythm-only",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.75,
            dense_content_score=0.73,
            semantic_match_type="dense",
        )
        rhythm_only.record["answer"] = (
            "셋잇단음표의 리듬을 정확하게 지켜 노래한다."
        )
        placement_answer = make_result(
            "placement-answer",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.68,
            dense_content_score=0.66,
            semantic_match_type="selector_candidate",
            source_ids=["placement-source"],
        )
        placement_answer.record["answer"] = (
            "3연음보 대목에서는 첫머리에 가사를 두고 노래한다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [rhythm_only]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [rhythm_only, placement_answer]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="셋잇단음표를 부를 때 모음은 어느 음에 놓아야 할까?",
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [placement_answer])

    def test_no_range_focus_keeps_named_technique_attachment_companion(
        self,
    ) -> None:
        general_technique = make_result(
            "general-technique",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.567,
            dense_content_score=0.438,
            content_concept_coverage=0.2,
            semantic_match_type="selector_candidate",
        )
        general_technique.record["answer"] = (
            "스케일과 트릴에서는 해당 단어의 모음으로 이어 가고 "
            "프레이즈가 끝나기 전에 단어를 끝맺는다."
        )
        named_technique = make_result(
            "named-technique",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.511,
            dense_content_score=0.520,
            content_concept_coverage=0.4,
            semantic_match_type="selector_candidate",
            source_ids=["named-technique-source"],
        )
        named_technique.record["answer"] = (
            "3연음보 대목에서는 첫머리에 가사를 두고, 같은 모음을 "
            "유지하면서 음정만 바꾸어 노래한다."
        )
        unrelated = make_result(
            "unrelated-technique",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.545,
            dense_content_score=0.521,
            content_concept_coverage=0.2,
            semantic_match_type="selector_candidate",
            source_ids=["unrelated-technique-source"],
        )
        unrelated.record["answer"] = (
            "빠른 기교에서는 리듬과 음정을 정확히 연습한다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [general_technique]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [general_technique, unrelated, named_technique]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="기교나 테크닉이 나올 때 가사는 어떻게 붙여야 할까?",
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [general_technique, named_technique])

    def test_no_range_focus_keeps_distinct_coordinated_relation(
        self,
    ) -> None:
        transition = make_result(
            "transition",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.614,
            dense_content_score=0.614,
            content_concept_coverage=2.0 / 3.0,
            semantic_match_type="selector_candidate",
        )
        transition.record["answer"] = (
            "어두운 분위기로 바뀌었다가 다시 희망적으로 회복되는 "
            "전환을 표현하는 것이 중요하다."
        )
        return_guidance = make_result(
            "return-guidance",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.466,
            dense_content_score=0.466,
            content_concept_coverage=4.0 / 9.0,
            semantic_match_type="selector_candidate",
            source_ids=["return-guidance-source"],
        )
        return_guidance.record["answer"] = (
            "다시 밝아질 때에는 크레센도와 표정을 활용하고 이어지는 "
            "대목은 가볍게 부른다."
        )
        nearby_but_incomplete = make_result(
            "nearby-but-incomplete",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.534,
            dense_content_score=0.499,
            content_concept_coverage=1.0 / 9.0,
            semantic_match_type="selector_candidate",
            source_ids=["nearby-source"],
        )
        nearby_but_incomplete.record["answer"] = (
            "곡이 높으면 가창자의 음역에 맞게 조성을 바꿀 수 있다."
        )
        repeated_transition = make_result(
            "repeated-transition",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.500,
            dense_content_score=0.500,
            content_concept_coverage=0.5,
            semantic_match_type="selector_candidate",
            source_ids=["repeated-transition-source"],
        )
        repeated_transition.record["answer"] = transition.record["answer"]

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [transition, nearby_but_incomplete]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [
                    transition,
                    nearby_but_incomplete,
                    repeated_transition,
                    return_guidance,
                ]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query=(
                "어두운 분위기로 바뀐 뒤 어떻게 달라지며, "
                "가창에서는 무엇을 중요하게 여겨야 할까?"
            ),
            piece="in-flowery-clouds",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [transition, return_guidance])

    def test_coordinated_relation_detection_requires_a_clause_boundary(
        self,
    ) -> None:
        self.assertFalse(
            answer_cli._query_requests_coordinated_relations(
                "어떤 음색과 느낌으로 부르는 것이 좋을까?"
            )
        )
        self.assertFalse(
            answer_cli._query_requests_coordinated_relations(
                "고음을 효과적으로 어떻게 불러야 할까?"
            )
        )

    def test_no_range_timbre_guidance_excludes_transposition_neighbor(
        self,
    ) -> None:
        tone = make_result(
            "tone",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.565,
            dense_content_score=0.565,
            content_concept_coverage=0.5,
            semantic_match_type="selector_candidate",
        )
        tone.record["answer"] = (
            "크게 지르기보다 가볍고 깔끔하게 부르는 것이 중요하다."
        )
        transposition = make_result(
            "transposition",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.527,
            dense_content_score=0.504,
            content_concept_coverage=0.5,
            semantic_match_type="selector_candidate",
            source_ids=["transposition-source"],
        )
        transposition.record["answer"] = (
            "조성이 너무 높거나 낮으면 자신의 목소리와 음색에 맞게 "
            "조성을 바꿀 수 있다."
        )
        imagery = make_result(
            "imagery",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.495,
            dense_content_score=0.495,
            content_concept_coverage=0.5,
            semantic_match_type="selector_candidate",
            source_ids=["imagery-source"],
        )
        imagery.record["answer"] = (
            "꽃가루와 바람이 흩뿌리는 느낌을 유지해 가볍게 부른다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [tone, transposition, imagery]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [tone, transposition, imagery]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="이 곡은 어떤 음색과 느낌으로 부르는 것이 좋을까?",
            piece="in-flowery-clouds",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [tone, imagery])

    def test_broad_multifacet_focus_preserves_router_coverage(self) -> None:
        rhythm = make_result(
            "rhythm",
            measure_status="whole_piece",
            scope_match="general_evidence",
            text_score=24.0,
            dense_score=0.618,
            dense_content_score=0.618,
            content_concept_coverage=1.0,
            semantic_match_type="broad_guidance",
        )
        rhythm.record["measure_range"] = []
        rhythm.record["answer"] = (
            "부점 리듬의 음 길이와 특징이 분명하게 드러나도록 "
            "주의하여 노래해야 한다."
        )
        diction = make_result(
            "diction",
            measure_status="whole_piece",
            scope_match="general_evidence",
            text_score=18.0,
            dense_score=0.537,
            dense_content_score=0.458,
            content_concept_coverage=1.0,
            semantic_match_type="broad_guidance",
        )
        diction.record["measure_range"] = []
        diction.record["answer"] = (
            "글자를 보이는 그대로 읽기보다 발음나는 대로 적어 "
            "연습하는 것이 좋다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [rhythm, diction]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [rhythm, diction]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query=(
                "이 곡을 잘 부르려면 리듬과 발음을 어떻게 "
                "준비해야 할까?"
            ),
            piece="in-flowery-clouds",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [rhythm, diction])

    def test_broad_multifacet_focus_rejects_slotless_semantic_leader(
        self,
    ) -> None:
        tone = make_result(
            "tone",
            measure_status="whole_piece",
            scope_match="general_evidence",
            text_score=27.0,
            dense_score=0.55,
            dense_content_score=0.55,
            content_concept_coverage=1.0,
            semantic_match_type="broad_guidance",
        )
        tone.record["measure_range"] = []
        tone.record["answer"] = (
            "새소리 같은 밝은 분위기와 느낌을 잘 드러내어 "
            "표현하는 것이 좋다."
        )
        high_note = make_result(
            "high-note",
            measure_status="whole_piece",
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.62,
            dense_content_score=0.62,
            content_concept_coverage=0.0,
            semantic_match_type="selector_candidate",
            source_ids=["high-note-source"],
        )
        high_note.record["measure_range"] = []
        high_note.record["answer"] = (
            "도약하는 고음을 날카롭게 내지 않도록 강박에 무게를 "
            "두고 노래해야 한다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [tone]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [high_note, tone]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query=(
                "이 곡을 잘 부르려면 음색과 호흡을 어떻게 "
                "준비해야 할까?"
            ),
            piece="la-capinera",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [tone])

    def test_no_range_focus_uses_semantic_leader_when_no_candidate_fills_slot(
        self,
    ) -> None:
        semantic_leader = make_result(
            "semantic-leader",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.66,
            dense_content_score=0.59,
            semantic_match_type="selector_candidate",
        )
        semantic_leader.record["answer"] = (
            "가사는 자연스러운 전달을 위해 정박보다 조금 앞이나 "
            "뒤에서 발음하는 경우가 일반적이다."
        )
        unrelated = make_result(
            "unrelated",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.54,
            dense_content_score=0.52,
            semantic_match_type="selector_candidate",
            source_ids=["unrelated-source"],
        )
        unrelated.record["answer"] = (
            "마이크 유무와 공연 공간에 따라 발성을 선택한다."
        )
        unrelated_neighbors = [
            make_result(
                f"unrelated-{index}",
                alias_score=0.0,
                scope_match="general_evidence",
                text_score=0.0,
                dense_score=score,
                dense_content_score=score,
                semantic_match_type="selector_candidate",
                source_ids=[f"unrelated-source-{index}"],
            )
            for index, score in enumerate((0.50, 0.48, 0.46, 0.44), start=2)
        ]
        for neighbor in unrelated_neighbors:
            neighbor.record["answer"] = "질문과 관계없는 일반 연주 조언이다."

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [semantic_leader, unrelated, *unrelated_neighbors]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [semantic_leader, unrelated, *unrelated_neighbors]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="각 가사를 어느 음정에 붙이는 것이 좋을까요?",
            piece="nella-fantasia",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [semantic_leader])

    def test_no_range_slot_fallback_does_not_expand_small_bundle(
        self,
    ) -> None:
        selected = make_result(
            "selected",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.61,
            dense_content_score=0.58,
        )
        selected.record["answer"] = (
            "가사는 정박보다 조금 앞이나 뒤에서 발음할 수 있다."
        )
        neighbor = make_result(
            "dense-neighbor",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.60,
            dense_content_score=0.57,
            source_ids=["neighbor-source"],
        )
        neighbor.record["answer"] = "다른 발성에 관한 조언이다."

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [selected]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [selected, neighbor]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="가사를 어느 음정에 붙여야 할까요?",
            piece="nella-fantasia",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [selected])

    def test_no_range_focus_is_invariant_to_unrelated_third_candidate(
        self,
    ) -> None:
        leader = make_result(
            "leader",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.575,
            dense_content_score=0.57,
            semantic_match_type="selector_candidate",
        )
        leader.record["measure_range"] = []
        leader.record["measure_status"] = "whole_piece"
        leader.record["answer"] = "호흡을 안정적으로 유지해 노래한다."
        companion = make_result(
            "companion",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.52,
            dense_content_score=0.51,
            semantic_match_type="selector_candidate",
            source_ids=["companion-source"],
        )
        companion.record["measure_range"] = []
        companion.record["measure_status"] = "whole_piece"
        companion.record["answer"] = "질문과 관계없는 공연장 정보를 설명한다."
        unrelated = make_result(
            "unrelated-third",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.45,
            dense_content_score=0.44,
            semantic_match_type="selector_candidate",
            source_ids=["third-source"],
        )
        unrelated.record["measure_range"] = []
        unrelated.record["measure_status"] = "whole_piece"
        unrelated.record["answer"] = "작곡가의 생애를 설명한다."

        def retrieve(pool: list[SearchResult]) -> list[SearchResult]:
            class Index:
                last_answer_candidates: list[SearchResult] = []

                @staticmethod
                def search(**_kwargs: Any) -> list[SearchResult]:
                    return list(pool)

                @staticmethod
                def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                    return list(pool)

            results, _messages = answer_cli.retrieve_generation_evidence(
                Index(),
                query="호흡을 어떻게 유지해야 할까?",
                piece="die-forelle",
                measure_ranges=[],
                measures="",
                topic=None,
                top_k=6,
            )
            return results

        self.assertEqual(
            retrieve([leader, companion]),
            retrieve([leader, companion, unrelated]),
        )

    def test_broad_guidance_keeps_exact_source_semantic_sibling(self) -> None:
        leader = make_result(
            "broad-leader",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=1.0,
            dense_score=0.62,
            dense_content_score=0.60,
            semantic_match_type="broad_guidance",
            source_ids=["one-reviewed-source"],
        )
        leader.record["measure_range"] = []
        leader.record["measure_status"] = "whole_piece"
        leader.record["answer"] = "밝고 탄력 있는 음색으로 노래한다."
        sibling = make_result(
            "broad-sibling",
            alias_score=0.0,
            scope_match="general_evidence",
            text_score=0.0,
            dense_score=0.42,
            dense_content_score=0.40,
            semantic_match_type="selector_candidate",
            source_ids=["one-reviewed-source"],
        )
        sibling.record["measure_range"] = []
        sibling.record["measure_status"] = "whole_piece"
        sibling.record["answer"] = "딕션의 장단을 정확히 공부한다."

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [leader]

            @staticmethod
            def semantic_candidates(**kwargs: Any) -> list[SearchResult]:
                self.assertGreaterEqual(kwargs["top_k"], 24)
                return [leader, sibling]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="이 곡을 잘 부르려면 무엇에 신경 써야 할까?",
            piece="die-forelle",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [leader, sibling])

    def test_no_range_broad_rescue_requires_recording_score_slot(
        self,
    ) -> None:
        lyric_only = make_result(
            "lyric-only",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.76,
            dense_content_score=0.74,
            semantic_match_type="selector_candidate",
        )
        lyric_only.record["answer"] = (
            "가사 안의 단어 구분이나 배치가 원곡과 다르면 원곡의 "
            "가사를 확인하고 원곡의 단어 구분에 따라 노래한다."
        )
        recording_score_answer = make_result(
            "recording-score-answer",
            alias_score=0.0,
            scope_match="local_example",
            text_score=0.0,
            dense_score=0.69,
            dense_content_score=0.67,
            semantic_match_type="selector_candidate",
            source_ids=["recording-score-source"],
        )
        recording_score_answer.record["answer"] = (
            "음원을 악보로 옮기는 과정과 음원 자체에는 목적에 따른 "
            "변화가 생길 수 있으므로, 가진 악보와 음원이 다르면 어떤 "
            "형태로 부를지 미리 확인하고 생략이나 반복을 계획한다."
        )

        class Index:
            last_answer_candidates = [lyric_only]

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return []

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [lyric_only, recording_score_answer]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query=(
                "음원과 악보가 서로 일치하지 않을 때 무엇을 기준으로 "
                "노래해야 할까?"
            ),
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [recording_score_answer])










    def test_ranged_semantic_focus_drops_range_only_neighbor(self) -> None:
        target = make_result(
            "in-flowery-clouds-ku-006",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.643116,
            dense_content_score=0.578318,
            semantic_match_type="dense",
        )
        target.record["answer"] = "높은 첫 음은 앞 프레이즈와 연결해 부른다."
        range_only_neighbor = make_result(
            "in-flowery-clouds-ku-005",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.459916,
            dense_content_score=0.459916,
            content_concept_coverage=0.142857,
            semantic_match_type="dense",
            source_ids=["unrelated-source"],
        )
        range_only_neighbor.record["answer"] = (
            "부점에서는 한국어 단어의 장단을 표현한다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [target, range_only_neighbor]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [target, range_only_neighbor]

        results, messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="첫 음을 내기가 어려우면 어떻게 해야 할까?",
            piece="die-forelle",
            measure_ranges=[[2, 5]],
            measures="2-5",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [target])
        self.assertIn(target.record["answer"], messages[1]["content"])
        self.assertNotIn(
            range_only_neighbor.record["answer"],
            messages[1]["content"],
        )

    def test_ranged_focus_can_choose_direct_whole_piece_over_local_neighbor(
        self,
    ) -> None:
        local_neighbor = make_result(
            "local-neighbor",
            alias_score=0.0,
            scope_match="overlaps_query_range",
            text_score=0.0,
            dense_score=0.45,
            dense_content_score=0.44,
            semantic_match_type="dense",
        )
        local_neighbor.record["answer"] = (
            "빠른 음표는 느린 박자부터 점차 빠르게 연습한다."
        )
        whole_piece_answer = make_result(
            "whole-piece-answer",
            alias_score=0.0,
            scope_match="global_context",
            text_score=0.0,
            dense_score=0.70,
            dense_content_score=0.64,
            semantic_match_type="selector_candidate",
            source_ids=["whole-piece-source"],
        )
        whole_piece_answer.record["measure_range"] = []
        whole_piece_answer.record["measure_status"] = "whole_piece"
        whole_piece_answer.record["answer"] = (
            "l’ingegno aguzzerò에서는 앞 단어의 o와 뒤 단어의 a를 "
            "연음으로 자연스럽게 이어 발음한다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [local_neighbor]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [whole_piece_answer, local_neighbor]

        results, messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query=(
                "l’ingegno aguzzerò의 발음을 자연스럽게 이어 "
                "붙이려면 어떻게 해야 할까?"
            ),
            piece="una-voce-poco-fa",
            measure_ranges=[[32, 32]],
            measures="32",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [whole_piece_answer])
        self.assertIn(whole_piece_answer.record["answer"], messages[1]["content"])
        self.assertNotIn(local_neighbor.record["answer"], messages[1]["content"])

    def test_ranged_focus_keeps_direct_local_exception_as_anchor(self) -> None:
        local_exception = make_result(
            "local-exception",
            alias_score=0.0,
            scope_match="overlaps_query_range",
            text_score=2.0,
            dense_score=0.63,
            dense_content_score=0.61,
            content_concept_coverage=0.5,
            semantic_match_type="hybrid",
            source_ids=["local-source"],
        )
        local_exception.record["answer"] = (
            "이 숨표에서는 앞뒤 프레이즈를 분리해 숨을 쉰다."
        )
        global_rule = make_result(
            "global-rule",
            alias_score=0.0,
            scope_match="global_context",
            text_score=0.0,
            dense_score=0.72,
            dense_content_score=0.67,
            content_concept_coverage=0.5,
            semantic_match_type="selector_candidate",
            source_ids=["global-source"],
        )
        global_rule.record["measure_range"] = []
        global_rule.record["measure_status"] = "whole_piece"
        global_rule.record["answer"] = (
            "이 곡의 프레이즈는 일반적으로 끊지 않고 연결한다."
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [local_exception, global_rule]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [global_rule, local_exception]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="이 부분의 숨표는 어떻게 처리해야 할까?",
            piece="die-forelle",
            measure_ranges=[[20, 20]],
            measures="20",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results[0], local_exception)
        self.assertIn(global_rule, results)

    def test_ranged_semantic_focus_preserves_supported_multiple_units(
        self,
    ) -> None:
        anchor = make_result(
            "anchor",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.70,
            dense_content_score=0.68,
            semantic_match_type="dense",
        )
        near_tie = make_result(
            "near-tie",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.66,
            dense_content_score=0.65,
            semantic_match_type="dense",
            source_ids=["near-tie-source"],
        )
        concept_supported = make_result(
            "concept-supported",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.64,
            dense_content_score=0.63,
            content_concept_coverage=0.5,
            semantic_match_type="dense",
            source_ids=["concept-source"],
        )
        directly_supported = make_result(
            "directly-supported",
            alias_score=0.0,
            text_score=4.0,
            dense_score=0.45,
            dense_content_score=0.45,
            content_concept_coverage=0.5,
            semantic_match_type="strict",
            source_ids=["direct-source"],
        )
        unrelated = make_result(
            "unrelated",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.60,
            dense_content_score=0.59,
            semantic_match_type="dense",
            source_ids=["unrelated-source"],
        )
        pool = [
            anchor,
            near_tie,
            concept_supported,
            directly_supported,
            unrelated,
        ]

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return list(pool)

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return list(pool)

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="서로 다른 두 표현을 함께 어떻게 살릴까?",
            piece="die-forelle",
            measure_ranges=[[2, 5]],
            measures="2-5",
            topic=None,
            top_k=6,
        )

        self.assertEqual(
            results,
            [anchor, near_tie, concept_supported, directly_supported],
        )

    def test_ranged_semantic_focus_leaves_weak_leader_unchanged(
        self,
    ) -> None:
        weak_leader = make_result(
            "weak-leader",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.57,
            dense_content_score=0.55,
            semantic_match_type="dense",
        )
        weak_neighbor = make_result(
            "weak-neighbor",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.45,
            dense_content_score=0.44,
            semantic_match_type="dense",
            source_ids=["other-source"],
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [weak_leader, weak_neighbor]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [weak_leader, weak_neighbor]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="의미 신호가 약한 범위 질문입니다.",
            piece="die-forelle",
            measure_ranges=[[2, 5]],
            measures="2-5",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [weak_leader, weak_neighbor])

    def test_ranged_alias_authority_skips_semantic_focus(self) -> None:
        first = make_result("alias-first", alias_score=1.0)
        second = make_result(
            "alias-second",
            alias_score=1.0,
            source_ids=["second-source"],
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [first, second]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                raise AssertionError(
                    "ranged semantic focus must preserve the alias path"
                )

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="원래 주석 질문입니다.",
            piece="die-forelle",
            measure_ranges=[[2, 5]],
            measures="2-5",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [first, second])

    def test_other_range_alias_does_not_disable_primary_range_focus(
        self,
    ) -> None:
        target = make_result(
            "target",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.70,
            dense_content_score=0.68,
            semantic_match_type="dense",
        )
        range_only_neighbor = make_result(
            "range-only-neighbor",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.47,
            dense_content_score=0.45,
            semantic_match_type="dense",
            source_ids=["neighbor-source"],
        )
        other_range_alias = make_result(
            "other-range-alias",
            alias_score=1.0,
            scope_match="other_range_context",
            source_ids=["other-range-source"],
        )

        class Index:
            last_answer_candidates: list[SearchResult] = []

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return [target, range_only_neighbor, other_range_alias]

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [target, range_only_neighbor]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="선택한 마디의 핵심 표현은 무엇인가요?",
            piece="die-forelle",
            measure_ranges=[[2, 5]],
            measures="2-5",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [target, other_range_alias])

    def test_ranged_broad_rescue_uses_same_semantic_focus(self) -> None:
        target = make_result(
            "rescued-target",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.72,
            dense_content_score=0.70,
            semantic_match_type="selector_candidate",
        )
        rejected_noise = make_result(
            "rejected-noise",
            alias_score=0.0,
            text_score=0.0,
            dense_score=0.48,
            dense_content_score=0.46,
            semantic_match_type="selector_candidate",
            source_ids=["rejected-source"],
        )

        class Index:
            last_answer_candidates = [rejected_noise]

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return []

            @staticmethod
            def semantic_candidates(**_kwargs: Any) -> list[SearchResult]:
                return [target, rejected_noise]

        results, _messages = answer_cli.retrieve_generation_evidence(
            Index(),
            query="범위 안에서 직접 답하는 조언은 무엇인가요?",
            piece="die-forelle",
            measure_ranges=[[2, 5]],
            measures="2-5",
            topic=None,
            top_k=6,
        )

        self.assertEqual(results, [target])

    def test_other_range_context_is_capped_at_two_records(self) -> None:
        overlap = make_result("overlap", alias_score=0.0)
        contexts = [
            make_result(
                f"other-range-{index}",
                alias_score=0.0,
                scope_match="other_range_context",
            )
            for index in range(3)
        ]

        selected = select_generation_evidence([overlap, *contexts])

        self.assertEqual(selected, [overlap, *contexts[:2]])

    def test_prompt_marks_other_range_evidence_as_context_only(self) -> None:
        overlap = make_result("overlap")
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
        )

        prompt = build_messages(
            "die-forelle",
            "2-5",
            "악센트는 무엇을 표현할까?",
            [overlap, other_range],
        )[1]["content"]

        self.assertIn("evidence_role: selected_range_support", prompt)
        self.assertIn("evidence_role: other_range_context_only", prompt)
        self.assertIn(
            "citation_label: (none; context-only evidence is not citable)",
            prompt,
        )
        self.assertIn("internal range-mismatch signal", prompt)
        self.assertIn(
            "do not infer, cite, paraphrase, or transfer",
            prompt,
        )

    def test_prompt_preserves_condition_boundaries_and_warning_polarity(
        self,
    ) -> None:
        prompt = build_messages(
            "die-forelle",
            "",
            "서로 다른 표시를 어떻게 구분해야 할까?",
            [make_result("expert", scope_match="general_evidence")],
        )[0]["content"]

        self.assertIn("Preserve condition boundaries and contrasts", prompt)
        self.assertIn("state each rule separately", prompt)
        self.assertIn("never transfer", prompt)
        self.assertIn("Never reverse polarity", prompt)
        self.assertIn("is not a recommendation", prompt)
        self.assertIn("to create that effect", prompt)

    def test_prompt_preserves_interpretive_and_causal_status(self) -> None:
        prompt = build_messages(
            "die-forelle",
            "",
            "이 표시가 쓰인 이유는 무엇인가요?",
            [make_result("expert", scope_match="general_evidence")],
        )[0]["content"]

        self.assertIn("canonical structure", prompt)
        self.assertIn("causal and determining", prompt)
        self.assertIn("premise embedded in the user's", prompt)
        self.assertIn("question is not evidence", prompt)
        self.assertIn("performer-facing recommendation", prompt)
        self.assertIn("Preserve every", prompt)
        self.assertIn("reporting attribution", prompt)
        self.assertIn("what determines", prompt)
        self.assertIn("proofread Korean verb conjugations", prompt)

    def test_local_example_keeps_canonical_ranges_out_of_no_range_prose(
        self,
    ) -> None:
        local = make_result(
            "una-voce-poco-fa-ku-028",
            scope_match="local_example",
        )
        local.record["piece"] = "una-voce-poco-fa"
        local.record["work"] = "Una voce poco fa"
        local.record["measure_range"] = [[63, 63], [82, 83]]
        local.record["answer"] = (
            "고음이 강박에 놓이지 않은 경우에는 단어와 음악의 "
            "강세 관계에 유의해 유연하게 노래해야 한다."
        )
        local.record["source_answer_context"] = [
            {
                "source_id": "yeon-una-voce-poco-fa-11",
                "question": "고음을 어떻게 불러야 할까?",
                "answer": (
                    "첫 마디가 아니라 82마디에서는 고음이 강박에 "
                    "놓이지 않는다. m. 51 및 measures 52–53의 "
                    "표기는 참고하지 말고, 68, 134마디는 지나치게 "
                    "강조하지 않는다."
                ),
                "legacy_measure_range_hints": [[82, 82]],
            }
        ]

        messages = build_messages(
            "una-voce-poco-fa",
            "",
            "이 노래를 부를 때 무엇에 유의해야 할까?",
            [local],
        )
        system_prompt = messages[0]["content"]
        context = messages[1]["content"]

        self.assertIn("Evidence_role=local_example", system_prompt)
        self.assertIn("evidence_role: local_example", context)
        self.assertNotIn("measure_range: 63, 82-83", context)
        self.assertIn("citable passage-specific advice", context)
        self.assertIn("hidden presentation metadata", context)
        self.assertIn("does not establish", context)
        self.assertNotIn("82마디에서는", context)
        self.assertNotIn("첫 마디", context)
        self.assertNotIn("m. 51", context)
        self.assertNotIn("measures 52–53", context)
        self.assertNotIn("68, 134마디", context)
        self.assertNotIn("해당 대목에서는", context)
        self.assertNotIn("expert_source_answer", context)
        self.assertIn(local.record["answer"], context)
        authority = expert_prompt_factuality_material(local.record, [])
        self.assertFalse(authority["range_partitioned"])
        self.assertEqual(
            authority,
            {
                "range_partitioned": False,
                "answer_texts": [local.record["answer"]],
                "question_contexts": [],
                "source_ids": [],
            },
        )
        self.assertEqual(local.record["measure_range"], [[63, 63], [82, 83]])
        extractive = build_extractive_answer([local])
        self.assertIn(
            (
                "고음이 강박에 놓이지 않은 경우에는 단어와 음악의 "
                "강세 관계에 유의해 유연하게 노래해야 합니다."
            ),
            extractive,
        )
        self.assertNotIn("범위 안내", extractive)
        self.assertNotIn("확인된 국소 예시", extractive)
        self.assertNotRegex(extractive, r"(?:63|82\s*[-–—~]\s*83)\s*마디")
        self.assertNotIn("[E1]", extractive)
        self.assertNotIn("una-voce-poco-fa-ku-028", extractive)

    def test_local_only_answer_cannot_claim_piece_wide_frequency(
        self,
    ) -> None:
        local = make_result(
            "la-capinera-ku-021",
            scope_match="local_example",
        )
        local.record["measure_range"] = [[30, 30], [33, 33]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "이 곡 전반에는 도약음이 많이 나오므로 항상 "
                    "고음을 약하게 불러야 한다. [E1]"
                ),
                [local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "질문하신 도약음에서는 지나치게 강조되지 않도록 "
                    "유의한다. [E1]"
                ),
                [local],
            )
        )
        for unsupported_claim in (
            "이 노래에는 도약음이 자주 나오므로 조심한다. [E1]",
            "고음이 강박에 있지 않은 경우가 많으므로 조심한다. [E1]",
            "대부분의 구간에서 고음을 가볍게 부른다. [E1]",
            "전 곡에 걸쳐 같은 방식으로 노래한다. [E1]",
            "모든 대목에서 같은 강세를 사용한다. [E1]",
        ):
            with self.subTest(unsupported_claim=unsupported_claim):
                self.assertTrue(
                    answer_overgeneralizes_local_examples(
                        unsupported_claim,
                        [local],
                    )
                )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "30마디와 33마디에서는 도약하는 고음이 "
                    "지나치게 강조되지 않도록 유의한다. [E1]"
                ),
                [local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "30마디에서는 도약음이 자주 나오지는 않는다고 "
                    "본다. [E1]"
                ),
                [local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "30마디의 근거만으로 곡 전반에 적용된다고 "
                    "확인되지 않았다. [E1]"
                ),
                [local],
            )
        )

        whole = make_result(
            "la-capinera-ku-whole",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "이 곡 전반의 가창에 유의해야 한다. [E1] [E2]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "이 곡 전반의 가창에 유의해야 한다. [E1]",
                [whole, local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "이 곡 전반의 가창에 유의하되, 30마디의 "
                    "도약음을 국소 예시로 참고한다. [E1] [E2]"
                ),
                [whole, local],
            )
        )

    def test_no_range_rejects_every_numeric_local_measure_locator(
        self,
    ) -> None:
        first = make_result(
            "una-voce-poco-fa-ku-028",
            scope_match="local_example",
        )
        first.record["measure_range"] = [[63, 63], [82, 83]]
        second = make_result(
            "una-voce-poco-fa-ku-020",
            scope_match="local_example",
        )
        second.record["measure_range"] = [[68, 69], [72, 73]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "63마디에서 고음을 유연하게 부른다. [E1] [E2]",
                [first, second],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "63마디에서 고음을 유연하게 부른다. [E1]\n"
                    "68–69마디에서 악센트를 가볍게 표현한다. "
                    "[E2]"
                ),
                [first, second],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "82마디부터 83마디에서 고음을 유연하게 부른다. [E1]",
                [first, second],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "64마디에서 고음을 유연하게 부른다. [E1]",
                [first, second],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "질문하신 고음에서는 유연하게 노래한다. [E1]\n"
                    "이어지는 악센트는 가볍게 표현한다. [E2]"
                ),
                [first, second],
            )
        )

    def test_unlabeled_body_cannot_be_attributed_to_retrieved_locals(
        self,
    ) -> None:
        whole = make_result(
            "whole",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        local = make_result("local", scope_match="local_example")
        local.record["measure_range"] = [[30, 30]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "도약 직전에 충분히 호흡한다.\n\n"
                    "제공된 검색 근거: [E1] [E2]"
                ),
                [whole, local],
            )
        )

    def test_measure_locators_need_query_or_cited_local_authority(
        self,
    ) -> None:
        whole = make_result(
            "la-capinera-ku-028",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        whole.record["source_answer_context"] = [
            {
                "source_id": "yeon-la-capinera-10",
                "question": "",
                "answer": "135마디에서 선율을 이어 부른다.",
                "legacy_measure_range_hints": [[135, 135]],
            }
        ]
        local = make_result(
            "la-capinera-ku-011",
            scope_match="local_example",
        )
        local.record["measure_range"] = [[41, 43]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "135마디에서 선율을 이어 부른다. [E1]",
                [whole],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "135마디에서 선율을 이어 부른다. [E1]",
                [whole, local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "41–43마디에서 레가토를 유지한다. [E1]",
                [whole, local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "41–43마디에서 레가토를 유지한다. [E2]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "기교를 더할 때는 모음으로 선율을 이어 부른다. [E1]",
                [whole, local],
            )
        )

        selected = make_result(
            "die-forelle-ku-002",
            scope_match="overlaps_query_range",
        )
        selected.record["measure_range"] = [[2, 27]]
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "2–5마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "3마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "1–6마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "작품의 분위기를 먼저 이해한다. [E1]",
                [whole, local],
            )
        )

    def test_extractive_answer_does_not_present_other_range_as_grounding(
        self,
    ) -> None:
        overlap = make_result("overlap")
        overlap.record["answer"] = "선택 범위 답변입니다."
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간 답변입니다."

        mixed = build_extractive_answer([overlap, other_range])
        secondary_only = build_extractive_answer([other_range])

        self.assertIn("선택 범위 답변", mixed)
        self.assertNotIn("다른 구간 답변", mixed)
        self.assertIn(
            "선택하신 마디에는 이 내용을 직접 적용하기 어렵습니다.",
            secondary_only,
        )
        self.assertIn("곡의 다른 부분에 해당합니다", secondary_only)
        self.assertNotIn("다른 구간의 관련 주석", secondary_only)
        self.assertIn("다른 구간 답변", secondary_only)

    def test_secondary_and_uncited_drafts_are_rejected(
        self,
    ) -> None:
        overlap = make_result("die-forelle-ku-001")
        overlap.record["answer"] = "선택 범위의 근거"
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간의 강약 교대"

        with self.assertRaises(GeneratedAnswerRejected):
            finalize_answer_citations(
                (
                    "다른 구간의 강약 교대 [E2] "
                    "[die-forelle-ku-002]"
                ),
                [overlap, other_range],
            )
        with self.assertRaises(GeneratedAnswerRejected):
            finalize_answer_citations(
                "인용을 생략한 답변",
                [overlap, other_range],
            )
        with self.assertRaisesRegex(
            GeneratedAnswerRejected,
            "uncited substantive claim",
        ):
            finalize_answer_citations(
                (
                    "부점 리듬에 주의해야 합니다. [E1] "
                    "이 부분은 리듬이 복잡하고 강약을 "
                    "조절해야 합니다."
                ),
                [overlap],
            )
        with self.assertRaisesRegex(
            GeneratedAnswerRejected,
            "uncited substantive claim",
        ):
            finalize_answer_citations(
                (
                    "부점 리듬에 주의해야 합니다. [E1]"
                    "이는 근거 없이 추가된 설명입니다."
                ),
                [overlap],
            )

        self.assertEqual(
            finalize_answer_citations(
                (
                    "부점 리듬에 주의해야 합니다. [E1] "
                    "이 부분에서는 템포를 유지해야 합니다. [E1]"
                ),
                [overlap],
            ),
            (
                "부점 리듬에 주의해야 합니다. "
                "이 부분에서는 템포를 유지해야 합니다."
            ),
        )


    def test_pipeline_scope_scaffolding_is_removed_but_answer_is_kept(
        self,
    ) -> None:
        legacy = (
            "범위 안내: 아래 내용은 표시된 마디에서 확인된 국소 "
            "예시입니다.\n\n"
            "확인된 국소 예시(마디 44-55): 아리아의 긴 간주에서는 "
            "표정과 움직임으로 극의 흐름에 반응해야 합니다."
        )
        markdown = (
            "**범위 안내:** 내부 범위입니다.\n\n"
            "### **확인된 국소 예시(마디 44–55):** 긴 간주에서는 "
            "다음 장면을 준비해야 합니다."
        )
        copied_prompt_fields = (
            "measure_range: 44-55\n"
            "evidence_role: local_example\n"
            "긴 간주에서는 앞뒤 감정의 변화를 이어야 합니다. [E1]"
        )

        self.assertEqual(
            finalize_user_visible_answer(legacy),
            (
                "아리아의 긴 간주에서는 표정과 움직임으로 극의 "
                "흐름에 반응해야 합니다."
            ),
        )
        self.assertEqual(
            finalize_user_visible_answer(markdown),
            "긴 간주에서는 다음 장면을 준비해야 합니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer(copied_prompt_fields),
            "긴 간주에서는 앞뒤 감정의 변화를 이어야 합니다.",
        )

    def test_inline_pipeline_fields_and_parenthesized_citations_never_leak(
        self,
    ) -> None:
        self.assertEqual(
            finalize_user_visible_answer(
                "정상 답변입니다. citation_label: E1 "
                "evidence_role: local_example"
            ),
            "정상 답변입니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer(
                "정상 답변입니다. 범위 안내: 내부 범위입니다."
            ),
            "정상 답변입니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer("정상 답변입니다. (E1)"),
            "정상 답변입니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer("citation_label E1"),
            "현재 확인된 정보만으로는 이 질문에 정확히 "
            "답변하기 어렵습니다.",
        )

        for answer in (
            "die_forelle_ku_010에 따르면 이렇게 부릅니다.",
            "[die forelle ku 010]에 따르면 이렇게 부릅니다.",
            "die–forelle–ku–010에 따르면 이렇게 부릅니다.",
        ):
            with self.subTest(answer=answer):
                finalized = finalize_user_visible_answer(answer)
                self.assertNotRegex(
                    finalized,
                    r"(?i)die[\s_–—-]+forelle[\s_–—-]+ku",
                )

        opaque_id_variants = (
            "webchunk test 01",
            "web_chunk_test_01",
            "web–chunk–test–01",
            "web source 01",
            "web-claim-test",
            "sqa 123",
            "sqa_123",
            "sqa–123",
        )
        unavailable = (
            "현재 확인된 정보만으로는 이 질문에 정확히 "
            "답변하기 어렵습니다."
        )
        for opaque_id in opaque_id_variants:
            with self.subTest(opaque_id=opaque_id, form="bare"):
                self.assertEqual(
                    finalize_user_visible_answer(
                        f"정상 답변입니다. {opaque_id}"
                    ),
                    unavailable,
                )
            with self.subTest(opaque_id=opaque_id, form="bracketed"):
                self.assertEqual(
                    finalize_user_visible_answer(
                        f"[{opaque_id}]에 따르면 이렇게 부릅니다."
                    ),
                    "이렇게 부릅니다.",
                )

        for answer in (
            "measure_range: 44-55",
            "범위 안내: 내부 범위입니다.",
            "제공된 검색 근거:",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(
                    finalize_user_visible_answer(answer),
                    "현재 확인된 정보만으로는 이 질문에 정확히 "
                    "답변하기 어렵습니다.",
                )

    def test_every_pipeline_field_blocks_separator_variants(self) -> None:
        unavailable = (
            "현재 확인된 정보만으로는 이 질문에 정확히 "
            "답변하기 어렵습니다."
        )
        separators = (" ", "-", "_", "**", "`", "–", "—")
        for field_name in answer_cli.PIPELINE_FIELD_NAMES:
            words = field_name.split("_")
            for separator in separators:
                variant = separator.join(words)
                with self.subTest(
                    field_name=field_name,
                    separator=separator,
                ):
                    self.assertEqual(
                        finalize_user_visible_answer(
                            f"정상 답변입니다. {variant} 값을 사용합니다."
                        ),
                        unavailable,
                    )

    def test_pipeline_guards_preserve_ordinary_korean_music_prose(self) -> None:
        answers = (
            "선택한 마디 범위에서는 호흡을 조절합니다.",
            "이 근거의 역할은 프레이즈의 방향을 설명하는 것입니다.",
            "국소적인 예를 들어 음색 변화를 설명합니다.",
            "웹 자료를 읽더라도 악보를 먼저 확인합니다.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertEqual(finalize_user_visible_answer(answer), answer)


    def test_cli_generation_error_propagates_without_retry(self) -> None:
        web = make_result(
            "web-only",
            evidence_type="web_database",
            scope_match="general_evidence",
        )
        web.record["answer"] = "렌더링하면 안 되는 웹 원문"
        web.content_concept_coverage = 1.0

        class WebIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "dense_result_returned"

            @staticmethod
            def search(**_kwargs):
                return [web]

        args = SimpleNamespace(
            settings=None,
            piece="die-forelle",
            measures="",
            measure_ranges=[],
            topic=None,
            question="이 자료는 무엇을 설명하나요?",
            top_k=6,
            no_generate=False,
            rebuild_corpus=False,
            json=True,
        )
        settings = {
            "corpus_path": "/tmp/corpus.json",
            "model_path": "/tmp/model.gguf",
            "llm": {},
        }
        with (
            mock.patch.object(answer_cli, "parse_args", return_value=args),
            mock.patch.object(
                answer_cli,
                "load_settings",
                return_value=settings,
            ),
            mock.patch.object(answer_cli, "validate_retrieval_requirements"),
            mock.patch.object(answer_cli, "validate_generation_requirements"),
            mock.patch.object(answer_cli, "ensure_corpus"),
            mock.patch.object(answer_cli, "load_corpus", return_value=[]),
            mock.patch.object(
                answer_cli,
                "build_retrieval_index",
                return_value=WebIndex(),
            ),
            mock.patch.object(
                answer_cli,
                "generate",
                side_effect=ValueError(
                    "requested tokens exceed the context window"
                ),
            ) as generate,
        ):
            with self.assertRaises(ValueError):
                answer_cli.main()
        generate.assert_called_once()

    def test_deictic_generic_range_question_is_underspecified(self) -> None:
        self.assertTrue(
            is_underspecified_ranged_query(
                "이 부분은 어떻게 노래해야 할까?",
                "nella-fantasia",
                [[11, 17]],
            )
        )
        self.assertFalse(
            is_underspecified_ranged_query(
                "이 부분에서 호흡은 어디에서 해야 할까?",
                "nella-fantasia",
                [[11, 17]],
            )
        )

    def test_secondary_citation_variants_are_rejected(self) -> None:
        overlap = make_result("die-forelle-ku-001")
        overlap.record["answer"] = "선택 범위의 근거"
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간 위험 주장"

        secondary_references = [
            "[E 2]",
            "[Evidence 2]",
            "[E–2]",
            "[Evidence no. 2]",
            "Evidence 2",
            "근거: 2",
            "DIE-FORELLE-KU-002",
        ]
        for reference in secondary_references:
            with self.subTest(reference=reference):
                with self.assertRaises(GeneratedAnswerRejected):
                    finalize_answer_citations(
                        f"다른 구간 위험 주장 {reference}",
                        [overlap, other_range],
                    )

    def test_secondary_citation_never_substitutes_primary_text(self) -> None:
        primary = [
            make_result(
                f"primary-{index}",
                scope_match="overlaps_query_range",
                source_ids=[f"source-{index}"],
            )
            for index in range(4)
        ]
        for index, result in enumerate(primary):
            result.record["answer"] = f"선택 근거 {index}입니다."
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
            source_ids=["other-source"],
        )

        with self.assertRaises(GeneratedAnswerRejected):
            finalize_answer_citations(
                "다른 구간의 위험 주장 [E5]",
                [*primary, other_range],
            )

    def test_music_and_korean_counts_are_not_secondary_citations(self) -> None:
        overlap = make_result("die-forelle-ku-001")
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )

        ordinary_uses = {
            "최저음은 E2이다. [E1]": "최저음은 E2입니다.",
            "최저음은 E2. [E1]": "최저음은 E2.",
            "음역 하한은 E2! [E1]": "음역 하한은 E2!",
            "근거 2가지를 설명한다. [E1]": "근거 2가지를 설명합니다.",
            "출처 2곳을 비교한다. [E1]": "출처 2곳을 비교합니다.",
        }
        for answer, expected in ordinary_uses.items():
            with self.subTest(answer=answer):
                finalized = finalize_answer_citations(
                    answer,
                    [overlap, other_range],
                )
                self.assertEqual(finalized, expected)
                self.assertNotIn("[E1]", finalized)
                self.assertNotIn("die-forelle-ku-001", finalized)
                if "E2" in answer:
                    self.assertIn("E2", finalized)

    def test_user_visible_answer_strips_internal_ids_and_formalizes_style(
        self,
    ) -> None:
        primary = make_result("die-forelle-ku-001")
        secondary = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )

        finalized = finalize_user_visible_answer(
            (
                "첫 문장은 근거를 설명한다. [E1, E2]\n"
                "둘째 문장은 이어진다. die-forelle-ku-001"
            ),
            [primary, secondary],
        )

        self.assertEqual(
            finalized,
            "첫 문장은 근거를 설명합니다.\n둘째 문장은 이어집니다.",
        )
        self.assertNotIn("[E1", finalized)
        self.assertNotIn("die-forelle-ku-001", finalized)
        self.assertNotIn("die-forelle-ku-002", finalized)

        citation_suffix_cases = {
            (
                "설명입니다. [E1]에 따르면, 전주가 있을 수 있습니다."
            ): "설명입니다. 전주가 있을 수 있습니다.",
            (
                "설명입니다. [die-forelle-ku-001]에서는 12마디를 "
                "확인합니다."
            ): "설명입니다. 12마디를 확인합니다.",
            (
                "설명입니다. [die-forelle-ku-001]에서 보듯이, 24마디를 "
                "확인합니다."
            ): "설명입니다. 24마디를 확인합니다.",
            (
                "설명입니다. die-forelle-ku-001에 의하면, 원본을 "
                "확인합니다."
            ): "설명입니다. 원본을 확인합니다.",
            (
                "첫 문장입니다. [E1] 이는 다음 설명입니다. [E1]"
            ): "첫 문장입니다. 이는 다음 설명입니다.",
            (
                "첫 문장입니다. [E1] 이 부분은 다음 "
                "설명입니다. [E1]"
            ): "첫 문장입니다. 이 부분은 다음 설명입니다.",
            (
                "첫 문장입니다. [E1]이는 다음 설명입니다. [E1]"
            ): "첫 문장입니다. 이는 다음 설명입니다.",
            (
                "첫 문장입니다. die-forelle-ku-001 이는 "
                "다음 설명입니다."
            ): "첫 문장입니다. 이는 다음 설명입니다.",
        }
        for answer, expected in citation_suffix_cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(
                    finalize_user_visible_answer(answer, [primary]),
                    expected,
                )

        already_orphaned_cases = {
            "설명입니다. 에 따르면, 전주가 있습니다.": (
                "설명입니다. 전주가 있습니다."
            ),
            "설명입니다. 에서는 12마디를 확인합니다.": (
                "설명입니다. 12마디를 확인합니다."
            ),
            "설명입니다. 에서 보듯이, 24마디를 확인합니다.": (
                "설명입니다. 24마디를 확인합니다."
            ),
        }
        for answer, expected in already_orphaned_cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(
                    finalize_user_visible_answer(answer, [primary]),
                    expected,
                )

        no_space_boundary = finalize_user_visible_answer(
            "악보대로 부르지 않는다.성악 성부를 확인한다."
        )
        self.assertEqual(
            no_space_boundary,
            "악보대로 부르지 않습니다. 성악 성부를 확인합니다.",
        )

        duplicated_source_spacing = finalize_user_visible_answer(
            "이 곡은 일관적으로  서정적이다.\t\t표현을 유지한다."
        )
        self.assertEqual(
            duplicated_source_spacing,
            "이 곡은 일관적으로 서정적입니다. 표현을 유지합니다.",
        )

        lexical_nida_cases = {
            "결코 쉬운 곡은 아니다.": "결코 쉬운 곡은 아닙니다.",
            "선율이 여러 성부를 다니다.": "선율이 여러 성부를 다닙니다.",
            "중요한 의미를 지니다.": "중요한 의미를 지닙니다.",
        }
        for answer, expected in lexical_nida_cases.items():
            with self.subTest(answer=answer):
                self.assertFalse(answer_uses_formal_polite_korean(answer))
                finalized = finalize_user_visible_answer(answer)
                self.assertEqual(finalized, expected)
                self.assertTrue(
                    answer_uses_formal_polite_korean(finalized)
                )

        for answer in (
            "결코 쉬운 곡은 아닙니다.",
            "선율이 여러 성부를 다닙니다.",
            "중요한 의미를 지닙니다.",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(finalize_user_visible_answer(answer), answer)
                self.assertTrue(answer_uses_formal_polite_korean(answer))

    def test_nonpolite_sentence_endings_fail_closed(self) -> None:
        for answer in (
            "이렇게 해 보자.",
            "어떻게 할까?",
            "숨을 쉬어라.",
            "이 부분을 조심해.",
            "Please sing softly.",
            "Use a lighter tone.",
            "OK",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(answer_uses_formal_polite_korean(answer))
                self.assertEqual(
                    finalize_user_visible_answer(answer),
                    "현재 확인된 정보만으로는 이 질문에 정확히 "
                    "답변하기 어렵습니다.",
                )

        for answer in (
            "이렇게 해 봅시다.",
            "어떻게 해야 합니까?",
            "숨을 쉬십시오.",
            "이 부분을 조심하세요.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(answer_uses_formal_polite_korean(answer))
                self.assertEqual(finalize_user_visible_answer(answer), answer)

    def test_pipeline_vocabulary_fails_closed_at_display_boundary(self) -> None:
        for term in (
            "knowledge unit",
            "expert annotation",
            "retrieved evidence",
            "retrieval",
            "지식 단위",
            "전문가 주석",
            "검색 파이프라인",
            "검색 결과",
        ):
            with self.subTest(term=term):
                self.assertEqual(
                    finalize_user_visible_answer(
                        f"{term}에 따르면 이렇게 부릅니다."
                    ),
                    "현재 확인된 정보만으로는 이 질문에 정확히 "
                    "답변하기 어렵습니다.",
                )

        for leaked_reference in (
            "retrieved corpus record에 따르면 이렇게 부릅니다.",
            "검색된 코퍼스 근거에 따르면 이렇게 부릅니다.",
            "data/corpus.json의 기록에 따르면 이렇게 부릅니다.",
        ):
            with self.subTest(leaked_reference=leaked_reference):
                self.assertEqual(
                    finalize_user_visible_answer(leaked_reference),
                    "현재 확인된 정보만으로는 이 질문에 정확히 "
                    "답변하기 어렵습니다.",
                )

    def test_domain_corpus_vocabulary_is_preserved(self) -> None:
        legitimate_answers = (
            (
                "해당 악보 이미지는 공개 라이선스가 없으므로 "
                "연구 코퍼스에 수록해서는 안 됩니다."
            ),
            "내부 연구와 공개 말뭉치 배포를 구분해야 합니다.",
            "This research corpus는 공개 원문만 수록합니다.",
            "제공된 말뭉치의 공개 라이선스를 확인해야 합니다.",
            "The provided corpus는 연구용으로만 사용해야 합니다.",
        )
        for answer in legitimate_answers:
            with self.subTest(answer=answer):
                self.assertEqual(finalize_user_visible_answer(answer), answer)

    def test_prompt_uses_curated_answer_instead_of_split_raw_source(
        self,
    ) -> None:
        result = make_result("la-capinera-ku-001")
        result.record["answer"] = "pianissimo로 시작하는 것이 좋다."
        result.record["source_answer_context"] = [
            {
                "source_id": "kim-la-capinera-01",
                "question": "어떻게 시작하고 발음할까?",
                "answer": (
                    "pianissimo로 시작하고 l과 r을 정확히 발음한다."
                ),
                "legacy_measure_range_hints": [[11, 14], [78, 81]],
                "linked_knowledge_unit_ids": [
                    "la-capinera-ku-001",
                    "la-capinera-ku-002",
                ],
            }
        ]
        messages = build_messages(
            "la-capinera",
            "78-81",
            "어떻게 시작할까?",
            [result],
        )
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        self.assertIn("authoritative claim text", system_prompt)
        self.assertIn(
            "Raw source annotations, legacy range hints, superseded wording",
            system_prompt,
        )
        self.assertNotIn("split_source_context_withheld", user_prompt)
        self.assertIn("pianissimo로 시작하는 것이 좋다.", user_prompt)
        self.assertNotIn("l과 r을 정확히 발음한다", user_prompt)

    def test_prompt_requests_complete_but_nonrepetitive_synthesis(self) -> None:
        messages = build_messages(
            "die-forelle",
            "",
            "이 곡을 잘 부르려면 어떻게 해야 할까?",
            [make_result("die-forelle-ku-009")],
        )
        system_prompt = messages[0]["content"]

        self.assertIn("all directly useful", system_prompt)
        self.assertIn("restrict the response to one item", system_prompt)
        self.assertIn("adjacent clauses", system_prompt)
        self.assertIn("parallel recommendations", system_prompt)
        self.assertIn("distinct practical action", system_prompt)
        self.assertIn('call its numbers "마디"', system_prompt)


class GenerationArtifactRepairTests(unittest.TestCase):
    def test_detects_unseen_camel_case_inside_korean_text(self) -> None:
        messages = [{"role": "user", "content": "꾸밈음을 가볍게 부른다."}]

        self.assertEqual(
            suspicious_generation_tokens(
                "꾸SizePolicy는 가볍게 부른다.",
                messages,
            ),
            ["SizePolicy"],
        )
        self.assertEqual(
            suspicious_generation_tokens(
                "Larghetto는 여유 있게 부른다.",
                messages,
            ),
            [],
        )






class RangeScopedSourceContextTests(unittest.TestCase):
    def test_no_range_uses_only_curated_answer_for_nonlocal_units(
        self,
    ) -> None:
        for scope_match, measure_status in (
            ("general_evidence", "whole_piece"),
            ("unspecified_scope", "unspecified"),
        ):
            with self.subTest(scope_match=scope_match):
                result = make_result(
                    f"source-{scope_match}",
                    scope_match=scope_match,
                    measure_status=measure_status,
                )
                result.record["measure_range"] = []
                result.record["answer"] = "검토가 완료된 지식 단위 답변이다."
                result.record["source_answer_context"] = [
                    {
                        "source_id": "source-a",
                        "question": "135마디에서 어떻게 부를까?",
                        "answer": "135마디에서 모음으로 선율을 이어 부른다.",
                        "legacy_measure_range_hints": [[135, 135]],
                    }
                ]

                prompt = build_messages(
                    "la-capinera",
                    "",
                    "이 곡을 부를 때 무엇에 유의할까?",
                    [result],
                )[1]["content"]
                authority = expert_prompt_factuality_material(
                    result.record,
                    [],
                )

                self.assertNotIn("135마디", prompt)
                self.assertNotIn("해당 대목에서", prompt)
                self.assertNotIn("expert_source_answer", prompt)
                self.assertIn(result.record["answer"], prompt)
                self.assertEqual(
                    authority,
                    {
                        "range_partitioned": False,
                        "answer_texts": [result.record["answer"]],
                        "question_contexts": [],
                        "source_ids": [],
                    },
                )

    def test_selected_range_uses_curated_answer_and_withholds_raw_sources(
        self,
    ) -> None:
        merged = make_result("la-capinera-ku-018")
        merged.record["measure_range"] = [[51, 59], [117, 121]]
        merged.record["answer"] = (
            "앞 구간의 두 블록 대비와 뒤 구간의 빠른 교대를 함께 설명한다."
        )
        merged.record["source_answer_context"] = [
            {
                "source_id": "kim-la-capinera-17",
                "question": (
                    "117마디의 ff와 121마디의 pp는 왜 빠르게 "
                    "번갈아 나올까?"
                ),
                "answer": "빠른 교대는 메아리 효과로 볼 수 있다.",
                "legacy_measure_range_hints": [[117, 121]],
            },
            {
                "source_id": "yeon-la-capinera-07",
                "question": "",
                "answer": (
                    "51-59마디의 ff와 pp는 51-54마디 "
                    "55-59마디의 대비가 드러나도록 표현한다."
                ),
                "legacy_measure_range_hints": [[51, 59]],
            },
        ]

        prompt = build_messages(
            "la-capinera",
            "51-58",
            "왜 빠르게 번갈아 나올까?",
            [merged],
        )[1]["content"]

        self.assertNotIn("range_partitioned_source_context", prompt)
        self.assertNotIn("expert_source_answer", prompt)
        self.assertNotIn("51-59마디의 ff와 pp", prompt)
        self.assertNotIn("빠른 교대는 메아리 효과로 볼 수 있다.", prompt)
        self.assertIn(merged.record["answer"], prompt)
        authority = expert_prompt_factuality_material(
            merged.record,
            [[51, 58]],
        )
        self.assertEqual(
            authority,
            {
                "range_partitioned": False,
                "answer_texts": [merged.record["answer"]],
                "question_contexts": [],
                "source_ids": [],
            },
        )

    def test_narrow_range_cannot_restore_conflicting_raw_timing(self) -> None:
        merged = make_result("in-flowery-clouds-ku-008")
        merged.record["measure_range"] = [[45, 47]]
        merged.record["answer"] = (
            "박자와 조성의 변화가 곡의 분위기를 전환한다."
        )
        merged.record["source_answer_context"] = [
            {
                "source_id": "kim-in-flowery-clouds-07",
                "question": "박자가 바뀌는 이유는 무엇일까?",
                "answer": "47마디에서 박자와 조성이 동시에 바뀐다.",
                "legacy_measure_range_hints": [[47, 47]],
            },
            {
                "source_id": "yeon-in-flowery-clouds-08",
                "question": "분위기는 언제 달라질까?",
                "answer": "45마디부터 장조에서 단조로 바뀐다.",
                "legacy_measure_range_hints": [[45, 45]],
            },
        ]

        prompt = build_messages(
            "in-flowery-clouds",
            "45",
            "선택한 부분에서 분위기가 달라지는 이유는 무엇일까?",
            [merged],
        )[1]["content"]
        authority = expert_prompt_factuality_material(
            merged.record,
            [[45, 45]],
        )

        self.assertIn(merged.record["answer"], prompt)
        self.assertNotIn("47마디에서 박자와 조성이 동시에", prompt)
        self.assertNotIn("45마디부터 장조에서 단조로", prompt)
        self.assertNotIn("expert_source_answer", prompt)
        self.assertEqual(authority["answer_texts"], [merged.record["answer"]])
        self.assertFalse(authority["range_partitioned"])
        self.assertEqual(authority["question_contexts"], [])
        self.assertEqual(authority["source_ids"], [])

    def test_no_range_withholds_raw_sources_and_uses_curated_rewrite(
        self,
    ) -> None:
        merged = make_result("la-capinera-ku-018")
        merged.record["answer"] = "두 범위를 함께 설명한다."
        merged.record["source_answer_context"] = [
            {
                "source_id": "source-a",
                "question": "",
                "answer": "앞 범위 설명",
                "legacy_measure_range_hints": [[51, 59]],
            },
            {
                "source_id": "source-b",
                "question": "",
                "answer": "뒤 범위 설명",
                "legacy_measure_range_hints": [[117, 121]],
            },
        ]

        prompt = build_messages(
            "la-capinera",
            "",
            "전체적으로 어떻게 표현할까?",
            [merged],
        )[1]["content"]

        self.assertNotIn("range_partitioned_source_context: true", prompt)
        self.assertNotIn("앞 범위 설명", prompt)
        self.assertNotIn("뒤 범위 설명", prompt)
        self.assertNotIn("expert_source_answer", prompt)
        self.assertIn("두 범위를 함께 설명한다.", prompt)
        authority = expert_prompt_factuality_material(
            merged.record,
            [],
        )
        self.assertFalse(authority["range_partitioned"])
        self.assertEqual(authority["answer_texts"], [merged.record["answer"]])
        self.assertEqual(authority["question_contexts"], [])
        self.assertEqual(authority["source_ids"], [])


if __name__ == "__main__":
    unittest.main()
