"""Regression tests for semantically direct local evidence without a range."""

from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest import mock

from soprano_qa import service as qa
from soprano_qa.answer import (
    build_extractive_answer,
    select_generation_evidence,
    semantically_dominant_no_range_local_expert_bundle,
)
from soprano_qa.dense import SOURCE_QUESTION_LED_MATCH_TYPE
from soprano_qa.retrieval import SearchResult


def make_result(
    record_id: str,
    *,
    scope_match: str,
    dense_score: float,
    dense_content_score: float,
    source_ids: list[str] | None = None,
    evidence_type: str = "expert_annotation",
    measure_status: str | None = None,
    retrieval_mode: str = "dense",
    retrieval_aliases: list[str] | None = None,
    semantic_match_type: str = "dense",
) -> SearchResult:
    local = scope_match == "local_example"
    record: Dict[str, Any] = {
        "id": record_id,
        "evidence_type": evidence_type,
        "piece": "test-piece",
        "work": "Test Piece",
        "topic": "performance",
        "answer": "finalized expert answer",
        "measure_range": [[1, 2]] if local else [],
        "measure_scope": "local" if local else "global",
        "measure_status": measure_status
        or ("specific" if local else "whole_piece"),
        "source_ids": (
            source_ids
            if source_ids is not None
            else ["source-" + record_id]
        ),
        "retrieval_aliases": (
            retrieval_aliases
            if retrieval_aliases is not None
            else []
        ),
        "annotators": ["tester"],
        "question": "",
        "web_source_ids": [],
        "claim_ids": [],
        "sources": [],
    }
    return SearchResult(
        record=record,
        score=dense_score,
        text_score=0.0,
        measure_score=0.75 if local else 1.5,
        piece_score=1.0,
        scope_match=scope_match,
        dense_score=dense_score,
        dense_content_score=dense_content_score,
        fusion_score=dense_score,
        retrieval_mode=retrieval_mode,
        semantic_match_type=semantic_match_type,
    )


class NoRangeSemanticFocusTests(unittest.TestCase):



    def test_disagreeing_dense_views_do_not_overpromote_local_detail(
        self,
    ) -> None:
        relevance_winner = make_result(
            "local-relevance-winner",
            scope_match="local_example",
            dense_score=0.66,
            dense_content_score=0.51,
        )
        content_winner = make_result(
            "whole-content-winner",
            scope_match="general_evidence",
            dense_score=0.60,
            dense_content_score=0.62,
        )
        results = [relevance_winner, content_winner]

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(results),
            [],
        )
        self.assertEqual(select_generation_evidence(results), results)

    def test_relevance_led_hybrid_winner_allows_small_content_deficit(
        self,
    ) -> None:
        winner = make_result(
            "hybrid-rank-one-local",
            scope_match="local_example",
            dense_score=0.68,
            dense_content_score=0.55,
            retrieval_mode="hybrid",
        )
        winner.text_score = 4.0
        content_near_tie = make_result(
            "content-near-tie",
            scope_match="general_evidence",
            dense_score=0.62,
            dense_content_score=0.58,
        )

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [winner, content_near_tie]
            ),
            [winner],
        )

    def test_relevance_led_dense_only_path_requires_very_high_score(
        self,
    ) -> None:
        winner = make_result(
            "dense-rank-one-local",
            scope_match="local_example",
            dense_score=0.71,
            dense_content_score=0.58,
        )
        content_near_tie = make_result(
            "content-near-tie",
            scope_match="general_evidence",
            dense_score=0.65,
            dense_content_score=0.62,
        )

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [winner, content_near_tie]
            ),
            [winner],
        )

        winner.dense_score = 0.69
        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [winner, content_near_tie]
            ),
            [],
        )

    def test_source_question_led_marker_bypasses_content_view_gates(
        self,
    ) -> None:
        winner = make_result(
            "source-question-led-local",
            scope_match="local_example",
            dense_score=0.68,
            dense_content_score=0.41,
            source_ids=["source-question"],
            retrieval_aliases=["authoritative source question"],
            semantic_match_type=SOURCE_QUESTION_LED_MATCH_TYPE,
        )
        content_winner = make_result(
            "other-source-content-winner",
            scope_match="general_evidence",
            dense_score=0.55,
            dense_content_score=0.58,
        )

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [winner, content_winner]
            ),
            [winner],
        )
        self.assertEqual(
            select_generation_evidence([winner, content_winner]),
            [winner],
        )

    def test_source_question_led_marker_revalidates_source_route_guards(
        self,
    ) -> None:
        def marked_winner() -> SearchResult:
            return make_result(
                "source-question-led-local",
                scope_match="local_example",
                dense_score=0.68,
                dense_content_score=0.41,
                source_ids=["source-question"],
                retrieval_aliases=["authoritative source question"],
                semantic_match_type=SOURCE_QUESTION_LED_MATCH_TYPE,
            )

        competitor = make_result(
            "other-source",
            scope_match="general_evidence",
            dense_score=0.55,
            dense_content_score=0.58,
        )
        cases = {
            "no-alias": lambda result: result.record.update(
                retrieval_aliases=[]
            ),
            "no-source": lambda result: result.record.update(
                source_ids=[]
            ),
            "no-range": lambda result: result.record.update(
                measure_range=[]
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                winner = marked_winner()
                mutate(winner)

                self.assertEqual(
                    semantically_dominant_no_range_local_expert_bundle(
                        [winner, competitor]
                    ),
                    [],
                )

    def test_source_question_led_marker_requires_clear_lead_and_rank_safety(
        self,
    ) -> None:
        winner = make_result(
            "source-question-led-local",
            scope_match="local_example",
            dense_score=0.68,
            dense_content_score=0.41,
            source_ids=["source-question"],
            retrieval_aliases=["authoritative source question"],
            semantic_match_type=SOURCE_QUESTION_LED_MATCH_TYPE,
        )
        near_tie = make_result(
            "other-source-near-tie",
            scope_match="general_evidence",
            dense_score=0.59,
            dense_content_score=0.58,
        )
        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [winner, near_tie]
            ),
            [],
        )

        unrelated_fusion_winner = make_result(
            "unrelated-fusion-winner",
            scope_match="general_evidence",
            dense_score=0.54,
            dense_content_score=0.52,
            retrieval_mode="hybrid",
        )
        unrelated_fusion_winner.text_score = 5.0
        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [unrelated_fusion_winner, winner]
            ),
            [],
        )

    def test_relevance_led_path_requires_clear_margin(
        self,
    ) -> None:
        winner = make_result(
            "semantic-winner",
            scope_match="local_example",
            dense_score=0.72,
            dense_content_score=0.58,
        )
        near_tie = make_result(
            "relevance-near-tie",
            scope_match="general_evidence",
            dense_score=0.68,
            dense_content_score=0.61,
        )
        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [winner, near_tie]
            ),
            [],
        )

    def test_dense_winner_may_follow_clean_same_source_fusion_sibling(
        self,
    ) -> None:
        sibling = make_result(
            "same-source-fusion-rank-one",
            scope_match="general_evidence",
            dense_score=0.61,
            dense_content_score=0.49,
            source_ids=["shared-source"],
            retrieval_mode="hybrid",
        )
        sibling.text_score = 5.0
        winner = make_result(
            "dense-local-rank-two",
            scope_match="local_example",
            dense_score=0.72,
            dense_content_score=0.62,
            source_ids=["shared-source"],
        )

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [sibling, winner]
            ),
            [winner, sibling],
        )

    def test_dense_winner_may_not_pass_unrelated_fusion_winner(
        self,
    ) -> None:
        unrelated_fusion_winner = make_result(
            "unrelated-fusion-rank-one",
            scope_match="general_evidence",
            dense_score=0.60,
            dense_content_score=0.55,
            retrieval_mode="hybrid",
        )
        unrelated_fusion_winner.text_score = 5.0
        dense_winner = make_result(
            "dense-local-rank-two",
            scope_match="local_example",
            dense_score=0.72,
            dense_content_score=0.62,
        )

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [unrelated_fusion_winner, dense_winner]
            ),
            [],
        )



    def test_whole_piece_semantic_winner_keeps_normal_bundle(self) -> None:
        whole = make_result(
            "direct-whole-piece",
            scope_match="general_evidence",
            dense_score=0.70,
            dense_content_score=0.68,
        )
        local = make_result(
            "weaker-local",
            scope_match="local_example",
            dense_score=0.62,
            dense_content_score=0.60,
        )
        results = [whole, local]

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(results),
            [],
        )
        self.assertEqual(select_generation_evidence(results), results)

    def test_range_selected_results_stay_on_existing_scope_path(self) -> None:
        overlap = make_result(
            "overlap",
            scope_match="overlaps_query_range",
            dense_score=0.75,
            dense_content_score=0.72,
            measure_status="specific",
        )
        overlap.record["measure_range"] = [[1, 2]]
        context = make_result(
            "other-range",
            scope_match="other_range_context",
            dense_score=0.60,
            dense_content_score=0.59,
            measure_status="specific",
        )
        context.record["measure_range"] = [[20, 21]]

        self.assertEqual(
            semantically_dominant_no_range_local_expert_bundle(
                [overlap, context]
            ),
            [],
        )
        self.assertEqual(
            select_generation_evidence([overlap, context]),
            [overlap, context],
        )


class NoRangeSemanticAuthorityServiceTests(unittest.TestCase):
    @staticmethod
    def _index_with_results(results: list[SearchResult]) -> Any:
        class FixedIndex:
            retrieval_mode = "hybrid"
            configured_retrieval_mode = "hybrid"
            last_search_mode = "hybrid"

            @staticmethod
            def search(**_kwargs: Any) -> list[SearchResult]:
                return results

        return FixedIndex()

    @staticmethod
    def _case_results(
        *,
        piece: str,
        local_id: str,
        local_range: list[list[int]],
        local_answer: str,
    ) -> list[SearchResult]:
        local = make_result(
            local_id,
            scope_match="local_example",
            dense_score=0.68,
            dense_content_score=0.62,
            source_ids=["target-source"],
        )
        local.record.update(
            {
                "piece": piece,
                "work": piece,
                "measure_range": local_range,
                "answer": local_answer,
            }
        )
        distractor = make_result(
            "unrelated-whole-piece",
            scope_match="general_evidence",
            dense_score=0.54,
            dense_content_score=0.51,
        )
        distractor.record.update(
            {
                "piece": piece,
                "work": piece,
                "answer": "질문과 무관한 곡 전체 설명이다.",
            }
        )
        return [local, distractor]

    def test_dominant_flowery_expert_is_used_for_llm_generation(
        self,
    ) -> None:
        results = self._case_results(
            piece="in-flowery-clouds",
            local_id="in-flowery-clouds-ku-001",
            local_range=[[1, 2]],
            local_answer=(
                "곡 첫머리의 페르마타 유무는 악보에 따른 편곡의 "
                "차이일 수 있다. 두 경우 모두 허용된다."
            ),
        )
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "validate_generation_requirements"),
            mock.patch.object(
                qa,
                "_get_index",
                return_value=self._index_with_results(results),
            ),
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=(
                    "악보와 음원의 페르마타 길이가 다르더라도 편곡 "
                    "차이일 수 있으므로, 두 경우를 모두 허용해 해석할 "
                    "수 있습니다. [E1]"
                ),
            ) as generate,
        ):
            response = qa.ask(
                piece_id="in-flowery-clouds",
                question=(
                    "악보의 페르마타 표시가 음원에서 들리는 음 길이와 "
                    "일치하지 않을 때는 어떻게 해야 할까"
                ),
                measure_range=None,
                generate=True,
                top_k=6,
            )

        generate.assert_called_once()
        self.assertEqual(response["generation_mode"], "llm")
        self.assertEqual(response["answer_basis"], "retrieved_evidence")
        self.assertIsNone(response["unavailable_reason"])
        self.assertIsNone(response["generation_validation_warning"])
        self.assertIn("페르마타", response["answer"])
        self.assertIn("두 경우를 모두 허용", response["answer"])
        self.assertNotIn("[E1]", response["answer"])
        self.assertNotIn("범위 안내", response["answer"])
        self.assertNotIn("확인된 국소 예시", response["answer"])
        self.assertEqual(
            [item["id"] for item in response["evidence"]],
            ["in-flowery-clouds-ku-001"],
        )
        self.assertEqual(response["evidence"][0]["measure_ranges"], [[1, 2]])

    def test_overgeneralized_llm_draft_is_returned_with_warning(
        self,
    ) -> None:
        results = self._case_results(
            piece="una-voce-poco-fa",
            local_id="una-voce-poco-fa-ku-016",
            local_range=[[44, 55]],
            local_answer=(
                "긴 간주는 극적 전환 구간일 수 있다. 성악가는 앞뒤 "
                "가사의 흐름에 맞추어 표정과 움직임으로 반응해야 한다."
            ),
        )
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        overgeneralized = (
            "이 곡 전반의 모든 간주에서는 항상 같은 연기를 해야 "
            "한다. [E1]"
        )
        with (
            mock.patch.object(qa, "validate_generation_requirements"),
            mock.patch.object(
                qa,
                "_get_index",
                return_value=self._index_with_results(results),
            ),
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=overgeneralized,
            ) as generate,
        ):
            response = qa.ask(
                piece_id="una-voce-poco-fa",
                question="이 곡에 간주가 너무 길어서 그동안 뭘 해야할지 모르겠다",
                measure_range=None,
                generate=True,
                top_k=6,
            )

        generate.assert_called_once()
        self.assertEqual(response["generation_mode"], "llm")
        self.assertEqual(response["answer_basis"], "retrieved_evidence")
        self.assertIsNone(response["unavailable_reason"])
        self.assertEqual(
            response["generation_validation_warning"],
            "Generated answer overgeneralizes local evidence",
        )
        self.assertIn("항상 같은 연기를 해야 합니다", response["answer"])
        self.assertNotIn("[E1]", response["answer"])
        self.assertEqual(
            [item["id"] for item in response["evidence"]],
            ["una-voce-poco-fa-ku-016"],
        )
        self.assertEqual(
            response["evidence"][0]["measure_ranges"],
            [[44, 55]],
        )


if __name__ == "__main__":
    unittest.main()
