"""Scope-aware supplementation for repeated whole-piece change questions."""

from __future__ import annotations

import unittest

from soprano_qa.answer import (
    _answer_has_whole_piece_change_trajectory,
    _query_requests_repeated_whole_piece_change,
    _whole_piece_change_trajectory_supplement,
)
from soprano_qa.retrieval import SearchResult


def make_result(
    record_id: str,
    *,
    answer: str,
    scope_match: str,
    dense_score: float,
    dense_content_score: float,
) -> SearchResult:
    local = scope_match == "local_example"
    return SearchResult(
        record={
            "id": record_id,
            "evidence_type": "expert_annotation",
            "piece": "test-piece",
            "work": "Test Piece",
            "answer": answer,
            "measure_range": [[10, 12]] if local else [],
            "measure_scope": "local" if local else "global",
            "measure_status": "specific" if local else "whole_piece",
            "source_ids": ["source-" + record_id],
            "retrieval_aliases": [],
        },
        score=dense_score,
        text_score=0.0,
        measure_score=0.0,
        piece_score=1.0,
        scope_match=scope_match,
        dense_score=dense_score,
        dense_content_score=dense_content_score,
        fusion_score=dense_score,
        retrieval_mode="dense",
    )


class WholePieceChangeSupplementTests(unittest.TestCase):
    def test_query_detector_requires_scope_change_and_explanation(self) -> None:
        positives = (
            "박자가 여러번 바뀌는 이유는 무엇인가?",
            "곡 전반에서 박자가 계속 바뀌는 이유는 무엇일까?",
            (
                "처음부터 끝까지 박자가 여러 차례 변하는 흐름은 "
                "어떻게 설명할 수 있을까?"
            ),
        )
        for question in positives:
            with self.subTest(question=question):
                self.assertTrue(
                    _query_requests_repeated_whole_piece_change(question)
                )

        negatives = (
            "박자가 바뀌는 이유는 무엇일까?",
            "박자가 여러 번 바뀌나요?",
            "여러 마디에서 박자가 바뀌는 부분은 어떻게 연습할까?",
            "이 구간을 여러 번 반복해서 연습하는 이유는 무엇일까?",
        )
        for question in negatives:
            with self.subTest(question=question):
                self.assertFalse(
                    _query_requests_repeated_whole_piece_change(question)
                )

    def test_only_matching_strong_global_trajectory_is_added(self) -> None:
        query = "이 곡에서 박자가 여러번 바뀌는 이유는 무엇인가?"
        leader = make_result(
            "local-meter-change",
            answer=(
                "이 대목에서는 박자와 조성이 바뀌어 곡의 분위기를 "
                "전환한다."
            ),
            scope_match="local_example",
            dense_score=0.677,
            dense_content_score=0.587,
        )
        trajectory = make_result(
            "whole-meter-trajectory",
            answer=(
                "이 곡은 2/4박으로 시작한 뒤 박자가 12/8박과 "
                "3/4박으로 변화하면서 극적 효과를 고조시키고, 다시 "
                "2/4박으로 돌아온다. 이러한 박자 변화는 강약 대비를 "
                "선명하게 드러내기 위한 것이다."
            ),
            scope_match="general_evidence",
            dense_score=0.572,
            dense_content_score=0.562,
        )
        wrong_subject = make_result(
            "whole-key-choice",
            answer=(
                "곡은 원래 조성으로 시작한 뒤 다른 조성으로 변화하고 "
                "다시 돌아오며, 성악가는 음역에 맞게 조성을 바꿀 수 있다."
            ),
            scope_match="general_evidence",
            dense_score=0.64,
            dense_content_score=0.61,
        )
        weak = make_result(
            "weak-meter-trajectory",
            answer=trajectory.record["answer"],
            scope_match="general_evidence",
            dense_score=0.49,
            dense_content_score=0.60,
        )

        self.assertEqual(
            _whole_piece_change_trajectory_supplement(
                leader,
                [leader, wrong_subject, trajectory, weak],
                query=query,
                piece="test-piece",
            ),
            [trajectory],
        )

    def test_rule_is_not_meter_specific(self) -> None:
        query = "곡 전반에서 템포가 여러 번 변하는 이유는 무엇일까?"
        leader = make_result(
            "local-tempo-change",
            answer="이 대목에서는 긴장을 높이기 위해 템포가 바뀐다.",
            scope_match="local_example",
            dense_score=0.68,
            dense_content_score=0.59,
        )
        trajectory = make_result(
            "whole-tempo-trajectory",
            answer=(
                "곡은 빠른 템포로 시작한 뒤 템포가 느려지고 다시 "
                "처음 템포로 돌아온다. 이러한 템포 변화는 전체 "
                "구조의 대비를 드러내기 위한 것이다."
            ),
            scope_match="general_evidence",
            dense_score=0.58,
            dense_content_score=0.56,
        )

        self.assertEqual(
            _whole_piece_change_trajectory_supplement(
                leader,
                [leader, trajectory],
                query=query,
                piece="test-piece",
            ),
            [trajectory],
        )

    def test_candidate_requires_a_real_trajectory(self) -> None:
        self.assertTrue(
            _answer_has_whole_piece_change_trajectory(
                "곡은 2/4박으로 시작한 뒤 변하고 다시 처음 박자로 돌아온다."
            )
        )
        self.assertFalse(
            _answer_has_whole_piece_change_trajectory(
                "성악가는 자신의 음역에 맞게 조성을 바꿀 수 있다."
            )
        )


if __name__ == "__main__":
    unittest.main()
