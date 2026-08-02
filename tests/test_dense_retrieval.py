"""Focused tests for dense candidate recall and hybrid rank fusion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence

from soprano_qa.dense import (
    DenseRetrievalUnavailable,
    HybridIndex,
    build_retrieval_index,
    dense_document_text,
    dense_query_text,
    missing_dense_answer_constraints,
    retrieval_diagnostics,
)
from soprano_qa.retrieval import BM25Index


def make_record(
    record_id: str,
    text: str,
    *,
    piece: str = "test-piece",
    measure_range: List[List[int]] | None = None,
    measure_status: str | None = None,
    retrieval_eligible: bool = True,
) -> Dict[str, Any]:
    ranges = measure_range or []
    return {
        "id": record_id,
        "piece": piece,
        "topic": "performance",
        "answer": text,
        "relevance_text": text,
        "retrieval_text": text,
        "retrieval_aliases": [],
        "measure_range": ranges,
        "measure_scope": "local" if ranges else "global",
        "measure_status": measure_status
        or ("specific" if ranges else "whole_piece"),
        "evidence_type": "expert_annotation",
        "retrieval_eligible": retrieval_eligible,
    }


class FakeEmbedder:
    def __init__(
        self,
        *,
        document_vectors: Dict[str, Sequence[float]],
        query_vectors: Dict[str, Sequence[float]],
    ) -> None:
        self.cache_identity = "fake-embedder-v1"
        self.document_vectors = document_vectors
        self.query_vectors = query_vectors
        self.document_calls: List[List[str]] = []
        self.query_calls: List[str] = []

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.document_calls.append(list(texts))
        return [list(self.document_vectors[text]) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        self.query_calls.append(text)
        return list(self.query_vectors[text])


class DenseRetrievalTests(unittest.TestCase):
    def test_dense_lane_rescues_unseen_korean_paraphrase(self) -> None:
        relevant = make_record(
            "relevant",
            "피아노 반주 두 번째 박자 악센트는 송어의 움직임을 표현한다",
            measure_range=[[2, 5]],
        )
        unrelated = make_record(
            "unrelated",
            "성악가는 모음을 열고 자연스럽게 호흡한다",
            measure_range=[[2, 5]],
        )
        question = "건반 반주에서 둘째 박을 세게 치는 것은 어떤 장면을 그린 것인가?"
        self.assertEqual(
            BM25Index([relevant, unrelated]).search(
                question,
                piece="test-piece",
                measure_ranges=[[2, 5]],
            ),
            [],
        )
        embedder = FakeEmbedder(
            document_vectors={
                dense_document_text(relevant): [1.0, 0.0],
                dense_document_text(unrelated): [0.0, 1.0],
            },
            query_vectors={question: [1.0, 0.0]},
        )

        index = HybridIndex(
            [relevant, unrelated],
            embedder=embedder,
            dense_min_score=0.75,
        )
        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual([result.record["id"] for result in results], ["relevant"])
        self.assertEqual(results[0].semantic_match_type, "dense")
        self.assertEqual(results[0].retrieval_mode, "dense")
        self.assertAlmostEqual(results[0].dense_score, 1.0)
        self.assertEqual(results[0].scope_match, "overlaps_query_range")
        diagnostics = retrieval_diagnostics(index)
        self.assertEqual(diagnostics["last_search_mode"], "hybrid")
        self.assertEqual(diagnostics["query_route"], "hybrid")
        self.assertTrue(diagnostics["dense_attempted"])
        self.assertTrue(diagnostics["dense_contributed"])
        self.assertFalse(diagnostics["fallback_used"])

    def test_intentional_broad_query_uses_explicit_lexical_route(self) -> None:
        record = make_record(
            "record",
            "노래를 부를 때에는 호흡과 발음을 세심하게 준비한다",
        )
        index = HybridIndex(
            [record],
            embedder=FakeEmbedder(
                document_vectors={record["answer"]: [1.0, 0.0]},
                query_vectors={},
            ),
        )

        index.search(
            "이 곡을 잘 부르기 위한 팁은?",
            piece="test-piece",
        )

        diagnostics = retrieval_diagnostics(index)
        self.assertEqual(diagnostics["last_search_mode"], "lexical_route")
        self.assertEqual(diagnostics["query_route"], "lexical")
        self.assertEqual(
            diagnostics["route_reason"],
            "broad_performance_guidance_router",
        )
        self.assertFalse(diagnostics["dense_attempted"])
        self.assertFalse(diagnostics["dense_contributed"])
        self.assertFalse(diagnostics["fallback_used"])

    def test_dense_attempt_without_returned_match_is_reported(self) -> None:
        record = make_record("record", "피아노 호흡")
        question = "피아노 호흡"
        index = HybridIndex(
            [record],
            embedder=FakeEmbedder(
                document_vectors={record["answer"]: [1.0, 0.0]},
                query_vectors={question: [0.0, 1.0]},
            ),
        )

        results = index.search(question, piece="test-piece")

        self.assertEqual([result.record["id"] for result in results], ["record"])
        self.assertTrue(all(result.retrieval_mode == "lexical" for result in results))
        diagnostics = retrieval_diagnostics(index)
        self.assertEqual(
            diagnostics["last_search_mode"],
            "hybrid_no_dense_match",
        )
        self.assertEqual(diagnostics["query_route"], "hybrid")
        self.assertTrue(diagnostics["dense_attempted"])
        self.assertFalse(diagnostics["dense_contributed"])
        self.assertFalse(diagnostics["fallback_used"])

    def test_dense_threshold_rejects_unrelated_measure_overlap(self) -> None:
        record = make_record(
            "overlap",
            "프레이즈를 길게 연결하고 모음을 유지한다",
            measure_range=[[2, 5]],
        )
        question = "FIFA World Cup winner"
        embedder = FakeEmbedder(
            document_vectors={dense_document_text(record): [0.0, 1.0]},
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex(
            [record],
            embedder=embedder,
            dense_min_score=0.75,
        ).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(results, [])

    def test_calibrated_thresholds_keep_balanced_semantic_paraphrase(self) -> None:
        record = make_record(
            "relevant",
            "프레이즈를 연결하는 발성 문제를 교정한다",
        )
        record["relevance_text"] = "끊어지는 가창 문제와 해결 방법"
        question = "노래 프레이즈가 토막 나면 어떻게 바로잡을까?"
        relevance_score = 0.45
        content_score = 0.435
        embedder = FakeEmbedder(
            document_vectors={
                record["relevance_text"]: [
                    relevance_score,
                    (1.0 - relevance_score**2) ** 0.5,
                ],
                record["answer"]: [
                    content_score,
                    (1.0 - content_score**2) ** 0.5,
                ],
            },
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex([record], embedder=embedder).search(
            question,
            piece="test-piece",
        )

        self.assertEqual([result.record["id"] for result in results], ["relevant"])

    def test_dense_lane_preserves_hard_filters_and_scope_order(self) -> None:
        records = [
            make_record("overlap", "overlap text", measure_range=[[2, 5]]),
            make_record("other-range", "other text", measure_range=[[30, 32]]),
            make_record(
                "wrong-piece",
                "wrong piece text",
                piece="other-piece",
                measure_range=[[2, 5]],
            ),
            make_record(
                "ineligible",
                "ineligible text",
                measure_range=[[2, 5]],
                retrieval_eligible=False,
            ),
            make_record(
                "pending",
                "pending text",
                measure_status="waiting_for_review",
            ),
        ]
        question = "피아노 semantic paraphrase"
        embedder = FakeEmbedder(
            document_vectors={
                "overlap text": [0.8, 0.6],
                "other text": [1.0, 0.0],
                "wrong piece text": [1.0, 0.0],
                "ineligible text": [1.0, 0.0],
                "pending text": [1.0, 0.0],
            },
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex(
            records,
            embedder=embedder,
            dense_min_score=0.7,
        ).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
            top_k=10,
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["overlap"],
        )
        self.assertEqual(
            [result.scope_match for result in results],
            ["overlaps_query_range"],
        )

    def test_confident_other_range_subject_blocks_dense_topic_substitution(
        self,
    ) -> None:
        requested_subject = make_record(
            "accent-other-range",
            "피아노 반주 두 번째 박자 악센트는 물고기의 움직임을 표현한다",
            measure_range=[[30, 32]],
        )
        different_overlap = make_record(
            "different-overlap",
            "피아노 반주의 음형은 낚시꾼이 물을 흐트러뜨리는 장면을 표현한다",
            measure_range=[[2, 5]],
        )
        question = "피아노 반주 두 번째 박자 악센트는 무엇을 표현하는가?"
        embedder = FakeEmbedder(
            document_vectors={
                requested_subject["answer"]: [0.8, 0.6],
                different_overlap["answer"]: [1.0, 0.0],
            },
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex(
            [different_overlap, requested_subject],
            embedder=embedder,
            dense_min_score=0.7,
            dense_min_content_score=0.7,
        ).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["accent-other-range"],
        )
        self.assertEqual(results[0].scope_match, "other_range_context")

    def test_dense_text_excludes_provenance_only_retrieval_metadata(self) -> None:
        record = make_record("record", "genre and form")
        record["retrieval_text"] = (
            "genre and form\nBibliothèque nationale de France"
        )
        question = "unseen semantic wording"
        embedder = FakeEmbedder(
            document_vectors={"genre and form": [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0]},
        )

        HybridIndex([record], embedder=embedder).search(
            question,
            piece="test-piece",
        )

        self.assertEqual(embedder.document_calls, [["genre and form"]])
        self.assertNotIn("Bibliothèque", embedder.document_calls[0][0])

    def test_document_embeddings_are_computed_once_per_index(self) -> None:
        record = make_record("record", "semantic passage")
        questions = ("피아노 first query", "피아노 second query")
        embedder = FakeEmbedder(
            document_vectors={"semantic passage": [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0] for question in questions},
        )
        index = HybridIndex([record], embedder=embedder)

        for question in questions:
            index.search(question, piece="test-piece")

        self.assertEqual(embedder.document_calls, [["semantic passage"]])
        self.assertEqual(embedder.query_calls, list(questions))

    def test_blank_query_does_not_run_embedding_inference(self) -> None:
        record = make_record("record", "semantic passage")
        embedder = FakeEmbedder(
            document_vectors={"semantic passage": [1.0, 0.0]},
            query_vectors={},
        )
        index = HybridIndex([record], embedder=embedder)

        self.assertEqual(index.search("   ", piece="test-piece"), [])
        self.assertEqual(embedder.query_calls, [])

    def test_content_free_query_does_not_run_embedding_inference(self) -> None:
        record = make_record("record", "semantic passage")
        embedder = FakeEmbedder(
            document_vectors={"semantic passage": [1.0, 0.0]},
            query_vectors={},
        )
        index = HybridIndex([record], embedder=embedder)

        for question in ("무엇인가요?", "설명해 주세요", "2-5마디는?"):
            with self.subTest(question=question):
                self.assertEqual(
                    index.search(
                        question,
                        piece="test-piece",
                        measure_ranges=[[2, 5]],
                    ),
                    [],
                )

        self.assertEqual(embedder.query_calls, [])

    def test_dense_query_text_removes_piece_and_measure_routing(self) -> None:
        prepared = dense_query_text(
            "Die Forelle 2-5마디에서 반주의 약박은 무엇을 그리나요?",
            "die-forelle",
        )

        self.assertNotIn("Die Forelle", prepared)
        self.assertNotIn("2-5", prepared)
        self.assertIn("반주의 약박", prepared)

    def test_repeated_query_reuses_bounded_embedding_cache(self) -> None:
        record = make_record("record", "semantic passage")
        question = "피아노 unseen semantic question"
        embedder = FakeEmbedder(
            document_vectors={"semantic passage": [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0]},
        )
        index = HybridIndex([record], embedder=embedder)

        index.search(question, piece="test-piece")
        index.search(question, piece="test-piece")

        self.assertEqual(embedder.query_calls, [question])

    def test_exact_alias_precedes_dense_disagreement_in_same_scope(self) -> None:
        exact = make_record(
            "exact",
            "피아노 반주의 악센트는 물고기의 움직임을 나타낸다",
            measure_range=[[2, 5]],
        )
        question = "두 번째 박자의 악센트는 무엇을 나타내는가?"
        exact["retrieval_aliases"] = [question]
        exact["relevance_text"] = "%s\n%s" % (exact["answer"], question)
        distractor = make_record(
            "dense-first",
            "강한 박자는 극적인 장면을 표현한다",
            measure_range=[[2, 5]],
        )
        embedder = FakeEmbedder(
            document_vectors={
                dense_document_text(exact): [0.8, 0.6],
                exact["answer"]: [0.8, 0.6],
                dense_document_text(distractor): [1.0, 0.0],
            },
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex(
            [distractor, exact],
            embedder=embedder,
            dense_min_score=0.7,
        ).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(results[0].record["id"], "exact")
        self.assertGreater(results[0].alias_score, 0)
        self.assertEqual(results[0].retrieval_mode, "hybrid")

    def test_dense_can_compete_with_non_alias_lexical_overlap(self) -> None:
        lexical = make_record(
            "lexical",
            "트릴은 어떻게 노래해야 할까? 트릴은 가볍게 노래한다",
            measure_range=[[2, 5]],
        )
        semantic = make_record(
            "semantic",
            "장식음을 충분히 굴린 뒤 악보대로 마무리한다",
            measure_range=[[2, 5]],
        )
        question = "트릴은 어떻게 노래해야 할까?"
        embedder = FakeEmbedder(
            document_vectors={
                lexical["answer"]: [0.0, 1.0],
                semantic["answer"]: [1.0, 0.0],
            },
            query_vectors={question: [1.0, 0.0]},
        )

        index = HybridIndex(
            [lexical, semantic],
            embedder=embedder,
            dense_min_score=0.7,
            dense_min_content_score=0.7,
        )
        lexical_results = index.lexical.search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )
        self.assertEqual(
            [result.record["id"] for result in lexical_results],
            ["lexical"],
        )
        self.assertEqual(lexical_results[0].alias_score, 0.0)

        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["semantic", "lexical"],
        )
        self.assertEqual(results[0].semantic_match_type, "dense")

    def test_alias_heavy_dense_only_match_requires_answer_support(self) -> None:
        record = make_record(
            "alias-heavy",
            "피아노 반주의 악센트는 물고기의 움직임을 나타낸다",
            measure_range=[[2, 5]],
        )
        alias = "피아노 반주의 악센트는 어떤 화음으로 구성되는가?"
        record["relevance_text"] = "%s\n%s" % (record["answer"], alias)
        question = "건반에서 강조된 박은 어떤 화음으로 이루어졌나요?"
        embedder = FakeEmbedder(
            document_vectors={
                dense_document_text(record): [1.0, 0.0],
                record["answer"]: [0.8, 0.6],
            },
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex(
            [record],
            embedder=embedder,
            dense_min_score=0.7,
            dense_min_content_score=0.7,
            dense_max_alias_gap=0.1,
        ).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(results, [])

    def test_dense_only_requested_constraints_require_answer_support(self) -> None:
        record = make_record(
            "record",
            "피아노 반주의 악센트는 물고기의 움직임을 표현한다",
            measure_range=[[2, 5]],
        )
        questions = (
            "피아노 악센트의 권장 손가락 번호는 무엇인가요?",
            "피아노 트레몰로는 왼손인가요 오른손인가요?",
            "피아노 악센트는 어떤 화음인가요?",
            "피아노 꾸밈음은 정확히 몇 개의 음인가요?",
            "피아노 템포의 정확한 메트로놈 수치는 몇 BPM인가요?",
            "노래할 때 혀의 어느 근육을 사용하나요?",
            "피아노의 ff는 몇 데시벨인가요?",
            "고음의 정확한 주파수는 몇 Hz인가요?",
            "피아노 악센트는 어느 페이지에 있나요?",
            "피아노 악센트와 슈베르트의 출생 연도를 함께 알려 주세요.",
            "오시아 보표의 가로 길이는 몇 cm인가요?",
            "피아노 꾸밈음의 정확한 음정 이름은 무엇인가요?",
            "피아노 반주를 기타 타브 악보로 변환해 주세요.",
        )
        embedder = FakeEmbedder(
            document_vectors={record["answer"]: [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0] for question in questions},
        )
        index = HybridIndex([record], embedder=embedder)

        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(
                    missing_dense_answer_constraints(
                        question,
                        record["answer"],
                    )
                )
                self.assertEqual(
                    index.search(
                        question,
                        piece="test-piece",
                        measure_ranges=[[2, 5]],
                    ),
                    [],
                )

    def test_contextual_handedness_does_not_become_a_required_slot(self) -> None:
        record = make_record(
            "record",
            "악센트는 흐르는 물에서 뛰어오르는 송어를 표현한다",
            measure_range=[[2, 5]],
        )
        question = (
            "오른손과 왼손을 오가는 둘째 박의 강세로 피아노가 "
            "그려낸 것은?"
        )
        embedder = FakeEmbedder(
            document_vectors={record["answer"]: [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex([record], embedder=embedder).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(
            missing_dense_answer_constraints(question, record["answer"]),
            [],
        )
        self.assertEqual([result.record["id"] for result in results], ["record"])

    def test_dense_only_out_of_domain_similarity_is_rejected(self) -> None:
        record = make_record(
            "record",
            "페르마타에서는 음을 평소보다 길게 지속한다",
            measure_range=[[2, 5]],
        )
        question = "파스타 면을 몇 분 삶아야 할까?"
        embedder = FakeEmbedder(
            document_vectors={record["answer"]: [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex([record], embedder=embedder).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(results, [])

    def test_dense_only_cross_domain_relation_is_rejected(self) -> None:
        record = make_record(
            "record",
            "피아노 반주는 성악 선율과 긴밀한 관계를 맺는다",
            measure_range=[[2, 5]],
        )
        questions = (
            "반주와 국제관계는 어떤 관련이 있을까?",
            "음악 형식과 형식주의 철학은 같은가?",
        )
        embedder = FakeEmbedder(
            document_vectors={record["answer"]: [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0] for question in questions},
        )
        index = HybridIndex([record], embedder=embedder)

        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    index.search(
                        question,
                        piece="test-piece",
                        measure_ranges=[[2, 5]],
                    ),
                    [],
                )

    def test_notation_page_requires_subject_and_location_linkage(self) -> None:
        answer = "이 판본은 모두 10쪽으로 구성되어 있다"
        question = "D.S. al Coda 표시는 이 판본의 몇 페이지에 있나요?"

        self.assertIn(
            "notation_page_subject",
            missing_dense_answer_constraints(question, answer),
        )

    def test_bare_which_side_is_not_a_page_location_request(self) -> None:
        question = "전주가 있는 판본과 없는 판본 중 어느 쪽을 따라야 할까?"

        self.assertNotIn(
            "page_location",
            missing_dense_answer_constraints(question, "두 판본 모두 가능하다"),
        )
        self.assertIn(
            "page_location",
            missing_dense_answer_constraints(
                "전주 악보는 어느 페이지에 있을까?",
                "전주가 포함된 판본이다",
            ),
        )
        self.assertIn(
            "page_location",
            missing_dense_answer_constraints(
                "전주 악보는 몇 쪽에 있을까?",
                "전주가 포함된 판본이다",
            ),
        )

    def test_requested_constraint_accepts_explicit_answer_shape(self) -> None:
        cases = (
            (
                "fingering",
                "권장 손가락 번호는 무엇인가요?",
                "3번 손가락을 사용한다.",
            ),
            (
                "handedness",
                "트레몰로는 왼손인가요 오른손인가요?",
                "오른손으로 연주한다.",
            ),
            (
                "roman_harmony",
                "화성 진행을 로마 숫자로 분석해 주세요.",
                "진행은 I-V-vi-IV이다.",
            ),
            (
                "chord_identity",
                "어떤 화음으로 구성되나요?",
                "Cmaj7 코드이다.",
            ),
            (
                "exact_count",
                "정확히 몇 개의 음표인가요?",
                "세 개의 음표이다.",
            ),
            (
                "metronome_value",
                "정확한 메트로놈 수치는 몇 BPM인가요?",
                "72 BPM으로 연주한다.",
            ),
            (
                "tongue_muscle",
                "혀의 어느 근육을 사용하나요?",
                "이설근을 사용한다.",
            ),
            ("decibels", "몇 데시벨인가요?", "약 70 dB이다."),
            ("frequency", "주파수는 몇 Hz인가요?", "440 Hz이다."),
            (
                "page_location",
                "D.S. al Coda는 몇 페이지에 있나요?",
                "D.S. al Coda는 12페이지에 있다.",
            ),
            (
                "biographical_date",
                "작곡가의 출생 연도는 언제인가요?",
                "1797년에 태어났다.",
            ),
            (
                "physical_dimension",
                "오시아 보표의 가로 폭은 몇 cm인가요?",
                "가로 폭은 4.5 cm이다.",
            ),
            (
                "pitch_name",
                "정확한 음정 이름은 무엇인가요?",
                "완전 5도이다.",
            ),
            (
                "score_conversion",
                "기타 타브 악보로 변환해 주세요.",
                "기타 타브로 편곡한다.",
            ),
        )
        for name, question, answer in cases:
            with self.subTest(name=name):
                self.assertIn(
                    name,
                    missing_dense_answer_constraints(question, ""),
                )
                self.assertEqual(
                    missing_dense_answer_constraints(question, answer),
                    [],
                )

    def test_dense_only_other_range_candidate_is_not_substituted(self) -> None:
        record = make_record(
            "record",
            "피아노 악센트는 극적인 장면을 표현한다",
            measure_range=[[30, 32]],
        )
        question = "건반의 강세는 어떤 장면을 그린 것인가요?"
        embedder = FakeEmbedder(
            document_vectors={record["answer"]: [1.0, 0.0]},
            query_vectors={question: [1.0, 0.0]},
        )

        results = HybridIndex([record], embedder=embedder).search(
            question,
            piece="test-piece",
            measure_ranges=[[2, 5]],
        )

        self.assertEqual(results, [])

    def test_query_embedding_failure_returns_lexical_results(self) -> None:
        record = make_record("record", "breath phrasing")

        class FailingQueryEmbedder(FakeEmbedder):
            def embed_query(self, text: str) -> List[float]:
                raise DenseRetrievalUnavailable("query backend stopped")

        index = HybridIndex(
            [record],
            embedder=FailingQueryEmbedder(
                document_vectors={"breath phrasing": [1.0, 0.0]},
                query_vectors={},
            ),
        )

        results = index.search("breath phrasing", piece="test-piece")

        self.assertEqual([result.record["id"] for result in results], ["record"])
        self.assertIn("query backend stopped", str(index.last_dense_error))
        diagnostics = retrieval_diagnostics(index)
        self.assertEqual(diagnostics["last_search_mode"], "lexical_fallback")
        self.assertTrue(diagnostics["dense_attempted"])
        self.assertFalse(diagnostics["dense_contributed"])
        self.assertTrue(diagnostics["fallback_used"])

    def test_runtime_fallback_can_be_disabled(self) -> None:
        record = make_record("record", "breath phrasing")

        class FailingQueryEmbedder(FakeEmbedder):
            def embed_query(self, text: str) -> List[float]:
                raise DenseRetrievalUnavailable("query backend stopped")

        index = HybridIndex(
            [record],
            embedder=FailingQueryEmbedder(
                document_vectors={"breath phrasing": [1.0, 0.0]},
                query_vectors={},
            ),
            fallback_to_lexical=False,
        )

        with self.assertRaisesRegex(
            DenseRetrievalUnavailable,
            "query backend stopped",
        ):
            index.search("breath phrasing", piece="test-piece")

    def test_persistent_document_cache_is_reused(self) -> None:
        record = make_record("record", "semantic passage")
        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "embeddings.json")
            first = FakeEmbedder(
                document_vectors={"semantic passage": [1.0, 0.0]},
                query_vectors={},
            )
            HybridIndex([record], embedder=first, cache_path=cache_path)
            second = FakeEmbedder(
                document_vectors={},
                query_vectors={},
            )
            HybridIndex([record], embedder=second, cache_path=cache_path)

        self.assertEqual(len(first.document_calls), 1)
        self.assertEqual(second.document_calls, [])

    def test_document_text_change_invalidates_persistent_cache(self) -> None:
        original = make_record("record", "first passage")
        changed = make_record("record", "changed passage")
        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "embeddings.json")
            first = FakeEmbedder(
                document_vectors={"first passage": [1.0, 0.0]},
                query_vectors={},
            )
            HybridIndex([original], embedder=first, cache_path=cache_path)
            second = FakeEmbedder(
                document_vectors={"changed passage": [0.0, 1.0]},
                query_vectors={},
            )
            HybridIndex([changed], embedder=second, cache_path=cache_path)

        self.assertEqual(second.document_calls, [["changed passage"]])

    def test_factory_reports_missing_checkpoint_lexical_fallback(self) -> None:
        record = make_record("record", "breath phrasing")
        with tempfile.TemporaryDirectory() as directory:
            index = build_retrieval_index(
                [record],
                {
                    "embedding_model_path": str(Path(directory) / "missing.gguf"),
                    "embedding_cache_path": str(Path(directory) / "cache.json"),
                    "retrieval": {
                        "mode": "hybrid",
                        "fallback_to_lexical": True,
                    },
                },
            )

        diagnostics = retrieval_diagnostics(index)
        self.assertEqual(diagnostics["configured_mode"], "hybrid")
        self.assertEqual(diagnostics["active_mode"], "lexical")
        self.assertFalse(diagnostics["dense_available"])
        self.assertIn("Embedding checkpoint not found", diagnostics["fallback_reason"])
        self.assertEqual(diagnostics["last_search_mode"], "lexical_fallback")
        self.assertTrue(diagnostics["dense_attempted"])
        self.assertFalse(diagnostics["dense_contributed"])
        self.assertTrue(diagnostics["fallback_used"])

    def test_factory_does_not_hide_invalid_dense_configuration(self) -> None:
        record = make_record("record", "semantic passage")
        embedder = FakeEmbedder(
            document_vectors={"semantic passage": [1.0, 0.0]},
            query_vectors={},
        )

        with self.assertRaisesRegex(ValueError, "dense_min_score"):
            build_retrieval_index(
                [record],
                {
                    "retrieval": {
                        "mode": "hybrid",
                        "dense_min_score": 2.0,
                    }
                },
                embedder=embedder,
            )

    def test_hybrid_requires_positive_weights_for_both_lanes(self) -> None:
        record = make_record("record", "semantic passage")
        embedder = FakeEmbedder(
            document_vectors={"semantic passage": [1.0, 0.0]},
            query_vectors={},
        )

        for lexical_weight, dense_weight in ((0.0, 1.0), (1.0, 0.0)):
            with self.subTest(
                lexical_weight=lexical_weight,
                dense_weight=dense_weight,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "must both be positive",
                ):
                    HybridIndex(
                        [record],
                        embedder=embedder,
                        lexical_weight=lexical_weight,
                        dense_weight=dense_weight,
                    )

    def test_malformed_persistent_cache_is_regenerated(self) -> None:
        record = make_record("record", "semantic passage")
        embedder = FakeEmbedder(
            document_vectors={"semantic passage": [1.0, 0.0]},
            query_vectors={},
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.json"
            cache_path.write_text("[]", encoding="utf-8")

            HybridIndex(
                [record],
                embedder=embedder,
                cache_path=str(cache_path),
            )

        self.assertEqual(embedder.document_calls, [["semantic passage"]])


if __name__ == "__main__":
    unittest.main()
