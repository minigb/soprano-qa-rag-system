"""Focused tests for dense candidate recall and hybrid rank fusion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
from unittest import mock

from soprano_qa import answer as answer_cli
from soprano_qa.dense import (
    DenseRetrievalUnavailable,
    HybridIndex,
    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
    SOURCE_QUESTION_LED_MATCH_TYPE,
    build_retrieval_index,
    dense_document_text,
    dense_query_text,
    has_causal_answer_authority,
    is_causal_answer_request,
    missing_dense_answer_constraints,
    retrieval_diagnostics,
    source_family_answer_authority,
    validate_retrieval_requirements,
)
from soprano_qa.retrieval import BM25Index, SearchResult


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


class SourceFamilyAnswerAuthorityTests(unittest.TestCase):
    def test_grammatical_causal_requests_are_detected(self) -> None:
        questions = (
            "이 리듬을 쓴 이유를 알려 주세요.",
            "작곡가가 이 리듬을 사용한 배경을 알려줘.",
            "이 리듬을 사용하게 된 계기가 무엇인가요?",
            "이 리듬은 무엇 때문에 쓰였나요?",
            "작곡가는 이 리듬으로 무엇을 노린 건가요?",
            "무엇을 위해 쓰였나요?",
            "이렇게 쓴 이유가 있나요?",
            "그 의도는 뭔가요?",
            "쓴 까닭이 뭔지 알려 주세요.",
            "이 리듬은 어떤 역할을 하나요?",
            "이 리듬을 사용한 이유와 노래하는 방법을 알려 주세요.",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(is_causal_answer_request(question))

    def test_incidental_causal_substrings_do_not_change_procedural_intent(
        self,
    ) -> None:
        procedural_questions = (
            "의도치 않게 부점 리듬이 흐트러질 때 어떻게 노래해야 할까?",
            "이유 없이 리듬이 밀리지 않게 하려면 어떻게 해야 할까?",
            "부점 리듬이 왜곡되지 않게 하려면 어떻게 해야 할까?",
            "특별한 이유와 상관없이 부점 리듬을 어떻게 유지할까요?",
            "작곡가의 의도와 무관하게 부점 리듬을 어떻게 해석할까요?",
            "무슨 의미인지 몰라도 부점 리듬을 어떻게 처리할까요?",
        )
        for question in procedural_questions:
            with self.subTest(question=question):
                self.assertFalse(is_causal_answer_request(question))

    def test_incidental_or_negated_nouns_do_not_authorize_causal_answer(
        self,
    ) -> None:
        answers = (
            "작곡가의 의도를 생각하며 정확한 리듬으로 노래해야 한다.",
            "특별한 이유 없이 흔들리지 않도록 박자를 세어 연습한다.",
            "리듬이 왜곡되지 않게 짧은 음표를 뒤에 놓아 부른다.",
            (
                "그 이유는 알 수 없습니다. 음악을 정확히 "
                "표현해야 합니다."
            ),
            (
                "작곡가의 의도는 확인되지 않았습니다. 이 리듬은 "
                "성격을 나타내며 정확히 노래해야 합니다."
            ),
        )
        for index, answer in enumerate(answers):
            record = make_record(
                f"incidental-causal-{index}",
                answer,
                measure_range=[[63, 63]],
            )
            result = SearchResult(
                record=record,
                score=1.0,
                text_score=1.0,
                measure_score=1.0,
                piece_score=1.0,
                scope_match="overlaps_query_range",
            )
            with self.subTest(answer=answer):
                self.assertFalse(has_causal_answer_authority(result))

    def test_explanatory_causal_forms_remain_authoritative(self) -> None:
        answers = (
            (
                "바뀌는 음형은 물고기를 잡는 모습을 그려 긴장감과 "
                "분위기 전환을 만든다."
            ),
            (
                "작곡가는 박자와 조성을 바꾸어 곡의 분위기를 "
                "전환한다."
            ),
            (
                "이는 문장의 표현을 극대화한다."
            ),
            (
                "둘째 박의 악센트는 송어가 뛰어노는 모습을 표현한 "
                "것으로 볼 수 있다."
            ),
            (
                "유명한 아리아는 한 곡만 따로 악보로 만들기도 하므로 "
                "여러 버전이 생길 수 있다."
            ),
            (
                "꾸밈음이 쓰이는 이유는 곡마다 다르지만 이 시대에는 "
                "화려한 기교를 중요하게 여겼다."
            ),
            "피아노 반주의 악센트는 물고기의 움직임을 나타낸다.",
            (
                "이 곡은 결코 쉬운 곡은 아니다. 상대적으로 템포가 "
                "빠르고 빠르게 발음해야 하는 단어가 많다."
            ),
        )
        for index, answer in enumerate(answers):
            with self.subTest(answer=answer):
                record = make_record(
                    f"causal-form-{index}",
                    answer,
                    measure_range=[[10, 20]],
                )
                record["source_ids"] = [f"source-{index}"]
                anchor = SearchResult(
                    record=record,
                    score=1.0,
                    text_score=1.0,
                    measure_score=1.0,
                    piece_score=1.0,
                    scope_match="local_example",
                    concept_coverage=1.0,
                    content_concept_coverage=1.0,
                    semantic_match_type=SOURCE_QUESTION_LED_MATCH_TYPE,
                    dense_score=0.7,
                    dense_content_score=0.7,
                    retrieval_mode="hybrid",
                )
                self.assertTrue(has_causal_answer_authority(anchor))
                self.assertIs(
                    source_family_answer_authority(
                        anchor,
                        [anchor],
                        query="왜 이렇게 음악을 바꾼 것일까?",
                    ),
                    anchor,
                )

    def test_strict_answer_side_sibling_displaces_dense_rescue_anchor(
        self,
    ) -> None:
        procedural_record = make_record(
            "procedural-split",
            "짧은 음표를 다음 음 가까이에 놓아 부르는 것이 좋다",
            measure_range=[[14, 15]],
        )
        procedural_record["source_ids"] = ["shared-source"]
        direct_record = make_record(
            "direct-causal-split",
            "과장된 리듬은 등장인물의 성격을 나타내기 위해 사용한다",
            measure_range=[[14, 15]],
        )
        direct_record["source_ids"] = ["shared-source"]
        anchor = SearchResult(
            record=procedural_record,
            score=29.0,
            text_score=28.0,
            measure_score=1.0,
            piece_score=1.0,
            scope_match="local_example",
            concept_coverage=1.0,
            content_concept_coverage=0.25,
            semantic_match_type=SOURCE_QUESTION_LED_MATCH_TYPE,
            dense_score=0.75,
            dense_content_score=0.68,
            retrieval_mode="hybrid",
        )
        direct = SearchResult(
            record=direct_record,
            score=41.0,
            text_score=40.0,
            measure_score=1.0,
            piece_score=1.0,
            scope_match="local_example",
            concept_coverage=1.0,
            content_concept_coverage=1.0,
            semantic_match_type="strict",
            dense_score=0.69,
            dense_content_score=0.65,
            retrieval_mode="hybrid",
        )

        self.assertIs(
            source_family_answer_authority(
                anchor,
                [anchor, direct],
                query="왜 이 리듬을 사용한 것일까?",
            ),
            direct,
        )

        self.assertIs(
            source_family_answer_authority(
                anchor,
                [anchor, direct],
                query="이 리듬은 어떻게 불러야 할까?",
            ),
            anchor,
        )

        direct.record["source_ids"] = ["different-source"]
        self.assertIsNone(
            source_family_answer_authority(
                anchor,
                [anchor, direct],
                query="왜 이 리듬을 사용한 것일까?",
                known_causal_family=True,
            )
        )

    def test_causal_request_without_same_scope_causal_answer_abstains(
        self,
    ) -> None:
        procedural_record = make_record(
            "procedural-only-split",
            "짧은 음표를 다음 음 가까이에 놓아 부르는 것이 좋다",
            measure_range=[[63, 63]],
        )
        procedural_record["source_ids"] = ["shared-source"]
        anchor = SearchResult(
            record=procedural_record,
            score=29.0,
            text_score=28.0,
            measure_score=1.0,
            piece_score=1.0,
            scope_match="overlaps_query_range",
            concept_coverage=1.0,
            content_concept_coverage=0.25,
            semantic_match_type=SOURCE_QUESTION_LED_MATCH_TYPE,
            dense_score=0.75,
            dense_content_score=0.68,
            retrieval_mode="hybrid",
        )

        self.assertIsNone(
            source_family_answer_authority(
                anchor,
                [anchor],
                query="왜 이 리듬을 사용한 것일까?",
                known_causal_family=True,
            )
        )
        self.assertIs(
            source_family_answer_authority(
                anchor,
                [anchor],
                query="이 리듬은 어떻게 불러야 할까?",
            ),
            anchor,
        )

    def test_weak_or_ambiguous_causal_siblings_do_not_authorize_anchor(
        self,
    ) -> None:
        procedural_record = make_record(
            "procedural-anchor",
            "짧은 음표를 다음 음 가까이에 놓아 부르는 것이 좋다",
            measure_range=[[14, 15]],
        )
        procedural_record["source_ids"] = ["shared-source"]
        anchor = SearchResult(
            record=procedural_record,
            score=29.0,
            text_score=28.0,
            measure_score=1.0,
            piece_score=1.0,
            scope_match="local_example",
            concept_coverage=1.0,
            content_concept_coverage=0.25,
            semantic_match_type=SOURCE_QUESTION_LED_MATCH_TYPE,
            dense_score=0.75,
            dense_content_score=0.68,
            retrieval_mode="hybrid",
        )

        def causal_sibling(record_id: str, coverage: float) -> SearchResult:
            record = make_record(
                record_id,
                "등장인물의 성격을 나타내기 위해 이 리듬을 사용한다",
                measure_range=[[14, 15]],
            )
            record["source_ids"] = ["shared-source"]
            return SearchResult(
                record=record,
                score=0.0,
                text_score=0.0,
                measure_score=1.0,
                piece_score=1.0,
                scope_match="local_example",
                concept_coverage=0.7,
                content_concept_coverage=coverage,
                semantic_match_type="dense",
                dense_score=0.66,
                dense_content_score=0.61,
                retrieval_mode="dense",
            )

        weak = causal_sibling("weak-causal", 0.40)
        self.assertIsNone(
            source_family_answer_authority(
                anchor,
                [anchor, weak],
                query="왜 이 리듬을 사용한 것일까?",
            )
        )

        first = causal_sibling("first-causal", 0.70)
        second = causal_sibling("second-causal", 0.72)
        self.assertIsNone(
            source_family_answer_authority(
                anchor,
                [anchor, first, second],
                query="왜 이 리듬을 사용한 것일까?",
            )
        )


def source_question_led_index(
    *,
    query: str = "악센트 표시가 첫 박이 아닌 이유는 무엇일까?",
    target_updates: Dict[str, Any] | None = None,
    target_relevance: float = 0.68,
    target_content: float = 0.41,
    competitor_relevance: float = 0.55,
) -> tuple[HybridIndex, str]:
    """Build a dense-only source-question fixture with separate views."""

    target = make_record(
        "source-question-target",
        "경쾌한 표현을 만들기 위해 첫 번째 박이 아닌 위치를 사용한다",
        measure_range=[[68, 69]],
    )
    target.update(
        {
            "relevance_text": "target semantic source question",
            "retrieval_text": "target semantic source question",
            "retrieval_aliases": ["authoritative source question"],
            "source_ids": ["target-source"],
        }
    )
    target.update(target_updates or {})
    competitor = make_record(
        "other-source-competitor",
        "competitor curated answer",
    )
    competitor.update(
        {
            "relevance_text": "competitor semantic passage",
            "retrieval_text": "competitor semantic passage",
            "source_ids": ["other-source"],
        }
    )

    def score_vector(score: float) -> List[float]:
        return [score, (1.0 - score**2) ** 0.5]

    embedder = FakeEmbedder(
        document_vectors={
            dense_document_text(target): score_vector(target_relevance),
            target["answer"]: score_vector(target_content),
            dense_document_text(competitor): score_vector(
                competitor_relevance
            ),
            competitor["answer"]: score_vector(0.50),
        },
        query_vectors={query: [1.0, 0.0]},
    )
    return HybridIndex([target, competitor], embedder=embedder), query


class DenseRetrievalTests(unittest.TestCase):
    def test_cli_missing_hybrid_model_exits_before_corpus_work(self) -> None:
        settings = {
            "embedding_model_path": "/does/not/exist.gguf",
            "retrieval": {"mode": "hybrid"},
        }
        with (
            mock.patch.object(
                answer_cli,
                "parse_args",
                return_value=SimpleNamespace(settings=None),
            ),
            mock.patch.object(
                answer_cli,
                "load_settings",
                return_value=settings,
            ),
            mock.patch.object(answer_cli, "ensure_corpus") as ensure_corpus,
        ):
            with self.assertRaisesRegex(
                SystemExit,
                r"download_embedding_model\.py.*never falls back silently",
            ):
                answer_cli.main()

        ensure_corpus.assert_not_called()

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
        ]
        question = "피아노 semantic paraphrase"
        embedder = FakeEmbedder(
            document_vectors={
                "overlap text": [0.8, 0.6],
                "other text": [1.0, 0.0],
                "wrong piece text": [1.0, 0.0],
                "ineligible text": [1.0, 0.0],
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

    def test_source_question_led_local_expert_bypasses_only_its_two_gates(
        self,
    ) -> None:
        index, question = source_question_led_index()

        results = index.search(
            question,
            piece="test-piece",
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["source-question-target"],
        )
        self.assertEqual(
            results[0].semantic_match_type,
            SOURCE_QUESTION_LED_MATCH_TYPE,
        )
        self.assertLess(
            results[0].dense_content_score,
            index.dense_min_content_score,
        )
        self.assertGreater(
            results[0].dense_score - results[0].dense_content_score,
            index.dense_max_alias_gap,
        )

    def test_source_question_led_expert_can_ground_its_selected_range(
        self,
    ) -> None:
        index, question = source_question_led_index()

        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[68, 69]],
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["source-question-target"],
        )
        self.assertEqual(
            results[0].semantic_match_type,
            SOURCE_QUESTION_LED_MATCH_TYPE,
        )
        self.assertEqual(results[0].scope_match, "overlaps_query_range")

    def test_single_procedural_family_cannot_answer_causal_request(
        self,
    ) -> None:
        question = "왜 이 리듬을 사용한 것일까?"
        index, _ = source_question_led_index(
            query=question,
            target_updates={
                "answer": "짧은 음표를 다음 음 가까이에 놓아 부르는 것이 좋다",
            },
        )

        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[68, 69]],
        )

        self.assertEqual(results, [])
        self.assertEqual(
            index.last_route_reason,
            SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
        )
        self.assertEqual(
            [
                result.record["id"]
                for result in index.last_answer_candidates
            ],
            ["source-question-target"],
        )

    def test_single_causal_family_cannot_answer_mixed_request(
        self,
    ) -> None:
        question = "왜 이 리듬을 사용했고 어떻게 노래해야 할까?"
        index, _ = source_question_led_index(
            query=question,
            target_updates={
                "answer": (
                    "경쾌한 표현을 만들기 위해 첫 번째 박이 아닌 "
                    "위치를 사용한다"
                ),
            },
        )

        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[68, 69]],
        )

        self.assertEqual(results, [])
        self.assertEqual(
            index.last_route_reason,
            SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
        )

    def test_source_question_led_focus_excludes_other_source_dense_hits(
        self,
    ) -> None:
        index, question = source_question_led_index(
            query="음악의 경쾌한 표현을 어떻게 이해할까?"
        )

        results = index.search(question, piece="test-piece")

        self.assertEqual(
            [result.record["id"] for result in results],
            ["source-question-target"],
        )

    def test_accent_placement_rejects_unrelated_overlapping_ossia(
        self,
    ) -> None:
        ossia = make_record(
            "ossia",
            (
                "작은 보표는 장식적 대안 선율이며 기본 선율과 "
                "동시에 부르지 않고 둘 중 하나를 선택한다"
            ),
            measure_range=[[90, 108]],
        )
        question = "첫 박을 벗어난 위치에 악센트를 표시한 이유는 무엇일까?"
        score = 0.50
        vector = [score, (1.0 - score**2) ** 0.5]
        index = HybridIndex(
            [ossia],
            embedder=FakeEmbedder(
                document_vectors={
                    dense_document_text(ossia): vector,
                    ossia["answer"]: vector,
                },
                query_vectors={question: [1.0, 0.0]},
            ),
        )

        self.assertEqual(
            missing_dense_answer_constraints(question, ossia["answer"]),
            ["accent_placement"],
        )
        self.assertEqual(
            index.search(
                question,
                piece="test-piece",
                measure_ranges=[[92, 93]],
            ),
            [],
        )

    def test_near_tied_unrelated_dense_only_range_candidates_abstain(
        self,
    ) -> None:
        first = make_record(
            "first-source",
            "공명을 유지하며 높은 음을 준비한다",
            measure_range=[[10, 20]],
        )
        first["source_ids"] = ["source-a"]
        second = make_record(
            "second-source",
            "호흡을 나누어 긴 선율을 마무리한다",
            measure_range=[[10, 20]],
        )
        second["source_ids"] = ["source-b"]
        question = "상행 도약을 자연스럽게 소화하려면 어떻게 해야 할까?"

        def score_vector(score: float) -> List[float]:
            return [score, (1.0 - score**2) ** 0.5]

        index = HybridIndex(
            [first, second],
            embedder=FakeEmbedder(
                document_vectors={
                    first["answer"]: score_vector(0.56),
                    second["answer"]: score_vector(0.555),
                },
                query_vectors={question: [1.0, 0.0]},
            ),
        )

        self.assertEqual(
            index.search(
                question,
                piece="test-piece",
                measure_ranges=[[12, 14]],
            ),
            [],
        )
        self.assertEqual(
            index.last_route_reason,
            "ambiguous_dense_only_range",
        )
        self.assertEqual(
            {
                result.record["id"]
                for result in index.last_answer_candidates
            },
            {"first-source", "second-source"},
        )

    def test_clear_dense_only_range_lead_is_not_treated_as_ambiguous(
        self,
    ) -> None:
        first = make_record(
            "clear-winner",
            "공명을 유지하며 높은 음을 준비한다",
            measure_range=[[10, 20]],
        )
        first["source_ids"] = ["source-a"]
        second = make_record(
            "runner-up",
            "호흡을 나누어 긴 선율을 마무리한다",
            measure_range=[[10, 20]],
        )
        second["source_ids"] = ["source-b"]
        question = "상행 도약을 자연스럽게 소화하려면 어떻게 해야 할까?"

        def score_vector(score: float) -> List[float]:
            return [score, (1.0 - score**2) ** 0.5]

        index = HybridIndex(
            [first, second],
            embedder=FakeEmbedder(
                document_vectors={
                    first["answer"]: score_vector(0.60),
                    second["answer"]: score_vector(0.575),
                },
                query_vectors={question: [1.0, 0.0]},
            ),
        )

        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[12, 14]],
        )

        self.assertEqual(results[0].record["id"], "clear-winner")
        self.assertEqual(index.dense_only_ambiguity_margin, 0.025)
        self.assertEqual(index.last_route_reason, "dense_result_returned")

    def test_near_tied_same_source_dense_range_siblings_are_preserved(
        self,
    ) -> None:
        first = make_record(
            "first-sibling",
            "공명을 유지하며 높은 음을 준비한다",
            measure_range=[[10, 20]],
        )
        second = make_record(
            "second-sibling",
            "호흡을 나누어 긴 선율을 마무리한다",
            measure_range=[[10, 20]],
        )
        for record in (first, second):
            record["source_ids"] = ["shared-source"]
        question = "상행 도약을 자연스럽게 소화하려면 어떻게 해야 할까?"

        def score_vector(score: float) -> List[float]:
            return [score, (1.0 - score**2) ** 0.5]

        index = HybridIndex(
            [first, second],
            embedder=FakeEmbedder(
                document_vectors={
                    first["answer"]: score_vector(0.56),
                    second["answer"]: score_vector(0.555),
                },
                query_vectors={question: [1.0, 0.0]},
            ),
        )

        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[12, 14]],
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["first-sibling", "second-sibling"],
        )

    def test_direct_alias_bypasses_dense_only_range_ambiguity_gate(
        self,
    ) -> None:
        direct = make_record(
            "direct",
            (
                "상행 도약을 자연스럽게 소화하려면 공명을 유지하며 "
                "높은 음을 준비한다"
            ),
            measure_range=[[10, 20]],
        )
        question = "상행 도약을 자연스럽게 소화하려면 어떻게 해야 할까?"
        direct["retrieval_aliases"] = [question]
        direct["source_ids"] = ["source-a"]
        competitor = make_record(
            "competitor",
            "호흡을 나누어 긴 선율을 마무리한다",
            measure_range=[[10, 20]],
        )
        competitor["source_ids"] = ["source-b"]

        def score_vector(score: float) -> List[float]:
            return [score, (1.0 - score**2) ** 0.5]

        index = HybridIndex(
            [direct, competitor],
            embedder=FakeEmbedder(
                document_vectors={
                    dense_document_text(direct): score_vector(0.56),
                    direct["answer"]: score_vector(0.56),
                    competitor["answer"]: score_vector(0.555),
                },
                query_vectors={question: [1.0, 0.0]},
            ),
        )

        results = index.search(
            question,
            piece="test-piece",
            measure_ranges=[[12, 14]],
        )

        self.assertEqual(results[0].record["id"], "direct")
        self.assertGreater(results[0].alias_score, 0)
        self.assertNotEqual(
            index.last_route_reason,
            "ambiguous_dense_only_range",
        )

    def test_source_question_led_path_revalidates_source_route_guards(
        self,
    ) -> None:
        cases = {
            "no-alias": {"retrieval_aliases": []},
            "no-source": {"source_ids": []},
            "no-range": {
                "measure_range": [],
                "measure_scope": "global",
                "measure_status": "whole_piece",
            },
        }
        for label, updates in cases.items():
            with self.subTest(label=label):
                index, question = source_question_led_index(
                    target_updates=updates,
                )

                results = index.search(question, piece="test-piece")

                self.assertNotIn(
                    "source-question-target",
                    [result.record["id"] for result in results],
                )

    def test_hybrid_index_rejects_legacy_review_records(self) -> None:
        cases = {
            "review-warning": {
                "retrieval_review_warning": "check this record"
            },
            "nonfinal-status": {"rewrite_status": "needs_review"},
        }
        for label, updates in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError,
                    "finalized release expert records",
                ):
                    source_question_led_index(target_updates=updates)

    def test_source_question_led_path_requires_clear_other_source_lead(
        self,
    ) -> None:
        index, question = source_question_led_index(
            competitor_relevance=0.60,
        )

        results = index.search(question, piece="test-piece")

        self.assertNotIn(
            "source-question-target",
            [result.record["id"] for result in results],
        )

    def test_source_question_led_path_preserves_explicit_constraints(
        self,
    ) -> None:
        question = "악센트는 정확히 몇 데시벨로 연주해야 할까?"
        index, _ = source_question_led_index(query=question)

        self.assertEqual(
            missing_dense_answer_constraints(
                question,
                "target curated answer",
            ),
            ["decibels"],
        )
        results = index.search(question, piece="test-piece")
        self.assertNotIn(
            "source-question-target",
            [result.record["id"] for result in results],
        )

    def test_source_question_led_path_preserves_domain_anchor_guard(
        self,
    ) -> None:
        question = "파스타 면은 몇 분 동안 삶아야 할까?"
        index, _ = source_question_led_index(query=question)

        self.assertEqual(index.search(question, piece="test-piece"), [])

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

    def test_query_embedding_failure_never_falls_back(self) -> None:
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

        with self.assertRaisesRegex(
            DenseRetrievalUnavailable,
            "query backend stopped",
        ):
            index.search("breath phrasing", piece="test-piece")

        self.assertIn("query backend stopped", str(index.last_dense_error))
        diagnostics = retrieval_diagnostics(index)
        self.assertEqual(diagnostics["last_search_mode"], "hybrid_error")
        self.assertTrue(diagnostics["dense_attempted"])
        self.assertFalse(diagnostics["dense_contributed"])

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

    def test_hybrid_preflight_rejects_missing_checkpoint(self) -> None:
        record = make_record("record", "breath phrasing")
        with tempfile.TemporaryDirectory() as directory:
            settings = {
                "embedding_model_path": str(Path(directory) / "missing.gguf"),
                "embedding_cache_path": str(Path(directory) / "cache.json"),
                "retrieval": {
                    "mode": "hybrid",
                },
            }
            with self.assertRaisesRegex(
                DenseRetrievalUnavailable,
                r"download_embedding_model\.py.*never falls back silently",
            ):
                validate_retrieval_requirements(settings)
            with self.assertRaisesRegex(
                DenseRetrievalUnavailable,
                "Required hybrid-retrieval embedding checkpoint not found",
            ):
                build_retrieval_index(
                    [record],
                    settings,
                )

    def test_explicit_lexical_mode_needs_no_embedding_checkpoint(self) -> None:
        record = make_record("record", "breath phrasing")
        index = build_retrieval_index(
            [record],
            {
                "embedding_model_path": "/does/not/exist.gguf",
                "retrieval": {
                    "mode": "lexical",
                },
            },
        )

        results = index.search(
            "breath phrasing",
            piece="test-piece",
        )
        self.assertEqual(
            [result.record["id"] for result in results],
            ["record"],
        )
        diagnostics = retrieval_diagnostics(index)
        self.assertEqual(diagnostics["configured_mode"], "lexical")

    def test_hybrid_preflight_rejects_missing_embedding_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "embedding.gguf"
            model_path.write_bytes(b"checkpoint")
            settings = {
                "embedding_model_path": str(model_path),
                "retrieval": {
                    "mode": "hybrid",
                },
            }
            with mock.patch(
                "soprano_qa.dense._require_llama_cpp_embedding_backend",
                side_effect=DenseRetrievalUnavailable(
                    "llama-cpp-python is unavailable"
                ),
            ):
                with self.assertRaisesRegex(
                    DenseRetrievalUnavailable,
                    "llama-cpp-python is unavailable",
                ):
                    validate_retrieval_requirements(settings)

    def test_hybrid_preflight_rejects_legacy_fallback_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "embedding.gguf"
            model_path.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(
                ValueError,
                "fallback_to_lexical is obsolete and unsupported",
            ):
                validate_retrieval_requirements(
                    {
                        "embedding_model_path": str(model_path),
                        "retrieval": {
                            "mode": "hybrid",
                            "fallback_to_lexical": True,
                        },
                    }
                )

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
