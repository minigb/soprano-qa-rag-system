"""Integration checks for the reusable QA service facade."""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from soprano_qa import service as qa
from soprano_qa.answer import (
    GeneratedAnswerRejected,
    NO_CORPUS_EVIDENCE_MESSAGE,
    answer_uses_formal_polite_korean,
)
from soprano_qa.dense import (
    AMBIGUOUS_DENSE_ONLY_RANGE_REASON,
    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
)
from soprano_qa.retrieval import SearchResult
from soprano_qa.settings import load_settings


class ServicePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        """Isolate generation while exercising configured strict retrieval."""
        self._generation_preflight_patch = mock.patch.object(
            qa,
            "validate_generation_requirements",
        )
        self.generation_preflight = self._generation_preflight_patch.start()

    def tearDown(self) -> None:
        self._generation_preflight_patch.stop()

    def assert_public_answer_contract(self, result: dict) -> None:
        """Check display prose while keeping provenance in structured evidence."""

        answer = result["answer"]
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip())
        self.assertTrue(
            answer_uses_formal_polite_korean(answer),
            msg=f"answer is not consistently formal-polite Korean: {answer!r}",
        )
        compact = re.sub(r"[\s*_~`]", "", answer)
        self.assertNotIn("제공된검색근거", compact)
        for internal_label in (
            "범위안내:",
            "확인된국소예시",
            "일반참고근거:",
            "citation_label",
            "evidence_role",
            "query_scope_relation",
            "measure_range",
            "measure_scope",
            "scope_match",
            "usage_constraint",
            "source_ids",
            "web_source_ids",
            "claim_ids",
            "annotators",
            "rewrite_status",
            "measure_status",
            "retrieval_review_warning",
            "record_id",
            "knowledge_unit_id",
        ):
            self.assertNotIn(internal_label, compact)
        self.assertNotRegex(
            answer,
            r"(?:검색(?:된|한)?|제공(?:된|한)?)\s*(?:코퍼스|말뭉치)",
        )
        self.assertNotIn("로컬 LLM", answer)
        self.assertNotRegex(
            answer,
            r"[\[【]\s*(?:E(?:vidence)?|근거|출처)\s*"
            r"(?:[:：#._–—-]\s*)?(?:no\.?\s*)?[0-9]+",
        )
        self.assertNotRegex(
            answer,
            r"(?<![A-Za-z0-9_-])(?:sqa-|webchunk-|source[_ -]?id\b)",
        )
        for evidence in result.get("evidence", []):
            evidence_id = evidence.get("id")
            self.assertIsInstance(evidence_id, str)
            self.assertTrue(evidence_id)
            self.assertNotIn(evidence_id, answer)
            for key in ("source_ids", "web_source_ids", "claim_ids"):
                for provenance_id in evidence.get(key, []):
                    self.assertNotIn(str(provenance_id), answer)
        if result.get("measure_range") is None:
            for evidence in result.get("evidence", []):
                if evidence.get("scope_match") != "local_example":
                    continue
                for start, end in evidence.get("measure_ranges", []):
                    if start == end:
                        locator_patterns = (
                            rf"(?<!\d){start}\s*(?:번째\s*)?마디",
                            rf"마디\s*{start}(?!\d)",
                        )
                    else:
                        locator_patterns = (
                            rf"(?<!\d){start}\s*[-–—~]\s*{end}\s*마디",
                            rf"마디\s*{start}\s*[-–—~]\s*{end}(?!\d)",
                            rf"(?<!\d){start}\s*마디부터\s*{end}\s*마디",
                        )
                    for pattern in locator_patterns:
                        self.assertNotRegex(answer, pattern)

    def test_generation_preflight_error_propagates_before_retrieval(self) -> None:
        failure = RuntimeError("required generation checkpoint is missing")
        with (
            mock.patch.object(
                qa,
                "validate_generation_requirements",
                side_effect=failure,
            ) as validate,
            mock.patch.object(qa, "_get_index") as get_index,
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            with self.assertRaises(RuntimeError) as raised:
                qa.ask(
                    piece_id="die-forelle",
                    question="어떻게 표현해야 하나요?",
                    measure_range=None,
                    generate=True,
                )

        self.assertIs(raised.exception, failure)
        validate.assert_called_once_with(
            qa.SETTINGS["model_path"],
            qa.SETTINGS["llm"],
        )
        get_index.assert_not_called()
        generate.assert_not_called()

    def test_missing_hybrid_checkpoint_fails_before_corpus_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing_model = Path(temporary) / "missing-embedding.gguf"
            settings = {
                **qa.SETTINGS,
                "embedding_model_path": str(missing_model),
                "retrieval": {
                    **qa.SETTINGS["retrieval"],
                    "mode": "hybrid",
                },
            }
            with (
                mock.patch.object(qa, "SETTINGS", settings),
                mock.patch.object(qa, "_index", None),
                mock.patch.object(qa, "ensure_corpus") as ensure_corpus,
                mock.patch.object(qa, "load_corpus") as load_corpus,
                mock.patch.object(qa, "generate_llm") as generate_llm,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"download_embedding_model\.py.*never falls back silently",
                ):
                    qa.ask(
                        piece_id="die-forelle",
                        question="어떻게 표현해야 하나요?",
                        measure_range=None,
                        generate=False,
                    )

            ensure_corpus.assert_not_called()
            load_corpus.assert_not_called()
            generate_llm.assert_not_called()

    def test_media_dataset_and_rag_dataset_overrides_do_not_collide(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SOPRANO_QA_DATASET_ROOT": "/tmp/score-dataset",
                "SOPRANO_QA_RAG_DATASET_ROOT": "/tmp/rag-dataset",
                "SOPRANO_QA_MODEL_PATH": "/tmp/model.gguf",
                "SOPRANO_QA_EMBEDDING_MODEL_PATH": "/tmp/embedding.gguf",
            },
            clear=False,
        ):
            settings = load_settings(use_legacy_dataset_env=False)

        self.assertEqual(settings["dataset_root"], "/tmp/rag-dataset")
        self.assertEqual(settings["model_path"], "/tmp/model.gguf")
        self.assertEqual(
            settings["embedding_model_path"],
            "/tmp/embedding.gguf",
        )

    def test_relative_overrides_resolve_from_pipeline_repository(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SOPRANO_QA_RAG_DATASET_ROOT": "../alternate-dataset",
                "SOPRANO_QA_MODEL_PATH": "models/alternate.gguf",
            },
            clear=False,
        ):
            settings = load_settings(use_legacy_dataset_env=False)

        pipeline_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            settings["dataset_root"],
            str((pipeline_root.parent / "alternate-dataset").resolve()),
        )
        self.assertEqual(
            settings["model_path"],
            str((pipeline_root / "models" / "alternate.gguf").resolve()),
        )

    def test_index_signature_changes_when_embedding_checkpoint_appears(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "corpus.json"
            stats_path = root / "stats.json"
            embedding_path = root / "embedding.gguf"
            corpus_path.write_text("[]", encoding="utf-8")
            stats_path.write_text("{}", encoding="utf-8")
            with mock.patch.dict(
                qa.SETTINGS,
                {
                    "corpus_path": str(corpus_path),
                    "stats_path": str(stats_path),
                    "embedding_model_path": str(embedding_path),
                },
            ):
                without_checkpoint = qa._current_corpus_signature()
                embedding_path.write_bytes(b"checkpoint")
                with_checkpoint = qa._current_corpus_signature()

        self.assertIsNone(without_checkpoint[2])
        self.assertIsNotNone(with_checkpoint[2])
        self.assertNotEqual(without_checkpoint, with_checkpoint)

    def test_warm_index_still_validates_corpus_on_every_request(self) -> None:
        cached_index = object()
        signature = ((1, 10), (2, 20), None)
        with (
            mock.patch.object(qa, "_index", cached_index),
            mock.patch.object(qa, "_corpus_signature", signature),
            mock.patch.object(qa, "ensure_corpus") as ensure_corpus,
            mock.patch.object(
                qa,
                "_current_corpus_signature",
                return_value=signature,
            ),
            mock.patch.object(qa, "build_retrieval_index") as build_index,
        ):
            self.assertIs(qa._get_index(), cached_index)
            self.assertIs(qa._get_index(), cached_index)

        self.assertEqual(ensure_corpus.call_count, 2)
        ensure_corpus.assert_called_with(qa.SETTINGS, rebuild=False)
        build_index.assert_not_called()

    def test_warm_index_propagates_release_readiness_failure(self) -> None:
        cached_index = object()
        signature = ((1, 10), (2, 20), None)
        with (
            mock.patch.object(qa, "_index", cached_index),
            mock.patch.object(qa, "_corpus_signature", signature),
            mock.patch.object(
                qa,
                "ensure_corpus",
                side_effect=ValueError(
                    "Expert review corpus is not release-ready"
                ),
            ) as ensure_corpus,
            mock.patch.object(
                qa,
                "_current_corpus_signature",
            ) as current_signature,
            mock.patch.object(qa, "build_retrieval_index") as build_index,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Expert review corpus is not release-ready",
            ):
                qa._get_index()

        ensure_corpus.assert_called_once_with(qa.SETTINGS, rebuild=False)
        current_signature.assert_not_called()
        build_index.assert_not_called()

    def test_measure_query_uses_pipeline_scope_and_ids(self) -> None:
        result = qa.ask(
            piece_id="die-forelle",
            question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
            measure_range=(28, 30),
            generate=False,
        )

        self.assertEqual(result["pipeline"], "soprano_qa")
        self.assertEqual(result["generation_mode"], "retrieval_only")
        self.assertEqual(result["answer_basis"], "retrieved_evidence")
        self.assertEqual(
            result["evidence"][0]["id"],
            "die-forelle-ku-010",
        )
        self.assertEqual(
            result["evidence"][0]["scope_match"],
            "overlaps_query_range",
        )
        self.assertEqual(
            result["evidence"][0]["generation_role"],
            "selected_range_support",
        )
        self.assertTrue(
            result["evidence"][0]["selected_range_claim_authority"]
        )
        self.assertTrue(result["has_selected_range_grounding"])
        self.assertIn("긴장감과 분위기 전환", result["answer"])
        self.assert_public_answer_contract(result)

    def test_korean_synonyms_retrieve_the_same_second_beat_evidence(self) -> None:
        questions = (
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 의미하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 표현하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 묘사하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 상징하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 가리키는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 보여주는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 드러내는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무슨 뜻인가?",
            (
                "피아노 반주에서 두 번째 박의 악센트는 무엇을 "
                "의미하는지 알려 주세요."
            ),
            (
                "피아노 반주에서 두 번째 박의 악센트가 어떤 뜻인지 "
                "궁금합니다."
            ),
        )
        for question in questions:
            with self.subTest(question=question):
                result = qa.ask(
                    piece_id="die-forelle",
                    question=question,
                    measure_range=(2, 5),
                    generate=False,
                )
                self.assertEqual(
                    [evidence["id"] for evidence in result["evidence"]],
                    ["die-forelle-ku-002"],
                )
                self.assertEqual(
                    result["evidence"][0]["id"],
                    "die-forelle-ku-002",
                )
                self.assertEqual(
                    result["evidence"][0]["scope_match"],
                    "overlaps_query_range",
                )
                self.assertIn("송어가 뛰어노는 모습", result["answer"])

    def test_dense_retrieval_keeps_intended_explanatory_units(self) -> None:
        cases = (
            (
                "la-capinera",
                "59-62마디의 악센트는 무엇을 의미하는가?",
                (59, 62),
                "la-capinera-ku-023",
            ),
            (
                "una-voce-poco-fa",
                "14-15마디의 겹점 리듬은 무엇을 의미하는가?",
                (14, 15),
                "una-voce-poco-fa-ku-004",
            ),
        )
        for piece_id, question, measure_range, expected_id in cases:
            with self.subTest(question=question):
                result = qa.ask(
                    piece_id=piece_id,
                    question=question,
                    measure_range=measure_range,
                    generate=False,
                )
                evidence_ids = [
                    evidence["id"] for evidence in result["evidence"]
                ]
                self.assertIn(expected_id, evidence_ids)
                target = next(
                    evidence
                    for evidence in result["evidence"]
                    if evidence["id"] == expected_id
                )
                self.assertEqual(
                    target["scope_match"],
                    "overlaps_query_range",
                )
                self.assertEqual(target["semantic_match_type"], "dense")
                self.assertGreater(target["dense_content_score"], 0.0)

    def test_terminal_answer_predicates_preserve_definition_queries(self) -> None:
        cases = (
            (
                "nella-fantasia",
                "D.S. al Coda는 무엇을 의미하는가?",
                "nella-fantasia-ku-012",
            ),
            (
                "la-capinera",
                "vuò(vuo’)는 무엇을 의미하는가?",
                "la-capinera-ku-017",
            ),
            (
                "una-voce-poco-fa",
                "꾸밈음의 사선 표시는 무엇을 의미하는가?",
                "una-voce-poco-fa-ku-008",
            ),
        )
        for piece_id, question, expected_id in cases:
            with self.subTest(question=question):
                result = qa.ask(
                    piece_id=piece_id,
                    question=question,
                    measure_range=None,
                    generate=False,
                )
                self.assertEqual(result["evidence"][0]["id"], expected_id)

    def test_natural_piano_part_paraphrase_retrieves_annotated_ranges(
        self,
    ) -> None:
        for measure_range in ((2, 27), (41, 54)):
            with self.subTest(measure_range=measure_range):
                result = qa.ask(
                    piece_id="die-forelle",
                    question=(
                        "피아노 파트에서 두 번째 박의 악센트는 "
                        "무엇을 표현할까?"
                    ),
                    measure_range=measure_range,
                    generate=False,
                )

                self.assertEqual(
                    result["evidence"][0]["id"],
                    "die-forelle-ku-002",
                )
                self.assertEqual(
                    result["evidence"][0]["scope_match"],
                    "overlaps_query_range",
                )
                self.assertIn("송어가 뛰어노는 모습", result["answer"])

    def test_broad_singer_question_uses_confirmed_local_examples(self) -> None:
        result = qa.ask(
            piece_id="la-capinera",
            question=(
                "이 노래를 부를 때 가창자의 입장에서 유의해야 할 "
                "점은 무엇인가?"
            ),
            measure_range=None,
            generate=False,
            top_k=10,
        )

        local = next(
            item
            for item in result["evidence"]
            if item["scope_match"] == "local_example"
        )
        self.assertEqual(local["scope_match"], "local_example")
        self.assertEqual(local["generation_role"], "local_example")
        self.assertTrue(local["in_requested_scope"])
        self.assertIsNone(local["selected_range_claim_authority"])
        self.assertEqual(local["measure_status"], "specific")
        self.assertTrue(local["measure_ranges"])
        self.assertTrue(result["has_confirmed_local_examples"])
        self.assertIn(local["id"], [item["id"] for item in result["evidence"]])
        self.assertNotIn("범위 안내", result["answer"])
        self.assertNotIn("확인된 국소 예시", result["answer"])
        self.assert_public_answer_contract(result)

    def test_retrieval_only_answer_is_bounded_but_evidence_is_complete(
        self,
    ) -> None:
        results = []
        for index in range(6):
            record = {
                "id": f"die-forelle-ku-test-{index}",
                "evidence_type": "expert_annotation",
                "piece": "die-forelle",
                "work": "Die Forelle",
                "topic": "performance",
                "question": "",
                "answer": f"검색 답변 {index}입니다.",
                "measure_range": [],
                "measure_scope": "whole_piece",
                "measure_status": "whole_piece",
                "source_ids": [f"source-{index}"],
                "web_source_ids": [],
                "claim_ids": [],
                "sources": [],
                "annotators": ["tester"],
                "source_answer_context": [],
            }
            results.append(
                SearchResult(
                    record=record,
                    score=1.0 - index * 0.01,
                    text_score=1.0,
                    measure_score=1.5,
                    piece_score=1.0,
                    scope_match="general_evidence",
                )
            )

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return results

        with mock.patch.object(qa, "_get_index", return_value=FixedIndex()):
            response = qa.ask(
                piece_id="die-forelle",
                question="어떻게 표현할까?",
                measure_range=None,
                generate=False,
                top_k=6,
            )

        self.assertEqual(len(response["evidence"]), 6)
        self.assertEqual(
            [item["id"] for item in response["evidence"]],
            [result.record["id"] for result in results],
        )
        for result in results[:4]:
            self.assertIn(result.record["answer"], response["answer"])
        self.assertNotIn(results[4].record["answer"], response["answer"])
        self.assertNotIn(results[5].record["answer"], response["answer"])
        self.assert_public_answer_contract(response)

    def test_broad_guidance_uses_the_same_grounded_llm_path(
        self,
    ) -> None:
        with mock.patch.object(
            qa,
            "generate_llm",
            return_value=(
                "가창자의 입장에서는 호흡과 표현을 함께 고려하는 "
                "것이 좋습니다. [E1]"
            ),
        ) as generate:
            result = qa.ask(
                piece_id="la-capinera",
                question=(
                    "이 노래를 부를 때 가창자의 입장에서 유의해야 "
                    "할 점은 무엇인가?"
                ),
                measure_range=None,
                generate=True,
                top_k=6,
            )

        generate.assert_called_once()
        self.generation_preflight.assert_called_once()
        self.assertEqual(result["generation_mode"], "llm")
        self.assertEqual(result["answer_basis"], "retrieved_evidence")
        self.assertIsNone(result["unavailable_reason"])
        self.assertTrue(result["evidence"])
        self.assertNotEqual(result["answer"], NO_CORPUS_EVIDENCE_MESSAGE)
        self.assertNotIn("확인된 국소 예시", result["answer"])
        self.assert_public_answer_contract(result)

    def test_ranged_query_retains_confirmed_other_range_context(self) -> None:
        result = qa.ask(
            piece_id="die-forelle",
            question=(
                "피아노 파트에서 두 번째 박의 악센트는 무엇을 표현할까?"
            ),
            measure_range=(28, 40),
            generate=False,
        )

        secondary = next(
            item
            for item in result["evidence"]
            if item["id"] == "die-forelle-ku-002"
        )
        self.assertEqual(
            secondary["scope_match"],
            "other_range_context",
        )
        self.assertEqual(
            secondary["generation_role"],
            "other_range_context_only",
        )
        self.assertFalse(secondary["in_requested_scope"])
        self.assertFalse(secondary["selected_range_claim_authority"])
        self.assertFalse(result["has_selected_range_grounding"])

    def test_secondary_only_context_is_not_generated_as_range_answer(
        self,
    ) -> None:
        record = {
            "id": "test-piece-ku-001",
            "evidence_type": "expert_annotation",
            "piece": "die-forelle",
            "work": "Die Forelle",
            "topic": "performance",
            "answer": "다른 구간에서만 적용되는 주석입니다.",
            "measure_range": [[30, 32]],
            "measure_scope": "local",
            "measure_status": "specific",
            "source_ids": ["kim-die-forelle-01"],
            "annotators": ["kim"],
            "source_answer_context": [],
        }
        secondary = SearchResult(
            record=record,
            score=1.0,
            text_score=2.0,
            measure_score=-1.0,
            piece_score=1.0,
            scope_match="other_range_context",
            alias_score=0.0,
        )

        class SecondaryOnlyIndex:
            @staticmethod
            def search(**_kwargs):
                return [secondary]

        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=SecondaryOnlyIndex(),
            ),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="악센트는 어떻게 표현할까?",
                measure_range=(10, 12),
                generate=True,
            )

        generate.assert_not_called()
        self.assertFalse(result["has_primary_grounding"])
        self.assertFalse(result["has_selected_range_grounding"])
        self.assertEqual(
            result["answer_basis"],
            "no_corpus_evidence",
        )
        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["unavailable_reason"], "no_corpus_evidence")
        self.assertEqual(result["answer"], NO_CORPUS_EVIDENCE_MESSAGE)
        self.assertEqual(
            result["evidence"][0]["generation_role"],
            "other_range_context_only",
        )
        self.generation_preflight.assert_called_once()
        self.assert_public_answer_contract(result)

    def test_finalized_expert_is_composed_by_the_grounded_llm(
        self,
    ) -> None:
        with mock.patch.object(
            qa,
            "generate_llm",
            return_value=(
                "긴장감과 분위기 전환을 자연스럽게 표현하는 것이 "
                "좋습니다. [E1]"
            ),
        ) as generate:
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        generate.assert_called_once()
        self.generation_preflight.assert_called_once_with(
            qa.SETTINGS["model_path"],
            qa.SETTINGS["llm"],
        )
        self.assertEqual(result["generation_mode"], "llm")
        self.assertEqual(result["answer_basis"], "retrieved_evidence")
        self.assertIsNone(result["unavailable_reason"])
        self.assertIn("긴장감과 분위기 전환", result["answer"])
        self.assertEqual(result["evidence"][0]["id"], "die-forelle-ku-010")
        self.assertNotIn("die-forelle-ku-010", result["answer"])
        self.assert_public_answer_contract(result)

    def test_ambiguous_dense_range_returns_explicit_unavailable_reason(
        self,
    ) -> None:
        class AmbiguousIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid_no_dense_match"
            last_route_reason = AMBIGUOUS_DENSE_ONLY_RANGE_REASON
            last_ambiguous_candidates = [
                {
                    "id": "nella-fantasia-ku-008",
                    "source_ids": ["source-b"],
                    "scope_match": "overlaps_query_range",
                    "dense_score": 0.563439,
                    "dense_content_score": 0.563237,
                },
                {
                    "id": "nella-fantasia-ku-006",
                    "source_ids": ["source-a"],
                    "scope_match": "overlaps_query_range",
                    "dense_score": 0.558445,
                    "dense_content_score": 0.537356,
                },
            ]

            @staticmethod
            def search(**_kwargs):
                return []

        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=AmbiguousIndex(),
            ),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            result = qa.ask(
                piece_id="nella-fantasia",
                question=(
                    "갑작스러운 상행 도약을 자연스럽게 소화하려면 "
                    "어떻게 해야 할까?"
                ),
                measure_range=(11, 17),
                generate=True,
            )

        generate.assert_not_called()
        self.generation_preflight.assert_called_once()
        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["answer_basis"], "no_corpus_evidence")
        self.assertEqual(
            result["unavailable_reason"],
            "ambiguous_dense_grounding",
        )
        self.assertEqual(result["evidence"], [])
        self.assertEqual(
            result["retrieval"]["route_reason"],
            AMBIGUOUS_DENSE_ONLY_RANGE_REASON,
        )
        self.assertEqual(
            [item["id"] for item in result["retrieval"]["ambiguous_candidates"]],
            ["nella-fantasia-ku-008", "nella-fantasia-ku-006"],
        )
        self.assertEqual(result["answer"], NO_CORPUS_EVIDENCE_MESSAGE)
        self.assert_public_answer_contract(result)

    def test_missing_scoped_causal_authority_returns_explicit_reason(
        self,
    ) -> None:
        class MissingCausalAuthorityIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid_no_dense_match"
            last_route_reason = SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON
            last_ambiguous_candidates = []

            def __init__(self) -> None:
                self.search_calls = []

            def search(self, **kwargs):
                self.search_calls.append(kwargs)
                return []

        index = MissingCausalAuthorityIndex()
        with (
            mock.patch.object(qa, "_get_index", return_value=index),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            result = qa.ask(
                piece_id="una-voce-poco-fa",
                question=(
                    "63마디에서 로시니가 겹부점 리듬을 사용한 "
                    "이유는 무엇일까?"
                ),
                measure_range=(14, 63),
                generate=True,
            )

        generate.assert_not_called()
        self.generation_preflight.assert_called_once()
        self.assertEqual(index.search_calls[0]["measure_ranges"], [[63, 63]])
        self.assertEqual(result["measure_range"], [14, 63])
        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["answer_basis"], "no_corpus_evidence")
        self.assertEqual(
            result["unavailable_reason"],
            "missing_scoped_causal_authority",
        )
        self.assertEqual(result["evidence"], [])
        self.assertEqual(
            result["retrieval"]["route_reason"],
            SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
        )
        self.assert_public_answer_contract(result)

    def test_underspecified_range_returns_explicit_unavailable_reason(
        self,
    ) -> None:
        class EmptyIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "no_lexical_match"
            last_ambiguous_candidates = []

            @staticmethod
            def search(**_kwargs):
                return []

        with (
            mock.patch.object(qa, "_get_index", return_value=EmptyIndex()),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            result = qa.ask(
                piece_id="nella-fantasia",
                question="이 부분은 어떻게 노래해야 할까?",
                measure_range=(11, 17),
                generate=True,
            )

        generate.assert_not_called()
        self.generation_preflight.assert_called_once()
        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["answer_basis"], "no_corpus_evidence")
        self.assertEqual(
            result["unavailable_reason"],
            "underspecified_ranged_question",
        )
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["answer"], NO_CORPUS_EVIDENCE_MESSAGE)
        self.assert_public_answer_contract(result)

    @staticmethod
    def _grounded_web_result() -> SearchResult:
        return SearchResult(
            record={
                "id": "webchunk-grounded-test",
                "evidence_type": "web_database",
                "piece": "die-forelle",
                "work": "Die Forelle",
                "topic": "historical_editions",
                "answer": "이 작품의 역사적 판본을 설명하는 검토 자료입니다.",
                "measure_range": [],
                "measure_scope": "global",
                "measure_status": "whole_piece",
                "source_ids": [],
                "web_source_ids": ["web-source"],
                "claim_ids": ["web-claim"],
                "sources": [],
            },
            score=0.75,
            text_score=2.0,
            measure_score=1.5,
            piece_score=1.0,
            scope_match="general_evidence",
            alias_score=0.0,
            concept_coverage=1.0,
            content_concept_coverage=1.0,
            semantic_match_type="strict",
            dense_score=0.55,
            dense_content_score=0.50,
            retrieval_mode="hybrid",
        )

    def test_generation_backend_error_propagates_after_one_call(self) -> None:
        class WebIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "dense_result_returned"

            @staticmethod
            def search(**_kwargs):
                return [ServicePipelineTests._grounded_web_result()]

        failure = RuntimeError("backend stopped")
        with (
            mock.patch.object(qa, "_get_index", return_value=WebIndex()),
            mock.patch.object(
                qa,
                "generate_llm",
                side_effect=failure,
            ) as generate,
        ):
            with self.assertRaises(RuntimeError) as raised:
                qa.ask(
                    piece_id="die-forelle",
                    question="이 작품의 역사적 판본은 무엇인가요?",
                    measure_range=None,
                    generate=True,
                )

        self.assertIs(raised.exception, failure)
        generate.assert_called_once()
        self.generation_preflight.assert_called_once()

    def test_rejected_generated_draft_propagates_after_one_call(self) -> None:
        class WebIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "dense_result_returned"

            @staticmethod
            def search(**_kwargs):
                return [ServicePipelineTests._grounded_web_result()]

        with (
            mock.patch.object(qa, "_get_index", return_value=WebIndex()),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value="<NO_GROUNDED_ANSWER>",
            ) as generate,
        ):
            with self.assertRaises(GeneratedAnswerRejected):
                qa.ask(
                    piece_id="die-forelle",
                    question="이 작품의 역사적 판본은 무엇인가요?",
                    measure_range=None,
                    generate=True,
                )

        generate.assert_called_once()
        self.generation_preflight.assert_called_once()

    def test_natural_insufficiency_draft_propagates_after_one_call(self) -> None:
        class WebIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "dense_result_returned"

            @staticmethod
            def search(**_kwargs):
                return [ServicePipelineTests._grounded_web_result()]

        refusal = (
            "제공된 근거만으로는 이 질문에 정확히 답변할 수 없습니다."
        )
        with (
            mock.patch.object(qa, "_get_index", return_value=WebIndex()),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=refusal,
            ) as generate,
        ):
            with self.assertRaises(GeneratedAnswerRejected):
                qa.ask(
                    piece_id="die-forelle",
                    question="이 작품의 역사적 판본은 무엇인가요?",
                    measure_range=None,
                    generate=True,
                )

        generate.assert_called_once()
        self.generation_preflight.assert_called_once()

    def test_empty_generated_draft_propagates_after_one_call(self) -> None:
        class WebIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "dense_result_returned"

            @staticmethod
            def search(**_kwargs):
                return [ServicePipelineTests._grounded_web_result()]

        with (
            mock.patch.object(qa, "_get_index", return_value=WebIndex()),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=" \n\t",
            ) as generate,
        ):
            with self.assertRaises(GeneratedAnswerRejected):
                qa.ask(
                    piece_id="die-forelle",
                    question="이 작품의 역사적 판본은 무엇인가요?",
                    measure_range=None,
                    generate=True,
                )

        generate.assert_called_once()
        self.generation_preflight.assert_called_once()

    def test_other_rejected_draft_remains_a_review_warning(self) -> None:
        class WebIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "dense_result_returned"

            @staticmethod
            def search(**_kwargs):
                return [ServicePipelineTests._grounded_web_result()]

        raw_answer = "판본에 따라 연주 방식이 달라집니다. [E1]"
        warning = "Generated answer overgeneralizes local evidence"
        with (
            mock.patch.object(qa, "_get_index", return_value=WebIndex()),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=raw_answer,
            ) as generate,
            mock.patch.object(
                qa,
                "finalize_grounded_generated_answer",
                side_effect=GeneratedAnswerRejected(warning),
            ),
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="이 작품의 역사적 판본은 무엇인가요?",
                measure_range=None,
                generate=True,
            )

        generate.assert_called_once()
        self.generation_preflight.assert_called_once()
        self.assertEqual(result["generation_mode"], "llm")
        self.assertEqual(result["answer_basis"], "retrieved_evidence")
        self.assertEqual(result["generation_validation_warning"], warning)
        self.assertEqual(result["answer"], "판본에 따라 연주 방식이 달라집니다.")

    def test_web_results_keep_deterministic_rights_notices(self) -> None:
        result = qa.ask(
            piece_id="die-forelle",
            question="이 작품의 정식 표제와 작품 번호는 무엇인가요?",
            measure_range=None,
            generate=False,
        )

        self.assertTrue(result["evidence"])
        self.assertTrue(all(item["kind"] == "web" for item in result["evidence"]))
        self.assertEqual(
            {item["record_id"] for item in result["evidence_notices"]},
            {item["id"] for item in result["evidence"]},
        )

    def test_corpus_stats_come_from_pipeline_snapshot(self) -> None:
        stats = qa.corpus_stats()
        self.assertEqual(stats["pipeline"], "soprano_qa")
        self.assertEqual(stats["corpus_schema_version"], 8)
        self.assertEqual(stats["total_records"], 223)
        self.assertEqual(
            stats["records_by_evidence_type"],
            {"expert_annotation": 122, "web_database": 101},
        )
        for deprecated in (
            "expert_records_by_rewrite_status",
            "expert_records_by_retrieval_exclusion_reason",
            "expert_records_by_retrieval_review_warning",
            "expert_retrieval_review_warnings",
        ):
            self.assertNotIn(deprecated, stats)


if __name__ == "__main__":
    unittest.main()
