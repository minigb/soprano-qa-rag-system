"""Integration checks for the reusable QA service facade."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from soprano_qa import service as qa
from soprano_qa.retrieval import SearchResult, format_measure_range
from soprano_qa.settings import load_settings


def _first_local_prompt_evidence(prompt: str) -> dict[str, str]:
    local_block = next(
        block
        for block in prompt.split("\n\n---\n\n")
        if "evidence_role: local_example" in block
    )
    return {
        line.split(": ", 1)[0]: line.split(": ", 1)[1]
        for line in local_block.splitlines()
        if ": " in line
    }


class ServicePipelineTests(unittest.TestCase):
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

    def test_measure_query_uses_pipeline_scope_and_ids(self) -> None:
        result = qa.ask(
            piece_id="die-forelle",
            question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
            measure_range=(28, 30),
            generate=False,
        )

        self.assertEqual(result["pipeline"], "soprano_qa")
        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
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
        self.assertIn("[die-forelle-ku-010]", result["answer"])

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

    def test_korean_relation_fallback_reranks_explanatory_units(self) -> None:
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
                self.assertEqual(
                    [evidence["id"] for evidence in result["evidence"]],
                    [expected_id],
                )
                self.assertEqual(
                    result["evidence"][0]["semantic_match_type"],
                    "answer_relation_fallback",
                )
                self.assertEqual(
                    result["evidence"][0]["answer_relation_score"],
                    1.0,
                )

    def test_relation_fallback_does_not_promote_alias_only_global_context(
        self,
    ) -> None:
        result = qa.ask(
            piece_id="die-forelle",
            question=(
                "피아노 반주에서 두 번째 박의 악센트는 무엇을 "
                "의미하는가?"
            ),
            measure_range=(28, 40),
            generate=False,
        )

        self.assertEqual(
            [evidence["id"] for evidence in result["evidence"]],
            ["die-forelle-ku-002"],
        )
        self.assertEqual(
            result["evidence"][0]["scope_match"],
            "other_range_context",
        )
        self.assertEqual(result["answer_basis"], "retrieved_secondary_context")
        self.assertFalse(result["has_primary_grounding"])

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
            if (
                item["scope_match"] == "local_example"
                and f'[{item["id"]}]' in result["answer"]
            )
        )
        self.assertEqual(local["scope_match"], "local_example")
        self.assertEqual(local["generation_role"], "local_example")
        self.assertTrue(local["in_requested_scope"])
        self.assertIsNone(local["selected_range_claim_authority"])
        self.assertEqual(local["measure_status"], "specific")
        self.assertTrue(local["measure_ranges"])
        canonical_range = format_measure_range(local["measure_ranges"])
        self.assertTrue(result["has_confirmed_local_examples"])
        self.assertIn(f'[{local["id"]}]', result["answer"])
        self.assertIn(
            f"확인된 국소 예시(마디 {canonical_range})",
            result["answer"],
        )

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
                "answer": f"검색 답변 {index}",
                "measure_range": [],
                "measure_scope": "whole_piece",
                "measure_status": "whole_piece",
                "measure_notes": "",
                "rewrite_status": "ready",
                "rewrite_notes": "",
                "retrieval_review_warning": "",
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

        unavailable = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": False,
            "llama_cpp_available": False,
        }
        with (
            mock.patch.object(qa, "_get_index", return_value=FixedIndex()),
            mock.patch.object(qa, "model_status", return_value=unavailable),
        ):
            response = qa.ask(
                piece_id="die-forelle",
                question="어떻게 표현할까?",
                measure_range=None,
                generate=False,
                top_k=6,
            )

        self.assertEqual(len(response["evidence"]), 6)
        cited_ids = [
            item["id"]
            for item in response["evidence"]
            if f'[{item["id"]}]' in response["answer"]
        ]
        self.assertEqual(cited_ids, [result.record["id"] for result in results[:4]])
        self.assertNotIn(results[4].record["answer"], response["answer"])

    def test_broad_rag_generation_can_cite_a_confirmed_local_example(
        self,
    ) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        selected_local = {}

        def grounded_local_example(
            _model_path: str,
            messages: list[dict[str, str]],
            _llm_settings: dict,
        ) -> str:
            prompt = messages[1]["content"]
            self.assertIn("evidence_role: local_example", prompt)
            self.assertIn(
                "canonical measure_range",
                prompt,
            )
            fields = _first_local_prompt_evidence(prompt)
            selected_local.update(
                {
                    "id": fields["id"],
                    "label": fields["citation_label"],
                    "measure_range": fields["measure_range"],
                }
            )
            return (
                f'{selected_local["measure_range"]}마디에서는 확인된 '
                "가창 조언에 유의한다. "
                f'[{selected_local["label"]}]'
            )

        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                side_effect=grounded_local_example,
            ),
        ):
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

        self.assertEqual(result["generation_mode"], "llm")
        self.assertEqual(result["answer_basis"], "retrieved_evidence")
        self.assertIn(selected_local["measure_range"], result["answer"])
        self.assertIn(f'[{selected_local["id"]}]', result["answer"])

    def test_broad_rag_rejects_locally_cited_piece_wide_claim(
        self,
    ) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        calls = []
        selected_local = {}

        def overgeneralized_answer(
            _model_path: str,
            messages: list[dict[str, str]],
            _llm_settings: dict,
        ) -> str:
            calls.append(messages)
            prompt = messages[1]["content"]
            fields = _first_local_prompt_evidence(prompt)
            selected_local.update(
                {
                    "id": fields["id"],
                    "label": fields["citation_label"],
                    "measure_range": fields["measure_range"],
                }
            )
            return (
                "이 곡 전반에는 도약음이 많이 나오므로 항상 고음을 "
                f'약하게 불러야 한다. [{selected_local["label"]}]'
            )

        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                side_effect=overgeneralized_answer,
            ),
        ):
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

        self.assertEqual(len(calls), 2)
        self.assertIn(
            "unsupported whole-piece or frequency claim",
            calls[1][-1]["content"],
        )
        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
        self.assertEqual(
            result["generation_fallback_reason"],
            "unsupported local-example generalization rejected",
        )
        self.assertNotIn("항상 고음을 약하게", result["answer"])
        self.assertNotIn(f'[{selected_local["id"]}]', result["answer"])
        cited = [
            item
            for item in result["evidence"]
            if f'[{item["id"]}]' in result["answer"]
        ]
        self.assertTrue(cited)
        self.assertTrue(
            all(item["scope_match"] != "local_example" for item in cited)
        )

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
            "answer": "다른 구간에서만 적용되는 주석",
            "measure_range": [[30, 32]],
            "measure_scope": "local",
            "measure_status": "specific",
            "measure_notes": "",
            "rewrite_status": "ready",
            "rewrite_notes": "",
            "retrieval_review_warning": "",
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

        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=SecondaryOnlyIndex(),
            ),
            mock.patch.object(qa, "model_status", return_value=available),
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
            "retrieved_secondary_context",
        )
        self.assertIn(
            "선택한 마디 범위를 직접 뒷받침하는 근거는 없습니다.",
            result["answer"],
        )
        self.assertEqual(
            result["evidence"][0]["generation_role"],
            "other_range_context_only",
        )

    def test_generation_runs_context_and_citation_pipeline(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value="검색 근거를 사용한 답변입니다. [E1]",
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        generate.assert_called_once()
        self.assertEqual(result["generation_mode"], "llm")
        self.assertEqual(result["answer_basis"], "retrieved_evidence")
        self.assertIn("[die-forelle-ku-010]", result["answer"])
        self.assertNotIn("[E1]", result["answer"])

    def test_secondary_citation_fallback_is_reported_as_extractive(
        self,
    ) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value="다른 구간의 주장을 잘못 사용했다. [E2]",
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question=(
                    "피아노 파트에서 두 번째 박의 악센트는 "
                    "무엇을 표현할까?"
                ),
                measure_range=(28, 40),
                generate=True,
            )

        generate.assert_called_once()
        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
        self.assertEqual(
            result["generation_fallback_reason"],
            "secondary evidence citation rejected",
        )
        self.assertIn("[die-forelle-ku-003]", result["answer"])
        self.assertNotIn("[die-forelle-ku-002]", result["answer"])
        self.assertNotIn("잘못 사용했다", result["answer"])

    def test_no_hit_uses_separate_internal_knowledge_prompt(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=(
                    "일반 음악 지식으로 답한 내용입니다. "
                    "[E1; sqa-0001] [sqa-0001, sqa-0002] "
                    "[die-forelle-ku-001] "
                    "[출처: E2] 【Evidence 3】 "
                    "[webchunk-a, webchunk-b] "
                    "현재 답변 가능 자료에서 이 질문을 뒷받침할 근거를 찾지 못했습니다."
                ),
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="2022년 FIFA 월드컵 우승팀은 어디인가요?",
                measure_range=None,
                generate=True,
            )

        generate.assert_called_once()
        messages = generate.call_args.args[1]
        self.assertIn("No matching corpus evidence", messages[0]["content"])
        self.assertNotIn("Retrieved evidence", messages[1]["content"])
        self.assertEqual(result["generation_mode"], "llm")
        self.assertEqual(result["answer_basis"], "internal_knowledge")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["evidence_notices"], [])
        self.assertIn("일반 음악 지식", result["answer"])
        self.assertNotIn("[E1]", result["answer"])
        self.assertNotIn("sqa-", result["answer"])
        self.assertNotIn("-ku-", result["answer"])
        self.assertNotIn("webchunk-", result["answer"])
        self.assertNotIn("Evidence", result["answer"])
        self.assertNotIn(
            "현재 답변 가능 자료에서 이 질문을 뒷받침할 근거를 찾지 못했습니다.",
            result["answer"],
        )

    def test_grounded_refusal_preserves_retrieved_extractive_evidence(
        self,
    ) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=(
                    "현재 답변 가능 자료에서 이 질문을 뒷받침할 "
                    "근거를 찾지 못했습니다."
                ),
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        generate.assert_called_once()
        self.assertIn(
            "Retrieved evidence",
            generate.call_args.args[1][1]["content"],
        )
        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
        self.assertTrue(result["evidence"])
        self.assertIn("[die-forelle-ku-010]", result["answer"])
        self.assertNotIn("현재 답변 가능 자료에서", result["answer"])
        self.assertEqual(
            result["generation_fallback_reason"],
            "grounded model returned no usable answer",
        )

    def test_empty_grounded_answer_preserves_retrieved_evidence(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(qa, "generate_llm", return_value="") as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        generate.assert_called_once()
        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
        self.assertTrue(result["evidence"])
        self.assertIn("[die-forelle-ku-010]", result["answer"])
        self.assertEqual(
            result["generation_fallback_reason"],
            "grounded model returned no usable answer",
        )

    def test_footer_only_grounded_answer_uses_extractive_evidence(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=(
                    "\"제공된 검색 근거\":\n"
                    "[Evidence no. 1] [근거 2]"
                ),
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        generate.assert_called_once()
        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
        self.assertIn("[die-forelle-ku-010]", result["answer"])
        self.assertNotIn("제공된 검색 근거", result["answer"])
        self.assertEqual(
            result["generation_fallback_reason"],
            "grounded model returned no usable answer",
        )

    def test_no_hit_can_disable_internal_knowledge_fallback(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="2022년 FIFA 월드컵 우승팀은 어디인가요?",
                measure_range=None,
                generate=True,
                allow_internal_knowledge=False,
            )

        generate.assert_not_called()
        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["answer_basis"], "no_corpus_evidence")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["answer"], "검색된 코퍼스 근거가 없습니다.")
        self.assertEqual(
            result["generation_fallback_reason"],
            "internal knowledge fallback disabled",
        )

    def test_grounded_secondary_limitation_does_not_trigger_internal_retry(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=(
                    "제공된 근거로는 판본 정보가 부족하지만, 28마디의 "
                    "분위기 변화는 어둡게 표현합니다. [E1]"
                ),
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        generate.assert_called_once()
        self.assertEqual(result["answer_basis"], "retrieved_evidence")
        self.assertTrue(result["evidence"])
        self.assertIn("[die-forelle-ku-010]", result["answer"])

    def test_empty_internal_answer_is_classified_as_unavailable(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=(
                    "현재 답변 가능 자료에서 이 질문을 뒷받침할 근거를 찾지 못했습니다."
                ),
            ),
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="2022년 FIFA 월드컵 우승팀은 어디인가요?",
                measure_range=None,
                generate=True,
            )

        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["answer_basis"], "generation_unavailable")
        self.assertIn("로컬 LLM을 사용할 수 없어", result["answer"])

    def test_no_hit_reports_model_unavailable_without_old_abstention(self) -> None:
        unavailable = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": False,
        }
        with mock.patch.object(qa, "model_status", return_value=unavailable):
            result = qa.ask(
                piece_id="die-forelle",
                question="2022년 FIFA 월드컵 우승팀은 어디인가요?",
                measure_range=None,
                generate=True,
            )

        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["answer_basis"], "generation_unavailable")
        self.assertIn("로컬 LLM을 사용할 수 없어", result["answer"])
        self.assertNotIn("현재 답변 가능 자료에서", result["answer"])
        self.assertEqual(result["evidence"], [])

    def test_no_hit_generation_failure_reports_unavailable(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                side_effect=RuntimeError("test generation failure"),
            ),
            mock.patch.object(qa.sys, "stderr"),
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="2022년 FIFA 월드컵 우승팀은 어디인가요?",
                measure_range=None,
                generate=True,
            )

        self.assertEqual(result["generation_mode"], "unavailable")
        self.assertEqual(result["answer_basis"], "generation_unavailable")
        self.assertEqual(result["generation_fallback_reason"], "local generation failed")
        self.assertNotIn("현재 답변 가능 자료에서", result["answer"])

    def test_exhausted_context_retry_keeps_extractive_evidence(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(
                qa,
                "generate_llm",
                side_effect=ValueError(
                    "requested tokens exceed the context window"
                ),
            ),
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
                top_k=2,
            )

        self.assertTrue(result["context_limited"])
        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
        self.assertTrue(result["evidence"])
        self.assertIn(f"[{result['evidence'][0]['id']}]", result["answer"])

    def test_generation_falls_back_to_cited_pipeline_results(self) -> None:
        unavailable = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": False,
        }
        with mock.patch.object(
            qa,
            "model_status",
            return_value=unavailable,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        self.assertEqual(result["generation_mode"], "extractive")
        self.assertEqual(result["answer_basis"], "retrieval_extractive")
        self.assertEqual(
            result["generation_fallback_reason"],
            "llama-cpp-python is unavailable",
        )
        self.assertIn("[die-forelle-ku-010]", result["answer"])

    def test_concurrent_generation_is_serialized(self) -> None:
        available = {
            "path": "/tmp/model.gguf",
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        }
        counter_lock = threading.Lock()
        active = 0
        peak = 0

        def fake_generate(*args, **kwargs) -> str:
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with counter_lock:
                active -= 1
            return "답변 [E1]"

        def ask_once() -> dict:
            return qa.ask(
                piece_id="die-forelle",
                question="28마디부터 분위기 변화를 어떻게 표현해야 하나요?",
                measure_range=(28, 30),
                generate=True,
            )

        with (
            mock.patch.object(qa, "model_status", return_value=available),
            mock.patch.object(qa, "generate_llm", side_effect=fake_generate),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(lambda _: ask_once(), range(2)))

        self.assertEqual(peak, 1)
        self.assertTrue(
            all(result["generation_mode"] == "llm" for result in results)
        )

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
        self.assertEqual(stats["total_records"], 223)
        self.assertEqual(stats["retrievable_records"], 223)


if __name__ == "__main__":
    unittest.main()
