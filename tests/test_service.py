"""Integration checks for the reusable QA service facade."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
import time
import unittest
from unittest import mock

from soprano_qa import service as qa
from soprano_qa.settings import load_settings


class ServicePipelineTests(unittest.TestCase):
    def test_media_dataset_and_rag_dataset_overrides_do_not_collide(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SOPRANO_QA_DATASET_ROOT": "/tmp/score-dataset",
                "SOPRANO_QA_RAG_DATASET_ROOT": "/tmp/rag-dataset",
                "SOPRANO_QA_MODEL_PATH": "/tmp/model.gguf",
            },
            clear=False,
        ):
            settings = load_settings(use_legacy_dataset_env=False)

        self.assertEqual(settings["dataset_root"], "/tmp/rag-dataset")
        self.assertEqual(settings["model_path"], "/tmp/model.gguf")

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
        self.assertEqual(result["evidence"][0]["id"], "sqa-0058")
        self.assertEqual(
            result["evidence"][0]["scope_match"],
            "overlaps_query_range",
        )
        self.assertIn("[sqa-0058]", result["answer"])

    def test_korean_paraphrase_retrieves_second_beat_accent_answer(self) -> None:
        result = qa.ask(
            piece_id="die-forelle",
            question="피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?",
            measure_range=(2, 5),
            generate=False,
        )

        self.assertEqual(result["evidence"][0]["id"], "sqa-0057")
        self.assertIn("송어가 뛰어노는 모습", result["answer"])

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
        self.assertIn("[sqa-0058]", result["answer"])
        self.assertNotIn("[E1]", result["answer"])

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
        self.assertNotIn("webchunk-", result["answer"])
        self.assertNotIn("Evidence", result["answer"])
        self.assertNotIn(
            "현재 답변 가능 자료에서 이 질문을 뒷받침할 근거를 찾지 못했습니다.",
            result["answer"],
        )

    def test_grounded_refusal_retries_with_internal_knowledge(self) -> None:
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
                side_effect=[
                    "현재 답변 가능 자료에서 이 질문을 뒷받침할 근거를 찾지 못했습니다.",
                    "일반 지식으로는 빠른 반주가 생동감과 추진력을 줄 수 있습니다.",
                ],
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="슈베르트는 왜 반주를 빠르게 연주했나요?",
                measure_range=None,
                generate=True,
            )

        self.assertEqual(generate.call_count, 2)
        self.assertIn("Retrieved evidence", generate.call_args_list[0].args[1][1]["content"])
        self.assertNotIn("Retrieved evidence", generate.call_args_list[1].args[1][1]["content"])
        self.assertEqual(result["answer_basis"], "internal_knowledge")
        self.assertEqual(result["evidence"], [])
        self.assertIn("생동감과 추진력", result["answer"])
        self.assertNotIn("현재 답변 가능 자료에서", result["answer"])

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
        self.assertIn("[sqa-0058]", result["answer"])

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
        self.assertIn("[sqa-0058]", result["answer"])

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
        self.assertEqual(stats["total_records"], 211)
        self.assertEqual(stats["retrievable_records"], 210)


if __name__ == "__main__":
    unittest.main()
