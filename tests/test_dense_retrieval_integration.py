"""Optional regression checks against the configured local embedding model."""

from __future__ import annotations

import importlib.util
import os
import unittest

from soprano_qa.dense import build_retrieval_index, retrieval_diagnostics
from soprano_qa.retrieval import load_corpus
from soprano_qa.settings import load_settings


class LocalDenseModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings()
        if importlib.util.find_spec("llama_cpp") is None:
            raise unittest.SkipTest("llama-cpp-python is not installed")
        if not os.path.isfile(cls.settings["embedding_model_path"]):
            raise unittest.SkipTest("local embedding checkpoint is unavailable")
        cls.index = build_retrieval_index(
            load_corpus(cls.settings["corpus_path"]),
            cls.settings,
        )
        diagnostics = retrieval_diagnostics(cls.index)
        if diagnostics["active_mode"] != "hybrid":
            raise AssertionError(
                "hybrid embedding backend did not start: %s"
                % diagnostics["fallback_reason"]
            )

    def test_unseen_korean_paraphrases_retrieve_second_beat_annotation(self) -> None:
        questions = (
            "건반 반주에서 둘째 박을 세게 치는 것은 어떤 장면을 그린 것인가?",
            "반주부의 2박에 붙은 강조는 무엇을 암시하나요?",
            "오른손과 왼손을 오가는 둘째 박의 강세로 슈베르트가 그려낸 것은?",
            "반주에서 두 번째 박이 유난히 도드라지는 까닭은 무엇인가?",
        )
        for question in questions:
            with self.subTest(question=question):
                results = self.index.search(
                    question,
                    piece="die-forelle",
                    measure_ranges=[[2, 5]],
                    top_k=6,
                )
                self.assertTrue(results)
                self.assertEqual(results[0].record["id"], "die-forelle-ku-002")
                self.assertEqual(
                    results[0].scope_match,
                    "overlaps_query_range",
                )
                self.assertEqual(
                    [result.record["id"] for result in results],
                    ["die-forelle-ku-002"],
                )

    def test_dense_answer_support_gate_rejects_near_miss_questions(self) -> None:
        questions = (
            "FIFA World Cup winner",
            "이 마디에서 바이올린 활의 재료는 무엇인가?",
            "피아노 반주의 두 번째 박 악센트는 어떤 화음으로 구성되는가?",
            "두 번째 박의 악센트와 슈베르트의 출생 연도를 함께 설명해 주세요.",
            "피아노 악센트를 연주할 때 권장 손가락 번호는 무엇인가요?",
            "ff와 pp는 각각 몇 데시벨로 연주해야 하나요?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    self.index.search(
                        question,
                        piece="die-forelle",
                        measure_ranges=[[2, 5]],
                        top_k=6,
                    ),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
