"""Optional regression checks against the configured local embedding model."""

from __future__ import annotations

import importlib.util
import os
import unittest

from soprano_qa.answer import (
    retrieve_generation_evidence,
    select_generation_evidence,
)
from soprano_qa.dense import (
    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
    SOURCE_QUESTION_LED_MATCH_TYPE,
    build_retrieval_index,
    retrieval_diagnostics,
)
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
                % (
                    diagnostics.get("last_dense_error")
                    or "no dense startup error was reported"
                )
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

    def test_measure_local_expert_semantics_win_without_selected_range(
        self,
    ) -> None:
        cases = (
            (
                "in-flowery-clouds",
                "악보의 페르마타 표시가 음원에서 들리는 음 길이와 "
                "일치하지 않을 때는 어떻게 해야 할까",
                "in-flowery-clouds-ku-001",
                [[1, 1]],
            ),
            (
                "una-voce-poco-fa",
                "이 곡에 간주가 너무 길어서 그동안 뭘 해야할지 모르겠다",
                "una-voce-poco-fa-ku-016",
                [[44, 51]],
            ),
        )
        for piece, question, expected_id, selected_range in cases:
            with self.subTest(piece=piece, scope="no-range"):
                results = self.index.search(
                    question,
                    piece=piece,
                    top_k=6,
                )
                self.assertTrue(results)
                self.assertEqual(results[0].record["id"], expected_id)
                self.assertEqual(results[0].scope_match, "local_example")
                self.assertEqual(
                    [
                        result.record["id"]
                        for result in select_generation_evidence(results)
                    ],
                    [expected_id],
                )

            with self.subTest(piece=piece, scope="selected-range"):
                ranged_results = self.index.search(
                    question,
                    piece=piece,
                    measure_ranges=selected_range,
                    top_k=6,
                )
                self.assertTrue(ranged_results)
                self.assertEqual(
                    ranged_results[0].record["id"],
                    expected_id,
                )
                self.assertEqual(
                    ranged_results[0].scope_match,
                    "overlaps_query_range",
                )

    def test_ranged_generation_focus_removes_overlap_only_neighbors(
        self,
    ) -> None:
        cases = (
            (
                "in-flowery-clouds",
                "첫 음을 내기가 어려우면 어떤 방법으로 해결해야 할까?",
                [[19, 21]],
                ["in-flowery-clouds-ku-006"],
            ),
            (
                "in-flowery-clouds",
                "《꽃구름 속에》의 전주가 주는 꽃가루와 바람의 "
                "느낌을 노래에 어떻게 이어 가야 할까?",
                [[1, 10]],
                ["in-flowery-clouds-ku-014"],
            ),
            (
                "in-flowery-clouds",
                "성악과 반주에 같은 멜로디가 함께 나올 때 어떻게 "
                "맞춰 연주해야 할까?",
                [[54, 60]],
                ["in-flowery-clouds-ku-020"],
            ),
            (
                "in-flowery-clouds",
                "곡이 다시 밝고 희망적인 분위기로 바뀔 때 어떻게 "
                "표현해야 할까?",
                [[60, 83]],
                ["in-flowery-clouds-ku-021"],
            ),
            (
                "la-capinera",
                "시작할 때 적절한 방법은 무엇일까?",
                [[11, 14]],
                ["la-capinera-ku-001"],
            ),
            (
                "la-capinera",
                "소리가 계속 끊겨 들리는 부분은 어떻게 노래해야 할까?",
                [[95, 96]],
                ["la-capinera-ku-010"],
            ),
        )
        for piece, question, measure_ranges, expected_ids in cases:
            with self.subTest(piece=piece, question=question):
                results, _messages = retrieve_generation_evidence(
                    self.index,
                    query=question,
                    piece=piece,
                    measure_ranges=measure_ranges,
                    measures="%d-%d" % tuple(measure_ranges[0]),
                    topic=None,
                    top_k=6,
                )

                self.assertEqual(
                    [result.record["id"] for result in results],
                    expected_ids,
                )

    def test_relevance_led_no_range_focus_covers_judge_regressions(
        self,
    ) -> None:
        cases = (
            (
                "die-forelle",
                "피아노 파트에서 두 번째 박의 악센트는 무엇을 표현할까?",
                ["die-forelle-ku-002"],
            ),
            (
                "in-flowery-clouds",
                "악보의 페르마타 표기와 음원에서 들리는 음의 길이가 "
                "다르면 어떻게 해야 할까?",
                ["in-flowery-clouds-ku-001"],
            ),
            (
                "la-capinera",
                "ff와 pp의 대비는 어떤 음악적 효과를 표현할까?",
                ["la-capinera-ku-018"],
            ),
            (
                "una-voce-poco-fa",
                "dolce amorosa를 이어 발음할 때 모음은 언제 어떻게 "
                "바꿀까?",
                ["una-voce-poco-fa-ku-017"],
            ),
            (
                "nella-fantasia",
                "노래하기 힘들거나 숨이 부족할 때 어떻게 해야 할까?",
                ["nella-fantasia-ku-008", "nella-fantasia-ku-009"],
            ),
        )
        for piece, question, expected_ids in cases:
            with self.subTest(piece=piece, question=question):
                raw_results = self.index.search(
                    question,
                    piece=piece,
                    top_k=6,
                )
                selected = select_generation_evidence(raw_results)

                self.assertEqual(
                    [result.record["id"] for result in selected],
                    expected_ids,
                )
                self.assertEqual(
                    selected[0].scope_match,
                    "local_example",
                )

    def test_source_question_led_focus_covers_synthesized_misses(
        self,
    ) -> None:
        cases = (
            (
                "dolce amorosa를 연결해 부를 때 모음을 바꾸는 시점과 "
                "방법은 무엇일까?",
                "una-voce-poco-fa-ku-017",
            ),
            (
                "dolce amorosa의 발음을 이을 때 모음은 언제 어떤 "
                "방식으로 바꿀까?",
                "una-voce-poco-fa-ku-017",
            ),
            (
                "왜 악센트가 첫 박이 아니라 다른 박에 놓여 있을까?",
                "una-voce-poco-fa-ku-020",
            ),
            (
                "첫 박을 벗어난 위치에 악센트를 표시한 이유는 "
                "무엇일까?",
                "una-voce-poco-fa-ku-020",
            ),
            (
                "악센트 표시가 첫 박 아닌 곳에 나오는 것은 어떤 "
                "까닭일까?",
                "una-voce-poco-fa-ku-020",
            ),
        )
        for question, expected_id in cases:
            with self.subTest(question=question):
                raw_results = self.index.search(
                    question,
                    piece="una-voce-poco-fa",
                    top_k=6,
                )
                selected = select_generation_evidence(raw_results)

                self.assertEqual(raw_results[0].record["id"], expected_id)
                self.assertEqual(
                    raw_results[0].semantic_match_type,
                    SOURCE_QUESTION_LED_MATCH_TYPE,
                )
                self.assertEqual(
                    [result.record["id"] for result in selected],
                    [expected_id],
                )

    def test_source_family_direct_answer_displaces_procedural_rescue(
        self,
    ) -> None:
        question = (
            "로시니가 일반적인 부점 리듬 대신 겹부점 리듬을 사용한 "
            "이유는 무엇일까?"
        )

        raw_results = self.index.search(
            question,
            piece="una-voce-poco-fa",
            top_k=6,
        )
        selected = select_generation_evidence(
            raw_results,
            query=question,
            piece="una-voce-poco-fa",
            measure_ranges=[],
        )

        self.assertEqual(
            [result.record["id"] for result in raw_results],
            ["una-voce-poco-fa-ku-004"],
        )
        self.assertEqual(
            raw_results[0].semantic_match_type,
            SOURCE_QUESTION_LED_MATCH_TYPE,
        )
        self.assertEqual(
            [result.record["id"] for result in selected],
            ["una-voce-poco-fa-ku-004"],
        )

    def test_no_range_generation_preserves_source_question_authority(
        self,
    ) -> None:
        question = (
            "로시니는 왜 보통의 부점 리듬이 아니라 "
            "겹부점 리듬을 선택했을까?"
        )

        results, _messages = retrieve_generation_evidence(
            self.index,
            query=question,
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["una-voce-poco-fa-ku-004"],
        )
        self.assertEqual(
            results[0].semantic_match_type,
            SOURCE_QUESTION_LED_MATCH_TYPE,
        )

    def test_no_range_generation_requires_explicit_requested_slots(
        self,
    ) -> None:
        cases = (
            (
                "una-voce-poco-fa",
                "셋잇단음표를 부를 때 모음은 어느 음에 놓아야 할까?",
                "una-voce-poco-fa-ku-010",
            ),
            (
                "nella-fantasia",
                "음원과 악보가 서로 일치하지 않을 때 무엇을 기준으로 "
                "노래해야 할까?",
                "nella-fantasia-ku-007",
            ),
        )
        for piece, question, expected_id in cases:
            with self.subTest(piece=piece, question=question):
                results, _messages = retrieve_generation_evidence(
                    self.index,
                    query=question,
                    piece=piece,
                    measure_ranges=[],
                    measures="",
                    topic=None,
                    top_k=6,
                )

                self.assertEqual(
                    [result.record["id"] for result in results],
                    [expected_id],
                )

    def test_ranged_causal_question_without_scoped_causal_unit_abstains(
        self,
    ) -> None:
        questions = (
            (
                "로시니가 일반적인 부점 리듬 대신 겹부점 리듬을 "
                "사용한 이유는 무엇일까?"
            ),
            "왜 로시니는 여기서 이 리듬을 썼는가?",
            "이 리듬은 어떤 효과를 내는가?",
            "이 리듬은 어떤 의미가 있는가?",
            "로시니가 이 리듬을 사용한 배경은 무엇인가?",
        )
        for question in questions:
            for measure in (63, 111):
                with self.subTest(question=question, measure=measure):
                    self.assertEqual(
                        self.index.search(
                            question,
                            piece="una-voce-poco-fa",
                            measure_ranges=[[measure, measure]],
                            top_k=6,
                        ),
                        [],
                    )
                    self.assertEqual(
                        self.index.last_route_reason,
                        SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
                    )

    def test_ranged_mixed_question_requires_scoped_causal_authority(
        self,
    ) -> None:
        questions = (
            (
                "로시니는 왜 이 리듬을 사용했고, 성악가는 "
                "어떻게 노래해야 할까?"
            ),
            "이 리듬을 사용한 이유와 노래하는 방법을 알려 주세요.",
            "이 리듬은 어떤 효과를 내며 어떻게 불러야 할까?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    self.index.search(
                        question,
                        piece="una-voce-poco-fa",
                        measure_ranges=[[63, 63]],
                        top_k=6,
                    ),
                    [],
                )
                self.assertEqual(
                    self.index.last_route_reason,
                    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
                )

    def test_causal_authority_covers_every_named_measure_location(
        self,
    ) -> None:
        questions = (
            (
                "14마디와 63마디에서 로시니가 겹부점 리듬을 "
                "사용한 이유는 무엇일까?"
            ),
            (
                "14마디와 63마디에서 로시니는 왜 이 리듬을 "
                "사용했고 성악가는 어떻게 노래해야 할까?"
            ),
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    self.index.search(
                        question,
                        piece="una-voce-poco-fa",
                        measure_ranges=[[14, 14], [63, 63]],
                        top_k=6,
                    ),
                    [],
                )
                self.assertEqual(
                    self.index.last_route_reason,
                    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
                )

    def test_performer_advice_cannot_answer_composer_choice_question(
        self,
    ) -> None:
        cases = (
            (
                "nella-fantasia",
                "작곡가는 이 부분에 음의 도약을 넣은 이유가 무엇인가요?",
                [[11, 17]],
            ),
            (
                "la-capinera",
                "작곡가가 어려운 발음 배치를 선택한 이유는 무엇인가요?",
                [[41, 43]],
            ),
            (
                "una-voce-poco-fa",
                "작곡가가 col canto와 a piacere를 표시한 이유는 무엇인가요?",
                [],
            ),
            (
                "in-flowery-clouds",
                "작곡가가 꽃바람에 스타카토를 쓰지 않은 이유는 무엇인가요?",
                [[11, 12]],
            ),
        )
        for piece, question, measure_ranges in cases:
            with self.subTest(piece=piece, question=question):
                self.assertEqual(
                    self.index.search(
                        question,
                        piece=piece,
                        measure_ranges=measure_ranges,
                        top_k=6,
                    ),
                    [],
                )
                self.assertEqual(
                    self.index.last_route_reason,
                    SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
                )

    def test_ranged_procedural_question_keeps_scoped_procedural_unit(
        self,
    ) -> None:
        questions = (
            "부점 리듬을 정확히 노래하려면 어떻게 해야 할까?",
            "의도치 않게 부점 리듬이 흐트러질 때 어떻게 노래해야 할까?",
            "이유 없이 부점 리듬이 밀리지 않게 하려면 어떻게 해야 할까?",
            "부점 리듬이 왜곡되지 않게 하려면 어떻게 해야 할까?",
        )
        for question in questions:
            for measure in (63, 111):
                with self.subTest(question=question, measure=measure):
                    results = self.index.search(
                        question,
                        piece="una-voce-poco-fa",
                        measure_ranges=[[measure, measure]],
                        top_k=6,
                    )
                    self.assertEqual(
                        [result.record["id"] for result in results],
                        ["una-voce-poco-fa-ku-005"],
                    )

    def test_mixed_causal_and_procedural_question_keeps_both_facets(
        self,
    ) -> None:
        results = self.index.search(
            (
                "로시니는 왜 겹부점 리듬을 사용했고, "
                "성악가는 이를 어떻게 노래해야 할까?"
            ),
            piece="una-voce-poco-fa",
            top_k=6,
        )

        expected_ids = [
            "una-voce-poco-fa-ku-004",
            "una-voce-poco-fa-ku-005",
        ]
        self.assertEqual(
            [result.record["id"] for result in results[:2]],
            expected_ids,
        )

        generation_results, _messages = retrieve_generation_evidence(
            self.index,
            query=(
                "로시니는 왜 겹부점 리듬을 사용했고, "
                "성악가는 이를 어떻게 노래해야 할까?"
            ),
            piece="una-voce-poco-fa",
            measure_ranges=[],
            measures="",
            topic=None,
            top_k=6,
        )
        self.assertEqual(
            {result.record["id"] for result in generation_results},
            set(expected_ids),
        )

    def test_mixed_question_requires_every_named_procedural_facet(
        self,
    ) -> None:
        results = self.index.search(
            (
                "박자와 조성을 바꾼 이유와 가사 발음을 어떻게 해야 "
                "하는지 알려 주세요."
            ),
            piece="in-flowery-clouds",
            measure_ranges=[[39, 49]],
            top_k=6,
        )

        self.assertEqual(results, [])
        self.assertEqual(
            retrieval_diagnostics(self.index)["route_reason"],
            SOURCE_FAMILY_MISSING_CAUSAL_AUTHORITY_REASON,
        )

    def test_mixed_question_keeps_complete_causal_and_expression_answer(
        self,
    ) -> None:
        results = self.index.search(
            (
                "박자와 조성을 바꾼 이유는 무엇이고 가창자는 "
                "어떻게 표현해야 할까?"
            ),
            piece="in-flowery-clouds",
            measure_ranges=[[39, 49]],
            top_k=6,
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["in-flowery-clouds-ku-008"],
        )

    def test_descriptive_how_change_is_not_procedural_advice(self) -> None:
        questions = (
            (
                "곡의 분위기는 박자가 바뀌면서 어떻게 달라지고, "
                "박자를 바꾼 이유는 무엇일까?"
            ),
            "왜 박자를 바꾸었으며 그에 따라 분위기는 어떻게 달라질까?",
        )
        for question in questions:
            with self.subTest(question=question):
                results = self.index.search(
                    question,
                    piece="in-flowery-clouds",
                    top_k=6,
                )
                self.assertEqual(
                    [result.record["id"] for result in results],
                    ["in-flowery-clouds-ku-008"],
                )

    def test_ranged_accent_variants_reject_ossia_and_keep_direct_unit(
        self,
    ) -> None:
        questions = (
            "왜 악센트가 첫 박이 아니라 다른 박에 놓여 있을까?",
            "첫 박을 벗어난 위치에 악센트를 표시한 이유는 무엇일까?",
            "악센트 표시가 첫 박 아닌 곳에 나오는 것은 어떤 까닭일까?",
        )
        ranges = ((68, 69), (72, 73), (92, 93), (96, 97))
        for question in questions:
            for start, end in ranges:
                with self.subTest(
                    question=question,
                    measure_range=(start, end),
                ):
                    results = self.index.search(
                        question,
                        piece="una-voce-poco-fa",
                        measure_ranges=[[start, end]],
                        top_k=6,
                    )
                    self.assertEqual(
                        [result.record["id"] for result in results],
                        ["una-voce-poco-fa-ku-020"],
                    )
                    self.assertEqual(
                        results[0].semantic_match_type,
                        SOURCE_QUESTION_LED_MATCH_TYPE,
                    )

    def test_near_tied_unrelated_nella_range_candidates_abstain(
        self,
    ) -> None:
        question = (
            "갑작스러운 상행 도약을 자연스럽게 소화하려면 "
            "어떻게 해야 할까?"
        )
        for start, end in ((11, 17), (28, 34), (40, 46)):
            with self.subTest(measure_range=(start, end)):
                results = self.index.search(
                    question,
                    piece="nella-fantasia",
                    measure_ranges=[[start, end]],
                    top_k=6,
                )

                self.assertEqual(results, [])
                self.assertEqual(
                    self.index.last_route_reason,
                    "ambiguous_dense_only_range",
                )


if __name__ == "__main__":
    unittest.main()
