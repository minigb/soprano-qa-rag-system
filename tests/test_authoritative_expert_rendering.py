"""Production safeguards for reviewed expert-answer semantic fidelity."""

from __future__ import annotations

import unittest
from unittest import mock

from soprano_qa import service as qa
from soprano_qa.answer import (
    GeneratedAnswerRejected,
    answer_uses_formal_polite_korean,
    formalize_korean_answer,
    render_authoritative_expert_bundle,
    select_authoritative_expert_bundle,
    select_expert_bundle_with_model,
)
from soprano_qa.retrieval import SearchResult


FLOWERY_ANSWER = (
    "앞선 간주에서 느껴지던 분위기는 이후 박자·조성·반주 리듬의 "
    "변화와 함께 전환되어, 가사 ‘추위와 주림에 시달리어’와 "
    "겨울·고난·시련 같은 이미지를 나타낸다. 가창자도 이 변화를 "
    "음색과 음악으로 표현하는 것이 중요하다. 이처럼 작곡가는 "
    "박자와 조성을 바꾸어 곡의 분위기를 전환한다."
)


def make_result(
    record_id: str,
    answer: str,
    *,
    evidence_type: str = "expert_annotation",
    scope_match: str = "local_example",
    measure_range: list[list[int]] | None = None,
    source_ids: list[str] | None = None,
    alias_score: float = 1.0,
) -> SearchResult:
    ranges = measure_range if measure_range is not None else [[39, 49]]
    record = {
        "id": record_id,
        "evidence_type": evidence_type,
        "piece": "in-flowery-clouds",
        "work": "Lee Heung-ryul - In Flowery Clouds",
        "topic": "performance",
        "question": "",
        "answer": answer,
        "measure_range": ranges,
        "measure_scope": "local" if ranges else "global",
        "measure_status": "specific" if ranges else "whole_piece",
        "source_ids": source_ids or ["kim-in-flowery-clouds-07"],
    }
    return SearchResult(
        record=record,
        score=10.0,
        text_score=5.0,
        measure_score=1.0,
        piece_score=1.0,
        scope_match=scope_match,
        alias_score=alias_score,
        concept_coverage=1.0,
        content_concept_coverage=1.0,
        dense_score=0.8,
        dense_content_score=0.8,
        fusion_score=0.9,
        retrieval_mode="hybrid",
    )


class FixedIndex:
    configured_retrieval_mode = "hybrid"
    retrieval_mode = "hybrid"
    last_search_mode = "hybrid"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, **_kwargs):
        return list(self.results)


class AuthoritativeExpertRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._generation_preflight_patch = mock.patch.object(
            qa,
            "validate_generation_requirements",
        )
        self.generation_preflight = self._generation_preflight_patch.start()

    def tearDown(self) -> None:
        self._generation_preflight_patch.stop()

    def test_bundle_keeps_complete_units_order_paragraphs_and_deduplicates(
        self,
    ) -> None:
        first = make_result(
            "expert-1",
            "작곡가는 박자를 바꾸어 분위기를 전환한다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["source-a"],
        )
        duplicate = make_result(
            "expert-duplicate",
            "작곡가는 박자를 바꾸어 분위기를 전환한다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["source-a"],
        )
        second = make_result(
            "expert-2",
            "가창자는 그 변화를 음색으로 표현하는 것이 중요하다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["source-a"],
        )
        web = make_result(
            "webchunk-test",
            "웹의 관련 설명이다.",
            evidence_type="web_database",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=[],
            alias_score=0.0,
        )

        selected = select_authoritative_expert_bundle(
            [first, duplicate, second, web]
        )
        rendered = render_authoritative_expert_bundle(selected)

        self.assertEqual(selected, [first, second])
        self.assertEqual(
            rendered,
            (
                "작곡가는 박자를 바꾸어 분위기를 전환합니다.\n\n"
                "가창자는 그 변화를 음색으로 표현하는 것이 중요합니다."
            ),
        )
        self.assertNotIn("웹의 관련 설명", rendered)

    def test_constrained_selector_can_choose_a_lower_semantic_candidate(
        self,
    ) -> None:
        related = make_result(
            "expert-related",
            "고음 뒤에는 쉼표를 활용해 호흡한다.",
            source_ids=["source-related"],
            alias_score=0.0,
        )
        direct = make_result(
            "expert-direct",
            "도약 전에는 이완된 상태에서 울림을 유지한다.",
            source_ids=["source-direct"],
            alias_score=0.0,
        )
        for result in (related, direct):
            result.semantic_match_type = "selector_candidate"
        with mock.patch(
            "soprano_qa.answer.generate_json",
            return_value={
                "selected_ids": ["expert-direct"],
                "coverage": "direct",
                "reason_code": "direct_answer",
            },
        ) as generate_json:
            selected = select_expert_bundle_with_model(
                "/models/answer.gguf",
                piece="nella-fantasia",
                measure_ranges=[],
                question="상행 도약을 자연스럽게 하려면?",
                results=[related, direct],
                llm_settings={"n_ctx": 8192},
            )

        self.assertEqual(selected, [direct])
        generate_json.assert_called_once()

    def test_constrained_selector_rejects_unknown_id_without_rank_fallback(
        self,
    ) -> None:
        candidate = make_result(
            "expert-candidate",
            "호흡을 충분히 준비한다.",
            alias_score=0.0,
        )
        candidate.semantic_match_type = "selector_candidate"
        with (
            mock.patch(
                "soprano_qa.answer.generate_json",
                return_value={
                    "selected_ids": ["invented-ku"],
                    "coverage": "direct",
                    "reason_code": "direct_answer",
                },
            ),
            self.assertRaisesRegex(
                GeneratedAnswerRejected,
                "outside its candidates",
            ),
        ):
            select_expert_bundle_with_model(
                "/models/answer.gguf",
                piece="in-flowery-clouds",
                measure_ranges=[],
                question="어떻게 노래할까?",
                results=[candidate],
                llm_settings={},
            )

    def _assert_flowery_service_answer(
        self,
        *,
        measure_range: tuple[int, int] | None,
        scope_match: str,
    ) -> None:
        evidence = make_result(
            "in-flowery-clouds-ku-008",
            FLOWERY_ANSWER,
            scope_match=scope_match,
        )
        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=FixedIndex([evidence]),
            ),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            response = qa.ask(
                piece_id="in-flowery-clouds",
                question="이 곡에서 박자가 여러번 바뀌는 이유는 무엇인가?",
                measure_range=measure_range,
                generate=True,
                top_k=6,
            )

        generate.assert_not_called()
        self.assertEqual(
            response["answer"],
            formalize_korean_answer(FLOWERY_ANSWER),
        )
        self.assertEqual(response["generation_mode"], "expert_verbatim")
        self.assertEqual(response["answer_basis"], "authoritative_expert")
        self.assertIsNone(response["unavailable_reason"])
        self.assertTrue(answer_uses_formal_polite_korean(response["answer"]))
        self.assertNotIn("in-flowery-clouds-ku-008", response["answer"])
        self.assertNotIn("중요하다고 합니다", response["answer"])
        self.assertNotIn(
            "작곡가는 박자와 조성을 바꾸어 추위와 주림",
            response["answer"],
        )
        self.assertEqual(
            [item["id"] for item in response["evidence"]],
            ["in-flowery-clouds-ku-008"],
        )
        self.assertEqual(response["evidence"][0]["measure_ranges"], [[39, 49]])
        self.generation_preflight.assert_called_once_with(
            qa.SETTINGS["model_path"],
            qa.SETTINGS["llm"],
        )

    def test_exact_flowery_no_range_query_renders_without_role_drift(
        self,
    ) -> None:
        self._assert_flowery_service_answer(
            measure_range=None,
            scope_match="local_example",
        )

    def test_exact_flowery_ranged_query_uses_the_same_fidelity_guard(
        self,
    ) -> None:
        self._assert_flowery_service_answer(
            measure_range=(47, 48),
            scope_match="overlaps_query_range",
        )

    def test_web_only_grounding_uses_one_grounded_generation(self) -> None:
        web = make_result(
            "webchunk-test",
            "웹 데이터베이스 설명이다.",
            evidence_type="web_database",
            measure_range=[],
            scope_match="general_evidence",
            source_ids=[],
            alias_score=0.0,
        )
        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=FixedIndex([web]),
            ),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value="웹 근거에 따른 답변입니다. [E1]",
            ) as generate,
        ):
            response = qa.ask(
                piece_id="in-flowery-clouds",
                question="작품의 배경은 무엇인가?",
                measure_range=None,
                generate=True,
            )

        generate.assert_called_once()
        self.assertEqual(response["generation_mode"], "llm")
        self.assertEqual(response["answer_basis"], "retrieved_evidence")
        self.assertIsNone(response["unavailable_reason"])
        self.assertNotIn("[E1]", response["answer"])
        self.assertNotIn("webchunk-test", response["answer"])
        self.assertTrue(answer_uses_formal_polite_korean(response["answer"]))
        self.generation_preflight.assert_called_once()

    def test_dense_direct_una_interlude_renders_finalized_expert_unit(self) -> None:
        evidence = make_result(
            "una-voce-poco-fa-ku-016",
            (
                "긴 간주는 극적 전환 구간일 수 있다. 성악가는 앞뒤 "
                "가사의 흐름에 맞추어 표정과 움직임으로 반응해야 한다."
            ),
            scope_match="local_example",
            measure_range=[[44, 55]],
            source_ids=["target-source"],
            alias_score=0.0,
        )
        evidence.record["piece"] = "una-voce-poco-fa"
        evidence.record["work"] = "Rossini - Una voce poco fa"
        evidence.semantic_match_type = "dense"
        evidence.concept_coverage = 0.2
        evidence.content_concept_coverage = 0.2
        evidence.dense_score = 0.612
        evidence.dense_content_score = 0.606
        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=FixedIndex([evidence]),
            ),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            response = qa.ask(
                piece_id="una-voce-poco-fa",
                question=(
                    "이 곡에 간주가 너무 길어서 그동안 뭘 해야할지 "
                    "모르겠다"
                ),
                measure_range=None,
                generate=True,
                top_k=6,
            )

        generate.assert_not_called()
        self.assertEqual(response["generation_mode"], "expert_verbatim")
        self.assertIn("극적 전환 구간일 수 있습니다", response["answer"])
        self.assertNotIn("항상", response["answer"])
        self.assertNotIn("una-voce-poco-fa-ku-016", response["answer"])
        self.assertTrue(answer_uses_formal_polite_korean(response["answer"]))
        self.generation_preflight.assert_called_once()

    def test_broad_guidance_without_positive_alias_is_not_direct_authority(
        self,
    ) -> None:
        breathing = make_result(
            "broad-breathing",
            "호흡 계획을 먼저 세운다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["source-breathing"],
            alias_score=0.0,
        )
        diction = make_result(
            "broad-diction",
            "가사의 자음을 분명하게 전달한다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["source-diction"],
            alias_score=0.0,
        )
        expression = make_result(
            "broad-expression",
            "프레이즈의 정서 변화를 표현한다.",
            scope_match="local_example",
            source_ids=["source-expression"],
            alias_score=0.0,
        )
        generic_appendage = make_result(
            "generic-appendage",
            "작곡가의 생애에 관한 일반 설명이다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["source-generic"],
            alias_score=0.0,
        )
        for result in (breathing, diction, expression):
            result.semantic_match_type = "broad_guidance"
        generic_appendage.semantic_match_type = "dense"

        selected = select_authoritative_expert_bundle(
            [breathing, diction, expression, generic_appendage]
        )

        self.assertEqual(selected, [])

    def test_direct_broad_alias_does_not_append_generic_broad_facets(
        self,
    ) -> None:
        tone = make_result(
            "tone-direct",
            "탄력 있고 밝은 음색으로 표현한다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["target-source"],
            alias_score=1.0,
        )
        diction = make_result(
            "diction-direct",
            "딕션을 정확히 공부해 단어를 표현한다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["target-source"],
            alias_score=1.0,
        )
        generic = make_result(
            "generic-broad",
            "호흡과 프레이즈를 전반적으로 연습한다.",
            scope_match="general_evidence",
            measure_range=[],
            source_ids=["other-source"],
            alias_score=0.0,
        )
        for result in (tone, diction, generic):
            result.semantic_match_type = "broad_guidance"

        selected = select_authoritative_expert_bundle(
            [tone, diction, generic]
        )

        self.assertEqual(selected, [tone, diction])

    def test_direct_web_drops_a_lower_unrelated_expert_from_generation(
        self,
    ) -> None:
        web = make_result(
            "webchunk-composition-date",
            "이 작품의 작곡 연도는 웹 자료에 기록되어 있다.",
            evidence_type="web_database",
            measure_range=[],
            scope_match="general_evidence",
            source_ids=[],
            alias_score=0.0,
        )
        unrelated_expert = make_result(
            "die-forelle-ku-003",
            "슈베르트는 가곡의 위상을 높여 가곡의 왕으로 불린다.",
            measure_range=[],
            scope_match="general_evidence",
            source_ids=["kim-die-forelle-02"],
            alias_score=0.0,
        )
        # Match the live shape: direct web records rank first and the merely
        # related expert biography appears farther down the dense ranking.
        web.dense_score = 0.52
        web.dense_content_score = 0.53
        unrelated_expert.dense_score = 0.45
        unrelated_expert.dense_content_score = 0.50
        def web_answer(
            _model_path: str,
            messages: list[dict[str, str]],
            _settings: dict,
        ) -> str:
            prompt = messages[1]["content"]
            self.assertIn(web.record["answer"], prompt)
            self.assertNotIn(unrelated_expert.record["answer"], prompt)
            return "작곡 연도는 웹 자료에서 확인할 수 있습니다. [E1]"

        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=FixedIndex([web, unrelated_expert]),
            ),
            mock.patch.object(
                qa,
                "generate_llm",
                side_effect=web_answer,
            ) as generate,
        ):
            response = qa.ask(
                piece_id="die-forelle",
                question="이 곡은 언제 작곡되었나요?",
                measure_range=None,
                generate=True,
            )

        generate.assert_called_once()
        self.assertEqual(response["generation_mode"], "llm")
        self.assertEqual(response["answer_basis"], "retrieved_evidence")
        self.assertNotIn("가곡의 왕", response["answer"])
        self.assertEqual(
            [item["id"] for item in response["evidence"]],
            ["webchunk-composition-date"],
        )
        self.assertTrue(answer_uses_formal_polite_korean(response["answer"]))
        self.assertNotIn("webchunk-composition-date", response["answer"])
        self.generation_preflight.assert_called_once()

    def test_exact_ranged_expert_does_not_append_partial_source_overlap(
        self,
    ) -> None:
        direct = make_result(
            "die-forelle-ku-002",
            (
                "두 번째 박자의 악센트는 송어가 뛰어노는 모습을 "
                "표현한 것으로 볼 수 있다."
            ),
            scope_match="overlaps_query_range",
            measure_range=[[2, 27], [41, 54]],
            source_ids=["kim-die-forelle-02", "yeon-die-forelle-06"],
            alias_score=0.0,
        )
        background = make_result(
            "die-forelle-ku-003",
            (
                "슈베르트는 시와 반주, 성악의 위상을 높였기 때문에 "
                "가곡의 왕으로 불린다."
            ),
            scope_match="global_context",
            measure_range=[],
            source_ids=["kim-die-forelle-02", "yeon-die-forelle-01"],
            alias_score=0.0,
        )
        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=FixedIndex([direct, background]),
            ),
            mock.patch.object(qa, "generate_llm") as generate,
        ):
            response = qa.ask(
                piece_id="die-forelle",
                question=(
                    "피아노 파트에서 두 번째 박의 악센트는 무엇을 "
                    "표현할까?"
                ),
                measure_range=(2, 27),
                generate=True,
            )

        generate.assert_not_called()
        self.assertEqual(response["generation_mode"], "expert_verbatim")
        self.assertIn("송어가 뛰어노는 모습", response["answer"])
        self.assertNotIn("가곡의 왕", response["answer"])
        self.assertEqual(
            [item["id"] for item in response["evidence"]],
            ["die-forelle-ku-002"],
        )
        self.assertTrue(answer_uses_formal_polite_korean(response["answer"]))
        self.assertNotIn("die-forelle-ku-002", response["answer"])
        self.generation_preflight.assert_called_once()

    def test_authoritative_renderer_errors_are_not_model_failures(self) -> None:
        evidence = make_result(
            "in-flowery-clouds-ku-008",
            FLOWERY_ANSWER,
        )
        with (
            mock.patch.object(
                qa,
                "_get_index",
                return_value=FixedIndex([evidence]),
            ),
            mock.patch.object(qa, "generate_llm") as generate,
            mock.patch.object(
                qa,
                "render_authoritative_expert_bundle",
                side_effect=ValueError("invalid curated answer"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "invalid curated answer"):
                qa.ask(
                    piece_id="in-flowery-clouds",
                    question="이 곡에서 박자가 여러번 바뀌는 이유는 무엇인가?",
                    measure_range=None,
                    generate=True,
                )

        generate.assert_not_called()
        self.generation_preflight.assert_called_once()


if __name__ == "__main__":
    unittest.main()
