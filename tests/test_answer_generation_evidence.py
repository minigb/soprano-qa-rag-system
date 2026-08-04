"""Focused tests for grounded-answer evidence selection and safeguards."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from typing import Any, Dict
from unittest import mock

from soprano_qa import answer as answer_cli
from soprano_qa.answer import (
    answer_overgeneralizes_local_examples,
    answer_references_secondary_evidence,
    answer_uses_formal_polite_korean,
    build_extractive_answer,
    build_messages,
    expert_prompt_factuality_material,
    GeneratedAnswerRejected,
    finalize_answer_citations,
    finalize_user_visible_answer,
    has_primary_grounding,
    is_grounded_insufficiency_answer,
    is_underspecified_ranged_query,
    select_generation_evidence,
    suspicious_generation_tokens,
)
from soprano_qa.retrieval import SearchResult


def make_result(
    record_id: str,
    *,
    evidence_type: str = "expert_annotation",
    alias_score: float = 0.0,
    measure_status: str = "specific",
    scope_match: str = "overlaps_query_range",
    source_ids: list[str] | None = None,
    score: float = 1.0,
    text_score: float = 1.0,
    semantic_match_type: str = "strict",
) -> SearchResult:
    record: Dict[str, Any] = {
        "id": record_id,
        "evidence_type": evidence_type,
        "piece": "die-forelle",
        "work": "Die Forelle",
        "topic": "performance",
        "measure_range": [[2, 5]],
        "measure_scope": "local",
        "source_ids": source_ids or ["kim-die-forelle-01"],
        "annotators": ["kim"],
        "measure_status": measure_status,
        "source_answer_context": [],
        "question": "",
        "answer": "전문가 답변",
    }
    return SearchResult(
        record=record,
        score=score,
        text_score=text_score,
        measure_score=0.0,
        piece_score=1.0,
        scope_match=scope_match,
        alias_score=alias_score,
        semantic_match_type=semantic_match_type,
    )


class GenerationEvidenceSelectionTests(unittest.TestCase):
    def test_keeps_only_maximum_positive_expert_alias_matches_in_input_order(
        self,
    ) -> None:
        weaker_expert = make_result("expert-weaker", alias_score=0.5)
        strongest_first = make_result("expert-strongest-first", alias_score=1.0)
        web_with_same_score = make_result(
            "web-strongest",
            evidence_type="web_database",
            alias_score=1.0,
        )
        unrelated_expert = make_result("expert-unrelated", alias_score=0.0)
        strongest_second = make_result("expert-strongest-second", alias_score=1.0)
        results = [
            weaker_expert,
            strongest_first,
            web_with_same_score,
            unrelated_expert,
            strongest_second,
        ]

        selected = select_generation_evidence(results)

        self.assertEqual(selected, [strongest_first, strongest_second])

    def test_returns_all_results_when_no_positive_expert_alias_exists(
        self,
    ) -> None:
        results = [
            make_result("expert-no-alias", alias_score=0.0),
            make_result(
                "web-positive-alias",
                evidence_type="web_database",
                alias_score=2.0,
            ),
        ]

        self.assertEqual(select_generation_evidence(results), results)

    def test_other_range_alias_is_retained_without_displacing_overlap(
        self,
    ) -> None:
        overlap = make_result(
            "overlap",
            alias_score=0.5,
            scope_match="overlaps_query_range",
        )
        stronger_other_range = make_result(
            "other-range",
            alias_score=1.0,
            scope_match="other_range_context",
        )

        selected = select_generation_evidence(
            [overlap, stronger_other_range]
        )

        self.assertEqual(selected, [overlap, stronger_other_range])
        self.assertTrue(has_primary_grounding(selected))










    def test_other_range_context_is_capped_at_two_records(self) -> None:
        overlap = make_result("overlap", alias_score=0.0)
        contexts = [
            make_result(
                f"other-range-{index}",
                alias_score=0.0,
                scope_match="other_range_context",
            )
            for index in range(3)
        ]

        selected = select_generation_evidence([overlap, *contexts])

        self.assertEqual(selected, [overlap, *contexts[:2]])

    def test_prompt_marks_other_range_evidence_as_context_only(self) -> None:
        overlap = make_result("overlap")
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
        )

        prompt = build_messages(
            "die-forelle",
            "2-5",
            "악센트는 무엇을 표현할까?",
            [overlap, other_range],
        )[1]["content"]

        self.assertIn("evidence_role: selected_range_support", prompt)
        self.assertIn("evidence_role: other_range_context_only", prompt)
        self.assertIn(
            "citation_label: (none; context-only evidence is not citable)",
            prompt,
        )
        self.assertIn("internal range-mismatch signal", prompt)
        self.assertIn(
            "do not infer, cite, paraphrase, or transfer",
            prompt,
        )

    def test_local_example_keeps_canonical_ranges_out_of_no_range_prose(
        self,
    ) -> None:
        local = make_result(
            "una-voce-poco-fa-ku-028",
            scope_match="local_example",
        )
        local.record["piece"] = "una-voce-poco-fa"
        local.record["work"] = "Una voce poco fa"
        local.record["measure_range"] = [[63, 63], [82, 83]]
        local.record["answer"] = (
            "고음이 강박에 놓이지 않은 경우에는 단어와 음악의 "
            "강세 관계에 유의해 유연하게 노래해야 한다."
        )
        local.record["source_answer_context"] = [
            {
                "source_id": "yeon-una-voce-poco-fa-11",
                "question": "고음을 어떻게 불러야 할까?",
                "answer": (
                    "첫 마디가 아니라 82마디에서는 고음이 강박에 "
                    "놓이지 않는다. m. 51 및 measures 52–53의 "
                    "표기는 참고하지 말고, 68, 134마디는 지나치게 "
                    "강조하지 않는다."
                ),
                "legacy_measure_range_hints": [[82, 82]],
            }
        ]

        messages = build_messages(
            "una-voce-poco-fa",
            "",
            "이 노래를 부를 때 무엇에 유의해야 할까?",
            [local],
        )
        system_prompt = messages[0]["content"]
        context = messages[1]["content"]

        self.assertIn("Evidence_role=local_example", system_prompt)
        self.assertIn("evidence_role: local_example", context)
        self.assertNotIn("measure_range: 63, 82-83", context)
        self.assertIn("citable passage-specific advice", context)
        self.assertIn("hidden presentation metadata", context)
        self.assertIn("does not establish", context)
        self.assertNotIn("82마디에서는", context)
        self.assertNotIn("첫 마디", context)
        self.assertNotIn("m. 51", context)
        self.assertNotIn("measures 52–53", context)
        self.assertNotIn("68, 134마디", context)
        self.assertNotIn("해당 대목에서는", context)
        self.assertNotIn("expert_source_answer", context)
        self.assertIn(local.record["answer"], context)
        authority = expert_prompt_factuality_material(local.record, [])
        self.assertFalse(authority["range_partitioned"])
        self.assertEqual(
            authority,
            {
                "range_partitioned": False,
                "answer_texts": [local.record["answer"]],
                "question_contexts": [],
                "source_ids": [],
            },
        )
        self.assertEqual(local.record["measure_range"], [[63, 63], [82, 83]])
        extractive = build_extractive_answer([local])
        self.assertIn(
            (
                "고음이 강박에 놓이지 않은 경우에는 단어와 음악의 "
                "강세 관계에 유의해 유연하게 노래해야 합니다."
            ),
            extractive,
        )
        self.assertNotIn("범위 안내", extractive)
        self.assertNotIn("확인된 국소 예시", extractive)
        self.assertNotRegex(extractive, r"(?:63|82\s*[-–—~]\s*83)\s*마디")
        self.assertNotIn("[E1]", extractive)
        self.assertNotIn("una-voce-poco-fa-ku-028", extractive)

    def test_local_only_answer_cannot_claim_piece_wide_frequency(
        self,
    ) -> None:
        local = make_result(
            "la-capinera-ku-021",
            scope_match="local_example",
        )
        local.record["measure_range"] = [[30, 30], [33, 33]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "이 곡 전반에는 도약음이 많이 나오므로 항상 "
                    "고음을 약하게 불러야 한다. [E1]"
                ),
                [local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "질문하신 도약음에서는 지나치게 강조되지 않도록 "
                    "유의한다. [E1]"
                ),
                [local],
            )
        )
        for unsupported_claim in (
            "이 노래에는 도약음이 자주 나오므로 조심한다. [E1]",
            "고음이 강박에 있지 않은 경우가 많으므로 조심한다. [E1]",
            "대부분의 구간에서 고음을 가볍게 부른다. [E1]",
            "전 곡에 걸쳐 같은 방식으로 노래한다. [E1]",
            "모든 대목에서 같은 강세를 사용한다. [E1]",
        ):
            with self.subTest(unsupported_claim=unsupported_claim):
                self.assertTrue(
                    answer_overgeneralizes_local_examples(
                        unsupported_claim,
                        [local],
                    )
                )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "30마디와 33마디에서는 도약하는 고음이 "
                    "지나치게 강조되지 않도록 유의한다. [E1]"
                ),
                [local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "30마디에서는 도약음이 자주 나오지는 않는다고 "
                    "본다. [E1]"
                ),
                [local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "30마디의 근거만으로 곡 전반에 적용된다고 "
                    "확인되지 않았다. [E1]"
                ),
                [local],
            )
        )

        whole = make_result(
            "la-capinera-ku-whole",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "이 곡 전반의 가창에 유의해야 한다. [E1] [E2]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "이 곡 전반의 가창에 유의해야 한다. [E1]",
                [whole, local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "이 곡 전반의 가창에 유의하되, 30마디의 "
                    "도약음을 국소 예시로 참고한다. [E1] [E2]"
                ),
                [whole, local],
            )
        )

    def test_no_range_rejects_every_numeric_local_measure_locator(
        self,
    ) -> None:
        first = make_result(
            "una-voce-poco-fa-ku-028",
            scope_match="local_example",
        )
        first.record["measure_range"] = [[63, 63], [82, 83]]
        second = make_result(
            "una-voce-poco-fa-ku-020",
            scope_match="local_example",
        )
        second.record["measure_range"] = [[68, 69], [72, 73]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "63마디에서 고음을 유연하게 부른다. [E1] [E2]",
                [first, second],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "63마디에서 고음을 유연하게 부른다. [E1]\n"
                    "68–69마디에서 악센트를 가볍게 표현한다. "
                    "[E2]"
                ),
                [first, second],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "82마디부터 83마디에서 고음을 유연하게 부른다. [E1]",
                [first, second],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "64마디에서 고음을 유연하게 부른다. [E1]",
                [first, second],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "질문하신 고음에서는 유연하게 노래한다. [E1]\n"
                    "이어지는 악센트는 가볍게 표현한다. [E2]"
                ),
                [first, second],
            )
        )

    def test_unlabeled_body_cannot_be_attributed_to_retrieved_locals(
        self,
    ) -> None:
        whole = make_result(
            "whole",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        local = make_result("local", scope_match="local_example")
        local.record["measure_range"] = [[30, 30]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "도약 직전에 충분히 호흡한다.\n\n"
                    "제공된 검색 근거: [E1] [E2]"
                ),
                [whole, local],
            )
        )

    def test_measure_locators_need_query_or_cited_local_authority(
        self,
    ) -> None:
        whole = make_result(
            "la-capinera-ku-028",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        whole.record["source_answer_context"] = [
            {
                "source_id": "yeon-la-capinera-10",
                "question": "",
                "answer": "135마디에서 선율을 이어 부른다.",
                "legacy_measure_range_hints": [[135, 135]],
            }
        ]
        local = make_result(
            "la-capinera-ku-011",
            scope_match="local_example",
        )
        local.record["measure_range"] = [[41, 43]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "135마디에서 선율을 이어 부른다. [E1]",
                [whole],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "135마디에서 선율을 이어 부른다. [E1]",
                [whole, local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "41–43마디에서 레가토를 유지한다. [E1]",
                [whole, local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "41–43마디에서 레가토를 유지한다. [E2]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "기교를 더할 때는 모음으로 선율을 이어 부른다. [E1]",
                [whole, local],
            )
        )

        selected = make_result(
            "die-forelle-ku-002",
            scope_match="overlaps_query_range",
        )
        selected.record["measure_range"] = [[2, 27]]
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "2–5마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "3마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "1–6마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "작품의 분위기를 먼저 이해한다. [E1]",
                [whole, local],
            )
        )

    def test_extractive_answer_does_not_present_other_range_as_grounding(
        self,
    ) -> None:
        overlap = make_result("overlap")
        overlap.record["answer"] = "선택 범위 답변입니다."
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간 답변입니다."

        mixed = build_extractive_answer([overlap, other_range])
        secondary_only = build_extractive_answer([other_range])

        self.assertIn("선택 범위 답변", mixed)
        self.assertNotIn("다른 구간 답변", mixed)
        self.assertIn(
            "선택하신 마디에는 이 내용을 직접 적용하기 어렵습니다.",
            secondary_only,
        )
        self.assertIn("곡의 다른 부분에 해당합니다", secondary_only)
        self.assertNotIn("다른 구간의 관련 주석", secondary_only)
        self.assertIn("다른 구간 답변", secondary_only)

    def test_secondary_and_uncited_drafts_are_rejected(
        self,
    ) -> None:
        overlap = make_result("die-forelle-ku-001")
        overlap.record["answer"] = "선택 범위의 근거"
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간의 강약 교대"

        with self.assertRaises(GeneratedAnswerRejected):
            finalize_answer_citations(
                (
                    "다른 구간의 강약 교대 [E2] "
                    "[die-forelle-ku-002]"
                ),
                [overlap, other_range],
            )
        with self.assertRaises(GeneratedAnswerRejected):
            finalize_answer_citations(
                "인용을 생략한 답변",
                [overlap, other_range],
            )


    def test_pipeline_scope_scaffolding_is_removed_but_answer_is_kept(
        self,
    ) -> None:
        legacy = (
            "범위 안내: 아래 내용은 표시된 마디에서 확인된 국소 "
            "예시입니다.\n\n"
            "확인된 국소 예시(마디 44-55): 아리아의 긴 간주에서는 "
            "표정과 움직임으로 극의 흐름에 반응해야 합니다."
        )
        markdown = (
            "**범위 안내:** 내부 범위입니다.\n\n"
            "### **확인된 국소 예시(마디 44–55):** 긴 간주에서는 "
            "다음 장면을 준비해야 합니다."
        )
        copied_prompt_fields = (
            "measure_range: 44-55\n"
            "evidence_role: local_example\n"
            "긴 간주에서는 앞뒤 감정의 변화를 이어야 합니다. [E1]"
        )

        self.assertEqual(
            finalize_user_visible_answer(legacy),
            (
                "아리아의 긴 간주에서는 표정과 움직임으로 극의 "
                "흐름에 반응해야 합니다."
            ),
        )
        self.assertEqual(
            finalize_user_visible_answer(markdown),
            "긴 간주에서는 다음 장면을 준비해야 합니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer(copied_prompt_fields),
            "긴 간주에서는 앞뒤 감정의 변화를 이어야 합니다.",
        )

    def test_inline_pipeline_fields_and_parenthesized_citations_never_leak(
        self,
    ) -> None:
        self.assertEqual(
            finalize_user_visible_answer(
                "정상 답변입니다. citation_label: E1 "
                "evidence_role: local_example"
            ),
            "정상 답변입니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer(
                "정상 답변입니다. 범위 안내: 내부 범위입니다."
            ),
            "정상 답변입니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer("정상 답변입니다. (E1)"),
            "정상 답변입니다.",
        )
        self.assertEqual(
            finalize_user_visible_answer("citation_label E1"),
            "현재 확인된 정보만으로는 이 질문에 정확히 "
            "답변하기 어렵습니다.",
        )

        for answer in (
            "die_forelle_ku_010에 따르면 이렇게 부릅니다.",
            "[die forelle ku 010]에 따르면 이렇게 부릅니다.",
            "die–forelle–ku–010에 따르면 이렇게 부릅니다.",
        ):
            with self.subTest(answer=answer):
                finalized = finalize_user_visible_answer(answer)
                self.assertNotRegex(
                    finalized,
                    r"(?i)die[\s_–—-]+forelle[\s_–—-]+ku",
                )

        opaque_id_variants = (
            "webchunk test 01",
            "web_chunk_test_01",
            "web–chunk–test–01",
            "web source 01",
            "web-claim-test",
            "sqa 123",
            "sqa_123",
            "sqa–123",
        )
        unavailable = (
            "현재 확인된 정보만으로는 이 질문에 정확히 "
            "답변하기 어렵습니다."
        )
        for opaque_id in opaque_id_variants:
            with self.subTest(opaque_id=opaque_id, form="bare"):
                self.assertEqual(
                    finalize_user_visible_answer(
                        f"정상 답변입니다. {opaque_id}"
                    ),
                    unavailable,
                )
            with self.subTest(opaque_id=opaque_id, form="bracketed"):
                self.assertEqual(
                    finalize_user_visible_answer(
                        f"[{opaque_id}]에 따르면 이렇게 부릅니다."
                    ),
                    "이렇게 부릅니다.",
                )

        for answer in (
            "measure_range: 44-55",
            "범위 안내: 내부 범위입니다.",
            "제공된 검색 근거:",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(
                    finalize_user_visible_answer(answer),
                    "현재 확인된 정보만으로는 이 질문에 정확히 "
                    "답변하기 어렵습니다.",
                )

    def test_every_pipeline_field_blocks_separator_variants(self) -> None:
        unavailable = (
            "현재 확인된 정보만으로는 이 질문에 정확히 "
            "답변하기 어렵습니다."
        )
        separators = (" ", "-", "_", "**", "`", "–", "—")
        for field_name in answer_cli.PIPELINE_FIELD_NAMES:
            words = field_name.split("_")
            for separator in separators:
                variant = separator.join(words)
                with self.subTest(
                    field_name=field_name,
                    separator=separator,
                ):
                    self.assertEqual(
                        finalize_user_visible_answer(
                            f"정상 답변입니다. {variant} 값을 사용합니다."
                        ),
                        unavailable,
                    )

    def test_pipeline_guards_preserve_ordinary_korean_music_prose(self) -> None:
        answers = (
            "선택한 마디 범위에서는 호흡을 조절합니다.",
            "이 근거의 역할은 프레이즈의 방향을 설명하는 것입니다.",
            "국소적인 예를 들어 음색 변화를 설명합니다.",
            "웹 자료를 읽더라도 악보를 먼저 확인합니다.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertEqual(finalize_user_visible_answer(answer), answer)


    def test_cli_generation_error_propagates_without_retry(self) -> None:
        web = make_result(
            "web-only",
            evidence_type="web_database",
            scope_match="general_evidence",
        )
        web.record["answer"] = "렌더링하면 안 되는 웹 원문"
        web.content_concept_coverage = 1.0

        class WebIndex:
            configured_retrieval_mode = "hybrid"
            retrieval_mode = "hybrid"
            last_search_mode = "hybrid"
            last_route_reason = "dense_result_returned"

            @staticmethod
            def search(**_kwargs):
                return [web]

        args = SimpleNamespace(
            settings=None,
            piece="die-forelle",
            measures="",
            measure_ranges=[],
            topic=None,
            question="이 자료는 무엇을 설명하나요?",
            top_k=6,
            no_generate=False,
            rebuild_corpus=False,
            json=True,
        )
        settings = {
            "corpus_path": "/tmp/corpus.json",
            "model_path": "/tmp/model.gguf",
            "llm": {},
        }
        with (
            mock.patch.object(answer_cli, "parse_args", return_value=args),
            mock.patch.object(
                answer_cli,
                "load_settings",
                return_value=settings,
            ),
            mock.patch.object(answer_cli, "validate_retrieval_requirements"),
            mock.patch.object(answer_cli, "validate_generation_requirements"),
            mock.patch.object(answer_cli, "ensure_corpus"),
            mock.patch.object(answer_cli, "load_corpus", return_value=[]),
            mock.patch.object(
                answer_cli,
                "build_retrieval_index",
                return_value=WebIndex(),
            ),
            mock.patch.object(
                answer_cli,
                "generate",
                side_effect=ValueError(
                    "requested tokens exceed the context window"
                ),
            ) as generate,
        ):
            with self.assertRaises(ValueError):
                answer_cli.main()
        generate.assert_called_once()

    def test_deictic_generic_range_question_is_underspecified(self) -> None:
        self.assertTrue(
            is_underspecified_ranged_query(
                "이 부분은 어떻게 노래해야 할까?",
                "nella-fantasia",
                [[11, 17]],
            )
        )
        self.assertFalse(
            is_underspecified_ranged_query(
                "이 부분에서 호흡은 어디에서 해야 할까?",
                "nella-fantasia",
                [[11, 17]],
            )
        )

    def test_secondary_citation_variants_are_rejected(self) -> None:
        overlap = make_result("die-forelle-ku-001")
        overlap.record["answer"] = "선택 범위의 근거"
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간 위험 주장"

        secondary_references = [
            "[E 2]",
            "[Evidence 2]",
            "[E–2]",
            "[Evidence no. 2]",
            "Evidence 2",
            "근거: 2",
            "DIE-FORELLE-KU-002",
        ]
        for reference in secondary_references:
            with self.subTest(reference=reference):
                with self.assertRaises(GeneratedAnswerRejected):
                    finalize_answer_citations(
                        f"다른 구간 위험 주장 {reference}",
                        [overlap, other_range],
                    )

    def test_secondary_citation_never_substitutes_primary_text(self) -> None:
        primary = [
            make_result(
                f"primary-{index}",
                scope_match="overlaps_query_range",
                source_ids=[f"source-{index}"],
            )
            for index in range(4)
        ]
        for index, result in enumerate(primary):
            result.record["answer"] = f"선택 근거 {index}입니다."
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
            source_ids=["other-source"],
        )

        with self.assertRaises(GeneratedAnswerRejected):
            finalize_answer_citations(
                "다른 구간의 위험 주장 [E5]",
                [*primary, other_range],
            )

    def test_music_and_korean_counts_are_not_secondary_citations(self) -> None:
        overlap = make_result("die-forelle-ku-001")
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )

        ordinary_uses = {
            "최저음은 E2이다. [E1]": "최저음은 E2입니다.",
            "최저음은 E2. [E1]": "최저음은 E2.",
            "음역 하한은 E2! [E1]": "음역 하한은 E2!",
            "근거 2가지를 설명한다. [E1]": "근거 2가지를 설명합니다.",
            "출처 2곳을 비교한다. [E1]": "출처 2곳을 비교합니다.",
        }
        for answer, expected in ordinary_uses.items():
            with self.subTest(answer=answer):
                finalized = finalize_answer_citations(
                    answer,
                    [overlap, other_range],
                )
                self.assertEqual(finalized, expected)
                self.assertNotIn("[E1]", finalized)
                self.assertNotIn("die-forelle-ku-001", finalized)
                if "E2" in answer:
                    self.assertIn("E2", finalized)

    def test_user_visible_answer_strips_internal_ids_and_formalizes_style(
        self,
    ) -> None:
        primary = make_result("die-forelle-ku-001")
        secondary = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )

        finalized = finalize_user_visible_answer(
            (
                "첫 문장은 근거를 설명한다. [E1, E2]\n"
                "둘째 문장은 이어진다. die-forelle-ku-001"
            ),
            [primary, secondary],
        )

        self.assertEqual(
            finalized,
            "첫 문장은 근거를 설명합니다.\n둘째 문장은 이어집니다.",
        )
        self.assertNotIn("[E1", finalized)
        self.assertNotIn("die-forelle-ku-001", finalized)
        self.assertNotIn("die-forelle-ku-002", finalized)

        citation_suffix_cases = {
            (
                "설명입니다. [E1]에 따르면, 전주가 있을 수 있습니다."
            ): "설명입니다. 전주가 있을 수 있습니다.",
            (
                "설명입니다. [die-forelle-ku-001]에서는 12마디를 "
                "확인합니다."
            ): "설명입니다. 12마디를 확인합니다.",
            (
                "설명입니다. [die-forelle-ku-001]에서 보듯이, 24마디를 "
                "확인합니다."
            ): "설명입니다. 24마디를 확인합니다.",
            (
                "설명입니다. die-forelle-ku-001에 의하면, 원본을 "
                "확인합니다."
            ): "설명입니다. 원본을 확인합니다.",
        }
        for answer, expected in citation_suffix_cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(
                    finalize_user_visible_answer(answer, [primary]),
                    expected,
                )

        already_orphaned_cases = {
            "설명입니다. 에 따르면, 전주가 있습니다.": (
                "설명입니다. 전주가 있습니다."
            ),
            "설명입니다. 에서는 12마디를 확인합니다.": (
                "설명입니다. 12마디를 확인합니다."
            ),
            "설명입니다. 에서 보듯이, 24마디를 확인합니다.": (
                "설명입니다. 24마디를 확인합니다."
            ),
        }
        for answer, expected in already_orphaned_cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(
                    finalize_user_visible_answer(answer, [primary]),
                    expected,
                )

        no_space_boundary = finalize_user_visible_answer(
            "악보대로 부르지 않는다.성악 성부를 확인한다."
        )
        self.assertEqual(
            no_space_boundary,
            "악보대로 부르지 않습니다. 성악 성부를 확인합니다.",
        )

        duplicated_source_spacing = finalize_user_visible_answer(
            "이 곡은 일관적으로  서정적이다.\t\t표현을 유지한다."
        )
        self.assertEqual(
            duplicated_source_spacing,
            "이 곡은 일관적으로 서정적입니다. 표현을 유지합니다.",
        )

        lexical_nida_cases = {
            "결코 쉬운 곡은 아니다.": "결코 쉬운 곡은 아닙니다.",
            "선율이 여러 성부를 다니다.": "선율이 여러 성부를 다닙니다.",
            "중요한 의미를 지니다.": "중요한 의미를 지닙니다.",
        }
        for answer, expected in lexical_nida_cases.items():
            with self.subTest(answer=answer):
                self.assertFalse(answer_uses_formal_polite_korean(answer))
                finalized = finalize_user_visible_answer(answer)
                self.assertEqual(finalized, expected)
                self.assertTrue(
                    answer_uses_formal_polite_korean(finalized)
                )

        for answer in (
            "결코 쉬운 곡은 아닙니다.",
            "선율이 여러 성부를 다닙니다.",
            "중요한 의미를 지닙니다.",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(finalize_user_visible_answer(answer), answer)
                self.assertTrue(answer_uses_formal_polite_korean(answer))

    def test_nonpolite_sentence_endings_fail_closed(self) -> None:
        for answer in (
            "이렇게 해 보자.",
            "어떻게 할까?",
            "숨을 쉬어라.",
            "이 부분을 조심해.",
            "Please sing softly.",
            "Use a lighter tone.",
            "OK",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(answer_uses_formal_polite_korean(answer))
                self.assertEqual(
                    finalize_user_visible_answer(answer),
                    "현재 확인된 정보만으로는 이 질문에 정확히 "
                    "답변하기 어렵습니다.",
                )

        for answer in (
            "이렇게 해 봅시다.",
            "어떻게 해야 합니까?",
            "숨을 쉬십시오.",
            "이 부분을 조심하세요.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(answer_uses_formal_polite_korean(answer))
                self.assertEqual(finalize_user_visible_answer(answer), answer)

    def test_pipeline_vocabulary_fails_closed_at_display_boundary(self) -> None:
        for term in (
            "knowledge unit",
            "expert annotation",
            "retrieved evidence",
            "retrieval",
            "corpus",
            "지식 단위",
            "전문가 주석",
            "검색 파이프라인",
            "검색 결과",
            "코퍼스",
        ):
            with self.subTest(term=term):
                self.assertEqual(
                    finalize_user_visible_answer(
                        f"{term}에 따르면 이렇게 부릅니다."
                    ),
                    "현재 확인된 정보만으로는 이 질문에 정확히 "
                    "답변하기 어렵습니다.",
                )

    def test_prompt_uses_curated_answer_instead_of_split_raw_source(
        self,
    ) -> None:
        result = make_result("la-capinera-ku-001")
        result.record["answer"] = "pianissimo로 시작하는 것이 좋다."
        result.record["source_answer_context"] = [
            {
                "source_id": "kim-la-capinera-01",
                "question": "어떻게 시작하고 발음할까?",
                "answer": (
                    "pianissimo로 시작하고 l과 r을 정확히 발음한다."
                ),
                "legacy_measure_range_hints": [[11, 14], [78, 81]],
                "linked_knowledge_unit_ids": [
                    "la-capinera-ku-001",
                    "la-capinera-ku-002",
                ],
            }
        ]
        messages = build_messages(
            "la-capinera",
            "78-81",
            "어떻게 시작할까?",
            [result],
        )
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        self.assertIn("authoritative claim text", system_prompt)
        self.assertIn(
            "Raw source annotations, legacy range hints, superseded wording",
            system_prompt,
        )
        self.assertNotIn("split_source_context_withheld", user_prompt)
        self.assertIn("pianissimo로 시작하는 것이 좋다.", user_prompt)
        self.assertNotIn("l과 r을 정확히 발음한다", user_prompt)


class GenerationArtifactRepairTests(unittest.TestCase):
    def test_detects_unseen_camel_case_inside_korean_text(self) -> None:
        messages = [{"role": "user", "content": "꾸밈음을 가볍게 부른다."}]

        self.assertEqual(
            suspicious_generation_tokens(
                "꾸SizePolicy는 가볍게 부른다.",
                messages,
            ),
            ["SizePolicy"],
        )
        self.assertEqual(
            suspicious_generation_tokens(
                "Larghetto는 여유 있게 부른다.",
                messages,
            ),
            [],
        )






class RangeScopedSourceContextTests(unittest.TestCase):
    def test_no_range_uses_only_curated_answer_for_nonlocal_units(
        self,
    ) -> None:
        for scope_match, measure_status in (
            ("general_evidence", "whole_piece"),
            ("unspecified_scope", "unspecified"),
        ):
            with self.subTest(scope_match=scope_match):
                result = make_result(
                    f"source-{scope_match}",
                    scope_match=scope_match,
                    measure_status=measure_status,
                )
                result.record["measure_range"] = []
                result.record["answer"] = "검토가 완료된 지식 단위 답변이다."
                result.record["source_answer_context"] = [
                    {
                        "source_id": "source-a",
                        "question": "135마디에서 어떻게 부를까?",
                        "answer": "135마디에서 모음으로 선율을 이어 부른다.",
                        "legacy_measure_range_hints": [[135, 135]],
                    }
                ]

                prompt = build_messages(
                    "la-capinera",
                    "",
                    "이 곡을 부를 때 무엇에 유의할까?",
                    [result],
                )[1]["content"]
                authority = expert_prompt_factuality_material(
                    result.record,
                    [],
                )

                self.assertNotIn("135마디", prompt)
                self.assertNotIn("해당 대목에서", prompt)
                self.assertNotIn("expert_source_answer", prompt)
                self.assertIn(result.record["answer"], prompt)
                self.assertEqual(
                    authority,
                    {
                        "range_partitioned": False,
                        "answer_texts": [result.record["answer"]],
                        "question_contexts": [],
                        "source_ids": [],
                    },
                )

    def test_selected_range_uses_curated_answer_and_withholds_raw_sources(
        self,
    ) -> None:
        merged = make_result("la-capinera-ku-018")
        merged.record["measure_range"] = [[51, 59], [117, 121]]
        merged.record["answer"] = (
            "앞 구간의 두 블록 대비와 뒤 구간의 빠른 교대를 함께 설명한다."
        )
        merged.record["source_answer_context"] = [
            {
                "source_id": "kim-la-capinera-17",
                "question": (
                    "117마디의 ff와 121마디의 pp는 왜 빠르게 "
                    "번갈아 나올까?"
                ),
                "answer": "빠른 교대는 메아리 효과로 볼 수 있다.",
                "legacy_measure_range_hints": [[117, 121]],
            },
            {
                "source_id": "yeon-la-capinera-07",
                "question": "",
                "answer": (
                    "51-59마디의 ff와 pp는 51-54마디 "
                    "55-59마디의 대비가 드러나도록 표현한다."
                ),
                "legacy_measure_range_hints": [[51, 59]],
            },
        ]

        prompt = build_messages(
            "la-capinera",
            "51-58",
            "왜 빠르게 번갈아 나올까?",
            [merged],
        )[1]["content"]

        self.assertNotIn("range_partitioned_source_context", prompt)
        self.assertNotIn("expert_source_answer", prompt)
        self.assertNotIn("51-59마디의 ff와 pp", prompt)
        self.assertNotIn("빠른 교대는 메아리 효과로 볼 수 있다.", prompt)
        self.assertIn(merged.record["answer"], prompt)
        authority = expert_prompt_factuality_material(
            merged.record,
            [[51, 58]],
        )
        self.assertEqual(
            authority,
            {
                "range_partitioned": False,
                "answer_texts": [merged.record["answer"]],
                "question_contexts": [],
                "source_ids": [],
            },
        )

    def test_narrow_range_cannot_restore_conflicting_raw_timing(self) -> None:
        merged = make_result("in-flowery-clouds-ku-008")
        merged.record["measure_range"] = [[45, 47]]
        merged.record["answer"] = (
            "박자와 조성의 변화가 곡의 분위기를 전환한다."
        )
        merged.record["source_answer_context"] = [
            {
                "source_id": "kim-in-flowery-clouds-07",
                "question": "박자가 바뀌는 이유는 무엇일까?",
                "answer": "47마디에서 박자와 조성이 동시에 바뀐다.",
                "legacy_measure_range_hints": [[47, 47]],
            },
            {
                "source_id": "yeon-in-flowery-clouds-08",
                "question": "분위기는 언제 달라질까?",
                "answer": "45마디부터 장조에서 단조로 바뀐다.",
                "legacy_measure_range_hints": [[45, 45]],
            },
        ]

        prompt = build_messages(
            "in-flowery-clouds",
            "45",
            "선택한 부분에서 분위기가 달라지는 이유는 무엇일까?",
            [merged],
        )[1]["content"]
        authority = expert_prompt_factuality_material(
            merged.record,
            [[45, 45]],
        )

        self.assertIn(merged.record["answer"], prompt)
        self.assertNotIn("47마디에서 박자와 조성이 동시에", prompt)
        self.assertNotIn("45마디부터 장조에서 단조로", prompt)
        self.assertNotIn("expert_source_answer", prompt)
        self.assertEqual(authority["answer_texts"], [merged.record["answer"]])
        self.assertFalse(authority["range_partitioned"])
        self.assertEqual(authority["question_contexts"], [])
        self.assertEqual(authority["source_ids"], [])

    def test_no_range_withholds_raw_sources_and_uses_curated_rewrite(
        self,
    ) -> None:
        merged = make_result("la-capinera-ku-018")
        merged.record["answer"] = "두 범위를 함께 설명한다."
        merged.record["source_answer_context"] = [
            {
                "source_id": "source-a",
                "question": "",
                "answer": "앞 범위 설명",
                "legacy_measure_range_hints": [[51, 59]],
            },
            {
                "source_id": "source-b",
                "question": "",
                "answer": "뒤 범위 설명",
                "legacy_measure_range_hints": [[117, 121]],
            },
        ]

        prompt = build_messages(
            "la-capinera",
            "",
            "전체적으로 어떻게 표현할까?",
            [merged],
        )[1]["content"]

        self.assertNotIn("range_partitioned_source_context: true", prompt)
        self.assertNotIn("앞 범위 설명", prompt)
        self.assertNotIn("뒤 범위 설명", prompt)
        self.assertNotIn("expert_source_answer", prompt)
        self.assertIn("두 범위를 함께 설명한다.", prompt)
        authority = expert_prompt_factuality_material(
            merged.record,
            [],
        )
        self.assertFalse(authority["range_partitioned"])
        self.assertEqual(authority["answer_texts"], [merged.record["answer"]])
        self.assertEqual(authority["question_contexts"], [])
        self.assertEqual(authority["source_ids"], [])


if __name__ == "__main__":
    unittest.main()
