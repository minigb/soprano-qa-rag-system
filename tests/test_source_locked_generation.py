"""Focused tests for the source-locked full-unit generation path."""

from __future__ import annotations

import io
import json
import re
from types import SimpleNamespace
import unittest
from unittest import mock

from soprano_qa import answer
from soprano_qa import service
from soprano_qa.answer import (
    answer_uses_formal_polite_korean,
    answer_uses_source_locked_formal_polite_korean,
    finalized_primary_grounding_results,
    formalize_korean_answer,
    generate_source_locked_grounded_answer,
)
from soprano_qa.source_locked import (
    RenderedSentence,
    RenderedSourceLockedUnit,
    SourceLockedProtocolError,
    build_source_locked_messages,
    build_source_locked_units,
    format_source_locked_answer,
    normalize_source_text,
    parse_source_locked_render,
    segment_source_sentences,
    source_locked_fidelity_warnings,
    source_locked_response_format,
)
from soprano_qa.retrieval import SearchResult


def expert_result(
    record_id: str,
    text: str,
    *,
    scope_match: str = "general_evidence",
    measure_status: str = "whole_piece",
    **extra,
):
    record = {
        "id": record_id,
        "evidence_type": "expert_annotation",
        "answer": text,
        "measure_status": measure_status,
        "measure_range": [],
        "source_ids": [],
        "web_source_ids": [],
        "claim_ids": [],
        **extra,
    }
    return SimpleNamespace(record=record, scope_match=scope_match)


def web_result(record_id: str, text: str, **extra):
    record = {
        "id": record_id,
        "evidence_type": "web_database",
        "answer": text,
        "measure_range": [],
        "source_ids": [],
        "web_source_ids": ["web-source-secret"],
        "claim_ids": ["claim-secret"],
        "sources": [{"url": "https://example.invalid/private-prompt"}],
        "usage_class": "research_open",
        "generated_text_license": "CC-BY-4.0",
        **extra,
    }
    return SimpleNamespace(record=record, scope_match="general_evidence")


def exact_copy_backend(calls):
    def generate(_model_path, messages, _settings, **kwargs):
        calls.append((messages, kwargs))
        prompt, marker = messages[-1]["content"].rsplit("\n/no_think", 1)
        assert marker == ""
        payload = json.loads(prompt)
        sources = payload["finalized_source_unit"]["source_sentences"]
        return json.dumps(
            {
                "sentences": [
                    {
                        "sentence_id": item["sentence_id"],
                        "text": item["exact_text"],
                    }
                    for item in sources
                ]
            },
            ensure_ascii=False,
        )

    return generate


class SentenceSegmentationTests(unittest.TestCase):
    def assert_rejoins(self, source: str, expected_count: int) -> None:
        normalized = normalize_source_text(source)
        sentences = segment_source_sentences(source)
        self.assertEqual(len(sentences), expected_count)
        self.assertEqual(" ".join(item.text for item in sentences), normalized)
        for sentence in sentences:
            self.assertEqual(
                normalized[sentence.start : sentence.end],
                sentence.text,
            )

    def test_score_abbreviations_initials_catalog_tokens_and_decimals(self) -> None:
        source = (
            "D.S. al Coda를 따른다. D.C. al Fine도 따른다. "
            "Op. 32와 op. 7, No. 2는 G. Verdi가 아니라 G. E. Blake와 "
            "O. Ditson, G. Schirmer 자료다. CC0 1.0과 청구기호 "
            "ML30.8b.S35를 기록한다."
        )
        self.assert_rejoins(source, 4)

    def test_uncertain_bracketed_dates_do_not_split(self) -> None:
        source = (
            "판본은 [185-?]로 추정하지만 확정하지 않는다. "
            "다른 판본의 연도도 [1816?]로 기록한다."
        )
        self.assert_rejoins(source, 2)

    def test_nella_ku_012_is_four_sentences(self) -> None:
        source = (
            "D.S. al Coda는 악보의 앞부분으로 돌아가 같은 부분을 다시 "
            "노래하기 위한 장치 중 하나이다. 이와 같은 음악적 지시 "
            "기호에는 도돌이표, Da Capo, Dal Segno, Coda 등이 있다. "
            "D.S.는 Dal Segno의 축약형으로, Segno 표시로 돌아간다는 "
            "뜻이다. al Coda는 Coda 표시에서 다음 Coda 사인으로 "
            "넘어간다는 뜻이다."
        )
        self.assert_rejoins(source, 4)

    def test_web_initials_remain_inside_their_sentences(self) -> None:
        source = (
            "필라델피아 G. E. Blake의 주소를 기록한다. "
            "보스턴 O. Ditson의 판본과 뉴욕 G. Schirmer 판본도 있다."
        )
        self.assert_rejoins(source, 2)

    def test_dynamic_abbreviation_with_attached_korean_particle_stays_whole(self) -> None:
        source = (
            "‘o mia gentil’의 dim.은 가사를 표현하기 위한 지시로 "
            "보는 것이 좋겠다. 뒤의 단어도 주의해 노래해야 한다."
        )
        self.assert_rejoins(source, 2)
        self.assertIn("dim.은", normalize_source_text(source))

    def test_quotes_and_korean_no_space_boundary_split(self) -> None:
        source = "첫 문장이다.”다음 문장이다! Last sentence?"
        self.assert_rejoins(source, 3)

    def test_quoted_stage_direction_stays_in_its_reporting_sentence(self) -> None:
        source = (
            "오페라 스코어의 처음에는 무대 구성을 설명하는 경우가 많다. "
            "이 아리아의 지문에는 “창문은 닫혀있다. 로지나는 손에 "
            "편지를 들고 있다.”라고 되어 있다. 이런 지문을 미리 알면 "
            "상황을 더 깊이 이해할 수 있다."
        )
        self.assert_rejoins(source, 3)
        normalized = normalize_source_text(source)
        self.assertIn("들고 있다.”라고", normalized)
        quoted_sentence = segment_source_sentences(source)[1].text
        self.assertIn("창문은 닫혀있다.", quoted_sentence)
        self.assertIn("들고 있다.”라고 되어 있다.", quoted_sentence)
        rendered = formalize_korean_answer(quoted_sentence)
        self.assertIn("창문은 닫혀있다.", rendered)
        self.assertIn("들고 있다.”라고 되어 있습니다.", rendered)


class FormalizationTests(unittest.TestCase):
    def test_plain_and_past_edge_cases_become_formal_polite(self) -> None:
        cases = {
            "아니다.": "아닙니다.",
            "다르다.": "다릅니다.",
            "쉽다.": "쉽습니다.",
            "어렵다.": "어렵습니다.",
            "하나이다.": "하나입니다.",
            "강조했다.": "강조했습니다.",
            "바뀌었다.": "바뀌었습니다.",
            "노래하는 것이 좋겠다.": "노래하는 것이 좋습니다.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                actual = formalize_korean_answer(source)
                self.assertEqual(actual, expected)
                self.assertTrue(answer_uses_formal_polite_korean(actual))

    def test_plain_anida_is_not_false_positive_polite(self) -> None:
        self.assertFalse(answer_uses_formal_polite_korean("아니다."))

    def test_trailing_non_korean_text_is_not_vacuously_polite(self) -> None:
        self.assertFalse(
            answer_uses_source_locked_formal_polite_korean(
                "검토된 문장입니다 HACK"
            )
        )


class SourceIsolationAndSchemaTests(unittest.TestCase):
    def test_finalized_primary_filter_keeps_expert_and_web_only(self) -> None:
        primary_expert = expert_result("piece-ku-001", "검토된 답이다.")
        primary_web = web_result("webchunk-001", "검토된 웹 요약이다.")
        provisional = expert_result(
            "piece-ku-002",
            "검토 전 답이다.",
            measure_status="waiting_for_review",
        )
        context = expert_result(
            "piece-ku-003",
            "다른 범위 문장이다.",
            scope_match="other_range_context",
            measure_status="specific",
        )
        overlap = expert_result(
            "piece-ku-004",
            "선택 범위 문장이다.",
            scope_match="overlaps_query_range",
            measure_status="specific",
        )
        filtered = finalized_primary_grounding_results(
            [overlap, context, provisional]
        )
        self.assertEqual(filtered, [overlap])
        self.assertEqual(
            finalized_primary_grounding_results([primary_expert, primary_web]),
            [primary_expert, primary_web],
        )

        malformed_web = web_result(
            "webchunk-malformed",
            "출처 연결이 없는 웹 요약이다.",
            claim_ids=[],
        )
        self.assertEqual(
            finalized_primary_grounding_results([malformed_web]),
            [],
        )

    def test_prompt_contains_only_one_answer_and_temporary_sentence_ids(self) -> None:
        secret_question = "raw-source-question-secret"
        record_id = "piece-ku-secret-999"
        source = "D.S. al Coda를 따른다. 다음 문장이다."
        unit = build_source_locked_units([(2, source)])[0]
        messages = build_source_locked_messages(unit)
        prompt = messages[-1]["content"]
        self.assertIn("D.S. al Coda를 따른다.", prompt)
        self.assertIn("다음 문장이다.", prompt)
        self.assertNotIn(secret_question, prompt)
        self.assertNotIn(record_id, prompt)
        self.assertNotIn("source_ids", prompt)
        self.assertNotIn("claim_ids", prompt)
        self.assertNotIn("knowledge_unit_id", prompt)
        self.assertNotIn("질문:", prompt)

    def test_schema_requires_exact_sentence_count_and_known_ids(self) -> None:
        unit = build_source_locked_units(
            [(0, "첫 문장이다. 둘째 문장이다.")]
        )[0]
        schema = source_locked_response_format(unit)["schema"]
        sentence_schema = schema["properties"]["sentences"]
        self.assertEqual(sentence_schema["minItems"], 2)
        self.assertEqual(sentence_schema["maxItems"], 2)
        self.assertEqual(
            sentence_schema["items"]["properties"]["sentence_id"]["enum"],
            ["S1", "S2"],
        )

    def test_parse_rejects_missing_or_reordered_sentence(self) -> None:
        unit = build_source_locked_units(
            [(0, "첫 문장이다. 둘째 문장이다.")]
        )[0]
        raw = json.dumps(
            {
                "sentences": [
                    {"sentence_id": "S2", "text": "둘째 문장입니다."},
                    {"sentence_id": "S1", "text": "첫 문장입니다."},
                ]
            },
            ensure_ascii=False,
        )
        with self.assertRaises(SourceLockedProtocolError):
            parse_source_locked_render(
                raw,
                unit,
                formalize_fn=formalize_korean_answer,
                polite_check_fn=answer_uses_source_locked_formal_polite_korean,
            )

    def test_generic_plain_model_output_is_formalized_before_validation(self) -> None:
        unit = build_source_locked_units([(0, "결코 쉬운 곡은 아니다.")])[0]
        raw = json.dumps(
            {"sentences": [{"sentence_id": "S1", "text": unit.source_text}]},
            ensure_ascii=False,
        )
        rendered = parse_source_locked_render(
            raw,
            unit,
            formalize_fn=formalize_korean_answer,
            polite_check_fn=answer_uses_source_locked_formal_polite_korean,
        )
        self.assertEqual(rendered.text, "결코 쉬운 곡은 아닙니다.")

    def test_catalog_identifier_that_resembles_sentence_id_is_preserved(self) -> None:
        source = "의회도서관 청구기호는 ML30.8b.S35 F7 1821이다."
        unit = build_source_locked_units([(0, source)])[0]
        raw = json.dumps(
            {"sentences": [{"sentence_id": "S1", "text": source}]},
            ensure_ascii=False,
        )
        rendered = parse_source_locked_render(
            raw,
            unit,
            formalize_fn=formalize_korean_answer,
            polite_check_fn=answer_uses_source_locked_formal_polite_korean,
        )
        self.assertIn("ML30.8b.S35", rendered.text)

    def test_new_internal_sentence_label_is_rejected(self) -> None:
        unit = build_source_locked_units([(0, "검토된 문장이다.")])[0]
        raw = json.dumps(
            {
                "sentences": [
                    {
                        "sentence_id": "S1",
                        "text": "검토된 문장입니다. E1",
                    }
                ]
            },
            ensure_ascii=False,
        )
        with self.assertRaises(SourceLockedProtocolError):
            parse_source_locked_render(
                raw,
                unit,
                formalize_fn=formalize_korean_answer,
                polite_check_fn=answer_uses_source_locked_formal_polite_korean,
            )

    def test_new_prompt_control_tokens_are_rejected(self) -> None:
        unit = build_source_locked_units([(0, "검토된 문장이다.")])[0]
        for token in (
            "finalized_source_unit",
            "source_sentences",
            "source_sentence",
            "/no_think",
        ):
            with self.subTest(token=token):
                raw = json.dumps(
                    {
                        "sentences": [
                            {
                                "sentence_id": "S1",
                                "text": "%s 검토된 문장입니다." % token,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
                with self.assertRaises(SourceLockedProtocolError):
                    parse_source_locked_render(
                        raw,
                        unit,
                        formalize_fn=formalize_korean_answer,
                        polite_check_fn=(
                            answer_uses_source_locked_formal_polite_korean
                        ),
                    )


class FidelityAndFormattingTests(unittest.TestCase):
    def render(self, source: str, output: str) -> RenderedSourceLockedUnit:
        unit = build_source_locked_units([(0, source)])[0]
        raw = json.dumps(
            {
                "sentences": [
                    {"sentence_id": "S1", "text": output}
                ]
            },
            ensure_ascii=False,
        )
        return parse_source_locked_render(
            raw,
            unit,
            formalize_fn=formalize_korean_answer,
            polite_check_fn=answer_uses_source_locked_formal_polite_korean,
        )

    def test_source_numeric_and_protected_marker_changes_are_warned(self) -> None:
        rendered = self.render(
            "가능하면 2~3배로 부르는 것이 좋다.",
            "반드시 2~4배로 부르는 것이 좋습니다.",
        )
        codes = {
            warning.code
            for warning in source_locked_fidelity_warnings(rendered)
        }
        self.assertIn("source_text_changed", codes)
        self.assertIn("numeric_content_changed", codes)
        self.assertIn("protected_marker_changed", codes)

    def test_recommendation_polite_conversion_has_no_fidelity_warning(self) -> None:
        rendered = self.render(
            "발랄한 이미지라고 생각하고 노래부르는 것이 좋겠다.",
            "발랄한 이미지라고 생각하고 노래부르는 것이 좋겠습니다.",
        )
        self.assertEqual(
            rendered.text,
            "발랄한 이미지라고 생각하고 노래부르는 것이 좋습니다.",
        )
        self.assertEqual(source_locked_fidelity_warnings(rendered), [])

    def test_exact_source_formalization_does_not_create_false_warnings(self) -> None:
        cases = (
            ("악보를 편집하기도 한다.", "악보를 편집하기도 합니다."),
            ("결코 쉬운 곡은 아니다.", "결코 쉬운 곡은 아닙니다."),
            ("연습하면 도움이 된다.", "연습하면 도움이 됩니다."),
            ("주의해서 공부해야 한다.", "주의해서 공부해야 합니다."),
        )
        for source, output in cases:
            with self.subTest(source=source):
                rendered = self.render(source, output)
                self.assertEqual(source_locked_fidelity_warnings(rendered), [])

    def test_multiple_units_use_generic_bullets_and_exact_title_stripping(self) -> None:
        def rendered(unit_id, result_index, text):
            source = build_source_locked_units([(result_index, text)])[0]
            return RenderedSourceLockedUnit(
                unit_id=unit_id,
                result_index=result_index,
                evidence_label="E%d" % (result_index + 1),
                source_text=source.source_text,
                source_sentences=source.sentences,
                rendered_sentences=(
                    RenderedSentence("S1", formalize_korean_answer(text)),
                ),
            )

        units = [
            rendered("U1", 0, "《Die Forelle》는 빠른 곡이다."),
            rendered("U2", 1, "《Die Forelle》에서는 딕션이 중요하다."),
            rendered("U3", 2, "《Die Forelle》는 밝은 음색이 중요하다."),
        ]
        formatted = format_source_locked_answer(units)
        self.assertTrue(formatted.startswith("다음 사항을 함께 참고"))
        self.assertEqual(formatted.count("\n- "), 3)
        self.assertEqual(formatted.count("《Die Forelle》"), 1)
        self.assertIn("- 딕션이 중요합니다.", formatted)
        self.assertIn("- 밝은 음색이 중요합니다.", formatted)

        two_units = format_source_locked_answer(units[:2])
        self.assertTrue(two_units.startswith("다음 사항을 함께 참고"))
        self.assertEqual(two_units.count("\n- "), 2)
        self.assertEqual(two_units.count("《Die Forelle》"), 1)
        self.assertIn("- 딕션이 중요합니다.", two_units)

    def test_repeated_title_stripping_supports_korean_case_particles(self) -> None:
        def rendered(unit_id: str, result_index: int, text: str):
            source = build_source_locked_units([(result_index, text)])[0]
            return RenderedSourceLockedUnit(
                unit_id=unit_id,
                result_index=result_index,
                evidence_label="E%d" % (result_index + 1),
                source_text=source.source_text,
                source_sentences=source.sentences,
                rendered_sentences=(
                    RenderedSentence("S1", formalize_korean_answer(text)),
                ),
            )

        for particle, expected in (
            ("에서", "호흡을 연결합니다."),
            ("의", "분위기를 살립니다."),
            ("를", "부를 때 딕션에 유의합니다."),
        ):
            with self.subTest(particle=particle):
                units = [
                    rendered("U1", 0, "《Nella fantasia》는 서정적인 곡이다."),
                    rendered(
                        "U2",
                        1,
                        "《Nella fantasia》%s %s" % (particle, expected),
                    ),
                ]
                formatted = format_source_locked_answer(units)
                self.assertEqual(formatted.count("《Nella fantasia》"), 1)
                self.assertIn("- " + expected, formatted)

        title_after_generic = [
            rendered("U1", 0, "호흡을 안정적으로 유지해야 한다."),
            rendered("U2", 1, "《Nella fantasia》는 서정적인 곡이다."),
            rendered("U3", 2, "《Nella fantasia》에서 호흡을 연결한다."),
        ]
        formatted = format_source_locked_answer(title_after_generic)
        self.assertEqual(formatted.count("《Nella fantasia》"), 1)
        self.assertIn("- 호흡을 연결합니다.", formatted)


class OrchestratorTests(unittest.TestCase):
    def test_every_primary_unit_is_rendered_in_isolated_schema_call(self) -> None:
        results = [
            expert_result("die-forelle-ku-009", "《Die Forelle》는 쉽지 않다."),
            expert_result("die-forelle-ku-017", "《Die Forelle》에서는 딕션이 중요하다."),
            expert_result("die-forelle-ku-016", "《Die Forelle》는 밝은 음색이 중요하다."),
        ]
        calls = []
        with (
            mock.patch.object(answer, "build_extractive_answer") as extractive,
            mock.patch.object(answer, "generate") as old_one_pass,
        ):
            visible, legacy, structured = generate_source_locked_grounded_answer(
                model_path="/models/qwen.gguf",
                llm_settings={"max_tokens": 128},
                results=results,
                measure_ranges=[],
                generate_fn=exact_copy_backend(calls),
            )
        extractive.assert_not_called()
        old_one_pass.assert_not_called()
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("response_format" in kwargs for _, kwargs in calls))
        for messages, _kwargs in calls:
            prompt = messages[-1]["content"]
            self.assertEqual(prompt.count("finalized_source_unit"), 1)
            self.assertNotIn("question", prompt.lower())
            self.assertNotIn("die-forelle-ku-", prompt)
        self.assertEqual(visible.count("《Die Forelle》"), 1)
        self.assertNotRegex(visible, r"\[(?:E|U|S)\d+\]")
        self.assertIsNone(legacy)
        self.assertEqual(structured, [])


class ServicePathTests(unittest.TestCase):
    @staticmethod
    def result(text: str) -> SearchResult:
        return SearchResult(
            record={
                "id": "die-forelle-ku-999",
                "piece": "die-forelle",
                "evidence_type": "expert_annotation",
                "answer": text,
                "measure_status": "whole_piece",
                "measure_range": [],
                "source_ids": [],
                "web_source_ids": [],
                "claim_ids": [],
            },
            score=1.0,
            text_score=1.0,
            measure_score=0.0,
            piece_score=1.0,
            scope_match="general_evidence",
        )

    def test_service_generation_uses_source_locked_orchestrator_fields(self) -> None:
        result = self.result("검토된 문장이다.")
        calls = []
        with (
            mock.patch.object(service, "validate_generation_requirements"),
            mock.patch.object(service, "_get_index", return_value=object()),
            mock.patch.object(
                service,
                "retrieve_generation_evidence",
                return_value=([result], []),
            ),
            mock.patch.object(
                service,
                "retrieval_diagnostics",
                return_value={},
            ),
            mock.patch.object(
                service,
                "generate_llm",
                side_effect=exact_copy_backend(calls),
            ),
            mock.patch.object(service, "build_extractive_answer") as extractive,
        ):
            response = service.ask(
                piece_id="die-forelle",
                question="이 문장은 무엇인가요?",
                measure_range=None,
                generate=True,
            )
        extractive.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertEqual(response["answer"], "검토된 문장입니다.")
        self.assertIsNone(response["generation_validation_warning"])
        self.assertEqual(response["generation_validation_warnings"], [])
        self.assertEqual(response["generation_mode"], "llm")
        self.assertEqual(response["answer_basis"], "retrieved_evidence")

    def test_cli_generation_uses_same_source_locked_orchestrator_fields(self) -> None:
        result = self.result("검토된 문장이다.")
        args = SimpleNamespace(
            settings=None,
            piece="die-forelle",
            measures="",
            selected_measure_ranges=[],
            measure_ranges=[],
            topic=None,
            question="이 문장은 무엇인가요?",
            top_k=6,
            no_generate=False,
            rebuild_corpus=False,
            json=True,
        )
        settings = {
            "model_path": "/models/qwen.gguf",
            "llm": {"max_tokens": 128},
            "corpus_path": "/data/corpus.json",
        }
        output = io.StringIO()
        with (
            mock.patch.object(answer, "parse_args", return_value=args),
            mock.patch.object(answer, "load_settings", return_value=settings),
            mock.patch.object(answer, "validate_retrieval_requirements"),
            mock.patch.object(answer, "validate_generation_requirements"),
            mock.patch.object(answer, "ensure_corpus"),
            mock.patch.object(answer, "load_corpus", return_value=[]),
            mock.patch.object(
                answer,
                "build_retrieval_index",
                return_value=object(),
            ),
            mock.patch.object(
                answer,
                "retrieve_generation_evidence",
                return_value=([result], []),
            ),
            mock.patch.object(
                answer,
                "generate_source_locked_grounded_answer",
                return_value=(
                    "검토된 문장입니다.",
                    "source_text_changed: 검토 경고",
                    [
                        {
                            "stage": "source_fidelity",
                            "code": "source_text_changed",
                            "message": "검토 경고",
                        }
                    ],
                ),
            ) as orchestrator,
            mock.patch.object(answer, "retrieval_diagnostics", return_value={}),
            mock.patch.object(answer, "build_evidence_notices", return_value=[]),
            mock.patch("sys.stdout", new=output),
        ):
            answer.main()

        orchestrator.assert_called_once_with(
            model_path="/models/qwen.gguf",
            llm_settings={"max_tokens": 128},
            results=[result],
            measure_ranges=[],
        )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["answer"], "검토된 문장입니다.")
        self.assertEqual(payload["generation_mode"], "llm")
        self.assertEqual(
            payload["generation_validation_warning"],
            "source_text_changed: 검토 경고",
        )
        self.assertEqual(
            payload["generation_validation_warnings"],
            [
                {
                    "stage": "source_fidelity",
                    "code": "source_text_changed",
                    "message": "검토 경고",
                }
            ],
        )

    def test_explicit_retrieval_only_mode_may_still_use_extraction(self) -> None:
        result = self.result("검토된 문장이다.")
        with (
            mock.patch.object(service, "_get_index") as get_index,
            mock.patch.object(
                get_index.return_value,
                "search",
                return_value=[result],
            ),
            mock.patch.object(
                service,
                "retrieval_diagnostics",
                return_value={},
            ),
            mock.patch.object(
                service,
                "build_extractive_answer",
                return_value="검토된 문장입니다.",
            ) as extractive,
            mock.patch.object(service, "generate_llm") as generate,
        ):
            response = service.ask(
                piece_id="die-forelle",
                question="검색만 합니다.",
                measure_range=None,
                generate=False,
            )
        extractive.assert_called_once()
        generate.assert_not_called()
        self.assertEqual(response["generation_validation_warnings"], [])

    def test_web_answer_is_source_locked_without_provenance_in_prompt(self) -> None:
        results = [
            web_result(
                "webchunk-001",
                "LOC 레코드는 G. E. Blake의 주소를 기록한다.",
            )
        ]
        calls = []
        visible, _legacy, _structured = generate_source_locked_grounded_answer(
            model_path="/models/qwen.gguf",
            llm_settings={},
            results=results,
            measure_ranges=[],
            generate_fn=exact_copy_backend(calls),
        )
        prompt = calls[0][0][-1]["content"]
        self.assertNotIn("webchunk-001", prompt)
        self.assertNotIn("web-source-secret", prompt)
        self.assertNotIn("claim-secret", prompt)
        self.assertNotIn("example.invalid", prompt)
        self.assertEqual(
            visible,
            "LOC 레코드는 G. E. Blake의 주소를 기록합니다.",
        )

    def test_fidelity_warning_never_replaces_nonempty_answer(self) -> None:
        results = [
            expert_result(
                "la-capinera-ku-013",
                "페르마타는 2~3배로 부르는 것이 좋다.",
            )
        ]

        def mutating_backend(_model, _messages, _settings, **_kwargs):
            return json.dumps(
                {
                    "sentences": [
                        {
                            "sentence_id": "S1",
                            "text": "페르마타는 2~4배로 부르는 것이 좋습니다.",
                        }
                    ]
                },
                ensure_ascii=False,
            )

        visible, legacy, structured = generate_source_locked_grounded_answer(
            model_path="/models/qwen.gguf",
            llm_settings={},
            results=results,
            measure_ranges=[],
            generate_fn=mutating_backend,
        )
        self.assertEqual(visible, "페르마타는 2~4배로 부르는 것이 좋습니다.")
        self.assertIsNotNone(legacy)
        self.assertIn(
            "numeric_content_changed",
            {warning["code"] for warning in structured},
        )

    def test_invalid_json_is_hard_error_without_fallback(self) -> None:
        results = [expert_result("piece-ku-001", "검토된 문장이다.")]
        with (
            mock.patch.object(answer, "build_extractive_answer") as extractive,
            self.assertRaises(SourceLockedProtocolError),
        ):
            generate_source_locked_grounded_answer(
                model_path="/models/qwen.gguf",
                llm_settings={},
                results=results,
                measure_ranges=[],
                generate_fn=lambda *_args, **_kwargs: "not json",
            )
        extractive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
