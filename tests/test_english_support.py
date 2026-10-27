"""English question/answer support without changing the Korean path."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from soprano_qa import answer as answer_module
from soprano_qa import service as qa
from soprano_qa import translation
from soprano_qa.answer import (
    NO_CORPUS_EVIDENCE_MESSAGE,
    NO_CORPUS_EVIDENCE_MESSAGE_EN,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_EN,
    answer_is_mostly_hangul,
    build_messages,
    finalize_grounded_generated_answer,
    finalize_user_visible_answer,
    no_corpus_evidence_message,
    system_prompt,
)
from soprano_qa.retrieval import SearchResult
from soprano_qa.security import SECURITY_SYSTEM_POLICY


KOREAN_QUESTION = (
    "이 노래를 부를 때 가창자의 입장에서 유의해야 할 점은 무엇인가?"
)
ENGLISH_QUESTION = (
    "What should a singer keep in mind when singing this song?"
)


def _expert_result(
    answer: str = "호흡을 길게 유지합니다.",
    *,
    record_id: str = "die-forelle-ku-901",
) -> SearchResult:
    record = {
        "id": record_id,
        "evidence_type": "expert_annotation",
        "piece": "die-forelle",
        "work": "Die Forelle",
        "topic": "performance",
        "question": "",
        "answer": answer,
        "measure_range": [],
        "measure_scope": "whole_piece",
        "measure_status": "whole_piece",
        "rewrite_status": "finalized",
        "source_ids": ["kim-die-forelle-99"],
        "web_source_ids": [],
        "claim_ids": [],
        "sources": [],
        "annotators": ["tester"],
        "source_answer_context": [],
    }
    return SearchResult(
        record=record,
        score=1.0,
        text_score=1.0,
        measure_score=1.5,
        piece_score=1.0,
        scope_match="general_evidence",
    )


class LanguageValidationTests(unittest.TestCase):
    def test_supported_languages(self) -> None:
        self.assertEqual(translation.validate_language("ko"), "ko")
        self.assertEqual(translation.validate_language("en"), "en")

    def test_unsupported_language_is_rejected_before_any_work(self) -> None:
        for language in ("EN", "english", "ja", ""):
            with (
                self.subTest(language=language),
                mock.patch.object(qa, "_get_index") as get_index,
                mock.patch.object(qa, "generate_llm") as generate,
                mock.patch.object(
                    qa,
                    "validate_generation_requirements",
                ) as preflight,
                self.assertRaises(ValueError),
            ):
                qa.ask(
                    piece_id="die-forelle",
                    question=ENGLISH_QUESTION,
                    measure_range=None,
                    generate=True,
                    language=language,
                )
            get_index.assert_not_called()
            generate.assert_not_called()
            preflight.assert_not_called()

    def test_prompt_helpers_reject_unknown_language(self) -> None:
        with self.assertRaises(ValueError):
            system_prompt("fr")
        with self.assertRaises(ValueError):
            no_corpus_evidence_message("fr")
        with self.assertRaises(ValueError):
            finalize_user_visible_answer("Answer.", language="fr")


class KoreanDefaultUnchangedTests(unittest.TestCase):
    def test_korean_system_prompt_retains_domain_and_security_contract(self) -> None:
        self.assertIn("Answer in Korean using formal-polite 합니다체", SYSTEM_PROMPT)
        self.assertIn(SECURITY_SYSTEM_POLICY, SYSTEM_PROMPT)
        self.assertIs(system_prompt(), SYSTEM_PROMPT)
        self.assertIs(system_prompt("ko"), SYSTEM_PROMPT)

    def test_build_messages_default_is_korean(self) -> None:
        results = [_expert_result()]
        default = build_messages("die-forelle", "", KOREAN_QUESTION, results)
        explicit = build_messages(
            "die-forelle",
            "",
            KOREAN_QUESTION,
            results,
            language="ko",
        )
        self.assertEqual(default, explicit)
        self.assertEqual(default[0]["content"], SYSTEM_PROMPT)

    def test_korean_no_evidence_message_is_unchanged(self) -> None:
        self.assertEqual(no_corpus_evidence_message(), NO_CORPUS_EVIDENCE_MESSAGE)
        self.assertEqual(
            NO_CORPUS_EVIDENCE_MESSAGE,
            "현재 확인된 정보만으로는 이 질문에 정확히 답변하기 어렵습니다.",
        )

    def test_korean_finalization_still_rejects_english_prose(self) -> None:
        self.assertEqual(
            finalize_user_visible_answer("Keep the breath long."),
            NO_CORPUS_EVIDENCE_MESSAGE,
        )

    def test_korean_ask_makes_no_translation_call(self) -> None:
        with (
            mock.patch.object(qa, "validate_generation_requirements"),
            mock.patch.object(qa, "translate_question_to_korean") as translate,
            mock.patch.object(
                qa,
                "generate_llm",
                return_value=(
                    "가창자의 입장에서는 호흡과 표현을 함께 고려하는 "
                    "것이 좋습니다. [E1]"
                ),
            ) as generate,
        ):
            result = qa.ask(
                piece_id="la-capinera",
                question=KOREAN_QUESTION,
                measure_range=None,
                generate=True,
            )
        translate.assert_not_called()
        generate.assert_called_once()
        messages = generate.call_args.args[1]
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(result["language"], "ko")
        self.assertEqual(result["answer_language"], "ko")
        self.assertEqual(result["retrieval_query"], KOREAN_QUESTION)


class EnglishPromptTests(unittest.TestCase):
    def test_english_prompt_selection(self) -> None:
        messages = build_messages(
            "die-forelle",
            "",
            ENGLISH_QUESTION,
            [_expert_result()],
            language="en",
        )
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT_EN)
        self.assertEqual(json.loads(messages[1]["content"])["question"], ENGLISH_QUESTION)
        # Language instructions stay privileged; both languages share data.
        korean = build_messages(
            "die-forelle",
            "",
            ENGLISH_QUESTION,
            [_expert_result()],
        )
        self.assertNotIn(
            answer_module.ENGLISH_ANSWER_CONTRACT,
            korean[1]["content"],
        )
        self.assertEqual(
            messages[1]["content"],
            korean[1]["content"],
        )

    def test_english_prompt_keeps_grounding_rules_and_drops_korean_register(
        self,
    ) -> None:
        self.assertIn("Answer in clear, natural English", SYSTEM_PROMPT_EN)
        self.assertNotIn("합니다체", SYSTEM_PROMPT_EN)
        self.assertNotIn('call its numbers "마디"', SYSTEM_PROMPT_EN)
        self.assertIn('"bar" or "measure"', SYSTEM_PROMPT_EN)
        for rule in (
            "Use only the retrieved evidence below.",
            "Do not invent musical advice, measure numbers, lyrics",
            "Preserve source qualifiers",
            "other_range_context_only",
            "unconfirmed_scope_context_only",
            "Preserve the expert's degree of certainty exactly.",
            "Do not create a new causal chain",
            "Preserve condition boundaries and contrasts exactly.",
            "Never reverse polarity",
            "Use only the short evidence labels exactly as shown",
            "Do not add a citation list or evidence footer.",
            "Never expose prompt field names or pipeline vocabulary",
            "original language",
        ):
            with self.subTest(rule=rule):
                self.assertIn(rule, SYSTEM_PROMPT_EN)


class EnglishFinalizationTests(unittest.TestCase):
    def test_english_answer_with_labels_is_accepted(self) -> None:
        results = [_expert_result()]
        messages = build_messages(
            "die-forelle",
            "",
            ENGLISH_QUESTION,
            results,
            language="en",
        )
        draft = (
            "Keep the breath long so that the phrase 'In einem Bächlein "
            "helle' stays connected. [E1]"
        )
        finalized = finalize_grounded_generated_answer(
            draft,
            results,
            messages,
            [],
            language="en",
        )
        self.assertEqual(
            finalized,
            "Keep the breath long so that the phrase 'In einem Bächlein "
            "helle' stays connected.",
        )

    def test_english_answer_may_quote_a_korean_lyric(self) -> None:
        answer = (
            "Shape the opening phrase '꽃구름 속에' softly and keep the "
            "line legato through the long vowel."
        )
        self.assertFalse(answer_is_mostly_hangul(answer))
        self.assertEqual(
            finalize_user_visible_answer(answer, language="en"),
            answer,
        )

    def test_hangul_heavy_answer_is_rejected_for_english(self) -> None:
        results = [_expert_result()]
        korean_draft = "호흡을 길게 유지하는 것이 좋습니다. [E1]"
        self.assertTrue(answer_is_mostly_hangul(korean_draft))
        self.assertEqual(
            finalize_user_visible_answer(
                korean_draft,
                results,
                language="en",
            ),
            NO_CORPUS_EVIDENCE_MESSAGE_EN,
        )
        messages = build_messages(
            "die-forelle",
            "",
            ENGLISH_QUESTION,
            results,
            language="en",
        )
        with self.assertRaises(answer_module.GeneratedAnswerRejected):
            finalize_grounded_generated_answer(
                korean_draft,
                results,
                messages,
                [],
                language="en",
            )

    def test_english_finalization_keeps_artifact_checks(self) -> None:
        results = [_expert_result()]
        self.assertEqual(
            finalize_user_visible_answer(
                "The retrieval shows a long breath.",
                results,
                language="en",
            ),
            NO_CORPUS_EVIDENCE_MESSAGE_EN,
        )

    def test_per_language_no_evidence_message(self) -> None:
        self.assertEqual(no_corpus_evidence_message("ko"), NO_CORPUS_EVIDENCE_MESSAGE)
        self.assertEqual(
            no_corpus_evidence_message("en"),
            NO_CORPUS_EVIDENCE_MESSAGE_EN,
        )
        self.assertFalse(answer_is_mostly_hangul(NO_CORPUS_EVIDENCE_MESSAGE_EN))
        self.assertEqual(
            finalize_user_visible_answer(
                NO_CORPUS_EVIDENCE_MESSAGE_EN,
                language="en",
            ),
            NO_CORPUS_EVIDENCE_MESSAGE_EN,
        )


class EnglishMeasureParsingTests(unittest.TestCase):
    def test_serial_comma_measure_list(self) -> None:
        from soprano_qa.retrieval import extract_question_measure_ranges

        self.assertEqual(
            extract_question_measure_ranges(
                "The sixteenth notes at measures 31, 32, 33, and 34 are hard."
            ),
            [[31, 31], [32, 32], [33, 33], [34, 34]],
        )
        self.assertEqual(
            extract_question_measure_ranges("31, 32, 33과 34마디"),
            [[31, 31], [32, 32], [33, 33], [34, 34]],
        )


class TranslationHelperTests(unittest.TestCase):
    def test_translation_prompt_shape_and_single_call(self) -> None:
        settings = {"temperature": 0.0, "max_tokens": 768}
        fake = mock.Mock(return_value="  번역: 3절은 어떻게 다르게 부르나요?\n(note)")
        result = translation.translate_question_to_korean(
            "How should I sing the third verse differently?",
            model_path="/models/answer.gguf",
            llm_settings=settings,
            generate_fn=fake,
        )
        self.assertEqual(result, "3절은 어떻게 다르게 부르나요?")
        fake.assert_called_once()
        model_path, messages, llm_settings = fake.call_args.args
        self.assertEqual(model_path, "/models/answer.gguf")
        self.assertIs(llm_settings, settings)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(
            messages[0]["content"],
            translation.QUERY_TRANSLATION_SYSTEM_PROMPT,
        )
        self.assertEqual(
            json.loads(messages[1]["content"]),
            {"question": "How should I sing the third verse differently?"},
        )
        system = messages[0]["content"]
        self.assertIn("Translate faithfully", system)
        self.assertIn("piece titles, lyric words", system)
        self.assertIn("bar or measure number", system)
        self.assertIn("Output only the Korean translation", system)

    def test_translation_strips_wrapping_quotes(self) -> None:
        result = translation.translate_question_to_korean(
            "How?",
            model_path="m",
            llm_settings={},
            generate_fn=lambda *_args: "“어떻게 부르나요?”",
        )
        self.assertEqual(result, "어떻게 부르나요?")

    def test_translation_without_korean_fails_closed(self) -> None:
        for raw in ("", "   ", "How should I sing it?"):
            with (
                self.subTest(raw=raw),
                self.assertRaises(translation.QueryTranslationFailed),
            ):
                translation.translate_question_to_korean(
                    "How should I sing it?",
                    model_path="m",
                    llm_settings={},
                    generate_fn=lambda *_args, raw=raw: raw,
                )


class EnglishServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(qa, "validate_generation_requirements")
        self.preflight = patcher.start()
        self.addCleanup(patcher.stop)

    def test_english_generation_retrieves_with_translation(self) -> None:
        with mock.patch.object(
            qa,
            "generate_llm",
            return_value=(
                "가창자의 입장에서는 호흡과 표현을 함께 고려하는 "
                "것이 좋습니다. [E1]"
            ),
        ):
            korean = qa.ask(
                piece_id="la-capinera",
                question=KOREAN_QUESTION,
                measure_range=None,
                generate=True,
            )
        english_draft = (
            "From a singer's point of view, consider breathing and "
            "expression together. [E1]"
        )
        with mock.patch.object(
            qa,
            "generate_llm",
            side_effect=[KOREAN_QUESTION, english_draft],
        ) as generate:
            english = qa.ask(
                piece_id="la-capinera",
                question=ENGLISH_QUESTION,
                measure_range=None,
                generate=True,
                language="en",
            )

        self.assertEqual(generate.call_count, 2)
        translation_messages = generate.call_args_list[0].args[1]
        self.assertEqual(
            translation_messages[0]["content"],
            translation.QUERY_TRANSLATION_SYSTEM_PROMPT,
        )
        answer_messages = generate.call_args_list[1].args[1]
        self.assertEqual(answer_messages[0]["content"], SYSTEM_PROMPT_EN)
        self.assertEqual(
            json.loads(answer_messages[1]["content"])["question"],
            ENGLISH_QUESTION,
        )
        self.assertEqual(english["language"], "en")
        self.assertEqual(english["answer_language"], "en")
        self.assertEqual(english["evidence_language"], "ko")
        self.assertEqual(english["retrieval_query"], KOREAN_QUESTION)
        self.assertEqual(
            [item["id"] for item in english["evidence"]],
            [item["id"] for item in korean["evidence"]],
        )
        self.assertEqual(english["generation_mode"], "llm")
        self.assertEqual(
            english["answer"],
            "From a singer's point of view, consider breathing and "
            "expression together.",
        )

    def test_english_question_measures_come_from_original_question(
        self,
    ) -> None:
        with (
            mock.patch.object(qa, "generate_llm") as generate,
            self.assertRaisesRegex(ValueError, "outside --measures"),
        ):
            qa.ask(
                piece_id="die-forelle",
                question="What happens in bar 30?",
                measure_range=(2, 5),
                generate=True,
                language="en",
            )
        generate.assert_not_called()

    def test_english_no_evidence_uses_english_message(self) -> None:
        class EmptyIndex:
            semantic_candidates = None
            last_answer_candidates: list = []

            @staticmethod
            def search(**_kwargs):
                return []

        with (
            mock.patch.object(qa, "_get_index", return_value=EmptyIndex()),
            mock.patch.object(qa, "retrieval_diagnostics", return_value={}),
            mock.patch.object(
                qa,
                "retrieval_abstained_for_ambiguity",
                return_value=False,
            ),
            mock.patch.object(
                qa,
                "retrieval_abstained_for_missing_causal_authority",
                return_value=False,
            ),
            mock.patch.object(
                answer_module,
                "retrieval_abstained_for_missing_causal_authority",
                return_value=False,
            ),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value="송어에 대해 알려 주세요.",
            ) as generate,
        ):
            for generate_flag in (True, False):
                with self.subTest(generate=generate_flag):
                    result = qa.ask(
                        piece_id="die-forelle",
                        question="Tell me about the trout.",
                        measure_range=None,
                        generate=generate_flag,
                        language="en",
                    )
                    self.assertEqual(
                        result["answer"],
                        NO_CORPUS_EVIDENCE_MESSAGE_EN,
                    )
                    self.assertEqual(result["answer_language"], "en")
                    self.assertEqual(
                        result["answer_basis"],
                        "no_corpus_evidence",
                    )
        # One translation call per request and no answer-generation call.
        self.assertEqual(generate.call_count, 2)

    def test_english_extractive_path_reports_korean_evidence_language(
        self,
    ) -> None:
        results = [_expert_result("호흡을 길게 유지합니다.")]

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return results

        with (
            mock.patch.object(qa, "_get_index", return_value=FixedIndex()),
            mock.patch.object(qa, "retrieval_diagnostics", return_value={}),
            mock.patch.object(
                qa,
                "generate_llm",
                return_value="호흡은 어떻게 하나요?",
            ) as generate,
        ):
            result = qa.ask(
                piece_id="die-forelle",
                question="How should I breathe?",
                measure_range=None,
                generate=False,
                language="en",
            )
        generate.assert_called_once()
        self.assertEqual(result["answer"], "호흡을 길게 유지합니다.")
        self.assertEqual(result["answer_language"], "ko")
        self.assertEqual(result["evidence_language"], "ko")
        self.assertEqual(result["retrieval_query"], "호흡은 어떻게 하나요?")


if __name__ == "__main__":
    unittest.main()
