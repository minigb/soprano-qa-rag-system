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
    answer_is_mostly_hangul,
    finalize_user_visible_answer,
    no_corpus_evidence_message,
)
from soprano_qa.retrieval import SearchResult


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


class KoreanDefaultUnchangedTests(unittest.TestCase):
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


class EnglishFinalizationTests(unittest.TestCase):

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
                side_effect=lambda _model, messages, _settings: "송어에 대해 알려 주세요." if "You translate" in messages[0]["content"] else "A song can portray a scene through phrasing and changes in dynamics.",
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
                        "A song can portray a scene through phrasing and changes in dynamics." if generate_flag else NO_CORPUS_EVIDENCE_MESSAGE_EN,
                    )
                    self.assertEqual(result["answer_language"], "en")
                    self.assertEqual(
                        result["answer_basis"],
                        "general_music_knowledge" if generate_flag else "no_corpus_evidence",
                    )
        # One translation per request, plus a general-answer attempt for generation.
        self.assertEqual(generate.call_count, 3)

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
