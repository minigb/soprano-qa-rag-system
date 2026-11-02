"""Explicit source facts may be rendered only within their supported scope."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import llm
from soprano_qa.security import SecurityViolation
from soprano_qa.source_comparison import opening_dynamics_comparison
from tests.test_security_integration import completion


SOFT = ("특히 반주부가 PP로 시작한다는 점을 고려한다면 노래 역시 지나치게 큰 소리로 "
        "시작하기보다는 부드러운 어택으로 시작하는 것이 적절해요. "
        "작은 음량 안에서도 호흡의 지지와 울림을 유지하는 것이 중요해요.")
PIANO = ("일반적인 가요나 팝 음악은 악보의 표기가 작곡가의 의도대로 되어 있지 않은 경우가 많다. "
         "이 악보에서는 시작 부분의 반주부가 piano (p)로 되어 있으므로, "
         "첫 소리를 크지 않게 시작하는 것이 좋아 보인다.")


def context(language="ko", first=SOFT, second=PIANO):
    return {"question": "처음 반주의 다이나믹 표기가 다른데 어떻게 이해해야 하나요?",
            "answer_language": language, "measure_range": "(none selected)",
            "Retrieved evidence": [
                {"citation_label": "E2", "answer": first, "evidence_role": "primary"},
                {"citation_label": "E4", "answer": second, "evidence_role": "primary"}]}


class SourceSynthesisTests(unittest.TestCase):
    def test_preserves_actor_values_suggestion_and_unknown_cause_in_both_languages(self):
        for language in ("ko", "en"):
            payload = context(language)
            original = deepcopy(payload)
            result = opening_dynamics_comparison(payload)
            self.assertEqual(payload, original)
            self.assertEqual(len(result.parts), 3)
            self.assertTrue(all(p["evidence"] == ["E2", "E4"] for p in result.parts))
            self.assertTrue(all(not p.get("source_quotation") for p in result.parts))
            self.assertIn("PP", result.parts[0]["text"])
            self.assertIn("p", result.parts[0]["text"])
            self.assertIn("반주" if language == "ko" else "accompaniment", result.parts[0]["text"])
            self.assertIn("확정할 수 없" if language == "ko" else "do not establish", result)
            self.assertIn("권합니다" if language == "ko" else "suggest", result)
            for absent in ("지지", "울림", "support", "resonance", "판본", "edition", "must"):
                self.assertNotIn(absent, result)

    def test_values_come_from_sources_not_questions_or_known_work_ids(self):
        one = "시작 반주가 mp로 되어 있습니다. 노래의 첫 소리는 크지 않게 시작하는 것이 좋습니다."
        two = "반주는 p로 시작합니다. 첫소리를 크지 않게 시작하는 것이 적절합니다."
        payload = context("en", one, two)
        payload["question"] = "Does the opening accompaniment really begin ff, and why do these markings differ?"
        result = opening_dynamics_comparison(payload)
        self.assertIn("as mp,", result)
        self.assertIn("as p.", result)
        self.assertNotIn("as ff", result)
        self.assertNotIn("PP", result)

    def test_ambiguous_negated_conditional_and_other_location_sources_keep_full_source_path(self):
        negative_sources = [
            "반주가 p로 시작하지 않습니다.",
            "시작 반주가 p로 되어 있지 않습니다.",
            "반주가 p로 시작한다면 조용할 것입니다.",
            "만약 반주가 p로 시작한다는 가정하에 설명합니다.",
            "반주가 p로 시작한다는 설명은 사실이 아닙니다.",
            "반주는 p 또는 mp로 시작합니다.",
            "후주의 반주는 p로 시작합니다.",
            "반주가 40마디에서 p로 시작합니다.",
            "노래가 p로 시작합니다. 반주는 셈여림 표시가 없습니다.",
            "p는 piano를 뜻합니다. 반주의 시작을 살펴봅니다.",
            "반주가 p로 시작합니다. 반주가 mp로 시작하는 부분도 있습니다.",
            "반주가 p로 시작합니다. 표기의 차이는 개정판 때문입니다.",
        ]
        for source in negative_sources:
            with self.subTest(source=source):
                self.assertIsNone(opening_dynamics_comparison(context(second=source +
                    " 첫소리를 크지 않게 시작하는 것이 좋습니다.")))

    def test_common_advice_requires_explicit_support_in_each_source(self):
        for advice in ("", "반주를 크지 않게 시작하는 것이 좋습니다.",
                       "노래의 첫소리를 크지 않게 시작하는 것은 적절하지 않습니다.",
                       "첫소리는 크게 시작하는 것이 좋습니다."):
            with self.subTest(advice=advice):
                self.assertIsNone(opening_dynamics_comparison(context(second="반주는 p로 시작합니다. " + advice)))

    def test_same_values_single_source_local_and_unrelated_questions_keep_full_source_path(self):
        self.assertIsNone(opening_dynamics_comparison(context(second=SOFT.replace("PP", "pp"))))
        changes = [{"Retrieved evidence": context()["Retrieved evidence"][:1]},
                   {"measure_range": [7, 8]}, {"question": "후반의 호흡은 어떻게 하나요?"},
                   {"answer_language": "fr"}]
        for change in changes:
            with self.subTest(change=change):
                self.assertIsNone(opening_dynamics_comparison({**context(), **change}))
        payload = context()
        payload["Retrieved evidence"][1]["evidence_role"] = "context"
        self.assertIsNone(opening_dynamics_comparison(payload))

    def test_comparison_never_bypasses_admission_or_output_review(self):
        for language in ("ko", "en"):
            for denied in (None, "admission", "output"):
                with self.subTest(language=language, denied=denied):
                    payload = context(language)
                    backend = Mock()
                    verdicts = ['{"task":"instruction_attack"}'] if denied == "admission" else [
                        '{"task":"music_question"}', '{"labels":["E2","E4"]}',
                        json.dumps({"allow": denied != "output"})]
                    backend.create_chat_completion.side_effect = [completion(x) for x in verdicts]
                    with patch.object(llm, "load_llama", return_value=backend), \
                         patch.object(llm, "translate_source_parts") as translate:
                        def run():
                            return llm.generate("fake", [{"role":"system", "content":"Music policy."},
                                                       {"role":"user", "content":json.dumps(payload)}],
                                                {"source_preservation":"verbatim_or_translation",
                                                 "evidence_translation":"reviewed_catalog"}, security_context=payload)
                        if denied:
                            with self.assertRaises(SecurityViolation) as error:
                                run()
                            self.assertEqual(error.exception.reason, f"security_{denied}_rejected")
                        else:
                            result = run()
                            self.assertEqual(result.generation_mode, "llm_guarded_source_comparison")
                            self.assertEqual(result.answer_basis, "retrieved_evidence")
                            self.assertIn("[E2, E4]", result)
                    translate.assert_not_called()
                    self.assertEqual(backend.create_chat_completion.call_count, len(verdicts))
                    if denied != "admission":
                        checked = json.loads(backend.create_chat_completion.call_args.kwargs["messages"][1]["content"])
                        self.assertEqual(checked["question"], payload["question"])
                        self.assertTrue(checked["source_synthesis"])
                        self.assertEqual(checked["composition"][0]["evidence"], ["E2", "E4"])


if __name__ == "__main__":
    unittest.main()
