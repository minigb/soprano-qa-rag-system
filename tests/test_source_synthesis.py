"""Multi-source prose keeps citations and requires ordinary output approval."""
import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import llm
from soprano_qa.security import SecurityViolation
from soprano_qa.source_comparison import render_comparison, sources_disagree, _classify_relationship
from tests.test_security_integration import completion


def contrast(feature, accounts):
    return {"feature": feature, "accounts": [{"evidence": label, "support": support, "value": value}
                                              for label, support, value in accounts]}


class SourceSynthesisTests(unittest.TestCase):
    def setUp(self):
        _classify_relationship.cache_clear()

    def test_comparison_values_need_real_excerpts_and_shared_advice_names_all_sources(self):
        payload = {"answer_language":"ko", "Retrieved evidence":[
            {"citation_label":"E2","answer":"첫 음을 작게 시작합니다."},
            {"citation_label":"E4","answer":"노래의 시작을 작게 합니다."}]}
        draft = {"contrast":contrast("설명 대상", [("E2","첫 음","첫 음"),("E4","노래의 시작","노래의 시작")]),
                 "shared_guidance":{"evidence":["E2","E4"], "text":"두 자료 모두 작게 시작하도록 권합니다"}, "cause":None}
        result = render_comparison(json.dumps(draft),payload)
        self.assertIn("첫 번째 자료에는",result)
        self.assertIn("두 번째 자료에는",result)
        self.assertTrue(result.parts[-1]["text"].endswith("."))
        for label, support in (("E4","호흡을 지지합니다."),("E9","노래의 시작"),("E4","")):
            invalid = contrast("설명 대상", [("E2","첫 음","첫 음"),(label,support,"노래의 시작")])
            with self.subTest(label=label,support=support), self.assertRaises(SecurityViolation):
                render_comparison(json.dumps({**draft,"contrast":invalid}),payload)
        with self.assertRaises(SecurityViolation):
            render_comparison(json.dumps({**draft,"shared_guidance":{"evidence":["E2"],"text":"Both agree."}}),payload)

    def test_missing_cause_stays_unknown_and_fabricated_quote_is_rejected(self):
        payload = {"answer_language": "en", "Retrieved evidence": [
            {"citation_label": "E1", "answer": "The opening is quiet."},
            {"citation_label": "E2", "answer": "The opening is loud."}]}
        draft = {"contrast": contrast("opening", [("E1","quiet","quiet"),("E2","loud","loud")]),
                 "shared_guidance": None, "cause": None}
        rendered = render_comparison(json.dumps(draft), payload)
        self.assertIn("do not establish why", rendered)
        self.assertEqual(rendered.parts[1]["evidence"], ["E1", "E2"])
        for bad in ({"evidence":"E1", "quote":"The edition changed."},
                    {"evidence":"E99", "quote":"The opening is quiet."},
                    {"evidence":"E1", "quote":""}):
            with self.subTest(bad=bad), self.assertRaises(SecurityViolation):
                render_comparison(json.dumps({**draft, "cause": bad}), payload)
        with self.assertRaises(SecurityViolation):
            render_comparison(json.dumps({**draft, "background": "An invented explanation."}), payload)

    def test_english_comparison_checks_each_prose_item_and_preserves_lyric_values(self):
        payload = {"answer_language":"en", "Retrieved evidence":[
            {"citation_label":"E1","answer":"가사에는 아리랑이라고 적혀 있습니다."},
            {"citation_label":"E2","answer":"가사에는 도라지라고 적혀 있습니다."}]}
        draft = {"contrast":contrast("lyric text",[("E1","아리랑","아리랑"),("E2","도라지","도라지")]),
                 "shared_guidance":None,"cause":None}
        self.assertIn("아리랑",render_comparison(json.dumps(draft),payload))
        with self.assertRaises(SecurityViolation) as error:
            render_comparison(json.dumps({**draft,"shared_guidance":{
                "evidence":["E1","E2"],"text":"영어 문단 사이에 한국어 설명을 숨기면 안 됩니다."}}),payload)
        self.assertEqual(error.exception.reason,"comparison_language_mismatch")

    def test_complementary_sources_and_single_source_do_not_get_comparison(self):
        payload = {"question": "How should I sing?", "Retrieved evidence": [
            {"citation_label":"E1", "answer":"Without a microphone, project the voice."},
            {"citation_label":"E2", "answer":"With a microphone, use a lighter sound."}]}
        generate = Mock(return_value='{"relationship":"complementary"}')
        self.assertFalse(sources_disagree("fake", {}, payload, generate_fn=generate))
        generate.assert_called_once()
        sent = json.loads(generate.call_args.args[1][1]["content"])
        self.assertNotIn("question", sent)
        generate.reset_mock()
        self.assertFalse(sources_disagree("fake", {}, {**payload,"Retrieved evidence":payload["Retrieved evidence"][:1]}, generate_fn=generate))
        generate.assert_not_called()
        for invalid in ('{"relationship":"disagreement","allow":true}', '{"relationship":true}',
                        '{"relationship":"disagreement","relationship":"complementary"}'):
            generate = Mock(return_value=invalid)
            with self.subTest(invalid=invalid), self.assertRaises(SecurityViolation):
                sources_disagree("fake", {}, payload, generate_fn=generate)

    def test_routing_cache_is_bound_to_sources_model_and_policy_not_a_security_verdict(self):
        sources = [{"citation_label":"E1","answer":"Quiet opening."},
                   {"citation_label":"E2","answer":"Loud opening."}]
        generate = Mock(return_value='{"relationship":"disagreement"}')
        payload = {"question":"First question", "Retrieved evidence":sources}
        self.assertTrue(sources_disagree("model", {}, payload, generate_fn=generate))
        self.assertTrue(sources_disagree("model", {}, {**payload,"question":"Paraphrase"}, generate_fn=generate))
        self.assertEqual(generate.call_count,1)
        sources_disagree("other-model", {}, payload, generate_fn=generate)
        sources_disagree("model", {"n_ctx":4096}, payload, generate_fn=generate)
        sources_disagree("model", {}, {**payload,"Retrieved evidence":[sources[0],{**sources[1],"answer":"Different account."}]},generate_fn=generate)
        self.assertEqual(generate.call_count,4)
        broken = Mock(return_value='{}')
        for _ in range(2):
            with self.assertRaises(SecurityViolation):
                sources_disagree("model", {}, payload, generate_fn=broken)
        self.assertEqual(broken.call_count,2)

    def test_comparison_is_cited_prose_and_never_a_catalog_translation(self):
        for language in ("ko", "en"):
            for allow in (True, False):
                with self.subTest(language=language, allow=allow):
                    _classify_relationship.cache_clear()
                    payload = {"question": "Compare the source recommendations.", "answer_language": language,
                               "Retrieved evidence": [
                                   {"citation_label": "E1", "answer": "First source advice."},
                                   {"citation_label": "E2", "answer": "Second source advice."},
                                   {"citation_label": None, "answer": "Context only."}]}
                    draft = json.dumps({"contrast": contrast("advice", [("E1","First source advice.","First source advice."),
                                                                 ("E2","Second source advice.","Second source advice.")]),
                                        "shared_guidance": None, "cause": None})
                    attempt = [draft, json.dumps({"allow": allow})]
                    backend = Mock()
                    backend.create_chat_completion.side_effect = [completion(x) for x in [
                        '{"task":"music_question"}', '{"labels":["E1","E2"]}', '{"relationship":"disagreement"}',
                        *attempt, *([] if allow else attempt)]]
                    with patch.object(llm, "load_llama", return_value=backend), \
                         patch.object(llm, "translate_source_parts") as translate:
                        def run():
                            return llm.generate("fake", [{"role":"system", "content":"Music policy."},
                                                       {"role":"user", "content":json.dumps(payload)}],
                                                {"source_preservation":"verbatim_or_translation",
                                                 "evidence_translation":"reviewed_catalog"}, security_context=payload)
                        if allow:
                            result = run()
                            self.assertIn("[E1, E2]", result)
                            self.assertFalse(any(p.get("source_quotation") for p in result.parts))
                        else:
                            with self.assertRaises(SecurityViolation) as error:
                                run()
                            self.assertEqual(error.exception.reason, "security_output_rejected")
                    translate.assert_not_called()
                    calls = backend.create_chat_completion.call_args_list
                    composed = json.loads(calls[3].kwargs["messages"][1]["content"])
                    self.assertNotIn("question", composed)
                    self.assertNotIn("source_preservation", composed)
                    self.assertEqual([r["citation_label"] for r in composed["Retrieved evidence"]], ["E1", "E2"])
                    checked = json.loads(calls[4].kwargs["messages"][1]["content"])
                    self.assertIn("First source advice.", checked["draft"])
                    self.assertEqual(checked["question"], payload["question"])
                    self.assertTrue(checked["source_synthesis"])
                    self.assertNotIn("response_format", calls[3].kwargs)
                    self.assertEqual(calls[3].kwargs["max_tokens"], 2560)
                    self.assertEqual(checked["composition"][0]["evidence"], ["E1", "E2"])


if __name__ == "__main__":
    unittest.main()
