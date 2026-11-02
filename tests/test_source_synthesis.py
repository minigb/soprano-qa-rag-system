"""Multi-source prose keeps citations and requires ordinary output approval."""
import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import llm
from soprano_qa.security import SecurityViolation
from soprano_qa.source_comparison import render_comparison, sources_disagree
from tests.test_security_integration import completion


class SourceSynthesisTests(unittest.TestCase):
    def test_missing_cause_stays_unknown_and_fabricated_quote_is_rejected(self):
        payload = {"answer_language": "en", "Retrieved evidence": [
            {"citation_label": "E1", "answer": "The opening is quiet."},
            {"citation_label": "E2", "answer": "The opening is loud."}]}
        draft = {"contrast": {"evidence": ["E1", "E2"], "text": "One account says quiet, the other loud."},
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

    def test_complementary_sources_and_single_source_do_not_get_comparison(self):
        payload = {"question": "How should I sing?", "Retrieved evidence": [
            {"citation_label":"E1", "answer":"Without a microphone, project the voice."},
            {"citation_label":"E2", "answer":"With a microphone, use a lighter sound."}]}
        generate = Mock(return_value='{"relationship":"complementary"}')
        self.assertFalse(sources_disagree("fake", {}, payload, generate_fn=generate))
        generate.assert_called_once()
        generate.reset_mock()
        self.assertFalse(sources_disagree("fake", {}, {**payload,"Retrieved evidence":payload["Retrieved evidence"][:1]}, generate_fn=generate))
        generate.assert_not_called()
        for invalid in ('{"relationship":"disagreement","allow":true}', '{"relationship":true}',
                        '{"relationship":"disagreement","relationship":"complementary"}'):
            generate.return_value = invalid
            with self.subTest(invalid=invalid), self.assertRaises(SecurityViolation):
                sources_disagree("fake", {}, payload, generate_fn=generate)

    def test_comparison_is_cited_prose_and_never_a_catalog_translation(self):
        for language in ("ko", "en"):
            for allow in (True, False):
                with self.subTest(language=language, allow=allow):
                    payload = {"question": "Compare the source recommendations.", "answer_language": language,
                               "Retrieved evidence": [
                                   {"citation_label": "E1", "answer": "First source advice."},
                                   {"citation_label": "E2", "answer": "Second source advice."},
                                   {"citation_label": None, "answer": "Context only."}]}
                    draft = json.dumps({"contrast": {"evidence": ["E1", "E2"], "text": "The sources disagree."},
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
                    self.assertTrue(composed["source_synthesis"])
                    self.assertNotIn("source_preservation", composed)
                    self.assertEqual([r["citation_label"] for r in composed["Retrieved evidence"]], ["E1", "E2"])
                    checked = json.loads(calls[4].kwargs["messages"][1]["content"])
                    self.assertIn("sources disagree", checked["draft"])
                    self.assertEqual(checked["composition"][0]["evidence"], ["E1", "E2"])


if __name__ == "__main__":
    unittest.main()
