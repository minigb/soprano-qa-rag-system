"""Multi-source prose keeps citations and requires ordinary output approval."""
import json
import unittest
from unittest.mock import Mock, patch

from soprano_qa import llm
from soprano_qa.security import SecurityViolation
from tests.test_security_integration import completion


class SourceSynthesisTests(unittest.TestCase):
    def test_comparison_is_cited_prose_and_never_a_catalog_translation(self):
        for language in ("ko", "en"):
            for allow in (True, False):
                with self.subTest(language=language, allow=allow):
                    payload = {"question": "Compare the source recommendations.", "answer_language": language,
                               "Retrieved evidence": [
                                   {"citation_label": "E1", "answer": "First source advice."},
                                   {"citation_label": "E2", "answer": "Second source advice."},
                                   {"citation_label": None, "answer": "Context only."}]}
                    draft = json.dumps({"sentences": [{"kind": "evidence", "evidence": ["E1", "E2"],
                                                       "text": "The sources disagree; their reason is not established."}]})
                    attempt = [draft, json.dumps({"allow": allow})]
                    backend = Mock()
                    backend.create_chat_completion.side_effect = [completion(x) for x in [
                        '{"task":"music_question"}', '{"labels":["E1","E2"]}',
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
                    composed = json.loads(calls[2].kwargs["messages"][1]["content"])
                    self.assertTrue(composed["source_synthesis"])
                    self.assertNotIn("source_preservation", composed)
                    self.assertEqual([r["citation_label"] for r in composed["Retrieved evidence"]], ["E1", "E2"])
                    checked = json.loads(calls[3].kwargs["messages"][1]["content"])
                    self.assertIn("sources disagree", checked["draft"])
                    self.assertEqual(checked["composition"][0]["evidence"], ["E1", "E2"])


if __name__ == "__main__":
    unittest.main()
