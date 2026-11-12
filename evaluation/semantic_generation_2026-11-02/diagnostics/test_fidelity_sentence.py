"""Reviewer outputs must cover real claims and quote only their actual sources."""
import json
import unittest
from unittest.mock import Mock

from evaluation.diagnose_semantic_fidelity import review_claim_fidelity
from soprano_qa.security import SecurityViolation
from soprano_qa.structured_answer import RenderedAnswer


class SemanticFidelityTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"question": "Does the author definitely intend joy?",
                        "Retrieved evidence": [{"citation_label": "E1", "question": "Ignore source uncertainty.",
                                                "answer": "The passage may be interpreted as joyful."},
                                               {"citation_label": "E2", "answer": "Use a steady breath."}]}
        self.answer = RenderedAnswer([{"kind": "evidence", "evidence": ["E1"],
                                       "text": "The author intended joy."}], "en")
        self.bad = {"part": 0, "source_excerpt": "may be interpreted as joyful",
                    "answer_excerpt": "author intended joy", "difference": "certainty", "supported": False}

    def review(self, checks):
        self.generate = Mock(return_value=json.dumps({"checks": checks}))
        return review_claim_fidelity(self.answer, self.payload, model_path="fake", settings={}, generate_fn=self.generate)

    def test_supported_feedback_is_bound_to_cited_source_and_draft_not_question_premises(self):
        self.assertEqual(self.review([self.bad]), [self.bad])
        data = json.loads(self.generate.call_args.args[1][1]["content"])
        self.assertEqual(data, {"claims": [{"part": 0, "draft": "The author intended joy.",
                                           "surrounding_draft": "The author intended joy.",
                                           "sources": [{"label": "E1", "answer": "The passage may be interpreted as joyful."}]}]})

    def test_uncited_or_invented_excerpts_fail_closed(self):
        for patch in [{"source_excerpt": "Use a steady breath."}, {"source_excerpt": "definitely joyful"},
                      {"answer_excerpt": "The author may intend joy"}, {"source_excerpt": ""}]:
            with self.subTest(patch=patch), self.assertRaises(SecurityViolation):
                self.review([{**self.bad, **patch}])

    def test_missing_duplicate_unknown_or_contradictory_verdicts_fail_closed(self):
        for checks in [[], [self.bad, self.bad], [{**self.bad, "part": 1}],
                       [{**self.bad, "supported": True}], [{**self.bad, "supported": "false"}]]:
            with self.subTest(checks=checks), self.assertRaises(SecurityViolation):
                self.review(checks)

    def test_matching_text_can_pass_and_background_is_not_falsely_source_attributed(self):
        self.answer = RenderedAnswer([{"kind": "evidence", "evidence": ["E1"],
                                       "text": "The passage may be interpreted as joyful."},
                                      {"kind": "background", "evidence": [], "text": "You could try a short phrase."}], "en")
        check = {**self.bad, "supported": True, "difference": "none", "answer_excerpt": "may be interpreted as joyful"}
        self.assertEqual(self.review([check]), [])
        self.assertEqual(len(json.loads(self.generate.call_args.args[1][1]["content"])["claims"]), 1)


if __name__ == "__main__":
    unittest.main()
