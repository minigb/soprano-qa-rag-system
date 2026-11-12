import unittest
from evaluation.audit_source_quotations import inspect, HEADER, BACKGROUND


class QuotationAuditTests(unittest.TestCase):
    def test_full_source_identity_allows_only_outer_display_whitespace(self):
        source = "두 모음 모두 e이다.\n조건이 맞을 때만 연음한다.\n"
        response = {"answer": HEADER + source.rstrip(), "evidence": [{"id": "a", "text": source}]}
        self.assertTrue(inspect(response)["verified"])
        for replacement in (source.replace("e이다", "i이다"), source.replace("때만", "때"),
                            source.replace("\n조건", " 조건")):
            response["answer"] = HEADER + replacement.rstrip()
            self.assertFalse(inspect(response)["verified"])

    def test_background_is_identified_without_claiming_it_is_verified(self):
        response = {"answer": HEADER + "첫 조건.\n\n둘째 조건.\n\n" + BACKGROUND + "별도 설명.",
                    "evidence": [{"id": "a", "text": "첫 조건."}, {"id": "b", "text": "둘째 조건."}]}
        result = inspect(response)
        self.assertEqual(result["exact_source_ids"], ["a", "b"])
        self.assertTrue(result["separate_background"])


if __name__ == "__main__":
    unittest.main()
