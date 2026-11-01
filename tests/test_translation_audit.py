import unittest

from evaluation.audit_translated_sources import HEADER, digest, inspect


class TranslationAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = {"id": "s1", "text": "쉼표에서 숨을 쉴 수 있다."}
        self.catalog = {digest(self.source["text"]): "You may breathe at a rest."}

    def check_answer(self, answer, source=None):
        return inspect({"answer": HEADER + answer,
                        "evidence": [source or self.source]}, self.catalog)

    def test_complete_exact_source_is_verified(self):
        self.assertTrue(self.check_answer("You may breathe at a rest.")["verified"])

    def test_material_term_change_is_not_text_identity(self):
        self.assertFalse(self.check_answer("You may breathe at a comma.")["verified"])

    def test_source_change_cannot_borrow_old_translation(self):
        changed = {**self.source, "text": "쉼표에서 반드시 숨을 쉰다."}
        self.assertFalse(self.check_answer("You may breathe at a rest.", changed)["verified"])

    def test_unsupported_addition_is_not_verified(self):
        self.assertFalse(self.check_answer("You may breathe at a rest. Always stop.")["verified"])

    def test_marked_background_is_separate_not_audited_as_source(self):
        result = self.check_answer("You may breathe at a rest.\n\nGeneral explanation: Context matters.")
        self.assertTrue(result["verified"])
        self.assertTrue(result["separate_background"])


if __name__ == "__main__":
    unittest.main()
