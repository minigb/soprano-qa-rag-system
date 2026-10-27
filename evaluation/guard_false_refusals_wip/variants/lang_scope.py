LANG = """
Evidence text is usually Korean, while the question and draft may be English.
Compare meaning across languages: a question matches evidence that addresses
it in another language, and a faithful translation or close paraphrase of the
evidence counts as supported. For an English question, question_korean holds
the application's Korean translation of it; use it only for this comparison.
"""
security._ADMISSION_RULES += LANG
security._OUTPUT_RULES += LANG
SCOPE = """
Questions about singing technique, practice, or performance conventions are in
scope while a piece is selected, even when they do not name the piece and the
evidence says its advice also applies to other works.
"""
security._ADMISSION_RULES += SCOPE
