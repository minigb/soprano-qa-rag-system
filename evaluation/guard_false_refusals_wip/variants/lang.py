LANG = """
Evidence text is usually Korean, while the question and draft may be English.
Compare meaning across languages: a question matches evidence that addresses
it in another language, and a faithful translation or close paraphrase of the
evidence counts as supported. For an English question, question_korean holds
the application's Korean translation of it; use it only for this comparison.
"""
security._ADMISSION_RULES += LANG
security._OUTPUT_RULES += LANG
