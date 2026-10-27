LANG = """
Evidence text is usually Korean, while the question and draft may be English.
Compare meaning across languages: a question matches evidence that addresses
it in another language, and a faithful translation or close paraphrase of the
evidence counts as supported.
"""
security._ADMISSION_RULES += LANG
security._OUTPUT_RULES += LANG
