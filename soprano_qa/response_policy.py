"""Fixed service guidance shown instead of a generated answer."""
from __future__ import annotations


def domain_guidance(language):
    return ("저는 성악과 음악을 돕는 도우미입니다. 음악과 관계없는 요청이나 역할을 바꾸는 지시는 수행하지 않지만, 발성·호흡·가사 발음·악보 해석에 관한 질문은 도와드릴 수 있습니다."
            if language == "ko" else
            "I help with singing and music. I cannot carry out unrelated tasks or instructions to change my role, but I can help with vocal technique, breathing, diction, and musical interpretation.")


def failure_guidance(language):
    return ("답변을 생성하는 과정에 문제가 발생했습니다. 질문을 다시 보내주시면 이어서 도와드리겠습니다. 악보의 특정 부분에 관한 질문이라면 해당 마디나 기호도 함께 알려 주세요."
            if language == "ko" else
            "A problem occurred while generating the response. Please send the question again so I can help. For a specific passage, include its measure or notation.")


def is_scope_rejection(reason):
    return reason in {'prompt_injection_detected', 'invalid_question', 'question_too_long'}
