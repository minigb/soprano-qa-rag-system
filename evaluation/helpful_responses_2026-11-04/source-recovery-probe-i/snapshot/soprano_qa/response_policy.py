"""Useful music responses with quality warnings and explicit service guidance."""
from __future__ import annotations

import json
from soprano_qa.security import SecurityViolation, screen_generated_answer, instruction_guard

GENERAL_FALLBACK_INSTRUCTIONS = """
You are a singing and music assistant. Answer the actual question using accurate
general music knowledge. No reliable source for this particular passage is supplied.
Explain what can be said generally, with a brief limit where needed. For an unknown
passage, describe conditional choices and when each fits, rather than prescribing
a change as the default. Respect the musical connection or constraint in the question.
Do not invent score markings, an expert's advice or a composer's intention.
Give useful connected prose, not just a statement that information is missing.
"""
GENERAL_FALLBACK_FORMAT = """
Return one JSON object with a sentences array containing one item. Its kind is
background, evidence is an empty array, and text is your actual answer to the
question. Write the answer itself in text, never a template or format description.
"""

class ResponseText(str):
    pass


def domain_guidance(language):
    return ("저는 성악과 음악을 돕는 도우미입니다. 음악과 관계없는 요청이나 역할을 바꾸는 지시는 수행하지 않지만, 발성·호흡·가사 발음·악보 해석에 관한 질문은 도와드릴 수 있습니다."
            if language == "ko" else
            "I help with singing and music. I cannot carry out unrelated tasks or instructions to change my role, but I can help with vocal technique, breathing, diction, and musical interpretation.")


def quality_notice(language):
    return ("다만, 일부 세부 내용은 추가 확인이 필요하므로 악보와 실제 연주 상황에 맞춰 참고해 주세요."
            if language == "ko" else
            "Some details still need confirmation; consider them alongside the score and your performance context.")


def failure_guidance(language):
    return ("답변을 생성하는 과정에 문제가 발생했습니다. 질문을 다시 보내주시면 이어서 도와드리겠습니다. 악보의 특정 부분에 관한 질문이라면 해당 마디나 기호도 함께 알려 주세요."
            if language == "ko" else
            "A problem occurred while generating the response. Please send the question again so I can help. For a specific passage, include its measure or notation.")


def is_scope_rejection(reason):
    return reason in {'security_admission_rejected', 'prompt_injection_detected', 'invalid_question', 'question_too_long'}


def compose_music_answer(model_path, messages, llm_settings, results, measure_ranges,
                         *, language, generate_fn, finalizer_fn=None, general_only=False):
    # Local imports avoid answer.py -> llm.py -> response_policy import cycles.
    from soprano_qa import answer as validation
    from soprano_qa.structured_answer import GENERAL_MUSIC_SYSTEM_PROMPT, GENERAL_MUSIC_FORMAT_POLICY
    from soprano_qa.security import SECURITY_SYSTEM_POLICY, serialize_untrusted

    def usable(text):
        plain = str(text).replace('General explanation:', '').replace('일반적인 음악 설명입니다.', '').strip()
        if plain in {'Your answer in the requested language.', 'Your answer.',
                     '자료에 근거한 답변', '필요할 때 보충하는 일반적인 설명'}:
            return False
        return (bool(text.strip()) and text != validation.no_corpus_evidence_message(language)
                and not validation.is_grounded_insufficiency_answer(text))

    def warning_finalize(draft, evidence):
        text = validation.finalize_user_visible_answer(draft, evidence, language=language)
        if not usable(text):
            raise validation.GeneratedAnswerRejected("No usable prose after presentation checks")
        return text + "\n\n" + quality_notice(language)

    def run(bundle, context_results, *, fallback=False):
        payload = json.loads(bundle[1]['content'])
        finalize = lambda draft: (finalizer_fn or validation.finalize_grounded_generated_answer)(
            draft, context_results, bundle, measure_ranges, language=language)
        raw = generate_fn(model_path, bundle, llm_settings, security_context=payload,
                          finalizer=finalize,
                          warning_finalizer=lambda draft: warning_finalize(draft, context_results),
                          **({'general_answer_fallback': True} if fallback else {}))
        text = getattr(raw, 'validated_text', None)
        warning = getattr(raw, 'generation_validation_warning', None)
        if text is None:
            try:
                text = finalize(raw)
            except validation.GeneratedAnswerRejected:
                if not usable(str(raw)):
                    raise
                screen_generated_answer(raw, payload)
                instruction_guard(model_path, llm_settings, {**payload, 'draft': str(raw)}, generate_fn=generate_fn)
                text = warning_finalize(raw, context_results)
                warning = 'answer_validation_warning'
        if not usable(text):
            raise validation.GeneratedAnswerRejected('No useful music answer')
        value = ResponseText(text)
        value.validated_text = text
        value.generation_mode = ('llm_general_fallback' if fallback else
                                 'llm_with_warning' if warning else getattr(raw, 'generation_mode', 'llm'))
        value.answer_basis = ('general_music_knowledge' if fallback else
                              getattr(raw, 'answer_basis', 'retrieved_evidence' if context_results else
                                      'score_context' if payload.get('Score context') else 'general_music_knowledge'))
        value.generation_validation_warning = warning or ('general_knowledge_fallback' if fallback else None)
        return value

    try:
        return run(messages, results, fallback=general_only)
    except (validation.GeneratedAnswerRejected, SecurityViolation) as exc:
        if isinstance(exc, SecurityViolation) and exc.reason in {
            'security_admission_rejected', 'invalid_question', 'question_too_long', 'prompt_injection_detected',
        }:
            raise
        # A source or draft failure does not mean the music question has no
        # useful answer. Retry once without source content, using the same 14B.
        source = json.loads(messages[1]['content'])
        payload = {key: source[key] for key in ('question', 'piece', 'measure_range', 'answer_language') if key in source}
        payload['Retrieved evidence'] = []
        prompt = GENERAL_MUSIC_FORMAT_POLICY + '\n' + GENERAL_MUSIC_SYSTEM_PROMPT + '\n' + SECURITY_SYSTEM_POLICY + GENERAL_FALLBACK_INSTRUCTIONS
        return run([{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': serialize_untrusted(payload)}], [], fallback=True)
