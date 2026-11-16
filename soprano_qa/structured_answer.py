"""Source-meaning hints for the writing prompt.

These helpers derive constraints and disagreements from the retrieved passages
themselves (conditions, negations, alternatives) and add them to the prompt.
They neither choose passages nor inspect an answer.
"""
from __future__ import annotations

import re
from typing import Any


def source_meaning_constraints(text: str) -> list[dict[str, str]]:
    """Surface literal source operators without adding musical facts."""
    hints = []
    for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
        if not sentence.strip():
            continue
        if re.search(r"것(?:으)?로\s*볼\s*수|보는\s*것이\s*일반적|(?:could|may)\s+be\s+(?:interpreted|seen)", sentence, re.I):
            hints.append({"relation": "interpretation_not_established_intention", "source_excerpt": sentence})
        contrast = re.search(r"보다(?:는)?\s+(?P<preferred>.+?(?:좋|적절|낫|권장)[^.!?。！？]*)", sentence)
        if contrast:
            hints.append({"relation": "preferred_alternative", "source_excerpt": sentence,
                          "preferred_clause": contrast.group("preferred")})
        if re.search(r"거나|또는|\bor\b", sentence, re.I):
            hint = {"relation": "alternatives_not_joint_requirements", "source_excerpt": sentence}
            condition = re.search(r"(?P<a>.+?)거나\s*(?P<b>.+?(?:경우|때))(?:에는|는|에)", sentence)
            if condition:
                hint.update(logic="OR", condition_fragment_1=condition["a"],
                            condition_fragment_2=condition["b"])
            hints.append(hint)
    return hints


def source_comparison_policy(language: str) -> str:
    """Generate a question-focused comparison from conflicting source facts."""
    return """서로 다른 연주 자료를 비교해 성악 질문에 간결하고 자연스럽게 답합니다.
공통으로 뒷받침되는 가창 방향을 먼저 설명하고, 표기에 대한 서로 다른 설명을 구분합니다.
source_disagreements의 accounts는 실제 자료에서 추출한 시작 반주의 표기입니다.
반주의 표기를 성악부에 적힌 기호로 바꾸지 않습니다. 어느 설명이 실제 악보와 일치하는지는
이 자료만으로 결정하지 않습니다. 차이의 원인을 묻는 경우, 원인을 확정할 근거가 없으면
그 한계를 짧게 말합니다. 판본, 작곡가 의도, 연주자 선택을 원인으로 추측하지 않습니다.
작곡가 의도에 관한 일반적인 주의 문구는 이 특정 차이의 원인을 설명하지 않습니다.
두 자료가 공통으로 말하는 내용과 한 자료만 덧붙이는 조언을 구분합니다.
공통 조언이 있다고 해서 두 표기가 같다는 뜻은 아닙니다. 고유한 호흡·울림 조언을
모두의 공통 권고로 바꾸지 않습니다. 기호 뜻은 질문에 필요한 경우에만 설명합니다.
원문 전체를 나열하지 말고 질문의 요점에 맞춰 직접 문장을 작성합니다.
본문에서는 '한 자료', '다른 자료' 같은 자연스러운 표현을 쓰며 E1·E2 같은 내부 라벨을
쓰지 않습니다. 서로 다른 설명을 함께 비교하는 문장에는 두 자료를 함께 인용할 수 있습니다.
JSON 한 개를 반환합니다:
{"sentences":[{"kind":"evidence","evidence":["E1","E2"],"text":"질문에 맞춰 작성한 비교 설명입니다."}]}
1–3개 항목으로 쓰며 text는 비우지 않습니다. 출처의 주장은 evidence 항목에만 쓰고
출처 라벨은 evidence 배열에만 넣습니다. 근거 없는 원인을 background 항목으로 만들지 않습니다.
""" + ("한국어 합니다체로 답합니다." if language == "ko" else "Answer in natural English.") + "\n/no_think\n"


def synthesis_evidence_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep claims/scope, omitting teaching-question premises after selection."""
    records = []
    for record in payload.get("Retrieved evidence", []):
        item = {key: value for key, value in record.items() if key != "question"}
        hints = source_meaning_constraints(record.get("answer", ""))
        if hints:
            item["meaning_constraints"] = hints
        records.append(item)
    view = {**payload, "Retrieved evidence": records}
    # Delay this import because the legacy comparison renderer imports this
    # module. Only source-derived labels/symbols enter the writing input; no
    # canned answer or evaluation-specific question is supplied.
    from soprano_qa.source_comparison import opening_dynamics_accounts
    accounts = opening_dynamics_accounts(payload)
    if accounts:
        view["source_disagreements"] = [{"topic": "opening accompaniment dynamic",
                                        "accounts": accounts, "cause_established": False}]
    return view


class PreservedSourceAnswer(str):
    """Visible source quotations already identity-checked by the finalizer.

    This internal type preserves a quoted author's register through the final
    display pass. Plain model prose and a prose heading cannot create it.
    """


