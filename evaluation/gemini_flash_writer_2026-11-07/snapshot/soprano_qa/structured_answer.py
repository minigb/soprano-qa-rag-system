"""Constrained answer structure with application-rendered model citations.

The schema guarantees neither truth nor entailment. A semantic guard must check
this module's rendered output, followed by the normal citation/scope validators.
The model selects evidence labels; the renderer never invents an attribution.
"""
from __future__ import annotations

import json
import re
from typing import Any

from soprano_qa.security import SecurityViolation
from soprano_qa.music_terms import MUSIC_TERMINOLOGY


MAX_SENTENCES = 6
MAX_SENTENCE_CHARS = 2000
# Keep this low-level module independent of answer.py, which imports llm.py.
NO_GROUNDED_ANSWER_SENTINEL = "<NO_GROUNDED_ANSWER>"

STRUCTURED_ANSWER_POLICY = """Return one JSON object, no markdown or extra keys:
{"sentences":[{"kind":"evidence","evidence":["E1"],"text":"Source-based answer."}]}
Choose each item's kind and evidence before writing its text:
- evidence: a faithful claim from the supplied reviewed answers; select its non-null
  citation_label values. At least one supporting label is required.
- background: general music knowledge, an explanation, or an explicitly optional
  practice suggestion. evidence must be []. It must not assert unverified facts
  about this piece/bar, quote an expert, explain an unestablished cause, or contradict
  the supplied advice. The application identifies it as general explanation.
- score_lyrics: only words sung in the supplied score context; evidence must be [].
Keep these kinds in separate items. Never attach a source citation to background.
Use 1-4 items, up to 6 if needed. Each text must be nonempty and at most 2000
characters. Do not pad, repeat, print labels/IDs, or write the kind name in prose.
If the source answers the question, do not add unnecessary background. If evidence
is missing, a useful general explanation is allowed, with an explicit limitation
when the question asks for a specific score fact. If no useful authorized answer
is possible, return {"sentences":[]}.
/no_think
"""


SEMANTIC_ANSWER_POLICY = """Write the answer yourself using the retrieved passages as evidence.
Your task is to answer the user's question, not display or translate complete
source records. Start with the relevant answer or practical recommendation.
Write a connected, concise explanation in the requested language and register.
Combine repeated advice once, retaining the conditions of every claim you use.
Include useful explanations and alternatives; omit passages that do not answer
the question. Do not list sources one after another or announce that you are
presenting reference text. For Korean, use natural, consistent 합니다체.

Preserve the musical meaning, not the source's paragraph structure. Check who
does what, negation, the condition under which advice applies, alternatives,
certainty, and the specific musical location or situation. A possibility or
interpretation must not become a definite fact or composer's intention. Clarify
a source's 'this part' using the supplied musical situation; if the situation
cannot be identified, state that limitation instead of generalizing its advice.
Retain quoted lyric words, pronunciation spellings and musical symbols exactly.
These short quotations do not require quoting the surrounding source paragraph.
For conflicting accounts, distinguish their claims and preserve uncertainty
about the cause; do not invent a reconciliation or call unique advice shared.

Return one JSON object with the following structure and no extra keys:
{"sentences":[{"kind":"evidence","evidence":["E1"],"text":"Your direct, source-supported answer."}]}
Write nonempty text in every item. Use evidence items for supported source
claims, selecting the labels of the passages supporting those claims. Use
background items with evidence=[] for useful general musical knowledge or
optional practice advice that the sources do not supply. Keep that provenance
separate; background must not invent a fact about a piece, edition or expert,
contradict a source, or explain an unknown source disagreement. score_lyrics
items may identify only sung words actually supplied in the score context.
Use 1-4 items, up to 6 if the question needs it. Each item can contain connected
sentences, at most 2000 characters. Do not print IDs, headings, citation labels,
or internal field names in the text. The application handles citations.
If no useful authorized answer can be given, return {"sentences":[]}.
/no_think
""" + MUSIC_TERMINOLOGY


def concise_synthesis_policy(language: str) -> str:
    language_rule = ("한국어 합니다체로 자연스럽게 답합니다." if language == "ko" else
                     "Answer in natural English; retain exact lyric words and pronunciation spellings in their original script.")
    return """당신은 성악과 음악에 관한 질문에 답하는 도우미입니다.
질문에 바로 답하고, 제공된 자료에서 필요한 내용만 연결하여 짧고 명확하게 설명합니다.
자료 목록이나 원문 모음을 반환하지 않습니다. 반복되는 내용은 한 번만 설명합니다.
관련 자료가 부족하면 정확한 일반 음악 지식과 선택적인 연습 제안도 사용할 수 있습니다.
자료만으로 충분히 답했다면 일반론을 덧붙이지 않습니다. 출처별 항목이 아니라
질문의 요점별로 문장을 연결합니다. 한 항목에 여러 출처를 함께 사용할 수 있습니다.
서로 연결되는 자료 내용은 evidence 라벨을 함께 넣은 한 항목으로 통합합니다.
같은 주장의 동의어 문장을 반복하지 말고 질문에 필요한 설명만 남깁니다.

자료의 문장 표현은 바꿀 수 있지만 다음 의미는 바꾸지 않습니다.
각 자료의 meaning_constraints는 원문에서 주의할 표현을 발췌한 것입니다.
해당 내용을 답변에 사용할 때 원문의 논리 관계를 반드시 유지합니다.
source_disagreements가 있으면 같은 대상에 대한 설명이 자료마다 다른 것입니다.
각각 누구의 설명인지 구분하고, 두 표기가 모두 확정된 사실인 것처럼 쓰지 않습니다.
- ‘볼 수 있다’, ‘해석하는 것이 일반적이다’는 가능한 해석입니다.
  이를 확정된 원인이나 ‘작곡가가 그렇게 의도했다’는 사실로 바꾸지 않습니다.
  질문이 작곡가의 의도를 단정하더라도 자료가 뒷받침하지 않으면 그대로 단정하지 않습니다.
- ‘A보다 B가 좋다’는 B를 권장한다는 뜻입니다. 뒤에 C라는 보조 방법이 있어도
  B를 피해야 한다거나 C만 써야 한다는 뜻이 아닙니다.
- 부정, 조건, 선택지, 권고의 강도, 적용되는 상황과 주체를 유지합니다.
  ‘또는’을 ‘그리고’로 바꾸거나, 가끔 나타나는 현상을 의무로 바꾸지 않습니다.
- 가사·발음 예시·음악 기호는 정확히 유지합니다. 내용을 모르는 구간·판본·작곡 의도를
  추측하지 않습니다. 서로 다른 자료의 설명은 구분하고 차이의 원인을 꾸며내지 않습니다.
- 일반 설명과 자료의 주장을 구분합니다. 일반 지식을 특정 전문가의 말처럼 인용하지 않습니다.

마디가 선택되어 있으면 selected_range_support만 그 마디의 사실을 뒷받침합니다.
일반 조언은 원래의 조건을 유지합니다. 마디 선택이 없으면 국소적 예시는 그 상황에
한해서 설명하며, 곡 전체의 법칙으로 확장하지 않습니다. 내부 마디 메타데이터를
출력하지 않고 원문에 있는 ‘첫머리’, ‘무반주 구간’ 같은 상황을 유지합니다.
악보 가사 문맥은 제공된 가사를 확인하는 용도이며 음정·리듬·표현법의 증거는 아닙니다.

출력은 다음 형식의 JSON 한 개입니다. 설명문이나 Markdown으로 감싸지 않습니다.
{"sentences":[{"kind":"evidence","evidence":["E1"],"text":"질문에 직접 답하는 문장입니다."}]}
자료가 뒷받침하는 문장은 kind=evidence, evidence에는 해당 자료의 라벨을 넣습니다.
일반 음악 설명이나 선택적인 연습 제안은 kind=background, evidence=[]입니다.
제공된 가사만 확인하는 문장은 kind=score_lyrics, evidence=[]입니다.
1–4개 항목으로 간결하게 쓰고 필요한 경우에만 최대 6개를 씁니다. text는 비우지 않습니다.
본문에 라벨·내부 용어·자료 반환 안내·제목을 쓰지 않습니다. 출처별로 문단을 나열하지 않습니다.
유용한 답변이 전혀 불가능할 때만 {"sentences":[]}를 반환합니다.

다음은 문장 관계를 지키는 예시입니다. 예시의 내용은 실제 답변의 근거가 아닙니다.
예시 1: 질문은 ‘작곡가는 왜 중간에 쉼을 넣었나요?’이고 자료 E1은
‘그 쉼은 인물의 망설임을 표현한 것으로 볼 수 있다. 쉼 뒤의 시작을 부드럽게 준비한다.’입니다.
올바른 답변: {"sentences":[{"kind":"evidence","evidence":["E1"],"text":"그 쉼은 인물의 망설임을 표현하는 것으로 해석할 수 있습니다. 쉼 뒤에는 부드럽게 시작할 수 있도록 준비합니다."}]}
‘작곡가는 망설임을 표현하려고 쉼을 넣었습니다’는 원문보다 강한 단정이므로 쓰지 않습니다.
예시 2: 자료 E1은 ‘빠르게 반복하기보다 천천히 연습하는 것이 좋다.
박자가 흔들릴 때는 손뼉을 치는 것도 도움이 될 수 있다.’입니다.
올바른 답변: {"sentences":[{"kind":"evidence","evidence":["E1"],"text":"천천히 연습하는 것이 좋습니다. 박자가 흔들린다면 손뼉을 치며 연습하는 방법도 도움이 될 수 있습니다."}]}
‘천천히 연습하지 말고 반드시 손뼉을 쳐야 합니다’는 권고와 보조 방법을 뒤집으므로 쓰지 않습니다.
예시 3: 자료 E1은 ‘혼자 연습하거나 반주가 없는 경우 녹음으로 확인할 수 있다.’입니다.
올바른 답변: {"sentences":[{"kind":"evidence","evidence":["E1"],"text":"혼자 연습할 때나 반주가 없을 때는 녹음해서 확인하는 방법도 있습니다."}]}
‘혼자 무반주로 연습할 때’는 두 조건을 모두 요구하므로 원문과 다릅니다.
예시 4: 자료 E1은 ‘첫 부분은 빠른 템포다. 호흡을 미리 준비한다.’이고
자료 E2는 ‘첫 부분은 보통 템포다. 호흡을 미리 준비하고 입모양도 준비한다.’입니다.
올바른 답변: {"sentences":[{"kind":"evidence","evidence":["E1","E2"],"text":"첫 부분의 템포에 대해서는 빠르다는 설명과 보통이라는 설명이 있습니다. 두 자료 모두 호흡을 미리 준비하도록 권합니다. 두 번째 자료는 입모양도 준비하도록 덧붙입니다."}]}
자료만으로 차이의 원인을 알 수 없다면 판본이나 연주자의 선택 때문이라고 추측하지 않습니다.
""" + language_rule + "\n/no_think\n"


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


SOURCE_PRESERVATION_POLICY = """Return one JSON object with the existing sentences/kind/evidence/text fields.
Select only passages that directly answer the actual question, preserving useful
complementary advice. An evidence item must select exactly ONE citation label;
never select the same label twice. Select complete reviewed answers, not isolated
clauses whose conditions could be lost. Do not merge conflicting passages into
one new claim or invent a reason for their disagreement.
Omit a second source if it only repeats advice already fully covered; retain
distinct qualifications, alternatives and genuine disagreements. A shared
musical subject does not make a source an answer to the requested task.

For Korean answers, every evidence item's text MUST be the empty string "".
Example: {"sentences":[{"kind":"evidence","evidence":["E1"],"text":""}]}
The application will quote that selected source's complete answer verbatim,
including its conditions, uncertainty, lyric referents and original register.
Do not rewrite it or add an explanation inside an evidence item.
For English answers, an evidence item's text MUST faithfully translate the
COMPLETE answer of its one selected source. Preserve every condition, actor,
negation, qualification, alternative and lyric referent. Do not summarize,
combine sources, strengthen an interpretation into composer intention, or add
a practice claim. Keep quoted sung words and vowel/consonant symbols unchanged.
Do not add romanization, transliteration, or a new phonetic spelling of sung
words. Retain their original script exactly. Explain a pronunciation spelling
only when the source itself supplies it. Preserve every term in a source's
list of techniques; do not substitute different familiar examples.

Background items require evidence=[] and nonempty text. They may give accurate
general music knowledge or explicitly optional suggestions, clearly separate
from the selected source. Add them only when the source alone leaves a useful
part of the question unanswered. Do not repeat, contradict, or invent a cause
for the source. Do not synthesize source advice in a background paragraph:
the selected quotations already supply that advice with its proper scope.
If a requested reason for source disagreement is unknown, state that limit
in one sentence; do not add possible causes, interpretations or shared advice.
Unknown local facts must remain unknown. score_lyrics items
remain limited to supplied sung words and require evidence=[]. All application
scope, safety and output-review requirements still apply to every item.
Use as few items as needed, at most six. Return {"sentences":[]} if no useful
authorized answer is possible. Never print labels or headings in item text.
/no_think
"""

SOURCE_PRESERVATION_POLICY += "\n" + MUSIC_TERMINOLOGY

SOURCE_SELECTION_POLICY = """Return one JSON object with sentences/kind/evidence/text fields.
Select complete reviewed answers that directly help answer the question.
For EVERY evidence item choose exactly one supplied citation label and set
text to the EMPTY STRING. Never select a label twice. Do not translate or
paraphrase a source here: the application translates selected sources separately.
Example: {"sentences":[{"kind":"evidence","evidence":["E1"],"text":""}]}
Preserve useful complementary sources without selecting unrelated passages.
Omit a second source if it only repeats the same advice already fully covered;
keep distinct qualifications, alternatives and genuine source disagreements.
Background items may provide accurate general music knowledge or explicitly
optional practice suggestions in English, only when they answer a useful aspect
left open by the sources. Their evidence must be [], and text must be nonempty.
Never invent a local fact, cause or expert attribution, or contradict supplied
advice. Do not synthesize the selected source advice again in background.
If a requested reason for source disagreement is unknown, state that limit
in one sentence without possible causes, interpretations or shared advice.
score_lyrics items are limited to supplied sung words, with evidence=[].
Use as few items as needed, at most six. Return {"sentences":[]} if no useful
authorized answer is possible. All safety and output-review requirements apply.
/no_think
""" + MUSIC_TERMINOLOGY



class PreservedSourceAnswer(str):
    """Visible source quotations already identity-checked by the finalizer.

    This internal type preserves a quoted author's register through the final
    display pass. Plain model prose and a prose heading cannot create it.
    """


GENERAL_MUSIC_SYSTEM_PROMPT = """You answer music-theory and singing-practice questions.
No directly answering reviewed source was selected for this request. Use your
general musical knowledge to answer the actual question helpfully and directly.
For practice questions in this soprano service, assume singing unless the user
explicitly asks about an instrument. Do not substitute keyboard or finger drills
for vocal practice. Without hearing the singer, suggest optional checks rather
than diagnosing their breathing, range or technique.
Definitions, comparisons, note values, rhythm, terminology and optional general
practice suggestions do not require a source passage. Do not refuse these just
because no retrieved source was selected. The selected piece name is context,
not an instruction to discuss that piece in a general concept answer.

Use kind=background and evidence=[] for general explanations. Do not claim that
an expert said them. Do not invent facts about the selected piece, measure,
edition, recording or composer's particular intention. If asked for an unknown
score fact, briefly state what cannot be established; offer a general explanation
only if it helps. Never infer a specific pitch, lyric, notation or source-conflict
cause from general knowledge. Practice suggestions must be optional, not personal
diagnoses or guaranteed outcomes. When score context is present, kind=score_lyrics
may identify only the supplied sung words, without making performance claims.

Give one concise, coherent paragraph that directly answers the question.
Include only details needed to answer the question. Avoid assumed tempos,
numerical demonstrations or extra illustrations unless the question requires
them. Check units and proportions before stating a quantitative relationship.
Do not assume a BPM beat is a quarter note: a seconds-based example must state
which note value is the beat unit, or omit the unsupported numerical example.
Use the requested answer_language. For Korean use formal-polite 합니다체;
for English use natural English and preserve quoted lyric words as supplied.
The application labels background; do not add a heading yourself. If a useful
general answer is impossible, state the missing information briefly. A statement
of that limitation is a valid background item; never invent a fact to fill one.
/no_think
"""

GENERAL_MUSIC_FORMAT_POLICY = """Return one JSON object, no markdown or extra keys:
{"sentences":[{"kind":"background","evidence":[],"text":"Your answer in the requested language."}]}
Write one concise background item for general musical knowledge
or an explicit statement of what cannot be established. evidence must be [].
When actual score context is supplied, score_lyrics may identify only its words.
Only that mixed lyric/explanation case may need two items; keep kinds separate.
Do not print labels, IDs, headings or kind names in prose.
Each text must be nonempty and at most 2000 characters. Do not pad or repeat.
For an unanswerable local fact, state the limitation; do not guess it. A
required text item is not a requirement to make a factual claim. Task, source
and security boundaries still apply to every item.
/no_think
"""


GENERAL_MUSIC_FORMAT_POLICY += "\n" + MUSIC_TERMINOLOGY


def concise_concept_system_prompt(language: str) -> str:
    return ("Explain the requested music concept accurately in TWO short sentences. "
            f"Answer in {language}. No examples, headings, lists or extra judgments.\n")


GENERAL_PRACTICE_SYSTEM_PROMPT = """Give ONE optional, practical singing exercise for the requested skill.
Use the requested answer_language, formal-polite 합니다체 for Korean. In this
soprano service, practice means singing unless an instrument is explicitly asked
for. Do not substitute an instrumental exercise or a different vocal technique.
Describe a simple action the singer can try, using a comfortable register and
short phrases with breaths as needed. Do not assume the singer's breath capacity,
range, health or technique, prescribe sustained duration, or promise improvement.
Do not add vibrato, slides, dynamic changes or range extensions unless the
requested skill requires them. A short note sequence is enough; avoid long holds.
For connected singing, connect adjacent notes within each short phrase and take
breaths BETWEEN phrases as needed. Shorten the phrase instead of detaching every
note. Do not prescribe seconds per note or invent note-name examples; use a brief
familiar phrase. 음을 분리하는 연습을 레가토라고 설명하지 마세요.
Keep the answer brief and concrete; avoid an essay defining the technique.
This is general practice advice, not a fact about the selected piece or a quoted
expert recommendation. Use background, evidence=[]; never invent local score facts.
/no_think
"""

GENERAL_PRACTICE_SYSTEM_PROMPT_KO = """질문에서 요청한 기법을 연습하는 성악 연습 한 가지를 짧고 구체적으로 설명합니다.
일반적인 연습 제안이며 전문가의 인용이나 선택한 곡에 대한 사실이 아닙니다.
사용자의 호흡량·음역·발성 상태를 알고 있다고 가정하거나 효과를 보장하지 않습니다.
사용자가 편한 음역에서 익숙한 짧은 구절을 골라 시도할 수 있도록 안내합니다.
음이름의 나열, 특정 곡의 가사, 음 하나를 몇 초씩 내라는 수치 예시는 쓰지 않습니다.
레가토 연습이라면 짧은 구절 안에서 이웃한 음을 매끄럽게 연결합니다. 호흡이
필요하면 구절을 더 짧게 나누고 구절 사이에서 숨을 쉽니다. 음표마다 끊어 부르는
연습이나 포르타멘토·비브라토를 레가토 대신 제안하지 않습니다.
요청한 기법과 관계없는 연습은 덧붙이지 않습니다. 편안한 범위에서 선택적으로
해 볼 수 있는 방법으로 설명하며, 합니다체를 사용합니다. kind는 background,
evidence는 []입니다. 악보의 알려지지 않은 사실을 만들어 내지 않습니다.
/no_think
"""


class RenderedAnswer(str):
    """Carry validated claim provenance until public answer finalization.

    A prose prefix alone never grants an exemption from citation validation.
    Only parts validated by render_structured_answer receive this metadata.
    """

    def __new__(cls, parts: list[dict[str, Any]], language: str = "ko"):
        grounded = "\n".join(_cite_text(p["text"], p["evidence"])
                             for p in parts if p["kind"] != "background")
        background = "\n".join(p["text"] for p in parts if p["kind"] == "background")
        prefix = "General explanation: " if language == "en" else "일반적인 음악 설명입니다.\n"
        visible = "\n\n".join(x for x in (grounded, prefix + background if background else "") if x)
        if any(p.get("source_quotation") for p in parts):
            visible = "참고 자료의 원문입니다.\n" + visible
        elif any(p.get("source_translation") for p in parts):
            visible = "Translated reference text:\n" + visible
        value = super().__new__(cls, visible)
        value.parts = tuple(parts)
        value.grounded_text = grounded
        value.background_text = background
        kinds = {p["kind"] for p in parts}
        value.answer_basis = (
            "retrieved_evidence_with_background" if kinds >= {"evidence", "background"}
            else "retrieved_evidence" if "evidence" in kinds
            else "score_context_with_background" if kinds >= {"score_lyrics", "background"}
            else "general_music_knowledge" if "background" in kinds
            else "score_context"
        )
        return value

_LABEL_RE = re.compile(r"E[1-9][0-9]*\Z")
_INLINE_CITATION_RE = re.compile(
    r"[\[【]\s*(?:E\s*[0-9]|Evidence\b|근거\s*[0-9])",
    flags=re.IGNORECASE,
)
_REDUNDANT_CITATION_RE = re.compile(r"\[(E[1-9][0-9]*(?:\s*,\s*E[1-9][0-9]*)*)\]")
# Match the downstream claim validator's punctuation boundaries. Decimal dots
# and abbreviation-internal dots have no following whitespace and stay intact.
# An abbreviation ending before whitespace is conservatively cited as well;
# its original characters are preserved and display-time stripping restores it.
# The downstream validator applies NFKC, which turns … into ... and fullwidth
# punctuation into ASCII. Cite these boundaries too, while preserving the
# original lyric typography in displayed text.
_SENTENCE_END_RE = re.compile(r"[.!?。！？…﹒．｡]+[\"'”’」』】)）]*(?=\s|$|[가-힣])")


def _allowed_labels(payload: dict[str, Any]) -> list[str]:
    evidence = payload.get("Retrieved evidence", [])
    if not isinstance(evidence, list):
        raise SecurityViolation()
    labels: list[str] = []
    for record in evidence:
        if not isinstance(record, dict):
            raise SecurityViolation()
        label = record.get("citation_label")
        if label is None:
            continue
        if not isinstance(label, str) or not _LABEL_RE.fullmatch(label):
            raise SecurityViolation()
        if label not in labels:
            labels.append(label)
    return labels


def answer_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON schema constrained to this bundle's primary labels."""
    labels = _allowed_labels(payload)
    evidence_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", **({"enum": labels} if labels else {})},
        "minItems": 0,
        "maxItems": len(labels),
        "uniqueItems": True,
    }
    schema = {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                # Empty-bundle generations otherwise collapsed to [] even for
                # elementary theory questions. Require an explanation OR an
                # explicit limitation; the semantic guard still checks both.
                "minItems": 0 if labels else 1,
                "maxItems": MAX_SENTENCES if labels else (2 if payload.get('Score context') else 1),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum":
                                 (["evidence"] if labels else []) + ["background"] +
                                 (["score_lyrics"] if payload.get("Score context") else [])},
                        "evidence": evidence_schema,
                        "text": {
                            "type": "string",
                            "minLength": 0 if labels and payload.get("source_preservation") and (payload.get("answer_language", "ko") == "ko" or payload.get("separate_source_translation")) else 1,
                            # llama.cpp expands bounded string repetition;
                            # a 2000-character schema bound exceeds its grammar
                            # complexity limit. Enforce that cap in the parser;
                            # generation remains bounded by max_tokens.
                        },
                    },
                    "required": ["kind", "evidence", "text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["sentences"],
        "additionalProperties": False,
    }
    if payload.get("source_preservation") is True and labels:
        native = payload.get("answer_language", "ko") == "ko" or payload.get("separate_source_translation") is True
        def item(kind, source=False):
            return {
                "type": "object",
                "properties": {
                    "kind": {"const": kind},
                    "evidence": {"type": "array", "items": {"type": "string", "enum": labels},
                                 "minItems": 1 if source else 0, "maxItems": 1 if source else 0},
                    "text": {"const": ""} if source and native else {"type": "string", "minLength": 1},
                },
                "required": ["kind", "evidence", "text"], "additionalProperties": False,
            }
        alternatives = [item("evidence", True), item("background")]
        if payload.get("Score context"):
            alternatives.append(item("score_lyrics"))
        schema["properties"]["sentences"]["items"] = {"oneOf": alternatives}
    return schema


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _cite_text(text: str, labels: list[str]) -> str:
    if not labels:
        return text
    suffix = " [" + ", ".join(labels) + "]"
    cited_lines = []
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content):]
        segments: list[str] = []
        start = 0
        for match in _SENTENCE_END_RE.finditer(content):
            segment = content[start:match.end()]
            segments.append(segment + suffix if segment.strip() else segment)
            start = match.end()
        trailing = content[start:]
        # Keep all existing whitespace: insert labels immediately after prose.
        if trailing.strip():
            prose = trailing.rstrip()
            segments.append(prose + suffix + trailing[len(prose):])
        else:
            segments.append(trailing)
        cited_lines.append("".join(segments) + newline)
    return "".join(cited_lines)


def render_structured_answer(raw: str, payload: dict[str, Any]) -> str:
    """Validate exact structure and render only the model-selected citations.

    Each item's labels apply to every sentence fragment in its text. This also
    covers a model that puts multiple sentences in one item; the subsequent
    semantic guard must verify support for every rendered claim and citation.
    An empty array represents an unsupported answer, never a successful response.
    """
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, RecursionError):
        raise SecurityViolation() from None
    if not isinstance(parsed, dict) or set(parsed) != {"sentences"}:
        raise SecurityViolation()
    sentences = parsed["sentences"]
    if not isinstance(sentences, list) or len(sentences) > MAX_SENTENCES:
        raise SecurityViolation()
    allowed_labels = set(_allowed_labels(payload))
    if not sentences:
        return NO_GROUNDED_ANSWER_SENTINEL
    rendered = []
    selected_quotes: set[str] = set()
    preserve = payload.get("source_preservation") is True
    source_by_label = {r.get("citation_label"): r for r in payload.get("Retrieved evidence", [])}
    for item in sentences:
        if not isinstance(item, dict) or set(item) not in ({"text", "evidence"}, {"kind", "text", "evidence"}):
            raise SecurityViolation()
        text, labels = item["text"], item["evidence"]
        # Legacy source-only drafts remain readable; new model output always
        # carries kind under the constrained schema.
        kind = item.get("kind", "evidence" if allowed_labels else "score_lyrics")
        if kind not in {"evidence", "background", "score_lyrics"}:
            raise SecurityViolation()
        native_quote = preserve and kind == "evidence" and (
            payload.get("answer_language", "ko") == "ko" or payload.get("separate_source_translation") is True)
        if (
            not isinstance(text, str)
            or (not text.strip() and not native_quote)
            or len(text) > MAX_SENTENCE_CHARS
            or not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
        ):
            raise SecurityViolation()
        if (
            len(labels) != len(set(labels))
            or not set(labels) <= allowed_labels
            or (kind == "evidence" and not labels)
            or (kind != "evidence" and labels)
            or (kind == "score_lyrics" and not payload.get("Score context"))
        ):
            raise SecurityViolation()
        if preserve and kind == "evidence":
            if len(labels) != 1 or labels[0] in selected_quotes:
                raise SecurityViolation()
            selected_quotes.add(labels[0])
        if native_quote:
            if text != "":
                raise SecurityViolation()
            source = source_by_label[labels[0]].get("answer")
            if (not isinstance(source, str) or not source.strip()
                    or len(source) > MAX_SENTENCE_CHARS or _INLINE_CITATION_RE.search(source)):
                raise SecurityViolation()
            # Text comes from the frozen primary record, never a model quote.
            # Existing security screening and semantic output review still
            # inspect this complete rendered text before it can be released.
            rendered.append({"kind": kind, "text": source, "evidence": labels,
                             "source_quotation": True})
            continue
        # Some models repeat a selected label in prose despite the schema
        # instruction. Remove only redundant labels already selected for this
        # item; an unknown, context-only or different label still fails closed.
        def remove_redundant(match: re.Match[str]) -> str:
            if not set(re.findall(r"E[1-9][0-9]*", match.group(1))) <= set(labels):
                raise SecurityViolation()
            return ""

        text = _REDUNDANT_CITATION_RE.sub(remove_redundant, text)
        if not text.strip() or _INLINE_CITATION_RE.search(text):
            raise SecurityViolation()
        rendered.append({"kind": kind, "text": text, "evidence": labels})
    return RenderedAnswer(rendered, payload.get("answer_language", "ko"))
