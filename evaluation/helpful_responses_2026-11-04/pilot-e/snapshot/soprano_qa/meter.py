"""Calculate conventional simple/compound beat units for basic meter questions.

This is a small domain calculator, not an expert KU or question-answer cache.
The rules are checked against Open Music Theory, Gotham and Hamm:
https://viva.pressbooks.pub/openmusictheory/chapter/compound-meters-and-time-signatures/
Only ordinary 2/3/4 simple and 6/9/12 compound readings are supported; alternate
groupings, score facts and performance interpretation remain outside its scope.
Admission and output review are still required by the caller.
"""
from __future__ import annotations

from fractions import Fraction
import re
from typing import Any

_SIGNATURE = re.compile(r"(?<![\w/])(\d{1,2})\s*/\s*(\d{1,2})(?![\d/])")
_TOPIC = re.compile(r"박자|\b(?:meter|metre|time|beat|beats)\b", re.I)
_COMPARISON = re.compile(r"차이|비교|다르|다른|다를|\b(?:compare|differ|different|difference|versus)\b", re.I)
_QUESTION = re.compile(r"정의|분할|큰\s*박|차이|비교|다르|다른|다를|몇\s*박|\b(?:define|explain|compare|differ|different|difference|beat|beats|grouping|divisions?|units?)\b", re.I)
_BEYOND_BASIC = re.compile(
    r"악보|이\s*곡|작곡|연주|부르|노래|가창|발성|연습|왜|이유|교대|변박|헤미올라|분수|날짜|"
    r"\b(?:score|song|piece|composer|performance|sing|singing|play|playing|practice|"
    r"conduct|conducting|why|hemiola|alternat\w*|irregular|syncopat\w*|fractions?|dates?|calendar)\b", re.I)
_KO_NOTES = {1: "온음표", 2: "2분음표", 4: "4분음표", 8: "8분음표", 16: "16분음표", 32: "32분음표", 64: "64분음표"}
_EN_NOTES = {1: "whole note", 2: "half note", 4: "quarter note", 8: "eighth note", 16: "sixteenth note", 32: "thirty-second note", 64: "sixty-fourth note"}


def meter_facts(numerator: int, denominator: int) -> dict[str, Any] | None:
    if numerator not in (2, 3, 4, 6, 9, 12) or denominator not in (2, 4, 8, 16, 32):
        return None
    compound = numerator in (6, 9, 12)
    divisions = 3 if compound else 2
    beats = numerator // 3 if compound else numerator
    return {"signature": f"{numerator}/{denominator}", "compound": compound,
            "beats": beats, "divisions": divisions,
            "beat_note_denominator": denominator // 2 if compound else denominator,
            "division_note_denominator": denominator if compound else 2 * denominator,
            "bar_duration": Fraction(numerator, denominator)}


def basic_meter_explanation(payload: dict[str, Any]) -> str | None:
    question = payload.get('question', '')
    if (not isinstance(question, str)
            or not _QUESTION.search(question) or _BEYOND_BASIC.search(question)
            or payload.get('measure_range') not in (None, '', '(none selected)')
            or payload.get('Score context')):
        return None
    compact = lambda text: re.sub(r'[\W_]+', '', text).casefold()
    sources = payload.get('Retrieved evidence', [])
    names = [payload.get('piece')]
    for source in sources:
        if source.get('question_alias_match') is True:
            return None
        names.extend(source.get(key) for key in ('work', 'piece_title'))
    if any(isinstance(name, str) and len(compact(name)) >= 3 and compact(name) in compact(question)
           for name in names):
        return None
    signatures = list(dict.fromkeys((int(n), int(d)) for n, d in _SIGNATURE.findall(question)))
    if not 1 <= len(signatures) <= 2:
        return None
    # A terse comparison can inherit the selected music context without
    # repeating "time signature". Explicit arithmetic/date tasks never do.
    if not (_TOPIC.search(question) or
            (len(signatures) == 2 and payload.get('piece') and _COMPARISON.search(question))):
        return None
    facts = [meter_facts(n, d) for n, d in signatures]
    if any(f is None for f in facts):
        return None
    korean = payload.get('answer_language', 'ko') == 'ko'
    parts = []
    for fact in facts:
        if korean:
            unit = ('점' if fact['compound'] else '') + _KO_NOTES[fact['beat_note_denominator']]
            division = _KO_NOTES[fact['division_note_denominator']]
            meter = ('겹' if fact['compound'] else '홑') + str(fact['beats']) + '박자'
            parts.append(f"일반적인 {fact['signature']}박자는 {meter}로, 한 마디에 {unit} 박이 "
                         f"{fact['beats']}개 있고 한 박은 {division} {fact['divisions']}개로 나뉩니다.")
        else:
            unit = ('dotted ' if fact['compound'] else '') + _EN_NOTES[fact['beat_note_denominator']]
            division = _EN_NOTES[fact['division_note_denominator']]
            meter = ('compound' if fact['compound'] else 'simple') + ' ' + {2: 'duple', 3: 'triple', 4: 'quadruple'}[fact['beats']]
            parts.append(f"In the usual {meter} reading of {fact['signature']}, a bar has {fact['beats']} "
                         f"{unit} beats, each divided into {fact['divisions']} {division}s.")
    if len(facts) == 2:
        equal = facts[0]['bar_duration'] == facts[1]['bar_duration']
        if korean:
            parts.append('한 마디에 담긴 음표 길이의 합은 같지만 박의 묶음은 다릅니다.' if equal
                         else '두 박자는 한 마디에 담긴 음표 길이의 합도 다릅니다.')
        else:
            parts.append('Their total written duration per bar is equal, but their beat grouping differs.' if equal
                         else 'Their total written duration per bar is also different.')
    return ' '.join(parts)
