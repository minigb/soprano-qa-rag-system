"""Small bilingual terminology aid, not expert evidence about a selected work.

Definitions were checked against Brian Blood's articulation lesson:
https://www.dolmetsch.com/musictheory21.htm
These are paraphrases, not copied textbook passages. Performance context can
modify articulation; no universal duration fraction is asserted for staccato.
The lexical translation 가성/falsetto was checked against the Korean Speech-
Language & Hearing Association dictionary (entry 6409):
https://ksha1990.or.kr/dict/6409?etype=F&keyword=&ktype=&page=5
Register labels vary across pedagogical traditions; this is not a claim of a
universal physiological register taxonomy. See NATS Music Theater Resources:
https://www.nats.org/Music_Theater_-_Resources.html
Meter/beat distinctions were checked against Mark Gotham and Chelsey Hamm,
Open Music Theory, "Compound Meter and Time Signatures":
https://viva.pressbooks.pub/openmusictheory/chapter/compound-meters-and-time-signatures/
The general rules below are not answers indexed by evaluation question.
"""

import re


# General theory aid checked against the authors' textbooks, not against
# evaluation references or a selected work's score:
# https://odp.library.tamu.edu/stepstomusictheory/chapter/more-intervals/
# https://openmusictheory.github.io/pitches.html
PITCH_CHANGE_TERMINOLOGY = """General pitch terminology; use only what answers the question.
Chromatic transposition moves every note in a passage by the same directed
interval, preserving intervals within that passage. A non-octave transposition
can change the key; transposition by a whole octave retains the pitch classes.
Moving only selected notes by an octave changes their register and can change
the intervals to surrounding notes, but retains those notes' pitch classes.
That alone does not change the song's key or necessarily damage its harmony.
Enharmonic respelling changes note names, not sounding pitch in twelve-tone
equal temperament. It is not a change of key or an octave displacement.
한국어: 조옮김은 구절의 모든 음을 같은 방향·같은 음정만큼 옮깁니다.
몇 음만 옥타브로 바꾸면 그 음들의 음역과 주변 음과의 간격이 달라질 수
있지만 음고류는 그대로입니다. 이것만으로 조성이 바뀌거나 화성이 망가진다고
하지 않습니다. 모든 음을 옥타브만큼 옮기는 조옮김도 음고류와 조성을 유지합니다.
평균율에서 이명동음은 같은 높이의 음을 다르게 적은 것이며 조옮김이 아닙니다.
No unrequested note-name examples, claims about artistic quality or score facts.
"""


def general_pitch_guidance(question: str) -> str:
    """Supply a small relevant terminology aid, never an answer or verdict."""
    return PITCH_CHANGE_TERMINOLOGY if re.search(
        r"조옮김|이조|옥타브|이명동음|\b(?:transpos\w*|octaves?|enharmonic\w*)\b",
        question, re.I) else ""

MUSIC_TERMINOLOGY = """General musical terminology (not a source about this piece)
Distinguish pitch (음높이), loudness (셈여림), and duration (음의 길이).
- legato / 레가토: smoothly connected notes, 음과 음을 매끄럽게 연결합니다.
- portamento / 포르타멘토: an audible continuous pitch glide between notes,
  한 음에서 다른 음으로 음높이를 연속적으로 미끄러뜨립니다. It is not a
  name for changing loudness or merely expressing emotion.
- staccato / 스타카토: detached notes, 음을 분리하여 짧게 냅니다. Its ordinary
  mark is a dot above or below the note, not a dotted line or an augmentation dot.
- tenuto / 테누토: sustain the full written value, 적힌 음가를 충분히 유지합니다.
  Context may add weight or slight flexibility; it does not invariably mean
  longer than the written value. Its mark is a short horizontal stroke above
  or below the note head, not through the stem.
- accent / 악센트: emphasize a note's attack, 음의 시작을 두드러지게 강조합니다.
  Emphasis does not by itself change the written note value.
- In translation, 가성 is falsetto, not mixed voice. Preserve a source's actual
  list of vocal techniques instead of substituting other familiar register
  names. These pedagogical labels are not a universal physiological taxonomy.
- Distinguish beats (박) from their divisions (박의 분할). In ordinary simple
  meter the beat divides in two; the time signature's numerator counts beats
  and its denominator identifies the beat's note value. In ordinary compound
  meter with numerator 6, 9 or 12, divide that numerator by three to count beats;
  the denominator identifies a division, and three divisions form one dotted
  beat. Equal total written duration does not imply equal beat grouping.
  Duple, triple and quadruple mean two, three and four beats respectively.
  A metrical grouping is not a requirement to force accents on every note.
  Do not invent strong/weak sequences or tempo differences from the fraction.
  Name note values as 음표 (e.g. quarter note = 4분음표), not as pitch intervals.
  한국어: 홑박자는 한 박이 둘로 나뉘고, 겹박자는 한 박이 셋으로 나뉩니다.
  홑박자에서 위 숫자 2·3·4는 한 마디의 박 수이며, 아래 숫자는 한 박의
  음표 단위입니다. 겹박자에서 위 숫자 6·9·12를 3으로 나누면 큰 박이
  각각 2·3·4개이고, 아래 숫자의 음표 세 개가 모여 큰 박 하나가 됩니다.
  아래 숫자 8이면 분할 단위는 8분음표, 큰 박은 점4분음표입니다.
  아래 숫자 4이면 분할 단위는 4분음표, 큰 박은 점2분음표입니다.
  겹박자의 분할 음표 개수를 큰 박의 개수와 혼동하지 않습니다.
Use these distinctions only when relevant, as general explanation without an
expert citation. They do not establish what is written in a particular score.
An expert may ask for a particular feeling while explicitly avoiding literal
articulation; preserve that contrast instead of replacing the advice with a
generic definition. For meaning/comparison questions explain the sound and
duration; omit symbol shapes unless notation is actually requested.
"""
