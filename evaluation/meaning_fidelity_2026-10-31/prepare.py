"""Freeze an authored meaning-fidelity diagnostic from existing failure records."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'evaluation/quality_refinement_2026-10-31'

def read(path):
    return json.loads(path.read_text())

def write(name, value):
    path = OUT / name
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Refusing to replace frozen input: {path}')
        return
    path.write_text(text)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

constraints = {
 'yeon-die-forelle-05': 'Broad-answer control: preserve useful diction, accent and tone guidance without an invented result.',
 'kim-in-flowery-clouds-01': 'Opening-only fermata, possible arrangement difference, both allowed; no invented decision procedure for all fermatas.',
 'kim-in-flowery-clouds-02': 'Preserve the lyric and consonant example exactly; no inserted vowels or invented syllables.',
 'kim-in-flowery-clouds-05': 'Half note is a duration, not the second note. Preserve the no-breath recommendation and conditional alternative if breathing is necessary.',
 'kim-in-flowery-clouds-07': 'Keep mood transition, hardship imagery and meter/key relations distinct; do not invent additional composer intention or emotional effects.',
 'kim-in-flowery-clouds-09': 'Natural crescendo is permitted, not a requirement to force or intentionally enlarge the sound.',
 'kim-la-capinera-01': 'Broad local control: keep pp entry and the l/r diction guidance when applicable.',
 'kim-la-capinera-11': 'Two-to-three times refers to duration, not pitch lowering; preserve the mor accent.',
 'kim-la-capinera-13': 'Keep the adjacent e vowels and placement of d near the following i; do not assign an e-initial vowel to fior.',
 'kim-la-capinera-14': 'Often produced like one consonant, with a two-articulation feeling and short pause; do not negate the one-consonant qualifier.',
 'kim-nella-fantasia-02': 'No male/female restriction does not establish unlimited vocal range or no tessitura restrictions.',
 'yeon-nella-fantasia-07': 'Broad-answer control: preserve breathing/connected lyrical line without guaranteed outcomes.',
 'kim-una-voce-poco-fa-03': 'Character expression is a common interpretation, not established composer intention; short-note placement is a separate recommendation.',
 'kim-una-voce-poco-fa-06': 'Preserve possible reasons, orchestral versus piano distinction, and the separate slashed/unslashed rules.',
 'kim-una-voce-poco-fa-08': 'Distinguish elision in l’ingegno from linking o-a in l’ingegno aguzzerò; do not swap examples.',
 'kim-una-voce-poco-fa-09': 'Practice slowly then faster; vowel-only practice omits consonants, not emphasizes them; preserve weak-syllable treatment.',
 'kim-una-voce-poco-fa-15': 'Interpretation/recommendation, not certain composer intent or a newly invented causal reason for the notated accent.',
 'kim-una-voce-poco-fa-18': 'Singer shapes freely under col canto, accompanist follows; a tempo returns to original beat and rhythm.',
 'kim-una-voce-poco-fa-19': 'Unslashed and slashed ornaments have different rules; preserve usual/often qualifiers and the conditional four-sixteenth-note example.',
}
corpus = {r['id']: r for r in read(ROOT / 'data/corpus.json')}
cases, historical = [], []
for group in read(ROOT / 'evaluation/qualitative.json')['questions']:
    sid = group['source_id']
    if sid not in constraints:
        continue
    for run in group['inference_runs']:
        request = run['inference_input']
        refs = request['case_reference_authority']['items']
        ids = list(dict.fromkeys(x['knowledge_unit_id'] for x in refs if x.get('knowledge_unit_id')))
        case = {'id': 'history-' + run['case_id'], 'kind': 'benign', 'language': 'ko',
                'pair_id': sid, 'piece_id': request['piece_id'], 'question': request['question'],
                'measure_range': request['measure_range'], 'category': 'historical_meaning_regression',
                'reference_answers': [corpus[i]['answer'] for i in ids], 'gold_evidence_ids': ids,
                'meaning_constraints': constraints[sid], 'injection': None}
        cases.append(case)
        historical.append({'case_id': case['id'], 'artifact': 'evaluation/qualitative.json',
                           'recorded_at': run.get('last_attempt_at'), 'response': run['pipeline_result']})

static_rows = [json.loads(line) for line in (PREVIOUS / 'static-preview/requests.jsonl').read_text().splitlines()]
static_choices = {
 7: ['una-voce-poco-fa-ku-012','una-voce-poco-fa-ku-013','una-voce-poco-fa-ku-014'],
 11: ['la-capinera-ku-018'], 15: ['la-capinera-ku-007'],
 17: ['nella-fantasia-ku-008','nella-fantasia-ku-009'], 18: ['nella-fantasia-ku-010'],
 19: ['nella-fantasia-ku-004','nella-fantasia-ku-014'],
 21: ['in-flowery-clouds-ku-002','in-flowery-clouds-ku-004'],
 23: ['in-flowery-clouds-ku-012'], 25: ['in-flowery-clouds-ku-006'],
}
for n, ids in static_choices.items():
    req = deepcopy(static_rows[n-1]['request']);req.pop('generate')
    cases.append({'id': f'static-{n:02d}-en', 'kind': 'benign', 'pair_id': f'static-{n:02d}',
                  **req, 'category': 'static_translation_regression',
                  'reference_answers': [corpus[i]['answer'] for i in ids], 'gold_evidence_ids': ids,
                  'meaning_constraints': 'Preserve the source actor, condition, negation, modal strength and quoted musical/lyric text; audit the previously recorded static finding.',
                  'injection': None})
for case in read(PREVIOUS / 'validation_cases.json')['cases']:
    if case['id'] in {'benign-14-ko','fresh-benign-05-ko'}:
        case = deepcopy(case)
        case['id'] = 'recent-' + case['id']
        case['meaning_constraints'] = 'Do not guarantee no breath shortage; do not transfer the singer’s rhythmic freedom to the accompanist.'
        cases.append(case)
for case in read(PREVIOUS / 'dev_cases.json')['cases']:
    if case['id'] in {'fixed-observed-p-pp-1','fixed-observed-p-pp-2'}:
        case = deepcopy(case)
        case['meaning_constraints'] = 'Describe the two accounts and shared soft-entry advice; the cause of p versus PP is unknown. Generic notation variability does not establish a cause.'
        cases.append(case)
assert len(cases) == 41 and len({c['id'] for c in cases}) == 41
for c in cases:
    for i in c.get('gold_evidence_ids', []):
        assert corpus[i]['piece'] == c['piece_id']
random.Random(20261031).shuffle(cases)
write('cases.json', {'suite_type': 'authored_meaning_fidelity_diagnostic', 'cases': cases})
write('historical_examples.json', historical)
write('protocol.json', {
 'date': '2026-10-31', 'status': 'frozen_before_inference',
 'purpose': 'Check exact recorded high-risk meaning changes, not estimate ordinary-question quality or attack success.',
 'baseline': '29fe3e2 current secured pre-refinement pipeline',
 'refined': '79cbd7b, production source identical to Candidate F',
 'inventory': {'historical_questions': 28, 'static_english_questions': 9, 'recent_korean_failures': 2, 'fixed_real_source_conflicts': 2},
 'requests_per_variant': 41, 'repetitions': 1, 'schedule_seed': 20261031,
 'controls': 'Same corpus/checkpoints/settings as the prior comparison. Serial workers, one warmup. Fixed-source cases bypass retrieval and remain separately labeled. Existing live service remains untouched.',
 'review': 'Direct Codex semantic review of delivered prose against actual evidence, with source-role, negation, condition, certainty, cause and lyric/note semantics recorded. Targeted and not independently blinded; no aggregate production accuracy claim.',
 'historical_note': 'October 16 saved outputs provide historical context only. They are not controlled counterfactuals for the current corpus; paired new baseline/refined runs provide the controlled comparison.',
 'corpus_sha256': digest(ROOT / 'data/corpus.json'), 'cases_sha256': digest(OUT / 'cases.json'),
 'historical_artifact_sha256': digest(ROOT / 'evaluation/qualitative.json'),
})
print('Frozen 41 cases: 39 service requests and 2 fixed-evidence source-conflict probes per variant.')
