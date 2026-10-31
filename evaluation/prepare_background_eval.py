"""Freeze expert references and source-group splits before candidate evaluation."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/ku_background_2026-10-31'


def read(path):
    return json.loads(path.read_text())


def freeze(name, data):
    path = OUT / name
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen artifact differs: ' + str(path))
    path.write_text(text)


def main():
    OUT.mkdir(exist_ok=True)
    corpus = {r['id']: r for r in read(ROOT / 'data/corpus.json')}
    inventory = read(ROOT / 'evaluation/qualitative.json')['questions']
    by_piece = defaultdict(list)
    for group in inventory:
        by_piece[group['piece_id']].append(group['source_id'])
    rng = random.Random(20261031)
    development = set()
    for piece in sorted(by_piece):
        ids = sorted(by_piece[piece]); rng.shuffle(ids)
        development.update(ids[:round(len(ids) * .4)])
    expert = []
    for group in inventory:
        split = 'development' if group['source_id'] in development else 'validation'
        for run in group['inference_runs']:
            req = run['inference_input']
            ids = list(dict.fromkeys(x['knowledge_unit_id'] for x in req['case_reference_authority']['items']))
            assert ids and all(corpus[i]['evidence_type'] == 'expert_annotation' for i in ids)
            assert all(corpus[i]['piece'] == req['piece_id'] for i in ids)
            reference = '\n'.join(corpus[i]['answer'].strip() for i in ids)
            expert.append({'id': 'expert-' + run['case_id'], 'kind': 'benign', 'language': 'ko',
                          'pair_id': group['source_id'], 'piece_id': req['piece_id'],
                          'question': req['question'], 'measure_range': req['measure_range'],
                          'category': 'expert_reference', 'split': split,
                          'gold_evidence_ids': ids, 'reference_answers': [reference],
                          'reference_policy': 'all applicable finalized expert KUs, concatenated in frozen order',
                          'injection': None})
    assert len(expert) == 105
    # Author general/background questions independently of generated outputs.
    concepts = [
        ('legato', '레가토와 포르타멘토는 어떤 차이가 있나요?',
         'What is the difference between legato and portamento?',
         'Connected articulation is not the same as an audible pitch slide.', None),
        ('dynamics', '크레센도와 디미누엔도는 무슨 뜻인가요?',
         'What do crescendo and diminuendo mean?',
         'Gradually louder versus gradually softer; do not equate dynamics with tempo.', None),
        ('duration', '2분음표는 4분음표보다 얼마나 긴가요? 항상 2초인가요?',
         'How long is a half note compared with a quarter note? Is it always two seconds?',
         'Twice a quarter note; seconds depend on tempo, not a fixed two seconds.', None),
        ('meter', '6/8박자와 3/4박자는 어떻게 다른가요?',
         'How do 6/8 and 3/4 differ?',
         'Typical grouping: two dotted-quarter beats versus three quarter-note beats; distinguish grouping from total duration.', None),
        ('fermata', '페르마타는 언제나 정확히 세 배 길이로 불러야 하나요?',
         'Must a fermata always last exactly three times the written duration?',
         'No universal exact multiplier; do not universalize the particular expert rule.', None),
        ('transpose', '곡을 조옮김하는 것과 몇 음만 옥타브 내려 부르는 것은 같은 건가요?',
         'Is transposing a song the same as lowering just a few notes by an octave?',
         'Transposition changes pitches consistently by an interval; changing selected notes is different.', None),
        ('slow_practice', '빠른 가사를 익히려고 느리게 연습할 때 어떤 점을 확인하면 좋을까요?',
         'What should I listen for when practicing rapid lyrics slowly?',
         'Optional general practice may supplement applicable diction advice; preserve consonant-omission exercises.', None),
        ('breath', '레가토 연습을 위한 일반적인 방법도 알려 주세요. 제 호흡 상태는 아직 모르시죠?',
         'Could you suggest a general legato exercise, without assuming you know my breathing ability?',
         'General optional exercise, no personal diagnosis or guaranteed breath capacity.', None),
        ('why_unknown', '이 악보의 103마디를 작곡가가 왜 이렇게 썼는지 확정할 수 있나요?',
         'Can you establish exactly why the composer wrote bar 103 this way?',
         'Do not invent a specific composer intention; explain evidence limit.', [103, 103]),
        ('pitch_unknown', '제가 보고 있는 판본의 103마디 첫 음 높이를 알려 주세요.',
         'What is the first pitch in bar 103 of the edition I am looking at?',
         'Unknown edition and no verified pitch evidence: do not invent a pitch.', [103, 103]),
        ('glossary', '스타카토, 테누토, 악센트의 차이를 간단히 설명해 주세요.',
         'Briefly explain the difference between staccato, tenuto, and an accent.',
         'Detached articulation, sustained/full value, emphasis; do not claim markings in selected score.', None),
        ('rubato', '루바토는 박자나 리듬을 전부 무시해도 된다는 뜻인가요?',
         'Does rubato mean I can ignore all meter and rhythm?',
         'Expressive flexibility, not unrestricted abandonment of rhythm; preserve source qualifications.', None),
    ]
    background = []
    for idx, (name, ko, en, constraint, measures) in enumerate(concepts):
        for language, question in [('ko', ko), ('en', en)]:
            background.append({'id': f'background-{name}-{language}', 'kind': 'benign',
                               'pair_id': 'background-' + name, 'piece_id': 'nella-fantasia',
                               'question': question, 'language': language, 'measure_range': measures,
                               'category': 'background_knowledge',
                               'split': 'development' if idx % 2 == 0 else 'validation',
                               'answerability': 'unknown_local_fact' if 'unknown' in name else 'general_music',
                               'meaning_constraints': constraint, 'injection': None})
    broad = deepcopy(read(ROOT / 'evaluation/quality_refinement_2026-10-31/validation_cases.json')['cases'])
    meaning = deepcopy(read(ROOT / 'evaluation/meaning_fidelity_2026-10-31/cases.json')['cases'])
    all_cases = expert + background + broad + meaning
    assert len({x['id'] for x in all_cases}) == len(all_cases)
    rng.shuffle(all_cases)
    dev = [x for x in all_cases if x.get('split') == 'development']
    # A small first probe, chosen before generation: one expert case per source,
    # plus general and unknown-score requests in both languages.
    probe = []; seen = set()
    for case in dev:
        if case['category'] == 'expert_reference' and case['pair_id'] not in seen:
            probe.append(case); seen.add(case['pair_id'])
            if len(probe) == 12:
                break
    probe += [c for c in background if c['split'] == 'development']
    for name, cases in [('expert_cases.json', expert), ('development_cases.json', dev),
                        ('probe_cases.json', probe), ('all_cases.json', all_cases)]:
        freeze(name, {'suite_type': 'bounded_background_evaluation', 'cases': cases})
    freeze('inventory.json', {'seed': 20261031, 'expert_cases': len(expert),
                             'expert_source_groups': len(inventory),
                             'expert_by_split': dict(Counter(c['split'] for c in expert)),
                             'expert_by_piece': dict(Counter(c['piece_id'] for c in expert)),
                             'background_cases': len(background), 'security_and_normal_cases': len(broad),
                             'meaning_cases': len(meaning), 'total_cases': len(all_cases),
                             'corpus_sha256': hashlib.sha256((ROOT / 'data/corpus.json').read_bytes()).hexdigest(),
                             'reference_sha256': hashlib.sha256(json.dumps(expert, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
    print(read(OUT / 'inventory.json'))


if __name__ == '__main__':
    main()
