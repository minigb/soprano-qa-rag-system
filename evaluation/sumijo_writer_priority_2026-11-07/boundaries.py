"""Six synthetic scope checks, kept separate from the expert benchmark.

These fictional passages test metadata-based routing; they are not statements
by Sumi Jo or any real expert and are never inserted into the corpus or demo.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from soprano_qa import single_pass, vertex_gemini
from soprano_qa.retrieval import SearchResult
from soprano_qa.service import _result_to_evidence
from run import ROOT, freeze, read, sha

FIXTURES = {
    'same_scope': {
        'ko': ('가상 연습곡의 첫 소리는 어떤 다이나믹으로 시작할까요?',
               '가상 연습곡의 첫 소리는 부드럽고 여리게 시작하도록 권합니다.',
               '가상 연습곡의 첫 소리는 강하게 시작하도록 권합니다.',
               '동일 범위의 상충하는 권고에서는 조수미 유형 자료를 우선 참고하여 부드럽고 여리게 시작하도록 안내한다. 이를 실제 악보 표기로 확정하거나 다른 설명을 객관적 오류로 단정하지 않는다.'),
        'en': ('How should I begin the first sung note in this fictional practice piece?',
               'Begin the first sung note of this fictional practice piece softly and gently.',
               'Begin the first sung note of this fictional practice piece strongly.',
               'Prefer the Sumi Jo-type passage for this same-scope conflict: start softly and gently. Do not claim to have verified the score or proven the other source wrong.')},
    'different_condition': {
        'ko': ('마이크 없이 큰 홀에서 이 가상 연습곡을 부를 때 어떤 발성을 쓸까요?',
               '마이크를 사용하는 작은 공간에서는 자연스럽고 친밀한 음색을 사용할 수 있습니다.',
               '마이크 없이 큰 홀에서 부를 때에는 충분한 공명과 호흡 지지를 바탕으로 한 성악적 발성이 필요할 수 있습니다.',
               '질문의 조건은 마이크가 없는 큰 홀이다. 이 조건에 맞는 공명과 호흡 지지의 성악적 발성을 설명하되 가능성을 의무로 바꾸지 않는다. 마이크가 있는 작은 공간의 조언을 덮어 적용하지 않는다.'),
        'en': ('What vocal approach should I use for this fictional piece in a large hall without a microphone?',
               'In a small space with a microphone, you may use a natural, intimate vocal tone.',
               'In a large hall without a microphone, classical vocal production with sufficient resonance and breath support may be needed.',
               'Use the large-hall, no-microphone guidance, retaining may be needed. Do not override it with the Sumi Jo-type advice about a different performance condition.')},
    'different_actor': {
        'ko': ('가상 연습곡의 성악부 첫 소리는 어떤 다이나믹으로 부르라고 하나요?',
               '이 가상 연습곡에서 피아니스트는 반주부의 첫 소리를 pp로 연주하도록 권합니다.',
               '이 가상 연습곡에서 성악가는 성악부의 첫 소리를 f로 부르도록 권합니다.',
               '질문은 성악부이므로 성악가의 첫 소리를 f로 부르라는 권고를 설명한다. 반주부의 pp를 성악부의 지시로 옮기지 않는다.'),
        'en': ('What dynamic is recommended for the singer’s first note in this fictional practice piece?',
               'The pianist is advised to play the opening accompaniment pp in this fictional practice piece.',
               'The singer is advised to sing the first vocal note f in this fictional practice piece.',
               'Answer f for the singer. Do not transfer the accompaniment pp advice to the vocal part.')},
}


def main():
    root = ROOT / 'boundaries'; root.mkdir(exist_ok=True)
    cases, packets = [], []
    for name, languages in FIXTURES.items():
        for language, (question, sumi, expert, expectation) in languages.items():
            cid = name + '-' + language
            # Lower-priority source deliberately comes first to test more than list order.
            results = []
            for kind, text in [('expert_annotation', expert), ('sumijo_feedback', sumi)]:
                record = {'id': 'fictional-' + cid + '-' + kind, 'piece': 'nella-fantasia',
                          'work': 'Synthetic practice fixture — not an actual score or quotation',
                          'evidence_type': kind, 'answer': text, 'measure_range': [], 'measure_scope': 'global'}
                if kind == 'sumijo_feedback': record['sumijo_source'] = {'speaker': 'Sumi Jo'}
                results.append(SearchResult(record, 1, 1, 0, 1, 'general_evidence'))
            messages = single_pass.build_messages('nella-fantasia', '', question, results, language=language)
            evidence = [_result_to_evidence(r, measure_ranges=[]) for r in results]
            cases.append({'id': cid, 'kind': 'benign', 'language': language, 'question': question,
                          'piece_id': 'nella-fantasia', 'measure_range': None, 'reference_answers': [expectation],
                          'reference_authority': {'status': 'synthetic_policy_fixture_not_expert_annotation'}})
            packets.append({'case_id': cid, 'messages': messages, 'evidence': evidence})
    freeze(root / 'cases.json', {'cases': cases})
    freeze(root / 'writer-inputs.json', packets)
    config = read(ROOT / 'writer-settings.json')
    freeze(root / 'contract.json', {'maximum_requests': 6, 'model': config['vertex_gemini'],
           'purpose': 'Synthetic, held separate from the existing expert benchmark and BERTScore',
           'inputs_sha256': sha(root / 'writer-inputs.json'), 'runner_sha256': sha(Path(__file__))})
    outputs = []; next_start = 0.
    for p in packets:
        saved = root / (p['case_id'] + '.json'); pending = root / (p['case_id'] + '.pending')
        if saved.exists(): row = read(saved)
        else:
            assert not pending.exists(), 'Unobserved request'
            delay = max(0, next_start - time.monotonic())
            if delay: time.sleep(delay)
            next_start = time.monotonic() + 10
            freeze(pending, {'case_id': p['case_id']}); started = time.perf_counter()
            text = vertex_gemini.generate(p['messages'], config)
            row = {'case_id': p['case_id'], 'variant': 'synthetic-boundary', 'status': 'response',
                   'elapsed_seconds': time.perf_counter() - started,
                   'response': {'answer': str(text), 'evidence': p['evidence'], 'generation_mode': 'llm_single_pass',
                                'generation_metadata': text.generation_metadata}}
            freeze(saved, row); pending.unlink()
        outputs.append(row)
        (root / 'current.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in outputs))
        print(json.dumps({'case_id': p['case_id'], 'status': row['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
