"""Show saved Gemini answers beside the unchanged pre-security originals.

Export only the 109 music cases with both outputs. No API calls or new scores.
Keep all 460-case historical datasets and browser note namespaces intact.
"""
from pathlib import Path
import copy
import hashlib
import json
import statistics

SITE = Path(__file__).resolve().parent
REPO = SITE.parents[1]
PRIMARY = REPO / 'evaluation/gemini_flash_writer_2026-11-07'
RECOVERY = REPO / 'evaluation/gemini_flash_recovery_2026-11-07'
ID = 'gemini-20261107'


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
    temp.replace(p)


def references(items):
    return [{k: r.get(k) for k in ['knowledge_unit_id', 'text', 'measure_ranges', 'reference_role']} for r in items]


def summarize(cases, version):
    result = {}
    for name, selected in [('all', cases), ('original', [c for c in cases if c['family'] == 'original'])]:
        versions = [c['versions'][version] for c in selected]
        result[name] = {'cases': len(selected),
                        'material_drifts': sum(v['judge']['source_fidelity'] == 'material_drift' for v in versions),
                        'naturalness_mean_effective_out_of_4': statistics.mean(v['judge']['naturalness'] if not v['judge']['refusal'] else 0 for v in versions),
                        'bertscore_effective_f1': statistics.mean(v['bertscore']['f1'] if v['bertscore']['scored'] else 0 for v in versions),
                        'latency_all': {'median_seconds': statistics.median(v['latency_seconds'] for v in versions)}}
    return result


def main():
    assert read(RECOVERY / 'verification.json')['verified']
    history = read(SITE / 'dist/data.json')
    old_cases = {c['id']: c for c in read(REPO / 'evaluation/expanded_460_2026-11-02/cases.json')['cases']}
    current_cases = {c['id']: c for c in read(PRIMARY / 'expert/cases.json')['cases']}
    outputs = {r['case_id']: r for r in rows(RECOVERY / 'combined/expert.jsonl')}
    judged = read(RECOVERY / 'combined/expert-judgments.json')
    metric = read(RECOVERY / 'combined/expert-bertscore.json')
    scored = {r['id']: r for r in metric['cases']}
    first = {r['case_id']: r for r in rows(PRIMARY / 'expert/current.jsonl')}
    previous = read(SITE / 'dist/data-helpful-14b-20261104.json')
    previous_refusals = {c['id'] for c in previous['cases'] if c['versions']['current'].get('previous_refusal')}
    assert len(outputs) == len(current_cases) == len(judged) == len(scored) == 109
    exported = []
    for historical in history['cases']:
        cid = historical['id']
        if cid not in outputs: continue
        case = current_cases[cid]; row = outputs[cid]; response = row['response']; m = scored[cid]
        assert row['status'] == 'response'
        assert response['generation_metadata']['model_version'] == 'gemini-3.8-flash'
        assert all(case[k] == historical[k] for k in ['question', 'piece_id', 'measure_range', 'family'])
        item = copy.deepcopy(historical)
        item['versions']['current'] = {
            'answer': response['answer'], 'status': row['status'], 'generation_mode': response['generation_mode'],
            'latency_seconds': row['elapsed_seconds'], 'technical_retry': row.get('attempt_number') == 2,
            'first_attempt_error': first[cid]['status'] == 'exception',
            'previous_refusal': cid in previous_refusals,
            'evidence': [{k: e.get(k) for k in ['id', 'text', 'measure_ranges', 'evidence_type', 'generation_role']}
                         for e in response['evidence']],
            'bertscore': {'scored': m['scored'], 'f1': m['f1'] if m['scored'] else None,
                          'reason': m['unscorable_reason'], 'candidate_tokens': m['candidate_tokens']},
            'judge': {'available': True, **{k: judged[cid][k] for k in historical['versions']['original']['judge']}},
            'secondary_note': None}
        item['evaluation_references'] = {
            'original': {'sources': historical['references'], 'bertscore_text': old_cases[cid]['reference_answers'][0]},
            'current': {'sources': references(case['reference_authority']['items']), 'bertscore_text': case['reference_answers'][0]}}
        exported.append(item)
    assert len(exported) == 109
    data = {'schema_version': 2, 'evaluation_id': ID, 'evaluation_date': '2026-11-07',
            'judge_model': 'Claude Opus 5.5', 'assessment_status': 'complete', 'pieces': history['pieces'],
            'cases': exported, 'summary': {v: summarize(exported, v) for v in ['original', 'current']},
            'naturalness_comparison': {'losses': sum(c['versions']['current']['judge']['naturalness'] < c['versions']['original']['judge']['naturalness'] for c in exported)},
            'coverage': {'current_music_cases': 109, 'historical_music_cases': 460, 'not_run_with_gemini': 351,
                         'expert_cases': 105, 'additional_regression_cases': 4, 'recovered_music_cases': 6},
            'source_hashes': {str(p.relative_to(REPO)): sha(p) for p in [SITE / 'dist/data.json',
                SITE / 'dist/data-helpful-14b-20261104.json',
                REPO / 'evaluation/expanded_460_2026-11-02/cases.json',
                PRIMARY / 'expert/cases.json', PRIMARY / 'expert/current.jsonl', RECOVERY / 'combined/expert.jsonl',
                RECOVERY / 'combined/expert-judgments.json', RECOVERY / 'combined/expert-bertscore.json',
                RECOVERY / 'verification.json']}}
    assert abs(data['summary']['current']['original']['bertscore_effective_f1'] - metric['summary']['expert105_after_retry']) < 1e-12
    save(SITE / 'dist' / ('data-'+ID+'.json'), data)
    entry = {'id': ID, 'label': '최신 Gemini · 원본과 109문항 비교', 'shortLabel': 'Gemini',
             'answerLabel': '최신 시스템 · Gemini', 'answerSubtitle': 'Gemini3.8Flash · 생성 답변',
             'featuredCase': 'expert-kim-in-flowery-clouds-02__m11-12', 'data': 'data-'+ID+'.json',
             'storageKey': 'soprano-answer-review:'+ID+':v1', 'date': '2026.11.07', 'caseCount': 109,
             'metricNote': '전문가 기반 105문항 · 버전별 당시 참조로 계산',
             'latencyComparable': False, 'latencyTitle': '시간 중앙값 · 측정 범위 다름',
             'latencyNote': '원본 전체 응답 / Gemini 생성만 · 직접 속도 비교 불가',
             'latencyLabels': {'original': '전체 응답', 'current': 'Gemini 생성 호출'},
             'footer': '최신 결과 109문항 · 실제 답변 218개 · 이전 460문항은 버전 목록에서 확인',
             'note': '왼쪽은 보안 적용 전 원본, 오른쪽은 최신 Gemini의 실제 생성 답변입니다. 최신 결과가 있는 109문항(전문가 기반 105 + 추가 4)만 비교합니다. API 오류 6건은 같은 입력으로 1회 재시도해 확보했습니다. 나머지 351문항은 Gemini로 실행하지 않았으며, 이전 460개 전체 결과는 버전 목록에서 볼 수 있습니다.',
             'methodIntro': '보안 적용 전 원본과 최신 Gemini 결과가 모두 있는 109문항을 비교합니다. 원본의 답변·근거·점수는 보존했고, Gemini 답변도 수정하거나 줄이지 않았습니다. 왼쪽은 초기 시스템의 결과이며 최신 14B 후보 J와 구분됩니다. 생성 API 오류 6건은 1회 재시도 결과와 함께 재시도 여부를 표시합니다.',
             'methodBert': '오른쪽은 기존 XLM-R-large, 18층, noIDF, raw F1 설정의 전체 답변 점수입니다. 원본 점수는 당시 평가 그대로입니다. 버전 사이에 정리된 참조가 달라진 문항이 있으므로 아래에서 양쪽 평가 참조와 실제 BERTScore 참조문을 구분해 확인하세요. 재시도 후 전문가 105문항 점수는 0.950992이며, 최초 API 실패를 포함한 0.897379는 별도 기록으로 보존했습니다.',
             'methodInventory': '현재 선택한 버전은 기존 460개 중 Gemini 답변이 있는 109개이며, 전문가 기반 105개와 이전 거절 사례의 추가 변형 4개입니다. 한국어·영어 공격 74개는 별도 평가이며 이 원본 음악 답변 비교에는 포함하지 않았습니다. 기존 개발 문항으로 독립적인 새 검증 집합은 아닙니다.',
             'methodScope': '원본과 최신 시스템은 프롬프트·자료·검색·생성 모델이 달라질 수 있어 전체 결과 비교로 해석합니다. Gemini는 후보 J의 검색 결과를 고정하고 생성했으며, 원본과 같은 검색 결과로 실행한 모델만의 비교는 아닙니다. 원본 시간은 검색 등을 포함한 전체 응답, Gemini 시간은 생성 API 호출만입니다. API 부하·실행 시점도 달라 직접 속도 비교에 사용할 수 없습니다.'}
    catalog = read(SITE / 'dist/evaluations.json')
    catalog['evaluations'] = [entry] + [e for e in catalog['evaluations'] if e['id'] != ID]
    catalog['default'] = ID
    save(SITE / 'dist/evaluations.json', catalog)
    print(json.dumps({'cases': 109, 'answers': 218, 'original_preserved': True, 'api_calls': 0, 'default': ID}))


if __name__ == '__main__':
    main()
