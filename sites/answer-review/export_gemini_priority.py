"""Export the latest writer revision while preserving previous demo datasets."""
from pathlib import Path
import copy
import json
from export_gemini import read, rows, sha, save, summarize

SITE = Path(__file__).resolve().parent
REPO = SITE.parents[1]
RUN = REPO / 'evaluation/sumijo_writer_priority_2026-11-07'
FINAL = RUN / 'v3'
ID = 'gemini-priority-20261107'


def main():
    assert read(RUN / 'verification.json')['verified']
    previous = read(SITE / 'dist/data-gemini-20261107.json')
    data = copy.deepcopy(previous)
    changed = {r['case_id']: r for r in rows(FINAL / 'after.jsonl')}
    judges = read(FINAL / 'combined/expert-judgments.json')
    scores = {r['id']: r for r in read(FINAL / 'combined/expert-bertscore.json')['cases']}
    for case in data['cases']:
        cid = case['id']
        if cid not in changed: continue
        row = changed[cid]; response = row['response']; metric = scores[cid]
        version = case['versions']['current']
        assert version['evidence'] == [{k: e.get(k) for k in ['id', 'text', 'measure_ranges', 'evidence_type', 'generation_role']}
                                        for e in response['evidence']]
        version.update(answer=response['answer'], latency_seconds=row['elapsed_seconds'],
                       technical_retry=False, first_attempt_error=False, writer_priority_updated=True,
                       judge={'available': True, **{k: judges[cid][k] for k in case['versions']['original']['judge']}},
                       bertscore={'scored': metric['scored'], 'f1': metric['f1'] if metric['scored'] else None,
                                  'reason': metric['unscorable_reason'], 'candidate_tokens': metric['candidate_tokens']})
    assert len(changed) == 8
    data['evaluation_id'] = ID
    data['summary'] = {v: summarize(data['cases'], v) for v in ['original', 'current']}
    data['naturalness_comparison'] = {'losses': sum(c['versions']['current']['judge']['naturalness'] < c['versions']['original']['judge']['naturalness'] for c in data['cases'])}
    data['coverage'].update(writer_priority_updated_cases=8, unchanged_prompt_results_reused=101)
    data['source_hashes'].update({str(p.relative_to(REPO)): sha(p) for p in [
        SITE / 'dist/data-gemini-20261107.json', RUN / 'verification.json', FINAL / 'after.jsonl',
        FINAL / 'combined/expert-judgments.json', FINAL / 'combined/expert-bertscore.json']})
    assert abs(data['summary']['current']['original']['bertscore_effective_f1'] - read(RUN / 'summary.json')['expert105_bertscore']['after']) < 1e-12
    save(SITE / 'dist' / ('data-' + ID + '.json'), data)
    catalog = read(SITE / 'dist/evaluations.json')
    old = next(e for e in catalog['evaluations'] if e['id'] == 'gemini-20261107')
    entry = copy.deepcopy(old)
    entry.update(id=ID, label='최신 Gemini · 조수미 우선 참고 · 109문항',
                 answerLabel='최신 Gemini · 우선 참고 적용', data='data-' + ID + '.json',
                 storageKey='soprano-answer-review:' + ID + ':v1',
                 featuredCase='expert-kim-nella-fantasia-03__no-range',
                 note='왼쪽은 보안 적용 전 원본, 오른쪽은 조수미 우선 참고 지침을 적용한 Gemini 결과입니다. 기존 109문항 중 지침이 추가되는 8문항을 다시 생성했고, 입력이 동일한 101문항은 기존 답변·평가를 재사용했습니다. 이번 변경 문항 필터로 8건을 확인할 수 있습니다. 변경 전 Gemini와 이전 460문항 결과는 버전 목록에 보존했습니다.',
                 footer='109문항 · 8문항 재생성 / 101문항 동일 입력의 기존 결과 재사용',
                 methodIntro='보안 적용 전 원본과 최신 Gemini의 109문항을 비교합니다. 조수미 자료와 다른 출처가 함께 있는 8문항만 새 지침으로 다시 생성했고 실제 출력 전문을 표시합니다. 다른 101문항은 입력 프롬프트의 동일성을 확인해 기존 결과를 재사용했습니다. 왼쪽은 초기 시스템이며 우선 참고 지침 적용 직전 Gemini와는 다릅니다.',
                 methodBert='XLM-R-large, 18층, noIDF, raw F1 설정을 유지했습니다. 전문가 105문항 중 변경된 5문항을 새 답변으로 계산하고 동일 입력의 100문항 점수는 재사용했습니다. 전체 F1은 0.950608입니다. 원본의 참조 및 점수는 당시 기록 그대로이며, 버전별 평가 참조문을 아래에서 확인할 수 있습니다.',
                 methodInventory='기존 460개 중 Gemini 결과가 있는 109개(전문가 기반 105 + 추가 4)입니다. 이번에는 8개를 새로 생성했고 101개는 재사용했습니다. 별도의 가상 한영 경계 사례 6개는 이 화면과 전문가 점수에 포함하지 않습니다. 반복 사용한 개발 문항이며 독립적인 새 검증 집합은 아닙니다.')
    old['label'] = 'Gemini · 우선 참고 지침 적용 전 · 109문항'
    old['answerLabel'] = 'Gemini · 우선 참고 지침 적용 전'
    catalog['evaluations'] = [entry] + [e for e in catalog['evaluations'] if e['id'] != ID]
    catalog['default'] = ID
    save(SITE / 'dist/evaluations.json', catalog)
    # Independent comparison against the previous artifact, including exact original outputs.
    before = {c['id']: c for c in previous['cases']}
    for c in data['cases']:
        assert c['versions']['original'] == before[c['id']]['versions']['original']
        if c['id'] not in changed: assert c == before[c['id']]
        else: assert c['versions']['current']['answer'] == changed[c['id']]['response']['answer']
    print(json.dumps({'cases': 109, 'updated': 8, 'reused': 101, 'original_preserved': True, 'default': ID}))


if __name__ == '__main__': main()
