"""Supplement the semantic comparison with the already frozen expert BERTScore setting.

No setting search, answer editing, quote insertion, or reranking occurs here.
Run only after generation and Claude evaluation complete.
"""
from pathlib import Path
import hashlib, json, subprocess, sys

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system/evaluation')
VARIANTS=['g0s0q0','security_only','validation_only','g0s1q0','g0s0q1','warning','helpful']
def read(p):return json.loads(p.read_text())
def write(p,data):
    text=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if p.exists() and p.read_text()!=text:raise ValueError('Frozen metric input changed: '+str(p))
    p.write_text(text)

def main():
    summary=read(ROOT/'summary.json')
    assert summary['contract']['questions']==233 and summary['contract']['outputs']==2796
    target=ROOT/'fixed-bertscore';target.mkdir(exist_ok=True)
    cases=[c for c in read(ROOT/'cases.json')['cases'] if c['cohort']=='expert105']
    assert len(cases)==105
    write(target/'cases.json',{'cases':cases})
    settings=SOURCE/'ku_background_2026-10-31/bertscore-development-b/selected_settings.json'
    write(target/'selected_settings.json',read(settings))
    write(target/'metric-length-amendment.json',read(SOURCE/'expanded_460_2026-11-02/metric-length-amendment.json'))
    write(target/'PROTOCOL.json',{
        'scope':'Supplementary diagnostic only; no optimization or production quality claim.',
        'expert_cases':105,'variants':VARIANTS,
        'setting_origin':str(settings),'setting_sha256':hashlib.sha256(settings.read_bytes()).hexdigest(),
        'text':'Full displayed text including warnings; no trimming, chunking, substitution or reference editing.',
        'refusals':'Original pipeline unavailable OR Claude refusal receives zero. Metric input retains the original visible text and generation_mode separately.',
        'length':'Overlength full answers are unscorable and retained as zero contributions; do not call these computed BERTScores.',
        'interpretation':'Raw encoder/layer-dependent similarity, not a calibrated correctness probability. References derive from the same expert KU corpus; this is not an independent holdout.'})
    assessments=read(ROOT/'assessments-by-variant.json');combined={}
    for variant in VARIANTS:
        rows=[json.loads(line) for line in (ROOT/(variant+'.jsonl')).read_text().splitlines()]
        indexed={r['case_id']:r for r in rows};prepared=[]
        for c in cases:
            row=indexed[c['id']]
            row['metric_original_generation_mode']=row['response']['generation_mode']
            row['metric_claude_refusal']=assessments[variant][c['id']]['refusal']
            if row['metric_claude_refusal']:row['response']['generation_mode']='unavailable'
            prepared.append(row)
        result=target/(variant+'.jsonl')
        text=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in prepared)
        if result.exists() and result.read_text()!=text:raise ValueError('Metric results changed')
        result.write_text(text)
        subprocess.run([sys.executable,str(ROOT/'score_expanded_bertscore_bounded.py'),
                        '--cases',str(target/'cases.json'),'--results',str(result),
                        '--settings',str(target/'selected_settings.json'),'--split','all',
                        '--output',str(target/variant),'--device','cuda'],check=True)
        combined[variant]=read(target/variant/'summary.json')
    write(target/'comparison.json',combined)
    lines=['# 고정 설정의 전문가 105문항 BERTScore','',
           '기존 설정(xlm-roberta-large, layer 18, IDF 없음, raw F1)을 그대로 사용했다. 설정 탐색이나 답변 수정은 하지 않았다.',
           '거절과 길이 초과는 유효 평균에 0으로 남기고, 길이 초과에는 계산된 점수가 있다고 주장하지 않는다.',
           '원문 유사도는 답변의 정확성·자연스러움을 보장하지 않는다. 전체 표시 문장과 경고를 평가했다.','',
           '| 조건 | 유효 F1 /105 | 실제 계산한 답변 | 계산한 답변만 F1 | 길이 초과 |',
           '|---|---:|---:|---:|---:|']
    for v,items in combined.items():
        s=items[0];conditional=s['conditional_scorable']['f1']
        lines.append(f"| {v} | {s['effective']['f1']:.6f} | {s['scored']} | {conditional:.6f} | {s['unscorable_overlength_cases']} |")
    lines+=['','105문항은 전문가 KU를 기준으로 구성한 기존 진단 문항이다. 전문가가 새로 판정한 독립 검증 집합은 아니다.',
            '같은 corpus의 정답 문장과 비교하므로 복사·분량의 영향을 의미 품질과 분리해서 봐야 한다.',
            '','[실제 답변 및 Claude 의미 평가](../report.md) · [고정 평가 계약](PROTOCOL.json)']
    (target/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
