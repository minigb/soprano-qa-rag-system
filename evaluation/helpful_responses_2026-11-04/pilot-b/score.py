"""Score all latest outputs with the existing fixed BERTScore setting after judging."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def assessments():
    judgments={}
    for batch in sorted((ROOT/'judge-batches').iterdir()):
        rows={r['id']:r['assessment'] for r in map(json.loads,(batch/'assessments.jsonl').read_text().splitlines())}
        for mapping in read(batch/'mapping.json'):
            judgments[mapping['case_id']]=rows.get(mapping['review_id'])
    assert len(judgments)==460
    return judgments
def main():
    raw=[json.loads(x) for x in (ROOT/'current.jsonl').read_text().splitlines()];assert len(raw)==460
    assert read(ROOT/'completion.json')['result_sha256']==sha(ROOT/'current.jsonl')
    judged=assessments();prepared=[]
    for source in raw:
        row=copy.deepcopy(source);judgment=judged[row['case_id']]
        row['metric_claude_refusal']=bool(judgment and judgment['refusal'])
        if row.get('response'):
            row['metric_original_generation_mode']=row['response'].get('generation_mode')
            if row['metric_claude_refusal']:row['response']['generation_mode']='unavailable'
        prepared.append(row)
    result=ROOT/'metric-input.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared)
    if result.exists() and result.read_text()!=text:raise ValueError('Frozen metric inputs changed')
    result.write_text(text)
    subprocess.run([sys.executable,str(ROOT/'score_expanded_bertscore_bounded.py'),'--cases',str(ROOT/'cases.json'),
        '--results',str(result),'--settings',str(ROOT/'selected-metric-settings.json'),'--split','all',
        '--output',str(ROOT/'bertscore'),'--device','cuda'],check=True)
    (ROOT/'assessment-complete.json').write_text(json.dumps({'results_sha256':sha(ROOT/'current.jsonl'),
        'metric_input_sha256':sha(result),'cases':460,'judge':'claude-opus-5-5',
        'policy':'First valid Claude judgment; same rubric as historical demo. Fixed BERTScore setting, full displayed text; semantic refusals, unavailable outputs and unscorable lengths contribute zero.'},indent=2)+'\n')
if __name__=='__main__':main()
