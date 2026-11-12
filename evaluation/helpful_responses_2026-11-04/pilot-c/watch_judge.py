"""Review each completed chunk using the same Claude rubric as the historical demo."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def save(p,x):
    text=json.dumps(x,ensure_ascii=False,indent=2)+'\n'
    if p.exists() and p.read_text()!=text:raise RuntimeError('Frozen judge input changed')
    p.write_text(text)
def main():
    cases=read(ROOT/'cases.json')['cases'];index={c['id']:c for c in cases};offset=0
    while offset<len(cases):
        lines=((ROOT/'current.jsonl').read_text() if (ROOT/'current.jsonl').exists() else '').splitlines(keepends=True)
        rows=[json.loads(x) for x in lines if x.endswith('\n')];count=min(20,len(cases)-offset)
        if len(rows)<offset+count:time.sleep(5);continue
        picked=rows[offset:offset+count];batch=ROOT/'judge-batches'/f'{offset//20:03d}';batch.mkdir(parents=True,exist_ok=True)
        save(batch/'cases.json',{'cases':[index[r['case_id']] for r in picked]})
        result=batch/'current.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in picked)
        if result.exists() and result.read_text()!=text:raise RuntimeError('Frozen outputs changed')
        result.write_text(text)
        with (batch/'run.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'prepare_expanded_judge_packets.py'),'--cases',str(batch/'cases.json'),
                '--results',str(result),'--output',str(batch)],stdout=log,stderr=subprocess.STDOUT,check=True)
            previous=[]
            for d in sorted((ROOT/'judge-batches').iterdir()):
                if d>=batch:continue
                for p in [d,d/'fresh']:
                    if (p/'aggregation.json').exists():previous.append(str(p))
            command=[sys.executable,str(ROOT/('run_reused_expanded_judge.py' if previous else 'run_joint_expanded_judge.py')),
                     '--directory',str(batch),'--workers','3','--model','claude-opus-5-5']
            if previous:command+=['--reuse-directory',*previous]
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        offset+=count;print(json.dumps({'judged':offset,'total':len(cases)}),flush=True)
    print('Complete',flush=True)
if __name__=='__main__':main()
