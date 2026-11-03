"""Judge completed case batches in completion order, never selected by quality."""
from pathlib import Path
import json, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def save(p,x):
    text=json.dumps(x,ensure_ascii=False,indent=2)+'\n'
    if p.exists() and p.read_text()!=text:raise ValueError('Frozen batch changed')
    p.write_text(text)
def main():
    cases=read(ROOT/'cases.json')['cases'];index={c['id']:c for c in cases};offset=0
    while offset<len(cases):
        p=ROOT/'results.jsonl'
        lines=p.read_text().splitlines(keepends=True) if p.exists() else []
        rows=[json.loads(x) for x in lines if x.endswith('\n')]
        count=min(20,len(cases)-offset)
        if len(rows)<offset+count:time.sleep(10);continue
        picked=rows[offset:offset+count];batch=ROOT/'judge-batches'/f'{offset//20:03d}';batch.mkdir(parents=True,exist_ok=True)
        save(batch/'cases.json',{'cases':[index[r['case_id']] for r in picked]})
        variants=sorted({o['variant'] for r in picked for o in r['outputs']});paths=[]
        for v in variants:
            out=batch/(v+'.jsonl');text=''.join(json.dumps(o,ensure_ascii=False)+'\n' for r in picked for o in r['outputs'] if o['variant']==v)
            if out.exists() and out.read_text()!=text:raise ValueError('Batch outputs changed')
            out.write_text(text);paths.append(str(out))
        with (batch/'run.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'prepare_expanded_judge_packets.py'),'--cases',str(batch/'cases.json'),'--results',*paths,'--output',str(batch)],stdout=log,stderr=subprocess.STDOUT,check=True)
            previous=[]
            for d in sorted((ROOT/'judge-batches').iterdir()):
                if d>=batch:continue
                for candidate in [d,d/'fresh']:
                    if (candidate/'aggregation.json').exists():previous.append(str(candidate))
            command=[sys.executable,str(ROOT/('run_reused_expanded_judge.py' if previous else 'run_joint_expanded_judge.py')),'--directory',str(batch),'--workers','3','--model','claude-opus-5-5']
            if previous:command+=['--reuse-directory',*previous]
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        offset+=count;print(json.dumps({'judged_cases':offset,'total':len(cases)}),flush=True)
    print('Complete',flush=True)
if __name__=='__main__':main()
