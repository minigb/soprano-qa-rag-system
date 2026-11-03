"""Audit the held-out generated-text pattern screen without generating or changing answers."""
import json
import run

def main():
    runner=run.Runner.__new__(run.Runner)
    cases={c['id']:c for c in run.read(run.ROOT/'cases.json')['cases']}
    completed=run.rows(run.ROOT/'results.jsonl');assert len(completed)==233
    rejected=[];new_combined=[];new_security_only=[]
    for batch in completed:
        for o in batch['outputs']:
            if o['variant'] not in ['g0s0q0','g0s1q0','g0s0q1','g0s1q1']:continue
            c=cases[o['case_id']];kept=set(o['diagnostic']['selected_ids'])
            pool=[p for p in c['fixed_evidence'] if p['record']['id'] in kept]
            try:run.security.screen_generated_answer(o['response']['answer'],runner.payload(c,pool))
            except run.security.SecurityViolation as e:
                record={'case_id':o['case_id'],'variant':o['variant'],'reason':e.reason}
                rejected.append(record)
                combined=next(x for x in batch['outputs'] if x['variant']=='g1'+o['variant'][2:])
                if combined['response']['generation_mode']!='unavailable':new_combined.append(record)
                if o['variant']=='g0s0q0' and not any(r!='answer_validation_failed' for r in combined['diagnostic']['rejections']):
                    new_security_only.append(record)
    result={'scope':'Post-run deterministic screen of unchanged displayed outputs; no new generation, judging or score changes.',
            'original_design':'G applied static input/source screens and semantic output review. The production generated-text regex screen was not in the frozen inference runner.',
            'rejected_saved_outputs':rejected,'additional_combined_withholding':new_combined,
            'additional_security_only_withholding':new_security_only,
            'interpretation':'Zero additional withholding means this omitted screen would not change these observed delivery decisions. This still is not a full-service replay.'}
    (run.ROOT/'static-output-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'screen_rejections':len(rejected),'additional_combined':len(new_combined),'additional_security_only':len(new_security_only)}))

if __name__=='__main__':main()
