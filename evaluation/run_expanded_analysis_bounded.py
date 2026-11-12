"""Wait for frozen generation, then score and judge without score-based reruns."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/expanded_460_2026-11-02'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((OUT/'config.json').read_text())
    script_names = ['score_expanded_bertscore_bounded.py', 'prepare_expanded_judge_packets.py',
                    'run_expanded_judge.py', 'run_joint_expanded_judge.py',
                    'summarize_expanded_comparison_bounded.py', Path(__file__).name]
    contract = {'config_sha256': sha(OUT/'config.json'), 'cases_sha256': sha(OUT/'cases.json'),
                'metric_settings_sha256': sha(ROOT/config['metric_settings']),
                'scripts': {n: sha(ROOT/'evaluation'/n) for n in script_names},
                'length_policy_sha256': sha(OUT/'metric-length-amendment.json'),
                'judge_rubric_sha256': sha(OUT/'judge-rubric.txt'),
                'judge_schema_sha256': sha(OUT/'judge-schema.json')}
    contract_path = OUT/'analysis-contract-v2.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Frozen analysis contract changed; do not reuse judgments')
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')

    def state(phase, **extra):
        value = {'phase': phase, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'production_candidate': 'E37', 'branch': 'codex/ku-background-knowledge',
                 'stop_on_usage_reset': False, 'notifier_used': False, **extra}
        temp=OUT/'analysis-state.tmp';temp.write_text(json.dumps(value,indent=2)+'\n')
        temp.replace(OUT/'analysis-state.json')
        print(json.dumps(value),flush=True)

    def ready():
        for system in config['systems']:
            p=OUT/(system['name']+'.jsonl')
            m=p.with_name(p.name+'.manifest.json')
            if not p.exists() or not m.exists():return False
            if len(p.read_text().splitlines()) != 460:return False
            manifest=json.loads(m.read_text())
            if manifest['sessions'][-1].get('status') not in ['complete','already_complete']:return False
        return True

    def run(label, command):
        for name, expected in contract['scripts'].items():
            if sha(ROOT/'evaluation'/name) != expected:
                raise ValueError('Analysis script changed after freeze: '+name)
        state(label)
        with (OUT/(label+'.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

    try:
        state('waiting_for_generation')
        while not ready():time.sleep(20)
        # A completed manifest is flushed just before the GPU process exits.
        time.sleep(5)
        for system in config['systems']:
            name=system['name']
            run('bertscore-'+name,[config['metric_python'],'-u','evaluation/score_expanded_bertscore_bounded.py',
                '--cases',config['cases'],'--results',str(OUT/(name+'.jsonl')),'--split','all',
                '--settings',config['metric_settings'],'--output',str(OUT/('bertscore-'+name))])
        run('prepare-judge',[sys.executable,'evaluation/prepare_expanded_judge_packets.py',
            '--cases',config['cases'],'--results',*[str(OUT/(s['name']+'.jsonl')) for s in config['systems']],
            '--output',str(OUT/'judge')])
        run('claude-judge',[config['generation_python'],'-u','evaluation/run_joint_expanded_judge.py',
            '--directory',str(OUT/'judge'),'--workers','3','--model','claude-opus-5-5'])
        run('summary',[sys.executable,'evaluation/summarize_expanded_comparison_bounded.py'])
        state('automated_evaluation_complete', next='Directly inspect flagged meaning/naturalness cases; write report and verify source/contracts.')
    except Exception as exc:
        state('error',error=type(exc).__name__+': '+str(exc))
        raise


if __name__ == '__main__':
    main()
