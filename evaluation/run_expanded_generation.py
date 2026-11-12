"""Resume two complete, frozen460-case runs without changing production code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/expanded_460_2026-11-02'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((OUT/'config.json').read_text())
    scripts = [Path(__file__), ROOT/'evaluation/run_refinement_probe.py', ROOT/'evaluation/run_security_comparison.py']
    contract = {'config_sha256': sha(OUT/'config.json'), 'cases_sha256': sha(ROOT/config['cases']),
                'scripts': {p.name: sha(p) for p in scripts}}
    path = OUT/'generation-contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Frozen generation contract changed')
    path.write_text(json.dumps(contract, indent=2)+'\n')
    for system in config['systems']:
        name = system['name']
        assert sha(ROOT/system['settings']) == system['settings_sha256']
        telemetry = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader'], capture_output=True, text=True)
        with (OUT/'gpu-observations.jsonl').open('a') as stream:
            stream.write(json.dumps({'before_system': name, 'time': datetime.now(timezone.utc).isoformat(),
                                     'returncode': telemetry.returncode, 'processes': telemetry.stdout})+'\n')
        print(json.dumps({'event': 'start', 'system': name, 'time': datetime.now(timezone.utc).isoformat()}), flush=True)
        with (OUT/(name+'.log')).open('a') as log:
            subprocess.run([config['generation_python'], '-u', 'evaluation/run_refinement_probe.py',
                '--system-root', system['root'], '--settings', system['settings'],
                '--cases', config['cases'], '--output', str(OUT/(name+'.jsonl')),
                '--variant', system['variant'], '--warmup-case', config['warmup_case']],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({'event': 'complete', 'system': name}), flush=True)


if __name__ == '__main__':
    main()
