"""Run a frozen multi-system comparison, serial GPU inference, then blind judging.

The configuration is explicit and hash-bound. This driver never chooses a
candidate, tunes a metric, drops failed cases, changes a source snapshot, or
promotes a deployment. Resume uses each component's existing immutable contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    scripts = ['run_refinement_probe.py', 'run_security_comparison.py', 'score_expert_bertscore.py',
               'prepare_judge_packets.py', 'run_joint_claude_judge.py', 'run_claude_judge.py',
               'summarize_background_eval.py', Path(__file__).name]
    contract = {'config_sha256': sha(args.config),
                'scripts': {name: sha(ROOT/'evaluation'/name) for name in scripts},
                'cases_sha256': sha(ROOT/config['cases']),
                'expert_cases_sha256': sha(ROOT/config['expert_cases']),
                'metric_settings_sha256': sha(ROOT/config['metric_settings'])}
    text = json.dumps(contract, indent=2)+'\n'
    manifest = output/'driver-contract.json'
    if manifest.exists() and manifest.read_text() != text:
        raise ValueError('Comparison contract changed; use a new output directory')
    manifest.write_text(text)

    def run(label, command):
        print(json.dumps({'event': 'start', 'step': label,
                          'time': datetime.now(timezone.utc).isoformat()}), flush=True)
        with (output/(label+'.log')).open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({'event': 'complete', 'step': label}), flush=True)

    def telemetry(name):
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                 '--format=csv,noheader'], capture_output=True, text=True)
        with (output/'gpu-observations.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': name, 'time': datetime.now(timezone.utc).isoformat(),
                                     'returncode': result.returncode, 'processes': result.stdout})+'\n')

    results = []
    for system in config['systems']:
        name = system['name']
        path = output/(name+'.jsonl')
        results.append(str(path))
        telemetry('before-'+name)
        run(name, [config['generation_python'], '-u', 'evaluation/run_refinement_probe.py',
                   '--system-root', system['root'], '--settings', system['settings'],
                   '--cases', config['cases'], '--output', str(path), '--variant', system['variant'],
                   '--warmup-case', config['warmup_case']])
        telemetry('after-'+name)
        # The inference process has exited before the GPU metric process starts.
        for split in ('development', 'validation'):
            run(name+'-bertscore-'+split, [config['metric_python'], '-u', 'evaluation/score_expert_bertscore.py',
                '--cases', config['expert_cases'], '--results', str(path), '--split', split,
                '--settings', config['metric_settings'], '--output', str(output/('bertscore-'+name+'-'+split))])
    judge_dir = output/'judge'
    run('prepare-judge', [sys.executable, 'evaluation/prepare_judge_packets.py',
        '--cases', config['cases'], '--results', *results, '--output', str(judge_dir)])
    run('judge', [config['generation_python'], '-u', 'evaluation/run_joint_claude_judge.py',
                 '--directory', str(judge_dir), '--workers', '3', '--model', 'claude-opus-5-5'])
    run('summary', [sys.executable, 'evaluation/summarize_background_eval.py',
        '--cases', config['cases'], '--results', *results, '--judge-directory', str(judge_dir),
        '--output', str(output/'review')])
    print(json.dumps({'event': 'finished', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
