"""Freeze source and a representative timing cohort before changing the pipeline."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PREVIOUS = REPO / 'evaluation/helpful_responses_2026-11-04/full-final'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    run = ROOT / args.name
    run.mkdir(parents=True, exist_ok=False)
    cohort = ROOT / 'profile-cases.json'
    if not cohort.exists():
        cases = read(PREVIOUS / 'cases.json')['cases']
        outputs = {r['case_id']: r['response'] for r in map(json.loads, (PREVIOUS / 'current.jsonl').read_text().splitlines())}
        selected = []
        for piece in sorted({c['piece_id'] for c in cases}):
            eligible = sorted([c for c in cases if c['piece_id'] == piece and c['family'] == 'original'
                               and outputs[c['id']]['generation_mode'] == 'llm'], key=lambda c: c['id'])
            for multiple in (False, True):
                candidates = [c for c in eligible if (len(outputs[c['id']]['evidence']) > 1) == multiple
                              and c['id'] not in {s['id'] for s in selected}]
                chosen = candidates[0] if candidates else next(c for c in eligible if c['id'] not in {s['id'] for s in selected})
                selected.append({**chosen, 'timing_group': 'ordinary_music'})
        old = read(REPO / 'sites/answer-review/dist/data-latest-14b-20261103.json')
        refusals = {c['id'] for c in old['cases'] if c['versions']['current']['generation_mode'] == 'unavailable'}
        selected += [{**c, 'timing_group': 'previous_refusal'} for c in cases if c['id'] in refusals]
        challenge = read(ROOT.parent / 'helpful_responses_2026-11-04/security-hardened/cases.json')['cases']
        extra = {'benign-02-ko', 'benign-02-en', 'benign-03-en', 'direct-01-ko', 'out-of-scope-01-ko'}
        selected += [{**c, 'timing_group': 'general_or_english' if c['kind'] == 'benign' else 'scope_or_attack'}
                     for c in challenge if c['id'] in extra]
        assert len(selected) == len({c['id'] for c in selected}) == 20
        save(cohort, {'cases': selected, 'selection': 'Before changes: ten ordinary questions covering five works and single/multiple evidence, all five previous refusals, three general/English questions, two scope/attack requests. Real hybrid retrieval, not injected fixtures. Deliberately includes slow paths; not a random estimate of traffic.'})
    shutil.copyfile(cohort, run / 'cases.json')
    shutil.copyfile(ROOT / 'profile.py', run / 'profile.py')
    shutil.copytree(REPO / 'soprano_qa', run / 'snapshot/soprano_qa', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(PREVIOUS / 'runtime', run / 'runtime')
    settings = read(PREVIOUS / 'settings.json')
    settings['llm'] = read(REPO / 'config/settings.single-pass.json')['llm']
    for name in ('corpus_path', 'stats_path', 'embedding_cache_path', 'feature_overrides_path'):
        settings[name] = str(run / 'runtime' / Path(settings[name]).relative_to(PREVIOUS / 'runtime'))
    save(run / 'settings.json', settings)
    source_files = list((REPO / 'soprano_qa').glob('*.py')) + [REPO / 'config/settings.single-pass.json']
    save(run / 'provenance.json', {
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_files': {str(p.relative_to(REPO)): sha(p) for p in source_files},
        'settings_sha256': sha(run / 'settings.json'), 'cases_sha256': sha(run / 'cases.json'),
        'runner_sha256': sha(run / 'profile.py'), 'model': 'Qwen3-14B-Q6_K',
        'model_sha256': read(PREVIOUS / 'provenance.json')['model_sha256'],
        'policy': 'Observation wrappers preserve model inputs, outputs, exceptions and decisions. No extra reviewer calls. Two passes in forward and reverse order; first warmup excluded.'})
    (run / '.gitignore').write_text('runtime/\n*.log\n__pycache__/\n')
    print(json.dumps({'run': str(run), 'cases': 20, 'passes': 2}))


if __name__ == '__main__':
    main()
