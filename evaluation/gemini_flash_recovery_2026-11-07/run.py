"""One technical retry per missing Flash answer on the newly authorized account.

No successful answer is regenerated. Original captures, scores and judgments
remain immutable; this supplement reports coverage after one technical retry.
"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT.parent / 'gemini_flash_writer_2026-11-07'
REPO = ROOT.parents[1]
PROJECT = 'project-44e5d3cc-7919-41c0-bb5'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == text, 'Frozen artifact changed: ' + str(path)
    else:
        path.write_text(text)


def run(command, log):
    with log.open('a') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    lock = (ROOT / 'run.json.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ['GOOGLE_CLOUD_PROJECT'] == PROJECT
    plan, cases, inputs = [], [], {}
    for cohort in ['expert', 'challenge']:
        directory = PRIMARY / cohort
        indexed = {c['id']: c for c in read(directory / 'cases.json')['cases']}
        packets = {r['case_id']: r for r in read(directory / 'writer-inputs.json')}
        for row in rows(directory / 'current.jsonl'):
            if row['status'] != 'exception':
                continue
            assert row.get('response') is None
            cid = row['case_id']; packet = packets[cid]
            assert hashlib.sha256(json.dumps(packet['messages'], ensure_ascii=False,
                                             sort_keys=True).encode()).hexdigest() == packet['messages_sha256']
            plan.append({'case_id': cid, 'cohort': cohort,
                         'original_exception': row['exception'],
                         'messages_sha256': packet['messages_sha256']})
            cases.append(indexed[cid]); inputs[cid] = packet
    assert len(plan) == 7 and sum(p['cohort'] == 'expert' for p in plan) == 6
    tracked = subprocess.check_output(['git', 'ls-files', '--', str(PRIMARY.relative_to(REPO))],
                                      cwd=REPO, text=True).splitlines()
    contract = {'runner_sha256': sha(Path(__file__)),
                'original_files': {p: sha(REPO / p) for p in tracked},
                'project': PROJECT, 'environment_source': '/home/minhee/gemini-vertex-4/env.sh',
                'model': read(PRIMARY / 'writer-settings.json')['vertex_gemini'],
                'minimum_request_start_interval_seconds': 10, 'concurrency': 1,
                'maximum_requests': 7, 'plan': plan,
                'authorization': 'User provided a newly configured Google API account and authorized its use if needed.',
                'policy': 'Exactly one additional attempt per original technical failure. Keep all first-attempt results and scores; never regenerate successful answers. Same model, prompt, inputs, metric and rubric.'}
    freeze(ROOT / 'contract.json', contract)
    freeze(ROOT / 'cases.json', {'cases': cases})
    sys.path.insert(0, str(PRIMARY / 'snapshot'))
    from soprano_qa import vertex_gemini
    config = read(PRIMARY / 'writer-settings.json')
    assert config['vertex_gemini']['model'] == 'gemini-3.8-flash'
    vertex_gemini.validate_requirements(config)
    journal = ROOT / 'requests'; journal.mkdir(exist_ok=True)
    output = []; next_start = 0.
    for item in plan:
        cid = item['case_id']; stem = hashlib.sha256(cid.encode()).hexdigest()[:24]
        saved = journal / (stem + '.json'); pending = journal / (stem + '.pending')
        if saved.exists():
            row = read(saved)
            assert row['contract_sha256'] == sha(ROOT / 'contract.json')
        else:
            assert not pending.exists(), 'Unobserved request; do not silently resubmit'
            delay = max(0., next_start - time.monotonic())
            if delay: time.sleep(delay)
            next_start = time.monotonic() + 10
            row = {'case_id': cid, 'cohort': item['cohort'],
                   'variant': 'gemini-3.8-flash-technical-retry1', 'attempt_number': 2,
                   'started_at': datetime.now(timezone.utc).isoformat(), 'api_project': PROJECT,
                   'contract_sha256': sha(ROOT / 'contract.json'),
                   'messages_sha256': item['messages_sha256'],
                   'original_exception': item['original_exception']}
            freeze(pending, row); started = time.perf_counter()
            try:
                answer = vertex_gemini.generate(inputs[cid]['messages'], config)
                response = copy.deepcopy(inputs[cid]['baseline_response'])
                response.update(answer=str(answer), generation_metadata=answer.generation_metadata,
                                model={'backend': 'vertex_gemini', 'model': 'gemini-3.8-flash'})
                row.update(status='response', response=response, exception=None)
            except Exception as exc:
                row.update(status='exception', response=None,
                           exception={'type': type(exc).__name__, 'message': str(exc)[:1000]})
            row['elapsed_seconds'] = time.perf_counter() - started
            freeze(saved, row); pending.unlink()
        output.append(row)
        (ROOT / 'current.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output))
        print(json.dumps({'case_id': cid, 'status': row['status'], 'seconds': row['elapsed_seconds']}, ensure_ascii=False), flush=True)
        if row['exception'] and any(code in row['exception']['message'] for code in ['401', '403', '429', 'BILLING']):
            raise RuntimeError('Access/capacity failure: no more requests on this account')
    freeze(ROOT / 'completion.json', {'cases': len(output), 'result_sha256': sha(ROOT / 'current.jsonl')})
    scripts = PRIMARY / 'expert'
    run([sys.executable, str(scripts / 'prepare_expanded_judge_packets.py'),
         '--cases', str(ROOT / 'cases.json'), '--results', str(ROOT / 'current.jsonl'),
         '--output', str(ROOT / 'judge')], ROOT / 'judge.log')
    run([sys.executable, str(scripts / 'run_joint_expanded_judge.py'),
         '--directory', str(ROOT / 'judge'), '--model', 'claude-opus-5-5', '--workers', '3'], ROOT / 'judge.log')
    print('Offline judgment complete', flush=True)
    assessed = {r['id']: r['assessment'] for r in rows(ROOT / 'judge/assessments.jsonl')}
    judgments = {r['case_id']: assessed.get(r['review_id']) for r in read(ROOT / 'judge/mapping.json')}
    expert = ROOT / 'expert'; expert.mkdir(exist_ok=True)
    expert_ids = {p['case_id'] for p in plan if p['cohort'] == 'expert'}
    freeze(expert / 'cases.json', {'cases': [c for c in cases if c['id'] in expert_ids]})
    freeze(expert / 'metric-length-amendment.json', read(scripts / 'metric-length-amendment.json'))
    metric_rows = []
    for source in output:
        if source['case_id'] not in expert_ids: continue
        row = copy.deepcopy(source); judgment = judgments[row['case_id']]
        row['metric_claude_refusal'] = bool(judgment and judgment['refusal'])
        if row.get('response') and row['metric_claude_refusal']:
            row['response']['generation_mode'] = 'unavailable'
        metric_rows.append(row)
    metric_path = expert / 'metric-input.jsonl'
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in metric_rows)
    assert not metric_path.exists() or metric_path.read_text() == text
    metric_path.write_text(text)
    assert read(scripts / 'selected-metric-settings.json')['settings']['idf'] is False
    os.environ['HF_HUB_OFFLINE'] = '1'
    run(['/home/minhee/.cache/soprano-rag-eval-venv/bin/python', str(scripts / 'score_expanded_bertscore_bounded.py'),
         '--cases', str(expert / 'cases.json'), '--results', str(metric_path), '--split', 'all',
         '--settings', str(scripts / 'selected-metric-settings.json'),
         '--output', str(expert / 'bertscore'), '--device', 'cuda'], ROOT / 'score.log')
    for name, digest in contract['original_files'].items(): assert sha(REPO / name) == digest
    print('Six missing expert scores complete; original artifacts unchanged', flush=True)


if __name__ == '__main__':
    main()
