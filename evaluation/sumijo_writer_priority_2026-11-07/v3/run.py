"""Frozen-input regression for the scoped Sumi Jo writer preference.

Regenerate only changed prompts, never select among successful outputs.
This evaluation does not modify historical results or invoke an online judge.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PRIMARY = ROOT.parents[1] / 'gemini_flash_writer_2026-11-07'
BASELINE = ROOT.parents[1] / 'gemini_flash_recovery_2026-11-07'
sys.path.insert(0, str(REPO))


def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(s) for s in path.read_text().splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def freeze(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists(): assert path.read_text() == text, str(path)
    else: path.write_text(text)


def prepare():
    from soprano_qa import single_pass
    from soprano_qa.retrieval import SearchResult
    from soprano_qa.single_pass_prompt import source_priority_policy
    spec = importlib.util.spec_from_file_location('previous_single_pass', PRIMARY / 'snapshot/soprano_qa/single_pass.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    corpus = {r['id']: r for r in read(REPO / 'data/corpus.json')}
    fields = {f.name for f in dataclasses.fields(SearchResult)} - {'record', 'scope_match'}
    outputs = {r['case_id']: r for r in rows(BASELINE / 'combined/expert.jsonl')}
    cases = {c['id']: c for c in read(PRIMARY / 'expert/cases.json')['cases']}
    audit, changed = [], []
    for packet in read(PRIMARY / 'expert/writer-inputs.json'):
        payload = json.loads(packet['messages'][1]['content'])
        results = [SearchResult(record=corpus[e['id']], scope_match=e['scope_match'],
                               **{f: e.get(f, 0) for f in fields}) for e in packet['baseline_response']['evidence']]
        args = (payload['piece'], payload['measure_range'], payload['question'], results)
        kwargs = {'score_context': json.dumps(payload['Score context'], ensure_ascii=False) if payload['Score context'] is not None else '',
                  'language': payload['answer_language']}
        before = old.build_messages(*args, **kwargs)
        after = single_pass.build_messages(*args, **kwargs)
        assert before == packet['messages'], 'Rebuild differs from frozen prompt'
        assert before[1:] == after[1:], 'Question/KU/score context changed'
        is_changed = before != after
        audit.append({'case_id': packet['case_id'], 'changed': is_changed,
                      'before_sha256': digest(before), 'after_sha256': digest(after)})
        if is_changed:
            changed.append({'case_id': packet['case_id'], 'messages': after,
                            'messages_sha256': digest(after)})
    assert len(audit) == 109 and len(changed) == 8
    challenge = read(PRIMARY / 'challenge/writer-inputs.json')
    for p in challenge:
        data = json.loads(p['messages'][1]['content'])
        assert not source_priority_policy(data['answer_language'], data['Retrieved evidence'])
    freeze(ROOT / 'input-audit.json', {'expert': audit, 'challenge_available_prompts': len(challenge),
                                    'challenge_changed_prompts': 0})
    freeze(ROOT / 'writer-inputs.json', changed)
    freeze(ROOT / 'cases.json', {'cases': [cases[p['case_id']] for p in changed]})
    freeze(ROOT / 'writer-settings.json', read(PRIMARY / 'writer-settings.json'))
    freeze(ROOT / 'metric-length-amendment.json', read(PRIMARY / 'expert/metric-length-amendment.json'))
    original_rows = [dict(outputs[p['case_id']], variant='before') for p in changed]
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in original_rows)
    previous = ROOT / 'before.jsonl'
    assert not previous.exists() or previous.read_text() == text
    previous.write_text(text)
    preserved = subprocess.check_output(['git', 'ls-files', '--', str(PRIMARY.relative_to(REPO)),
                 str(BASELINE.relative_to(REPO)), 'sites/answer-review/dist'], cwd=REPO, text=True).splitlines()
    source = {str(p.relative_to(REPO)): sha(p) for p in (REPO / 'soprano_qa').glob('*.py')}
    freeze(ROOT / 'contract.json', {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_files': source, 'historical_files': {p: sha(REPO / p) for p in preserved},
        'writer_inputs_sha256': sha(ROOT / 'writer-inputs.json'), 'cases_sha256': sha(ROOT / 'cases.json'),
        'writer_settings_sha256': sha(ROOT / 'writer-settings.json'), 'corpus_sha256': sha(REPO / 'data/corpus.json'),
        'policy': 'Only eight changed prompts regenerated. Reuse 101 unchanged exact prompts/results; report this explicitly. First output retained, no quality-based retries. Same model, decoding, retrieval, metric and quality rubric.',
        'concurrency': 1, 'start_interval_seconds': 10, 'maximum_requests': 8,
        'baseline': str(BASELINE.relative_to(REPO)), 'credential_environment': '~/gemini-vertex-4/env.sh'})
    print(json.dumps({'changed': 8, 'unchanged': 101, 'challenge_unchanged': len(challenge)}), flush=True)


def generate():
    from soprano_qa import vertex_gemini
    lock = (ROOT / 'generation.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = read(ROOT / 'contract.json')
    for rel, value in contract['source_files'].items(): assert sha(REPO / rel) == value
    assert sha(ROOT / 'writer-inputs.json') == contract['writer_inputs_sha256']
    assert sha(ROOT / 'writer-settings.json') == contract['writer_settings_sha256']
    config = read(ROOT / 'writer-settings.json'); assert config['vertex_gemini']['model'] == 'gemini-3.8-flash'
    vertex_gemini.validate_requirements(config)
    before = {r['case_id']: r for r in rows(ROOT / 'before.jsonl')}
    journal = ROOT / 'requests'; journal.mkdir(exist_ok=True)
    outputs = []; next_start = 0.
    for packet in read(ROOT / 'writer-inputs.json'):
        cid = packet['case_id']; stem = hashlib.sha256(cid.encode()).hexdigest()[:24]
        saved = journal / (stem + '.json'); pending = journal / (stem + '.pending')
        if saved.exists():
            row = read(saved); assert row['messages_sha256'] == packet['messages_sha256']
        else:
            assert not pending.exists(), 'Unobserved call: inspect before retrying'
            delay = max(0., next_start - time.monotonic())
            if delay: time.sleep(delay)
            next_start = time.monotonic() + 10
            row = {'case_id': cid, 'variant': 'sumijo-priority-v3', 'messages_sha256': packet['messages_sha256'],
                   'started_at': datetime.now(timezone.utc).isoformat(), 'api_project': os.environ['GOOGLE_CLOUD_PROJECT'],
                   'contract_sha256': sha(ROOT / 'contract.json'), 'latency_scope': 'generation_api_only'}
            freeze(pending, row); started = time.perf_counter()
            try:
                answer = vertex_gemini.generate(packet['messages'], config)
                response = copy.deepcopy(before[cid]['response'])
                response.update(answer=str(answer), generation_metadata=answer.generation_metadata)
                row.update(status='response', response=response, exception=None)
            except Exception as exc:
                row.update(status='exception', response=None, exception={'type': type(exc).__name__, 'message': str(exc)[:1000]})
            row['elapsed_seconds'] = time.perf_counter() - started
            freeze(saved, row); pending.unlink()
        outputs.append(row)
        (ROOT / 'after.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in outputs))
        print(json.dumps({'case_id': cid, 'status': row['status'], 'seconds': row['elapsed_seconds']}, ensure_ascii=False), flush=True)
        if row['status'] != 'response': raise RuntimeError('Provider failure preserved; inspect before more requests')
    for rel, value in contract['historical_files'].items(): assert sha(REPO / rel) == value
    freeze(ROOT / 'completion.json', {'cases': len(outputs), 'after_sha256': sha(ROOT / 'after.jsonl')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'generate'])
    globals()[parser.parse_args().action]()
