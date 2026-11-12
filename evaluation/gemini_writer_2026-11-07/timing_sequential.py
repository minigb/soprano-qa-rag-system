"""Separate serial timing on the previous ten ordinary-music questions.

These additional responses are saved only as timing observations. They never
replace quality-capture answers or enter judging/BERTScore.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'prompt_fidelity_2026-11-06'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Quality API requests finish before these serial latency observations.
    while not all((ROOT / name / 'completion.json').exists() for name in ['expert', 'challenge']):
        time.sleep(10)
    contract = read(ROOT / 'contract.json')
    for rel, digest in contract['source_files'].items(): assert sha(ROOT / 'snapshot' / rel) == digest
    assert sha(ROOT / 'writer-settings.json') == contract['writer_settings_sha256']
    cases = [c for c in read(OLD / 'profile-cases.json')['cases'] if c['timing_group'] == 'ordinary_music']
    assert len(cases) == 10
    inputs = {r['case_id']: r for r in read(ROOT / 'expert/writer-inputs.json')}
    timing_contract = {'case_ids': [c['id'] for c in cases], 'run_sha256': sha(Path(__file__)),
                       'writer_contract_sha256': sha(ROOT / 'contract.json'),
                       'policy': 'One serial observation per case; descriptive comparison with historical two-repetition Qwen writer times. First API connection included. Not used for answer-quality selection.'}
    p = ROOT / 'timing-contract.json'
    if p.exists(): assert read(p) == timing_contract
    else: p.write_text(json.dumps(timing_contract, indent=2)+'\n')
    sys.path.insert(0, str(ROOT / 'snapshot'))
    from soprano_qa import vertex_gemini
    config = read(ROOT / 'writer-settings.json'); vertex_gemini.validate_requirements(config)
    output = ROOT / 'timing-sequential.jsonl'
    done = {json.loads(l)['case_id'] for l in output.read_text().splitlines()} if output.exists() else set()
    inflight = ROOT / 'timing-inflight.json'
    if inflight.exists() and read(inflight)['case_id'] not in done:
        raise RuntimeError('Unobserved timing request needs explicit recovery')
    with output.open('a') as stream:
        for case in cases:
            cid = case['id']
            if cid in done: continue
            row = {'case_id': cid, 'started_at': datetime.now(timezone.utc).isoformat()}
            inflight.write_text(json.dumps(row)+'\n'); started = time.perf_counter()
            try:
                answer = vertex_gemini.generate(inputs[cid]['messages'], config)
                row.update(status='response', answer=str(answer), generation_metadata=answer.generation_metadata, exception=None)
            except Exception as exc:
                row.update(status='exception', answer=None, exception={'type': type(exc).__name__, 'message': str(exc)[:1000]})
            row['elapsed_seconds'] = time.perf_counter()-started
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush(); inflight.unlink()
            print(json.dumps({'case_id': cid, 'seconds': row['elapsed_seconds'], 'status': row['status']}), flush=True)
    print('Serial timing complete', flush=True)


if __name__ == '__main__':
    main()
