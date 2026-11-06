"""Checkpointed sequential Gemini writer replay with frozen inputs and code."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    contract = read(ROOT / 'contract.json')
    assert sha(Path(__file__)) == contract['run_sha256']
    assert sha(ROOT / 'writer-settings.json') == contract['writer_settings_sha256']
    for name, digest in contract['source_files'].items():
        assert sha(ROOT / 'snapshot' / name) == digest
    sys.path.insert(0, str(ROOT / 'snapshot'))
    from soprano_qa import vertex_gemini
    settings = read(ROOT / 'writer-settings.json')
    vertex_gemini.validate_requirements(settings)
    for cohort in ['expert', 'challenge']:
        directory = ROOT / cohort
        assert sha(directory / 'writer-inputs.json') == contract['cohorts'][cohort]['inputs_sha256']
        assert sha(directory / 'cases.json') == contract['cohorts'][cohort]['cases_sha256']
        inputs = {r['case_id']: r for r in read(directory / 'writer-inputs.json')}
        exclusions = {r['case_id']: r for r in read(directory / 'excluded.json')}
        cases = read(directory / 'cases.json')['cases']
        output = directory / 'current.jsonl'
        old = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        done = {r['case_id'] for r in old}
        assert len(old) == len(done) and done <= {c['id'] for c in cases}
        inflight = directory / 'inflight.json'
        if inflight.exists():
            active = read(inflight)
            if active['case_id'] not in done:
                raise RuntimeError('Ambiguous prior API call; inspect inflight.json before explicit recovery')
            inflight.unlink()
        with output.open('a') as stream:
            for case in cases:
                cid = case['id']
                if cid in done: continue
                row = {'case_id': cid, 'variant': 'gemini-4-argon-low-frozen-j',
                       'started_at': datetime.now(timezone.utc).isoformat(),
                       'source_contract_sha256': sha(ROOT / 'contract.json'),
                       'latency_scope': 'writer_api_wall_clock_excluding_retrieval_and_translation'}
                if cid in exclusions:
                    row.update(status='excluded', response=None, exception=exclusions[cid]['baseline_exception'],
                               exclusion_reason=exclusions[cid]['reason'], elapsed_seconds=0)
                else:
                    packet = inputs[cid]
                    inflight.write_text(json.dumps({'case_id': cid, 'started_at': row['started_at']}) + '\n')
                    started = time.perf_counter()
                    try:
                        answer = vertex_gemini.generate(packet['messages'], settings)
                        response = copy.deepcopy(packet['baseline_response'])
                        response.update(answer=str(answer), model={'backend': 'vertex_gemini', 'model': settings['vertex_gemini']['model']},
                                        generation_metadata=answer.generation_metadata)
                        row.update(status='response', response=response, exception=None)
                    except Exception as exc:
                        row.update(status='exception', response=None,
                                   exception={'type': type(exc).__name__, 'message': str(exc)[:1000]},
                                   generation_metadata=getattr(exc, 'metadata', None),
                                   partial_text=getattr(exc, 'partial_text', None))
                    row.update(elapsed_seconds=time.perf_counter()-started, messages_sha256=packet['messages_sha256'])
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
                if inflight.exists(): inflight.unlink()
                done.add(cid)
                print(json.dumps({'cohort': cohort, 'completed': len(done), 'total': len(cases),
                                  'case_id': cid, 'status': row['status'], 'seconds': row['elapsed_seconds']}, ensure_ascii=False), flush=True)
                if row['status'] == 'exception':
                    # Keep the failed case, and stop on account/auth/quota failures
                    # instead of blindly submitting the rest of the cohort.
                    message = row['exception']['message']
                    if any(s in message for s in ('401', '403', '429', 'BILLING', 'UNAUTHENTICATED')):
                        raise RuntimeError('Provider access/quota error; inspect recorded exception before resuming')
        (directory / 'completion.json').write_text(json.dumps({'cases': len(done), 'result_sha256': sha(output),
            'source_contract_sha256': sha(ROOT / 'contract.json')}, indent=2) + '\n')


if __name__ == '__main__':
    main()
