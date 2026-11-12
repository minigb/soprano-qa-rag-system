"""Resume the same frozen writer experiment with bounded, spaced API requests.

Per-case journals checkpoint completed calls before ordered aggregation. A
planned interruption of the old sequential runner is recorded in the scheduling
amendment; stored outcomes (including failures) are never replaced.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import fcntl
import hashlib
import json
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--start-interval', type=float, default=5.0)
    args = parser.parse_args()
    if args.start_interval < 0: raise ValueError('Negative request interval')
    lock = (ROOT / 'execution.json.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = read(ROOT / 'contract.json'); amendment = read(ROOT / 'scheduling-amendment.json')
    assert sha(ROOT / 'writer-settings.json') == contract['writer_settings_sha256']
    for rel, digest in contract['source_files'].items(): assert sha(ROOT / 'snapshot' / rel) == digest
    prefix = (ROOT / 'expert/current.jsonl').read_text().splitlines(keepends=True)[:amendment['sequential_prefix_count']]
    assert hashlib.sha256(''.join(prefix).encode()).hexdigest() == amendment['prefix_sha256']
    execution = {'runner_sha256': sha(Path(__file__)), 'contract_sha256': sha(ROOT / 'contract.json'),
                 'amendment_sha256': sha(ROOT / 'scheduling-amendment.json'), 'workers': args.workers,
                 'minimum_request_start_interval_seconds': args.start_interval,
                 'reason': 'Reduce concurrency and smooth request starts after a second 429. Original runner saved under runners/. Stored results and provider inputs unchanged.'}
    p = ROOT / ('execution-contract-v2-w%d-interval%s.json' % (args.workers, args.start_interval))
    if p.exists(): assert read(p) == execution
    else: atomic(p, execution)
    sys.path.insert(0, str(ROOT / 'snapshot'))
    from soprano_qa import vertex_gemini
    config = read(ROOT / 'writer-settings.json')
    vertex_gemini.validate_requirements(config)
    fatal = threading.Event()
    rate_lock = threading.Lock()
    next_start = [0.0]
    accepted_contracts = {sha(f) for f in ROOT.glob('execution-contract*.json')
                          if read(f)['contract_sha256'] == sha(ROOT / 'contract.json')}
    for cohort in ['expert', 'challenge']:
        directory = ROOT / cohort; journal = directory / 'request-journal'
        journal.mkdir(exist_ok=True)
        assert sha(directory / 'writer-inputs.json') == contract['cohorts'][cohort]['inputs_sha256']
        assert sha(directory / 'cases.json') == contract['cohorts'][cohort]['cases_sha256']
        inputs = {r['case_id']: r for r in read(directory / 'writer-inputs.json')}
        exclusions = {r['case_id']: r for r in read(directory / 'excluded.json')}
        cases = read(directory / 'cases.json')['cases']; output = directory / 'current.jsonl'
        previous = [json.loads(l) for l in output.read_text().splitlines()] if output.exists() else []
        done = {r['case_id'] for r in previous}
        assert len(previous) == len(done) and done <= {c['id'] for c in cases}
        for r in previous: assert r['source_contract_sha256'] == sha(ROOT / 'contract.json')
        old_inflight = directory / 'inflight.json'
        if old_inflight.exists():
            state = read(old_inflight)
            assert state == amendment['interrupted_unobserved_call'] or state['case_id'] in done
            old_inflight.rename(directory / 'interrupted-sequential-call.json')

        def invoke(case):
            cid = case['id']; stem = hashlib.sha256(cid.encode()).hexdigest()[:24]
            saved = journal / (stem + '.json'); pending = journal / (stem + '.pending')
            if saved.exists():
                row = read(saved)
                assert row['case_id'] == cid and row['execution_contract_sha256'] in accepted_contracts
                return row
            if pending.exists(): raise RuntimeError('Unobserved API request requires explicit recovery: ' + str(pending))
            if fatal.is_set(): raise RuntimeError('Provider access error stopped new submissions')
            row = {'case_id': cid, 'variant': 'gemini-3.8-flash-low-frozen-j',
                   'started_at': datetime.now(timezone.utc).isoformat(), 'source_contract_sha256': sha(ROOT / 'contract.json'),
                   'execution_contract_sha256': sha(p), 'api_concurrency_limit': args.workers,
                   'latency_scope': 'writer_api_wall_clock_concurrency%d_excluding_retrieval_and_translation' % args.workers}
            if cid in exclusions:
                row.update(status='excluded', response=None, exception=exclusions[cid]['baseline_exception'],
                           exclusion_reason=exclusions[cid]['reason'], elapsed_seconds=0)
            else:
                packet = inputs[cid]
                with rate_lock:
                    wait = max(0, next_start[0] - time.monotonic())
                    if wait: time.sleep(wait)
                    if fatal.is_set(): raise RuntimeError('Provider access error stopped new submissions')
                    next_start[0] = time.monotonic() + args.start_interval
                row['started_at'] = datetime.now(timezone.utc).isoformat()
                atomic(pending, {'case_id': cid, 'started_at': row['started_at']})
                started = time.perf_counter()
                try:
                    answer = vertex_gemini.generate(packet['messages'], config)
                    response = copy.deepcopy(packet['baseline_response'])
                    response.update(answer=str(answer), model={'backend': 'vertex_gemini', 'model': config['vertex_gemini']['model']},
                                    generation_metadata=answer.generation_metadata)
                    row.update(status='response', response=response, exception=None)
                except Exception as exc:
                    message = str(exc)[:1000]
                    row.update(status='exception', response=None, exception={'type': type(exc).__name__, 'message': message},
                               generation_metadata=getattr(exc, 'metadata', None), partial_text=getattr(exc, 'partial_text', None))
                    if any(s in message for s in ('401', '403', '429', 'BILLING', 'UNAUTHENTICATED')): fatal.set()
                row.update(elapsed_seconds=time.perf_counter()-started, messages_sha256=packet['messages_sha256'])
            atomic(saved, row)
            if pending.exists(): pending.unlink()
            print(json.dumps({'cohort': cohort, 'case_id': cid, 'status': row['status'], 'seconds': row['elapsed_seconds']}, ensure_ascii=False), flush=True)
            return row

        with output.open('a') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [(c, pool.submit(invoke, c)) for c in cases if c['id'] not in done]
            for case, future in futures:
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); done.add(case['id'])
                print(json.dumps({'cohort': cohort, 'aggregated': len(done), 'total': len(cases)}), flush=True)
        atomic(directory / 'completion.json', {'cases': len(done), 'result_sha256': sha(output),
            'source_contract_sha256': sha(ROOT / 'contract.json'), 'execution_contract_sha256': sha(p)})


if __name__ == '__main__':
    main()
