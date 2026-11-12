"""Observe the five saved refusals without changing any decision or model input."""
from pathlib import Path
import functools
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT.parent / 'latest_demo_2026-11-03'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provenance = read(CAPTURE / 'provenance.json')
    for rel, digest in provenance['source_files'].items():
        assert sha(Path(provenance['source_root']) / rel) == digest
        if rel.startswith('soprano_qa/'):
            assert sha(CAPTURE / 'snapshot' / rel) == digest
    assert sha(CAPTURE / 'settings.json') == provenance['settings_sha256']
    assert sha(CAPTURE / 'current.jsonl') == read(CAPTURE / 'completion.json')['result_sha256']
    cases = {c['id']: c for c in read(CAPTURE / 'cases.json')['cases']}
    failures = [r for r in map(json.loads, (CAPTURE / 'current.jsonl').read_text().splitlines())
                if (r.get('response') or {}).get('generation_mode') == 'unavailable']
    assert len(failures) == 5
    out = ROOT / 'traces.jsonl'
    done = {r['case_id'] for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    if len(done) == len(failures):
        print('All five diagnosis cases are complete.')
        return
    for name in ['SOPRANO_QA_MODEL_PATH', 'SOPRANO_QA_RAG_DATASET_ROOT',
                 'SOPRANO_QA_DATASET_ROOT', 'SOPRANO_QA_EMBEDDING_MODEL_PATH']:
        os.environ.pop(name, None)
    os.environ['SOPRANO_QA_SETTINGS'] = str(CAPTURE / 'settings.json')
    sys.path.insert(0, str(CAPTURE / 'snapshot'))
    from soprano_qa import service as qa, llm
    assert Path(qa.SETTINGS['model_path']).name == 'Qwen3-14B-Q6_K.gguf'
    qa._get_index()
    cfg = qa.SETTINGS['llm']
    llm.load_llama(model_path=qa.SETTINGS['model_path'], n_ctx=cfg['n_ctx'],
                   n_gpu_layers=cfg['n_gpu_layers'], chat_format=llm._configured_chat_format(cfg))
    events = []

    def observe(name, fn, describe):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            event = {'stage': name, **describe(*args, **kwargs)}
            events.append(event)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                event.update(outcome='raised', exception=type(exc).__name__, message=str(exc),
                             reason=getattr(exc, 'reason', None))
                raise
            event['outcome'] = 'returned'
            if name == 'retrieve':
                event['retrieved_ids'] = [r.record['id'] for r in result[0]]
                event['retrieved_answers'] = [r.record.get('answer') for r in result[0]]
            elif name == 'select':
                event['selected_evidence'] = result.get('Retrieved evidence', [])
            elif name == 'finalize':
                event['visible_answer'] = str(result)
            return result
        return wrapped

    qa.retrieve_generation_evidence = observe('retrieve', qa.retrieve_generation_evidence,
                                              lambda *a, **k: {})
    llm.select_question_evidence = observe('select', llm.select_question_evidence,
                                          lambda *a, **k: {})
    llm.semantic_guard = observe('guard', llm.semantic_guard,
        lambda model_path, settings, payload, **kw: {
            'guard_stage': kw['stage'], 'draft': payload.get('draft'),
            'composition': payload.get('composition'),
            'evidence': payload.get('Retrieved evidence', [])})
    qa.finalize_grounded_generated_answer = observe('finalize', qa.finalize_grounded_generated_answer,
        lambda draft, *a, **k: {'draft': str(draft), 'parts': getattr(draft, 'parts', None)})
    with out.open('a') as stream:
        for saved in failures:
            if saved['case_id'] in done:
                continue
            c = cases[saved['case_id']]
            events.clear()
            started = time.perf_counter()
            response = qa.ask(piece_id=c['piece_id'], question=c['question'],
                measure_range=tuple(c['measure_range']) if c.get('measure_range') else None,
                generate=True, top_k=c.get('top_k', 6), language=c['language'])
            row = {'case_id': c['id'], 'question': c['question'], 'events': list(events),
                'response': response, 'diagnostic_seconds': time.perf_counter() - started,
                'saved_reason': saved['response']['unavailable_reason'],
                'same_public_answer': response['answer'] == saved['response']['answer'],
                'same_reason': response['unavailable_reason'] == saved['response']['unavailable_reason']}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({'case_id': c['id'], 'same_public_answer': row['same_public_answer'],
                'same_reason': row['same_reason'], 'failures': [
                    {'stage': e['stage'], 'message': e.get('message'), 'reason': e.get('reason')}
                    for e in events if e['outcome'] == 'raised']}, ensure_ascii=False), flush=True)
    for rel, digest in provenance['source_files'].items():
        assert sha(Path(provenance['source_root']) / rel) == digest
    (ROOT / 'verification.json').write_text(json.dumps({
        'cases': 5, 'script_sha256': sha(Path(__file__)), 'traces_sha256': sha(out),
        'capture_sha256': sha(CAPTURE / 'current.jsonl'),
        'source_contract_sha256': sha(CAPTURE / 'provenance.json'),
        'source_files_unchanged': True, 'model': 'Qwen3-14B-Q6_K',
        'intervention': 'Observation wrappers only; original arguments, return values, exceptions, and decisions preserved. No extra model calls. Saved demo outputs were not replaced.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
