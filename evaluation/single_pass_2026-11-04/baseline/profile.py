"""Measure actual local inference stages without changing their behavior."""
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime, timezone
import collections
import functools
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = read(ROOT / 'provenance.json')
    assert sha(ROOT / 'settings.json') == contract['settings_sha256']
    assert sha(ROOT / 'cases.json') == contract['cases_sha256']
    assert sha(Path(__file__)) == contract['runner_sha256']
    for rel, digest in contract['source_files'].items():
        if rel.startswith('soprano_qa/'):
            assert sha(ROOT / 'snapshot' / rel) == digest
    out = ROOT / 'timings.jsonl'
    done = {r['id'] for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    if len(done) == 40:
        print('All measurements already complete.')
        return
    for key in ('SOPRANO_QA_MODEL_PATH', 'SOPRANO_QA_RAG_DATASET_ROOT', 'SOPRANO_QA_DATASET_ROOT', 'SOPRANO_QA_EMBEDDING_MODEL_PATH'):
        os.environ.pop(key, None)
    os.environ['SOPRANO_QA_SETTINGS'] = str(ROOT / 'settings.json')
    sys.path.insert(0, str(ROOT / 'snapshot'))
    cold = time.perf_counter()
    with redirect_stdout(sys.stderr):
        from soprano_qa import service as qa, llm, security, response_policy
        qa._get_index()
        cfg = qa.SETTINGS['llm']
        model = llm.load_llama(qa.SETTINGS['model_path'], cfg['n_ctx'], cfg['n_gpu_layers'], llm._configured_chat_format(cfg))
    cold_seconds = time.perf_counter() - cold
    cases = read(ROOT / 'cases.json')['cases']

    def ask(case):
        with redirect_stdout(sys.stderr):
            return qa.ask(piece_id=case['piece_id'], question=case['question'],
                          measure_range=tuple(case['measure_range']) if case.get('measure_range') else None,
                          generate=True, top_k=case.get('top_k', 6), language=case['language'])

    # Exclude first prompt/pipeline initialization, separately from model loading.
    warmup_start = time.perf_counter()
    ask(next(c for c in cases if c['timing_group'] == 'ordinary_music'))
    warmup_seconds = time.perf_counter() - warmup_start
    stack, completions, spans = [], [], []

    def wrap(fn, label):
        @functools.wraps(fn)
        def observed(*args, **kwargs):
            stage = label(*args, **kwargs) if callable(label) else label
            stack.append(stage)
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                spans.append({'stage': stage, 'seconds': time.perf_counter() - started})
                stack.pop()
        return observed

    qa.retrieve_generation_evidence = wrap(qa.retrieve_generation_evidence, 'retrieval_and_prompt')
    qa.translate_question_to_korean = wrap(qa.translate_question_to_korean, 'question_translation')
    qa.finalize_grounded_generated_answer = wrap(qa.finalize_grounded_generated_answer, 'deterministic_quality_check')
    llm.select_question_evidence = wrap(llm.select_question_evidence, 'ku_selection')
    llm.semantic_guard = wrap(llm.semantic_guard, lambda *a, **k: 'input_admission' if k['stage'] == 'admission' else 'output_review')
    guarded_instruction = wrap(security.instruction_guard, 'instruction_review')
    security.instruction_guard = response_policy.instruction_guard = guarded_instruction
    complete = model.create_chat_completion

    @functools.wraps(complete)
    def timed_complete(*args, **kwargs):
        stage = stack[-1] if stack else 'answer_generation'
        started = time.perf_counter()
        result = complete(*args, **kwargs)
        completions.append({'stage': stage, 'seconds': time.perf_counter() - started,
                            'prompt_chars': sum(len(m.get('content', '')) for m in kwargs.get('messages', [])),
                            'prompt_tokens': (result.get('usage') or {}).get('prompt_tokens'),
                            'completion_tokens': (result.get('usage') or {}).get('completion_tokens')})
        return result

    model.create_chat_completion = timed_complete
    with out.open('a') as stream:
        for repetition in range(2):
            for case in cases if repetition == 0 else reversed(cases):
                identity = f'{repetition}:{case["id"]}'
                if identity in done:
                    continue
                stack.clear(); completions.clear(); spans.clear()
                started_at = datetime.now(timezone.utc).isoformat()
                start = time.perf_counter()
                response = error = None
                try:
                    response = ask(case)
                except Exception as exc:
                    error = {'type': type(exc).__name__, 'message': str(exc)}
                elapsed = time.perf_counter() - start
                stage_seconds = collections.defaultdict(float)
                for item in completions:
                    stage_seconds[item['stage']] += item['seconds']
                retrieval_seconds = sum(s['seconds'] for s in spans if s['stage'] == 'retrieval_and_prompt')
                row = {'id': identity, 'case_id': case['id'], 'repetition': repetition,
                       'timing_group': case['timing_group'], 'language': case['language'],
                       'started_at': started_at, 'elapsed_seconds': elapsed,
                       'llm_calls': len(completions), 'llm_seconds_by_stage': dict(stage_seconds),
                       'retrieval_seconds': retrieval_seconds,
                       'remaining_seconds': elapsed - retrieval_seconds - sum(stage_seconds.values()),
                       'completions': list(completions), 'spans_inclusive': list(spans),
                       'response': response, 'exception': error}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); done.add(identity)
                print(json.dumps({'completed': len(done), 'total': 40, 'case': case['id'], 'group': case['timing_group'],
                                  'seconds': round(elapsed, 3), 'calls': len(completions),
                                  'stages': dict(stage_seconds)}, ensure_ascii=False), flush=True)
    (ROOT / 'completion.json').write_text(json.dumps({
        'measurements': len(done), 'results_sha256': sha(out), 'contract_sha256': sha(ROOT / 'provenance.json'),
        'cold_load_seconds': cold_seconds, 'excluded_warmup_seconds': warmup_seconds,
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'accounting': 'Request wall time = low-level 14B completion time + retrieval/prompt time + remaining CPU/I/O. Inclusive wrapper spans are diagnostic only; do not sum them again.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
