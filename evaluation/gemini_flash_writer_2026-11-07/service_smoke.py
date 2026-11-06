"""Exercise real retrieval and provider routing once per language, after capture."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['ko', 'en'], required=True)
    language = parser.parse_args().language
    output = ROOT / ('service-smoke.json' if language == 'ko' else 'service-smoke-en.json')
    assert not output.exists(), 'Do not overwrite a completed smoke request'
    assert os.environ['GOOGLE_CLOUD_PROJECT'] == 'project-04eb605e-5bee-4a0d-aa3'
    assert all((ROOT / name / 'completion.json').exists() for name in ['expert', 'challenge'])
    assert len((ROOT / 'timing-sequential.jsonl').read_text().splitlines()) == 10
    os.environ['SOPRANO_QA_SETTINGS'] = str(REPO / 'config/settings.gemini.json')
    sys.path.insert(0, str(REPO))
    from soprano_qa import service, llm, translation

    question = ('col canto에서는 누가 누구를 따라가나요?' if language == 'ko'
                else 'In col canto, who follows whom?')
    local_calls = []
    original = service.generate_llm

    def tracked(model_path, messages, settings, **kwargs):
        assert language == 'en', 'Korean Gemini path must not use local generation'
        assert messages == translation.build_translation_messages(question, prompt_only=True)
        local_calls.append('local_query_translation')
        return original(model_path, messages, settings, **kwargs)

    service.generate_llm = tracked
    record = {'language': language, 'question': question}
    started = time.perf_counter()
    try:
        response = service.ask(piece_id='una-voce-poco-fa', question=question,
                               measure_range=None, generate=True, language=language)
        record.update(status='response', response=response, local_calls=local_calls,
                      local_writer_loaded=llm.load_llama.cache_info().currsize)
    except Exception as exc:
        record.update(status='exception', exception={'type': type(exc).__name__, 'message': str(exc)[:1000]})
    record['wall_seconds'] = time.perf_counter() - started
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    assert record['status'] == 'response', record.get('exception')
    assert response['model']['backend'] == 'vertex_gemini'
    assert response['generation_metadata']['model_version'] == 'gemini-3.8-flash'
    assert response['retrieval']['active_mode'] == 'hybrid'
    assert response['generation_mode'] == 'llm_single_pass'
    assert local_calls == ([] if language == 'ko' else ['local_query_translation'])
    if language == 'ko': assert record['local_writer_loaded'] == 0
    print(json.dumps({'language': language, 'seconds': record['wall_seconds'],
                      'local_calls': local_calls, 'answer': response['answer']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
