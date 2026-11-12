#!/usr/bin/env python3
"""Export real loopback demo responses into a review-only static snapshot.

The server is owned by this invocation and always closed. No running service or
published docs are touched. The same usage cap as the comparison gates every
model call. Only requests and delivered responses are exported, and the
imported pipeline identity is recorded explicitly.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import run_refinement_probe as probe


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lab-root', 'system-root', 'settings', 'output', 'usage-rollout', 'usage-state'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'answers').exists():
        raise ValueError('Use a new preview directory; do not overwrite a reviewed snapshot')
    os.environ['SOPRANO_QA_PIPELINE_ROOT'] = str(args.system_root.resolve())
    os.environ['SOPRANO_QA_SETTINGS'] = str(args.settings.resolve())
    settings = json.loads(args.settings.read_text())
    os.environ['SOPRANO_QA_DATASET_ROOT'] = settings['dataset_root']
    os.environ['SOPRANO_QA_RAG_DATASET_ROOT'] = settings['dataset_root']
    guard = probe.UsageGuard(args.usage_rollout, args.usage_state, args.output / 'usage-checks.jsonl')
    guard.check()
    server = load_module('refinement_demo_server', args.lab_root / 'demo/server.py')
    from soprano_qa import llm
    if Path(llm.__file__).resolve().parent != (args.system_root / 'soprano_qa').resolve():
        raise RuntimeError('Unexpected imported pipeline')
    counter = probe.TraceCounter(guard)
    llm.load_llama = counter.wrap_loader(llm.load_llama)
    actual_ask = server.qa.ask

    def guarded_ask(*, piece_id, question, measure_range=None, generate=True, language='ko'):
        guard.check()
        counter.reset(None, {})
        started = time.perf_counter()
        request = dict(piece_id=piece_id, question=question, measure_range=measure_range,
                       generate=generate, language=language)
        response = actual_ask(**request)
        row = {'request': request, 'response': response, 'elapsed_seconds': time.perf_counter() - started}
        with (args.output / 'requests.jsonl').open('a') as stream:
            probe.common.append_row(stream, row)
        return response

    server.qa.ask = guarded_ask
    httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    exporter = load_module('refinement_static_export', args.lab_root / 'docs/tools/export.py')
    shutil.copyfile(args.lab_root / 'docs/app.js', args.output / 'app.js')
    exporter.SITE_ROOT = args.output
    try:
        exporter.export_answers('http://127.0.0.1:' + str(httpd.server_port))
        guard.check()
        identity = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'pipeline_source_sha256': probe.common.source_sha256(args.system_root),
            'settings_sha256': hashlib.sha256(args.settings.read_bytes()).hexdigest(),
            'demo_server_sha256': hashlib.sha256((args.lab_root / 'demo/server.py').read_bytes()).hexdigest(),
            'scope': 'Real requests to a temporary loopback demo; preview artifacts only. No publication.',
        }
        (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
