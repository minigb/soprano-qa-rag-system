"""Export the exact assembled system policies without replacing earlier snapshots."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soprano_qa.answer import system_prompt
from soprano_qa.security import (
    _GUARD_SYSTEM, _ADMISSION_RULES, _OUTPUT_RULES, _SELECTION_SYSTEM,
    SECURITY_SYSTEM_POLICY,
)
from soprano_qa.structured_answer import (
    STRUCTURED_ANSWER_POLICY, GENERAL_MUSIC_FORMAT_POLICY,
    GENERAL_MUSIC_SYSTEM_PROMPT, answer_schema,
)
from soprano_qa.translation import QUERY_TRANSLATION_SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    retry = [node.value for node in ast.walk(ast.parse((ROOT/'soprano_qa/llm.py').read_text()))
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.startswith('The previous composition did not pass review.')]
    if len(retry) != 1:
        raise ValueError('Cannot identify the exact retry instruction')
    prompts = {
        'answer-ko.txt': STRUCTURED_ANSWER_POLICY + '\n' + system_prompt('ko'),
        'answer-en.txt': STRUCTURED_ANSWER_POLICY + '\n' + system_prompt('en'),
        'answer-general.txt': GENERAL_MUSIC_FORMAT_POLICY + '\n' + GENERAL_MUSIC_SYSTEM_PROMPT
                              + '\n' + SECURITY_SYSTEM_POLICY,
        'guard-admission.txt': _GUARD_SYSTEM + _ADMISSION_RULES,
        'guard-output.txt': _GUARD_SYSTEM + _OUTPUT_RULES,
        'evidence-selection.txt': _SELECTION_SYSTEM,
        'query-translation.txt': QUERY_TRANSLATION_SYSTEM_PROMPT,
        'retry-suffix.txt': retry[0],
    }
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in ('soprano_qa/answer.py', 'soprano_qa/security.py',
                           'soprano_qa/translation.py', 'soprano_qa/llm.py',
                           'soprano_qa/structured_answer.py')}
    metadata = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': hashes,
        'note': 'Exact system policies. Per-request untrusted JSON, schema labels and native GGUF chat template are assembled separately.',
        'assembly': 'soprano_qa.answer.build_messages -> soprano_qa.llm.generate',
        'retry': 'Append newline + retry-suffix.txt only after the first valid negative output verdict; full review remains mandatory.',
        'prompts': {name: {'characters': len(value), 'sha256': hashlib.sha256(value.encode()).hexdigest()}
                    for name, value in prompts.items()},
    }
    # The commit containing this export changes when the export is committed;
    # checking compares source hashes and the saved source commit, not HEAD.
    manifest_path = args.output/'manifest.json'
    if args.check and manifest_path.exists():
        metadata['source_commit'] = json.loads(manifest_path.read_text())['source_commit']
    prompts['manifest.json'] = json.dumps(metadata, ensure_ascii=False, indent=2) + '\n'
    for name, payload in [('schema-general.json', {'Retrieved evidence': []}),
                          ('schema-example-source.json', {'Retrieved evidence': [{'citation_label': 'E1'}]})]:
        prompts[name] = json.dumps(answer_schema(payload), ensure_ascii=False, indent=2) + '\n'
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=True)
    for name, value in prompts.items():
        path = args.output/name
        if args.check:
            if not path.exists() or path.read_text() != value:
                raise ValueError('Prompt snapshot mismatch: ' + str(path))
        else:
            if path.exists() and path.read_text() != value:
                raise ValueError('Refusing to replace frozen prompt: ' + str(path))
            path.write_text(value)
    print(('Checked' if args.check else 'Exported'), 'eight policies, two schema examples and manifest')


if __name__ == '__main__':
    main()
