"""Export the assembled prompts, or check that saved prompts match the code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from soprano_qa.answer import system_prompt
from soprano_qa.security import _GUARD_SYSTEM, _ADMISSION_RULES, _OUTPUT_RULES
from soprano_qa.structured_answer import STRUCTURED_ANSWER_POLICY
from soprano_qa.translation import QUERY_TRANSLATION_SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    prompts = {
        'answer-ko.txt': STRUCTURED_ANSWER_POLICY + '\n' + system_prompt('ko'),
        'answer-en.txt': STRUCTURED_ANSWER_POLICY + '\n' + system_prompt('en'),
        'guard-admission.txt': _GUARD_SYSTEM + _ADMISSION_RULES,
        'guard-output.txt': _GUARD_SYSTEM + _OUTPUT_RULES,
        'query-translation.txt': QUERY_TRANSLATION_SYSTEM_PROMPT,
    }
    metadata = {
        'evaluated_revision': '79cbd7b',
        'note': 'Exact system message contents. Per-request question/evidence JSON and the native chat template are assembled separately; these are not complete tokenized requests.',
        'assembly': 'soprano_qa.answer.build_messages, then soprano_qa.llm.generate prepends STRUCTURED_ANSWER_POLICY',
        'schema': 'soprano_qa.structured_answer.answer_schema, applied by soprano_qa.llm.generate',
        'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ('soprano_qa/answer.py', 'soprano_qa/security.py',
                                 'soprano_qa/translation.py', 'soprano_qa/llm.py',
                                 'soprano_qa/structured_answer.py')},
        'prompts': {name: {'characters': len(value),
                           'sha256': hashlib.sha256(value.encode()).hexdigest()}
                    for name, value in prompts.items()},
    }
    prompts['manifest.json'] = json.dumps(metadata, ensure_ascii=False, indent=2) + '\n'
    directory = HERE / 'prompts'
    if not args.check:
        directory.mkdir(exist_ok=True)
    for name, value in prompts.items():
        path = directory / name
        if args.check:
            if not path.exists() or path.read_text() != value:
                raise ValueError('Prompt snapshot mismatch: ' + str(path))
        else:
            if path.exists() and path.read_text() != value:
                raise ValueError('Refusing to replace frozen prompt: ' + str(path))
            path.write_text(value, encoding='utf-8')
    print(('Checked' if args.check else 'Exported'), 'five system prompts and their manifest')


if __name__ == '__main__':
    main()
