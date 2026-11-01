"""Build the maintained source translations from reviewed, versioned artifacts.

The revision contains the entire catalog but reviews only edited translations.
Unchanged translations retain their first review. Neither questions nor answer
similarity scores participate in building the runtime lookup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soprano_qa.security import screen_text


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed(directory):
    packets = {p['id']: p for p in json.loads((directory/'packets.json').read_text())}
    rows = [json.loads(line) for line in (directory/'assessments.jsonl').read_text().splitlines()]
    if len(rows) != len(packets) or {r['id'] for r in rows} != set(packets):
        raise ValueError('Incomplete review: ' + str(directory))
    result = {}
    for row in rows:
        packet = packets[row['id']]
        expected = hashlib.sha256(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if row['status'] != 'ok' or row['model'] != 'claude-opus-5-5' or row['packet_hash'] != expected:
            raise ValueError('Invalid or changed review packet')
        source = packet['evidence'][0]
        result[source['id']] = (source['text'], packet['answer'], row['assessment'])
    if len(result) != len(rows):
        raise ValueError('Duplicate source review')
    return result


def build(base, revision):
    original = {r['source_id']: r for r in json.loads((base/'records.json').read_text())}
    records = json.loads((revision/'records.json').read_text())
    if len(records) != len(original) or {r['source_id'] for r in records} != set(original):
        raise ValueError('Revision must retain the entire source catalog')
    old_reviews, new_reviews = reviewed(base), reviewed(revision)
    changed = {r['source_id'] for r in records if r['english'] != original[r['source_id']]['english']}
    if set(old_reviews) != set(original) or set(new_reviews) != changed:
        raise ValueError('Every original and every edited translation must be reviewed')
    translations = {}
    source_findings = []
    for record in records:
        source_id = record['source_id']
        before = original[source_id]
        if {k:v for k,v in record.items() if k != 'english'} != {k:v for k,v in before.items() if k != 'english'}:
            raise ValueError('Source records must remain unchanged')
        source_hash = hashlib.sha256(record['source'].encode()).hexdigest()
        if source_hash != record['source_sha256']:
            raise ValueError('Source hash mismatch')
        source, english, assessment = (new_reviews if source_id in changed else old_reviews)[source_id]
        if (source, english) != (record['source'], record['english']) or assessment['source_fidelity'] != 'faithful':
            raise ValueError('Final translation lacks a faithful review: ' + source_id)
        if not isinstance(english, str) or not english.strip() or len(english) > 2000:
            raise ValueError('Invalid translation length')
        screen_text(source, source='evidence')
        screen_text(english, source='translation')
        if source_hash in translations and translations[source_hash] != english:
            raise ValueError('Ambiguous source translation')
        translations[source_hash] = english
        if assessment['factual_correctness'] < 4:
            source_findings.append(source_id)
    provenance = {
        'corpus_sha256': json.loads((base/'preparation.json').read_text())['corpus_sha256'],
        'records': len(records),
        'author': 'Codex; machine-authored, checked by the coding agent, not human-expert English annotations.',
        'reviewer': 'claude-opus-5-5',
        'review': f'{len(original)} original translations reviewed; {len(changed)} edited translations reviewed again. All final translations judged faithful. This is model assessment, not proof of correctness.',
        'source_factual_findings': sorted(source_findings),
        'artifacts': {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): digest(path)
                      for folder in (base, revision)
                      for path in (folder/'records.json', folder/'packets.json', folder/'assessments.jsonl')},
        'reference_policy': 'Original Korean expert references remain unchanged. Lookup uses exact source text only, never evaluation questions or case IDs.',
    }
    header = '''"""Maintained English translations of curated Korean source answers.

Generated with evaluation/build_source_translation_catalog.py from versioned
source preparations and independent model reviews. This is not an answer cache
or a set of human-expert English annotations. Matching requires the exact UTF-8
SHA256 of the original source. Request-time safety and output review still run.
Faithful translation does not establish the factual correctness of the source.
"""

'''
    return header + 'PROVENANCE = ' + pprint.pformat(provenance, sort_dicts=False, width=100) + '\n\nTRANSLATIONS_EN = ' + pprint.pformat(dict(sorted(translations.items())), width=100) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-review', type=Path, required=True)
    parser.add_argument('--revision-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = build(args.base_review.resolve(), args.revision_review.resolve())
    if args.check:
        if args.output.read_text() != result:
            raise ValueError('Generated catalog differs from reviewed inputs')
    else:
        args.output.write_text(result)
    print(('Checked' if args.check else 'Built') + ' reviewed source translation catalog')


if __name__ == '__main__':
    main()
