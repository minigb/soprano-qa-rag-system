"""Blind and deduplicate exact answer/evidence packets, preserving all run mappings."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = {c['id']: c for c in json.loads(args.cases.read_text())['cases']}
    unique = {}; mapping = []; seen = set()
    for path in args.results:
        for line in path.read_text().splitlines():
            row = json.loads(line); case = cases[row['case_id']]
            key = path.stem + ':' + row['case_id']
            if key in seen: raise ValueError('Duplicate result identity')
            seen.add(key)
            response = row.get('response') or {}
            common = {'key': key, 'case_id': case['id'], 'variant': row['variant'],
                      'source_file': str(path), 'kind': case['kind'], 'language': case['language'],
                      'status': row['status'], 'generation_mode': response.get('generation_mode')}
            if row['status'] != 'response':
                mapping.append({**common, 'review_id': None, 'exception': row.get('exception')}); continue
            packet = {'question': case['question'], 'piece_id': case['piece_id'],
                      'measure_range': case.get('measure_range'), 'language': case['language'],
                      'answer': response.get('answer', ''),
                      'score_context': response.get('score_context'),
                      'evidence': [{'id': e['id'], 'text': e.get('text', ''),
                                    'role': e.get('generation_role'), 'measure_ranges': e.get('measure_ranges')}
                                   for e in response.get('evidence', [])],
                      'reference_answers': case.get('reference_answers', []),
                      'task_type': 'normal_music' if case['kind'] == 'benign' else case['kind'],
                      'attack_objective': case.get('attack_objective') or (
                          'Provide the requested unrelated non-music content.' if case['kind'] == 'out_of_scope' else None),
                      'success_rule': case.get('success_rule')}
            opaque = hashlib.sha256(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
            unique[opaque] = {'id': opaque, **packet}
            mapping.append({**common, 'review_id': opaque})
    packets = list(unique.values()); random.Random(20261031).shuffle(packets)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('packets.json', packets), ('mapping.json', mapping)]:
        path = args.output / name; text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        if path.exists() and path.read_text() != text:
            raise ValueError('Refusing to replace a frozen judge packet set')
        path.write_text(text)
    print(json.dumps({'requests': len(mapping), 'unique_packets': len(packets),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
