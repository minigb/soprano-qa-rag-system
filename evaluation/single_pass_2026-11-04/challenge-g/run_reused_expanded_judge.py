"""Reuse a completed review only for identical packets under the same rubric.

Original assessments (including unfavorable ones) remain byte-for-byte records
with their original contracts. Changed question, source, answer, reference or
task packets receive a new independent assessment. No quality-based selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from . import run_expanded_judge as judge
    from . import run_joint_expanded_judge as joint
else:
    import run_expanded_judge as judge
    import run_joint_expanded_judge as joint


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def completed_review(directory, model):
    """Accept a fully validated ordinary joint review, never partial shards."""
    aggregation = read(directory/'aggregation.json')
    packets = read(directory/'packets.json')
    assessed = rows(directory/'assessments.jsonl')
    contract = read(directory/'joint-contract.json')
    if (aggregation['contract'] != contract or contract['model'] != model
            or contract['packets_sha256'] != joint.digest(directory/'packets.json')
            or aggregation['assessment_sha256'] != joint.digest(directory/'assessments.jsonl')
            or contract['judge_runner_sha256'] != joint.digest(Path(judge.__file__))
            or contract['runner_sha256'] != joint.digest(Path(joint.__file__))):
        raise ValueError('Source review contract or contents changed')
    expected = {'model': model, 'runner_sha256': joint.digest(Path(judge.__file__)),
                'rubric_sha256': hashlib.sha256(judge.SYSTEM.encode()).hexdigest(),
                'schema_sha256': judge.sha(judge.schema([])),
                'acceptance': 'first valid batch; never retry based on a score'}
    contracts = set()
    for name, hashes in aggregation['shards'].items():
        shard = directory/name
        manifest = read(shard/'assessments.jsonl.manifest.json')
        if (joint.digest(shard/'assessments.jsonl') != hashes['assessment_sha256']
                or joint.digest(shard/'assessments.jsonl.manifest.json') != hashes['manifest_sha256']
                or manifest['packet_file_sha256'] != joint.digest(shard/'packets.json')
                or any(manifest.get(k) != v for k, v in expected.items())):
            raise ValueError('Source rubric, model, schema or shard changed')
        contracts.add(judge.sha(manifest))
    if any(row['contract_id'] not in contracts for row in assessed):
        raise ValueError('Unknown source assessment contract')
    joint.validate_coverage(packets, assessed, model)
    return packets, assessed


def partition(packets, source_packets, source_rows):
    by_id = {row['id']: row for row in source_rows}
    by_hash = {judge.sha(packet): by_id[packet['id']] for packet in source_packets}
    reused, pending = [], []
    for packet in packets:
        previous = by_hash.get(judge.sha(packet))
        if previous is None:
            pending.append(packet)
        else:
            reused.append(previous)
    return reused, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--reuse-directory', type=Path, nargs='+', required=True)
    parser.add_argument('--model', default='claude-opus-5-5')
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    packets = read(args.directory/'packets.json')
    source_packets, source_rows, sources, seen = [], [], [], set()
    for directory in args.reuse_directory:
        previous_packets, previous_rows = completed_review(directory, args.model)
        indexed = {row['id']: row for row in previous_rows}
        for packet in previous_packets:
            identity = judge.sha(packet)
            if identity not in seen:
                source_packets.append(packet)
                source_rows.append(indexed[packet['id']])
                seen.add(identity)
        sources.append({'directory': str(directory.resolve()),
                        'aggregation_sha256': joint.digest(directory/'aggregation.json'),
                        'assessments_sha256': joint.digest(directory/'assessments.jsonl')})
    reused, pending = partition(packets, source_packets, source_rows)
    contract = {'packets_sha256': joint.digest(args.directory/'packets.json'),
                'sources_in_priority_order': sources,
                'model': args.model, 'workers': args.workers,
                'runner_sha256': joint.digest(Path(__file__)),
                'judge_runner_sha256': joint.digest(Path(judge.__file__)),
                'policy': 'Exact full packet identity only; retain the first valid prior judgment regardless of score.'}
    joint.frozen_write(args.directory/'reuse-contract.json', contract)
    fresh_dir = args.directory/'fresh'
    fresh_rows = []
    if pending:
        fresh_dir.mkdir(exist_ok=True)
        joint.frozen_write(fresh_dir/'packets.json', pending)
        subprocess.run([sys.executable, str(Path(joint.__file__).resolve()),
                        '--directory', str(fresh_dir), '--model', args.model,
                        '--workers', str(min(args.workers, len(pending)))], check=True)
        fresh_rows = rows(fresh_dir/'assessments.jsonl')
        joint.validate_coverage(pending, fresh_rows, args.model)
    assessed = reused + fresh_rows
    joint.validate_coverage(packets, assessed, args.model)
    indexed = {row['id']: row for row in assessed}
    output = ''.join(json.dumps(indexed[p['id']], ensure_ascii=False)+'\n' for p in packets)
    path = args.directory/'assessments.jsonl'
    if path.exists() and path.read_text() != output:
        raise ValueError('Refusing to replace a completed review')
    path.write_text(output)
    joint.frozen_write(args.directory/'reuse-aggregation.json', {
        'contract': contract, 'packets': len(packets), 'reused': len(reused), 'fresh': len(fresh_rows),
        'reused_ids': [r['id'] for r in reused],
        'fresh_ids': [r['id'] for r in fresh_rows],
        'assessment_sha256': joint.digest(path),
        'fresh_aggregation_sha256': joint.digest(fresh_dir/'aggregation.json') if pending else None,
        'note': 'Rows retain their original contract/model/timing. Reused assessments are not new CLI calls.'})
    print(json.dumps({'complete': len(assessed), 'reused': len(reused), 'fresh': len(fresh_rows)}))


if __name__ == '__main__':
    main()
