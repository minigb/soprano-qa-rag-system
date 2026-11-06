"""Assess one frozen, deduplicated packet set with bounded parallel CLI workers.

Each worker uses the frozen meaning/naturalness rubric and checkpointed runner.
This does not select among repeated judgments: every packet belongs to exactly
one shard, and aggregation requires complete, valid coverage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from . import run_expanded_judge as judge
else:
    import run_expanded_judge as judge


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_write(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen input changed: ' + str(path))
    path.write_text(text)


def validate_coverage(packets, rows, model):
    expected = {p['id']: p for p in packets}
    if len(expected) != len(packets) or len(rows) != len(expected):
        raise ValueError('Incomplete or duplicate assessment coverage')
    if {r['id'] for r in rows} != set(expected):
        raise ValueError('Assessment IDs do not match the frozen packets')
    for row in rows:
        packet = expected[row['id']]
        if row['status'] != 'ok' or row['model'] != model or row['packet_hash'] != judge.sha(packet):
            raise ValueError('Failed, mismatched or altered assessment')
        judge.validate([row['assessment']], [packet])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True,
                        help='Directory created by prepare_judge_packets.py')
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--model', default='claude-opus-5-5')
    args = parser.parse_args()
    packet_path = args.directory/'packets.json'
    packets = json.loads(packet_path.read_text())
    if not packets or len({p['id'] for p in packets}) != len(packets):
        raise ValueError('Expected nonempty deduplicated packets')
    contract = {'packets_sha256': digest(packet_path), 'workers': args.workers, 'model': args.model,
                'runner_sha256': digest(Path(__file__)), 'judge_runner_sha256': digest(Path(judge.__file__)),
                'allocation': 'frozen packet order, round-robin; exactly one judgment per packet'}
    frozen_write(args.directory/'joint-contract.json', contract)
    workers = []
    for index in range(args.workers):
        directory = args.directory/('shard-%02d' % index)
        directory.mkdir(exist_ok=True)
        frozen_write(directory/'packets.json', packets[index::args.workers])
        workers.append(directory)

    def run(directory):
        with (directory/'run.log').open('a') as log:
            result = subprocess.run([sys.executable, str(Path(judge.__file__).resolve()),
                '--packets', str((directory/'packets.json').resolve()),
                '--output', str((directory/'assessments.jsonl').resolve()), '--model', args.model],
                stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'shard': directory.name, 'exit_code': result.returncode}), flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, workers))
    if any(codes):
        raise RuntimeError('A judge worker failed; valid checkpoints remain, no complete review was published')
    rows = [json.loads(line) for directory in workers
            for line in (directory/'assessments.jsonl').read_text().splitlines()]
    validate_coverage(packets, rows, args.model)
    indexed = {r['id']: r for r in rows}
    output = ''.join(json.dumps(indexed[p['id']], ensure_ascii=False)+'\n' for p in packets)
    target = args.directory/'assessments.jsonl'
    if target.exists() and target.read_text() != output:
        raise ValueError('Refusing to replace an existing completed review')
    target.write_text(output)
    frozen_write(args.directory/'aggregation.json', {
        'contract': contract, 'assessment_sha256': digest(target), 'packets': len(packets),
        'shards': {directory.name: {'assessment_sha256': digest(directory/'assessments.jsonl'),
                  'manifest_sha256': digest(directory/'assessments.jsonl.manifest.json')}
                   for directory in workers},
    })
    print(json.dumps({'completed': len(rows), 'model': args.model}), flush=True)


if __name__ == '__main__':
    main()
