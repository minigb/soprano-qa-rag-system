"""Build a full-packet meaning-review worklist; never invent manual judgments.

An earlier manual review is reusable only when its saved full packet hash is
verified and the entire new packet is identical. Every changed normal answer
and newly changed delivered attack remains pending direct review.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--previous-directory', type=Path, required=True)
    parser.add_argument('--previous-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    packets = {p['id']: p for p in read(args.directory/'judge/packets.json')}
    old_packets = {p['id']: p for p in read(args.previous_directory/'judge/packets.json')}
    mapping = {m['key']: m for m in read(args.directory/'judge/mapping.json')}
    old_mapping = {m['case_id']: m for m in read(args.previous_directory/'judge/mapping.json')
                   if m['key'].startswith('candidate:')}
    candidates = {m['case_id']: m for m in mapping.values() if m['key'].startswith('candidate:')}
    previous = read(args.previous_review)
    if len(candidates) != len(old_mapping):
        raise ValueError('Expected identical complete cohorts')
    changed, reused, pending, unchanged_attacks = [], {}, {}, []
    old_reviews = {**previous['cases'], **previous['additional_checks']}
    for case_id, row in candidates.items():
        packet = packets.get(row['review_id'])
        old = old_packets.get(old_mapping[case_id]['review_id'])
        same = packet == old
        if not same:
            changed.append(case_id)
        delivered_attack = row['kind'] != 'benign' and row['generation_mode'] != 'unavailable'
        needs_review = case_id in old_reviews or (row['kind'] == 'benign' and not same) or delivered_attack
        if not needs_review:
            continue
        review = old_reviews.get(case_id)
        if same and review and digest(old) == review.get('judge_packet_sha256'):
            reused[case_id] = {**review, 'review_reused_from': str(args.previous_review),
                               'reuse_basis': 'Verified previous full packet SHA256 and exact new packet equality'}
        elif same and delivered_attack:
            unchanged_attacks.append(case_id)
        else:
            pending[case_id] = {'status': 'pending_direct_review', 'kind': row['kind'],
                'packet_sha256': digest(packet), 'question': packet['question'] if packet else None,
                'before': old, 'after': packet}
    result = {'status': 'worklist_not_a_completed_review', 'candidate_directory': str(args.directory),
        'previous_review_sha256': hashlib.sha256(args.previous_review.read_bytes()).hexdigest(),
        'changed_packet_case_ids': sorted(changed), 'exact_previous_reviews_reused': reused,
        'unchanged_delivered_attack_packets': sorted(unchanged_attacks), 'pending_direct_review': pending}
    text = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError('Refusing to replace a frozen worklist')
    args.output.write_text(text)
    print(json.dumps({'changed_packets': len(changed), 'reused_manual_reviews': len(reused),
                      'pending_direct_review': len(pending), 'unchanged_delivered_attacks': len(unchanged_attacks)}))


if __name__ == '__main__':
    main()
