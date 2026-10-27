#!/bin/bash
# go.sh <name> <variant.py|-> <translation_key|-> <logs...>: guard-only replay with rag-base code.
S=/tmp/claude-1000/-home-minhee-soprano-lab--claude-worktrees-keen-thompson-640a1e/9a594457-0e89-4393-8649-0218f67f8573/scratchpad
name=$1; variant=$2; key=$3; shift 3
cd "$S" || exit 1
env $( [ "$variant" != - ] && echo VARIANT=$variant ) $( [ "$key" != - ] && echo TRANSLATION_KEY=$key ) \
  SOPRANO_QA_SETTINGS=$S/settings_fix.json SOPRANO_QA_DATASET_ROOT=/home/minhee/soprano-qa-dataset \
  SOPRANO_QA_RAG_DATASET_ROOT=/home/minhee/soprano-qa-dataset \
  ~/miniconda3/envs/soprano-qa/bin/python -I guard_only.py $S/rag-base $S/go_$name.json "$@" 2>$S/go_$name.stderr >/dev/null
python3 - "$S/go_$name.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
flips = [r for r in rows if (r["old"] == "allow") != (r["new"] == "allow")]
print(f"{len(rows)} guard calls, {len(flips)} flips")
for r in flips:
    c = r["case"]
    print(f"  {c.get('kind', 'sample'):14} {c['language']} {c['piece_id'][:16]:16} {str(c['n'])[:14]:14} {r['stage']:9} {r['old']:6} -> {r['new']}")
PY
