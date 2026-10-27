#!/bin/bash
# run_diag.sh <rag_root> <cases.json> <out.json>: guard diagnostics on the GPU.
S=/tmp/claude-1000/-home-minhee-soprano-lab--claude-worktrees-keen-thompson-640a1e/9a594457-0e89-4393-8649-0218f67f8573/scratchpad
cd "$S" || exit 1
SOPRANO_QA_SETTINGS=$S/settings_fix.json \
SOPRANO_QA_DATASET_ROOT=/home/minhee/soprano-qa-dataset \
SOPRANO_QA_RAG_DATASET_ROOT=/home/minhee/soprano-qa-dataset \
  ~/miniconda3/envs/soprano-qa/bin/python -I "$S/diagnose_guards.py" "$1" "$2" "$3" 2>"$3.stderr" \
  | grep --line-buffered -E -- "->"
