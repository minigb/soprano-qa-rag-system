# E37 재현/재개

완료된 결과는 followup-report.md, 소스는414a6c0, 양쪽 브랜치는codex/ku-background-knowledge다.
기존 완료 실행을 다시 생성할 필요는 없다. 중단된 실행을 재개할 때만 아래 동일 계약을 사용한다.

```bash
cd /home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/run_background_comparison.py --config evaluation/ku_background_2026-10-31/comparison-e37.config.json
/home/minhee/miniconda3/envs/soprano-qa/bin/python evaluation/supplement_background_comparison.py --config evaluation/ku_background_2026-10-31/comparison-e37.config.json
/home/minhee/miniconda3/envs/soprano-qa/bin/python evaluation/export_current_prompts.py --output evaluation/ku_background_2026-10-31/prompts-e37 --check
```

GPU inference/metric worker는 하나만 사용한다. 새 코드는 새 snapshot/config/output 경로로 평가한다.
동결된 결과·판정·전문가 정답·BERTScore 설정을 덮어쓰지 않는다. `runtime/`과 private-traces는 Git에서 제외하지만 이 기계에는 남아 있다.
다른 체크아웃에서 새로 실행하려면 e37-provenance.json의 source/model/corpus 해시를 맞추고 새 계약을 만든다.
기존 절대 경로나 package/source/settings 해시가 바뀐 상태를 기존 실행의 resume로 처리하지 않는다.

전체 결과는 comparison-2026-11-02-e37/, 추가 진단은 probe-e37.jsonl 및 probe-e37-judge/다.
추가23개 진단 점수/시간을 전체290개와 섞지 않았다. 일반 source 비교 및 긴 교과서 prompt가 실패한 E26–E36 출력도 보존한다.
`diagnose_*_legacy.py` 및 역사적 진단 스크립트의 docstring은 필요한 당시 source revision을 명시한다.
현재 production에서 제거된 API를 쓰는 역사적 실험은 해당 revision의 격리된 체크아웃에서만 재현한다.

생성/테스트 Python: /home/minhee/miniconda3/envs/soprano-qa/bin/python
BERTScore Python: /home/minhee/.cache/soprano-rag-eval-venv/bin/python
독립 judge: claude-opus-5-5 CLI. tools/hooks/MCP/session persistence를 끈 기존 runner를 사용한다.
main/live/dataset은 수정하지 않았고 reset-stop/notifier도 사용하지 않았다.
