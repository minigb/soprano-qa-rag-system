# 생성 프롬프트 의미 보존 비교 — 2026-11-06

작업 브랜치: `codex/helpful-rag-responses`.
작업 경로: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`.
현재 코드는 의미 우선 후보 J이며 기본 설정 전환은 보류했습니다. 최종 수치·적용 여부는 `report.md`와 `decision.json`을 확인합니다.

## 비교 계약

- 기존 전문가 기반 105개 (`family=original`) + 이전 거절 변형 4개 = 109개. 기존 평가 문항을 그대로 사용합니다.
- 한국어/영어 challenge 74개: 직접 공격30, 자료 주입20, 범위 밖12, 정상12.
- 지연: 기존20문항을 정순/역순으로 두 번. 모델·색인 로딩과 첫 워밍업 제외.
- Qwen3-14B-Q6_K, hybrid retrieval, corpus, 생성 설정, 입력 자료를 고정합니다.
- 온라인 입력/KU/출력 LLM 검사, 품질 재작성, 원문 대입을 추가하지 않습니다.
- Claude Opus 5.5 CLI는 오프라인 평가에만 사용합니다. 기존 rubric/schema와 첫 유효 판정 규칙을 유지합니다.
- BERTScore는 기존 XLM-R-large layer18/noIDF/rawF1/revision을 그대로 사용하고 답변 전문을 계산합니다.
- 개발에 이미 사용한 문항입니다. 새 holdout이나 일반적인 보안 보증으로 해석하지 않습니다.

## 실행 기록

- `timing-g`: 이전 단일 생성 후보 G의 소스 그대로 새 지연 측정.
- `expert-h`, `challenge-h`, `timing-h`: 간결한 의미 보존 규칙과 추가 보안 문구. 자연스러움·공격 방어 회귀로 미채택.
- `expert-i`, `challenge-i`, `timing-i`: 의미 보존 예시 추가 snapshot. 실제 생성 전 J로 보완했으며 한 번도 실행하지 않았습니다.
- `expert-j`, `challenge-j`, `timing-j`: 의미 보존 예시와 뒤쪽 범위 지시. 전문가 오류는 감소했으나 직접 공격9/30으로 G의8/30보다 증가했습니다.
- `challenge-k`: J에 가상 범위 예시 추가. 직접 공격12/30으로 실패하여 전문가/지연 평가를 시작하지 않았습니다. 전체 challenge 출력은 보존합니다.
- `expert-l`, `challenge-l`, `timing-l`: 최종 분리 비교. **G 대비 `single_pass_prompt.py` 하나만 변경**하고 G의 보안 문구와 배치, 입력 구성, 검색, 모델을 그대로 유지합니다.

`attempts.json`과 `selection-policy.json`에 변경 이유와 평가 전에 기록한 선택 기준·수정을 남겼습니다.
실패한 출력은 제거하지 않으며 완료된 답변/판정을 더 좋은 점수가 나올 때까지 재실행하지 않습니다.
J와 K에서 발생한 영어 질문 번역 오류는 공격 방어 성공으로 세지 않습니다.

## 재개 및 확인

```bash
python evaluation/prompt_fidelity_2026-11-06/status.py
python evaluation/prompt_fidelity_2026-11-06/analyze.py --candidate j
python evaluation/prompt_fidelity_2026-11-06/summarize_profile.py
python evaluation/prompt_fidelity_2026-11-06/verify.py --candidate j
```

완료된 생성 작업은 다시 실행하지 않습니다. 미완료 캡처만 해당 run의 `run.py`/`profile.py`로
이어가며, 먼저 기존 프로세스와 GPU 사용 상태를 확인해 중복 추론을 피합니다.
각 캡처는 자체 snapshot/runtime/settings와 SHA256 계약을 사용하므로 이후 소스 편집에 영향을 받지 않습니다.
source_head는 freeze 시점의 Git HEAD이며 실제 평가 소스는 source_files의 해시로 식별합니다.

오프라인 평가가 미완료인 경우 해당 expert/challenge 디렉터리의 `watch_judge.py`를 사용합니다.
첫 유효 판정은 보존하며, 재시도는 CLI/형식/인용 검증 실패에만 허용합니다.
질문 번역 같은 실행 오류로 답변이 없는 문항에는 Claude 판정을 만들어 넣지 않습니다.
실패를 정상적인 거절이나 방어 성공으로 합치지 않고 별도로 집계합니다.

모든 GPU 생성 종료와 전문가 평가 완료 뒤에만 고정 BERTScore를 실행합니다.

```bash
HF_HUB_OFFLINE=1 /home/minhee/.cache/soprano-rag-eval-venv/bin/python -u \
  evaluation/prompt_fidelity_2026-11-06/expert-j/score.py
```

기존 460개 데모, 원본/main 및 Lab 백엔드는 이번 부분 평가로 덮어쓰지 않습니다.
Discord/notifier 및 사용량/reset 중지 조건을 사용하지 않습니다.

현재 후보를 실행하려면 이 checkout에서 다음 설정을 명시합니다.

```bash
conda run -n soprano-qa python scripts/ask.py \
  --settings config/settings.single-pass.json \
  --piece una-voce-poco-fa --question '박자는 어떻게 처리할까?'
```

`config/settings.json`은 이전 검증 경로입니다. 이번 변경은 명시적 단일 생성 경로에 적용됐으며,
입력/KU/출력 모델 검사가 다시 추가된 것이 아닙니다. `report.md`와 `decision.json`에
의미 개선 및 공격 방어 미달을 함께 기록했습니다. 완료된 평가를 다시 실행할 필요는 없습니다.
