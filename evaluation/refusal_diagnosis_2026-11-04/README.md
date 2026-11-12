# 고정 거절 문구의 발생 경로 진단

현재 데모의 거절 5건을 같은 Qwen3-14B-Q6_K, 설정, 질문, 코드 스냅샷으로
재실행했다. 5건 모두 기존 사용자 응답과 `unavailable_reason`이 재현됐다.
현재 서비스 코드, 원본 시스템, 데모 답변은 수정하지 않았다.
진단 브랜치는 `codex/refusal-path-audit`이다.

## 직접적인 변경 지점

`29fe3e2db66dde5bb2604652c8f4f8daddb0de46`의 `soprano_qa/service.py` 변경이
기존 `except GeneratedAnswerRejected` 복구 경로를 제거했다. 원본은 비어 있지
않은 초안을 표시용으로 정리한 뒤 반환하고, 일부 검증 실패를
`generation_validation_warning = str(exc)`로 기록했다. 보안 도입 후에는
`SecurityViolation`과 `GeneratedAnswerRejected`를 하나의 공개 예외 처리에서
받아 고정 문구로 교체한다.

아래 행 번호는 현재 작업 소스
`/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
기준이다. 동일한 코드 복사본은 이 브랜치의
`evaluation/latest_demo_2026-11-03/snapshot/soprano_qa/`에 있다.

- `soprano_qa/answer.py:144`: 고정 한국어 문구 정의.
- `soprano_qa/service.py:478`: 두 종류의 예외를 같은 응답으로 처리.
- `soprano_qa/service.py:492`: `no_corpus_evidence_message(language)` 반환.
- `soprano_qa/service.py:494`: 실제 원인과 무관하게 `answer_basis=no_corpus_evidence`.
- `soprano_qa/service.py:496`: 경고 필드를 `None`으로 설정.
- `soprano_qa/service.py:503`: 검색 진단과 근거 목록을 비움.
- `soprano_qa/llm.py:371`: 한 번 다시 작성해도 출력 검사나 최종 검증에 실패하면
  예외를 다시 올림. 유용한 부분 답변을 돌려주는 경로는 없음.

원본의 대응 경로는 보존된 pre-injection 체크아웃의
`soprano_qa/service.py:392`에 있다. 현재 생성 프롬프트는 일반 음악 지식을
허용하지만, 이 허용이 검증 실패 후의 부분 답변이나 일반 설명 반환을 보장하지
않는다. 두 종류의 문제를 함께 차단하는 코드가 그 뒤에 남아 있다.

## 5건에서 실제 발생한 일

모든 문항에 관련 KU가 검색됐고, 입력 검사는 정상 음악 질문으로 허용했다.
추가 KU 선택도 관련 자료를 유지했다. 각 문항에서 초안과 한 번의 재작성 후
최종적으로 고정 문구가 반환됐다. 이번 5건은 입력 공격 판정이나 검색 실패가 아니다.

| 사례 | 건수 | 실제 차단 지점 |
| --- | ---: | --- |
| Nella fantasia 마지막 음의 길이 및 표현 변형 | 4 | LLM 출력 검사는 통과. 그 뒤 규칙 검사가 `Generated answer overgeneralizes local evidence`를 발생시킴 |
| Die Forelle에서 다이나믹으로 이야기·메시지를 표현하는 방법 | 1 | 출처와 동일한 문장의 두 초안 모두 LLM 출력 검사에서 거부됨 |

마지막 음 질문은 `sumijo-nella-fantasia-039-ku-01`과
`nella-fantasia-ku-010`을 함께 검색·선택했다. 생성 초안에는 악보의 음가를
존중하되 장르와 반주 흐름에 맞춰 자연스럽게 마무리하라는 설명이 있었다.
출력 검사도 두 번 모두 허용했다.

하지만 `answer_overgeneralizes_local_examples`는 같은 문장에 `local_example`
자료가 인용되고 `general_evidence` 자료가 없으면, `곡 전체`, `경우가 많` 등
표현을 감지해 차단한다 (`soprano_qa/answer.py:1397`). 함께 인용된 조수미
자료는 `unspecified_scope`여서 일반 근거 예외에 들어가지 않는다. 원문 문장
일치 예외인 `_copies_scoped_source_sentence`는 인용된 local 자료만 검사한다.
따라서 다른 인용 출처에 해당 표현이 있어도, 또는 원문의 조건을 유지한
바꿔 쓰기여도 원문 일치 예외를 통과하지 못할 수 있다.

예를 들어 실제로 차단된 문장 중에는 다음이 있다.

> 특히 대중적인 레파토리에서는 반주의 마지막 울림까지 성악의 소리를 연결하여 곡 전체를 하나의 호흡으로 마무리하는 경우가 있어요.

이 문장은 조수미 자료의 문장과 같은 내용이지만, 내부 인용이 `[E1, E2]`여서
국소 자료 인용에 대한 규칙에 걸렸다. 일부 다른 초안에는 빈도 표현의 강화나
중복도 있어 모든 문장을 무조건 적절하다고 판정한 것은 아니다. 핵심 문제는
검증 의심이 유용한 나머지 답변까지 일괄 폐기하게 된다는 점이다.

Die Forelle의 경우 두 초안 모두 다음 전문가 KU 문장과 정확히 같았다.

> 《Die Forelle》에서는 다이나믹을 잘 활용하여 이야기가 전하는 내용과 메시지를 장면마다 표현하는 데 집중해 연습하는 것이 좋다.

그럼에도 출력 검사가 두 번 거부했다. 이 검사는 근거·의미·보안 문제를 하나의
`allow` 참/거짓으로 판정하며, 불확실하면 거부하도록 지시한다
(`security.py:246`, `security.py:281`, `security.py:511`). 판정 이유는 반환하지
않으므로 이 한 건에서 모델이 어떤 규칙을 문제 삼았는지는 알 수 없다.

## 이전 요청과 현재 동작의 차이

`codex/rag-component-ablation`의 별도 실험에서는 경고와 답변 제공, 일반 지식
보완 정책을 비교했지만 실제 서비스에는 채택하지 않았다. 해당 실험의
`ablation/README.md:9`에도 미적용 상태가 명시돼 있다. 이후 비교 데모는 그
실험 조건이 아니라 실제 현재 서비스 코드를 캡처했으므로 차단 경로가 유지됐다.

사용자가 요청한 동작을 구현하려면 보안·업무 범위 판정과 답변 품질 경고의
처리를 구분해야 한다. 정상 음악 질문의 근거·표현·인용 검증 문제는 가능한
부분 답변, 일반 음악 설명, 불확실성 안내로 이어져야 한다. 비음악 요청에는
성악 도우미의 범위를 설명하는 응답을 제공해야 한다. 악성 지시를 따르거나
알 수 없는 구체적 사실을 확정하는 것까지 허용하자는 의미는 아니다.
이번 작업에서는 원인만 진단했고 이 정책 변경은 구현하지 않았다.

## 검증과 재현

`run.py`는 원래 함수의 인자·반환값·예외를 보존하는 관찰용 래퍼만 적용한다.
모델 호출을 추가하거나 판정·초안을 수정하지 않는다. 완료 사례를 덮어쓰지
않고, 최초 저장한 진단 결과를 유지한다. `traces.jsonl`에는 생성 후의 초안,
검색·선택된 자료, 검사 통과/실패와 예외가 있다. 모델의 비공개 추론은 수집하지
않았다. `verification.json`은 원본 캡처와 현재 소스의 해시 확인 결과를 남긴다.

`analyze.py`는 저장된 자료로 순수 검증 함수만 다시 호출하여 차단된 문장과
일치한 표현을 `findings.json`에 기록한다. 추가 LLM 호출은 없다.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python evaluation/refusal_diagnosis_2026-11-04/run.py
PYTHONDONTWRITEBYTECODE=1 /home/minhee/miniconda3/envs/soprano-qa/bin/python evaluation/refusal_diagnosis_2026-11-04/analyze.py
```

진단 결과는 460문항 데모를 새 평가 결과로 교체하지 않으며, 정책 변경의 품질
효과를 측정한 실험도 아니다. 보존된 출력의 차단 경로를 확인한 재현 기록이다.
