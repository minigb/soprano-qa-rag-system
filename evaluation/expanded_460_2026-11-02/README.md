# 기존460개 확장 평가 — 원본 시스템과 E37

**평가 완료.** [결과 보고서](report.md): 의미 변경27→0건, 자연스러움3.50→2.89/4.
원질문 검증62개 BERTScore0.972987. [전체 답변 비교](review/review.html).
현재 실행 중인 작업은 없으며, 아래 명령은 재현/재개 절차다.
[최종 완료 기록](completion.json)은 자동 평가 후 직접 대조와 보고서 작성까지
포함한다. `analysis-state.json`은 자동 평가 controller가 종료할 때 남긴 기록이다.

이 폴더는 2026-11-02 사용자가 요청한 확장 평가다. 기존 문항·마디 조건·전문가
참조를 그대로 사용하며 두 버전에서 각각460개, 총920개 요청을 새로 실행한다.
이전 시스템 출력이나 이전 Claude 점수를 이번 결과로 재사용하지 않는다.

## 고정된 구성

| 유형 | 케이스 | 질문 출처 |
| --- | ---: | --- |
| original |105| 전문가 원질문79개와 기존 마디 조건 |
| paraphrase |306| 같은 원질문의237개 표현 변형과 기존 마디 조건 |
| ku_derived |49| 원질문이 없던 검토된 KU에서 만든36개 질문과 기존 마디 조건 |
| 합계 |460| 모두 한국어 |

정확히 같은 요청 입력은459개다. `in-flowery-clouds`의 첫 페르마타 질문1개가 원질문 회귀
조건과 표현 변형에 중복된다. 기존460개 정의를 유지하며 제거하지 않았다.
원질문105개는 개발43개·검증62개이며 이전 분할이 그대로다. 참조는 과거 모델
답변이 아니라 해당 문항에 적용되는 최종 전문가 KU 본문이다. 모든 참조를
현재 corpus와 과거 평가 파일에 대조했다. 질문 작성/출처 정보와 해시는
[inventory.json](inventory.json), 실제 입력은 [cases.json](cases.json)에 있다.

## 비교 조건

- 원본: 보안 적용 전 `f3074700103d60cf4bd8862fd655e51656f66f0c`.
- 현재: E37 `414a6c0a830ffaf9c822574a66853ddcf46ea42c`의 production 코드.
- 같은 Qwen3-14B Q6_K, Qwen3-Embedding-0.6B Q8_0, A6000, 기본 생성 설정.
- 각 버전의 native public service를 사용한다. E37의 원문 보존/비교 기능과
  원본의 생성·검증 동작을 그대로 평가한다. 평가 중 production 변경은 없다.
- **자료 차이:** 원본946개, E37 953개. 공통946개와 전문가 KU122개는 내용이
  동일하다. E37에는 조수미 피드백7개가 더 있고 원본은 이 자료형을 지원하지
  않는다. 따라서 전체 시스템의 비교이며 코드 변경만의 인과 효과가 아니다.
  이전290건 보고서의 “같은 corpus”라는 표현도 이 차이를 생략한 것이므로,
  동일한 공통 자료를 사용했다는 범위로 정정한다. 과거 측정 파일은 보존한다.
- 동일한 고정 순서로 E37을 먼저 실행한 뒤 원본을 실행한다. 버전별 warmup1개는
  평가에서 제외한다. GPU 추론·지표 계산은 순차 실행한다. 지연은 단일 실행의
  기술통계로 해석하며 통계적 속도 향상을 주장하지 않는다.

## Claude 의미·자연스러움 평가

Claude Opus5.5 CLI에 시스템명·버전·BERTScore를 숨긴 문항/답변/실제 근거/참조를
제공한다. 도구·MCP·세션 저장을 끄고 원문/질문 내부 지시를 평가 데이터로 다룬다.
질문·답변·근거·참조가 완전히 같은 packet만 중복 평가를 합친다. 낮은 점수는
재평가 사유가 아니다. 구조·인용·범위 검증에 실패한 호출만 최대3회 시도하며,
최초 유효 판정을 수용하고 실패 로그를 보존한다.

- 출처 의미: faithful / minor_drift / material_drift / not_applicable / uncertain.
- 사실성:0–4, 귀속·관련성·완전성:각0–2, 언어 적절성:0–1.
- **자연스러움:0–4 별도 척도.** 문장·연결·반복·불필요한 길이를 평가한다.
  유창한 오답이 가능하므로 사실성과 혼동하지 않는다. 유용하게 제시한 인용은
  허용하지만 관련 없는 긴 인용을 이어 붙이는 답변은 감점할 수 있다.
- 정상 답변의 자연스러움과 실패/거절을0으로 포함한 유효 평균을 함께 보고한다.
- 양쪽에서 답한 같은 문항의 승/무/패 및 source group을 묶은 paired bootstrap을
  보고한다. 이 구간은 문항 표본 불확실성만 다루며 평가자 변동을 포함하지 않는다.

[루브릭](judge-rubric.txt), [JSON 스키마](judge-schema.json), [별도 평가자 점검](calibration/purpose.json).
점검용3개는 사람이 주석한 benchmark가 아닌 명시적 합성 대조군이며460개에
포함하지 않는다. 유창한 오답과 정확하지만 반복적인 답변을 분리하는지 확인한다.

## BERTScore

이전 설정을 고정한다: `xlm-roberta-large`,18층,IDF=false,raw/unrescaled F1,
slow tokenizer, revision `c23d21b0620b635a76227c604d44e43a9f0ee389`.
설정 탐색·정답 수정·불리한 문항 제외는 하지 않는다. 답변 전체를 사용하고
길이 제한을 넘는 답변은 계산 불가로 명시하고 보수적 집계에0점을 배정한다.
pipeline의 unavailable/error도0점이다. 길이 초과의0은 실제 계산한 BERTScore가 아니다.
Claude가 추가로 판정한 순수 거절까지0점으로 환산한 보조 평균도 구분한다.

원질문105개, 기존 개발43개·검증62개, 표현 변형306개, KU 파생49개, 전체460개의
점수를 각각 보고한다. 기존 **원질문 검증62개 F1≥0.90**이 우선 확인 기준이며,
확장 문항도 별도로0.90 충족 여부를 확인한다. 이들은 재사용한 내부 평가이며
새 독립 holdout이 아니다. 질문 변형들은 같은 전문가 출처를 공유한다.

## 실행 및 재개

RAG worktree `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`,
branch `codex/ku-background-knowledge`에서 실행한다.

```bash
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/run_expanded_generation.py
/home/minhee/miniconda3/envs/soprano-qa/bin/python -u evaluation/run_expanded_analysis_bounded.py
```

첫 명령은 완료된 요청을 해시 계약으로 확인해 건너뛰고 미실행 요청만 실행한다.
기록된 실행 오류는 완료된 시도로 유지한다. 둘째 명령은 두 생성의 완료를 기다린
후 BERTScore와 Claude 평가, 비교 HTML/JSON을 만든다. 동일 controller를 중복
실행하지 않는다. 모든 작업의 완료 후 flagged 사례 직접 검토와 최종 보고서를
작성한다. 진행 상태는 `analysis-state.json`, 생성 진행은 각 버전의JSONL행 수다.
생성·판정의 비공개 원문 추적과 실패 원문은 Git에서 제외한다.

사용량/reset 중지 조건·Discord·process-stop notifier를 사용하지 않는다.
main/live/dataset을 수정하지 않는다.

## 길이 사전 검사에 따른 측정 제한 처리

E37 전체 생성 후, 지표·benchmark Claude 채점 전의 tokenizer 검사에서 표현 변형
2개가600/730토큰으로 확인됐다. 원질문105개와 모든 참조는512토큰 이내다.
답변·문항·참조·encoder·layer 등 설정은 그대로 두고, 초과 문항을 제외하지 않고
계산 불가 사유/길이와0점 기여를 명시하는 보수적 집계를 추가했다. 양쪽 시스템에
같은 정책을 적용한다. Claude는 초과 답변도 전문을 평가한다.

기존 분석 controller는 생성 완료를 기다리던 상태에서만 종료했다. 생성은 계속
진행됐다. v1스크립트·계약을 보존하고 v2분석 계약을 새로 만든다. 근거는
[길이 처리 명세](metric-length-amendment.json), [해당 문항](overlength-candidate-cases.json),
[독립 검증](metric-length-policy-verification.json), [v1종료 기록](analysis-v1-stop.json)에 있다.
