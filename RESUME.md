# 확장460개 평가 실행 중 — 2026-11-02

사용자 요청: 기존 확장 평가460개를 현재 E37과 보안 적용 전 원본으로 새로 실행하고,
Claude Opus5.5로 의미 보존·자연스러움을 비교하며 전문가 기반 BERTScore≥0.90 확인.
같은 `codex/ku-background-knowledge` 브랜치. production E37은 변경하지 않는다.

현재 문서: [평가 방법/재개](evaluation/expanded_460_2026-11-02/README.md)
진행 상태: [analysis-state](evaluation/expanded_460_2026-11-02/analysis-state.json)
입력: [cases](evaluation/expanded_460_2026-11-02/cases.json)
설정: [config](evaluation/expanded_460_2026-11-02/config.json)

E37→원본 순으로 각각460개를 새로 생성한다. 생성 controller와 생성 완료 후
분석을 시작하는 controller가 실행 중이다. 새로 시작하기 전에 기존 프로세스를
확인해서 중복 실행하지 않는다. 분석은 지표→눈가림 Claude→집계 순서다. 현재 분석 재개 명령은
`evaluation/run_expanded_analysis_bounded.py`다. v1은 길이 사전 검사 후 대기 상태에서 종료했고
초과 답변2개를 계산 불가·보수적0점 기여로 명시하는 v2계약을 사용한다. 원질문105개는
길이 제한 이내이며, encoder·설정·문항·답변·참조는 변경하지 않았다.
전체 끝난 뒤 낮은 의미/자연스러움 평가를 직접 확인하고, 보고서·검증·Lab동기화·커밋이 남는다.
원본946개 vs현재953개 자료 차이를 발견했다. 공통946개/전문가KU122개는 동일하며
추가7개는 조수미 피드백이다. 이전 보고서의 동일corpus 표현을 정확히 한정해
새 README에 명시했다. 과거 결과와 참조를 바꾸지 않았다.

사용량/reset 중지와 notifier는 사용하지 않는다. 아래는 직전 E37 완료 기록이다.

---

# 후속 개선 완료 — E37, 2026-11-02

요청한 오거절3건과 p/PP 설명 개선을 마쳤다. BERTScore를 높이기 위한 재튜닝 없이
의미·자연스러움·전체 품질을 확인했다. 실행 중인 평가나 남은 필수 작업은 없다.

## 위치와 브랜치

- RAG: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
- Lab: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`, RAG는 `rag/`
- 양쪽 `codex/ku-background-knowledge`. 평가한 production 소스는 `414a6c0a830ffaf9c822574a66853ddcf46ea42c`.
- main/live/dataset은 변경하지 않았다. 사용량/reset 중지 조건과 Discord/notifier를 사용하지 않았다.

## 완료된 결과

- 전체290건 실행 오류0, 정상 오거절 **3/216→0/216**.
- 전문가 한국어 검증62/62응답, 고정 BERTScore F1 **0.9729869404**, 잘린 텍스트0.
- 주입 공격 성공0/58, 범위 밖 요청 성공0/12.
- Claude 중대한 출처 의미 변경0, 실질적 사실 오류0, 혼합/불확실 사실1.
- 같은 packet 평가의 종합 품질 **94.8653→96.5488/100**.
- 정상 서비스 지연 중앙값5.269→5.347초. 같은211개 성공요청의 요청별 차이 중앙값+0.039초.
- 이전 의미41개 전부 검토:39개 전체 packet 동일, 변경된p/PP2개 직접 대조.
- 정상220개 중211개 packet 동일, 변경된9개 전부 직접 검토.
- 원문165/165·영어번역38/38 일치. Root RAG648, Lab RAG648, Lab demo44 tests 통과.
- 양쪽21개 production 모듈·LLM설정이 평가 snapshot과 일치한다.

## 읽을 문서

- [현재 보고서](evaluation/ku_background_2026-10-31/followup-report.md)
- [전체 답변 비교](evaluation/ku_background_2026-10-31/comparison-2026-11-02-e37/review/review.html)
- [직접 의미 검토](evaluation/ku_background_2026-10-31/e37-manual-meaning-review.json)
- [현재 프롬프트](evaluation/ku_background_2026-10-31/prompts-e37/manifest.json)
- [최종 검증](evaluation/ku_background_2026-10-31/e37-verification.json)
- [수용 판정](evaluation/ku_background_2026-10-31/e37-quality-decision.json)
- [재현/재개 방법](evaluation/ku_background_2026-10-31/followup-reproduce.md)
- [실패 후보를 포함한 결정 기록](evaluation/ku_background_2026-10-31/followup-decision-log.json)

## 실제 구현과 해석

입력 판정은 음악 질문/공격/비음악 요청 범주를 사용하고, 일반 연습 라우팅과
Rosina 성격 검색 어휘를 보완했다. p/PP는 실제 출처의 시작 반주 표기와 공통
첫소리 조언만 확인해 비교하며, 원인·판본·작곡가 의도를 추측하지 않는다.
`llm_guarded_source_comparison` 답변도 모든 입력·출력·인용·범위 검토를 거친다.

출처 없는 일반 음높이 개념 질문은 짧은 질문/언어 입력과 제한된 추론을 사용한다.
전체 원래 문맥은 입력·출력 검토에 유지된다. 실패한 일반 비교 생성/캐시,
긴 교과서 정의 prompt는 제거했고 E26–E36 실패 기록은 보존했다.

일반 영어 조옮김의 옥타브 예외 표현, 레가토 연습의 구체성, 한국어 느린 연습의
표현과 기존 마이크 원문의 모호성·반복은 남는다. BERTScore는 사실 정확도가 아니다.
고정 전문가 split은 원질문 그룹이 분리됐지만 KU6개를 공유하고, 평가를 수정에
사용했으므로 새 holdout이 아니다. 공격0/58은 모든 미래 공격에 대한 보증이 아니다.

완료된 실행은 다시 돌리지 않는다. 추가 요청이 오면 새 snapshot/config/output을
만들고 기존 결과·정답·불리한 판정을 보존한다. 이전 E25 완료 상태는
`evaluation/ku_background_2026-10-31/resume-state.json` 안에 보존했다.
