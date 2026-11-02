# 460개 답변 비교 localhost 데모 추가

사용자가 답변을 직접 확인할 웹 데모를 요청했고, 이후 **localhost 사용**을 지정했다.
전체 460문항 / 920개 답변을 `sites/answer-review/dist/`에 연결했다.
곡·출처·의미/자연스러움 필터, KU/참조 전문, 평가 숨기기, 문항 링크,
브라우저 로컬 메모·북마크 및 JSON 내보내기를 지원한다.

- 실행 주소: <http://localhost:8766/>. 루프백에 바인딩한 정적 서버다.
- 재시작: `python -m http.server 8766 --bind 127.0.0.1 --directory sites/answer-review/dist`.
- 원본 평가 답변 920개, 참조와 점수 매핑 검증 통과. 브라우저 동작과
  1440/1100/800/390px 화면 확인 통과. 평가 자체와 RAG 생성 코드는 변경하지 않았다.
- Lab에는 `demo/static/answer-review/`를 복사하고 기존 데모 상단에 진입 링크를 추가했다.
- 개인 메모는 브라우저별 localStorage이며 서버로 보내지 않는다.
- localhost 지정 전에 생성된 비공개 Sites 등록은 미배포 상태로 남겼다.
  소스나 평가 데이터는 업로드하지 않았다. 등록 ID는 원래 로컬 `.openai/hosting.json`에
  보존하고 Git에서 제외했다. 사용자의 요청 없이 외부 배포를 이어가지 않는다.

[데모 실행/검증 방법](sites/answer-review/README.md)

---

# 확장460개 평가 완료 — 자연스러움 개선 필요, 2026-11-02

현재 E37과 보안 적용 전 원본을 각각460개, 총920개로 새로 평가했다.
Claude Opus5.5의918개 고유packet 평가와20개 대표 사례 직접 대조를 완료했다.
실행 중인 평가 controller/worker는 없다. 요청한 평가는 완료했으며 production은 E37 그대로다.

- 같은 `codex/ku-background-knowledge` 브랜치: RAG와 Lab의 기존 worktree.
- 중대한 출처 의미 변경27/460→0/460, 경미한 변경은 현재2건.
- 자연스러움3.502→2.889/4, 같은460건 중 개선6/동률182/하락272건.
- 원질문105개 BERTScore0.972613, 기존 검증62개0.972987로0.90기준 충족.
- 전체460개 보수적평균0.967741. 표현 변형2개가600/730토큰으로512한도초과:
  전문 계산불가라고 표시하고0점 기여로 포함했다. 답변/참조/설정 변경·잘림 없음.
- 양쪽 실행 오류·거절0. 원문 인용459/459일치. 실제 모델·코드·입력 계약 검증 통과.
- 의미 보존은 개선됐지만 자연스러움까지 유지됐다고 주장할 수 없다.

[최종 보고서](evaluation/expanded_460_2026-11-02/report.md)
[답변 비교](evaluation/expanded_460_2026-11-02/review/review.html)
[평가 결론](evaluation/expanded_460_2026-11-02/decision.json)
[직접 검토](evaluation/expanded_460_2026-11-02/manual-review.json)
[검증](evaluation/expanded_460_2026-11-02/verification.json)
[재현 방법](evaluation/expanded_460_2026-11-02/README.md)

후속 품질 개선이 요청되면 중복 인용·문장 연결·원문의 국소적 지시어부터 다룬다.
조건/부정/귀속/범위를 보존하며 의미와 자연스러움을 별도로 검증해야 한다.
현재2개 크레센도 질문에는 선택 구간 없이 ‘이 부분’이 남고, Larghetto 설명의
모호성은 원문에도 있다. 이 평가는 개선 완료 선언이나 배포가 아니다.

원본946개와현재953개 corpus 차이도 확인했다. 공통946개/전문가KU122개는 동일하고
추가7개는 조수미 피드백이다. 이전 동일corpus 표현을 새 보고서에서 정정했다.
main/live/dataset 및 기존 평가 파일은 변경하지 않았다. 사용량/reset 중지와 notifier 미사용.

아래는 직전 E37 완료 기록이다.

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
