# 후속 개선 진행 — 2026-11-02

E25 완료 후 사용자가 남은 오거절 3건과 p/PP 설명 개선을 요청했다.
현재 같은 브랜치에서 E26 후보를 검증 중이며 완료 판정 전이다.
입력 판정은 음악/공격/비음악 범주, 일반 연습은 일반 설명 경로, 성격 묘사는
검색 어휘 보완, 여러 출처는 인용을 갖춘 비교 설명을 시험한다.
BERTScore 설정/원문/평가 정답은 변경하지 않았다.
진단: e26-input-diagnostic.json, e26-admission-diagnostic.json, e26_probe_cases.json.
전체 평가와 Claude 판단, Lab 동기화는 아직 남았다.

이하 E25의 이전 완료 기록:

요청한 BERTScore 목표와 의미·품질 검토를 완료했다. 최종 구현은 E25,
측정 소스 `a93b108`이며 평가한 reviewed_catalog/1536 설정을 기본값으로 채택했다.
실행 중인 평가나 남은 구현 작업은 없다. 사용량/리셋 중지 조건과 notifier는 사용하지 않았다.

## 결과와 위치

- RAG: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
- Lab: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`, RAG는 `rag/`
- 양쪽 `codex/ku-background-knowledge` 브랜치. main/live/dataset은 기존 상태 그대로다.
- [최종 보고서](evaluation/ku_background_2026-10-31/report.md)
- [전체 답변 비교](evaluation/ku_background_2026-10-31/comparison-2026-11-02-e25/review/review.html)
- [현재 프롬프트](evaluation/ku_background_2026-10-31/prompts-e25/)
- [최종 검증](evaluation/ku_background_2026-10-31/final-verification.json)
- [재현 방법](evaluation/ku_background_2026-10-31/reproduce.md)

## 실제 확인한 결과

전문가 Korean validation62/62응답, 고정 BERTScore F1 **0.9734575239**.
전체290요청 실행오류0, 주입공격 성공0/58, 범위밖요청 성공0/12,
정상 오거절3/216, Claude 중대한 의미 변경0, 실질적 사실 오류0,
혼합/불확실 사실2, 종합루브릭94.8232/100.

알려진 의미41개와 D.S./마이크조건/박자설명을 대조했다.
E24→E25 정상220개 전체 packet은 동일하며, 인코딩 지시문을 그대로 인용하던
공격1개만 차단 응답으로 바뀌었다. LLM판정과 별도의 직접 검토 지적을 보존했다.
Root RAG637, Lab RAG637, Lab demo44 tests와 catalog/prompt 검증이 통과했다.
양쪽 source20모듈·corpus·LLM설정은 평가 snapshot과 일치한다.

## 해석과 이후 작업

BERTScore0.90은 충분한 기준이며 점수를 더 높이는 튜닝은 하지 않았다.
원문인용이 유사도를 높이므로0.97346은97%정확도가 아니다.
평가는 실패를 수정에 사용한 재사용 내부평가다. 전문가dev43/validation62는
원질문그룹이 겹치지 않지만 KU6개를 공유한다. 독립 일반화 성능을 주장하지 않는다.

남은 문제는 영어조옮김·영어호흡연습·한국어Rosina해석의 거절3개,
p/PP자료불일치 설명 부족, 작은 hedge/귀속 문제, 마이크원문 모호성, 긴반복인용이다.
최초 대비 공통209개 정상질문 중앙값은3.35→5.25초다.
영어catalog129는 기계작성/모델검토이며 전문가영어정답이 아니다.
기본박자는 계산+guard로 처리하며 API에계산임을기록한다.
간접공격은 검색후 강제노출이며0/58은모든미래공격의방어보증이아니다.

완료된 실행을 다시 시작할 필요는 없다. 추가 변경을 요청받으면 새 snapshot,
config, output을 만들고 기존 실패 후보·reference·판정을 유지한다.
[resume-state.json](evaluation/ku_background_2026-10-31/resume-state.json)에
완료 상태와 검증 파일 해시를 기록했다.
