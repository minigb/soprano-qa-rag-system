# E25 평가 완료 — 최종 통합 검증 중

사용자 요청을 이어서 수행했고 의미와 전체 품질을 기준으로 E25를 채택했다.
BERTScore는 더 최적화하지 않는다. 사용량/리셋 중지 조건과 notifier는 없다.

- RAG: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
- Lab: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`, RAG는 `rag/`
- 두 저장소 모두 `codex/ku-background-knowledge`. main/live/dataset은 변경하지 않는다.
- 측정 소스 `a93b108`, 기본 LLM 설정은 평가 snapshot과 일치하도록 채택했다.
- E25 full290, BERTScore, Claude 평가, 직접 의미 검토 모두 완료. 실행 중인 평가 없음.
- 결과: `evaluation/ku_background_2026-10-31/report.md`.
- 실제 validationF1 **0.9734575239** /62개 모두 응답, 실행오류0.
- 공격 성공0/58, 정상 오거절3/216, Claude중대 의미변경0,
  실질적 사실오류0, 혼합/불확실 사실2, 종합루브릭94.8232/100.
- E24→E25 정상220개 전체 packet 동일. 달라진 것은 인코딩 공격1개이며
  자료의 해독/최종답변 지시 복사가 차단됨. 기존문제41개 검토도 그대로 유지.
- Root RAG 전체637 tests 통과(23.299초), catalog/prompt 검증 통과.

## 아직 완료할 작업

1. 최종 결과/default설정/report를 commit하고 Lab subtree에 동기화.
2. Lab/rag637 및 Lab/demo44 tests, 양쪽 실제 source/config 일치와 main/dataset불변 확인.
3. `final-verification.json`과 최종 RESUME/state 기록, docs만 최종 동기화, clean커밋 확인.
4. 그 뒤 goal complete. PR/push/deploy는 요청되지 않았다.

## 의미/평가 한계

남은 정상거절은 transposeEN, breathEN, RosinaKO다. p/PP자료의 불일치 설명,
약한 hedge강화와 E1/E2노출, 마이크번역의 원문 모호성, 반복인용도 남는다.
같은p/PP답변에 E24 judge는minor, E25는faithful을 주었지만 직접 검토 지적은 보존했다.

105개 전문가 한국어 reference(dev43/validation62), 원질문그룹 겹침0/KU겹침6.
관측실패를 수정에 사용한 재사용 내부평가다. 새 독립 testset 성능을 주장하지 않는다.
xlm-roberta-large layer18, IDFfalse, raw/unrescaled, slow tokenizer, langko,
pinned revision. `config/bertscore.json`은 개발18설정 sweep 후 변경하지 않았다.
원문인용이 유사도를 높이므로0.97346은97%정확도가 아니다.
영어 catalog129는 기계 작성/모델 검토이며 전문가 영어 정답이 아니다.
기본박자2문항은 계산+guard로 처리하며API에계산임을기록한다.
간접공격은 검색후 강제노출, 지연은 단일순차측정이다. 최초 대비공통209질문
중앙값3.35→5.25초의 보안비용을 명시했다.

기록 경로는 모두 `evaluation/ku_background_2026-10-31/` 기준:
`e25-quality-decision.json`, `e25-manual-meaning-review.json`,
`comparison-2026-11-02-e25/`, `e25-provenance.json`, `prompts-e25/`.
`reproduce.md`, `protocol.md`, `development.md`에 재현/실패후보/연구한계가 있다.
E19/E24를 포함한 실패 후보와 모든 원래판정은 삭제/덮어쓰지 않았다.
