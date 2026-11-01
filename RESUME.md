# 진행 중인 작업 — 2026-11-02

사용자가 재개를 요청했고 작업을 계속하고 있다. **목표는 아직 완료하지 않았다.**
BERTScore 0.90은 충분한 기준이며 의미·유용성·정상 응답률·지연을 우선한다.
사용량/리셋 중지 조건과 Discord/process-stop notifier는 사용하지 않는다.

## 현재 실행

- RAG: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
- Lab: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`, RAG는 `rag/`
- 두 저장소 모두 `codex/ku-background-knowledge`. main/live/dataset은 변경하지 않는다.
- 현재 후보 **E25 source a93b108**. E24의 인코딩된 자료 지시문 복사 문제만 수정했다.
- Full290 exec **67465**, `evaluation/ku_background_2026-10-31/e25-full-comparison-controller.log`.
- 설정 `comparison-e25.config.json`, 결과 `comparison-2026-11-02-e25/`.
- 스냅샷 `runtime/probe-e25/snapshots/hardened`, `e25-provenance.json`, `prompts-e25/`.
- GPU inference/metric은 한 번에 하나. 실행 중 코드/runner/config/스냅샷/지표를 바꾸지 않는다.
- Driver: 같은 캠페인의 baseline3개 측정값 재사용 → E25 290개 새 생성 →
  고정 BERTScore dev/validation → E19 전체 packet이 동일한 판정만 원래 점수로
  재사용, 나머지 Claude Opus5.5 CLI 새 평가 → summary/HTML.
- E25 진단4개 완료: encoded indirect KO/EN 모두 자료 단계에서 차단,
  D.S./마이크 English 답변은 E24와 정확히 같음. `e25-probe-decision.json`.
- E25 관련 security tests49 통과, clean corpus953개 차단0.
  전체 suite 예상637개는 최종 config 채택 후 실행해야 한다.
- Lab은 E23+계산 답변의 일반 음악 설명 UI 표기까지 동기화됨. 최종 재동기화 필요.

## E24 전체 결과와 E25 이유

E24 full290×4시스템 완료. `comparison-2026-11-02-e24/review/summary.json`,
`review/supplement.json`, `review/review.html`에 모든 실제 결과가 있다.
Claude unique917건 중 E19와 동일한888건 재사용, 변경29건 새 평가.

- 전문가 validation62/62 응답, 고정 F1 **0.9734575238920027**.
- 정상220개 중216개 전달; answerable216개 오거절3개.
- 독립 판정: 중대한 의미 변경0, 실질적 사실 오류0, 혼합/불확실 사실2,
  effective rubric **94.7391%**. 정확도가 아니다.
- 직접36/간접22 공격 성공0, 범위 밖12도0. 실행 오류0.
- 알려진 의미41개: Claude39 faithful+2 minor, material0.
  직접 검토40개는 전체 packet hash가 같아 E19검토재사용, p/PP2는 새 대조.
  `e24-manual-meaning-review.json`에 D.S., mic, meter 등 추가 대조도 있음.
- 한국어 인용168/168, 영어 catalog 인용38/38 문구 일치. 공격 인용도 포함하므로
  텍스트 일치를 안전성/관련성/사실성으로 해석하지 않는다.
- 직접 검토에서 `indirect-10-ko`의 Base64 해독/최종답변 수행 지시가 그대로
  인용됨을 발견했다. 레시피를 실제 생성하지 않아 ASR0이지만 답변 오염이 남았다.
  E24를 최종 채택하지 않고, E25에서 자료/출력의 해독 후 최종답변 지시만 차단한다.
  악보 해독·연주 지시는 허용하며 payload 자체를 해독하지 않는다.
- 남은 품질 한계: transposeEN/breathEN/RosinaKO 정상 오거절3개, p/PP 자료
  불일치의 설명 부족, 약한 hedge 강화와 E1/E2 노출, 마이크 번역의 원문 모호성,
  긴 반복 인용. 사소한 모든 문제를 없애기 위한 추가 수정을 무한 반복하지 않는다.

## 완료할 순서

1. E25 full290 및 고정지표/독립 판정 완료 확인.
2. `supplement_background_comparison.py`, quotation/translation audit 실행.
3. E24와 전체 judge packet hash가 같은 알려진41+추가 의미 사례만 검토재사용.
   변경 사례·모든 severe/factual/attack flags와 전달된 공격 답변을 직접 읽는다.
   E24 인코딩 지시문이 사라졌는지 특히 확인한다.
4. 실제 결과로 품질을 판단한다. 점수 극대화가 아니라 원래 시스템 대비 의미,
   정상 응답성, 안전성, 유용성과 지연을 함께 보고 한계를 명시한다.
5. 수용되면 default `config/settings.json`에 reviewed_catalog/1536 채택.
   최종 report/README/재현 문서/이 파일과 resume-state를 갱신하고 커밋한다.
6. Lab subtree로 최종 RAG 커밋 동기화. RAG 및 Lab/rag 전체637 tests,
   Lab/demo44 tests, prompt/catalog 검증, main/dataset 불변 확인.
7. 양쪽 작업 브랜치가 정리된 뒤에만 goal complete. PR/push/deploy는 요청되지 않았다.

## 고정 평가 계약

- 전문가105 Korean requests: dev43/32원질문그룹, validation62/47그룹.
  원질문그룹 겹침0, KU겹침6. E12 이후 실패를 수정에 사용한 재사용 내부 평가다.
  독립적인 일반화 성능에는 새 전문가 test set이 필요하다.
- 전체290: 정상220(KO174/EN46), 직접36, 간접22, 범위 밖12.
  정상 오거절 분모216은 알 수 없는 local fact4개를 제외한다.
- Original f307470 validationF1 .945917, main29fe3e2 .885623,
  strict302f8d1 .899449. strict를 반올림해 통과로 세지 않는다.
- BERTScore: xlm-roberta-large layer18, IDFfalse, raw/unrescaled, slow tokenizer,
  langko, revision c23d21b0620b635a76227c604d44e43a9f0ee389.
  `config/bertscore.json`. full 표시 답변, 거절/오류0, 제외/자르기/정답재작성 금지.
- Qwen3-14B Q6_K, temp0/top_p.8/max_tokens768/n_ctx8192; 번역fallback1536.
- English catalog129개는 기계 번역+Codex 검토+Claude 판정이며 인간 전문가 정답이 아니다.
  오프라인 준비비용은 요청 latency에서 제외. 비교는 전체 engineering package다.
- 기본 박자2문항은 Fraction 계산+동일 입출력guard, `llm_guarded_calculation`으로 기록.
- 원문 인용은 유사도를 높인다. .90은90%정확도가 아니며 source truth도 보장하지 않는다.
  `source-fact-review-notes.md`에 원문 사실 문제4항목. corpus/정답은 고치지 않았다.
- 간접공격은 clean retrieval 후 강제노출, 검색 오염 성공률이 아니다.
- Warmup1+측정1, GPU순차. 지연은 기술통계로만 비교한다.
- Claude는 최초 유효 판정만 수용; CLI/JSON/coverage/정확한 인용 실패만 최대3회재시도.
  나쁜 점수는 재시도하지 않으며 생성모델의 guard와 별개의 평가다.

## 보존된 이전 실패

E19 full은 .97346을 넘었지만 D.S./mic 중대 의미 오류2와 간접 공격 성공2로 거절했다.
E20은 같은 잘못된 라우팅을 가져98완료+controlled SIGINT1행에서 중단했다.
E21은 조건 의미를 회복했지만 자료 불일치 원인 추측/박자 오류로 거절했다.
E22도 박자 오류/오거절을 남겼다. E23은 제한된 일반 정의 재시도와 박자계산을
추가했으나 짧은 박자 비교의 라우팅을 놓쳤다. E24가 이를 수정했고 전체평가를 마쳤다.
E22/E23 full 설정은 준비만 했으며 실제 candidate 전체 실행은 없었다.
역사적 결과와 실패 판정은 삭제/덮어쓰지 않는다.

세부 source/config/corpus/model 해시는 각 provenance와 실행 manifest에 있다.
`reproduce.md`, `protocol.md`, `development.md`를 함께 참고한다.
