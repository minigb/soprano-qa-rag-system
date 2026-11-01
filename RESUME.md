# 진행 중인 작업 — 2026-11-02

사용자 요청으로 재개했다. **목표는 아직 완료하지 않았다.** BERTScore 0.90은
충분한 통과 기준이며 의미·유용성·정상 질문 응답률·속도를 우선한다.
사용량/리셋 중지 조건과 Discord/process-stop notifier는 사용하지 않는다.

## 작업 위치

- RAG: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
- Lab: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`, RAG는 `rag/`
- 두 저장소 모두 `codex/ku-background-knowledge`. main/live는 변경하지 않는다.
- 현재 후보 **E22 source e0a2cae**. 624개 전체 테스트와 이후 추가한 평가 무결성
  검사를 포함한 14개 targeted tests가 통과했다. 전체 suite는 현재 627개다.
- Lab은 E19/a61a064까지 동기화했다. 최종 코드·결과·설정 동기화가 필요하다.

## 현재 실행

경로는 아래에서 모두 `evaluation/ku_background_2026-10-31/` 기준이다.
GPU inference/metric은 한 번에 하나만 실행한다. PID만 보고 종료하지 않는다.
재부팅 후 live GPU 모델은 없었으며 서비스를 다시 시작하지 않았다.

- E22 진단50 생성 → independent Claude 판정 → 요약 큐 실행 중.
- exec session **34864**, `e22-probe-controller.log`.
- 스냅샷 `runtime/probe-e22/snapshots/hardened`, `e22-provenance.json`.
- `e21_probe_cases.json`의 동일50개, 결과 `probe-e22.jsonl`, `judge-e22/`,
  `review-e22/`. 새 입력으로 유리한 결과를 고르는 재실행이 아니다.
- 현재 프롬프트 `prompts-e22/`. BERTScore/정답/corpus/catalog는 그대로다.
- E22 전체290 설정·실행은 진단을 직접 확인한 뒤 준비해야 한다.
- default settings의 reviewed_catalog 채택은 전체 비교 후에 결정한다.

## 완료·보존된 실패 평가

**E19** 전체290×4시스템과 unique919 independent Claude 판정 완료.
`comparison-2026-11-01/review/summary.json`, `review/review.html`,
`review/supplement.json`에 원래/초기보안/엄격보안/후보 비교가 있다.
E19 validation F1 **0.9734575**, 62/62응답이지만 채택하지 않았다.
D.S.를 Da Capo로 바꾸고 마이크 사용 조건의 발성 의미를 바꾼 중대 오류2건,
간접공격 성공2/22, 정상 오거절3/216, 잘못된 일반 박자 설명이 확인됐다.
후보 effective rubric93.4343%, 원래89.6465%는 정확도나 통과 판정이 아니다.
자료 제어문 복사4건과 judge가 성공으로 센2건을 구별한다.

**E20** `f72fd24`: 역할재현/자료의 응답형식 지시 검사. 정상 corpus953개 차단0.
E19 독립 판정에서 의미 오류가 드러나, 같은 경로를 가진 E20 실행을 중단했다.
98개 완료+controlled SIGINT1개 행, `comparison-2026-11-02-e20/interruption.json`.
부분 실행을 전체 비교/모델 실행오류로 주장하지 않는다. 추가 practice8개는 생성 완료.

**E21** `8d0548a`: 정의 질문도 실제 KU 관련성을 검사하고, 작품명이 있는 비교는
자료 경로를 유지했다. D.S./마이크 의미는 회복했지만 자료 불일치의 원인을 추측하고
한국어 박자/새9/8진단도 틀렸다. 진단50과 독립 평가 완료, 채택하지 않았다.
특히 p/PP 추측 한 건은 judge가 놓쳤으므로 직접 대조 결과도 보존한다.

**E22**는 간결한 음악 task admission, 정의용 passage selection 지시,
자료 조언을 배경 문장으로 다시 합성하지 않기, 같은 박자 규칙의 한국어 설명,
성악 서비스에서 일반 연습을 가창으로 해석하는 지침을 적용한다.
`e22-refinement-plan.json` 참조. 아직 결과를 통과로 가정하지 않는다.

## 이어서 완료할 순서

1. E22 진단50 결과/Claude 판정을 확인한다. D.S., mic KO/EN, p/PP1·2,
   KO/EN meter+새 transfer4, legato EN, transposition EN을 직접 대조한다.
2. 의미/응답성에 수용할 수 없는 오류가 없으면 새 comparison-e22 config와
   output을 준비한다. baseline3개의 실제290행/metric은 E19에서 byte-identical
   복사하고 baseline-reuse 해시를 기록한다. candidate290은 새로 생성한다.
3. `run_background_comparison.py`는 이제 optional `judge_reuse_directory`를 받는다.
   E19 완료 `comparison-2026-11-01/judge`를 지정하면 동일 packet+model/rubric/schema
   판정만 원래 contract 그대로 재사용한다. 불리한 점수도 유지한다. 다른 packet은
   새 independent Claude 판정이며 `reuse-aggregation.json`에 구별한다.
   이전 E19 driver hash는 historical artifact이므로 현재 driver로 E19를 resume하지 않는다.
4. 전체 validation 실제F1, 41개 알려진 의미 사례, 모든 material/factual/attack
   판정을 직접 확인한다. 기존 직접 대조는 전체 judge packet hash가 같을 때만 재사용.
   quotation/translation audit와 language×kind/paired-latency supplement를 실행한다.
5. 수용할 수 있을 때 기본 `config/settings.json`에 reviewed_catalog/1536을 채택한다.
   README, 연구/최종 보고서, 재현/이 체크포인트를 실제 결과로 갱신한다.
6. commit → Lab subtree 동기화 → RAG/Lab RAG 전체 tests 및 Lab demo44,
   catalog/prompt 검증, main/dataset 불변 확인. 그 뒤에만 goal complete.

## 고정 평가 계약

- 전문가105: 개발43/validation62, 원질문그룹 겹침0, KU겹침6.
- E12 이후 validation와 관측된 실패를 수정에 사용한 **재사용 내부 평가**다.
  독립 일반화 성능에는 새 전문가 test set이 필요하다.
- 전체290: 정상220(KO174/EN46), 직접36, 간접22, 범위 밖12.
  정상 오거절 분모216은 알 수 없는 local fact4개를 제외한다.
- Fresh baseline validation: pre-security f307470 **0.945917**, main29fe3e2
  **0.885623**, strict302f8d1 **0.899449**. strict를 반올림해 통과로 세지 않는다.
- BERTScore: xlm-roberta-large layer18, IDF false, raw/unrescaled, slow tokenizer,
  lang ko, revision c23d21b0620b635a76227c604d44e43a9f0ee389. 설정 변경 금지.
  전체 표시 답변, 거절/오류0, 실패 제외/답변 자르기/reference 재작성 금지.
- Qwen3-14B Q6_K, temp0/top_p.8/max_tokens768/n_ctx8192; 번역 fallback1536.
- 영어 catalog129개는 전문가 source의 기계 번역+Codex 검토+별도 Claude 판정이며
  인간 전문가 영어 정답이 아니다. 준비비용은 요청 latency에 포함되지 않는다.
- 원문 인용은 높은 중복도를 만든다. 유사도는 의미/사실/관련성/안전성의 증명이 아니다.
  원문 사실 문제는 `source-fact-review-notes.md`에 분리했고 평가중 정답을 고치지 않았다.
- 간접공격은 clean retrieval 후 강제노출, 실제 검색 오염 성공률이 아니다.
- warmup1+측정1, 단일GPU순차. 지연은 기술 통계이며 통계적 speedup 주장이 아니다.

정확한 해시는 `evaluation/ku_background_2026-10-31/resume-state.json`에 있다.
