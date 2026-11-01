# 진행 중인 작업 — 2026-11-02 KST

사용자 요청으로 11월 1일에 재개했다. **목표는 아직 달성하지 않았다.**
BERTScore 0.90은 충분한 통과 기준이며, 의미·유용성·정상 질문 응답률·속도를
우선한다. 사용량/리셋 중지 조건은 없다. Discord와 process-stop notifier도
사용하지 않는다. main과 실제 서비스를 변경하지 않는다.

## 작업 위치

- RAG: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
- Lab: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`, RAG는 `rag/`
- 둘 다 `codex/ku-background-knowledge`. Lab 동기화는 최종 비교 후 필요하다.
- E17 소스: `ba65680` (E16의 번역 catalog를 유지). 새 영어 번역 경로는 평가 스냅샷에만 활성화되어 있다.
- 데이터셋: `/home/minhee/soprano-qa-dataset`, `1ae33e8`.
- 재부팅 후 GPU에 live 모델은 없었다. 서비스를 다시 시작하지 않았다.
- 이 파일은 실행 중 체크포인트이다. 완료된 종료 체크포인트가 아니다.

## 최근 결과와 결정

E15 개발 전문가 43개 F1은 **0.972074**였지만, 알려진 영어 9개에서
발음 권고 반전·쉼표/rest를 comma로 번역·마디를 phrase로 번역하는
중대한 오류가 나와 **채택하지 않았다**. 원본 결과와 판정은 보존했다.

E16은 전문가/성악가 피드백 KU 전체 129개에 대한 영어 번역을 준비했다.
평가 질문별 답변 캐시가 아니라 **정확한 KU 원문 SHA256**에만 대응한다.
Codex가 작성·확인하고 Claude Opus 5.5가 별도로 검토했다. 인간 전문가의
영어 주석이 아니며, 고정 한국어 BERTScore reference를 변경하지 않는다.
최초 검토 124 faithful/5 minor drift/0 material drift; 실제 수정한 5개를
다시 검토하여 최종 129개 모두 faithful 판정. 번역 충실도는 원문 사실성,
관련성, 전체 서비스 품질을 증명하지 않는다. 원문 사실 오류 3개는 별도 기록.

원문이 변경되거나 새로 추가되면 기존 segmented 번역을 사용한다. 원문/출력
공격 검사와 최종 의미 검토를 그대로 실행한다. 일반 음악 지식은 별도 생성한다.
오프라인 번역 준비 비용은 요청 지연에 포함되지 않는 추가 자원임을 공개한다.
회귀 테스트 **606개 통과**. 실제 모델 E17 진단/전체 비교는 아직 필요하다.

E16의 실제 모델 probe를 시작하기 전에, 이전 E15 일반 연습 질문에서
KU를 반복 나열하고 질문에 직접 답하지 않는 문제를 확인했다. E17은 selector에
KU의 원래 질문을 데이터로 추가하고 요청한 과제와 관련성을 판단하도록 한다.
한국어 구성에도 중복 자료 생략 지침을 적용하고 BPM의 박 단위를 명시하도록 한다.
E16 번역 catalog와 모든 평가 문항/reference/metric은 그대로 유지한다.

## 현재 실행과 이어갈 순서

먼저 프로세스와 로그를 확인한다. GPU 평가 프로세스는 한 번에 하나만 실행한다.
PID는 종료 후 재사용될 수 있으므로 명령행을 확인하지 않고 종료하지 않는다.

- `evaluation/ku_background_2026-10-31/e17-queue.log`
- 로컬 큐 스크립트 `evaluation/ku_background_2026-10-31/runtime/run_e17_queue.py`
- 새 비교 폴더 `evaluation/ku_background_2026-10-31/comparison-2026-11-01/`
- E17 스냅샷 `runtime/probe-e17/snapshots/hardened`, 공개 `e17-provenance.json`

원래 시스템 `f307470` 290개는 완료했고 개발/validation F1은
**0.948348 / 0.945917**이다. 실행 오류를 포함한 전체 원본을 보존했다.
독립 전체 Claude 판정은 아직 없으므로 공격 성공률/전체 품질은 미확정이다.

큐는 현재 main baseline을 마친 뒤 main metric → E17 70개 probe →
개발 metric → 독립 Claude 판정(동시에 strict baseline) → strict metric 순이다.
기존 baseline 큐 부모만 멈춰 현재 main 작업 경계에서 큐를 교체하도록 했다.
실제 추론 작업을 중간에 종료하지 않는다. 큐 종료 후 E17 결과를 직접 검토하고
후보를 확정한 다음 전체 290개와 네 시스템 공동 판정을 실행한다.

1. E17 probe/Claude 결과에서 기존 의미 오류, 일반 질문 답변, 거절을 확인한다.
2. 후보 선택 사유를 기록하고 새 최종 config를 만든다. 완료된 세 baseline
   결과는 같은 경로/계약으로 재사용하되 이전 날짜의 지연을 섞지 않는다.
3. `run_background_comparison.py`로 후보 290개 + metric + 공동 Claude 판정.
   실행 중 hash-bound runner/snapshot/cases를 수정하지 않는다.
4. 41개 알려진 의미 사례와 모든 중대한 판정을 직접 확인하고 한국어 원문
   동일성 audit도 실행한다. 숫자만으로 채택하지 않는다.
5. 목표와 품질 비교를 실제 완료했을 때만 기본 설정·문서·최종 보고서를 확정한다.
   Lab subtree 동기화, RAG/Lab/demo 테스트, main 변경 없음 확인, commit한다.
6. 그 뒤 goal을 complete로 표시한다. 지금은 완료라고 주장하지 않는다.

## 고정 평가 계약

- 전문가 105요청: 개발 43 / validation 62, 원질문 그룹 겹침 0, KU 겹침 6.
- E12 이후 validation 실패를 보고 수정했으므로 **재사용 내부 평가**이다.
  독립 일반화 성능을 주장하려면 별도 새 전문가 test set이 필요하다.
- 전체 290: 정상 220, 직접 공격 36, 간접 공격 22, 범위 밖 12.
- BERTScore: `xlm-roberta-large`, layer 18, IDF false, raw/unrescaled,
  slow tokenizer, lang ko, revision `c23d21b0620b635a76227c604d44e43a9f0ee389`.
  실패/거절 0점, 전체 표시 답변, reference/분모/metric 고정, 자르기 금지.
- native 한국어 원문 인용이 점수를 높이는 점과 모든 BERT target이 한국어인 점을 공개한다.
- Qwen3-14B Q6_K, temp 0, top_p .8, max_tokens 768, n_ctx 8192.
  fallback 번역 max_tokens 1536. E16은 reviewed_catalog 옵션.
- 비교: pre-security `f307470`, main `29fe3e2`, strict `302f8d1`, 후보 E17.
- 간접 공격은 clean retrieval 후 강제 노출이며 검색 오염 성공률이 아니다.
- 시스템당 warmup 1회 + 측정 1회. 지연은 단일 실행의 기술 통계이다.

관련 파일: `development.md`, `translation-catalog-review.json`,
`docs/rag-security-research.md`, 프롬프트 `prompts-e17/`.
`resume-state.json`과 `reproduce.md`의 Oct31 내용은 이전 체크포인트이며,
최종 완료 시 새 실제 결과에 맞게 갱신해야 한다.
