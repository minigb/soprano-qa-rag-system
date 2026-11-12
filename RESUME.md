# 현재 사용자 요청: 보안 대응 전 시스템 보존과 작업 위치 안내 (2026-11-03 KST)

사용자가 작업 범위가 불명확해졌다고 지적하고, prompt injection 대응 전 시스템을
보존한 뒤 직접 비교할 수 있도록 현재 브랜치와 위치를 알려 달라고 요청했다.
추가 개선/평가는 실행하지 않고 다음 비교용 worktree를 만들었다. 실행 중인 평가 작업은 없다.

- 원본 RAG: `/home/minhee/.codex/worktrees/rag-pre-injection/soprano-qa-rag-system`
  — `codex/pre-injection-baseline`, `f3074700103d60cf4bd8862fd655e51656f66f0c`.
- 원본 Lab: `/home/minhee/.codex/worktrees/rag-pre-injection/soprano-lab`
  — 같은 브랜치명, `5768b942daaab5e6a07e0bd5eaab5ab6fbab1833`.
  보안 변경을 가져온 `120f8a3`의 직전 first-parent다. 이 Lab의 RAG는
  `737fbf1`이며 평가 원본 `f307470`과 README 외 코드/설정은 동일하다.
- 현재 RAG 작업: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-qa-rag-system`
  — `codex/ku-background-knowledge`, HEAD `146b4fb` + 미커밋 생성형 복구 변경.
- 현재 Lab 작업: `/home/minhee/.codex/worktrees/rag-background-knowledge/soprano-lab`
  — `codex/ku-background-knowledge`, HEAD `4ae2e1e`. 최신 생성형 복구 코드는 아직 동기화하지 않았다.
- `/home/minhee/soprano-qa-rag-system`의 main에는 이미 첫 보안 변경 `29fe3e2`가
  들어 있으므로 보안 적용 전 원본으로 간주하면 안 된다. main/live/dataset은 수정하지 않았다.

마지막 실험 상태: E54 전문가105 BERTScore0.926264이나 중대한 의미 변경8건으로
미채택. E55 32개에서 중대한 의미 변경0/정상22, 오거절0, 자연스러움3.409였으나
별도 repair8에서 p/PP 원인 추측과 거절1건이 남았다. E56 repair8은 전부 거절됐다.
E56의 평가용 model proxy가 chat_handler 쓰기를 실제 모델에 전달하지 않는 문제가
의심되므로 추론 품질 비교로 해석하면 안 된다. 해당 실행/실패는 보존했으며 고치거나
재실행하지 않았다. 새 모델/설정은 기본값으로 채택하지 않았다.

아래는 이 비교 요청 전까지의 상세 개발 기록이다. 이전 완료 문구는 역사적 기록이다.

# 진행 중: 생성형 RAG 복구 — E37 원문 인용 방식 폐기

사용자는 E37이 질문에 답변을 생성하지 않고 KU 원문을 반환하는 것이 과제의
본질에 어긋난다고 지적했다. 이 지적을 수용했다. BERTScore 0.90은 자연스럽고
의미가 정확한 생성형 답변에 적용할 제약이며 원문 인용으로 달성한 수치는
생성형 RAG 품질 목표 달성을 의미하지 않는다.

- 같은 작업 브랜치에서 기본 source_preservation을 semantic으로 변경했다.
- LLM이 비어 있지 않은 답변을 직접 작성하며, KU 선택 후 원문 대입을 하지 않는다.
- E38 첫 생성형 시험에서 확실성 강화와 발음 대안 왜곡이 재발했다. 수용 불가.
  evaluation/semantic_generation_2026-11-02/ 아래 실패 결과도 보존한다.
- E39 reasoning 후보는 22개 정상 질문에서 중대한 의미 변경9건으로 악화되어 폐기했다.
- E40–E42의 4문항 점검에서도 단정/발음 대안 오류가 남았다. 실패 산출물을 보존했다.
- 별도 14B 의미 검토기는 알려진 오류4건을 모두 놓쳤다. 기본 경로에 넣지 않았으며
  evaluation/semantic_generation_2026-11-02/diagnostics/에 실험 소스와 결과를 보존했다.
- E43 진단32개: 정상22개 중 중대한 의미 변경1/22(원본3/22), 오거절0,
  자연스러움3.227/4(원본 유효평균3.136). 한국어/영어 공격·범위 밖10개 성공0.
- E44 진단32개: 의미 변경1/22, 오거절0, 자연스러움3.364/4, 공격 성공0/10.
  p/PP 자료를 모순 설명 없이 나열하고 OR 조건을 좁히는 문제가 남아 채택 보류.
- E45는 압축 입력+reasoning으로16개를 점검했으나 발음/조건 변형이 남았다. 폐기.
- E46 압축 입력에서는 p/PP·OR 조건이 나아졌지만 다른 해석·범위 오류가 남았다.
- E47은 원래 입력 문맥을 유지하고 추가 의미 단서만 전달했다. 정상22개 중
  중대한 의미 변경3건, 자연스러움3.318/4, 오거절0, 공격·범위 밖 성공0/10.
  p를 ‘포코 피아노’로 설명하는 신규 오류가 있어 채택할 수 없다.
- E48은 E47의 사용하지 않는 추론/검토 실험 옵션을 제거한 동일 동작 후보다.
  전문가105개 완료: 3개 오거절을0점 포함한 고정 BERTScore0.919024,
  검증62개0.915978. 의미 변경2/105(원본10), 유효 자연스러움3.410/4(원본3.448).
  p의 잘못된 확장, 검증 실패3건이 남아 채택하지 않는다. Claude208고유packet 새 평가 완료.
- Qwen3-32B-Q6_K를 공식 repo에서 별도 ignored runtime/models에 다운로드하고
  SHA256 검증 완료. revision938a7432affaec9157f883a87164e2646ae17555,
  SHA256 c4c7c3cb6da260df1fe1d3cfd090a32dc7cc348f1278158be18e301f390d6f6e.
  model-comparison-download.json / verification.json에 경로와 고정 출처가 있다.
- E49는 E48과 동일한21개소스/프롬프트/코퍼스에32B만 바꾼 후보다.
  32문항 완료: 정상22개 중 중대한 의미 변경0, 경미7, 오거절0,
  자연스러움3.773/4, 공격·범위 밖 성공0/10, 정상 지연 중앙값17.22초.
  전문가105 실행은 중간에 새 후처리 실패2건이 확인되어 소스 선택/검증 수정으로 중단했다.
  e49-expert.jsonl은 불완전 진단이며 완성된105개 평균처럼 사용하면 안 된다.
- E50은 한 자료를 관련 있다고 선택하면, 실제로 다른 시작 반주 표기를 말하는
  상대 자료를 중복 조언으로 버리지 않도록 한다. 답변 템플릿을 반환하지 않는다.
- E51은 output guard와 최종 검증 실패에 하나의 재작성 예산(최대2번 작성)을 공유한다.
  보안 위반/잘못된판정/백엔드오류는 재시도하지 않는다. 기본 guard는 모두 유지한다.
  ‘음표가 많은 부분은’은 ‘많은 부분에서’와 달리 빈도 일반화가 아니라는 검사도 추가했다.
  전체659 tests 통과. 8개 집중 진단에서 p/PP 나열/작곡 의도 추측과2거절이 남았다.
- E52 현재 후보(32B snapshot): 실제 출처 불일치에는 짧은 비교 작성 프롬프트를 사용하고
  기존 비교 출력검토 규칙도 적용한다. 어떤 악보가 맞는지/차이의 원인을 지어내지 않는다.
  KU 해설과 score_lyrics를 섞을 때 가사에 잘못된 KU인용을 강제하던 후처리 충돌을 수정했다.
  가사는 제공된 score context가 있어야 하고 사용자 선택 범위를 지켜야 한다.
  출력 심사 거절 후에는 출처의 연주 지침과 보편적 이론법칙을 구분해 귀속하도록 안내한다.
  검증을 느슨하게 풀거나 참고답안/지표를 변경하지 않았다.
- run_refinement_probe.py의 fixed-context adapter도 production finalizer callback을 사용하도록
  수정했다(지원하는 새 snapshot에만). 이전 runner는 diagnostics/에 보존했고 과거 결과는
  변경하지 않았다. 새 코드와도 동일한 중복 인용 검사를 적용한다. E52 전체660 tests 통과(25.173초).
- E52 repair8개에서는 가사 인용 충돌은 해결됐지만 p/PP 원인 추측과 페르마타 거절이 남았다.
- E53은32B 작성 단계에서 권장 샘플링의 추론을 허용했다. 최종 JSON 코드블록을
  기존 parser가 거절하는 문제가 확인되어3개 실행 후 중단했다. 부분 결과는 보존했다.
- E54 현재: 추론 작성의 최종 JSON 전체를 감싼 코드블록만 제거한 뒤 기존 엄격한
  구조/인용/출력/범위 검증을 모두 적용한다. 추가 설명이 붙은 응답은 허용하지 않는다.
  전체661 tests 통과. root 기본 설정은 임시 WIP14B semantic(추론 off)이며 Lab production은 아직 E37이다.
  채택할 모델/설정은 전문가105개 결과를 본 뒤 결정한다. 새 기본값 채택 완료 아님.
- E54 전문가105 실행: session82605, e54-expert.jsonl / e54-expert.log.
  15개씩 먼저 완료된 응답을 순서대로 Claude 평가하는 별도 작업: session66858,
  e54-expert-batches/ 및 e54-expert-batches.log. 품질을 보고 문항/판정을 고르지 않는다.
  run_reused_expanded_judge.py는 검증된 완성 batch들을 정확한 packet 일치로만 재사용한다.
- E54 첫30개 Claude 평가에서 중대한 의미 변경3건(간주→무반주, 자연스러운
  음량 증가 허용→힘을 주라는 지시, 첫머리의 페르마타 선택→곡의 모든 페르마타)이
  확인됐다. 채택 가능으로 간주하지 않는다. 실패 결과/판정은 그대로 남긴다.
- Qwen3.5-35B-A3B Q8_0 별도 비교 준비: ggml-org 공식 양자화 repo,
  revision3127ef0b7fd4f12626506419ceefde54fa8db53c,36,903,139,584bytes,
  SHA2568a83fbfde74b0366feb76b075c10df46bbece6aa21ed14a6f291a7904f2b7d67.
  model-qwen35-download.json에 공개 출처/용도, session12430에서 runtime/models로
  다운로드·검증 중. 전송 데이터는 공개 모델뿐이며 코퍼스 업로드 없음.
  현재 llama-cpp-python0.3.32에는 qwen35moe backend가 있다.
  Qwen3.5는 /no_think soft switch를 지원하지 않아 기본 native template의
  enable_thinking=false를 명시하는 native_non_thinking 옵션을 llm.py에 추가했다.
  root 현재 소스는 E54 이후 이 옵션만 추가한 WIP이며 실행 중 E54는 고정되어 있다.
  새 옵션 테스트19개 통과. 새 모델은 아직 추론/평가/채택하지 않았다.
- localhost UI는 E37 과거 원문 인용 결과라고 명확히 표시했고 버전 선택/메모
  분리 기반을 마련했다. 데이터920개는 바꾸지 않았다. 서버 없이 Playwright
  route로 전체 화면/기능 검사 통과. Root146b4fb, Lab4ae2e1e 커밋.
- Qwen3.5 다운로드/전체 SHA256검증 완료(986.64초). model-qwen35-verification.json.
  CPU vocab/template 검사에서 qwen35moe와 공식 enable_thinking=false의 닫힌
  thinking prefill을 확인했다(model-qwen35-template-check.json). 답변 추론은 아직 미실행.
- Qwen3.5 CPU-only 실제 추론 smoke는 SIGILL(exit132)로 실패했다.
  model-qwen35-cpu-smoke.json에 실패 기록이 있다. metadata 지원이 실제 동작 검증은
  아니다. 아직 GPU 추론은 안 했으며 E55 probe가 실제 GPU 호환성/품질을 확인한다.
  현재 global Conda/라이브러리는 바꾸지 않았다. 필요하면 작업 전용 환경을 사용한다.
- E55 snapshot 고정 완료: Qwen3.5-35B-A3B-Q8_0, native_non_thinking,
  source_preservation=semantic, 기존 concise/비교 프롬프트, temperature0.7/top_p0.8,
  top_k20/min_p0/seed42, max_tokens768. reasoning off. 생성만큼 guard/번역/선택도
  같은 모델을 사용하므로 모델/모드/샘플링을 포함한 전체 설정 비교다.
  기본 root 설정은 바꾸지 않았고 --llm-settings로 별도 후보를 고정했다.
- native template/pinned downloader 변경 후 전체665 tests 통과(28.786초).
  평가한 새 weight를 재현하도록 download_model.py가 optional revision/SHA256를 지원한다.
- E54 첫75개 Claude 평가: 중대한 의미 변경4건, 거절0, 자연스러움3.453/4.
  완성105개 평균은 아직 아니다. 기록된 실행은 끝까지 진행 중이며 채택은 보류한다.
  batch7개가 모두 끝나면 E54/원본105개를 집계하는 session17883이 대기 중이다
  (runtime/finalize-e54-judge.py, e54-expert-finalize.log). 현재 E54 판정을 우선 유지하고
  원본 exact packet만 E48의 기존 판정으로 채운다. 점수에 따라 판정을 고르지 않는다.
- GPU 직렬 대기열: session14824, runtime/continue-e54-e55.py, e54-e55-queue.log.
  E54 worker PID553021 종료+105개 완성 확인 → E54 고정 BERTScore → E55 probe32
  → Claude(e43-original-probe-judge의 동일 packet 첫 판정만 재사용) → e55-probe-summary.json 순서다.
  별도 GPU 작업을 중복 실행하지 않는다. E54 완성 전까지 대기열은 GPU를 사용하지 않는다.
- E55 이후 전문가 판정에는 expert-review-cases.json을 사용하면 full460과 동일한
  reference_authority를 유지하며 판정을 재사용할 수 있다. 생성/BERT 원래105개
  요청/참고답은 동일하다. E54 전문가판정은 원래 expert-cases 기준이므로 과거대로 보존한다.
- assemble_semantic_cohort.py는105/355의 요청 동일성과 pipeline계약을 확인하고
  실패까지 원래 행/계약으로 합친다. summarize_generated_comparison.py와
  sites/answer-review/export_generated_data.py는 완성460개만 받아 새 버전으로
  내보낸다. 새 생성형460개 데이터는 아직 없으며 UI기본값은 여전히 과거E37이다.
- E54 전문가105개 완료: 실행오류0/거절0, 고정 BERTScore0.926264,
  Claude 의미충실56/경미41/중대8, 자연스러움3.4095/4(원본3.4476),
  사실성3.4/4(원본3.4952), 정상지연중앙26.076초(원본2.714초).
  점수0.90을 넘었지만 품질이 부족하여 채택하지 않는다. e54-expert-summary.json,
  e54-expert-bertscore/에 전체 결과가 있다. 거절해결만으로 의미품질 성공을 주장하지 않는다.
- E55는 GPU full offload에서 실제 추론이 동작했다. CPU-only SIGILL은 별도 실패다.
  현재 session14824 대기열에서 probe32 실행 중(e55-probe.jsonl/log). 끝나면 자동
  Claude/원본비교가 이어진다. GPU worker 추가 실행 금지; 완료/종료를 먼저 확인한다.
- 다음: 전문가105개 전체 Claude/고정BERT → 같은 소스로 남은355개 및 한영185개.
  모델/소스가 바뀌면 새 snapshot에서 전체를 다시 검증한다. 실패 결과를 지우지 않는다.
- 모델 고정 여부를 선택 질문으로 남겼으나 답변이 없다. 14B 제약이 명시되지 않았다는
  가정으로 별도 32B 비교를 진행한다고 사용자에게 알렸다. 외부 API는 Claude 평가 용도다.
- E48 전체654 tests 통과(25.846초), E51 전체659 tests 통과(24.912초). E47 실제 생성 trace 25건은 모두 비어 있지 않은
  모델 작성 문장이며 자동 원문 반환0이다. 이는 의미/자연스러움 통과 판정은 아니다.
- 새 후보의 Claude 평가, 고정 전문가 BERTScore, 확대 검증은 아직 완료하지 않았다.
- 이전 460개 웹 비교는 과거 E37 원문 인용 방식이라고 명확히 표시했다.
  과거 답변·지표를 새 결과처럼 고치지 않았다.
- 새 스냅샷은 prepare_semantic_refinement.py --candidate eNN으로 고정한다.
  실행 중 스냅샷 수정 금지. private-traces 및 runtime은 Git에서 제외한다.
- root 변경은 아직 Lab production에 동기화하지 않았다. main/live/dataset 수정 금지.
- 사용량 중지/알림/notifier 미사용. 사용자가 localhost는 직접 실행하겠다고 했다.

---

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
