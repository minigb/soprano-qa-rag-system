# 완료 — 원본 / 최신 Gemini 답변 비교 데모 (2026-11-07)

작업 위치: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`
브랜치: `codex/helpful-rag-responses`.

사용자 요청에 따라 기존 localhost 답변 비교 데모의 기본 버전을
`gemini-20261107`로 갱신했습니다. 왼쪽은 보안 적용 전 원본, 오른쪽은
복구 완료된 Gemini3.8Flash의 실제 생성 답변입니다. 후보 J를 원본으로 표시하지 않습니다.

- 최신 결과가 존재하는 음악 109문항(전문가 기반 105 + 추가 회귀 4)을 표시합니다.
  나머지 351문항은 Gemini로 실행하지 않았습니다. 이전 460문항 버전 3개는 선택 목록에 보존됩니다.
- 원본 답변/근거/판정/점수와 기존 데이터 JSON 3개는 변경하지 않았습니다.
  원문 대입, 답변 편집, 재생성, 재평가, 추가 API 호출은 없습니다.
- 전문가 105문항의 최신 BERTScore는 저장된 0.950992를 그대로 표시합니다.
  양쪽 평가 참조와 BERTScore 참조문을 구분하며, 요약은 표시된 109문항 기준입니다.
  원본 전체 응답 시간과 Gemini 생성 호출 시간은 직접 속도 비교가 불가함을 표시합니다.
- 음악 API 오류 6건의 재시도 여부를 별도 표시합니다. 이전 거절 5건 필터,
  전체 답변/근거 열람, 버전별 메모, 자연스러움 하락 필터를 유지했습니다.
- 데이터 검증 및 최신 109/과거 460 화면의 브라우저 검사 통과.
  문항 수, 답변 전문, 평가 참조, 필터, 메모 저장/내보내기, 버전 전환,
  없는 문항 안내, 4가지 화면 너비를 확인했습니다.
- 기존 background-knowledge 작업트리의 정적 데모 복사본 3곳도 동기화했습니다.
  메인/원본/백엔드는 변경하지 않았고 서버도 시작하지 않았습니다.

실행 (사용자가 직접):

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory /home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system/sites/answer-review/dist
```

URL: `http://localhost:8766/?evaluation=gemini-20261107`
이전 URL에 다른 `evaluation=` 값이 있으면 해당 과거 버전이 계속 열립니다.
`sites/answer-review/README.md`와 `evaluation/gemini_demo_2026-11-07/`에
실행/검증/동기화 기록이 있습니다. 데이터만 재내보내려면
`python sites/answer-review/export_gemini.py` 후 `verify_gemini.py`를 실행합니다.
이 데모 갱신을 위해 모델을 다시 호출할 필요는 없습니다.

---

# 완료 — 새 계정으로 누락 답변7건 복구 (2026-11-07, Asia/Seoul)

작업 위치: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`
브랜치: `codex/helpful-rag-responses`.

사용자가 새 Google API 계정을 설정하고 필요하면 사용하도록 허용했습니다.
`~/gemini-vertex-4`의 ADC가 이제4번 프로젝트와 일치하고3번 계정과도 달라졌음을 확인했습니다.
이 최신 승인은 아래 과거의 ‘3번만 사용’ 제한을 새 계정 사용에 대해 갱신합니다.

같은 Gemini3.8Flash로 기술 실패7건(전문가6+영어직접공격1)만1회씩 재시도했고 모두 답변됐습니다.
기존 성공175개 답변·판정·최초평가464개 파일은 그대로 보존했습니다. 추가 API 호출은7회입니다.

- 전문가105개 모두 확보. 중대한 의미 오류 로컬4→Gemini0, 경미한 오류38→7.
- 자연스러움3.486→3.762/4, 충분한 답변92→101/105, 음악 오거절0.
- **재시도 후 전체105 BERTScore0.950992로0.9 충족.** 첫 시도0.897379는 별도로 보존합니다.
  기존99개 점수 유지, 누락6개만 같은 noIDF 설정/encoder revision으로 점수 계산. 잘림0.
- 직접 공격 전체30개: 로컬9→Gemini0 성공. 나머지 기존 답변은 모두 그대로입니다.
- 기존5거절 문항의 자연스러움3.6→3.2 하락도 그대로 유지; 새 크레센도 답변의 범위 확대는 minor입니다.
- 새 모델·새 프롬프트·성공 답변 재생성·지연 벤치마크 재실행·서빙 코드 변경은 없습니다.
- 기존 자료와 입력/응답 모델/근거/metric/packages 보존 검증 통과. 현재 실행 중인 이 평가 작업은 없습니다.

먼저 [복구 후 보고서](evaluation/gemini_flash_recovery_2026-11-07/report.md),
[전체 수치](evaluation/gemini_flash_recovery_2026-11-07/summary.json),
[실제 답변 비교](evaluation/gemini_flash_recovery_2026-11-07/expert-comparison.json),
[검증](evaluation/gemini_flash_recovery_2026-11-07/verification.json)을 읽으세요.

첫 시도와 재시도 후 지표를 혼동하지 마세요. 기존20.448초는 이전 공유 API 환경의 관측이고,
이번 계정의 대표 지연을 다시 측정한 것은 아닙니다. 자동 재시도나 계정 fallback을 서빙에 추가하지 않았습니다.
기본/원본/main/기존460데모 보존. YouTube 작업은 건드리지 않았습니다.
완료된 답변을 다시 생성하거나 다른 API 모델을 탐색하지 마세요. 알림/notifier를 재설치하지 마세요.

---

# 완료 — Gemini 단일 모델과 로컬14B 비교 (2026-11-07)

작업 위치: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`
브랜치: `codex/helpful-rag-responses`. Vertex 연결 구현 커밋: `5989b70`.

**Gemini3.8Flash LOW 하나와 Qwen3-14B 후보J를 비교 완료했습니다.**
같은 질문·KU·생성 프롬프트를 고정했습니다. API 모델 탐색을 더 하지 않습니다.
선택 설정은 `config/settings.gemini.json`이며, 기본/local 설정·원본/main·corpus·
기존460개 데모는 그대로입니다. 생성에 원문 대입이나 온라인 출력 검사를 추가하지 않았습니다.

전문가 주석 기반105개+기존 거절변형4개=109개, 한영challenge74개 목록을 처리했습니다.
challenge의 기존 영어 번역 실패1개는 생성 입력이 없어 paired 비교에서 제외했습니다.
품질 평가182회 결과는 답변175/API오류7개입니다. 첫 출력/오류와 첫 유효 Claude 판정을 보존했습니다.

- 전문가105개: Gemini 답변99/API오류6. 같은99개에서 중대한 의미 변경4→0,
  경미한 변경35→6, 자연스러움3.495→3.778/4, 충분한 답변88→97. 음악 오거절0.
- **전체105 BERTScore는0.897379로0.9 미달**입니다. 반환된99개만의0.951765와 구분합니다.
  같은99개 로컬 점수는0.940459, 로컬 전체105는0.939943입니다.
  기존 XLM-R-large layer18/noIDF/rawF1과 revision 유지; 잘림/인코더길이초과0, 재탐색0.
- 기존 거절5개는 모두 답변/faithful이지만, 자연스러움은3.6→3.2로 하락했습니다.
  기존4개 중대한 오류는 개선됐고, 조옮김 답변에는 의무 표현으로 강화하는 경미한 오류가 남았습니다.
- 직접 공격 같은29개: 로컬9→Gemini0 성공. 추가1개는 API오류로 방어 성공에 포함하지 않습니다.
  KU주입0/20(기존 자료 차단 포함), 정상음악 오거절0/12, 범위밖 과제 수행1/11→0/11.
- 별도 일반10문항 순차 생성 시간: 로컬 평균2.690초/중앙값2.381초(2회씩),
  Gemini 평균20.448초/중앙값11.003초(1회씩). 검색·번역·요청간격 제외.
  같은 프로젝트의 YouTube 작업을 유지했으므로 모델 고유 속도 차이로 단정하지 않습니다.
- 기술 오류7개는429 네 건,504 두 건,4번 프로젝트 설정403 한 건입니다.
  403을 모델 의미 오류로 해석하지 마세요. 이 설정 실패도 전체 기록에 보존했습니다.
- 693개 전체 테스트, Flash 설정 변경 후19개 관련 테스트, pip check 통과.
  실제 한국어/영어 서비스 호출 모두 통과: 한국어 로컬생성0회, 영어14B검색번역1회+Gemini생성.
  source/input/provider/model/완료수와 원본 보존 검증 모두 통과했습니다.

먼저 다음 자료를 읽으세요.

- [최종 비교 보고서](evaluation/gemini_flash_writer_2026-11-07/report.md)
- [수치와 실패 목록](evaluation/gemini_flash_writer_2026-11-07/summary.json)
- [판단과 미달 목표](evaluation/gemini_flash_writer_2026-11-07/decision.json)
- [전문가 실제 답변 비교](evaluation/gemini_flash_writer_2026-11-07/expert-comparison.json)
- [공격/정상 질문 비교](evaluation/gemini_flash_writer_2026-11-07/challenge-comparison.json)
- [직접 의미 검토](evaluation/gemini_flash_writer_2026-11-07/manual-review.json)
- [검증 결과](evaluation/gemini_flash_writer_2026-11-07/verification.json)
- [연결 및 실행 방법](docs/gemini-vertex.md)

현재 이 작업의 생성·평가·완료 대기·서버 프로세스는 없습니다. 완료된 평가를 다시 실행하지 마세요.
후속 변경은 새 snapshot/output으로 진행하고, 점수를 올리려고 저장된 답변이나 실패를 교체하지 않습니다.

사용자는4번 계정의 fallback 사용을 철회했습니다. 앞으로 **3번만 사용**합니다.
인증/프로젝트를 바꾸거나 다른 API 모델을 호출하지 마세요. 별도 YouTube 작업은 중지하지 않습니다.
이전 Argon 실험은 `evaluation/gemini_writer_2026-11-07/`의 부분 기록으로 보존했습니다.
87개 시도/80답변/7오류이며, 전체 평가가 아닙니다. 해당 생성/판정/시간측정 스크립트를 재개하지 마세요.
Discord/프로세스 종료 notifier 및 usage-reset 중지 조건은 없습니다.

---

# 완료 — 출력 검사 없는 프롬프트 개선·비교, 기본 전환 보류 (2026-11-06)

작업 위치: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`
브랜치: `codex/helpful-rag-responses`, 구현 커밋 `9da2bda`.
현재 소스는 의미 보존 우선 후보 J입니다. `config/settings.single-pass.json`으로 실행합니다.
**기본 `config/settings.json`과 기존460개 데모는 이전 검증 상태 그대로입니다.**
원본/main/Lab 백엔드도 보존했습니다. 출력 검사는 새 후보에 추가하지 않았습니다.

14B·hybrid·corpus·생성 설정을 고정하고 동일 전문가 기반105개+거절 변형4개,
한국어/영어 challenge74개, 고정20문항×2회 지연을 비교했습니다.

- J 전문가105개: 중대한 의미 변경 **11→4**, 자연스러움 **3.448→3.486/4**, 오거절0.
- 새 중대한 오류3개가 생겼고 기존1개가 남았습니다. 완전 해결은 아닙니다.
- 충분한 답변 판정94→92개. 자연스러움18개 상승/73동일/14하락.
- 기존 거절5개 모두 답변·의미 보존5/5, 자연스러움3.6/4.
- 일반 음악10문항×2회: **3.976→2.769초**. 입력/KU/출력 모델 판정0회.
- 한국어 생성1회, 영어 검색 번역+생성2회. 14B 유지.
- 직접 공격 성공 **8/30→9/30**, 자료 주입0/20, 정상 대조 오거절0/12.
- 범위 밖1/11 과제 수행 + 영어 질문 번역 실패1건. 실패를 방어 성공으로 세지 않습니다.
- 고정 BERTScore 전문가105개 **0.939943**, 전체109개0.939042. 설정 재탐색·답변 잘림0.
- 684개 테스트, 소스/model/corpus/retrieval/기존 checkout/demo 해시 검증 통과.

생성 프롬프트는 조건·주체·범위·확실성을 구체적으로 유지하고 불필요한 부연을 줄이며,
가상 작성 예시를 사용합니다. 일반 음악 지식 보충은 허용합니다. 원문 대입은 없습니다.
페르마타 m1–2/col canto/p와PP 구분은 J에서 개선됐습니다. 꽃바람 발음, 겹자음,
조옮김→편곡 확장, 기교 가사의 조건 누락4건과 다른 경미한 문제는 남아 있습니다.

프롬프트만으로 보안과 품질을 모두 해결한 기본 전환을 달성하지는 못했습니다.
추가 보안 문구 H/K는 회귀했습니다. 보안 문구를 그대로 둔 작성-only L은
의미 오류8개·자연스러움3.571·직접 공격8/30·BERTScore0.932087이지만,
페르마타/p-PP 세부 의미가 J보다 나빠 검토용 후보는 J를 남겼습니다.
H/J/L 전체 출력, K challenge 출력, 실행하지 않은 I 및 K expert/timing 상태도 보존했습니다.

먼저 다음 자료를 읽으세요.

- [최종 비교 보고서](evaluation/prompt_fidelity_2026-11-06/report.md)
- [선택과 한계](evaluation/prompt_fidelity_2026-11-06/decision.json)
- [실제 답변 비교](evaluation/prompt_fidelity_2026-11-06/answer-comparison-j.json)
- [직접 의미 검토](evaluation/prompt_fidelity_2026-11-06/manual-review-j.json)
- [검증 결과](evaluation/prompt_fidelity_2026-11-06/verification.json)
- [재현/실행 기록](evaluation/prompt_fidelity_2026-11-06/README.md)

요청한 프롬프트 개선과 비교 작업은 완료했습니다. 실행 중인 생성/평가/서버 작업은 없습니다.
완료된 답변이나 첫 유효 Claude 판정을 좋은 점수가 나올 때까지 재실행하지 않습니다.
후속 요청은 새 snapshot/output으로 진행합니다. 새 holdout 평가가 아닙니다.
Discord/notifier 및 사용량/reset 중지 조건은 사용하지 않습니다.

---

# 단계별 지연 측정 완료 — 단일 생성 후보는 미채택 (2026-11-05)

작업 위치: `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`
브랜치: `codex/helpful-rag-responses`. 후보 경로 구현 커밋은 `5822dd9`입니다.
이후 품질 회귀를 확인해 기본 설정을 직전 검증 버전으로 복구하고 후보 설정을 분리했습니다.
**기본 `config/settings.json`과 460개 데모는 이전 검증 상태입니다.**
새 후보는 `config/settings.single-pass.json`으로만 활성화합니다. 14B는 유지했습니다.

사용자는 단계별 시간 측정, 14B 입력/KU 판정 제거, 기존 hybrid 검색 유지,
출력 판정 제거를 요청했습니다. 이를 단일 생성 후보로 구현·비교했으나,
이전에 강조한 의미·품질 유지 조건을 만족하지 않아 기본 경로로 채택하지 않았습니다.
현재 실행 중인 생성/평가 작업이나 필수 마무리 작업은 없습니다.
이 작업은 측정/실험 완료이며, 품질을 유지하는 단일 생성 전환을 달성했다는 뜻은 아닙니다.

- 일반 음악 10문항 × 2회: 8.720→3.968초. 검색 0.033초, 기존 입력 판정 0.422초,
  추가 KU 선택 0.339초, 생성 6.322초, 출력 판정 1.556초. 후보는 생성 3.890초,
  나머지 LLM 판정 0회입니다. 생성 방식도 달라져 전체 차이를 검사 제거만의 효과로 보지 않습니다.
- 이전 거절 5문항 × 2회: 23.289→4.367초. 후보는 한국어 1회, 영어는 검색 번역+생성 2회입니다.
- 전문가 105개: 중대한 의미 변경 2→11건, 자연스러움 3.543→3.448/4.
- 고정 BERTScore 0.928871로 0.9를 넘지만 **품질 유지에는 실패**했습니다. 설정 재탐색/잘림 0.
- 이전 거절 5건 모두 답변, 의미 보존 4/5·경미한 확장 1/5, 자연스러움 3.8/4.
  p/PP 최종 후보 답변은 두 설명을 구분해 의미 보존 판정입니다.
- 직접 공격 성공 0/30→8/30, 간접 공격 0/20, 범위 밖 과제 수행 0/12→3/12.
  프롬프트만으로 서비스 범위를 유지한다는 기대는 이번 14B 후보에서 충족되지 않았습니다.
- 684개 테스트 통과. 원본/main/기존 백엔드와 데모 세 곳 보존, 후보 소스·모델·corpus 해시 검증 완료.

먼저 [보고서](evaluation/single_pass_2026-11-04/report.md),
[판정](evaluation/single_pass_2026-11-04/decision.json),
[답변 비교](evaluation/single_pass_2026-11-04/answer-comparison.json),
[재현/실험 목록](evaluation/single_pass_2026-11-04/README.md)을 읽으세요.
최종 유효 capture는 `single-pass-g-clean`, `expert-g`(109개), `challenge-g`(74개)입니다.
`single-pass-final`/`expert-final`은 이름과 달리 실패한 D 후보이며 최종 결과가 아닙니다.
`baseline`과 `single-pass-g`는 계측 오류/동시 추론 때문에 무효화했습니다. 불리한 결과도 보존했습니다.
새 460개 평가/데모를 만들지 않았으며, 기존 데모를 새 후보 결과처럼 표시하지 않았습니다.

후속 개선을 요청받으면 새 snapshot/output을 사용하고, 완료된 결과를 좋은 점수가 나올 때까지
재실행하지 마세요. 온라인 출력 검사를 다시 추가하지 않는다는 사용자 방향을 유지하면서
경량 입력 범위 처리와 생성 자체의 의미 보존을 별도로 검토해야 합니다. BERTScore만으로
선택하거나 원문 대입을 사용하지 않습니다. Discord/notifier 및 사용량/reset 중지 조건은 없습니다.

---

아래는 이전 검증 버전의 완료 기록입니다.

# 완료 — 답변 제공 정책 적용 및 최신 데모 갱신 (2026-11-04)

현재 작업 위치는 `/home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system`,
브랜치는 `codex/helpful-rag-responses`입니다. 구현 커밋은 `2f01ba7`, 분류 메타데이터 보완 커밋은 `e9d523c`입니다.
원본/main과 이전 작업 브랜치의 백엔드는 보존했습니다. 생성 모델은 Qwen3-14B입니다.

전체 460건 재생성·Claude 평가·고정 BERTScore와 한국어/영어 보안 74건을 완료했습니다.
기존 거절 5건은 실제 전체 실행에서 모두 답변됐고 Claude의 의미 보존 판정은 5/5입니다.
자연스러움은 모두 3/4이며, 초기의 같은 5건 평균 3.6/4까지 회복한 것은 아닙니다.
전문가 105문항 BERTScore는 0.945026입니다. 원문 대입·설정 재탐색·답변 잘림은 없습니다.
674개 테스트, 920개 답변/점수 매핑, 브라우저 기능과 모바일 화면 검사도 통과했습니다.

11/03 대비 답변 미제공 5→0, 중대한 의미 변경 18→17, 새 중대한 의미 변경 0입니다.
전체 자연스러움 평균은 3.5239→3.5522지만, 원래 답변하던 455건만 보면
3.5626→3.5582이고 2건은 하락했습니다. 다른 1건의 정보 누락·1건의 귀속 약화,
공격 자료 제외 후 영문 일반 지식 오류 2건 등 한계를 보고서에 남겼습니다.
449개 동일 평가 packet은 계약 확인 후 기존 Claude 판정을 재사용했고 변경 11건은 새로 평가했습니다.

[최종 보고서](evaluation/helpful_responses_2026-11-04/report.md),
[직접 비교](evaluation/helpful_responses_2026-11-04/manual-review.json),
[검증](evaluation/helpful_responses_2026-11-04/verification.json),
[재현 방법](evaluation/helpful_responses_2026-11-04/README.md)을 먼저 읽으세요.
최종 실행은 `security-hardened/`, `full-final/`입니다. 완료된 평가는 다시 돌리지 않습니다.
후속 수정은 새 snapshot/output으로 진행하며 기존의 불리한 결과도 보존합니다.
이전 `security-final/`은 74건 중 자료 제외 후 의미 역전 1건을 발견한 기록이고,
`full/`은 그 문제를 보완하려고 18건에서 중단한 예비 실행입니다. 둘 다 최종 결과가 아닙니다.
`source-recovery-probe*`는 실패 후보까지 보존했습니다. 최종 L은 공격성 topic을
생성 입력에서 제외하고 정상 KU 본문을 유지해 한·영 모두 의미 보존으로 복구했습니다.
일반 지식 프롬프트 변경·추가 reasoning 후보는 채택하지 않았습니다.
최신 데모는 이 checkout의 `sites/answer-review/dist`이며 기존 작업 데모 세 곳에도
정적 파일만 반영했습니다. 초기·E37·11/03 기록과 메모 키는 보존했습니다.
기존 Lab 백엔드는 새 구현으로 전환하지 않았습니다.

```bash
python -m http.server 8766 --bind 127.0.0.1 \
  --directory /home/minhee/.codex/worktrees/rag-helpful-responses/soprano-qa-rag-system/sites/answer-review/dist
```

<http://localhost:8766/?evaluation=helpful-14b-20261104>에서 ‘이전 거절 5건’을 선택하면 됩니다.
서버는 사용자가 직접 띄우며 Discord/notifier 및 사용량/reset 중지 조건을 사용하지 않습니다.

---

아래는 이전 작업의 기록입니다.

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
