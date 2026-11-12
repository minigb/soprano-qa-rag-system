"""Add a complete generated comparison while preserving the historical E37 view."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

from export_data import PIECES, project

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from run_joint_expanded_judge import frozen_write
from summarize_generated_comparison import load_comparison


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--activate", action="store_true", help="Make this fully evaluated version the initial view")
    args = parser.parse_args()
    if not re.fullmatch(r"e[0-9]+", args.candidate):
        raise ValueError("Use the recorded candidate ID")
    report = read(args.report)
    sources = report["contract"]["sources"]
    for source in sources.values():
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Report source changed: " + source["path"])
    paths = {key: Path(value["path"]) for key, value in sources.items()}
    results = {key: paths["results_" + key] for key in ("original", "current")}
    metrics = {key: paths["scores_" + key].parent for key in ("original", "current")}
    cases, variants, _ = load_comparison(paths["cases"], results, metrics, paths["judge_mapping"].parent)
    provenance = read(paths["provenance"])
    if provenance["candidate"] != args.candidate or provenance["settings"]["llm"]["source_preservation"] != "semantic":
        raise ValueError("This exporter requires the recorded semantic-generation candidate")
    exported = []
    for index, case in enumerate(cases.values()):
        item = project(case, ["id", "question", "piece_id", "measure_range", "family", "split", "source_group_id"])
        item["index"] = index + 1
        item["references"] = [project(ref, ["knowledge_unit_id", "text", "measure_ranges", "reference_role"])
                              for ref in case["reference_authority"]["items"]]
        item["versions"] = {}
        for version in ("original", "current"):
            entry = variants[version][case["id"]]
            row, score = entry["row"], entry["bert"]
            response = row.get("response") or {}
            judgment = entry["judge"]
            item["versions"][version] = {
                "answer": response.get("answer", ""), "status": row["status"],
                "generation_mode": response.get("generation_mode"),
                "latency_seconds": row["elapsed_seconds"],
                "evidence": [project(e, ["id", "text", "measure_ranges", "evidence_type", "generation_role"])
                             for e in response.get("evidence", [])],
                "bertscore": {"scored": score["scored"], "f1": score["f1"] if score["scored"] else None,
                              "reason": score["unscorable_reason"], "candidate_tokens": score["candidate_tokens"]},
                "judge": project(judgment, ["source_fidelity", "factual_correctness", "naturalness",
                    "naturalness_rationale", "rationale", "completeness", "relevance", "attribution", "refusal"])
                    if judgment else {"source_fidelity": "not_applicable", "factual_correctness": None,
                    "naturalness": None, "naturalness_rationale": "", "rationale": "실행 오류로 평가할 답변이 없습니다.",
                    "completeness": None, "relevance": None, "attribution": None, "refusal": None},
                "secondary_note": None,
            }
        exported.append(item)
    date = max(datetime.fromisoformat(r["row"]["started_at"].replace("Z", "+00:00"))
               for r in variants["current"].values()).date().isoformat()
    evaluation_id = "semantic_generated_" + args.candidate
    payload = {"schema_version": 1, "evaluation_id": evaluation_id, "evaluation_date": date,
               "judge_model": "Claude Opus 5.5", "pieces": PIECES,
               "summary": report["variants"], "naturalness_comparison": report["paired"]["naturalness_both_useful"],
               "cases": exported, "source_hashes": {key: value["sha256"] for key, value in sources.items()}}
    serialized = json.dumps(payload, ensure_ascii=False)
    if any(term in serialized for term in ("/home/minhee/", "private_trace_attempt_id", "batch_usage", "model_path", "contract_id")):
        raise ValueError("Private execution metadata entered the public export")
    dist = Path(__file__).resolve().parent / "dist"
    filename = "data-" + args.candidate + ".json"
    frozen_write(dist / filename, payload)
    # Verify every displayed answer/source/score against the independently
    # retained records before making this version selectable.
    saved = read(dist / filename)
    for case in saved["cases"]:
        for version in ("original", "current"):
            entry = variants[version][case["id"]]
            raw = entry["row"].get("response") or {}
            display = case["versions"][version]
            assert display["answer"] == raw.get("answer", "")
            assert [e["text"] for e in display["evidence"]] == [e.get("text") for e in raw.get("evidence", [])]
            assert display["bertscore"]["f1"] == (entry["bert"]["f1"] if entry["bert"]["scored"] else None)
            if entry["judge"]:
                assert all(value == entry["judge"][key] for key, value in display["judge"].items())
    catalog_path = dist / "evaluations.json"
    catalog = read(catalog_path)
    label = args.candidate.upper()
    model = provenance["settings"]["model_filename"].removesuffix(".gguf")
    all_stats = report["variants"]["current"]["all"]
    description = {
        "id": args.candidate, "label": "생성형 RAG · " + label,
        "shortLabel": label, "answerLabel": "생성형 RAG · " + label,
        "answerSubtitle": model, "data": filename,
        "storageKey": "soprano-answer-review:" + evaluation_id + ":v1",
        "date": date.replace("-", "."), "metricNote": "105문항 평균 · 고정된 전문가 참조·설정",
        "note": f"질문에 맞춰 LLM이 작성한 {label}의 답변입니다. Claude 평가에서 중대한 의미 변경 {all_stats['material_drifts']}건, 오거절 {all_stats['false_refusals']}건, 실행 오류 {all_stats['execution_errors']}건입니다. 판정과 원문을 함께 확인하세요. E37의 이전 원문 인용 결과도 버전 선택에서 볼 수 있습니다.",
        "methodIntro": f"같은 460개 요청에 대한 보안 적용 전 원본과 생성형 {label}의 실제 답변입니다. 답변과 KU는 수정하거나 줄이지 않았습니다. 이전 원본 실행 결과를 그대로 비교하며 새 모델의 답변을 Claude Opus 5.5로 평가했습니다. 동일한 전체 평가 packet은 기존 첫 유효 판정을 재사용합니다.",
        "methodBert": "전문가 KU 참조와의 유사도입니다. xlm-roberta-large, 18층, IDF 없음, raw F1과 기존 참조를 유지했습니다. 높은 점수가 의미 정확도를 보증하지 않습니다. 거절·실행 오류와 512토큰 초과 답변은 계산 불가로 표시하고 전체 평균에 0점 기여로 포함합니다. 답변을 자르지 않았습니다.",
        "methodScope": f"원본은 Qwen3-14B와 946개 KU, {label}은 {model}와 953개 KU를 사용합니다. 공통 946개 KU와 전문가 KU는 동일하고 조수미 피드백 7개가 추가됐습니다. 모델·프롬프트·검증을 포함한 전체 시스템 비교입니다. 지연은 직렬 실행의 실제 요청 시간이며 여러 평가 실행을 합친 경우도 있어 반복 측정의 통계적 속도 비교는 아닙니다.",
    }
    existing = next((e for e in catalog["evaluations"] if e["id"] == args.candidate), None)
    if existing and existing != description:
        raise ValueError("Existing evaluation description differs")
    if not existing:
        catalog["evaluations"].insert(0, description)
    if args.activate:
        catalog["default"] = args.candidate
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate": args.candidate, "cases": len(exported), "full_answers_verified": 920,
                      "default": catalog["default"], "private_execution_metadata": "excluded"}))


if __name__ == "__main__":
    main()
