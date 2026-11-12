#!/usr/bin/env python3
"""Run a reproducible Qwen quantization panel; performs no judging."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path("/home/minhee/soprano-qa-rag-system")
BASELINE_ROOT = Path(
    "/home/minhee/soprano-qa-evaluation-final-20260804-v6-OZcJD5"
)
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.server import (  # noqa: E402
    _comparison_runs,
    build_comparison_payload,
)
from soprano_qa import service  # noqa: E402
from soprano_qa.llm import validate_generation_requirements  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def input_fingerprint(model: Path, panel_ids: list[str]) -> str:
    paths = [
        PROJECT_ROOT / "soprano_qa" / "answer.py",
        PROJECT_ROOT / "soprano_qa" / "dense.py",
        PROJECT_ROOT / "soprano_qa" / "retrieval.py",
        PROJECT_ROOT / "soprano_qa" / "service.py",
        PROJECT_ROOT / "soprano_qa" / "llm.py",
        PROJECT_ROOT / "config" / "settings.json",
        PROJECT_ROOT / "data" / "corpus.json",
        PROJECT_ROOT / "data" / "corpus_stats.json",
        model,
    ]
    payload = {
        "model": file_record(model),
        "files": [file_record(path) for path in paths],
        "panel_ids": panel_ids,
        "top_k": 6,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def choose_panel(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result, run in _comparison_runs(comparison):
        review = run["quality_review"]
        rows.append(
            {
                "case_id": run["case_id"],
                "piece_id": result["piece_id"],
                "question_origin": run["viewer_question_origin"],
                "question_kind": run["viewer_question_kind"],
                "range_kind": (
                    "range"
                    if run["inference_input"].get("measure_range")
                    else "whole"
                ),
                "baseline_red_flag": review["red_flag"],
                "baseline_quality_review": deepcopy(review),
                "inference_input": deepcopy(run["inference_input"]),
                "baseline_generated_answer": deepcopy(
                    run["generated_answer"]
                ),
            }
        )
    flags = sorted(
        (row for row in rows if row["baseline_red_flag"]),
        key=lambda row: row["case_id"],
    )
    passes = sorted(
        (row for row in rows if not row["baseline_red_flag"]),
        key=lambda row: row["case_id"],
    )
    controls: list[dict[str, Any]] = []
    control_ids: set[str] = set()
    kinds = (
        ("original", "whole", None),
        ("original", "range", None),
        ("synthesized", "whole", "source_question_paraphrase"),
        ("synthesized", "range", "source_question_paraphrase"),
        ("synthesized", "whole", "knowledge_unit_derived"),
        ("synthesized", "range", "knowledge_unit_derived"),
    )
    pieces = (
        "die-forelle",
        "in-flowery-clouds",
        "la-capinera",
        "nella-fantasia",
        "una-voce-poco-fa",
    )
    for piece in pieces:
        for origin, range_kind, question_kind in kinds:
            candidate = next(
                (
                    row
                    for row in passes
                    if row["case_id"] not in control_ids
                    and row["piece_id"] == piece
                    and row["question_origin"] == origin
                    and row["range_kind"] == range_kind
                    and (
                        question_kind is None
                        or row["question_kind"] == question_kind
                    )
                ),
                None,
            )
            if candidate is not None:
                controls.append(candidate)
                control_ids.add(candidate["case_id"])
    for row in passes:
        if len(controls) >= 30:
            break
        if row["case_id"] not in control_ids:
            controls.append(row)
            control_ids.add(row["case_id"])
    assert len(flags) == 55, len(flags)
    assert len(controls) == 30, len(controls)
    return [*flags, *controls]


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    output = args.output.resolve()
    original = load_json(BASELINE_ROOT / "qualitative.json")
    synthesized = load_json(
        BASELINE_ROOT / "synthesized_question_results.json"
    )
    review = load_json(BASELINE_ROOT / "semantic_quality_review.json")
    comparison = build_comparison_payload(original, synthesized, review)
    panel = choose_panel(comparison)
    panel_ids = [row["case_id"] for row in panel]
    fingerprint = input_fingerprint(model, panel_ids)

    completed: dict[str, dict[str, Any]] = {}
    if output.exists():
        prior = load_json(output)
        if prior.get("input_fingerprint") != fingerprint:
            raise RuntimeError("Existing panel output has a stale fingerprint")
        completed = {
            item["case_id"]: item for item in prior.get("results") or []
        }

    service.SETTINGS["model_path"] = str(model)
    validate_generation_requirements(
        str(model),
        service.SETTINGS["llm"],
    )
    payload: dict[str, Any] = {
        "artifact_type": "soprano_qa_model_precision_panel",
        "schema_version": "1.0",
        "status": "partial",
        "input_fingerprint": fingerprint,
        "model": file_record(model),
        "top_k": 6,
        "selection": {
            "baseline_flagged": 55,
            "passing_controls": 30,
            "total": 85,
            "policy": "all_baseline_flags_plus_stratified_passing_controls",
        },
        "results": [],
    }
    for index, row in enumerate(panel, start=1):
        case_id = row["case_id"]
        if case_id not in completed:
            inference_input = row["inference_input"]
            measure = inference_input.get("measure_range")
            try:
                response = service.ask(
                    piece_id=inference_input["piece_id"],
                    question=inference_input["question"],
                    measure_range=(tuple(measure) if measure else None),
                    generate=True,
                    top_k=6,
                )
                completed[case_id] = {
                    **row,
                    "status": "completed",
                    "current_generated_answer": response,
                    "error": None,
                }
            except Exception as error:  # record; never substitute an answer
                completed[case_id] = {
                    **row,
                    "status": "error",
                    "current_generated_answer": None,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            print(f"[{index:02d}/85] {case_id} {completed[case_id]['status']}", flush=True)
        payload["results"] = [
            completed[panel_id]
            for panel_id in panel_ids
            if panel_id in completed
        ]
        atomic_write(output, payload)
    errors = sum(item["status"] == "error" for item in payload["results"])
    payload["status"] = "complete" if errors == 0 else "complete_with_errors"
    payload["summary"] = {
        "completed": len(payload["results"]) - errors,
        "errors": errors,
        "total": len(payload["results"]),
    }
    atomic_write(output, payload)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
