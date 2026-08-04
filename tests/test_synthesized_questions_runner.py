"""Focused tests for the five-piece synthesized hybrid RAG+LLM runner."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluation import run_synthesized_questions as runner
from evaluation import run_qualitative as original_runner


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def add_piece(
    root: Path,
    *,
    piece_id: str,
    schema_version: str,
    token: str,
    include_excluded: bool = False,
    specific_range: bool = False,
) -> None:
    validator_path = (
        root / "expert_curation" / "derived_knowledge_unit_questions.py"
    )
    if not validator_path.exists():
        validator_path.parent.mkdir(parents=True, exist_ok=True)
        validator_path.write_text(
            '"""Fixture validator bound into evaluation fingerprints."""\n',
            encoding="utf-8",
        )
    source_id = f"kim-{piece_id}-02" if include_excluded else f"kim-{piece_id}-01"
    source_question = f"{token} 원래 질문은 무엇인가?"
    source_answer = f"{token} 전문가 답변이다."
    knowledge_unit_answer = f"{token} 최종 지식 단위 답변이다."
    sources = []
    questions = []
    if include_excluded:
        excluded_id = f"kim-{piece_id}-01"
        sources.append(
            {
                "source_id": excluded_id,
                "annotator": "kim",
                "source_text": "보존된 제외 원문",
                "question": f"{token} 제외 질문은 무엇인가?",
                "answer": "기존 웹자료 부탁드립니다.",
                "legacy_measure_ranges": [],
                "curation_status": "excluded_unanswerable",
                "curation_notes": "전문가 답변이 아님",
            }
        )
        questions.append(
            {
                "source_id": excluded_id,
                "annotator": "kim",
                "original_question": f"{token} 제외 질문은 무엇인가?",
                "paraphrased_question": f"{token} 제외 질문을 바꾸면?",
                "knowledge_unit_ids": [],
                "retrieval_eligible_knowledge_unit_ids": [],
                "measure_range_hints": [],
                "inference_measure_ranges": [],
                "inference_scope": "no_range",
                "review_status": "excluded_unanswerable",
            }
        )
    sources.append(
        {
            "source_id": source_id,
            "annotator": "kim",
            "source_text": f"{source_question}\n{source_answer}",
            "question": source_question,
            "answer": source_answer,
            "legacy_measure_ranges": [],
            "curation_status": "included",
            "curation_notes": "",
        }
    )
    unit_id = f"{piece_id}-ku-001"
    unit = {
        "knowledge_unit_id": unit_id,
        "source_ids": [source_id],
        "answer": knowledge_unit_answer,
        "rewrite_status": "ready",
        "rewrite_notes": "",
        "measure_range_hints": [],
        "measure_status": "specific" if specific_range else "whole_piece",
        "measure_ranges": [[2, 5]] if specific_range else [],
        "measure_notes": "",
    }
    base_question = {
        "source_id": source_id,
        "annotator": "kim",
        "original_question": source_question,
        "paraphrased_question": f"{token} 기본 질문을 어떻게 답할까?",
        "knowledge_unit_ids": [unit_id],
        "retrieval_eligible_knowledge_unit_ids": [unit_id],
        "measure_range_hints": [],
        "inference_measure_ranges": [[2, 5]] if specific_range else [],
        "inference_scope": "measure_range" if specific_range else "no_range",
        "review_status": "retrievable",
    }
    if schema_version == "1.3":
        base_question["reference_claim_scope"] = {
            "source_answer_sha256": sha256(source_answer.encode()).hexdigest(),
            "sentence_items": [
                {
                    "sentence_index": 1,
                    "text": source_answer,
                    "scope": "direct_required",
                    "flags": [],
                }
            ],
        }
    questions.append(base_question)
    review = {
        "schema_version": "1.0",
        "piece_id": piece_id,
        "source_files": [],
        "source_annotations": sources,
        "knowledge_units": [unit],
    }
    inventory = {
        "schema_version": schema_version,
        "piece_id": piece_id,
        "questions": questions,
    }
    variants = {
        "schema_version": "1.0",
        "piece_id": piece_id,
        "source_inventory_schema_version": schema_version,
        "variants_per_question": 3,
        "questions": [
            {
                "source_id": source_id,
                "original_question": source_question,
                "base_paraphrased_question": base_question[
                    "paraphrased_question"
                ],
                "variants": [
                    {
                        "variant_id": f"{source_id}-syn-{index:02d}",
                        "question": f"{token} 합성 질문 {label}는 어떻게 답할까?",
                        "transformations": ["lexical_substitution"],
                    }
                    for index, label in enumerate(("하나", "둘", "셋"), start=1)
                ],
            }
        ],
    }
    derived_questions = {
        "schema_version": runner.DERIVED_CATALOG_SCHEMA_VERSION,
        "piece_id": piece_id,
        "provenance": deepcopy(runner.DERIVED_CATALOG_PROVENANCE),
        "questions": [],
    }
    write_json(
        root / "expert_curation" / "review" / f"{piece_id}.json",
        review,
    )
    write_json(
        root
        / "expert_curation"
        / "evaluation_questions"
        / f"{piece_id}.json",
        inventory,
    )
    write_json(
        root
        / "expert_curation"
        / "evaluation_question_variants"
        / f"{piece_id}.json",
        variants,
    )
    write_json(
        root
        / "expert_curation"
        / "derived_knowledge_unit_questions"
        / f"{piece_id}.json",
        derived_questions,
    )


def add_derived_knowledge_unit_question(
    root: Path,
    *,
    piece_id: str = "die-forelle",
) -> tuple[Path, str, str]:
    review_path = root / "expert_curation" / "review" / f"{piece_id}.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    source_id = f"yeon-{piece_id}-99"
    unit_id = f"{piece_id}-ku-999"
    unit_answer = "여러 후보 구간에서는 숨을 미리 준비해 자연스럽게 이어 부릅니다."
    review["source_annotations"].append(
        {
            "source_id": source_id,
            "annotator": "yeon",
            "source_text": unit_answer,
            "question": "",
            "answer": unit_answer,
            "legacy_measure_ranges": [],
            "curation_status": "included",
            "curation_notes": "",
        }
    )
    review["knowledge_units"].append(
        {
            "knowledge_unit_id": unit_id,
            "source_ids": [source_id],
            "answer": unit_answer,
            "rewrite_status": "ready",
            "rewrite_notes": "",
            "measure_range_hints": [],
            "measure_status": "specific",
            "measure_ranges": [[4, 4], [12, 12], [20, 21]],
            "measure_notes": "",
        }
    )
    write_json(review_path, review)

    catalog_path = (
        root
        / "expert_curation"
        / "derived_knowledge_unit_questions"
        / f"{piece_id}.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["questions"].append(
        {
            "knowledge_unit_id": unit_id,
            "source_ids": [source_id],
            "knowledge_unit_answer_sha256": sha256(
                unit_answer.encode("utf-8")
            ).hexdigest(),
            "question": "여러 후보 구간에서는 호흡을 어떻게 준비해야 할까?",
            "origin": runner.DERIVED_QUESTION_ORIGIN,
            "review_status": runner.DERIVED_REVIEW_STATUS,
            "range_policy": "no_range_and_representative_range",
        }
    )
    write_json(catalog_path, catalog)
    return catalog_path, source_id, unit_id


def clean_hybrid_response(
    model_path: Path,
    *,
    unavailable_reason: str | None = None,
    generation_mode: str | None = None,
    answer_basis: str | None = None,
) -> dict:
    resolved_mode = generation_mode or (
        "llm" if unavailable_reason is None else "unavailable"
    )
    resolved_basis = answer_basis or (
        "retrieved_evidence"
        if unavailable_reason is None
        else "no_corpus_evidence"
    )
    return {
        "piece_id": "die-forelle",
        "scope": "whole_piece",
        "measure_range": None,
        "answer": "근거 기반 생성 답변",
        "generation_mode": resolved_mode,
        "answer_basis": resolved_basis,
        "unavailable_reason": unavailable_reason,
        "pipeline": "soprano_qa",
        "model": {
            "path": str(model_path),
            "checkpoint_exists": True,
            "backend": "llama-cpp-python",
        },
        "retrieval": {
            "configured_mode": "hybrid",
            "active_mode": "hybrid",
            "dense_available": True,
            "last_dense_error": None,
            "last_search_mode": "hybrid",
            "query_route": "hybrid",
            "route_reason": "semantic_query",
            "dense_attempted": True,
            "dense_contributed": True,
        },
        "evidence": [
            {
                "id": "die-forelle-ku-001",
                "kind": "expert",
                "source_ids": ["kim-die-forelle-01"],
                "in_requested_scope": True,
                "retrieval_mode": "dense",
            }
        ],
        "evidence_notices": [],
    }


def build_case_snapshot(model_path: Path, *, eligible: bool = True) -> dict:
    unit_ids = ["die-forelle-ku-001"] if eligible else []
    return {
        "run": {
            "model_contract": {
                "generation_model": {"path": str(model_path)}
            },
            "pipeline_error": None,
        },
        "results": [
            {
                "source_id": "kim-die-forelle-01",
                "piece_id": "die-forelle",
                "synthesized_question_variants": [],
                "inference_runs": [
                    {
                        "case_id": "kim-die-forelle-01-syn-01__no-range",
                        "status": "pending",
                        "attempt_count": 0,
                        "last_attempt_at": None,
                        "completed_at": None,
                        "inference_input": {
                            "piece_id": "die-forelle",
                            "source_id": "kim-die-forelle-01",
                            "question": "합성 질문",
                            "synthesized_variant_id": (
                                "kim-die-forelle-01-syn-01"
                            ),
                            "variant_index": 1,
                            "measure_range": None,
                            "expected_knowledge_unit_ids": [
                                "die-forelle-ku-001"
                            ],
                            "range_applicable_knowledge_unit_ids": [
                                "die-forelle-ku-001"
                            ],
                            "expected_retrieval_eligible_knowledge_unit_ids": (
                                unit_ids
                            ),
                        },
                        "retrieval_probe": None,
                        "generated_answer": None,
                        "error": None,
                    }
                ],
            }
        ],
    }


class SynthesizedQuestionsRunnerTests(unittest.TestCase):
    def test_default_dataset_root_uses_primary_dataset_repository(self) -> None:
        self.assertEqual(runner.DEFAULT_DATASET_ROOT.name, "soprano-qa-dataset")

    def test_active_dataset_has_all_synthesized_no_range_semantic_cases(
        self,
    ) -> None:
        benchmark = runner.load_benchmark(runner.DEFAULT_DATASET_ROOT)
        cases = [
            case
            for result in benchmark.results
            for case in result["inference_runs"]
        ]
        cohort = [
            case
            for case in cases
            if case["inference_input"].get("evaluation_cohort")
            == runner.NO_RANGE_COHORT_NAME
        ]

        self.assertEqual(
            len(benchmark.results),
            runner.EXPECTED_QUESTION_GROUP_COUNT,
        )
        self.assertEqual(len(cases), runner.EXPECTED_SYNTHESIZED_CASE_COUNT)
        self.assertEqual(
            Counter(
                case["inference_input"]["piece_id"] for case in cases
            ),
            runner.EXPECTED_SYNTHESIZED_CASE_COUNTS,
        )
        self.assertEqual(len(cohort), 69)
        self.assertEqual(
            Counter(
                case["inference_input"]["piece_id"]
                for case in cohort
            ),
            {
                "die-forelle": 9,
                "in-flowery-clouds": 12,
                "la-capinera": 18,
                "nella-fantasia": 12,
                "una-voce-poco-fa": 18,
            },
        )
        by_source = {result["source_id"]: result for result in benchmark.results}
        for source_id in runner.INTRINSIC_RANGE_ONLY_SOURCES:
            with self.subTest(source_id=source_id):
                runs = by_source[source_id]["inference_runs"]
                self.assertEqual(len(runs), runner.VARIANTS_PER_QUESTION)
                self.assertTrue(
                    all(
                        run["inference_input"]["measure_range"] is not None
                        and run["inference_input"]["case_kind"]
                        == (
                            runner.REVIEWED_HINT_CONTEXT_CASE_KIND
                            if source_id
                            in runner.REVIEWED_HINT_CONTEXT_RANGE_SOURCES
                            else "confirmed_measure_range"
                        )
                        for run in runs
                    )
                )
        lyric_runs = by_source["kim-la-capinera-13"]["inference_runs"]
        self.assertEqual(
            [run["inference_input"]["measure_range"] for run in lyric_runs],
            [[80, 80]] * runner.VARIANTS_PER_QUESTION,
        )
        hint_context = by_source["kim-una-voce-poco-fa-08"]
        self.assertEqual(hint_context["inference_measure_ranges"], [])
        self.assertEqual(
            hint_context["evaluation_context_measure_ranges"],
            [[32, 32]],
        )
        self.assertEqual(hint_context["inference_scope"], "no_range")
        self.assertEqual(
            [
                run["inference_input"]["measure_range"]
                for run in hint_context["inference_runs"]
            ],
            [[32, 32]] * runner.VARIANTS_PER_QUESTION,
        )
        self.assertTrue(
            all(
                run["inference_input"]["measure_range_provenance"]
                == runner.REVIEWED_HINT_CONTEXT_RANGE_PROVENANCE
                for run in hint_context["inference_runs"]
            )
        )
        self.assertEqual(
            len({case["case_id"] for case in cases}),
            len(cases),
        )
        derived_catalog_paths = [
            path
            for path in benchmark.input_paths
            if path.parent.name == "derived_knowledge_unit_questions"
        ]
        self.assertEqual(
            {path.name for path in derived_catalog_paths},
            {f"{piece_id}.json" for piece_id in runner.TARGET_PIECES},
        )
        self.assertIn(
            runner.DEFAULT_DATASET_ROOT
            / "expert_curation"
            / "derived_knowledge_unit_questions.py",
            benchmark.input_paths,
        )
        source_variant_results = [
            result
            for result in benchmark.results
            if result["question_provenance"]
            == runner.SOURCE_VARIANT_PROVENANCE
        ]
        derived_results = [
            result
            for result in benchmark.results
            if result["question_provenance"]
            == runner.DERIVED_QUESTION_PROVENANCE
        ]
        derived_cases = [
            case
            for result in derived_results
            for case in result["inference_runs"]
        ]
        self.assertEqual(
            len(source_variant_results),
            runner.EXPECTED_ACTIVE_QUESTION_COUNT,
        )
        self.assertEqual(
            sum(
                len(result["synthesized_question_variants"])
                for result in source_variant_results
            ),
            runner.EXPECTED_VARIANT_FORMULATION_COUNT,
        )
        self.assertEqual(
            len(derived_results),
            runner.EXPECTED_DERIVED_QUESTION_COUNT,
        )
        self.assertEqual(
            len(derived_cases),
            sum(runner.DERIVED_EXPECTED_CASE_COUNTS.values()),
        )
        self.assertEqual(
            Counter(
                case["inference_input"]["piece_id"]
                for case in derived_cases
            ),
            runner.DERIVED_EXPECTED_CASE_COUNTS,
        )
        self.assertEqual(
            sum(
                len(result["synthesized_question_variants"])
                if result["question_provenance"]
                == runner.SOURCE_VARIANT_PROVENANCE
                else 1
                for result in benchmark.results
            ),
            runner.EXPECTED_FORMULATION_COUNT,
        )
        self.assertTrue(
            all(
                result["synthesized_question_variants"] == []
                for result in derived_results
            )
        )
        snapshot = {
            "run": {},
            "results": deepcopy(benchmark.results),
        }
        runner.refresh_summary(snapshot)
        summary = snapshot["summary"]
        self.assertEqual(
            summary["synthesized_variant_formulation_count"],
            runner.EXPECTED_VARIANT_FORMULATION_COUNT,
        )
        self.assertEqual(
            summary["knowledge_unit_derived_formulation_count"],
            runner.EXPECTED_DERIVED_QUESTION_COUNT,
        )
        self.assertEqual(
            summary["formulation_count"],
            runner.EXPECTED_FORMULATION_COUNT,
        )
        self.assertEqual(
            summary["knowledge_unit_derived_case_count"],
            sum(runner.DERIVED_EXPECTED_CASE_COUNTS.values()),
        )
        self.assertEqual(
            summary["by_question_provenance"][
                runner.DERIVED_QUESTION_PROVENANCE
            ]["cases"],
            sum(runner.DERIVED_EXPECTED_CASE_COUNTS.values()),
        )
        self.assertEqual(
            summary["evaluation_cohorts"][
                runner.DERIVED_EVALUATION_COHORT
            ]["cases"],
            sum(runner.DERIVED_EXPECTED_CASE_COUNTS.values()),
        )
        deictic = next(
            result
            for result in derived_results
            if result["source_id"]
            == "in-flowery-clouds-ku-019-derived-q01"
        )
        expected_representative = runner.select_representative_range(
            deictic["source_id"],
            deictic["inference_measure_ranges"],
        )
        self.assertEqual(
            deictic["representative_inference_measure_range"],
            expected_representative,
        )
        self.assertEqual(len(deictic["inference_runs"]), 1)
        deictic_input = deictic["inference_runs"][0]["inference_input"]
        self.assertEqual(deictic_input["measure_range"], expected_representative)
        for unit_id in runner.DERIVED_INTRINSIC_RANGE_ONLY_KNOWLEDGE_UNIT_IDS:
            with self.subTest(derived_intrinsic_range_only=unit_id):
                result = next(
                    item
                    for item in derived_results
                    if item["source_id"] == f"{unit_id}-derived-q01"
                )
                self.assertEqual(len(result["inference_runs"]), 1)
                self.assertIsNotNone(
                    result["inference_runs"][0]["inference_input"][
                        "measure_range"
                    ]
                )

        hint_derived = next(
            item
            for item in derived_results
            if item["source_id"]
            == "in-flowery-clouds-ku-003-derived-q01"
        )
        self.assertEqual(hint_derived["inference_measure_ranges"], [])
        self.assertEqual(hint_derived["measure_range_hints"], [[11, 11]])
        self.assertEqual(
            hint_derived["evaluation_context_measure_ranges"],
            [[11, 11]],
        )
        hint_derived_input = hint_derived["inference_runs"][0][
            "inference_input"
        ]
        self.assertEqual(hint_derived_input["measure_range"], [11, 11])
        self.assertEqual(
            hint_derived_input["measure_range_provenance"],
            runner.REVIEWED_HINT_CONTEXT_RANGE_PROVENANCE,
        )
        self.assertEqual(
            hint_derived_input["range_source_knowledge_unit_ids"],
            [],
        )
        hint_authority = hint_derived_input["case_reference_authority"]["items"]
        self.assertEqual(hint_authority[0]["measure_status"], "whole_piece")
        self.assertEqual(hint_authority[0]["measure_ranges"], [])
        self.assertEqual(
            deictic_input["evaluation_question_id"],
            "in-flowery-clouds-ku-019-derived-q01",
        )
        self.assertIsNone(deictic_input["synthesized_variant_id"])
        self.assertEqual(
            deictic_input["question_provenance"],
            runner.DERIVED_QUESTION_PROVENANCE,
        )
        self.assertEqual(
            deictic_input["expected_knowledge_unit_ids"],
            ["in-flowery-clouds-ku-019"],
        )
        self.assertEqual(
            [
                item["knowledge_unit_id"]
                for item in deictic_input["case_reference_authority"]["items"]
            ],
            ["in-flowery-clouds-ku-019"],
        )
        flowery = next(
            case
            for case in cohort
            if case["case_id"]
            == "kim-in-flowery-clouds-01-syn-01__no-range"
        )
        self.assertEqual(
            flowery["inference_input"]["question"],
            "악보의 페르마타 표시가 음원에서 들리는 음 길이와 "
            "일치하지 않을 때는 어떻게 해야 할까?",
        )
        self.assertEqual(
            flowery["inference_input"][
                "semantic_required_any_knowledge_unit_ids"
            ],
            ["in-flowery-clouds-ku-001"],
        )
        flowery_authority = flowery["inference_input"][
            "case_reference_authority"
        ]["items"][0]
        self.assertEqual(flowery_authority["reference_role"], "required_any")
        self.assertEqual(flowery_authority["measure_status"], "specific")
        self.assertEqual(flowery_authority["measure_ranges"], [[1, 2]])

    def test_original_and_synthesized_use_the_same_representative_ranges(
        self,
    ) -> None:
        original_questions, _ = original_runner.load_qualitative_questions(
            runner.DEFAULT_DATASET_ROOT
        )
        benchmark = runner.load_benchmark(runner.DEFAULT_DATASET_ROOT)
        original_ranges = {
            question["source_id"]: question[
                "representative_inference_measure_range"
            ]
            for question in original_questions
        }

        for result in benchmark.results:
            if (
                result["question_provenance"]
                != runner.SOURCE_VARIANT_PROVENANCE
            ):
                continue
            expected = original_ranges[result["source_id"]]
            self.assertEqual(
                result["representative_inference_measure_range"],
                expected,
            )
            regular_ranges = {
                tuple(case["inference_input"]["measure_range"])
                if case["inference_input"]["measure_range"] is not None
                else None
                for case in result["inference_runs"]
                if case["inference_input"].get("evaluation_cohort") is None
            }
            self.assertEqual(
                regular_ranges,
                {tuple(expected) if expected is not None else None},
            )

    def test_loader_binds_mixed_schemas_and_skips_excluded_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            add_piece(
                root,
                piece_id="die-forelle",
                schema_version="1.3",
                token="첫째",
                specific_range=True,
            )
            add_piece(
                root,
                piece_id="nella-fantasia",
                schema_version="1.1",
                token="둘째",
                include_excluded=True,
            )
            benchmark = runner.load_benchmark(
                root,
                piece_ids=("die-forelle", "nella-fantasia"),
                enforce_expected_counts=False,
            )

        self.assertEqual(len(benchmark.results), 2)
        self.assertEqual(
            sum(len(item["synthesized_question_variants"]) for item in benchmark.results),
            6,
        )
        self.assertEqual(
            sum(len(item["inference_runs"]) for item in benchmark.results),
            6,
        )
        self.assertNotIn(
            "kim-nella-fantasia-01",
            {item["source_id"] for item in benchmark.results},
        )
        scoped = benchmark.results[0]["inference_runs"][0]["inference_input"][
            "case_reference_authority"
        ]
        unscoped = benchmark.results[1]["inference_runs"][0]["inference_input"][
            "case_reference_authority"
        ]
        self.assertEqual(
            scoped["source"],
            "applicable_finalized_knowledge_units",
        )
        self.assertFalse(scoped["manual_review_required"])
        self.assertEqual(
            unscoped["source"],
            "applicable_finalized_knowledge_units",
        )
        self.assertFalse(unscoped["manual_review_required"])
        for authority, token in ((scoped, "첫째"), (unscoped, "둘째")):
            self.assertEqual(
                authority["items"][0]["scope"],
                "curated_knowledge_unit_answer",
            )
            self.assertEqual(
                authority["items"][0]["scope_authority"],
                "finalized_knowledge_unit",
            )
            self.assertEqual(
                authority["items"][0]["text"],
                f"{token} 최종 지식 단위 답변이다.",
            )
            self.assertNotIn("전문가 답변", authority["items"][0]["text"])
            self.assertEqual(
                authority["items"][0]["reference_role"],
                "required_any",
            )
        self.assertEqual(scoped["items"][0]["measure_status"], "specific")
        self.assertEqual(scoped["items"][0]["measure_ranges"], [[2, 5]])
        self.assertEqual(
            unscoped["items"][0]["measure_status"],
            "whole_piece",
        )
        self.assertEqual(unscoped["items"][0]["measure_ranges"], [])
        self.assertEqual(
            [item["kind"] for item in benchmark.exclusions],
            ["question"],
        )
        self.assertEqual(
            benchmark.exclusions[0]["source_id"],
            "kim-nella-fantasia-01",
        )

    def test_loader_adds_one_evaluation_only_question_for_a_questionless_unit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            add_piece(
                root,
                piece_id="die-forelle",
                schema_version="1.3",
                token="첫째",
            )
            catalog_path, source_id, unit_id = (
                add_derived_knowledge_unit_question(root)
            )
            benchmark = runner.load_benchmark(
                root,
                piece_ids=("die-forelle",),
                enforce_expected_counts=False,
            )

        self.assertEqual(len(benchmark.results), 2)
        self.assertEqual(
            sum(len(result["inference_runs"]) for result in benchmark.results),
            5,
        )
        self.assertIn(catalog_path, benchmark.input_paths)
        self.assertIn(
            root / "expert_curation" / "derived_knowledge_unit_questions.py",
            benchmark.input_paths,
        )
        derived_id = f"{unit_id}-derived-q01"
        derived = next(
            result
            for result in benchmark.results
            if result["source_id"] == derived_id
        )
        representative = runner.select_representative_range(
            derived_id,
            [[4, 4], [12, 12], [20, 21]],
        )
        self.assertEqual(
            derived["representative_inference_measure_range"],
            representative,
        )
        self.assertEqual(
            [
                case["inference_input"]["measure_range"]
                for case in derived["inference_runs"]
            ],
            [None, representative],
        )
        self.assertEqual(derived["source_annotation_ids"], [source_id])
        self.assertEqual(
            derived["knowledge_unit_ids"],
            [unit_id],
        )
        for case in derived["inference_runs"]:
            case_input = case["inference_input"]
            self.assertEqual(case_input["evaluation_question_id"], derived_id)
            self.assertEqual(case_input["lineage_source_ids"], [source_id])
            self.assertEqual(
                case_input["expected_knowledge_unit_ids"],
                [unit_id],
            )
            self.assertEqual(
                [
                    item["knowledge_unit_id"]
                    for item in case_input["case_reference_authority"]["items"]
                ],
                [unit_id],
            )

    def test_loader_rejects_derived_catalog_drift_or_retrieval_aliases(
        self,
    ) -> None:
        for drift, expected in (
            ("answer_hash", "answer hash drift"),
            ("range_policy", "expected range_policy"),
            ("retrieval_alias", "duplicates a retrieval alias"),
            ("source_question_type", "question: expected a string"),
        ):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                add_piece(
                    root,
                    piece_id="die-forelle",
                    schema_version="1.3",
                    token="첫째",
                )
                catalog_path, derived_source_id, _ = (
                    add_derived_knowledge_unit_question(root)
                )
                catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                item = catalog["questions"][0]
                if drift == "answer_hash":
                    item["knowledge_unit_answer_sha256"] = "0" * 64
                elif drift == "range_policy":
                    item["range_policy"] = "no_range_only"
                elif drift == "retrieval_alias":
                    item["question"] = "첫째 원래 질문은 무엇인가?"
                else:
                    review_path = (
                        root
                        / "expert_curation"
                        / "review"
                        / "die-forelle.json"
                    )
                    review = json.loads(review_path.read_text(encoding="utf-8"))
                    derived_source = next(
                        source
                        for source in review["source_annotations"]
                        if source["source_id"] == derived_source_id
                    )
                    derived_source["question"] = 123
                    write_json(review_path, review)
                if drift != "source_question_type":
                    write_json(catalog_path, catalog)

                with self.assertRaisesRegex(
                    runner.EvaluationInputError,
                    expected,
                ):
                    runner.load_benchmark(
                        root,
                        piece_ids=("die-forelle",),
                        enforce_expected_counts=False,
                    )

    def test_loader_rejects_nonfinal_knowledge_unit_case_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            add_piece(
                root,
                piece_id="die-forelle",
                schema_version="1.3",
                token="첫째",
            )
            review_path = (
                root / "expert_curation" / "review" / "die-forelle.json"
            )
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["knowledge_units"][0]["rewrite_status"] = "needs_review"
            write_json(review_path, review)
            inventory_path = (
                root
                / "expert_curation"
                / "evaluation_questions"
                / "die-forelle.json"
            )
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["questions"][0][
                "retrieval_eligible_knowledge_unit_ids"
            ] = []
            write_json(inventory_path, inventory)

            with self.assertRaisesRegex(
                runner.EvaluationInputError,
                "is not a finalized knowledge unit",
            ):
                runner.load_benchmark(
                    root,
                    piece_ids=("die-forelle",),
                    enforce_expected_counts=False,
                )

    def test_loader_rejects_excluded_question_identity_or_status_drift(self) -> None:
        for drift in ("identity", "review_status"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                add_piece(
                    root,
                    piece_id="nella-fantasia",
                    schema_version="1.1",
                    token="둘째",
                    include_excluded=True,
                )
                inventory_path = (
                    root
                    / "expert_curation"
                    / "evaluation_questions"
                    / "nella-fantasia.json"
                )
                inventory = runner.load_json(inventory_path)
                if drift == "review_status":
                    inventory["questions"][0]["review_status"] = "retrievable"
                    expected = "review_status drift"
                else:
                    review_path = (
                        root
                        / "expert_curation"
                        / "review"
                        / "nella-fantasia.json"
                    )
                    review = runner.load_json(review_path)
                    review["source_annotations"][0]["source_id"] = (
                        "kim-nella-fantasia-99"
                    )
                    inventory["questions"][0]["source_id"] = (
                        "kim-nella-fantasia-99"
                    )
                    write_json(review_path, review)
                    expected = "unexpected excluded question"
                write_json(inventory_path, inventory)
                with self.assertRaisesRegex(runner.EvaluationInputError, expected):
                    runner.load_benchmark(
                        root,
                        piece_ids=("nella-fantasia",),
                        enforce_expected_counts=False,
                    )

    def test_preflight_requires_hybrid_and_both_fingerprinted_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding = root / "embedding.gguf"
            generation = root / "generation.gguf"
            embedding.write_bytes(b"embedding")
            generation.write_bytes(b"generation")
            settings = {
                "embedding_model_path": str(embedding),
                "model_path": str(generation),
                "retrieval": {"mode": "hybrid"},
                "llm": {},
            }
            state = {
                "optional_assets": {
                    "embedding_model": runner.file_record(embedding),
                    "generation_model": runner.file_record(generation),
                }
            }
            status = {
                "path": str(generation),
                "checkpoint_exists": True,
                "backend": "llama-cpp-python",
            }
            contract = runner.validate_hybrid_generation_preflight(
                settings,
                system_state=state,
                model_status=status,
                runtime_state={"fingerprint": "runtime"},
            )
            self.assertEqual(
                contract["generation_model"]["sha256"],
                runner.file_sha256(generation),
            )
            state["optional_assets"]["generation_model"] = {
                "kind": "missing"
            }
            with self.assertRaisesRegex(
                runner.ModelPreflightError,
                "generation model checkpoint not found",
            ):
                runner.validate_hybrid_generation_preflight(
                    settings,
                    system_state=state,
                    model_status=status,
                    runtime_state={"fingerprint": "runtime"},
                )

    def test_pipeline_corpus_authentication_requires_declared_schema_8(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            corpus_path = root / "corpus.json"
            corpus_path.write_text("[]\n", encoding="utf-8")
            input_path = dataset_root / "review.json"
            input_path.write_text("{}\n", encoding="utf-8")
            stats_path = root / "corpus_stats.json"
            settings = {
                "corpus_path": str(corpus_path),
                "stats_path": str(stats_path),
                "dataset_root": str(dataset_root),
                "web_export_files": ["research-open.jsonl"],
            }
            stats = {
                "corpus_schema_version": 8,
                "web_export_files": ["research-open.jsonl"],
                "dataset_root": str(dataset_root.resolve()),
                "input_fingerprint": "fingerprint",
                "corpus_sha256": runner.file_sha256(corpus_path),
            }
            write_json(stats_path, stats)
            corpus_module = SimpleNamespace(
                CORPUS_SCHEMA_VERSION=8,
                corpus_input_fingerprint=lambda _settings: "fingerprint",
                corpus_input_paths=lambda _settings: [
                    ("review", input_path)
                ],
            )

            state = runner.authenticate_pipeline_corpus(
                settings,
                corpus_module,
            )
            self.assertEqual(state["input_fingerprint"], "fingerprint")

            for label, declared_schema, stats_schema, expected_failure in (
                ("stale stats", 8, 7, "schema"),
                ("stale target", 7, 7, "schema_declaration"),
                ("missing declaration", None, 8, "schema_declaration"),
            ):
                with self.subTest(label=label):
                    stats["corpus_schema_version"] = stats_schema
                    write_json(stats_path, stats)
                    if declared_schema is None:
                        del corpus_module.CORPUS_SCHEMA_VERSION
                    else:
                        corpus_module.CORPUS_SCHEMA_VERSION = declared_schema
                    with self.assertRaisesRegex(
                        runner.EvaluationInputError,
                        expected_failure,
                    ):
                        runner.authenticate_pipeline_corpus(
                            settings,
                            corpus_module,
                        )

    def test_target_retrieval_preflight_wraps_missing_asset_or_backend(self) -> None:
        for message in (
            "embedding checkpoint missing",
            "llama-cpp backend unavailable",
        ):
            with self.subTest(message=message):
                def fail(_settings, failure=message):
                    raise RuntimeError(failure)

                dense = SimpleNamespace(
                    validate_retrieval_requirements=fail,
                )
                with self.assertRaisesRegex(
                    runner.ModelPreflightError,
                    message,
                ):
                    runner.validate_target_retrieval_requirements(
                        {"retrieval": {"mode": "hybrid"}},
                        dense,
                    )

    def test_eager_initialization_checks_llm_before_corpus_index(self) -> None:
        calls = []

        def corpus_stats():
            calls.append("retrieval")
            return {
                "retrieval": {
                    "configured_mode": "hybrid",
                    "active_mode": "hybrid",
                    "dense_available": True,
                    "last_dense_error": None,
                }
            }

        def validate_generation_requirements(model_path, llm_settings):
            calls.append(("generation", model_path, llm_settings))

        target = SimpleNamespace(
            corpus_stats=corpus_stats,
            validate_generation_requirements=(
                validate_generation_requirements
            ),
            pipeline_settings={
                "model_path": "/models/generation.gguf",
                "llm": {
                    "n_ctx": 4096,
                    "n_gpu_layers": -1,
                    "chat_format": "chatml",
                },
            },
        )
        initialized = runner.initialize_target_models(target)

        self.assertEqual(calls[0][0], "generation")
        self.assertEqual(calls[0][1], "/models/generation.gguf")
        self.assertEqual(calls[0][2]["n_ctx"], 4096)
        self.assertEqual(calls[1], "retrieval")
        self.assertEqual(initialized["generation"]["status"], "validated")
        self.assertEqual(initialized["generation"]["chat_format"], "chatml")

    def test_eager_initialization_records_configured_chat_format(self) -> None:
        cases = (
            ("omitted", {}, None),
            ("blank", {"chat_format": "  \t"}, None),
            (
                "explicit",
                {"chat_format": "  qwen3-native  "},
                "qwen3-native",
            ),
        )

        for label, llm_settings, expected in cases:
            with self.subTest(label=label):
                target = SimpleNamespace(
                    corpus_stats=lambda: {
                        "retrieval": {
                            "configured_mode": "hybrid",
                            "active_mode": "hybrid",
                            "dense_available": True,
                            "last_dense_error": None,
                        }
                    },
                    validate_generation_requirements=lambda *_args: None,
                    pipeline_settings={
                        "model_path": "/models/generation.gguf",
                        "llm": llm_settings,
                    },
                )

                initialized = runner.initialize_target_models(target)

                self.assertEqual(
                    initialized["generation"]["chat_format"],
                    expected,
                )

    def test_run_case_uses_grounded_generation_and_preserves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
            calls = []

            def ask(**kwargs):
                calls.append(kwargs)
                return clean_hybrid_response(model_path)

            attempted = runner.run_cases(
                snapshot,
                ask=ask,
                top_k=6,
                limit=None,
                checkpoint=lambda: None,
            )

        case = snapshot["results"][0]["inference_runs"][0]
        self.assertEqual(attempted, 1)
        self.assertTrue(calls[0]["generate"])
        self.assertNotIn("allow_internal_knowledge", calls[0])
        self.assertEqual(case["status"], "completed")
        self.assertEqual(case["generated_answer"]["answer"], "근거 기반 생성 답변")
        for removed_field in (
            "rag_llm_succeeded",
            "authoritative_expert_succeeded",
            "grounded_answer_succeeded",
        ):
            self.assertNotIn(removed_field, case["generated_answer"])
        rollup = runner._rollup([case])
        self.assertEqual(rollup["generation_modes"], {"llm": 1})
        self.assertEqual(rollup["answer_bases"], {"retrieved_evidence": 1})
        for removed_field in (
            "rag_llm_succeeded_cases",
            "authoritative_expert_succeeded_cases",
            "grounded_answer_succeeded_cases",
            "unavailable_reasons",
        ):
            self.assertNotIn(removed_field, rollup)
        self.assertEqual(
            case["retrieval_probe"]["diagnostics"]["retrieval_diagnostics"][
                "active_mode"
            ],
            "hybrid",
        )

    def test_legacy_authoritative_expert_answer_fails_the_case_and_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
            with self.assertRaisesRegex(
                runner.EvaluationInputError,
                "requires an llm/retrieved_evidence answer path",
            ):
                runner.run_cases(
                    snapshot,
                    ask=lambda **_: clean_hybrid_response(
                        model_path,
                        generation_mode="expert_verbatim",
                        answer_basis="authoritative_expert",
                    ),
                    top_k=6,
                    limit=None,
                    checkpoint=lambda: None,
                )

        case = snapshot["results"][0]["inference_runs"][0]
        self.assertEqual(case["status"], "error")
        self.assertIsNone(case["generated_answer"])
        rollup = runner._rollup([case])
        self.assertEqual(rollup["completed_cases"], 0)
        self.assertEqual(rollup["error_cases"], 1)

    def test_explicit_unavailable_answer_fails_the_case_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
            with self.assertRaisesRegex(
                runner.EvaluationInputError,
                "requires an llm/retrieved_evidence answer path",
            ):
                runner.run_cases(
                    snapshot,
                    ask=lambda **_: clean_hybrid_response(
                        model_path,
                        unavailable_reason=(
                            "insufficient_authoritative_evidence"
                        ),
                    ),
                    top_k=6,
                    limit=None,
                    checkpoint=lambda: None,
                )

        case = snapshot["results"][0]["inference_runs"][0]
        self.assertEqual(case["status"], "error")
        self.assertIsNone(case["generated_answer"])
        self.assertEqual(runner._rollup([case])["error_cases"], 1)

    def test_natural_language_refusal_fails_the_case_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
            response = clean_hybrid_response(model_path)
            response["answer"] = (
                "현재 제공된 정보만으로는 정확히 답변하기 어렵습니다."
            )

            with self.assertRaisesRegex(
                runner.EvaluationInputError,
                "grounded-insufficiency refusal",
            ):
                runner.run_cases(
                    snapshot,
                    ask=lambda **_: response,
                    top_k=6,
                    limit=None,
                    checkpoint=lambda: None,
                )

        case = snapshot["results"][0]["inference_runs"][0]
        self.assertEqual(case["status"], "error")
        self.assertIsNone(case["generated_answer"])
        self.assertEqual(runner._rollup([case])["error_cases"], 1)

    def test_unsupported_answer_paths_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            invalid_responses = (
                clean_hybrid_response(
                    model_path,
                    generation_mode="expert_verbatim",
                    answer_basis="retrieved_evidence",
                ),
                clean_hybrid_response(
                    model_path,
                    generation_mode="extractive",
                    answer_basis="retrieval_extractive",
                ),
                clean_hybrid_response(
                    model_path,
                    unavailable_reason=(
                        "grounded model returned no usable answer"
                    ),
                    generation_mode="expert_verbatim",
                    answer_basis="authoritative_expert",
                ),
            )
            for response in invalid_responses:
                with self.subTest(response=response):
                    snapshot = build_case_snapshot(model_path)
                    with self.assertRaises(runner.EvaluationInputError):
                        runner.run_cases(
                            snapshot,
                            ask=lambda response=response, **_: response,
                            top_k=6,
                            limit=None,
                            checkpoint=lambda: None,
                        )

    def test_llm_retrieved_answer_requires_nonempty_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
            response = clean_hybrid_response(model_path)
            response["evidence"] = []

            with self.assertRaisesRegex(
                runner.EvaluationInputError,
                "retrieved-evidence LLM answers require evidence",
            ):
                runner.run_cases(
                    snapshot,
                    ask=lambda **_: response,
                    top_k=6,
                    limit=None,
                    checkpoint=lambda: None,
                )

        case = snapshot["results"][0]["inference_runs"][0]
        self.assertEqual(case["status"], "error")
        self.assertIsNone(case["generated_answer"])

    def test_removed_generation_metadata_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            removed_fields = {
                "generation_fallback_reason": "local generation failed",
                "rag_llm_succeeded": True,
                "authoritative_expert_succeeded": False,
                "grounded_answer_succeeded": True,
            }
            for field, value in removed_fields.items():
                with self.subTest(field=field):
                    snapshot = build_case_snapshot(model_path)
                    response = clean_hybrid_response(model_path)
                    response[field] = value

                    with self.assertRaisesRegex(
                        runner.EvaluationInputError,
                        "removed generation fields",
                    ):
                        runner.run_cases(
                            snapshot,
                            ask=lambda **_: response,
                            top_k=6,
                            limit=None,
                            checkpoint=lambda: None,
                        )

                    case = snapshot["results"][0]["inference_runs"][0]
                    self.assertEqual(case["status"], "error")
                    self.assertEqual(case["attempt_count"], 1)
                    self.assertIsNone(case["generated_answer"])

    def test_empty_eligible_lane_is_not_counted_as_a_retrieval_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path, eligible=False)
            runner.run_cases(
                snapshot,
                ask=lambda **_: clean_hybrid_response(model_path),
                top_k=6,
                limit=None,
                checkpoint=lambda: None,
            )

        case = snapshot["results"][0]["inference_runs"][0]
        rollup = runner._rollup([case])
        self.assertEqual(
            rollup["retrieval_eligible"]["target_available_cases"],
            0,
        )
        self.assertEqual(
            rollup["retrieval_eligible"]["target_unavailable_cases"],
            1,
        )
        self.assertIsNone(rollup["retrieval_eligible"]["hit_at_k_rate"])

    def test_no_range_semantic_rollup_reports_hit_at_3_without_changing_hit_at_k(
        self,
    ) -> None:
        cases = []
        for index, first_rank in enumerate((1, 3, 4, None), start=1):
            hit = first_rank is not None
            cases.append(
                {
                    "case_id": f"semantic-case-{index}",
                    "status": "completed",
                    "inference_input": {
                        "piece_id": "die-forelle",
                        "evaluation_cohort": runner.NO_RANGE_COHORT_NAME,
                    },
                    "retrieval_probe": {
                        "diagnostics": {
                            "semantic_required_any": {
                                "first_rank": first_rank,
                                "reciprocal_rank": (
                                    1 / first_rank if hit else 0.0
                                ),
                                "hit_at_1": first_rank == 1,
                                "hit_at_k": hit,
                            }
                        }
                    },
                }
            )

        rollup = runner._semantic_no_range_cohort_rollup(cases)

        self.assertEqual(rollup["completed_cases"], 4)
        self.assertEqual(rollup["semantic_target_hit_at_1_cases"], 1)
        self.assertEqual(rollup["semantic_target_hit_at_3_cases"], 2)
        self.assertEqual(rollup["semantic_target_hit_at_3_rate"], 0.5)
        # Hit@k remains the configured-depth result: rank 4 still counts.
        self.assertEqual(rollup["semantic_target_hit_at_k_cases"], 3)
        self.assertEqual(rollup["semantic_target_hit_at_k_rate"], 0.75)
        self.assertEqual(
            rollup["semantic_target_mean_reciprocal_rank"],
            0.395833,
        )

    def test_resume_rejects_corrupt_completed_case_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            current = build_case_snapshot(model_path)
            immutable_run = {
                "integrity": {
                    "status": "pending",
                    "validated_at": None,
                    "error": None,
                },
                "input_fingerprint": "input",
                "system_fingerprint": "system",
                "run_fingerprint": "run",
                "benchmark_dataset_root": "/benchmark",
                "pipeline_dataset_root": "/pipeline-dataset",
                "system_root": "/system",
                "target_pieces": ["die-forelle"],
                "top_k": 6,
                "generate": True,
                "required_retrieval_mode": "hybrid",
                "input_files": [],
                "system_state": {},
                "runtime_fingerprint": "runtime",
                "pipeline_settings": {},
                "pipeline_corpus_state": {},
                "target_module_paths": {},
                "excluded_cases": [],
            }
            current["run"].update(deepcopy(immutable_run))
            current.update(
                artifact_type=runner.ARTIFACT_TYPE,
                schema_version=runner.SCHEMA_VERSION,
            )
            expected = deepcopy(current)
            runner.run_cases(
                current,
                ask=lambda **_: clean_hybrid_response(model_path),
                top_k=6,
                limit=None,
                checkpoint=lambda: None,
            )
            runner.validate_resume_snapshot(current, expected)

            prior_schema = deepcopy(current)
            prior_schema["schema_version"] = "1.1"
            with self.assertRaisesRegex(
                runner.ResumeMismatchError,
                "schema version is incompatible",
            ):
                runner.validate_resume_snapshot(prior_schema, expected)

            corruptions = {
                "missing generated output": lambda case: case.update(
                    generated_answer=None
                ),
                "mismatched piece": lambda case: case[
                    "generated_answer"
                ].update(piece_id="other-piece"),
                "invalid attempt type": lambda case: case.update(
                    attempt_count=True
                ),
            }
            for label, corrupt in corruptions.items():
                with self.subTest(label=label):
                    damaged = deepcopy(current)
                    damaged_case = damaged["results"][0]["inference_runs"][0]
                    corrupt(damaged_case)
                    with self.assertRaises(runner.ResumeMismatchError):
                        runner.validate_resume_snapshot(damaged, expected)


if __name__ == "__main__":
    unittest.main()
