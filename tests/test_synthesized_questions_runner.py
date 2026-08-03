"""Focused tests for the five-piece synthesized hybrid RAG+LLM runner."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluation import run_synthesized_questions as runner


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
    source_id = f"kim-{piece_id}-02" if include_excluded else f"kim-{piece_id}-01"
    source_question = f"{token} 원래 질문은 무엇인가?"
    source_answer = f"{token} 전문가 답변이다."
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
        "answer": source_answer,
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


def clean_hybrid_response(model_path: Path, *, reason=None) -> dict:
    return {
        "piece_id": "die-forelle",
        "scope": "whole_piece",
        "measure_range": None,
        "answer": "근거 기반 생성 답변",
        "generation_mode": "llm" if reason is None else "extractive",
        "answer_basis": (
            "retrieved_evidence" if reason is None else "retrieval_extractive"
        ),
        "generation_fallback_reason": reason,
        "context_limited": False,
        "pipeline": "soprano_qa",
        "model": {
            "path": str(model_path),
            "checkpoint_exists": True,
            "llama_cpp_available": True,
        },
        "retrieval": {
            "configured_mode": "hybrid",
            "active_mode": "hybrid",
            "dense_available": True,
            "fallback_reason": None,
            "last_dense_error": None,
            "last_search_mode": "hybrid",
            "query_route": "hybrid",
            "route_reason": "semantic_query",
            "dense_attempted": True,
            "dense_contributed": True,
            "fallback_used": False,
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
        self.assertEqual(scoped["source"], "reference_claim_scope")
        self.assertFalse(scoped["manual_review_required"])
        self.assertEqual(unscoped["source"], "unscoped_source_answer")
        self.assertTrue(unscoped["manual_review_required"])
        self.assertEqual(unscoped["items"][0]["scope"], "unscoped_source_answer")
        self.assertEqual(
            [item["kind"] for item in benchmark.exclusions],
            ["question"],
        )
        self.assertEqual(
            benchmark.exclusions[0]["source_id"],
            "kim-nella-fantasia-01",
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
                "llama_cpp_available": True,
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

    def test_eager_model_initialization_loads_retrieval_then_llm(self) -> None:
        calls = []

        def corpus_stats():
            calls.append("retrieval")
            return {
                "retrieval": {
                    "configured_mode": "hybrid",
                    "active_mode": "hybrid",
                    "dense_available": True,
                    "fallback_reason": None,
                    "last_dense_error": None,
                }
            }

        def load_generation_model(**kwargs):
            calls.append(("generation", kwargs))
            return object()

        target = SimpleNamespace(
            corpus_stats=corpus_stats,
            load_generation_model=load_generation_model,
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

        self.assertEqual(calls[0], "retrieval")
        self.assertEqual(calls[1][0], "generation")
        self.assertEqual(calls[1][1]["n_ctx"], 4096)
        self.assertEqual(initialized["generation"]["status"], "loaded")

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
        self.assertFalse(calls[0]["allow_internal_knowledge"])
        self.assertEqual(case["status"], "completed")
        self.assertEqual(case["generated_answer"]["answer"], "근거 기반 생성 답변")
        self.assertEqual(
            case["retrieval_probe"]["diagnostics"]["retrieval_diagnostics"][
                "active_mode"
            ],
            "hybrid",
        )

    def test_operational_generation_fallback_aborts_resumable_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)

            with self.assertRaisesRegex(
                runner.PipelineExecutionError,
                "Operational generation failure",
            ):
                runner.run_cases(
                    snapshot,
                    ask=lambda **_: clean_hybrid_response(
                        model_path,
                        reason="local generation failed",
                    ),
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
                "allow_internal_knowledge": False,
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
