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

        self.assertEqual(len(benchmark.results), 79)
        self.assertEqual(len(cases), 483)
        self.assertEqual(len(cohort), 114)
        self.assertEqual(
            Counter(
                case["inference_input"]["piece_id"]
                for case in cohort
            ),
            {
                "die-forelle": 9,
                "in-flowery-clouds": 15,
                "la-capinera": 48,
                "nella-fantasia": 12,
                "una-voce-poco-fa": 30,
            },
        )
        self.assertEqual(
            len({case["case_id"] for case in cases}),
            len(cases),
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
        self.assertTrue(case["generated_answer"]["rag_llm_succeeded"])
        self.assertFalse(
            case["generated_answer"]["authoritative_expert_succeeded"]
        )
        self.assertTrue(
            case["generated_answer"]["grounded_answer_succeeded"]
        )
        self.assertEqual(
            case["retrieval_probe"]["diagnostics"]["retrieval_diagnostics"][
                "active_mode"
            ],
            "hybrid",
        )

    def test_authoritative_expert_answer_is_grounded_without_becoming_rag_llm(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
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
        generated = case["generated_answer"]
        self.assertEqual(case["status"], "completed")
        self.assertEqual(generated["generation_mode"], "expert_verbatim")
        self.assertEqual(generated["answer_basis"], "authoritative_expert")
        self.assertIsNone(generated["unavailable_reason"])
        self.assertFalse(generated["rag_llm_succeeded"])
        self.assertTrue(generated["authoritative_expert_succeeded"])
        self.assertTrue(generated["grounded_answer_succeeded"])

        rollup = runner._rollup([case])
        self.assertEqual(rollup["generation_modes"], {"expert_verbatim": 1})
        self.assertEqual(rollup["answer_bases"], {"authoritative_expert": 1})
        self.assertEqual(rollup["unavailable_reasons"], {})
        self.assertEqual(rollup["rag_llm_succeeded_cases"], 0)
        self.assertEqual(rollup["authoritative_expert_succeeded_cases"], 1)
        self.assertEqual(rollup["grounded_answer_succeeded_cases"], 1)

    def test_explicit_unavailable_answer_is_not_counted_as_grounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
            runner.run_cases(
                snapshot,
                ask=lambda **_: clean_hybrid_response(
                    model_path,
                    unavailable_reason="insufficient_authoritative_evidence",
                ),
                top_k=6,
                limit=None,
                checkpoint=lambda: None,
            )

        case = snapshot["results"][0]["inference_runs"][0]
        generated = case["generated_answer"]
        self.assertEqual(generated["generation_mode"], "unavailable")
        self.assertEqual(generated["answer_basis"], "no_corpus_evidence")
        self.assertFalse(generated["rag_llm_succeeded"])
        self.assertFalse(generated["authoritative_expert_succeeded"])
        self.assertFalse(generated["grounded_answer_succeeded"])
        self.assertEqual(
            runner._rollup([case])["unavailable_reasons"],
            {"insufficient_authoritative_evidence": 1},
        )

    def test_authoritative_expert_metadata_fails_closed_when_inconsistent(
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

    def test_removed_generation_fallback_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "generation.gguf"
            model_path.write_bytes(b"model")
            snapshot = build_case_snapshot(model_path)
            response = clean_hybrid_response(model_path)
            response["generation_fallback_reason"] = "local generation failed"

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
