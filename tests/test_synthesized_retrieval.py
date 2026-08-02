"""Focused tests for the synthesized development retrieval evaluator."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation import run_synthesized_retrieval as synthesized


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class SynthesizedBenchmarkTests(unittest.TestCase):
    def build_dataset(self, root: Path) -> Path:
        piece_id = "die-forelle"
        source_id = "kim-die-forelle-01"
        unit_id = "die-forelle-ku-001"
        original = "원래 질문은 무엇일까?"
        base = "기본 질문은 어떻게 바꿀까?"
        inventory = {
            "schema_version": "1.3",
            "piece_id": piece_id,
            "questions": [
                {
                    "source_id": source_id,
                    "annotator": "kim",
                    "original_question": original,
                    "paraphrased_question": base,
                    "knowledge_unit_ids": [unit_id],
                    "retrieval_eligible_knowledge_unit_ids": [unit_id],
                    "measure_range_hints": [],
                    "inference_measure_ranges": [[2, 5], [10, 12]],
                    "inference_scope": "measure_range",
                    "review_status": "retrievable",
                }
            ],
        }
        review = {
            "schema_version": "1.0",
            "piece_id": piece_id,
            "source_annotations": [
                {
                    "source_id": source_id,
                    "question": original,
                    "answer": "전문가 답변",
                    "curation_status": "included",
                }
            ],
            "knowledge_units": [
                {
                    "knowledge_unit_id": unit_id,
                    "source_ids": [source_id],
                    "answer": "전문가 답변",
                    "measure_status": "specific",
                    "measure_ranges": [[2, 5], [10, 12]],
                    "rewrite_status": "ready",
                }
            ],
        }
        variants = {
            "schema_version": "1.0",
            "piece_id": piece_id,
            "source_inventory_schema_version": "1.3",
            "variants_per_question": 3,
            "questions": [
                {
                    "source_id": source_id,
                    "original_question": original,
                    "base_paraphrased_question": base,
                    "variants": [
                        {
                            "variant_id": f"{source_id}-syn-01",
                            "question": "첫 번째 표현으로 무엇을 물어볼까?",
                            "transformations": ["lexical_substitution"],
                        },
                        {
                            "variant_id": f"{source_id}-syn-02",
                            "question": "무엇을 물을지 두 번째 표현으로 바꾸면?",
                            "transformations": ["syntactic_reframing"],
                        },
                        {
                            "variant_id": f"{source_id}-syn-03",
                            "question": "세 번째 방식이라면 질문을 어떻게 할까?",
                            "transformations": [
                                "information_structure",
                                "word_order",
                            ],
                        },
                    ],
                }
            ],
        }
        write_json(
            root / "expert_curation" / "evaluation_questions" / f"{piece_id}.json",
            inventory,
        )
        write_json(
            root / "expert_curation" / "review" / f"{piece_id}.json",
            review,
        )
        variant_path = (
            root
            / "expert_curation"
            / "evaluation_question_variants"
            / f"{piece_id}.json"
        )
        write_json(variant_path, variants)
        return variant_path

    def test_loads_three_variants_over_the_same_canonical_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_dataset(root)
            benchmark = synthesized.load_benchmark(
                root,
                piece_ids=("die-forelle",),
                enforce_expected_counts=False,
            )

        self.assertEqual(len(benchmark.cases), 6)
        self.assertEqual(
            [case["variant_index"] for case in benchmark.cases],
            [1, 1, 2, 2, 3, 3],
        )
        self.assertEqual(
            {case["formulation_id"] for case in benchmark.cases},
            {
                "kim-die-forelle-01__m2-5",
                "kim-die-forelle-01__m10-12",
            },
        )
        self.assertTrue(
            all(
                case["applicable_knowledge_unit_ids"]
                == ["die-forelle-ku-001"]
                for case in benchmark.cases
            )
        )

    def test_rejects_base_binding_drift_and_missing_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variant_path = self.build_dataset(root)
            payload = json.loads(variant_path.read_text(encoding="utf-8"))
            payload["questions"][0]["base_paraphrased_question"] = "바뀐 기준"
            write_json(variant_path, payload)
            with self.assertRaisesRegex(
                synthesized.EvaluationInputError,
                "base paraphrase binding drift",
            ):
                synthesized.load_benchmark(
                    root,
                    piece_ids=("die-forelle",),
                    enforce_expected_counts=False,
                )

            payload["questions"][0]["base_paraphrased_question"] = (
                "기본 질문은 어떻게 바꿀까?"
            )
            payload["questions"][0]["variants"].pop()
            write_json(variant_path, payload)
            with self.assertRaisesRegex(
                synthesized.EvaluationInputError,
                "exactly three variants",
            ):
                synthesized.load_benchmark(
                    root,
                    piece_ids=("die-forelle",),
                    enforce_expected_counts=False,
                )


class RetrievalDiagnosticsTests(unittest.TestCase):
    CASE = {
        "source_id": "source-01",
        "expected_knowledge_unit_ids": ["ku-1", "ku-2"],
        "applicable_knowledge_unit_ids": ["ku-2"],
    }

    def test_records_ordered_ranks_hits_and_source_grounding(self) -> None:
        result = {
            "evidence": [
                {
                    "id": "other",
                    "kind": "web",
                    "source_ids": [],
                    "in_requested_scope": True,
                },
                {
                    "id": "ku-2",
                    "kind": "expert",
                    "source_ids": ["source-01"],
                    "in_requested_scope": True,
                },
                {
                    "id": "ku-1",
                    "kind": "expert",
                    "source_ids": ["source-01"],
                    "in_requested_scope": False,
                },
            ]
        }
        diagnosed = synthesized.diagnose_result(
            result,
            self.CASE,
            top_k=3,
            required_mode="lexical",
        )

        self.assertEqual(
            diagnosed["ordered_evidence_ids"],
            ["other", "ku-2", "ku-1"],
        )
        self.assertEqual(diagnosed["expected"]["first_rank"], 2)
        self.assertEqual(diagnosed["expected"]["reciprocal_rank"], 0.5)
        self.assertFalse(diagnosed["expected"]["hit_at_1"])
        self.assertTrue(diagnosed["expected"]["hit_at_k"])
        self.assertTrue(diagnosed["expected"]["all_retrieved"])
        self.assertEqual(diagnosed["applicable"]["first_rank"], 2)
        self.assertTrue(diagnosed["target_source_grounded"])
        self.assertEqual(
            diagnosed["retrieval_diagnostics"]["diagnostic_source"],
            "implicit_legacy_lexical",
        )

    def test_required_hybrid_rejects_lexical_fallback(self) -> None:
        result = {
            "evidence": [],
            "retrieval": {
                "configured_mode": "hybrid",
                "active_mode": "lexical",
                "dense_available": False,
                "fallback_reason": "checkpoint missing",
                "last_dense_error": None,
                "last_search_mode": "lexical_fallback",
            },
        }
        with self.assertRaises(synthesized.RetrievalModeMismatchError):
            synthesized.diagnose_result(
                result,
                self.CASE,
                top_k=6,
                required_mode="hybrid",
            )

    def test_required_hybrid_accepts_explicit_nonfallback_routes(self) -> None:
        for last_search_mode, query_route, dense_attempted in (
            ("lexical_route", "lexical", False),
            ("hybrid_no_dense_match", "hybrid", True),
        ):
            with self.subTest(last_search_mode=last_search_mode):
                result = {
                    "evidence": [],
                    "retrieval": {
                        "configured_mode": "hybrid",
                        "active_mode": "hybrid",
                        "dense_available": True,
                        "fallback_reason": None,
                        "last_dense_error": None,
                        "last_search_mode": last_search_mode,
                        "query_route": query_route,
                        "route_reason": "intentional_test_route",
                        "dense_attempted": dense_attempted,
                        "dense_contributed": False,
                        "fallback_used": False,
                    },
                }

                diagnosed = synthesized.diagnose_result(
                    result,
                    self.CASE,
                    top_k=6,
                    required_mode="hybrid",
                )

                diagnostics = diagnosed["retrieval_diagnostics"]
                self.assertEqual(
                    diagnostics["last_search_mode"],
                    last_search_mode,
                )
                self.assertFalse(diagnostics["fallback_used"])

    def test_required_hybrid_rejects_explicit_fallback_flag(self) -> None:
        result = {
            "evidence": [],
            "retrieval": {
                "configured_mode": "hybrid",
                "active_mode": "hybrid",
                "dense_available": True,
                "fallback_reason": None,
                "last_dense_error": None,
                "last_search_mode": "hybrid_no_dense_match",
                "query_route": "hybrid",
                "dense_attempted": True,
                "dense_contributed": False,
                "fallback_used": True,
            },
        }

        with self.assertRaisesRegex(
            synthesized.RetrievalModeMismatchError,
            "fallback_used",
        ):
            synthesized.diagnose_result(
                result,
                self.CASE,
                top_k=6,
                required_mode="hybrid",
            )


class ResumeAndAggregationTests(unittest.TestCase):
    @staticmethod
    def build_snapshot() -> dict:
        case_inputs = []
        for index in range(1, 4):
            case_inputs.append(
                {
                    "case_id": f"source-syn-{index:02d}__no-range",
                    "formulation_id": "source__no-range",
                    "piece_id": "die-forelle",
                    "source_id": "source",
                    "variant_id": f"source-syn-{index:02d}",
                    "variant_index": index,
                    "question": f"질문 {index}",
                    "transformations": ["word_order"],
                    "measure_range": None,
                    "expected_knowledge_unit_ids": ["ku"],
                    "retrieval_eligible_knowledge_unit_ids": ["ku"],
                    "applicable_knowledge_unit_ids": ["ku"],
                }
            )
        benchmark = synthesized.Benchmark(case_inputs, [], [])
        target = synthesized.TargetRuntime(
            ask=lambda **_: {},
            pipeline_settings={"dataset_root": "/pipeline"},
            pipeline_dataset_root=Path("/pipeline"),
            module_paths={"soprano_qa.service": "/system/soprano_qa/service.py"},
        )
        return synthesized.new_snapshot(
            benchmark=benchmark,
            benchmark_root=Path("/benchmark"),
            system_state={
                "system_root": "/system",
                "fingerprint": "system-fingerprint",
            },
            target=target,
            top_k=6,
            required_mode="lexical",
            input_files=[],
        )

    def test_summary_reports_three_formulation_hits_and_variant_rollups(self) -> None:
        snapshot = self.build_snapshot()
        for case in snapshot["cases"]:
            case["status"] = "completed"
            hit = case["input"]["variant_index"] != 2
            metric = {
                "first_rank": 1 if hit else None,
                "reciprocal_rank": 1.0 if hit else 0.0,
                "hit_at_1": hit,
                "hit_at_k": hit,
                "all_retrieved": hit,
                "retrieved_ids": ["ku"] if hit else [],
            }
            case["retrieval"] = {
                "ordered_evidence_ids": ["ku"] if hit else [],
                "expected": deepcopy(metric),
                "applicable": deepcopy(metric),
                "target_source_grounded": hit,
                "retrieval_diagnostics": {
                    "last_search_mode": (
                        "hybrid" if hit else "hybrid_no_dense_match"
                    ),
                    "query_route": "hybrid",
                    "dense_attempted": True,
                    "dense_contributed": hit,
                    "fallback_used": False,
                },
            }
        synthesized.refresh_summary(snapshot)

        consistency = snapshot["summary"]["formulation_consistency"]
        self.assertEqual(consistency["groups"], 1)
        self.assertEqual(consistency["completed_groups"], 1)
        self.assertEqual(consistency["all_variants_hit_at_k_groups"], 0)
        self.assertEqual(len(consistency["inconsistent_groups"]), 1)
        self.assertEqual(
            snapshot["summary"]["by_variant_index"]["2"]["expected"][
                "hit_at_k_rate"
            ],
            0.0,
        )
        overall = snapshot["summary"]["overall"]
        self.assertEqual(
            overall["retrieval_modes"],
            {"hybrid": 2, "hybrid_no_dense_match": 1},
        )
        self.assertEqual(overall["dense_attempted_cases"], 3)
        self.assertEqual(overall["dense_contributed_cases"], 2)
        self.assertEqual(overall["fallback_used_cases"], 0)

    def test_integrity_must_be_validated_before_completion(self) -> None:
        snapshot = self.build_snapshot()
        for case in snapshot["cases"]:
            case["status"] = "completed"
            metric = {
                "first_rank": 1,
                "reciprocal_rank": 1.0,
                "hit_at_1": True,
                "hit_at_k": True,
                "all_retrieved": True,
                "retrieved_ids": ["ku"],
            }
            case["retrieval"] = {
                "ordered_evidence_ids": ["ku"],
                "expected": deepcopy(metric),
                "applicable": deepcopy(metric),
                "target_source_grounded": True,
                "retrieval_diagnostics": {
                    "last_search_mode": "lexical",
                    "query_route": "lexical",
                    "dense_attempted": False,
                    "dense_contributed": False,
                    "fallback_used": False,
                },
            }

        synthesized.refresh_summary(snapshot)
        self.assertEqual(snapshot["summary"]["status"], "partial")
        self.assertEqual(snapshot["summary"]["integrity_status"], "pending")

        snapshot["run"]["integrity"]["status"] = "validated"
        synthesized.refresh_summary(snapshot)
        self.assertEqual(snapshot["summary"]["status"], "complete")

        snapshot["run"]["integrity"]["status"] = "invalid"
        synthesized.refresh_summary(snapshot)
        self.assertEqual(snapshot["summary"]["status"], "invalid")

    def test_post_run_integrity_failure_persists_invalid_artifact(self) -> None:
        template = self.build_snapshot()
        benchmark = synthesized.Benchmark(
            cases=[case["input"] for case in template["cases"]],
            input_paths=[],
            exclusions=[],
        )

        def ask(**_: object) -> dict:
            return {
                "evidence": [
                    {
                        "id": "ku",
                        "kind": "expert",
                        "source_ids": ["source"],
                        "in_requested_scope": True,
                    }
                ]
            }

        target = synthesized.TargetRuntime(
            ask=ask,
            pipeline_settings={"dataset_root": "/pipeline"},
            pipeline_dataset_root=Path("/pipeline"),
            module_paths={"soprano_qa.service": "/system/service.py"},
            pipeline_corpus_state={"input_files": []},
        )

        @contextmanager
        def failing_target_runtime(*_: object, **__: object):
            yield target
            raise synthesized.IntegrityValidationError("temporary corpus mutated")

        system_state = {
            "system_root": "/system",
            "fingerprint": "system-fingerprint",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            with (
                patch.object(synthesized, "load_benchmark", return_value=benchmark),
                patch.object(
                    synthesized,
                    "collect_system_state",
                    return_value=system_state,
                ),
                patch.object(
                    synthesized,
                    "target_runtime",
                    failing_target_runtime,
                ),
            ):
                with self.assertRaisesRegex(
                    synthesized.IntegrityValidationError,
                    "temporary corpus mutated",
                ):
                    synthesized.run_evaluation(
                        system_root=Path("/system"),
                        dataset_root=Path("/benchmark"),
                        output=output,
                        top_k=6,
                        required_mode="lexical",
                    )
            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(artifact["summary"]["status"], "invalid")
        self.assertEqual(
            artifact["run"]["integrity"]["status"],
            "invalid",
        )
        self.assertEqual(
            {case["status"] for case in artifact["cases"]},
            {"completed"},
        )

    def test_resume_refuses_a_different_system_fingerprint(self) -> None:
        existing = self.build_snapshot()
        changed = deepcopy(existing)
        changed["run"]["system_fingerprint"] = "different"
        with self.assertRaisesRegex(
            synthesized.ResumeMismatchError,
            "system_fingerprint",
        ):
            synthesized.validate_resume_snapshot(existing, changed)

    def test_successful_retry_clears_stale_retrieval_mode_error(self) -> None:
        snapshot = self.build_snapshot()
        snapshot["run"]["mode_error"] = {
            "type": "RetrievalModeMismatchError",
            "message": "dense runtime was unavailable",
        }

        def ask(**_: object) -> dict:
            return {
                "evidence": [
                    {
                        "id": "ku",
                        "kind": "expert",
                        "source_ids": ["source"],
                        "in_requested_scope": True,
                    }
                ],
                "retrieval": {
                    "configured_mode": "hybrid",
                    "active_mode": "hybrid",
                    "dense_available": True,
                    "fallback_reason": None,
                    "last_dense_error": None,
                    "last_search_mode": "hybrid",
                },
            }

        synthesized.run_cases(
            snapshot,
            ask=ask,
            top_k=6,
            required_mode="hybrid",
            checkpoint=lambda: None,
        )

        self.assertIsNone(snapshot["run"]["mode_error"])
        self.assertEqual(
            {case["status"] for case in snapshot["cases"]},
            {"completed"},
        )


class TargetIsolationTests(unittest.TestCase):
    def test_imports_only_target_package_and_does_not_write_bytecode(self) -> None:
        import soprano_qa.retrieval as original_retrieval

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "soprano_qa"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            corpus = root / "data" / "corpus.json"
            stats = root / "data" / "corpus_stats.json"
            dataset = root / "dataset"
            corpus_input = dataset / "input.json"
            write_json(corpus_input, {"source": "authenticated"})
            write_json(corpus, [])
            fingerprint = hashlib.sha256()
            fingerprint.update(b"input.json\0")
            fingerprint.update(corpus_input.read_bytes())
            fingerprint.update(b"\0")
            stats_payload = {
                "corpus_schema_version": 6,
                "web_export_files": ["research-open.jsonl"],
                "dataset_root": str(dataset.resolve()),
                "input_fingerprint": fingerprint.hexdigest(),
                "corpus_sha256": synthesized.file_sha256(corpus),
            }
            write_json(stats, stats_payload)
            corpus_source = '''\
import hashlib
from pathlib import Path

def corpus_input_paths(settings):
    path = Path(settings["dataset_root"]) / "input.json"
    return [("input.json", str(path))]

def corpus_input_fingerprint(settings):
    digest = hashlib.sha256()
    for label, path in corpus_input_paths(settings):
        digest.update(label.encode("utf-8"))
        digest.update(b"\\0")
        digest.update(Path(path).read_bytes())
        digest.update(b"\\0")
    return digest.hexdigest()
'''
            (package / "corpus.py").write_text(corpus_source, encoding="utf-8")
            service_source = f'''\
SETTINGS = {{
    "dataset_root": {str(dataset)!r},
    "corpus_path": {str(corpus)!r},
    "stats_path": {str(stats)!r},
}}
def ask(**kwargs):
    return {{"evidence": [], "kwargs": kwargs}}
'''
            (package / "service.py").write_text(service_source, encoding="utf-8")

            with synthesized.target_runtime(root) as target:
                response = target.ask(generate=False)
                self.assertFalse(response["kwargs"]["generate"])
                self.assertTrue(
                    Path(target.module_paths["soprano_qa.service"])
                    .resolve()
                    .is_relative_to(root)
                )
                self.assertEqual(
                    target.pipeline_corpus_state["input_fingerprint"],
                    fingerprint.hexdigest(),
                )

            self.assertFalse((package / "__pycache__").exists())
            self.assertEqual(json.loads(corpus.read_text(encoding="utf-8")), [])
            self.assertEqual(
                json.loads(stats.read_text(encoding="utf-8")),
                stats_payload,
            )
            self.assertIs(
                sys.modules["soprano_qa.retrieval"],
                original_retrieval,
            )

    def test_rejects_a_stale_prebuilt_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            corpus_input = dataset / "input.json"
            corpus = root / "corpus.json"
            stats = root / "stats.json"
            write_json(corpus_input, {"source": "current"})
            write_json(corpus, [])
            write_json(
                stats,
                {
                    "corpus_schema_version": 6,
                    "web_export_files": ["research-open.jsonl"],
                    "dataset_root": str(dataset.resolve()),
                    "input_fingerprint": "stale",
                    "corpus_sha256": synthesized.file_sha256(corpus),
                },
            )

            class CorpusModule:
                @staticmethod
                def corpus_input_paths(settings: dict) -> list[tuple[str, str]]:
                    return [("input.json", str(corpus_input))]

                @staticmethod
                def corpus_input_fingerprint(settings: dict) -> str:
                    return "current"

            with self.assertRaisesRegex(
                synthesized.EvaluationInputError,
                "stale or unauthenticated.*input_fingerprint",
            ):
                synthesized.authenticate_pipeline_corpus(
                    {
                        "dataset_root": str(dataset),
                        "corpus_path": str(corpus),
                        "stats_path": str(stats),
                    },
                    CorpusModule(),
                )


if __name__ == "__main__":
    unittest.main()
