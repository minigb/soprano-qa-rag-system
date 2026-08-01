"""Five-piece retrieval coverage for the annotator-question inventory.

This is intentionally a retrieval-only regression.  The paraphrased benchmark
questions stay outside the corpus: the test asks the normal BM25 index and
checks whether its top-six results contain knowledge units that are linked to
the authoritative annotator source.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest
from typing import Any

from soprano_qa.corpus import corpus_input_fingerprint
from soprano_qa.retrieval import BM25Index, load_corpus, ranges_overlap
from soprano_qa.settings import load_settings


PIECES = (
    "die-forelle",
    "in-flowery-clouds",
    "la-capinera",
    "nella-fantasia",
    "una-voce-poco-fa",
)
TOP_K = 6
MINIMUM_QUESTION_COVERAGE = 0.90

# This pairing is deliberately omitted by the semantic evaluator as well.  A
# reviewed unit transfers an opening pp idea to the reprise, but its lyric
# anchor does not match measures 78-81.  The annotation and range remain in
# the dataset; only this question/range benchmark pairing is invalid.
EXCLUDED_CASES = {("kim-la-capinera-01", (78, 81))}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _range_applicable_expected_ids(
    expected_ids: set[str],
    records_by_id: dict[str, dict[str, Any]],
    measure_range: list[int] | None,
) -> set[str]:
    if measure_range is None:
        return set(expected_ids)

    applicable = set()
    for record_id in expected_ids:
        record = records_by_id[record_id]
        if record.get("measure_status") == "whole_piece":
            applicable.add(record_id)
        elif (
            record.get("measure_status") == "specific"
            and ranges_overlap(
                record.get("measure_range") or [],
                [measure_range],
            )
        ):
            applicable.add(record_id)
    return applicable


def evaluate_five_piece_retrieval(
    *,
    index: BM25Index,
    dataset_root: Path,
    minimum_question_coverage: float = MINIMUM_QUESTION_COVERAGE,
) -> dict[str, Any]:
    """Return per-piece linked-KU retrieval metrics and concrete misses."""

    records_by_id = {
        record["id"]: record
        for record in index.records
        if record.get("evidence_type") == "expert_annotation"
    }
    inventory_root = (
        dataset_root / "expert_curation" / "evaluation_questions"
    )
    piece_reports: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = []

    for piece_id in PIECES:
        inventory = _load_json(inventory_root / f"{piece_id}.json")
        if inventory.get("piece_id") != piece_id:
            raise AssertionError(
                f"{piece_id}: evaluation inventory piece_id mismatch"
            )

        questions_total = 0
        questions_covered = 0
        questions_all_expected = 0
        cases_total = 0
        cases_covered = 0
        cases_all_expected = 0
        cases_top1 = 0
        expected_unit_occurrences = 0
        retrieved_unit_occurrences = 0
        piece_failures: list[dict[str, Any]] = []

        for question in inventory["questions"]:
            if question.get("review_status") == "excluded_unanswerable":
                continue

            source_id = question["source_id"]
            expected_ids = set(question["knowledge_unit_ids"])
            if not expected_ids:
                raise AssertionError(
                    f"{source_id}: answerable question has no linked units"
                )
            missing_records = expected_ids - records_by_id.keys()
            if missing_records:
                raise AssertionError(
                    f"{source_id}: linked units absent from corpus: "
                    f"{sorted(missing_records)}"
                )
            for expected_id in expected_ids:
                if source_id not in records_by_id[expected_id].get(
                    "source_ids", []
                ):
                    raise AssertionError(
                        f"{source_id}: corpus unit {expected_id} does not "
                        "link back to its annotator source"
                    )

            active_ranges: list[list[int] | None] = []
            raw_ranges = question.get("inference_measure_ranges") or []
            for measure_range in raw_ranges:
                if (source_id, tuple(measure_range)) not in EXCLUDED_CASES:
                    active_ranges.append(measure_range)
            if not raw_ranges:
                active_ranges.append(None)
            if not active_ranges:
                # Every answerable inventory question must retain at least one
                # valid inference case after documented exclusions.
                raise AssertionError(
                    f"{source_id}: all inference cases were excluded"
                )

            questions_total += 1
            question_case_hits: list[bool] = []
            question_all_expected_hits: list[bool] = []
            for measure_range in active_ranges:
                applicable_ids = _range_applicable_expected_ids(
                    expected_ids,
                    records_by_id,
                    measure_range,
                )
                if not applicable_ids:
                    raise AssertionError(
                        f"{source_id} {measure_range}: no linked unit is "
                        "applicable to this inference range"
                    )

                results = index.search(
                    question["paraphrased_question"],
                    piece=piece_id,
                    measure_ranges=(
                        [measure_range]
                        if measure_range is not None
                        else None
                    ),
                    top_k=TOP_K,
                )
                retrieved_ids = [
                    result.record["id"] for result in results
                ]
                retrieved_expected = applicable_ids & set(retrieved_ids)
                # Exact KU membership and the source backlink are both part
                # of the hit contract.  The backlink check above also fails
                # fast if corpus provenance has drifted.
                case_hit = bool(retrieved_expected)
                all_expected_hit = applicable_ids <= set(retrieved_ids)

                cases_total += 1
                cases_covered += int(case_hit)
                cases_all_expected += int(all_expected_hit)
                cases_top1 += int(
                    bool(retrieved_ids)
                    and retrieved_ids[0] in applicable_ids
                )
                expected_unit_occurrences += len(applicable_ids)
                retrieved_unit_occurrences += len(retrieved_expected)
                question_case_hits.append(case_hit)
                question_all_expected_hits.append(all_expected_hit)

                if not case_hit:
                    failure = {
                        "piece_id": piece_id,
                        "source_id": source_id,
                        "question": question["paraphrased_question"],
                        "measure_range": measure_range,
                        "expected_knowledge_unit_ids": sorted(
                            applicable_ids
                        ),
                        "retrieved_ids": retrieved_ids,
                        "retrieved_scope_matches": [
                            result.scope_match for result in results
                        ],
                    }
                    piece_failures.append(failure)
                    all_failures.append(failure)

            questions_covered += int(all(question_case_hits))
            questions_all_expected += int(
                all(question_all_expected_hits)
            )

        question_coverage = (
            questions_covered / questions_total if questions_total else 0.0
        )
        piece_reports[piece_id] = {
            "questions": questions_total,
            "questions_covered": questions_covered,
            "question_coverage": round(question_coverage, 6),
            "questions_with_all_expected_units": questions_all_expected,
            "cases": cases_total,
            "cases_covered": cases_covered,
            "case_coverage": round(
                cases_covered / cases_total if cases_total else 0.0,
                6,
            ),
            "cases_top1_expected": cases_top1,
            "cases_with_all_expected_units": cases_all_expected,
            "expected_unit_occurrences": expected_unit_occurrences,
            "retrieved_expected_unit_occurrences": (
                retrieved_unit_occurrences
            ),
            "expected_unit_recall": round(
                retrieved_unit_occurrences / expected_unit_occurrences
                if expected_unit_occurrences
                else 0.0,
                6,
            ),
            "failures": piece_failures,
        }

    totals = Counter()
    for piece_report in piece_reports.values():
        totals.update(
            {
                "questions": piece_report["questions"],
                "questions_covered": piece_report[
                    "questions_covered"
                ],
                "questions_with_all_expected_units": piece_report[
                    "questions_with_all_expected_units"
                ],
                "cases": piece_report["cases"],
                "cases_covered": piece_report["cases_covered"],
                "cases_top1_expected": piece_report[
                    "cases_top1_expected"
                ],
                "cases_with_all_expected_units": piece_report[
                    "cases_with_all_expected_units"
                ],
                "expected_unit_occurrences": piece_report[
                    "expected_unit_occurrences"
                ],
                "retrieved_expected_unit_occurrences": piece_report[
                    "retrieved_expected_unit_occurrences"
                ],
            }
        )
    overall_question_coverage = (
        totals["questions_covered"] / totals["questions"]
    )
    below_threshold = [
        piece_id
        for piece_id, piece_report in piece_reports.items()
        if piece_report["question_coverage"] < minimum_question_coverage
    ]
    return {
        "minimum_question_coverage": minimum_question_coverage,
        "pieces": piece_reports,
        "overall": {
            **dict(totals),
            "question_coverage": round(overall_question_coverage, 6),
            "case_coverage": round(
                totals["cases_covered"] / totals["cases"],
                6,
            ),
            "expected_unit_recall": round(
                totals["retrieved_expected_unit_occurrences"]
                / totals["expected_unit_occurrences"],
                6,
            ),
        },
        "pieces_below_threshold": below_threshold,
        "failures": all_failures,
    }


class FivePieceRetrievalCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings(use_legacy_dataset_env=False)
        stats = _load_json(Path(cls.settings["stats_path"]))
        current_fingerprint = corpus_input_fingerprint(cls.settings)
        if stats.get("input_fingerprint") != current_fingerprint:
            raise AssertionError(
                "The checked corpus is stale for the active dataset; run "
                "python3 scripts/build_corpus.py before this coverage test"
            )
        cls.index = BM25Index(load_corpus(cls.settings["corpus_path"]))

    def test_all_five_pieces_retrieve_linked_annotator_knowledge(self) -> None:
        report = evaluate_five_piece_retrieval(
            index=self.index,
            dataset_root=Path(self.settings["dataset_root"]),
        )
        failure_report = json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self.assertGreaterEqual(
            report["overall"]["question_coverage"],
            MINIMUM_QUESTION_COVERAGE,
            failure_report,
        )
        self.assertEqual(
            report["pieces_below_threshold"],
            [],
            failure_report,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
