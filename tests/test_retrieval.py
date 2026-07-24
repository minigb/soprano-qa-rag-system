"""Focused tests for measure-aware retrieval routing."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from soprano_qa.retrieval import (
    BM25Index,
    extract_question_measure_ranges,
    parse_measure_ranges,
    query_components,
    ranges_cover,
    token_groups,
    tokenize,
    validate_question_measure_contract,
)


def make_record(
    record_id: str,
    measure_range: List[List[int]],
    *,
    text: str = "breath phrasing",
    evidence_type: str = "expert_annotation",
    retrieval_eligible: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return the smallest realistic corpus record needed by BM25Index."""
    record: Dict[str, Any] = {
        "id": record_id,
        "piece": "test-piece",
        "topic": "performance",
        "answer": text,
        "retrieval_text": text,
        "measure_range": measure_range,
        "measure_scope": "local" if measure_range else "global",
        "evidence_type": evidence_type,
    }
    if retrieval_eligible is not None:
        record["retrieval_eligible"] = retrieval_eligible
    return record


def result_ids(index: BM25Index, **search_args: Any) -> List[str]:
    return [result.record["id"] for result in index.search(**search_args)]


class ParseMeasureRangesTests(unittest.TestCase):
    def test_parses_positive_inclusive_ranges_with_hyphen_and_en_dash(self) -> None:
        self.assertEqual(
            parse_measure_ranges("1, 3 - 5, 7–9"),
            [[1, 1], [3, 5], [7, 9]],
        )

    def test_empty_input_means_no_measure_range(self) -> None:
        self.assertEqual(parse_measure_ranges(""), [])

    def test_rejects_non_positive_reversed_and_malformed_ranges(self) -> None:
        invalid_inputs = (
            "0",
            "-1",
            "5-3",
            "1-",
            "1-2-3",
            "1,,2",
            ",1",
            "1,",
            "measure 3",
        )
        for value in invalid_inputs:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_measure_ranges(value)

    def test_extracts_measure_mentions_from_natural_language(self) -> None:
        self.assertEqual(
            extract_question_measure_ranges(
                "28마디부터 30까지와 40부터 42마디까지, 51번째 마디를 비교해 주세요"
            ),
            [[28, 30], [40, 42], [51, 51]],
        )
        self.assertEqual(
            extract_question_measure_ranges("제28마디와 30마디의 분위기"),
            [[28, 28], [30, 30]],
        )
        self.assertEqual(
            extract_question_measure_ranges("이 곡은 4마디 프레이즈로 구성되나요?"),
            [],
        )
        self.assertEqual(
            extract_question_measure_ranges("4마디로 구성된 구절과 8마디 악절"),
            [],
        )
        self.assertEqual(
            extract_question_measure_ranges("40마디 분위기와 42마디 근처를 비교해 주세요"),
            [[40, 40], [42, 42]],
        )

    def test_extracts_abbreviated_english_ordinal_and_list_measure_forms(self) -> None:
        cases = {
            "28마디부터 30까지": [[28, 30]],
            "28부터 30마디까지": [[28, 30]],
            "bars 40-42": [[40, 42]],
            "measure 40 to 42": [[40, 42]],
            "28-30 bars": [[28, 30]],
            "28 bars to 30 bars": [[28, 30]],
            "28마디~30마디": [[28, 30]],
            "m.40": [[40, 40]],
            "40번 마디": [[40, 40]],
            "첫 번째 마디": [[1, 1]],
            "first bar": [[1, 1]],
            "첫 4마디": [[1, 4]],
            "opening 4 bars": [[1, 4]],
            "28, 30마디": [[28, 28], [30, 30]],
            "28과 30마디": [[28, 28], [30, 30]],
            "28번과 30번 마디": [[28, 28], [30, 30]],
            "measures 28 and 30": [[28, 28], [30, 30]],
            "bars 28, 30": [[28, 28], [30, 30]],
            "28, 30-32마디": [[28, 28], [30, 32]],
            "28-30, 40마디": [[28, 30], [40, 40]],
            "measures 28, 30-32": [[28, 28], [30, 32]],
            "bars 28-30, 40": [[28, 30], [40, 40]],
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(extract_question_measure_ranges(question), expected)

    def test_total_counts_do_not_become_locations_or_reversed_ranges(self) -> None:
        for question in (
            "이 곡은 총 67마디인가요?",
            "이 곡은 전체 67마디인가요?",
            "이 곡의 총 마디 수는 67마디인가요?",
            "이 곡은 67마디인가요?",
            "이 노래가 67마디예요?",
            "곡 전체 길이는 67마디인가요?",
            "이 곡은 모두 67마디인가요?",
            "이 곡은 67마디에요?",
            "이 작품의 길이는 67마디에요?",
            "이 곡은 67마디죠?",
        ):
            with self.subTest(question=question):
                self.assertEqual(extract_question_measure_ranges(question), [])
        self.assertEqual(
            extract_question_measure_ranges("총 40마디에서 28마디의 분위기는?"),
            [[28, 28]],
        )
        with self.assertRaisesRegex(ValueError, "Invalid measure range"):
            extract_question_measure_ranges("40마디에서 28마디의 분위기는?")

    def test_unresolved_end_locations_require_but_accept_an_explicit_scope(self) -> None:
        for question in ("마지막 마디의 호흡", "끝 4마디", "last bar phrasing"):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "explicit numeric"):
                    validate_question_measure_contract(question, [])
                self.assertEqual(
                    validate_question_measure_contract(question, [[40, 42]]),
                    [],
                )

    def test_measure_contract_checks_opening_counts_lists_and_english_forms(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside --measures"):
            validate_question_measure_contract("첫 4마디의 분위기", [[4, 4]])
        self.assertEqual(
            validate_question_measure_contract("첫 4마디의 분위기", [[1, 4]]),
            [[1, 4]],
        )
        with self.assertRaisesRegex(ValueError, "outside --measures"):
            validate_question_measure_contract("28, 30마디", [[30, 30]])
        with self.assertRaisesRegex(ValueError, "outside --measures"):
            validate_question_measure_contract("What changes at bar 40?", [[28, 28]])

    def test_query_particle_normalization_removes_the_whole_suffix(self) -> None:
        normalized = {}
        for word in ("형식으로", "발음으로", "호흡에서는", "작품으로는", "가사까지는"):
            query, _ = query_components(word, None)
            normalized[word] = [group[0] for group in token_groups(query)]
        self.assertEqual(
            normalized,
            {
                "형식으로": ["형식"],
                "발음으로": ["발음"],
                "호흡에서는": ["호흡"],
                "작품으로는": ["작품"],
                "가사까지는": ["가사"],
            },
        )

    def test_question_measure_contract_accepts_covered_mentions(self) -> None:
        self.assertTrue(ranges_cover([[28, 29], [30, 32]], [[28, 30]]))
        self.assertEqual(
            validate_question_measure_contract(
                "28마디부터 분위기가 어떻게 바뀌나요?",
                [[28, 30]],
            ),
            [[28, 28]],
        )

    def test_question_measure_contract_rejects_missing_or_conflicting_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "pass a matching --measures"):
            validate_question_measure_contract("40마디의 분위기는 어떤가요?", [])
        with self.assertRaisesRegex(ValueError, "outside --measures"):
            validate_question_measure_contract(
                "40마디의 분위기는 어떤가요?",
                [[28, 28]],
            )
        with self.assertRaisesRegex(ValueError, "outside --measures"):
            validate_question_measure_contract(
                "28마디부터 30까지의 분위기는 어떤가요?",
                [[28, 28]],
            )


class MeasureAwareSearchTests(unittest.TestCase):
    def test_document_tokenization_preserves_term_frequency(self) -> None:
        self.assertEqual(tokenize("breath breath breath"), ["breath"] * 3)

    def test_ranged_search_keeps_overlap_and_global_but_excludes_other_local(self) -> None:
        records = [
            make_record("overlap", [[10, 12]]),
            make_record("global", []),
            make_record("other-local", [[30, 32]]),
        ]

        ids = result_ids(
            BM25Index(records),
            query="breath phrasing",
            piece="test-piece",
            measure_ranges=[[11, 11]],
            top_k=10,
        )

        self.assertEqual(set(ids), {"overlap", "global"})
        self.assertNotIn("other-local", ids)

    def test_no_range_ranks_global_above_equally_relevant_local(self) -> None:
        records = [
            make_record("local", [[10, 12]]),
            make_record("global", []),
        ]

        results = BM25Index(records).search(
            query="breath phrasing",
            piece="test-piece",
            measure_ranges=None,
            top_k=10,
        )

        self.assertEqual([result.record["id"] for result in results], ["global", "local"])
        self.assertGreater(results[0].measure_score, results[1].measure_score)

    def test_ineligible_records_are_skipped_and_missing_flag_defaults_true(self) -> None:
        records = [
            make_record("default-eligible", []),
            make_record("ineligible", [], retrieval_eligible=False),
        ]

        ids = result_ids(
            BM25Index(records),
            query="breath phrasing",
            piece="test-piece",
            top_k=10,
        )

        self.assertEqual(ids, ["default-eligible"])

    def test_zero_text_relevance_does_not_return_arbitrary_global_context(self) -> None:
        index = BM25Index([make_record("global", [], text="breath phrasing")])

        results = index.search(
            query="ornament trill",
            piece="test-piece",
            measure_ranges=None,
            top_k=10,
        )

        self.assertEqual(results, [])

    def test_measure_overlap_does_not_replace_textual_relevance(self) -> None:
        index = BM25Index([make_record("overlap", [[10, 12]], text="breath phrasing")])

        results = index.search(
            query="FIFA World Cup winner",
            piece="test-piece",
            measure_ranges=[[11, 11]],
            top_k=10,
        )

        self.assertEqual(results, [])

    def test_web_database_evidence_is_global_context_for_ranged_questions(self) -> None:
        records = [
            make_record(
                "web-global",
                [],
                text="composer biography",
                evidence_type="web_database",
            ),
            make_record("unrelated-local", [[40, 41]], text="composer biography"),
        ]

        ids = result_ids(
            BM25Index(records),
            query="composer biography",
            piece="test-piece",
            measure_ranges=[[10, 12]],
            top_k=10,
        )

        self.assertEqual(ids, ["web-global"])

    def test_provenance_metadata_cannot_establish_semantic_relevance(self) -> None:
        record = make_record("semantic", [], text="genre and form")
        record["retrieval_text"] += "\nBibliothèque nationale de France"
        record["relevance_text"] = "genre and form"
        index = BM25Index([record])
        self.assertEqual(index.search("France", piece="test-piece"), [])
        self.assertEqual(
            result_ids(index, query="genre", piece="test-piece"),
            ["semantic"],
        )

    def test_compound_queries_use_minimal_full_coverage(self) -> None:
        records = [
            make_record("pronunciation", [], text="pronunciation"),
            make_record("form", [], text="form"),
            make_record("both", [], text="pronunciation rhythm"),
        ]
        index = BM25Index(records)
        self.assertEqual(
            set(
                result_ids(
                    index,
                    query="pronunciation form",
                    piece="test-piece",
                    top_k=2,
                )
            ),
            {"pronunciation", "form"},
        )
        self.assertEqual(
            result_ids(
                index,
                query="pronunciation rhythm",
                piece="test-piece",
                top_k=1,
            ),
            ["both"],
        )
        self.assertEqual(
            index.search("pronunciation unsupported", piece="test-piece", top_k=2),
            [],
        )


if __name__ == "__main__":
    unittest.main()
