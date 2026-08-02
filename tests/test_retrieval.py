"""Focused tests for measure-aware retrieval routing."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from soprano_qa.retrieval import (
    BM25Index,
    answer_relation_query_concepts,
    extract_question_measure_ranges,
    is_broad_performance_guidance_query,
    parse_measure_ranges,
    query_components,
    ranges_cover,
    scope_evidence_role,
    semantic_concepts,
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
        "measure_status": "specific" if measure_range else "whole_piece",
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

    def test_terminal_answer_relations_are_classified_as_soft_intent(self) -> None:
        base_concepts = [
            "피아노",
            "반주",
            "번째",
            "박자",
            "악센트",
        ]
        questions = (
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 의미하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 표현하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 묘사하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 상징하는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 가리키는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 보여주는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 드러내는가?",
            "피아노 반주에서 두 번째 박의 악센트는 무슨 뜻인가?",
            (
                "피아노 반주에서 두 번째 박의 악센트는 무엇을 "
                "의미하는지 알려 주세요."
            ),
            (
                "피아노 반주에서 두 번째 박의 악센트가 어떤 뜻인지 "
                "궁금합니다."
            ),
        )
        for question in questions:
            with self.subTest(question=question):
                concepts = semantic_concepts(
                    question,
                    "die-forelle",
                    query=True,
                )
                self.assertEqual(concepts[:-1], base_concepts)
                self.assertEqual(
                    answer_relation_query_concepts(question, concepts),
                    {concepts[-1]},
                )

    def test_answer_predicate_rule_preserves_content_uses(self) -> None:
        noun_concepts = semantic_concepts(
            "가사의 의미",
            None,
            query=True,
        )
        self.assertIn("의미", noun_concepts)
        self.assertEqual(
            answer_relation_query_concepts("가사의 의미", noun_concepts),
            set(),
        )
        performance_question = "분위기를 어떻게 표현해야 하나요?"
        performance_concepts = semantic_concepts(
            performance_question,
            None,
            query=True,
        )
        self.assertIn("표현", performance_concepts)
        self.assertEqual(
            answer_relation_query_concepts(
                performance_question,
                performance_concepts,
            ),
            set(),
        )
        concepts = semantic_concepts(
            "악센트는 국제관계를 의미하는가?",
            None,
            query=True,
        )
        self.assertIn("국제관계", concepts)
        self.assertIn("의미하", concepts)
        self.assertEqual(
            answer_relation_query_concepts(
                "악센트는 국제관계를 의미하는가?",
                concepts,
            ),
            set(),
        )
        multi_relation_question = (
            "표현주의 음악은 무엇을 의미하는가?"
        )
        multi_relation_concepts = semantic_concepts(
            multi_relation_question,
            None,
            query=True,
        )
        self.assertIn("표현주", multi_relation_concepts)
        self.assertEqual(
            answer_relation_query_concepts(
                multi_relation_question,
                multi_relation_concepts,
            ),
            {multi_relation_concepts[-1]},
        )
        self.assertNotIn(
            "표현주",
            answer_relation_query_concepts(
                multi_relation_question,
                multi_relation_concepts,
            ),
        )
        unsupported = (
            "피아노 반주 악센트는 무엇을 의미없는질문이라고 하는가?"
        )
        unsupported_concepts = semantic_concepts(
            unsupported,
            None,
            query=True,
        )
        self.assertEqual(
            answer_relation_query_concepts(
                unsupported,
                unsupported_concepts,
            ),
            set(),
        )
        natural_query, _ = query_components(
            "피아노 파트에서 두 번째 박의 악센트는 무엇을 표현할까?",
            "die-forelle",
        )
        self.assertEqual(
            [group[0] for group in token_groups(natural_query)],
            ["피아노", "반주", "번째", "박자", "악센트", "표현"],
        )

    def test_annotator_paraphrases_use_bidirectional_concept_families(self) -> None:
        equivalent_pairs = (
            ("페르마타 표기", "페르마타 유무"),
            ("음의 길이가 다르면", "음이 길게 들리는 차이"),
            ("분위기는 달라질까", "분위기가 변화한다"),
            ("이전 박자로 돌아온다", "처음 박자로 돌아온다"),
            ("기교가 들어갈 때", "기교가 들어가면"),
            ("반주가 없는 곳에서 박자를 잡아야", "반주가 없는 대목에서 박자를 잡아가나요"),
        )
        for paraphrase, source_wording in equivalent_pairs:
            with self.subTest(paraphrase=paraphrase):
                paraphrase_concepts = set(
                    semantic_concepts(paraphrase, None, query=True)
                )
                source_concepts = set(
                    semantic_concepts(source_wording, None, query=False)
                )
                self.assertTrue(paraphrase_concepts)
                self.assertTrue(paraphrase_concepts.issubset(source_concepts))

    def test_question_scaffolding_is_rechecked_after_stemming(self) -> None:
        self.assertEqual(
            semantic_concepts(
                "박자는 어떻게 처리할까?",
                "una-voce-poco-fa",
                query=True,
            ),
            ["박자"],
        )

    def test_completed_five_piece_annotations_share_natural_concepts(self) -> None:
        equivalent_pairs = (
            ("성별이나 성부의 제한", "성별 구분 없이 성부를 지정"),
            ("셋잇단음표에서 모음을 붙일까", "3연음보에서 모음의 위치"),
            ("겹부점 리듬을 사용한 이유", "겹점 리듬을 사용했을까"),
            ("악보의 표기와 영상에서 들리는 방식", "악보와 영상의 소리"),
            ("단어를 나누는 방식", "단어 구분"),
        )
        for paraphrase, source_wording in equivalent_pairs:
            with self.subTest(paraphrase=paraphrase):
                paraphrase_concepts = set(
                    semantic_concepts(paraphrase, None, query=True)
                )
                source_concepts = set(
                    semantic_concepts(source_wording, None, query=False)
                )
                self.assertTrue(paraphrase_concepts)
                self.assertTrue(paraphrase_concepts.issubset(source_concepts))

    def test_named_dotted_figure_does_not_require_redundant_rhythm_word(self) -> None:
        self.assertEqual(
            semantic_concepts(
                "일반적인 부점 리듬 대신 겹부점 리듬을 사용한 이유",
                None,
                query=True,
            ),
            ["일반적인", "부점", "겹점", "사용"],
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

    def test_dynamic_p_is_a_musical_concept_not_discarded_noise(self) -> None:
        exact = make_record(
            "exact-dynamic",
            [],
            text="멜로디 반주 악보 표기 p로 시작한다",
        )
        distractor = make_record(
            "other-marking",
            [],
            text="멜로디 반주 악보 표기를 해석한다",
        )
        index = BM25Index([exact, distractor])
        self.assertEqual(
            result_ids(
                index,
                query="멜로디 파트에 p가 표시된 이유는 무엇일까?",
                piece="test-piece",
            ),
            ["exact-dynamic"],
        )

    def test_ranged_search_retains_other_local_as_secondary_context(self) -> None:
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

        self.assertEqual(ids, ["overlap", "global", "other-local"])
        results = BM25Index(records).search(
            query="breath phrasing",
            piece="test-piece",
            measure_ranges=[[11, 11]],
            top_k=10,
        )
        self.assertEqual(
            [result.scope_match for result in results],
            [
                "overlaps_query_range",
                "global_context",
                "other_range_context",
            ],
        )
        self.assertEqual(results[-1].measure_score, -1.0)

    def test_ranged_search_still_excludes_unconfirmed_locations(self) -> None:
        pending = make_record("pending", [])
        pending["measure_status"] = "waiting_for_review"
        unspecified = make_record("unspecified", [])
        unspecified["measure_status"] = "unspecified"
        confirmed_other = make_record("confirmed-other", [[30, 32]])

        ids = result_ids(
            BM25Index([pending, unspecified, confirmed_other]),
            query="breath phrasing",
            piece="test-piece",
            measure_ranges=[[11, 11]],
            top_k=10,
        )

        self.assertEqual(ids, ["confirmed-other"])

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

    def test_broad_singer_question_keeps_confirmed_local_examples(self) -> None:
        whole = make_record(
            "whole-guidance",
            [],
            text=(
                "곡 전체에서는 명료한 발음과 자연스러운 호흡을 "
                "유지하며 노래하는 것이 중요하다."
            ),
        )
        local = make_record(
            "local-guidance",
            [[63, 63], [82, 83]],
            text=(
                "고음이 강박에 놓이지 않은 경우에는 단어의 강세와 "
                "음악적 강세에 유의해 고음을 유연하게 노래해야 한다."
            ),
        )
        pending = make_record(
            "pending-guidance",
            [],
            text="프레이즈와 호흡에 유의하며 연습하는 것이 중요하다.",
        )
        pending["measure_status"] = "waiting_for_review"
        distractor = make_record(
            "web-biography",
            [],
            text="작곡가의 출생과 생애를 설명한다.",
            evidence_type="web_database",
        )
        index = BM25Index([whole, local, pending, distractor])
        question = (
            "이 노래를 부를 때 가창자의 입장에서 유의해야 할 "
            "점은 무엇인가?"
        )

        self.assertTrue(
            is_broad_performance_guidance_query(
                question,
                "test-piece",
            )
        )
        for natural_variant in (
            "이 곡은 어떻게 불러야 할까?",
            "이 곡을 부를 때 가장 유의할 점은?",
            "이 곡을 노래할 때 중요한 가창 포인트는?",
            "이 곡에서 가창자가 신경 써야 할 사항은?",
            "이 곡을 잘 부르기 위한 팁은?",
            "이 노래를 잘 노래하려면 무엇을 해야 할까요?",
            "이 작품을 제대로 소화할 때 신경 쓸 점은?",
            "어떻게 해야 이 노래를 잘 부를 수 있나요?",
            "이 곡을 부르는 법을 알려줘",
            "이 작품의 전반적인 가창 조언을 줘",
            "How should I sing this piece?",
            "What should a singer focus on in this work?",
            "Give me overall singing advice for this piece.",
        ):
            with self.subTest(natural_variant=natural_variant):
                self.assertTrue(
                    is_broad_performance_guidance_query(
                        natural_variant,
                        "test-piece",
                    )
                )
        results = index.search(
            question,
            piece="test-piece",
            top_k=6,
        )

        self.assertEqual(
            {result.record["id"] for result in results},
            {"whole-guidance", "local-guidance"},
        )
        local_result = next(
            result
            for result in results
            if result.record["id"] == "local-guidance"
        )
        self.assertEqual(local_result.scope_match, "local_example")
        self.assertEqual(
            scope_evidence_role(local_result.scope_match),
            "local_example",
        )
        self.assertEqual(local_result.alias_score, 0.0)
        self.assertFalse(
            is_broad_performance_guidance_query(
                "고음을 부를 때 어떤 점에 유의해야 할까?",
                "test-piece",
            )
        )
        for homonym in (
            "이 작품에서 성악가가 제대하려면 어떻게 해야 할까?",
            "이 작품에서 소화를 잘하려면 무엇을 해야 할까?",
        ):
            with self.subTest(homonym=homonym):
                self.assertFalse(
                    is_broad_performance_guidance_query(
                        homonym,
                        "test-piece",
                    )
                )

    def test_no_range_prefers_confirmed_local_over_pending_scope(self) -> None:
        local = make_record("local", [[10, 12]])
        pending = make_record("pending", [])
        pending["measure_status"] = "waiting_for_review"

        results = BM25Index([pending, local]).search(
            query="breath phrasing",
            piece="test-piece",
            top_k=10,
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["local", "pending"],
        )

    def test_broad_singer_question_can_request_a_musical_facet(
        self,
    ) -> None:
        breathing = make_record(
            "breathing",
            [[10, 12]],
            text=(
                "호흡과 울림을 유지하도록 주의하고 충분히 "
                "연습하는 것이 중요하다."
            ),
        )
        expression = make_record(
            "expression",
            [[20, 22]],
            text=(
                "반주의 분위기와 음악적 해석을 생각하며 장면을 "
                "섬세하게 표현하는 것이 중요하다."
            ),
        )
        diction = make_record(
            "diction",
            [[30, 31]],
            text=(
                "자음과 모음을 명료하게 발음하도록 주의하는 것이 "
                "중요하다."
            ),
        )
        index = BM25Index([breathing, expression, diction])

        expectations = {
            "이 곡을 부를 때 호흡에 유의해야 할 점은?": {
                "breathing"
            },
            "이 곡을 부를 때 표현에 유의해야 할 점은?": {
                "expression"
            },
            "이 곡을 부를 때 음악적으로 유의해야 할 점은?": {
                "expression"
            },
        }
        for question, expected_ids in expectations.items():
            with self.subTest(question=question):
                self.assertTrue(
                    is_broad_performance_guidance_query(
                        question,
                        "test-piece",
                    )
                )
                results = index.search(
                    question,
                    piece="test-piece",
                    top_k=10,
                )
                self.assertEqual(
                    {result.record["id"] for result in results},
                    expected_ids,
                )
                self.assertTrue(
                    all(
                        result.scope_match == "local_example"
                        for result in results
                    )
                )

        self.assertFalse(
            is_broad_performance_guidance_query(
                "고음을 부를 때 어떤 점에 유의해야 할까?",
                "test-piece",
            )
        )

    def test_broad_guidance_keeps_complementary_same_source_units(self) -> None:
        general = make_record(
            "general",
            [],
            text="자음과 모음의 악센트에 유의해 노래하는 것이 중요하다.",
        )
        tone = make_record(
            "tone",
            [],
            text="탄력 있고 밝고 깔끔한 음색으로 표현하는 것이 중요하다.",
        )
        diction = make_record(
            "diction-sibling",
            [],
            text="딕션의 장단을 정확히 공부해 단어를 표현해야 한다.",
        )
        tone["source_ids"] = ["expert-source"]
        diction["source_ids"] = ["expert-source"]

        ids = result_ids(
            BM25Index([general, tone, diction]),
            query="이 곡을 잘 부르려면 무엇에 유의해야 할까?",
            piece="test-piece",
            top_k=3,
        )

        self.assertIn("tone", ids)
        self.assertIn("diction-sibling", ids)

    def test_broad_local_only_results_are_capped_at_three_examples(
        self,
    ) -> None:
        records = [
            make_record(
                f"local-{index}",
                [[index, index]],
                text=(
                    "고음과 프레이즈를 연습할 때 강박에 유의하고 "
                    "명료한 발음을 유지하는 것이 중요하다."
                ),
            )
            for index in range(1, 6)
        ]

        results = BM25Index(records).search(
            query="이 곡을 부르는 법을 알려줘",
            piece="test-piece",
            top_k=10,
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(
            all(result.scope_match == "local_example" for result in results)
        )

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

        self.assertEqual(ids, ["web-global", "unrelated-local"])

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

    def test_compound_queries_require_candidate_local_full_coverage(self) -> None:
        records = [
            make_record("pronunciation", [], text="pronunciation"),
            make_record("form", [], text="form"),
            make_record("both", [], text="pronunciation rhythm"),
        ]
        index = BM25Index(records)
        self.assertEqual(
            index.search(
                "pronunciation form",
                piece="test-piece",
                top_k=2,
            ),
            [],
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

    def test_korean_answer_relation_uses_a_single_anchored_fallback(self) -> None:
        relevant = make_record(
            "relevant",
            [[2, 5]],
            text=(
                "피아노 반주 두 번째 박자 악센트는 송어의 움직임을 "
                "표현한다"
            ),
        )
        unrelated = make_record(
            "unrelated",
            [[2, 5]],
            text="피아노 반주 박자는 국제관계를 설명한다",
        )
        index = BM25Index([relevant, unrelated])

        results = index.search(
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 묘사하는가?",
            piece="test-piece",
            measure_ranges=[[2, 5]],
            top_k=6,
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["relevant"],
        )
        self.assertEqual(results[0].semantic_match_type, "answer_relation_fallback")
        self.assertAlmostEqual(results[0].concept_coverage, 5 / 6)
        self.assertEqual(results[0].alias_score, 0.0)

    def test_answer_relation_fallback_keeps_content_concepts_strict(self) -> None:
        relevant = make_record(
            "relevant",
            [[2, 5]],
            text="피아노 반주 두 번째 박자 악센트는 움직임을 표현한다",
        )
        index = BM25Index([relevant])

        self.assertEqual(
            index.search(
                (
                    "피아노 반주 두 번째 박자 악센트와 국제관계는 "
                    "무엇을 의미하는가?"
                ),
                piece="test-piece",
                measure_ranges=[[2, 5]],
                top_k=6,
            ),
            [],
        )

    def test_noun_meaning_does_not_enable_relation_fallback(self) -> None:
        index = BM25Index(
            [
                make_record(
                    "representation",
                    [],
                    text="가사가 봄의 장면을 나타낸다",
                )
            ]
        )

        self.assertEqual(
            index.search(
                "가사의 의미",
                piece="test-piece",
                top_k=6,
            ),
            [],
        )

    def test_alias_only_concepts_cannot_establish_relation_fallback(self) -> None:
        record = make_record(
            "alias-only",
            [[2, 5]],
            text="작곡가의 생애와 작품 연도를 소개한다",
        )
        alias = "피아노 반주 두 번째 박자 악센트는 움직임을 표현한다"
        record["retrieval_aliases"] = [alias]
        record["relevance_text"] = f"{record['answer']}\n{alias}"

        results = BM25Index([record]).search(
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 묘사하는가?",
            piece="test-piece",
            measure_ranges=[[2, 5]],
            top_k=6,
        )

        self.assertEqual(results, [])

    def test_relation_fallback_requires_explanation_in_answer_content(self) -> None:
        records = [
            make_record(
                "global",
                [],
                text="피아노 반주 두 번째 박자 악센트",
            ),
            make_record(
                "other-range",
                [[30, 32]],
                text="피아노 반주 두 번째 박자 악센트",
            ),
        ]
        index = BM25Index(records)
        question = (
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 묘사하는가?"
        )

        self.assertEqual(
            index.search(question, piece="test-piece", top_k=6),
            [],
        )
        self.assertEqual(
            index.search(
                question,
                piece="test-piece",
                measure_ranges=[[2, 5]],
                top_k=6,
            ),
            [],
        )

    def test_relation_nouns_do_not_count_as_answer_relations(self) -> None:
        cases = (
            ("표현주의 음악", "표현주의 음악은 무엇을 묘사하는가?"),
            (
                "음악적 표현 기호",
                "음악적 표현 기호는 무엇을 묘사하는가?",
            ),
            ("반주에 대한 설명", "반주는 무엇을 묘사하는가?"),
            ("악센트의 의미", "악센트는 무엇을 묘사하는가?"),
        )
        for answer, question in cases:
            with self.subTest(question=question):
                index = BM25Index(
                    [make_record("noun-only", [[2, 5]], text=answer)]
                )
                self.assertEqual(
                    index.search(
                        question,
                        piece="test-piece",
                        measure_ranges=[[2, 5]],
                        top_k=6,
                    ),
                    [],
                )

    def test_relation_fallback_prefers_answer_side_content_coverage(self) -> None:
        strong = make_record(
            "strong",
            [[2, 5]],
            text="피아노 반주 악센트는 움직임을 표현한다",
        )
        weak = make_record(
            "weak",
            [[2, 5]],
            text="피아노 반주는 작곡가의 생각을 표현한다",
        )
        repeated_alias = " ".join(
            ["피아노 반주 악센트"] * 8
        )
        weak["retrieval_aliases"] = ["피아노 반주 악센트"]
        weak["relevance_text"] = f"{weak['answer']}\n{repeated_alias}"
        weak["retrieval_text"] = weak["relevance_text"]

        results = BM25Index([weak, strong]).search(
            "피아노 반주 악센트는 무엇을 묘사하는가?",
            piece="test-piece",
            measure_ranges=[[2, 5]],
            top_k=6,
        )

        self.assertEqual(
            [result.record["id"] for result in results],
            ["strong"],
        )
        self.assertEqual(results[0].content_concept_coverage, 1.0)

    def test_strict_same_scope_match_suppresses_relation_fallback(self) -> None:
        strict = make_record(
            "strict",
            [[2, 5]],
            text="피아노 반주 두 번째 박자 악센트는 움직임을 묘사한다",
        )
        fallback = make_record(
            "fallback",
            [[2, 5]],
            text=(
                "피아노 피아노 반주 반주 두 번째 박자 악센트는 "
                "움직임을 표현한다"
            ),
        )

        results = BM25Index([fallback, strict]).search(
            "피아노 반주에서 두 번째 박의 악센트는 무엇을 묘사하는가?",
            piece="test-piece",
            measure_ranges=[[2, 5]],
            top_k=6,
        )

        self.assertEqual([result.record["id"] for result in results], ["strict"])
        self.assertEqual(results[0].semantic_match_type, "strict")


if __name__ == "__main__":
    unittest.main()
