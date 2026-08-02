"""Focused tests for grounded-answer evidence selection and review cautions."""

from __future__ import annotations

import unittest
from typing import Any, Dict

from soprano_qa.answer import (
    answer_overgeneralizes_local_examples,
    build_extractive_answer,
    build_messages,
    build_review_constraints,
    build_review_disclosure,
    build_review_disclosure_from_records,
    ensure_review_disclosure,
    expert_prompt_factuality_material,
    finalize_answer_citations,
    generate_with_context_retry,
    has_primary_grounding,
    select_extractive_fallback_evidence,
    select_generation_evidence,
    suspicious_generation_tokens,
)
from soprano_qa.retrieval import SearchResult


def make_result(
    record_id: str,
    *,
    evidence_type: str = "expert_annotation",
    alias_score: float = 0.0,
    rewrite_status: str = "ready",
    rewrite_notes: str = "",
    measure_status: str = "specific",
    measure_notes: str = "",
    review_warning: str = "",
    scope_match: str = "overlaps_query_range",
    source_ids: list[str] | None = None,
    score: float = 1.0,
    text_score: float = 1.0,
    semantic_match_type: str = "strict",
) -> SearchResult:
    record: Dict[str, Any] = {
        "id": record_id,
        "evidence_type": evidence_type,
        "piece": "die-forelle",
        "work": "Die Forelle",
        "topic": "performance",
        "measure_range": [[2, 5]],
        "measure_scope": "local",
        "source_ids": source_ids or ["kim-die-forelle-01"],
        "annotators": ["kim"],
        "rewrite_status": rewrite_status,
        "rewrite_notes": rewrite_notes,
        "measure_status": measure_status,
        "measure_notes": measure_notes,
        "retrieval_review_warning": review_warning,
        "source_answer_context": [],
        "question": "",
        "answer": "전문가 답변",
    }
    return SearchResult(
        record=record,
        score=score,
        text_score=text_score,
        measure_score=0.0,
        piece_score=1.0,
        scope_match=scope_match,
        alias_score=alias_score,
        semantic_match_type=semantic_match_type,
    )


class GenerationEvidenceSelectionTests(unittest.TestCase):
    def test_keeps_only_maximum_positive_expert_alias_matches_in_input_order(
        self,
    ) -> None:
        weaker_expert = make_result("expert-weaker", alias_score=0.5)
        strongest_first = make_result("expert-strongest-first", alias_score=1.0)
        web_with_same_score = make_result(
            "web-strongest",
            evidence_type="web_database",
            alias_score=1.0,
        )
        unrelated_expert = make_result("expert-unrelated", alias_score=0.0)
        strongest_second = make_result("expert-strongest-second", alias_score=1.0)
        results = [
            weaker_expert,
            strongest_first,
            web_with_same_score,
            unrelated_expert,
            strongest_second,
        ]

        selected = select_generation_evidence(results)

        self.assertEqual(selected, [strongest_first, strongest_second])

    def test_returns_all_results_when_no_positive_expert_alias_exists(
        self,
    ) -> None:
        results = [
            make_result("expert-no-alias", alias_score=0.0),
            make_result(
                "web-positive-alias",
                evidence_type="web_database",
                alias_score=2.0,
            ),
        ]

        self.assertEqual(select_generation_evidence(results), results)

    def test_other_range_alias_is_retained_without_displacing_overlap(
        self,
    ) -> None:
        overlap = make_result(
            "overlap",
            alias_score=0.5,
            scope_match="overlaps_query_range",
        )
        stronger_other_range = make_result(
            "other-range",
            alias_score=1.0,
            scope_match="other_range_context",
        )

        selected = select_generation_evidence(
            [overlap, stronger_other_range]
        )

        self.assertEqual(selected, [overlap, stronger_other_range])
        self.assertTrue(has_primary_grounding(selected))

    def test_extractive_fallback_prefers_expert_anchor_over_web_evidence(
        self,
    ) -> None:
        web = make_result(
            "web",
            evidence_type="web_database",
            scope_match="general_evidence",
        )
        expert = make_result(
            "expert",
            scope_match="general_evidence",
            source_ids=["source-direct"],
        )

        self.assertEqual(
            select_extractive_fallback_evidence([web, expert]),
            [expert],
        )

    def test_extractive_fallback_preserves_ranked_local_example(
        self,
    ) -> None:
        local = make_result(
            "local",
            scope_match="local_example",
            source_ids=["source-local"],
        )
        general = make_result(
            "general",
            scope_match="general_evidence",
            source_ids=["source-general"],
        )

        self.assertEqual(
            select_extractive_fallback_evidence([local, general]),
            [local],
        )

    def test_local_rejection_keeps_a_near_tied_dense_contender(
        self,
    ) -> None:
        generic = make_result(
            "generic",
            scope_match="general_evidence",
            source_ids=["generic-source"],
            score=0.62,
            text_score=0.0,
            semantic_match_type="dense",
        )
        direct = make_result(
            "direct",
            scope_match="general_evidence",
            source_ids=["direct-source"],
            score=0.54,
            text_score=0.0,
            semantic_match_type="dense",
        )
        sibling = make_result(
            "generic-sibling",
            scope_match="general_evidence",
            source_ids=["generic-source"],
            score=0.51,
            text_score=0.0,
            semantic_match_type="dense",
        )

        selected = select_extractive_fallback_evidence(
            [generic, direct, sibling],
            exclude_local_examples=True,
        )

        self.assertEqual(selected, [generic, direct, sibling])

    def test_dense_fallback_can_reanchor_on_directness_signal(self) -> None:
        generic = [
            make_result(
                f"generic-{index}",
                scope_match="general_evidence",
                source_ids=[f"generic-source-{index}"],
                score=0.60 - index * 0.02,
                semantic_match_type="dense",
            )
            for index in range(5)
        ]
        direct = make_result(
            "direct-concept-match",
            scope_match="unspecified_scope",
            source_ids=["direct-source"],
            score=0.44,
            text_score=0.0,
            semantic_match_type="dense",
        )
        direct.concept_coverage = 0.25

        selected = select_extractive_fallback_evidence(
            [*generic, direct]
        )

        self.assertEqual(selected[0], direct)
        self.assertEqual(selected, [direct, generic[0]])

    def test_artifact_shaped_old_omissions_keep_direct_expected_units(
        self,
    ) -> None:
        web = make_result(
            "web-distractor",
            evidence_type="web_database",
            scope_match="general_evidence",
            source_ids=[],
            score=0.548,
            text_score=0.0,
            semantic_match_type="dense",
        )
        generic = make_result(
            "generic-difficulty",
            scope_match="general_evidence",
            source_ids=["generic"],
            score=0.544,
            text_score=0.0,
            semantic_match_type="dense",
        )
        interrupted_phrase = make_result(
            "die-forelle-ku-006",
            scope_match="unspecified_scope",
            source_ids=["kim-die-forelle-04"],
            score=0.444,
            text_score=0.0,
            semantic_match_type="dense",
        )
        interrupted_phrase.concept_coverage = 0.142857
        interrupted_phrase.content_concept_coverage = 0.142857

        self.assertEqual(
            select_extractive_fallback_evidence(
                [web, generic, interrupted_phrase]
            ),
            [interrupted_phrase, generic],
        )

        difficult_generic = make_result(
            "die-forelle-ku-009",
            scope_match="general_evidence",
            source_ids=["generic-difficulty"],
            score=0.625,
            text_score=0.0,
            semantic_match_type="dense",
        )
        difficult_generic.concept_coverage = 0.5
        difficult_generic.content_concept_coverage = 0.5
        difficult_direct = make_result(
            "die-forelle-ku-011",
            scope_match="general_evidence",
            source_ids=["kim-die-forelle-08"],
            score=0.544,
            text_score=0.0,
            semantic_match_type="dense",
        )
        unrelated_local = make_result(
            "die-forelle-ku-010",
            scope_match="local_example",
            source_ids=["kim-die-forelle-07"],
            score=0.513,
            text_score=0.0,
            semantic_match_type="dense",
        )
        unrelated_local.concept_coverage = 0.5
        unrelated_local.content_concept_coverage = 0.5

        selected = select_extractive_fallback_evidence(
            [difficult_generic, difficult_direct, unrelated_local],
            exclude_local_examples=True,
        )
        self.assertEqual(selected, [difficult_generic, difficult_direct])

    def test_meter_return_fallback_anchors_whole_piece_unit_only(self) -> None:
        m39_local = make_result(
            "in-flowery-clouds-ku-008",
            scope_match="local_example",
            source_ids=["meter-change"],
            score=0.597,
            text_score=0.0,
            semantic_match_type="dense",
            review_warning="rewrite_review_pending",
        )
        m39_local.concept_coverage = 0.25
        m39_local.content_concept_coverage = 0.25
        meter_return = make_result(
            "in-flowery-clouds-ku-011",
            scope_match="general_evidence",
            source_ids=["meter-return"],
            score=0.517,
            text_score=0.0,
            semantic_match_type="dense",
            review_warning="rewrite_review_pending",
        )
        meter_return.concept_coverage = 0.75
        meter_return.content_concept_coverage = 0.75
        m19_local = make_result(
            "in-flowery-clouds-ku-006",
            scope_match="local_example",
            source_ids=["breath-example"],
            score=0.455,
            text_score=0.0,
            semantic_match_type="dense",
        )

        self.assertEqual(
            select_extractive_fallback_evidence(
                [m39_local, meter_return, m19_local]
            ),
            [meter_return],
        )

    def test_transposition_fallback_omits_fermata_warning(self) -> None:
        transposition = make_result(
            "in-flowery-clouds-ku-007",
            scope_match="general_evidence",
            source_ids=["transposition"],
            score=0.702,
            text_score=0.0,
            semantic_match_type="dense",
        )
        transposition.concept_coverage = 0.428571
        transposition.content_concept_coverage = 0.142857
        optional_note = make_result(
            "in-flowery-clouds-ku-013",
            scope_match="general_evidence",
            source_ids=["optional-note"],
            score=0.599,
            text_score=0.0,
            semantic_match_type="dense",
        )
        fermata = make_result(
            "in-flowery-clouds-ku-001",
            scope_match="local_example",
            source_ids=["fermata"],
            score=0.560,
            text_score=0.0,
            semantic_match_type="dense",
            review_warning="rewrite_review_pending",
        )

        selected = select_extractive_fallback_evidence(
            [transposition, optional_note, fermata]
        )

        self.assertEqual(selected, [transposition, optional_note])
        self.assertFalse(
            any(
                result.record.get("retrieval_review_warning")
                for result in selected
            )
        )

    def test_broad_fallback_reanchors_and_keeps_same_source_bundle(
        self,
    ) -> None:
        distractor = make_result(
            "distractor",
            scope_match="general_evidence",
            source_ids=["source-other"],
            score=21.5,
            semantic_match_type="broad_guidance",
        )
        local = make_result(
            "local",
            scope_match="local_example",
            source_ids=["source-local"],
            score=23.0,
            semantic_match_type="broad_guidance",
        )
        sibling_one = make_result(
            "sibling-one",
            scope_match="general_evidence",
            source_ids=["source-bundle"],
            score=18.5,
            semantic_match_type="broad_guidance",
        )
        sibling_two = make_result(
            "sibling-two",
            scope_match="general_evidence",
            source_ids=["source-bundle"],
            score=16.5,
            semantic_match_type="broad_guidance",
        )
        anchor = make_result(
            "anchor",
            scope_match="general_evidence",
            source_ids=["source-bundle", "source-secondary"],
            score=22.5,
            semantic_match_type="broad_guidance",
        )

        self.assertEqual(
            select_extractive_fallback_evidence(
                [distractor, local, sibling_one, sibling_two, anchor]
            ),
            [anchor, sibling_one, sibling_two],
        )

    def test_extractive_fallback_keeps_unscoped_direct_anchor(self) -> None:
        direct = make_result(
            "direct",
            scope_match="unspecified_scope",
            source_ids=["source-direct"],
        )
        general = make_result(
            "general",
            scope_match="general_evidence",
            source_ids=["source-other"],
        )

        self.assertEqual(
            select_extractive_fallback_evidence([direct, general]),
            [direct],
        )

    def test_other_range_context_is_capped_at_two_records(self) -> None:
        overlap = make_result("overlap", alias_score=0.0)
        contexts = [
            make_result(
                f"other-range-{index}",
                alias_score=0.0,
                scope_match="other_range_context",
            )
            for index in range(3)
        ]

        selected = select_generation_evidence([overlap, *contexts])

        self.assertEqual(selected, [overlap, *contexts[:2]])

    def test_unconfirmed_range_context_cannot_become_primary(self) -> None:
        pending = make_result(
            "pending",
            alias_score=1.0,
            scope_match="unscoped_pending_context",
        )
        unspecified = make_result(
            "unspecified",
            alias_score=0.0,
            scope_match="unspecified_context",
        )

        self.assertFalse(has_primary_grounding([pending, unspecified]))
        self.assertEqual(
            select_generation_evidence([pending, unspecified]),
            [],
        )

    def test_prompt_marks_other_range_evidence_as_context_only(self) -> None:
        overlap = make_result("overlap")
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
        )

        prompt = build_messages(
            "die-forelle",
            "2-5",
            "악센트는 무엇을 표현할까?",
            [overlap, other_range],
        )[1]["content"]

        self.assertIn("evidence_role: selected_range_support", prompt)
        self.assertIn("evidence_role: other_range_context_only", prompt)
        self.assertIn(
            "citation_label: (none; context-only evidence is not citable)",
            prompt,
        )
        self.assertIn("internal range-mismatch signal", prompt)
        self.assertIn(
            "do not infer, cite, paraphrase, or transfer",
            prompt,
        )

    def test_local_example_uses_only_canonical_ranges_and_stays_local(
        self,
    ) -> None:
        local = make_result(
            "una-voce-poco-fa-ku-028",
            scope_match="local_example",
        )
        local.record["piece"] = "una-voce-poco-fa"
        local.record["work"] = "Una voce poco fa"
        local.record["measure_range"] = [[63, 63], [82, 83]]
        local.record["answer"] = (
            "고음이 강박에 놓이지 않은 경우에는 단어와 음악의 "
            "강세 관계에 유의해 유연하게 노래해야 한다."
        )
        local.record["source_answer_context"] = [
            {
                "source_id": "yeon-una-voce-poco-fa-11",
                "question": "고음을 어떻게 불러야 할까?",
                "answer": (
                    "첫 마디가 아니라 82마디에서는 고음이 강박에 "
                    "놓이지 않는다. m. 51 및 measures 52–53의 "
                    "표기는 참고하지 말고, 68, 134마디는 지나치게 "
                    "강조하지 않는다."
                ),
                "legacy_measure_range_hints": [[82, 82]],
            }
        ]

        messages = build_messages(
            "una-voce-poco-fa",
            "",
            "이 노래를 부를 때 무엇에 유의해야 할까?",
            [local],
        )
        system_prompt = messages[0]["content"]
        context = messages[1]["content"]

        self.assertIn("Evidence_role=local_example", system_prompt)
        self.assertIn("evidence_role: local_example", context)
        self.assertIn("measure_range: 63, 82-83", context)
        self.assertIn("citable confirmed local example", context)
        self.assertIn("does not establish", context)
        self.assertNotIn("82마디에서는", context)
        self.assertNotIn("첫 마디", context)
        self.assertNotIn("m. 51", context)
        self.assertNotIn("measures 52–53", context)
        self.assertNotIn("68, 134마디", context)
        self.assertIn("해당 대목에서는", context)
        authority = expert_prompt_factuality_material(local.record, [])
        self.assertFalse(authority["range_partitioned"])
        self.assertFalse(
            any(
                locator in text
                for text in authority["answer_texts"]
                for locator in (
                    "첫 마디",
                    "82마디",
                    "m. 51",
                    "measures 52–53",
                    "68, 134마디",
                )
            )
        )
        extractive = build_extractive_answer([local])
        self.assertIn("확인된 국소 예시(마디 63, 82-83)", extractive)
        self.assertIn(local.record["answer"], extractive)
        self.assertIn("[una-voce-poco-fa-ku-028]", extractive)

    def test_local_only_answer_cannot_claim_piece_wide_frequency(
        self,
    ) -> None:
        local = make_result(
            "la-capinera-ku-021",
            scope_match="local_example",
        )
        local.record["measure_range"] = [[30, 30], [33, 33]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "이 곡 전반에는 도약음이 많이 나오므로 항상 "
                    "고음을 약하게 불러야 한다. [E1]"
                ),
                [local],
            )
        )
        for unsupported_claim in (
            "이 노래에는 도약음이 자주 나오므로 조심한다. [E1]",
            "고음이 강박에 있지 않은 경우가 많으므로 조심한다. [E1]",
            "대부분의 구간에서 고음을 가볍게 부른다. [E1]",
            "전 곡에 걸쳐 같은 방식으로 노래한다. [E1]",
            "모든 대목에서 같은 강세를 사용한다. [E1]",
        ):
            with self.subTest(unsupported_claim=unsupported_claim):
                self.assertTrue(
                    answer_overgeneralizes_local_examples(
                        unsupported_claim,
                        [local],
                    )
                )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "30마디와 33마디에서는 도약하는 고음이 "
                    "지나치게 강조되지 않도록 유의한다. [E1]"
                ),
                [local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "30마디에서는 도약음이 자주 나오지는 않는다고 "
                    "본다. [E1]"
                ),
                [local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "30마디의 근거만으로 곡 전반에 적용된다고 "
                    "확인되지 않았다. [E1]"
                ),
                [local],
            )
        )

        whole = make_result(
            "la-capinera-ku-whole",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "이 곡 전반의 가창에 유의해야 한다. [E1] [E2]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "이 곡 전반의 가창에 유의해야 한다. [E1]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "이 곡 전반의 가창에 유의하되, 30마디의 "
                    "도약음을 국소 예시로 참고한다. [E1] [E2]"
                ),
                [whole, local],
            )
        )

    def test_each_cited_local_example_requires_its_own_canonical_range(
        self,
    ) -> None:
        first = make_result(
            "una-voce-poco-fa-ku-028",
            scope_match="local_example",
        )
        first.record["measure_range"] = [[63, 63], [82, 83]]
        second = make_result(
            "una-voce-poco-fa-ku-020",
            scope_match="local_example",
        )
        second.record["measure_range"] = [[68, 69], [72, 73]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "63마디에서 고음을 유연하게 부른다. [E1] [E2]",
                [first, second],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                (
                    "63마디에서 고음을 유연하게 부른다. [E1]\n"
                    "68–69마디에서 악센트를 가볍게 표현한다. "
                    "[E2]"
                ),
                [first, second],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "82마디부터 83마디에서 고음을 유연하게 부른다. [E1]",
                [first, second],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "64마디에서 고음을 유연하게 부른다. [E1]",
                [first, second],
            )
        )

    def test_unlabeled_body_cannot_be_attributed_to_retrieved_locals(
        self,
    ) -> None:
        whole = make_result(
            "whole",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        local = make_result("local", scope_match="local_example")
        local.record["measure_range"] = [[30, 30]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                (
                    "도약 직전에 충분히 호흡한다.\n\n"
                    "제공된 검색 근거: [E1] [E2]"
                ),
                [whole, local],
            )
        )

    def test_measure_locators_need_query_or_cited_local_authority(
        self,
    ) -> None:
        whole = make_result(
            "la-capinera-ku-028",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        whole.record["source_answer_context"] = [
            {
                "source_id": "yeon-la-capinera-10",
                "question": "",
                "answer": "135마디에서 선율을 이어 부른다.",
                "legacy_measure_range_hints": [[135, 135]],
            }
        ]
        local = make_result(
            "la-capinera-ku-011",
            scope_match="local_example",
        )
        local.record["measure_range"] = [[41, 43]]

        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "135마디에서 선율을 이어 부른다. [E1]",
                [whole],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "135마디에서 선율을 이어 부른다. [E1]",
                [whole, local],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "41–43마디에서 레가토를 유지한다. [E1]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "41–43마디에서 레가토를 유지한다. [E2]",
                [whole, local],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "기교를 더할 때는 모음으로 선율을 이어 부른다. [E1]",
                [whole, local],
            )
        )

        selected = make_result(
            "die-forelle-ku-002",
            scope_match="overlaps_query_range",
        )
        selected.record["measure_range"] = [[2, 27]]
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "2–5마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "3마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertTrue(
            answer_overgeneralizes_local_examples(
                "1–6마디에서 악센트를 살린다. [E1]",
                [selected],
                [[2, 5]],
            )
        )
        self.assertFalse(
            answer_overgeneralizes_local_examples(
                "작품의 분위기를 먼저 이해한다. [E1]",
                [whole, local],
            )
        )

    def test_extractive_answer_does_not_present_other_range_as_grounding(
        self,
    ) -> None:
        overlap = make_result("overlap")
        overlap.record["answer"] = "선택 범위 답변"
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간 답변"

        mixed = build_extractive_answer([overlap, other_range])
        secondary_only = build_extractive_answer([other_range])

        self.assertIn("선택 범위 답변", mixed)
        self.assertNotIn("다른 구간 답변", mixed)
        self.assertIn(
            "선택한 마디 범위를 직접 뒷받침하는 근거는 없습니다.",
            secondary_only,
        )
        self.assertIn("다른 구간의 관련 주석", secondary_only)
        self.assertIn("다른 구간 답변", secondary_only)

    def test_secondary_citations_are_removed_from_answer_and_footer(
        self,
    ) -> None:
        overlap = make_result("die-forelle-ku-001")
        overlap.record["answer"] = "선택 범위의 근거"
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간의 강약 교대"

        explicit = finalize_answer_citations(
            (
                "다른 구간의 강약 교대 [E2] "
                "[die-forelle-ku-002]"
            ),
            [overlap, other_range],
        )
        automatic_footer = finalize_answer_citations(
            "인용을 생략한 답변",
            [overlap, other_range],
        )

        self.assertIn("[die-forelle-ku-001]", explicit)
        self.assertNotIn("[die-forelle-ku-002]", explicit)
        self.assertIn("선택 범위의 근거", explicit)
        self.assertNotIn("다른 구간의 강약 교대", explicit)
        self.assertIn("[die-forelle-ku-001]", automatic_footer)
        self.assertNotIn(
            "[die-forelle-ku-002]",
            automatic_footer,
        )

    def test_secondary_citation_variants_trigger_safe_fallback(self) -> None:
        overlap = make_result("die-forelle-ku-001")
        overlap.record["answer"] = "선택 범위의 근거"
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )
        other_range.record["answer"] = "다른 구간 위험 주장"

        secondary_references = [
            "[E 2]",
            "[Evidence 2]",
            "[E–2]",
            "[Evidence no. 2]",
            "Evidence 2",
            "근거: 2",
            "DIE-FORELLE-KU-002",
        ]
        for reference in secondary_references:
            with self.subTest(reference=reference):
                finalized = finalize_answer_citations(
                    f"다른 구간 위험 주장 {reference}",
                    [overlap, other_range],
                )
                self.assertIn("선택 범위의 근거", finalized)
                self.assertIn("[die-forelle-ku-001]", finalized)
                self.assertNotIn("다른 구간 위험 주장", finalized)
                self.assertNotIn(
                    "die-forelle-ku-002",
                    finalized.lower(),
                )

    def test_secondary_citation_fallback_uses_compact_selection(self) -> None:
        primary = [
            make_result(
                f"primary-{index}",
                scope_match="overlaps_query_range",
                source_ids=[f"source-{index}"],
            )
            for index in range(4)
        ]
        other_range = make_result(
            "other-range",
            scope_match="other_range_context",
            source_ids=["other-source"],
        )

        finalized = finalize_answer_citations(
            "다른 구간의 위험 주장 [E5]",
            [*primary, other_range],
        )

        cited_primary = [
            result.record["id"]
            for result in primary
            if f'[{result.record["id"]}]' in finalized
        ]
        self.assertEqual(
            cited_primary,
            [primary[0].record["id"], primary[1].record["id"]],
        )
        self.assertNotIn("다른 구간의 위험 주장", finalized)

    def test_music_and_korean_counts_are_not_secondary_citations(self) -> None:
        overlap = make_result("die-forelle-ku-001")
        other_range = make_result(
            "die-forelle-ku-002",
            scope_match="other_range_context",
        )

        ordinary_uses = [
            "최저음은 E2이다. [E1]",
            "최저음은 E2. [E1]",
            "음역 하한은 E2! [E1]",
            "근거 2가지를 설명한다. [E1]",
            "출처 2곳을 비교한다. [E1]",
        ]
        for answer in ordinary_uses:
            with self.subTest(answer=answer):
                finalized = finalize_answer_citations(
                    answer,
                    [overlap, other_range],
                )
                self.assertIn(answer.rsplit(" ", 1)[0], finalized)
                self.assertIn("[die-forelle-ku-001]", finalized)

    def test_prompt_forbids_claims_split_into_an_inapplicable_unit(
        self,
    ) -> None:
        result = make_result("la-capinera-ku-001")
        result.record["answer"] = "pianissimo로 시작하는 것이 좋다."
        result.record["source_answer_context"] = [
            {
                "source_id": "kim-la-capinera-01",
                "question": "어떻게 시작하고 발음할까?",
                "answer": (
                    "pianissimo로 시작하고 l과 r을 정확히 발음한다."
                ),
                "legacy_measure_range_hints": [[11, 14], [78, 81]],
                "linked_knowledge_unit_ids": [
                    "la-capinera-ku-001",
                    "la-capinera-ku-002",
                ],
            }
        ]
        messages = build_messages(
            "la-capinera",
            "78-81",
            "어떻게 시작할까?",
            [result],
        )
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        self.assertIn(
            "do not use a source-answer claim absent from this record's",
            system_prompt,
        )
        self.assertIn(
            "another retrieved, range-applicable record",
            system_prompt,
        )
        self.assertIn("split_source_context_withheld", user_prompt)
        self.assertIn("pianissimo로 시작하는 것이 좋다.", user_prompt)
        self.assertNotIn("l과 r을 정확히 발음한다", user_prompt)


class GenerationArtifactRepairTests(unittest.TestCase):
    def test_detects_unseen_camel_case_inside_korean_text(self) -> None:
        messages = [{"role": "user", "content": "꾸밈음을 가볍게 부른다."}]

        self.assertEqual(
            suspicious_generation_tokens(
                "꾸SizePolicy는 가볍게 부른다.",
                messages,
            ),
            ["SizePolicy"],
        )
        self.assertEqual(
            suspicious_generation_tokens(
                "Larghetto는 여유 있게 부른다.",
                messages,
            ),
            [],
        )

    def test_generation_retries_and_repairs_corrupted_token(self) -> None:
        evidence = make_result("la-capinera-ku-009")

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return [evidence]

        calls = []

        def generator(messages):
            calls.append(messages)
            if len(calls) == 1:
                return "꾸SizePolicy는 새소리를 표현한다. [E1]"
            self.assertIn("SizePolicy", messages[-1]["content"])
            return "꾸밈음은 새소리를 표현한다. [E1]"

        answer, results, _messages, context_limited = (
            generate_with_context_retry(
                FixedIndex(),
                query="꾸밈음은 어떻게 표현할까?",
                piece="la-capinera",
                measure_ranges=[[28, 35]],
                measures="28-35",
                topic=None,
                top_k=6,
                generator=generator,
            )
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(answer, "꾸밈음은 새소리를 표현한다. [E1]")
        self.assertEqual(results, [evidence])
        self.assertFalse(context_limited)

    def test_generation_repairs_unbound_local_with_whole_evidence_present(
        self,
    ) -> None:
        whole = make_result(
            "una-voce-poco-fa-ku-whole",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        local = make_result(
            "una-voce-poco-fa-ku-028",
            scope_match="local_example",
        )
        local.record["measure_range"] = [[63, 63], [82, 83]]

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return [whole, local]

        calls = []

        def generator(messages):
            calls.append(messages)
            if len(calls) == 1:
                return (
                    "고음이 강박에 놓이지 않으면 유연하게 "
                    "노래한다. [E2]"
                )
            self.assertIn(
                "must cite that item's [E#] label",
                messages[-1]["content"],
            )
            self.assertIn(
                "canonical measure_range",
                messages[-1]["content"],
            )
            return (
                "63마디에서는 고음이 강박에 놓이지 않으므로 "
                "유연하게 노래한다. [E2]"
            )

        answer, results, _messages, context_limited = (
            generate_with_context_retry(
                FixedIndex(),
                query="이 곡을 부를 때 무엇에 유의할까?",
                piece="una-voce-poco-fa",
                measure_ranges=[],
                measures="",
                topic=None,
                top_k=6,
                generator=generator,
            )
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(results, [whole, local])
        self.assertIn("63마디", answer)
        self.assertIn("[E2]", answer)
        self.assertFalse(context_limited)

    def test_generation_removes_source_only_measure_from_whole_record(
        self,
    ) -> None:
        whole = make_result(
            "la-capinera-ku-028",
            scope_match="general_evidence",
            measure_status="whole_piece",
        )
        whole.record["measure_range"] = []
        whole.record["source_answer_context"] = [
            {
                "source_id": "yeon-la-capinera-10",
                "question": "",
                "answer": "135마디에서 모음으로 선율을 이어 부른다.",
                "legacy_measure_range_hints": [[135, 135]],
            }
        ]

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return [whole]

        calls = []

        def generator(messages):
            calls.append(messages)
            if len(calls) == 1:
                return "135마디에서 모음으로 선율을 이어 부른다. [E1]"
            self.assertIn(
                "general_support item never authorizes a measure number",
                messages[-1]["content"],
            )
            return "기교를 더할 때는 모음으로 선율을 이어 부른다. [E1]"

        answer, results, _messages, context_limited = (
            generate_with_context_retry(
                FixedIndex(),
                query="이 곡을 부를 때 무엇에 유의할까?",
                piece="la-capinera",
                measure_ranges=[],
                measures="",
                topic=None,
                top_k=6,
                generator=generator,
            )
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(results, [whole])
        self.assertNotIn("135마디", answer)
        self.assertIn("[E1]", answer)
        self.assertFalse(context_limited)


class ReviewConstraintPromptTests(unittest.TestCase):
    def test_unrelated_warning_is_not_added_to_cited_safe_evidence(
        self,
    ) -> None:
        safe = make_result(
            "safe",
            scope_match="general_evidence",
        )
        warning = make_result(
            "unrelated-warning",
            rewrite_status="needs_review",
            rewrite_notes="다른 음정 표기를 확인해야 한다.",
            review_warning="rewrite_review_pending",
            scope_match="local_example",
        )

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return [safe, warning]

        unrelated_disclosure = build_review_disclosure([warning])
        answer, _results, _messages, _limited = (
            generate_with_context_retry(
                FixedIndex(),
                query="전주는 어떻게 선택할까?",
                piece="die-forelle",
                measure_ranges=[],
                measures="",
                topic=None,
                top_k=6,
                generator=lambda _messages: (
                    unrelated_disclosure
                    + "\n\n안전한 근거의 답변이다. [E1]"
                ),
            )
        )

        self.assertEqual(answer, "안전한 근거의 답변이다. [E1]")
        self.assertNotIn("검토 주의:", answer)
        self.assertNotIn("다른 음정", answer)

    def test_unusable_citation_with_warning_candidate_fails_closed(
        self,
    ) -> None:
        safe = make_result(
            "safe",
            scope_match="general_evidence",
        )
        warning = make_result(
            "warning",
            rewrite_status="needs_review",
            rewrite_notes="표기를 확인해야 한다.",
            review_warning="rewrite_review_pending",
            scope_match="local_example",
        )

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return [safe, warning]

        answer, _results, _messages, _limited = (
            generate_with_context_retry(
                FixedIndex(),
                query="조성을 바꿔도 될까?",
                piece="in-flowery-clouds",
                measure_ranges=[],
                measures="",
                topic=None,
                top_k=6,
                generator=lambda _messages: "조성을 바꿀 수 있다. [E99]",
            )
        )

        self.assertIn("<NO_GROUNDED_ANSWER>", answer)

    def test_warning_candidates_make_uncited_draft_fail_closed(self) -> None:
        warning = make_result(
            "warning",
            rewrite_status="needs_review",
            rewrite_notes="표기를 확인해야 한다.",
            review_warning="rewrite_review_pending",
        )

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return [warning]

        answer, _results, _messages, _limited = (
            generate_with_context_retry(
                FixedIndex(),
                query="표현은 어떻게 할까?",
                piece="die-forelle",
                measure_ranges=[[2, 5]],
                measures="2-5",
                topic=None,
                top_k=6,
                generator=lambda _messages: "인용 없는 답변이다.",
            )
        )

        self.assertIn("<NO_GROUNDED_ANSWER>", answer)

    def test_needs_review_draft_gets_compliance_repair_pass(self) -> None:
        warning = make_result(
            "in-flowery-clouds-ku-warning",
            rewrite_status="needs_review",
            rewrite_notes=(
                "한 원문은 47마디와 동시에라고 하고 다른 원문은 "
                "45마디부터라고 하므로 확인해야 한다."
            ),
            review_warning="rewrite_review_pending",
        )

        class FixedIndex:
            @staticmethod
            def search(**_kwargs):
                return [warning]

        calls = []

        def generator(messages):
            calls.append(messages)
            if len(calls) == 1:
                return "박자와 조성이 동시에 바뀐다. [E1]"
            self.assertIn(
                "warning prefix alone is not enough",
                messages[-1]["content"],
            )
            return (
                "자료마다 조성 변화 시점은 다르게 기록되어 있지만, "
                "공통적으로 분위기 전환을 만든다. [E1]"
            )

        answer, _results, _messages, _limited = (
            generate_with_context_retry(
                FixedIndex(),
                query="분위기는 왜 바뀔까?",
                piece="in-flowery-clouds",
                measure_ranges=[[39, 49]],
                measures="39-49",
                topic=None,
                top_k=6,
                generator=generator,
            )
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(answer.startswith("검토 주의:"))
        self.assertNotIn("동시에 바뀐다", answer)
        self.assertIn("시점은 다르게 기록", answer)

    def test_warning_metadata_becomes_a_mandatory_prompt_constraint(self) -> None:
        warning = make_result(
            "die-forelle-ku-warning",
            rewrite_status="needs_review",
            rewrite_notes="원문 주장 사이의 정확성 충돌을 확인해야 함",
            measure_status="waiting_for_review",
            measure_notes="정확한 적용 위치가 아직 확인되지 않음",
            review_warning="rewrite_and_measure_review_pending",
        )

        constraints = build_review_constraints([warning])
        user_prompt = build_messages(
            "die-forelle",
            "2-5",
            "어떻게 표현할까?",
            [warning],
        )[1]["content"]

        self.assertIn("MANDATORY REVIEW CONSTRAINTS", constraints)
        self.assertIn("actually\nuses and cites", constraints)
        self.assertIn("answer omits", constraints)
        self.assertIn("rewrite_status: needs_review", constraints)
        self.assertIn("원문 주장 사이의 정확성 충돌을 확인해야 함", constraints)
        self.assertIn("measure_status: waiting_for_review", constraints)
        self.assertIn("정확한 적용 위치가 아직 확인되지 않음", constraints)
        self.assertIn(
            build_review_disclosure([warning]),
            constraints,
        )
        self.assertIn(constraints, user_prompt)

    def test_local_example_disclosure_hides_disputed_measure_numbers(
        self,
    ) -> None:
        warning = make_result(
            "in-flowery-clouds-ku-008",
            rewrite_status="needs_review",
            rewrite_notes=(
                "한 원문은 47마디와 동시에 바뀐다고 하고, 다른 "
                "원문은 45마디부터 바뀐다고 하므로 확인해야 한다."
            ),
            review_warning="rewrite_review_pending",
            scope_match="local_example",
        )
        warning.record["measure_range"] = [[39, 49]]

        disclosure = build_review_disclosure([warning])
        context = build_messages(
            "in-flowery-clouds",
            "",
            "이 곡을 부를 때 표현에 유의해야 할 점은?",
            [warning],
        )[1]["content"]

        self.assertIn("미확인 위치 A", disclosure)
        self.assertIn("미확인 위치 B", disclosure)
        self.assertNotIn("47마디", disclosure)
        self.assertNotIn("45마디", disclosure)
        self.assertIn("measure_range: 39-49", context)
        self.assertNotIn("47마디", context)
        self.assertNotIn("45마디", context)

    def test_warning_disclosure_is_natural_visible_and_idempotent(self) -> None:
        warning = make_result(
            "in-flowery-clouds-ku-warning",
            rewrite_status="needs_review",
            rewrite_notes="박자 표기의 정확성을 확인해야 한다.",
            measure_status="waiting_for_review",
            measure_notes="다른 악보나 곡이었나 확인 필요",
            review_warning="rewrite_and_measure_review_pending",
        )

        disclosure = build_review_disclosure([warning])
        answer = ensure_review_disclosure("전문가 설명이다.", [warning])

        self.assertTrue(disclosure.startswith("검토 주의:"))
        self.assertIn("박자 표기의 정확성을 확인해야 한다.", disclosure)
        self.assertIn("정확한 적용 위치", disclosure)
        self.assertIn("현재 선택한 곡에 해당하는지", disclosure)
        self.assertEqual(
            answer,
            disclosure + "\n\n전문가 설명이다.",
        )
        self.assertEqual(
            ensure_review_disclosure(answer, [warning]),
            answer,
        )

    def test_record_helper_matches_result_helper(self) -> None:
        warning = make_result(
            "in-flowery-clouds-ku-warning",
            rewrite_status="needs_review",
            rewrite_notes="박자 표기의 정확성을 확인해야 한다.",
            measure_status="waiting_for_review",
            measure_notes="다른 악보나 곡이었나 확인 필요",
            review_warning="rewrite_and_measure_review_pending",
        )

        self.assertEqual(
            build_review_disclosure_from_records([warning.record]),
            build_review_disclosure([warning]),
        )

    def test_exact_leading_disclosure_is_idempotent(self) -> None:
        warning = make_result(
            "die-forelle-ku-warning",
            rewrite_notes="원문 주장을 확인해야 한다.",
            review_warning="rewrite_review_pending",
        )
        disclosure = build_review_disclosure([warning])
        answer = disclosure + "\n\n전문가 설명이다."

        self.assertEqual(
            ensure_review_disclosure(answer, [warning]),
            answer,
        )

    def test_leading_disclosure_requires_a_whitespace_boundary(self) -> None:
        warning = make_result(
            "die-forelle-ku-warning",
            rewrite_notes="원문 주장을 확인해야 한다.",
            review_warning="rewrite_review_pending",
        )
        disclosure = build_review_disclosure([warning])
        malformed = disclosure + "전문가 설명이다."

        self.assertEqual(
            ensure_review_disclosure(malformed, [warning]),
            disclosure + "\n\n" + malformed,
        )

    def test_non_leading_disclosure_is_repaired(self) -> None:
        warning = make_result(
            "die-forelle-ku-warning",
            rewrite_notes="원문 주장을 확인해야 한다.",
            review_warning="rewrite_review_pending",
        )
        disclosure = build_review_disclosure([warning])
        misplaced = "전문가 설명이다.\n\n" + disclosure

        self.assertEqual(
            ensure_review_disclosure(misplaced, [warning]),
            disclosure + "\n\n" + misplaced,
        )

    def test_ready_reviewed_evidence_does_not_get_a_false_warning(self) -> None:
        ready = make_result("die-forelle-ku-ready")

        constraints = build_review_constraints([ready])
        user_prompt = build_messages(
            "die-forelle",
            "2-5",
            "어떻게 표현할까?",
            [ready],
        )[1]["content"]

        self.assertEqual(constraints, "")
        self.assertEqual(build_review_disclosure([ready]), "")
        self.assertEqual(
            ensure_review_disclosure("확정 답변", [ready]),
            "확정 답변",
        )
        self.assertNotIn("MANDATORY REVIEW CONSTRAINTS", user_prompt)


class RangeScopedSourceContextTests(unittest.TestCase):
    def test_no_range_neutralizes_source_locators_for_nonlocal_units(
        self,
    ) -> None:
        for scope_match, measure_status in (
            ("general_evidence", "whole_piece"),
            ("unspecified_scope", "unspecified"),
        ):
            with self.subTest(scope_match=scope_match):
                result = make_result(
                    f"source-{scope_match}",
                    scope_match=scope_match,
                    measure_status=measure_status,
                )
                result.record["measure_range"] = []
                result.record["source_answer_context"] = [
                    {
                        "source_id": "source-a",
                        "question": "135마디에서 어떻게 부를까?",
                        "answer": "135마디에서 모음으로 선율을 이어 부른다.",
                        "legacy_measure_range_hints": [[135, 135]],
                    }
                ]

                prompt = build_messages(
                    "la-capinera",
                    "",
                    "이 곡을 부를 때 무엇에 유의할까?",
                    [result],
                )[1]["content"]
                authority = expert_prompt_factuality_material(
                    result.record,
                    [],
                )

                self.assertNotIn("135마디", prompt)
                self.assertIn("해당 대목에서", prompt)
                self.assertIn(
                    "no_range_expert_source_answer_"
                    "with_source_locators_neutralized",
                    prompt,
                )
                self.assertFalse(
                    any(
                        "135마디" in text
                        for text in authority["answer_texts"]
                        + authority["question_contexts"]
                    )
                )

    def test_selected_range_omits_other_range_source_and_combined_rewrite(
        self,
    ) -> None:
        merged = make_result("la-capinera-ku-018")
        merged.record["measure_range"] = [[51, 59], [117, 121]]
        merged.record["answer"] = (
            "앞 구간의 두 블록 대비와 뒤 구간의 빠른 교대를 함께 설명한다."
        )
        merged.record["source_answer_context"] = [
            {
                "source_id": "kim-la-capinera-17",
                "question": (
                    "117마디의 ff와 121마디의 pp는 왜 빠르게 "
                    "번갈아 나올까?"
                ),
                "answer": "빠른 교대는 메아리 효과로 볼 수 있다.",
                "legacy_measure_range_hints": [[117, 121]],
            },
            {
                "source_id": "yeon-la-capinera-07",
                "question": "",
                "answer": (
                    "51-59마디의 ff와 pp는 51-54마디 "
                    "55-59마디의 대비가 드러나도록 표현한다."
                ),
                "legacy_measure_range_hints": [[51, 59]],
            },
        ]

        prompt = build_messages(
            "la-capinera",
            "51-58",
            "왜 빠르게 번갈아 나올까?",
            [merged],
        )[1]["content"]

        self.assertIn("range_partitioned_source_context: true", prompt)
        self.assertIn(
            "Do not repeat any measure number from the source text",
            prompt,
        )
        self.assertIn(
            "range_scoped_expert_source_answer_"
            "with_source_locators_neutralized",
            prompt,
        )
        self.assertIn(
            "해당 대목의 ff와 pp는 두 부분의 대비가 드러나도록 "
            "표현한다.",
            prompt,
        )
        self.assertNotIn("빠른 교대는 메아리 효과로 볼 수 있다.", prompt)
        self.assertNotIn("117마디", prompt)
        self.assertNotIn("121마디", prompt)
        self.assertNotIn("51-59마디", prompt)
        self.assertNotIn(
            "앞 구간의 두 블록 대비와 뒤 구간의 빠른 교대를 함께 설명한다.",
            prompt,
        )
        authority = expert_prompt_factuality_material(
            merged.record,
            [[51, 58]],
        )
        self.assertTrue(authority["range_partitioned"])
        self.assertEqual(
            authority["answer_texts"],
            [
                "해당 대목의 ff와 pp는 두 부분의 대비가 드러나도록 "
                "표현한다."
            ],
        )
        self.assertNotIn("빠른 교대", " ".join(authority["answer_texts"]))
        self.assertNotIn(
            merged.record["answer"],
            authority["answer_texts"],
        )

    def test_no_range_keeps_all_source_context_and_combined_rewrite(
        self,
    ) -> None:
        merged = make_result("la-capinera-ku-018")
        merged.record["answer"] = "두 범위를 함께 설명한다."
        merged.record["source_answer_context"] = [
            {
                "source_id": "source-a",
                "question": "",
                "answer": "앞 범위 설명",
                "legacy_measure_range_hints": [[51, 59]],
            },
            {
                "source_id": "source-b",
                "question": "",
                "answer": "뒤 범위 설명",
                "legacy_measure_range_hints": [[117, 121]],
            },
        ]

        prompt = build_messages(
            "la-capinera",
            "",
            "전체적으로 어떻게 표현할까?",
            [merged],
        )[1]["content"]

        self.assertNotIn("range_partitioned_source_context: true", prompt)
        self.assertIn("앞 범위 설명", prompt)
        self.assertIn("뒤 범위 설명", prompt)
        self.assertIn("두 범위를 함께 설명한다.", prompt)
        authority = expert_prompt_factuality_material(
            merged.record,
            [],
        )
        self.assertFalse(authority["range_partitioned"])
        self.assertEqual(
            authority["answer_texts"],
            [
                "앞 범위 설명",
                "뒤 범위 설명",
                "두 범위를 함께 설명한다.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
