"""Focused tests for the two-stage atomic evidence protocol."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from soprano_qa import answer
from soprano_qa.answer import (
    ATOMIC_PLAN_RELATIONS,
    GeneratedAnswerRejected,
    atomic_claim_planner_response_format,
    atomic_claim_renderer_response_format,
    build_atomic_claim_planner_messages,
    build_atomic_claim_renderer_messages,
    build_atomic_evidence_units,
    build_atomic_source_claims,
    generate_atomic_grounded_answer,
    parse_atomic_claim_plan,
    parse_atomic_claim_render,
    validate_atomic_render_fidelity,
)
from soprano_qa.retrieval import SearchResult


def make_result(
    record_id: str,
    text: str,
    *,
    scope_match: str = "general_evidence",
    measure_range: list[list[int]] | None = None,
    evidence_type: str = "expert_annotation",
) -> SearchResult:
    ranges = measure_range or []
    return SearchResult(
        record={
            "id": record_id,
            "evidence_type": evidence_type,
            "piece": "die-forelle",
            "work": "Die Forelle",
            "answer": text,
            "measure_range": ranges,
            "measure_scope": "local" if ranges else "global",
            "measure_status": "specific" if ranges else "whole_piece",
            "source_ids": ["source-" + record_id],
            "web_source_ids": [],
            "claim_ids": [],
            "sources": [],
        },
        score=1.0,
        text_score=1.0,
        measure_score=1.0,
        piece_score=1.0,
        scope_match=scope_match,
    )


def planner_json(
    units,
    *relations: str,
    selected_unit_ids: list[str] | None = None,
) -> str:
    if not relations:
        relations = tuple("direct_answer" for _ in units)
    if selected_unit_ids is None:
        selected_unit_ids = [
            unit.unit_id
            for unit, relation in zip(units, relations)
            if relation in ATOMIC_PLAN_RELATIONS[:4]
        ]
    return json.dumps(
        {
            "selected_unit_ids": selected_unit_ids,
            "units": [
                {"unit_id": unit.unit_id, "relation": relation}
                for unit, relation in zip(units, relations)
            ]
        }
    )


class AtomicEvidenceUnitTests(unittest.TestCase):
    def test_complete_units_are_exact_and_context_only_text_is_hidden(
        self,
    ) -> None:
        source = (
            "D.S. al Coda 표시는 그대로 따른다. Op. 32 표기도 유지한다. "
            "G. Verdi 이름도 나누지 않는다. 이는 앞 문장을 설명한다."
        )
        primary = make_result(
            "die-forelle-ku-001",
            source,
            scope_match="overlaps_query_range",
            measure_range=[[10, 12]],
        )
        context_only = make_result(
            "die-forelle-ku-002",
            "다른 구간에서만 적용되는 문장이다.",
            scope_match="other_range_context",
            measure_range=[[30, 32]],
        )

        units = build_atomic_evidence_units([primary, context_only])

        self.assertEqual(
            [
                (
                    item.unit_id,
                    item.evidence_label,
                    item.result_index,
                    item.text,
                )
                for item in units
            ],
            [("U1", "E1", 0, source)],
        )
        prompt_messages = build_atomic_claim_planner_messages(
            "die-forelle",
            "10-12",
            "어떻게 해야 할까?",
            units,
        )
        self.assertIn("[U1|E1] " + source, prompt_messages[1]["content"])
        self.assertNotIn("다른 구간에서만", prompt_messages[1]["content"])
        self.assertNotIn("die-forelle-ku-001", prompt_messages[1]["content"])
        self.assertNotIn("source-die-forelle-ku-001", prompt_messages[1]["content"])
        self.assertNotIn("전문가", prompt_messages[0]["content"])
        self.assertIn("검토 완료된 근거", prompt_messages[0]["content"])

    def test_selected_unit_is_rendered_whole_without_orphaning(self) -> None:
        text = "호흡을 먼저 준비한다. 이는 다음 진입을 안정시킨다."
        units = build_atomic_evidence_units(
            [make_result("die-forelle-ku-001", text)]
        )

        plan = parse_atomic_claim_plan(
            planner_json(units, "direct_answer"),
            units,
        )
        claims = build_atomic_source_claims(plan.selected_units)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, text)
        self.assertEqual(claims[0].unit_id, "U1")
        renderer_prompt = build_atomic_claim_renderer_messages(
            "어떻게 해야 할까?",
            claims,
        )[1]["content"]
        self.assertIn(text, renderer_prompt)

    def test_dynamic_schemas_require_total_planning_and_full_blocks(
        self,
    ) -> None:
        units = build_atomic_evidence_units(
            [
                make_result("die-forelle-ku-001", "첫 묶음이다."),
                make_result("die-forelle-ku-002", "둘째 묶음이다."),
            ]
        )

        planner = atomic_claim_planner_response_format(units)
        self.assertEqual(
            planner["schema"]["required"],
            ["selected_unit_ids", "units"],
        )
        selection_schema = planner["schema"]["properties"][
            "selected_unit_ids"
        ]
        self.assertEqual(selection_schema["minItems"], 1)
        self.assertEqual(selection_schema["maxItems"], 2)
        self.assertTrue(selection_schema["uniqueItems"])
        self.assertEqual(selection_schema["items"]["enum"], ["U1", "U2"])
        unit_schema = planner["schema"]["properties"]["units"]
        self.assertEqual(unit_schema["minItems"], 2)
        self.assertEqual(unit_schema["maxItems"], 2)
        self.assertEqual(
            unit_schema["items"]["properties"]["unit_id"]["enum"],
            ["U1", "U2"],
        )
        self.assertEqual(
            unit_schema["items"]["properties"]["relation"]["enum"],
            list(ATOMIC_PLAN_RELATIONS),
        )
        claims = build_atomic_source_claims(units[:1])
        renderer = atomic_claim_renderer_response_format(claims)
        block_schema = renderer["schema"]["properties"]["sentences"]
        self.assertEqual(block_schema["minItems"], 1)
        self.assertEqual(
            block_schema["items"]["properties"]["claim_id"]["enum"],
            ["C1"],
        )


class AtomicProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.units = build_atomic_evidence_units(
            [
                make_result("die-forelle-ku-001", "첫 묶음이다."),
                make_result("die-forelle-ku-002", "둘째 묶음이다."),
                make_result("die-forelle-ku-003", "셋째 묶음이다."),
            ]
        )

    def test_plan_accounts_for_every_unit_and_includes_four_relations(
        self,
    ) -> None:
        plan = parse_atomic_claim_plan(
            planner_json(
                self.units,
                "direct_answer",
                "redundant",
                "necessary_reason",
            ),
            self.units,
        )

        self.assertEqual(
            [item.unit_id for item in plan.selected_units],
            ["U1", "U3"],
        )
        self.assertIsNone(plan.validation_warning)

    def test_model_selection_remains_answerable_when_relations_exclude_all(
        self,
    ) -> None:
        plan = parse_atomic_claim_plan(
            planner_json(
                self.units,
                "background",
                "different_scope",
                "unrelated",
                selected_unit_ids=["U2"],
            ),
            self.units,
        )

        self.assertEqual(
            [item.unit_id for item in plan.selected_units],
            ["U2"],
        )
        self.assertEqual(
            plan.validation_warning,
            "Atomic claim planner selection/relation mismatch",
        )

    def test_plan_rejects_invalid_or_empty_model_selection(self) -> None:
        valid_payload = json.loads(
            planner_json(
                self.units,
                "direct_answer",
                "distinct_method",
                "unrelated",
            )
        )
        invalid_selections = (
            [],
            ["U9"],
            ["U1", "U1"],
            ["U2", "U1"],
        )
        for selected_unit_ids in invalid_selections:
            payload = {
                **valid_payload,
                "selected_unit_ids": selected_unit_ids,
            }
            with self.subTest(selected_unit_ids=selected_unit_ids):
                with self.assertRaises(GeneratedAnswerRejected):
                    parse_atomic_claim_plan(
                        json.dumps(payload),
                        self.units,
                    )

    def test_plan_rejects_missing_duplicate_reordered_and_invalid_units(
        self,
    ) -> None:
        invalid_plans = (
            '{"units":[{"unit_id":"U1","relation":"direct_answer"}]}',
            '{"units":['
            '{"unit_id":"U1","relation":"direct_answer"},'
            '{"unit_id":"U1","relation":"redundant"},'
            '{"unit_id":"U3","relation":"unrelated"}]}',
            '{"units":['
            '{"unit_id":"U2","relation":"direct_answer"},'
            '{"unit_id":"U1","relation":"redundant"},'
            '{"unit_id":"U3","relation":"unrelated"}]}',
            '{"units":['
            '{"unit_id":"U1","relation":"direct_answer"},'
            '{"unit_id":"U2","relation":"redundant"},'
            '{"unit_id":"U9","relation":"unrelated"}]}',
            '{"units":['
            '{"unit_id":"U1","relation":"wrong"},'
            '{"unit_id":"U2","relation":"redundant"},'
            '{"unit_id":"U3","relation":"unrelated"}]}',
            planner_json(
                self.units,
                "background",
                "different_scope",
                "unrelated",
            ),
            '{"claim_ids":["C1"]}',
            "not json",
        )
        for raw_plan in invalid_plans:
            with self.subTest(raw_plan=raw_plan):
                with self.assertRaises(GeneratedAnswerRejected):
                    parse_atomic_claim_plan(raw_plan, self.units)

    def test_render_preserves_complete_blocks_and_source_order(self) -> None:
        claims = build_atomic_source_claims(self.units[:2])
        rendered = parse_atomic_claim_render(
            json.dumps(
                {
                    "sentences": [
                        {"claim_id": "C1", "text": "첫 묶음입니다."},
                        {"claim_id": "C2", "text": "둘째 묶음입니다."},
                    ]
                },
                ensure_ascii=False,
            ),
            claims,
        )

        self.assertEqual(
            [item.text for item in rendered],
            ["첫 묶음입니다.", "둘째 묶음입니다."],
        )

    def test_render_rejects_internal_labels_schema_keys_and_json_text(
        self,
    ) -> None:
        claims = build_atomic_source_claims(self.units[:1])
        leaked_texts = (
            "첫 묶음입니다. C1",
            "첫 묶음입니다. U1",
            "첫 묶음입니다. E1",
            "첫 묶음입니다. claim_id",
            "첫 묶음입니다. claim_ids",
            "첫 묶음입니다. unit_id",
            "첫 묶음입니다. selected_unit_ids",
            "첫 묶음입니다. sentences",
            "첫 묶음입니다. direct_answer",
            '{"sentences":[]}',
            '[{"text":"첫 묶음입니다."}]',
        )
        for leaked_text in leaked_texts:
            raw = json.dumps(
                {
                    "sentences": [
                        {"claim_id": "C1", "text": leaked_text}
                    ]
                },
                ensure_ascii=False,
            )
            with self.subTest(leaked_text=leaked_text):
                with self.assertRaises(GeneratedAnswerRejected):
                    parse_atomic_claim_render(raw, claims)

    def test_legitimate_relation_word_is_not_treated_as_schema_leak(self) -> None:
        units = build_atomic_evidence_units(
            [
                make_result(
                    "die-forelle-ku-004",
                    "The relation between phrases is clear.",
                )
            ]
        )
        claims = build_atomic_source_claims(units)

        rendered = parse_atomic_claim_render(
            '{"sentences":[{"claim_id":"C1",'
            '"text":"The relation between phrases is clear."}]}',
            claims,
        )

        self.assertEqual(
            rendered[0].text,
            "The relation between phrases is clear.",
        )

    def test_render_rejects_reordered_or_duplicate_mapping(self) -> None:
        claims = build_atomic_source_claims(self.units[:2])
        for ids in (("C2", "C1"), ("C1", "C1")):
            raw = json.dumps(
                {
                    "sentences": [
                        {"claim_id": ids[0], "text": "첫 묶음입니다."},
                        {"claim_id": ids[1], "text": "둘째 묶음입니다."},
                    ]
                },
                ensure_ascii=False,
            )
            with self.subTest(ids=ids):
                with self.assertRaises(GeneratedAnswerRejected):
                    parse_atomic_claim_render(raw, claims)


class AtomicFidelityTests(unittest.TestCase):
    def test_deterministic_formal_polite_edit_is_allowed(self) -> None:
        validate_atomic_render_fidelity(
            "끌어 부르는 것이 좋다.",
            "끌어 부르는 것이 좋습니다.",
        )

    def test_meaning_number_and_polarity_changes_are_rejected(self) -> None:
        changed_pairs = (
            (
                "노래하는 사람은 이를 이해하고 피아노가 만드는 "
                "긴장감을 이용해 가사를 표현하며, 전환이 느껴지도록 "
                "주의해서 부르는 것이 좋다.",
                "노래하는 사람은 이를 이해하고 피아노가 만드는 "
                "긴장감을 이용해 가사를 표현하며, 전환이 느껴지도록 "
                "주의해서 끌어내리는 것이 좋습니다.",
            ),
            (
                "박자와 조성을 바꾸어 곡의 분위기를 전환한다.",
                "작곡가는 추위를 표현하기 위해 박자와 조성을 바꿉니다.",
            ),
            (
                "과도하게 부르지 않는다.",
                "과도하게 부릅니다.",
            ),
            (
                "47-48마디에서 바뀐다.",
                "49-50마디에서 바뀝니다.",
            ),
        )
        for source, rendered in changed_pairs:
            with self.subTest(rendered=rendered):
                with self.assertRaises(GeneratedAnswerRejected):
                    validate_atomic_render_fidelity(source, rendered)


class AtomicCompositionTests(unittest.TestCase):
    def test_uniform_two_stage_path_keeps_complete_units_hidden(
        self,
    ) -> None:
        results = [
            make_result(
                "die-forelle-ku-001",
                "가능성으로 보는 것이 일반적이다. 표현하는 것이 좋다.",
            ),
            make_result(
                "die-forelle-ku-002",
                "밝은 음색도 중요하다.",
            ),
        ]
        units = build_atomic_evidence_units(results)
        responses = iter(
            (
                planner_json(
                    units,
                    "direct_answer",
                    "distinct_method",
                ),
                json.dumps(
                    {
                        "sentences": [
                            {
                                "claim_id": "C1",
                                "text": (
                                    "가능성으로 보는 것이 일반적입니다. "
                                    "표현하는 것이 좋습니다."
                                ),
                            },
                            {
                                "claim_id": "C2",
                                "text": "밝은 음색도 중요합니다.",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
            )
        )
        calls = []

        def fake_generate(model_path, messages, settings, **kwargs):
            calls.append((model_path, messages, settings, kwargs))
            return next(responses)

        final, warning = generate_atomic_grounded_answer(
            model_path="/models/qwen.gguf",
            llm_settings={"max_tokens": 768},
            piece="die-forelle",
            measures="",
            question="어떻게 표현해야 할까?",
            results=results,
            measure_ranges=[],
            generate_fn=fake_generate,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call[3]["response_format"]["type"] for call in calls],
            ["json_object", "json_object"],
        )
        self.assertTrue(all(call[2]["max_tokens"] == 1024 for call in calls))
        self.assertEqual(
            final,
            "가능성으로 보는 것이 일반적입니다. 표현하는 것이 좋습니다. "
            "밝은 음색도 중요합니다.",
        )
        self.assertIsNone(warning)
        for hidden in (
            "C1",
            "U1",
            "E1",
            "claim_id",
            "sentences",
            "die-forelle-ku-001",
        ):
            self.assertNotIn(hidden, final)

    def test_relation_mismatch_returns_selected_answer_with_warning(self) -> None:
        result = make_result(
            "die-forelle-ku-001",
            "가장 직접 도움이 되는 근거이다.",
        )
        units = build_atomic_evidence_units([result])
        responses = iter(
            (
                planner_json(
                    units,
                    "unrelated",
                    selected_unit_ids=["U1"],
                ),
                '{"sentences":[{"claim_id":"C1",'
                '"text":"가장 직접 도움이 되는 근거입니다."}]}',
            )
        )

        final, warning = generate_atomic_grounded_answer(
            model_path="/models/qwen.gguf",
            llm_settings={},
            piece="die-forelle",
            measures="",
            question="무엇이 도움이 될까?",
            results=[result],
            measure_ranges=[],
            generate_fn=lambda *_args, **_kwargs: next(responses),
        )

        self.assertEqual(final, "가장 직접 도움이 되는 근거입니다.")
        self.assertEqual(
            warning,
            "Atomic claim planner selection/relation mismatch",
        )

    def test_semantic_rejection_returns_rendered_text_as_warning_not_json(
        self,
    ) -> None:
        result = make_result(
            "die-forelle-ku-001",
            "국소적인 가능성이다.",
        )
        units = build_atomic_evidence_units([result])
        responses = iter(
            (
                planner_json(units, "direct_answer"),
                '{"sentences":[{"claim_id":"C1",'
                '"text":"국소적인 가능성입니다."}]}',
            )
        )

        with mock.patch.object(
            answer,
            "finalize_grounded_generated_answer",
            side_effect=GeneratedAnswerRejected("semantic warning"),
        ):
            final, warning = generate_atomic_grounded_answer(
                model_path="/models/qwen.gguf",
                llm_settings={},
                piece="die-forelle",
                measures="",
                question="무엇일까?",
                results=[result],
                measure_ranges=[],
                generate_fn=lambda *_args, **_kwargs: next(responses),
            )

        self.assertEqual(final, "국소적인 가능성입니다.")
        self.assertEqual(warning, "semantic warning")
        self.assertNotIn("sentences", final)

    def test_fidelity_drift_returns_answer_with_review_warning(self) -> None:
        source = (
            "노래하는 사람은 이를 이해하고 피아노가 만드는 긴장감을 "
            "이용해 가사를 표현하며, 전환이 느껴지도록 주의해서 "
            "부르는 것이 좋다."
        )
        rendered = (
            "노래하는 사람은 이를 이해하고 피아노가 만드는 긴장감을 "
            "이용해 가사를 표현하며, 전환이 느껴지도록 주의해서 "
            "끌어내리는 것이 좋습니다."
        )
        result = make_result("die-forelle-ku-001", source)
        units = build_atomic_evidence_units([result])
        responses = iter(
            (
                planner_json(units, "direct_answer"),
                json.dumps(
                    {
                        "sentences": [
                            {"claim_id": "C1", "text": rendered}
                        ]
                    },
                    ensure_ascii=False,
                ),
            )
        )

        final, warning = generate_atomic_grounded_answer(
            model_path="/models/qwen.gguf",
            llm_settings={},
            piece="die-forelle",
            measures="",
            question="어떻게 표현해야 할까?",
            results=[result],
            measure_ranges=[],
            generate_fn=lambda *_args, **_kwargs: next(responses),
        )

        self.assertEqual(final, rendered)
        self.assertIsNotNone(warning)
        self.assertIn("changed reviewed source wording", warning)

    def test_runtime_default_resolves_patched_generation_backend(self) -> None:
        result = make_result("die-forelle-ku-001", "검토된 문장이다.")
        units = build_atomic_evidence_units([result])
        with mock.patch.object(
            answer,
            "generate",
            side_effect=(
                planner_json(units, "direct_answer"),
                '{"sentences":[{"claim_id":"C1",'
                '"text":"검토된 문장입니다."}]}',
            ),
        ) as backend:
            final, warning = generate_atomic_grounded_answer(
                model_path="/models/qwen.gguf",
                llm_settings={},
                piece="die-forelle",
                measures="",
                question="무엇일까?",
                results=[result],
                measure_ranges=[],
            )

        self.assertEqual(backend.call_count, 2)
        self.assertEqual(final, "검토된 문장입니다.")
        self.assertIsNone(warning)

    def test_invalid_planner_is_hard_error_without_renderer_call(self) -> None:
        calls = []

        def invalid_generate(*_args, **_kwargs):
            calls.append(True)
            return "not json"

        with self.assertRaises(GeneratedAnswerRejected):
            generate_atomic_grounded_answer(
                model_path="/models/qwen.gguf",
                llm_settings={},
                piece="die-forelle",
                measures="",
                question="무엇일까?",
                results=[
                    make_result(
                        "die-forelle-ku-001",
                        "검토된 문장이다.",
                    )
                ],
                measure_ranges=[],
                generate_fn=invalid_generate,
            )

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
