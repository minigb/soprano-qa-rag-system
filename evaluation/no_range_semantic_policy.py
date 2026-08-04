"""Curated policy for semantically self-contained no-range evaluation.

The dataset's ``inference_scope=measure_range`` field describes the confirmed
scope of linked knowledge units.  It does not say whether a location-neutral
question remains understandable when the user has not selected score
measures.  This module makes that separate evaluation decision explicit and
exhaustive without creating another result JSON file.
"""

from __future__ import annotations

from typing import Final


COHORT_NAME: Final = "measure_scoped_question_no_range"
SHADOW_CASE_KIND: Final = "measure_annotated_no_range_semantic"
REPORTED_CASE_KIND: Final = "reported_no_range_regression"


# ``required_any_knowledge_unit_ids`` defines semantic retrieval success: at
# least one listed finalized KU must be retrieved.  ``supporting`` KUs are
# valid additional answer authority but are not independently required.
NO_RANGE_SEMANTIC_CASES: Final = {
    "kim-die-forelle-02": {
        "piece_id": "die-forelle",
        "required_any_knowledge_unit_ids": ("die-forelle-ku-002",),
        "supporting_knowledge_unit_ids": ("die-forelle-ku-003",),
    },
    "kim-die-forelle-07": {
        "piece_id": "die-forelle",
        "required_any_knowledge_unit_ids": ("die-forelle-ku-010",),
        "supporting_knowledge_unit_ids": (),
    },
    "yeon-die-forelle-04": {
        "piece_id": "die-forelle",
        "required_any_knowledge_unit_ids": ("die-forelle-ku-010",),
        "supporting_knowledge_unit_ids": ("die-forelle-ku-015",),
    },
    "kim-in-flowery-clouds-01": {
        "piece_id": "in-flowery-clouds",
        "required_any_knowledge_unit_ids": ("in-flowery-clouds-ku-001",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-in-flowery-clouds-02": {
        "piece_id": "in-flowery-clouds",
        "required_any_knowledge_unit_ids": ("in-flowery-clouds-ku-002",),
        "supporting_knowledge_unit_ids": ("in-flowery-clouds-ku-004",),
    },
    "kim-in-flowery-clouds-07": {
        "piece_id": "in-flowery-clouds",
        "required_any_knowledge_unit_ids": ("in-flowery-clouds-ku-008",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-in-flowery-clouds-09": {
        "piece_id": "in-flowery-clouds",
        "required_any_knowledge_unit_ids": ("in-flowery-clouds-ku-010",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-in-flowery-clouds-11": {
        "piece_id": "in-flowery-clouds",
        "required_any_knowledge_unit_ids": ("in-flowery-clouds-ku-012",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-02": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-003",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-03": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-004",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-05": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-006",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-06": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-008",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-07": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-009",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-09": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-011",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-10": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-012",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-11": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-013",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-12": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-014",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-13": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-007",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-14": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-015",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-15": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-016",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-16": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-017",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-17": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-018",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-la-capinera-18": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-019",),
        "supporting_knowledge_unit_ids": (),
    },
    "yeon-la-capinera-12": {
        "piece_id": "la-capinera",
        "required_any_knowledge_unit_ids": ("la-capinera-ku-029",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-nella-fantasia-03": {
        "piece_id": "nella-fantasia",
        "required_any_knowledge_unit_ids": ("nella-fantasia-ku-002",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-nella-fantasia-07": {
        "piece_id": "nella-fantasia",
        "required_any_knowledge_unit_ids": ("nella-fantasia-ku-006",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-nella-fantasia-09": {
        "piece_id": "nella-fantasia",
        "required_any_knowledge_unit_ids": ("nella-fantasia-ku-008",),
        "supporting_knowledge_unit_ids": ("nella-fantasia-ku-009",),
    },
    "kim-nella-fantasia-10": {
        "piece_id": "nella-fantasia",
        "required_any_knowledge_unit_ids": ("nella-fantasia-ku-010",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-03": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-004",),
        "supporting_knowledge_unit_ids": ("una-voce-poco-fa-ku-005",),
    },
    "kim-una-voce-poco-fa-05": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-007",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-07": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-010",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-09": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": (
            "una-voce-poco-fa-ku-012",
            "una-voce-poco-fa-ku-013",
            "una-voce-poco-fa-ku-014",
        ),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-10": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-015",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-11": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-016",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-13": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-017",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-14": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-019",),
        "supporting_knowledge_unit_ids": ("una-voce-poco-fa-ku-018",),
    },
    "kim-una-voce-poco-fa-15": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-020",),
        "supporting_knowledge_unit_ids": (),
    },
    "kim-una-voce-poco-fa-20": {
        "piece_id": "una-voce-poco-fa",
        "required_any_knowledge_unit_ids": ("una-voce-poco-fa-ku-005",),
        "supporting_knowledge_unit_ids": ("una-voce-poco-fa-ku-004",),
    },
}


NO_RANGE_SEMANTIC_EXCLUSIONS: Final = {
    "kim-in-flowery-clouds-05": {
        "piece_id": "in-flowery-clouds",
        "reason": (
            "'첫 소리' is ambiguous with the song opening, while the linked "
            "KU concerns the later high phrase at measures 19-21."
        ),
    },
    "kim-in-flowery-clouds-08": {
        "piece_id": "in-flowery-clouds",
        "reason": (
            "The question does not identify the visible accompaniment mark "
            "as tremolo and is not answerable without score or range context."
        ),
    },
    "kim-la-capinera-01": {
        "piece_id": "la-capinera",
        "reason": (
            "'시작' is ambiguous between the opening and reprise, and one "
            "linked reprise case is already excluded for a lyric-anchor "
            "mismatch."
        ),
    },
    "kim-la-capinera-04": {
        "piece_id": "la-capinera",
        "reason": (
            "The question does not identify which dim. marking or name the "
            "associated 'o mia gentil' phrase."
        ),
    },
    "kim-la-capinera-08": {
        "piece_id": "la-capinera",
        "reason": (
            "The question omits the ornament and high-leap feature needed "
            "to identify the intended local KU."
        ),
    },
}


REPORTED_REGRESSION_FORMULATIONS: Final = {
    "kim-in-flowery-clouds-01": (
        {
            "case_suffix": "reported-no-range-01",
            "question": (
                "악보의 페르마타 표시가 음원에서 들리는 음 길이와 "
                "일치하지 않을 때는 어떻게 해야 할까?"
            ),
        },
    ),
    "kim-in-flowery-clouds-07": (
        {
            "case_suffix": "reported-no-range-01",
            "question": "이 곡에서 박자가 여러번 바뀌는 이유는 무엇인가?",
        },
    ),
    "kim-una-voce-poco-fa-11": (
        {
            "case_suffix": "reported-no-range-01",
            "question": "이 곡에 간주가 너무 길어서 그동안 뭘 해야할지 모르겠다",
        },
    ),
}


EXPECTED_INCLUDED_PER_PIECE: Final = {
    "die-forelle": 3,
    "in-flowery-clouds": 5,
    "la-capinera": 16,
    "nella-fantasia": 4,
    "una-voce-poco-fa": 10,
}
