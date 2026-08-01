#!/usr/bin/env python3
"""Calibrate the dual-frame semantic judge against fixed risk controls.

The controls are deliberately small and discriminative.  Their references
are rebuilt from the versioned expert review files on every new run, while
each model attempt is checkpointed by atomic replacement.  Some deliberately
ambiguous controls allow ``human_review`` as the safe outcome; unsafe controls
must never become an automatic pass, and every control must emit its intended
semantic risk signal.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import run_question_evaluation as evaluator


SCHEMA_VERSION = evaluator.CALIBRATION_SCHEMA_VERSION
CONTROL_CONTRACT_VERSION = evaluator.CALIBRATION_CONTRACT_VERSION
MINIMUM_ACCURACY = 1.0
EXPECTED_CONTROL_COUNT = evaluator.CALIBRATION_EXPECTED_CONTROL_COUNT
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "judge_calibration_results.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "soprano-qa-dataset"
QWEN3_4B_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-4B"
)

JudgeFunction = Callable[[list[dict[str, str]]], str]
CheckpointFunction = Callable[[], None]


def runtime_versions() -> dict[str, str]:
    """Return judge-runtime versions that affect tokenization/inference."""

    versions = {"python": sys.version.split()[0]}
    for distribution in ("torch", "transformers", "tokenizers"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


CONTROL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "good_retrieved_expert_supplement",
        "description": "필수 원문 답변과 검색된 전문가 보충 조언을 함께 충족",
        "source_id": "kim-die-forelle-01",
        "measure_range": None,
        "candidate_answer": (
            "《Die Forelle》는 슈베르트가 처음 작곡한 뒤 여러 차례 "
            "개정하고 즉흥적으로 기보하는 등 여러 버전의 악보 자료를 "
            "남겼기 때문에, 악보에 따라 전주가 있거나 없을 수 있다. "
            "두 경우 모두 가능하지만 일반적으로는 전주가 있는 악보를 "
            "많이 사용한다. 전주 유무가 다른 악보는 원본과 비교해 "
            "보는 것이 좋다."
        ),
        "expectation": {
            "verdict": "pass",
            "allowed_verdicts": ["pass"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "supplement_only_cannot_replace_target",
        "description": "검색된 전문가 보충 조언만으로 필수 원문을 대체할 수 없음",
        "source_id": "kim-die-forelle-01",
        "measure_range": None,
        "candidate_answer": (
            "전주 유무가 다른 악보는 원본과 비교해 보는 것이 좋다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_verdicts": ["fail", "human_review"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
                "unrelated",
            ],
            "required_signal": (
                "target_incomplete_or_s_substitution_guard"
            ),
        },
    },
    {
        "control_id": "good_second_beat_accent",
        "description": "명확히 충실한 두 번째 박 악센트 답변",
        "source_id": "kim-die-forelle-02",
        "measure_range": [2, 27],
        "candidate_answer": (
            "두 번째 박의 악센트는 흐르는 물에서 송어가 튀어 오르며 "
            "노는 생동감을 나타낸다고 볼 수 있다. 악센트가 위아래 "
            "성부를 오가며 반복되는 움직임도 그 이미지를 강화한다."
        ),
        "expectation": {
            # Two historical sentences are optional background. The mixed
            # second sentence is fully covered, including motion/repetition
            # and effect, so this complete core answer should pass.
            "verdict": "pass",
            "allowed_verdicts": ["pass"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "minor_omission_lerbe_e_melisma",
        "description": "핵심 e 연음은 맞고 d 발음 조언만 생략한 답변",
        "source_id": "kim-la-capinera-13",
        "measure_range": [80, 80],
        "candidate_answer": "이어지는 e 모음들을 부드럽게 연음하면 된다.",
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["pass", "human_review"],
            # The d-placement detail is secondary but genuinely absent.
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
            ],
            "required_signal": "none",
            "forbidden_covered_reference_markers": [
                "d 발음",
                "d발음",
            ],
        },
    },
    {
        "control_id": "contradictory_fermata_mor",
        "description": "페르마타 음가와 mor 악센트를 모두 뒤집은 답변",
        "source_id": "kim-la-capinera-11",
        "measure_range": [67, 69],
        "candidate_answer": (
            "페르마타도 원래 음가보다 늘이지 말고, l’amor의 mor도 "
            "강조하지 않는다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_relationships": [
                "minor_omission",
                # Either frame may conservatively identify the negated
                # claims as unsupported/unreliable rather than link them as
                # direct contradictions. The aggregate must still fail and
                # at least one frame must emit the contradiction signal.
                "materially_unreliable",
                "contradictory",
            ],
            "required_signal": "contradiction",
            "required_signal_frames": "any",
        },
    },
    {
        "control_id": "unsupported_mandatory_training_regimen",
        "description": "근거 없는 의무적 30분 발성 훈련을 추가한 답변",
        "source_id": "kim-die-forelle-03",
        "measure_range": None,
        "candidate_answer": (
            "느리게 연습하고 음정 없이 발음만 연습할 수 있다. 여기에 "
            "매일 30분씩 립트릴과 복식호흡을 반드시 병행해야 한다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_relationships": [
                "materially_incomplete",
                "materially_unreliable",
            ],
            "required_signal": "unsupported_mandatory_advice",
            "required_signal_frames": "any",
        },
    },
    {
        "control_id": "cautious_unresolved_meter_change",
        "description": "음악적 핵심과 미해결 전환 시점 불확실성을 보존한 답변",
        "source_id": "kim-in-flowery-clouds-07",
        "measure_range": [39, 49],
        "candidate_answer": (
            "박자·조성·반주 리듬의 변화가 ‘추위와 주림에 시달리어’의 "
            "고난과 겨울 이미지를 만들며 분위기를 전환한다. 다만 "
            "자료마다 조성 변화 시점 설명이 달라 정확한 시점은 "
            "단정하기 어렵다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_answer_quality_verdicts": [
                "pass",
                "human_review",
            ],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "overconfident_fermata_original",
        "description": "근거가 미해결인 원본 페르마타 주장을 단정한 답변",
        "source_id": "kim-in-flowery-clouds-01",
        "measure_range": [1, 2],
        "candidate_answer": (
            "원본에 가까운 악보에는 반드시 페르마타가 없으므로 "
            "페르마타 없이 불러야 한다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_relationships": [
                "minor_omission",
                "materially_incomplete",
                "contradictory",
            ],
            "required_signal": "contradiction",
            "required_signal_frames": "any",
        },
    },
    {
        "control_id": "overconfident_unresolved_meter_simultaneity",
        "description": "시점 충돌을 언급한 뒤 동시 변화를 다시 단정한 답변",
        "source_id": "kim-in-flowery-clouds-07",
        "measure_range": [39, 49],
        "candidate_answer": (
            "자료마다 조성 변화 시점 설명이 다르다. 하지만 박자와 "
            "조성은 반드시 같은 순간에 동시에 바뀐다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["human_review", "fail"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
                "materially_unreliable",
                "contradictory",
            ],
            # Exact bidirectional claim links are preferred. If the judge
            # detects the unresolved-source contradiction but cannot anchor
            # it consistently, the validator preserves a review-only
            # warning; that still demonstrates risk detection and can never
            # produce a pass.
            "required_signal": "contradiction_or_review_warning",
            "required_signal_frames": "any",
        },
    },
    {
        "control_id": "good_early_echo_contrast",
        "description": "앞 ff·pp 범위의 두 블록 메아리 대비를 보존한 답변",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "이 대목에서는 ff 부분과 pp 부분을 두 블록으로 대비해 "
            "산속에서 새소리가 메아리처럼 울리는 느낌을 표현한다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["human_review"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "good_early_shared_vocabulary_paraphrase",
        "description": "뒤 범위와 어휘를 공유하지만 앞 범위의 두 덩어리 대비인 답변",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "ff와 pp를 두 덩어리로 대비해 산새의 반향 효과가 "
            "느껴지게 한다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["human_review"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "good_early_negated_correction",
        "description": "뒤 범위의 빠른 교대를 명시적으로 부정하고 앞 대비를 설명",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "ff와 pp는 빠르게 번갈아 나오는 것이 아니라 두 블록으로 "
            "대비되어 산속 새소리의 메아리를 표현한다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["human_review"],
            "allowed_relationships": list(evaluator.RELATIONSHIPS),
            "required_signal": "range_negation",
        },
    },
    {
        "control_id": "wrong_later_dynamics_projected_to_early_echo",
        "description": "뒤 범위의 빠른 교대·경쾌함을 앞 범위에 투사한 답변",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "이 대목의 ff와 pp는 한 프레이즈 안에서 빠르게 번갈아 "
            "나오며 산새의 경쾌한 움직임을 표현한다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
                "contradictory",
                "materially_unreliable",
            ],
            "required_signal": "wrong_measure_application",
        },
    },
    {
        "control_id": "wrong_later_dynamics_paraphrased_at_early_echo",
        "description": "뒤 범위의 빠른 교대를 다른 어휘로 앞 범위에 투사",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "강약이 잇달아 재빨리 교차하며 작은 새의 발랄한 몸짓을 "
            "보여 준다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["fail", "human_review"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
                "contradictory",
                "materially_unreliable",
            ],
            "required_signal": "range_risk",
        },
    },
    {
        "control_id": "good_later_rapid_alternation",
        "description": "뒤 ff·pp 범위의 빠른 교대와 경쾌함을 보존한 답변",
        "source_id": "kim-la-capinera-17",
        "measure_range": [117, 124],
        "candidate_answer": (
            "이 대목의 ff와 pp는 한 프레이즈 안에서 빠르게 번갈아 "
            "나오며 산새의 경쾌한 움직임과 반향 효과를 표현한다."
        ),
        "expectation": {
            "verdict": "pass",
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "wrong_early_echo_projected_to_later_dynamics",
        "description": "앞 범위의 두 블록 대비를 뒤 빠른 교대 범위에 투사",
        "source_id": "kim-la-capinera-17",
        "measure_range": [117, 124],
        "candidate_answer": (
            "이 대목에서는 ff 부분과 pp 부분을 두 블록으로 나누어 "
            "산속에서 새소리가 메아리처럼 울리는 대비를 표현한다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
                "materially_incomplete",
                "contradictory",
                "materially_unreliable",
            ],
            "required_signal": "wrong_measure_application",
        },
    },
    {
        "control_id": "unrelated_same_composer_background_only",
        "description": "같은 작곡가 배경만 말하고 질문 핵심을 답하지 않는 경우",
        "source_id": "kim-die-forelle-02",
        "measure_range": [2, 27],
        "candidate_answer": "슈베르트는 가곡의 왕으로 불린다.",
        "expectation": {
            "verdict": "fail",
            "allowed_relationships": [
                "materially_incomplete",
                "materially_unreliable",
                "unrelated",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "good_second_beat_semantic_paraphrase",
        "description": "어휘가 달라도 두 번째 박 악센트의 핵심을 보존한 답변",
        "source_id": "kim-die-forelle-02",
        "measure_range": [2, 27],
        "candidate_answer": (
            "두 번째 박의 강조는 물 위로 솟구쳐 노는 송어의 "
            "생동감을 그린다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["human_review"],
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
            ],
            "required_signal": "none",
            "forbidden_covered_reference_markers": [
                "높은음자리표와 낮은음자리표",
            ],
        },
    },
    {
        "control_id": "wrong_early_unseen_dense_alternation",
        "description": "새 어휘로 뒤의 빠른 교대를 앞 범위에 투사",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "강한 소리와 여린 소리가 한 프레이즈에서 촘촘하게 "
            "교대하며 작은 새의 발랄한 몸짓을 그린다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["fail", "human_review"],
            "allowed_relationships": list(evaluator.RELATIONSHIPS),
            "required_signal": "range_risk",
        },
    },
    {
        "control_id": "wrong_early_mixed_final_assertion",
        "description": "앞 설명 뒤에 실제 빠른 교대를 최종적으로 단정",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "두 블록으로 대비되기도 하지만 실제로는 강약이 쉴 새 "
            "없이 교대한다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_relationships": list(evaluator.RELATIONSHIPS),
            "required_signal": "wrong_measure_application",
        },
    },
    {
        "control_id": "wrong_early_double_negation",
        "description": "이중 부정으로 뒤의 빠른 교대를 긍정",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "ff와 pp가 빠르게 번갈아 나오지 않는 게 아니라고 볼 수 "
            "있다."
        ),
        "expectation": {
            "verdict": "fail",
            "allowed_verdicts": ["fail", "human_review"],
            "allowed_relationships": list(evaluator.RELATIONSHIPS),
            "required_signal": "range_uncertainty",
        },
    },
    {
        "control_id": "good_early_natural_negation",
        "description": "자연스러운 부정 뒤 앞 범위의 두 블록 대비를 설명",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "ff와 pp가 빠르게 번갈아 나오는 대목은 아니다. ff와 pp는 두 "
            "블록으로 대비해 메아리를 표현한다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["human_review"],
            "allowed_relationships": list(evaluator.RELATIONSHIPS),
            "required_signal": "range_negation",
        },
    },
    {
        "control_id": "wrong_later_unseen_separated_blocks",
        "description": "새 어휘로 앞의 분리된 블록을 뒤 범위에 투사",
        "source_id": "kim-la-capinera-17",
        "measure_range": [117, 124],
        "candidate_answer": (
            "강한 부분과 여린 부분을 각각 묶어 멀리 떨어뜨려 "
            "제시한다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["fail", "human_review"],
            "allowed_relationships": list(evaluator.RELATIONSHIPS),
            "required_signal": "range_risk",
        },
    },
    {
        "control_id": "good_later_rapid_with_two_parts_contrast",
        "description": "두 부분이라는 공유 어휘가 있어도 뒤의 빠른 교대를 설명",
        "source_id": "kim-la-capinera-17",
        "measure_range": [117, 124],
        "candidate_answer": (
            "ff와 pp를 빠르게 번갈아 내되 두 부분의 강약을 명확히 "
            "대비해 산새의 경쾌함을 표현한다."
        ),
        "expectation": {
            "verdict": "pass",
            "allowed_relationships": [
                "equivalent",
                "minor_omission",
            ],
            "required_signal": "none",
        },
    },
    {
        "control_id": "uncertain_early_range_projection",
        "description": "다른 범위의 빠른 교대를 가능성으로만 제시한 경우",
        "source_id": "kim-la-capinera-17",
        "measure_range": [51, 58],
        "candidate_answer": (
            "이 대목이 두 블록의 메아리 대비인지, 강약이 빠르게 "
            "교대하는 효과인지 확실하지 않다."
        ),
        "expectation": {
            "verdict": "human_review",
            "allowed_verdicts": ["human_review", "fail"],
            "allowed_relationships": list(evaluator.RELATIONSHIPS),
            "required_signal": "range_uncertainty",
        },
    },
)


class CalibrationInputError(ValueError):
    """Raised when calibration inputs do not meet the fixed contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_qwen3_4b_path() -> Path:
    """Resolve the locally cached immutable Qwen3-4B snapshot, if present."""

    revision_file = QWEN3_4B_CACHE_ROOT / "refs" / "main"
    if revision_file.is_file():
        revision = revision_file.read_text(encoding="utf-8").strip()
        if revision:
            return QWEN3_4B_CACHE_ROOT / "snapshots" / revision
    return QWEN3_4B_CACHE_ROOT


def validate_qwen3_4b_checkpoint(model_path: Path) -> dict[str, Any]:
    """Validate that ``model_path`` is the Transformers Qwen3-4B model."""

    if not model_path.is_dir():
        raise CalibrationInputError(
            f"Qwen3-4B judge directory not found: {model_path}"
        )
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise CalibrationInputError(
            f"Transformers config.json not found in {model_path}"
        )
    try:
        config = evaluator.load_json(config_path)
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationInputError(
            f"Cannot read Qwen3-4B config: {error}"
        ) from error

    expected = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
    }
    mismatches = {
        name: {"expected": value, "actual": config.get(name)}
        for name, value in expected.items()
        if config.get(name) != value
    }
    architectures = config.get("architectures") or []
    if "Qwen3ForCausalLM" not in architectures:
        mismatches["architectures"] = {
            "expected": ["Qwen3ForCausalLM"],
            "actual": architectures,
        }
    if mismatches:
        raise CalibrationInputError(
            "Judge checkpoint is not the expected Qwen3-4B architecture: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return {
        "declared_model": "Qwen/Qwen3-4B",
        "config_path": str(config_path.resolve()),
        "architecture_checks": expected,
    }


def _select_question_and_case(
    questions: Sequence[Mapping[str, Any]],
    *,
    source_id: str,
    measure_range: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        question = next(
            item for item in questions if item["source_id"] == source_id
        )
    except StopIteration as error:
        raise CalibrationInputError(
            f"Calibration source is absent from inventory: {source_id}"
        ) from error

    matches = [
        case
        for case in question["inference_runs"]
        if case["inference_input"]["measure_range"] == measure_range
    ]
    if len(matches) != 1:
        raise CalibrationInputError(
            f"{source_id}: expected one inference case for {measure_range}, "
            f"found {len(matches)}"
        )
    return deepcopy(question), deepcopy(matches[0])


def _expert_evidence(
    question: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    trusted_expert_catalog: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> list[dict[str, Any]]:
    applicable_ids = set(
        case["inference_input"]["range_applicable_knowledge_unit_ids"]
    )
    evidence = []
    for unit in question["authoritative_reference"][
        "linked_knowledge_units"
    ]:
        if unit["knowledge_unit_id"] not in applicable_ids:
            continue
        record = (
            trusted_expert_catalog.get(unit["knowledge_unit_id"])
            if trusted_expert_catalog is not None
            else None
        )
        if record is not None:
            from soprano_qa.answer import (
                expert_prompt_factuality_material,
            )
            from soprano_qa.retrieval import (
                measure_scope_match,
                scope_evidence_role,
            )

            selected_range = case["inference_input"]["measure_range"]
            selected_ranges = (
                [selected_range]
                if selected_range is not None
                else []
            )
            scope_match = measure_scope_match(
                dict(record),
                selected_ranges,
            )
            evidence.append(
                {
                    "id": record["id"],
                    "kind": "expert",
                    "evidence_type": "expert_annotation",
                    "piece_id": record["piece"],
                    "source_ids": list(record["source_ids"]),
                    "measure_ranges": deepcopy(
                        record.get("measure_range", [])
                    ),
                    "scope_match": scope_match,
                    "generation_role": scope_evidence_role(scope_match),
                    "selected_range_claim_authority": (
                        scope_match == "overlaps_query_range"
                        if selected_range is not None
                        else None
                    ),
                    "in_requested_scope": (
                        scope_match == "overlaps_query_range"
                        if selected_range is not None
                        else scope_match
                        in {
                            "general_evidence",
                            "local_example",
                            "unscoped_pending_review",
                            "unspecified_scope",
                        }
                    ),
                    "text": record["answer"],
                    "rewrite_status": record["rewrite_status"],
                    "rewrite_notes": record.get("rewrite_notes", ""),
                    "measure_status": record["measure_status"],
                    "measure_notes": record.get("measure_notes", ""),
                    "retrieval_review_warning": record.get(
                        "retrieval_review_warning",
                        "",
                    ),
                    "generation_expert_authority": (
                        expert_prompt_factuality_material(
                            dict(record),
                            selected_ranges,
                        )
                    ),
                }
            )
            continue
        evidence.append(
            {
                "id": unit["knowledge_unit_id"],
                "kind": "expert",
                "evidence_type": "expert_annotation",
                "piece_id": question["piece_id"],
                "source_ids": list(unit["source_ids"]),
                "measure_ranges": deepcopy(unit["measure_ranges"]),
                "scope_match": "calibration_reference",
                "in_requested_scope": True,
                "text": unit["answer"],
                "rewrite_status": unit["rewrite_status"],
                "measure_status": unit["measure_status"],
                "retrieval_review_warning": (
                    "rewrite_review_pending"
                    if unit["rewrite_status"] != "ready"
                    else ""
                ),
            }
        )
    if not evidence:
        raise CalibrationInputError(
            f"{question['source_id']}: no range-applicable expert evidence"
        )
    return evidence


def build_controls(
    dataset_root: Path,
    *,
    trusted_expert_catalog: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Build immutable control packets from authoritative dataset records."""

    if trusted_expert_catalog is None:
        trusted_expert_catalog = (
            evaluator.load_trusted_expert_evidence_catalog(
                PROJECT_ROOT / "data" / "corpus.json"
            )
        )
    questions, input_paths = evaluator.load_evaluation_questions(
        dataset_root
    )
    controls = []
    seen_ids: set[str] = set()
    for definition in CONTROL_DEFINITIONS:
        control_id = definition["control_id"]
        if control_id in seen_ids:
            raise CalibrationInputError(
                f"Duplicate control_id: {control_id}"
            )
        seen_ids.add(control_id)
        question, case = _select_question_and_case(
            questions,
            source_id=definition["source_id"],
            measure_range=definition["measure_range"],
        )
        evidence = _expert_evidence(
            question,
            case,
            trusted_expert_catalog=trusted_expert_catalog,
        )
        from soprano_qa.answer import (
            build_review_disclosure_from_records,
        )

        disclosure_records = [
            trusted_expert_catalog[item["id"]]
            for item in evidence
            if item["id"] in trusted_expert_catalog
        ]
        review_disclosure = build_review_disclosure_from_records(
            disclosure_records,
            neutralize_specific_measure_locators=any(
                item.get("generation_role") == "local_example"
                for item in evidence
            ),
        )
        raw_generated_answer = definition["candidate_answer"]
        if review_disclosure:
            raw_generated_answer = (
                review_disclosure
                + "\n\n"
                + raw_generated_answer
            )
        generated_answer = {
            "answer": raw_generated_answer,
            "piece_id": question["piece_id"],
            "measure_range": deepcopy(definition["measure_range"]),
            "generation_mode": "llm",
            "answer_basis": "retrieved_evidence",
            "evidence": evidence,
            "diagnostics": {"target_source_grounded": True},
        }
        case["generated_answer"] = generated_answer
        packet = evaluator._reference_packet(
            question,
            case,
            trusted_expert_catalog=trusted_expert_catalog,
        )
        range_requirement = evaluator.range_scope_requirement(packet)
        controls.append(
            {
                "control_id": control_id,
                "description": definition["description"],
                "source_id": definition["source_id"],
                "piece_id": question["piece_id"],
                "question": packet["question"],
                "selected_measure_range": deepcopy(
                    packet["selected_measure_range"]
                ),
                "candidate_answer": definition["candidate_answer"],
                "expected": deepcopy(definition["expectation"]),
                "reference_packet": packet,
                "generated_answer": generated_answer,
                "frames": {
                    frame: {"status": "pending", "attempts": []}
                    for frame in evaluator.FRAME_NAMES
                },
                "range_scope": (
                    {
                        "status": "pending",
                        "attempts": [],
                        "requirement": range_requirement,
                    }
                    if range_requirement["applicable"]
                    else {
                        **range_requirement,
                        "attempts": [],
                        "assessment": {
                            "status": "not_applicable",
                            "applicable": False,
                            "wrong_measure_application": False,
                        },
                    }
                ),
                "reference_review": (
                    evaluator.reference_review_requirement(packet)
                ),
                "aggregate": None,
                "expectation_evaluation": None,
            }
        )

    if len(controls) != EXPECTED_CONTROL_COUNT:
        raise CalibrationInputError(
            "Expected exactly "
            f"{EXPECTED_CONTROL_COUNT} controls, found {len(controls)}"
        )
    return controls, input_paths


def _assessment_has_required_signal(
    assessment: Mapping[str, Any],
    signal: str,
) -> bool:
    if signal == "none":
        return True
    if signal == "contradiction":
        return (
            assessment["relationship"]
            in {"contradictory", "materially_unreliable"}
            and bool(
                any(
                    item["status"] == "contradicted"
                    for item in assessment["reference_assessments"]
                )
                or any(
                    item["status"] == "contradicted"
                    for item in assessment["candidate_assessments"]
                )
                or "contradicts_expert"
                in assessment["critical_error_types"]
            )
        )
    if signal == "contradiction_or_review_warning":
        return (
            _assessment_has_required_signal(
                assessment,
                "contradiction",
            )
            or "contradicts_expert"
            in assessment.get(
                "unanchored_critical_error_warnings",
                [],
            )
        )
    if signal == "unsupported_mandatory_advice":
        risk_errors = {
            "invents_specific_fact",
            "unsafe_vocal_advice",
            "ungrounded_as_fact",
            "other_material_error",
        }
        return (
            any(
                item["status"] in {"unsupported", "mixed"}
                for item in assessment["candidate_assessments"]
            )
            and (
                assessment["scores"]["factual_safety"] < 3
                or bool(
                    risk_errors.intersection(
                        assessment["critical_error_types"]
                    )
                )
            )
        )
    if signal == "target_incomplete_or_s_substitution_guard":
        missing_target = any(
            item.get("scope_authority")
            == evaluator.QUESTION_SCOPE_AUTHORITY
            and item.get("curator_scope")
            in {"direct_required", "mixed_or_ambiguous"}
            and item.get("status") in {"missing", "contradicted"}
            for item in assessment["reference_assessments"]
        )
        substitution_guard = any(
            item.get("guard_scope")
            == "supplemental_substitution_required_claim"
            for item in assessment.get(
                "atomic_claim_entailment_warnings",
                [],
            )
        )
        return missing_target or substitution_guard
    if signal == "wrong_measure_application":
        raise CalibrationInputError(
            "wrong_measure_application must be evaluated by range_scope"
        )
    if signal == "range_risk":
        raise CalibrationInputError(
            "range_risk must be evaluated by range_scope"
        )
    if signal == "range_uncertainty":
        raise CalibrationInputError(
            "range_uncertainty must be evaluated by range_scope"
        )
    if signal == "range_negation":
        raise CalibrationInputError(
            "range_negation must be evaluated by range_scope"
        )
    raise CalibrationInputError(f"Unknown required signal: {signal}")


def evaluate_control_expectation(
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly compare one completed dual-frame result to its expectation."""

    expected = control["expected"]
    aggregate = control.get("aggregate")
    complete_frames = {
        frame: control["frames"][frame]
        for frame in evaluator.FRAME_NAMES
        if control["frames"][frame].get("status") == "complete"
    }
    relationships = {
        frame: state["assessment"]["relationship"]
        for frame, state in complete_frames.items()
    }
    frame_passes = {
        frame: state["computed"]["pass"]
        for frame, state in complete_frames.items()
    }
    required_signal = expected["required_signal"]
    range_assessment = (
        control.get("range_scope", {}).get("assessment") or {}
    )
    if required_signal in {
        "wrong_measure_application",
        "range_risk",
        "range_uncertainty",
        "range_negation",
    }:
        signal_matches = {}
    else:
        signal_matches = {
            frame: _assessment_has_required_signal(
                state["assessment"],
                required_signal,
            )
            for frame, state in complete_frames.items()
        }
    range_signal_match = (
        (
            required_signal == "wrong_measure_application"
            and range_assessment.get("status") == "fail"
            and range_assessment.get("wrong_measure_application") is True
            and any(
                item.get("relation") == "asserted"
                for item in range_assessment.get(
                    "excluded_claim_assessments",
                    [],
                )
            )
        )
        or (
            required_signal == "range_risk"
            and range_assessment.get("status")
            in {"fail", "human_review"}
            and any(
                item.get("relation") in {"asserted", "uncertain"}
                for item in range_assessment.get(
                    "excluded_claim_assessments",
                    [],
                )
            )
        )
        or (
            required_signal == "range_uncertainty"
            and range_assessment.get("status") == "human_review"
            and any(
                item.get("relation") == "uncertain"
                for item in range_assessment.get(
                    "excluded_claim_assessments",
                    [],
                )
            )
        )
        or (
            required_signal == "range_negation"
            and range_assessment.get("status")
            in {"pass", "human_review"}
            and any(
                item.get("relation") == "negated"
                for item in range_assessment.get(
                    "excluded_claim_assessments",
                    [],
                )
            )
        )
    )
    forbidden_markers = [
        marker.casefold()
        for marker in expected.get(
            "forbidden_covered_reference_markers",
            [],
        )
    ]
    raw_forbidden_coverage_matches = {}
    literal_anchor_guard_rejections = {}
    atomic_claim_guard_rejections = {}
    forbidden_coverage_matches = {}
    for frame, state in complete_frames.items():
        assessment = state["assessment"]
        raw_matches = [
            item
            for item in assessment["reference_assessments"]
            if item.get("status") == "covered"
            if any(
                marker in item.get("reference_text", "").casefold()
                for marker in forbidden_markers
            )
        ]
        literal_guarded_ids = {
            item["reference_id"]
            for item in assessment.get(
                "literal_anchor_coverage_warnings",
                [],
            )
        }
        atomic_guarded_ids = {
            item["reference_id"]
            for item in assessment.get(
                "atomic_claim_entailment_warnings",
                [],
            )
        }
        guarded_ids = literal_guarded_ids | atomic_guarded_ids
        raw_forbidden_coverage_matches[frame] = [
            item.get("reference_text", "")
            for item in raw_matches
        ]
        literal_anchor_guard_rejections[frame] = [
            item.get("reference_text", "")
            for item in raw_matches
            if item.get("reference_id") in literal_guarded_ids
        ]
        atomic_claim_guard_rejections[frame] = [
            item.get("reference_text", "")
            for item in raw_matches
            if item.get("reference_id") in atomic_guarded_ids
        ]
        forbidden_coverage_matches[frame] = [
            item.get("reference_text", "")
            for item in raw_matches
            if item.get("reference_id") not in guarded_ids
        ]
    actual_verdict = aggregate.get("status") if aggregate else "incomplete"
    actual_answer_quality_verdict = (
        aggregate.get("answer_quality_status")
        if aggregate
        else "incomplete"
    )
    allowed_verdicts = expected.get(
        "allowed_verdicts",
        [expected["verdict"]],
    )
    allowed_answer_quality_verdicts = expected.get(
        "allowed_answer_quality_verdicts",
        allowed_verdicts,
    )
    if any(
        verdict not in {"pass", "fail", "human_review"}
        for verdict in [
            *allowed_verdicts,
            *allowed_answer_quality_verdicts,
        ]
    ):
        raise CalibrationInputError(
            "Unknown allowed verdict: "
            f"{allowed_verdicts}, {allowed_answer_quality_verdicts}"
        )
    signal_policy = expected.get("required_signal_frames", "both")
    if required_signal in {
        "wrong_measure_application",
        "range_risk",
        "range_uncertainty",
        "range_negation",
    }:
        signal_policy = "range_scope_nli"
        required_signal_matches = range_signal_match
    elif signal_policy == "both":
        required_signal_matches = (
            len(signal_matches) == len(evaluator.FRAME_NAMES)
            and all(signal_matches.values())
        )
    elif signal_policy == "any":
        required_signal_matches = (
            len(signal_matches) == len(evaluator.FRAME_NAMES)
            and any(signal_matches.values())
        )
    else:
        raise CalibrationInputError(
            f"Unknown required_signal_frames policy: {signal_policy}"
        )
    checks = {
        "both_frames_complete": (
            set(complete_frames) == set(evaluator.FRAME_NAMES)
        ),
        "aggregate_verdict_allowed": actual_verdict in allowed_verdicts,
        "answer_quality_verdict_allowed": (
            actual_answer_quality_verdict
            in allowed_answer_quality_verdicts
        ),
        "relationships_allowed": (
            len(relationships) == len(evaluator.FRAME_NAMES)
            and all(
                relationship in expected["allowed_relationships"]
                for relationship in relationships.values()
            )
        ),
        "required_signal_in_required_frames": required_signal_matches,
        "forbidden_reference_points_not_claimed": (
            len(forbidden_coverage_matches) == len(evaluator.FRAME_NAMES)
            and all(
                not matches
                for matches in forbidden_coverage_matches.values()
            )
        ),
    }
    return {
        "match": all(checks.values()),
        "expected_verdict": expected["verdict"],
        "allowed_verdicts": allowed_verdicts,
        "actual_verdict": actual_verdict,
        "actual_answer_quality_verdict": (
            actual_answer_quality_verdict
        ),
        "allowed_answer_quality_verdicts": (
            allowed_answer_quality_verdicts
        ),
        "relationships": relationships,
        "frame_passes": frame_passes,
        "required_signal": required_signal,
        "required_signal_frames": signal_policy,
        "signal_matches": signal_matches,
        "range_scope_signal_match": range_signal_match,
        "raw_forbidden_coverage_matches": (
            raw_forbidden_coverage_matches
        ),
        "literal_anchor_guard_rejections": (
            literal_anchor_guard_rejections
        ),
        "atomic_claim_guard_rejections": (
            atomic_claim_guard_rejections
        ),
        "forbidden_coverage_matches": forbidden_coverage_matches,
        "checks": checks,
        "mismatch_reasons": [
            name for name, passed in checks.items() if not passed
        ],
    }


def refresh_summary(snapshot: dict[str, Any]) -> None:
    controls = snapshot["controls"]
    complete = [
        control
        for control in controls
        if control.get("expectation_evaluation") is not None
    ]
    matched = [
        control
        for control in complete
        if control["expectation_evaluation"]["match"]
    ]
    frame_errors = sum(
        control["frames"][frame].get("status") == "error"
        for control in controls
        for frame in evaluator.FRAME_NAMES
    ) + sum(
        control.get("range_scope", {}).get("status") == "error"
        for control in controls
    )
    verdict_counts: dict[str, int] = {}
    for control in complete:
        verdict = control["expectation_evaluation"]["actual_verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    all_complete = len(complete) == len(controls)
    accuracy = len(matched) / len(controls) if all_complete else None
    required_matches = math.ceil(MINIMUM_ACCURACY * len(controls))
    snapshot["summary"] = {
        "total_controls": len(controls),
        "completed_controls": len(complete),
        "matched_controls": len(matched),
        "required_matches": required_matches,
        "accuracy": round(accuracy, 6) if accuracy is not None else None,
        "minimum_accuracy": MINIMUM_ACCURACY,
        "quality_gate_passed": (
            all_complete and len(matched) >= required_matches
        ),
        "actual_verdicts": dict(sorted(verdict_counts.items())),
        "frame_errors": frame_errors,
        "mismatched_control_ids": [
            control["control_id"]
            for control in complete
            if not control["expectation_evaluation"]["match"]
        ],
    }
    snapshot["run"]["status"] = (
        "complete" if all_complete else "in_progress"
    )
    snapshot["run"]["updated_at"] = utc_now()
    if all_complete:
        snapshot["run"]["finished_at"] = (
            snapshot["run"].get("finished_at") or utc_now()
        )
    else:
        snapshot["run"]["finished_at"] = None


def run_calibration(
    snapshot: dict[str, Any],
    *,
    judge_fn: JudgeFunction,
    range_signal_fn: evaluator.RangeSignalFunction,
    coverage_signal_fn: evaluator.CoverageSignalFunction,
    range_nli_threshold: float,
    range_embedding_delta_threshold: float,
    checkpoint: CheckpointFunction,
    max_attempts: int,
    rerun: bool = False,
    selected_control_ids: set[str] | None = None,
) -> None:
    """Run or resume all controls through both runner judge frames."""

    for control in snapshot["controls"]:
        if (
            selected_control_ids is not None
            and control["control_id"] not in selected_control_ids
        ):
            continue
        if rerun:
            control["frames"] = {
                frame: {"status": "pending", "attempts": []}
                for frame in evaluator.FRAME_NAMES
            }
            requirement = evaluator.range_scope_requirement(
                control["reference_packet"]
            )
            control["range_scope"] = (
                {
                    "status": "pending",
                    "attempts": [],
                    "requirement": requirement,
                }
                if requirement["applicable"]
                else {
                    **requirement,
                    "attempts": [],
                    "assessment": {
                        "status": "not_applicable",
                        "applicable": False,
                        "wrong_measure_application": False,
                    },
                }
            )
            control["aggregate"] = None
            control["expectation_evaluation"] = None
            checkpoint()
        if control.get("expectation_evaluation") is not None:
            continue

        all_frames_complete = True
        for frame in evaluator.FRAME_NAMES:
            frame_state = control["frames"][frame]
            if frame_state.get("status") == "complete":
                continue
            completed = evaluator._run_one_judge_frame(
                frame=frame,
                packet=control["reference_packet"],
                judge_fn=judge_fn,
                coverage_signal_fn=coverage_signal_fn,
                coverage_nli_threshold=range_nli_threshold,
                frame_state=frame_state,
                checkpoint=checkpoint,
                max_attempts=max_attempts,
            )
            if not completed:
                all_frames_complete = False
                break

        range_state = control["range_scope"]
        requirement = evaluator.range_scope_requirement(
            control["reference_packet"]
        )
        if (
            all_frames_complete
            and requirement["applicable"]
            and range_state.get("status") != "complete"
        ):
            all_frames_complete = evaluator._run_one_range_frame(
                packet=control["reference_packet"],
                judge_fn=judge_fn,
                range_signal_fn=range_signal_fn,
                nli_threshold=range_nli_threshold,
                embedding_delta_threshold=(
                    range_embedding_delta_threshold
                ),
                frame_state=range_state,
                checkpoint=checkpoint,
                max_attempts=max_attempts,
            )

        if all_frames_complete and all(
            control["frames"][frame].get("status") == "complete"
            for frame in evaluator.FRAME_NAMES
        ) and (
            not requirement["applicable"]
            or range_state.get("status") == "complete"
        ):
            control["aggregate"] = evaluator.aggregate_judge_frames(
                control["frames"],
                generated_answer=control["generated_answer"],
                range_scope_evaluation=range_state["assessment"],
                reference_review=control["reference_review"],
                supplemental_evidence_validation=control[
                    "reference_packet"
                ]["supplemental_evidence_validation"],
                review_disclosure_validation=control[
                    "reference_packet"
                ]["review_disclosure_validation"],
            )
            control["expectation_evaluation"] = (
                evaluate_control_expectation(control)
            )
        refresh_summary(snapshot)
        checkpoint()

    refresh_summary(snapshot)
    checkpoint()


def _input_fingerprint(
    *,
    controls: Sequence[Mapping[str, Any]],
    input_files: Sequence[Mapping[str, str]],
    judge_model: Mapping[str, Any],
    dataset_root: Path,
    runtime: Mapping[str, str],
    max_tokens: int,
    range_guard: Mapping[str, Any],
) -> str:
    control_contract = [
        {
            "control_id": item["control_id"],
            "source_id": item["source_id"],
            "selected_measure_range": item["selected_measure_range"],
            "candidate_answer": item["candidate_answer"],
            "generated_answer": item["generated_answer"],
            "expected": item["expected"],
            "reference_packet": item["reference_packet"],
        }
        for item in controls
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "control_contract_version": CONTROL_CONTRACT_VERSION,
        "runner_judge_protocol": evaluator.JUDGE_PROTOCOL_VERSION,
        "frames": list(evaluator.FRAME_NAMES),
        "controls": control_contract,
        "input_files": list(input_files),
        "judge_model": dict(judge_model),
        "dataset_root": str(dataset_root.resolve()),
        "runtime_versions": dict(runtime),
        "max_tokens": max_tokens,
        "range_guard": dict(range_guard),
    }
    return hashlib.sha256(
        evaluator.canonical_json_bytes(payload)
    ).hexdigest()


def new_snapshot(
    *,
    controls: list[dict[str, Any]],
    dataset_root: Path,
    input_files: list[dict[str, str]],
    input_fingerprint: str,
    judge_model: Mapping[str, Any],
    range_guard: Mapping[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    now = utc_now()
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "status": "in_progress",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "dataset_root": str(dataset_root.resolve()),
            "input_fingerprint": input_fingerprint,
            "input_files": input_files,
            "control_contract_version": CONTROL_CONTRACT_VERSION,
            "runner_judge_protocol": evaluator.JUDGE_PROTOCOL_VERSION,
            "frames": list(evaluator.FRAME_NAMES),
            "judge_backend": "transformers",
            "judge_model": dict(judge_model),
            "range_guard": deepcopy(dict(range_guard)),
            "judge_max_tokens": max_tokens,
            "same_model_dual_prompt_is_independent": False,
            "human_review_remains_authoritative": True,
        },
        "summary": {},
        "controls": controls,
    }
    refresh_summary(snapshot)
    return snapshot


def validate_resume_snapshot(
    snapshot: Mapping[str, Any],
    *,
    input_fingerprint: str,
) -> None:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationInputError(
            "Existing calibration output has a different schema; "
            "choose a new --output"
        )
    if snapshot.get("run", {}).get("input_fingerprint") != input_fingerprint:
        raise CalibrationInputError(
            "Calibration inputs, controls, runner, or model changed. "
            "Choose a new --output rather than mixing runs."
        )


def expected_calibration_inputs(
    *,
    dataset_root: Path,
    judge_model_path: Path,
    range_embedding_model_path: Path,
    range_nli_model_path: Path,
    range_nli_threshold: float,
    range_embedding_delta_threshold: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Reconstruct the complete immutable calibration contract."""

    dataset_root = dataset_root.resolve()
    judge_model_path = judge_model_path.resolve()
    architecture_record = validate_qwen3_4b_checkpoint(judge_model_path)
    settings = evaluator._load_runtime_settings()
    generator_model_path = Path(settings["model_path"]).resolve()
    if judge_model_path == generator_model_path:
        raise CalibrationInputError(
            "Judge checkpoint must be distinct from the answer generator"
        )
    corpus_path = Path(settings["corpus_path"]).resolve()
    stats_path = Path(settings["stats_path"]).resolve()
    trusted_expert_catalog = (
        evaluator.load_trusted_expert_evidence_catalog(corpus_path)
    )
    controls, dataset_paths = build_controls(
        dataset_root,
        trusted_expert_catalog=trusted_expert_catalog,
    )
    input_files = evaluator.input_file_records(
        [
            *dataset_paths,
            Path(__file__).resolve(),
            Path(evaluator.__file__).resolve(),
            PROJECT_ROOT / "soprano_qa" / "answer.py",
            PROJECT_ROOT / "soprano_qa" / "corpus.py",
            PROJECT_ROOT / "soprano_qa" / "retrieval.py",
            PROJECT_ROOT / "soprano_qa" / "service.py",
            corpus_path,
            stats_path,
        ]
    )
    judge_model = evaluator._model_record(
        judge_model_path,
        backend="transformers",
    )
    judge_model.update(architecture_record)
    judge_model["distinct_from_generator"] = True
    judge_model["generator_model_path"] = str(generator_model_path)
    generator_model = evaluator._model_record(
        generator_model_path,
        backend="llama-cpp",
    )
    judge_model["generator_model_sha256"] = generator_model["sha256"]
    range_guard = evaluator.build_range_guard_record(
        embedding_model_path=range_embedding_model_path.resolve(),
        nli_model_path=range_nli_model_path.resolve(),
        nli_threshold=range_nli_threshold,
        embedding_delta_threshold=range_embedding_delta_threshold,
    )
    runtime = runtime_versions()
    fingerprint = _input_fingerprint(
        controls=controls,
        input_files=input_files,
        judge_model=judge_model,
        dataset_root=dataset_root,
        runtime=runtime,
        max_tokens=max_tokens,
        range_guard=range_guard,
    )
    return {
        "controls": controls,
        "input_files": input_files,
        "judge_model": judge_model,
        "runtime_versions": runtime,
        "range_guard": range_guard,
        "input_fingerprint": fingerprint,
    }


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-model-path",
        type=Path,
        default=default_qwen3_4b_path(),
        help="Local Transformers Qwen3-4B snapshot directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--judge-max-attempts", type=int, default=4)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--control-id",
        action="append",
        default=None,
        help=(
            "Run only this fixed control ID; repeat for a focused, resumable "
            "calibration subset."
        ),
    )
    parser.add_argument(
        "--range-embedding-model-path",
        type=Path,
        default=Path(os.environ.get(
            "SOPRANO_QA_RANGE_EMBEDDING_MODEL_PATH",
            str(evaluator.DEFAULT_RANGE_EMBEDDING_MODEL_PATH),
        )),
    )
    parser.add_argument(
        "--range-nli-model-path",
        type=Path,
        default=Path(os.environ.get(
            "SOPRANO_QA_RANGE_NLI_MODEL_PATH",
            str(evaluator.DEFAULT_RANGE_NLI_MODEL_PATH),
        )),
    )
    parser.add_argument(
        "--range-nli-threshold",
        type=float,
        default=float(os.environ.get(
            "SOPRANO_QA_RANGE_NLI_THRESHOLD",
            str(evaluator.DEFAULT_RANGE_NLI_THRESHOLD),
        )),
    )
    parser.add_argument(
        "--range-embedding-delta-threshold",
        type=float,
        default=float(os.environ.get(
            "SOPRANO_QA_RANGE_EMBEDDING_DELTA_THRESHOLD",
            str(evaluator.DEFAULT_RANGE_EMBEDDING_DELTA_THRESHOLD),
        )),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Clear completed frame results and rerun all controls.",
    )
    arguments = parser.parse_args(argv)
    if arguments.judge_max_attempts < 1:
        parser.error("--judge-max-attempts must be positive")
    if arguments.judge_max_tokens < 128:
        parser.error("--judge-max-tokens must be at least 128")
    if not 0 <= arguments.range_nli_threshold <= 1:
        parser.error("--range-nli-threshold must be between 0 and 1")
    if not -2 <= arguments.range_embedding_delta_threshold <= 2:
        parser.error(
            "--range-embedding-delta-threshold must be between -2 and 2"
        )
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    dataset_root = arguments.dataset_root.resolve()
    output = arguments.output.resolve()
    judge_model_path = arguments.judge_model_path.resolve()
    expected = expected_calibration_inputs(
        dataset_root=dataset_root,
        judge_model_path=judge_model_path,
        range_embedding_model_path=(
            arguments.range_embedding_model_path.resolve()
        ),
        range_nli_model_path=arguments.range_nli_model_path.resolve(),
        range_nli_threshold=arguments.range_nli_threshold,
        range_embedding_delta_threshold=(
            arguments.range_embedding_delta_threshold
        ),
        max_tokens=arguments.judge_max_tokens,
    )
    controls = expected["controls"]
    selected_control_ids = (
        set(arguments.control_id)
        if arguments.control_id
        else None
    )
    known_control_ids = {
        control["control_id"] for control in controls
    }
    if (
        selected_control_ids is not None
        and not selected_control_ids.issubset(known_control_ids)
    ):
        unknown = sorted(selected_control_ids - known_control_ids)
        raise CalibrationInputError(
            f"Unknown --control-id value(s): {unknown}"
        )
    input_files = expected["input_files"]
    judge_model = expected["judge_model"]
    fingerprint = expected["input_fingerprint"]
    range_guard = expected["range_guard"]

    with evaluator.exclusive_output_lock(output):
        if output.exists():
            snapshot = evaluator.load_json(output)
            validate_resume_snapshot(
                snapshot,
                input_fingerprint=fingerprint,
            )
        else:
            snapshot = new_snapshot(
                controls=controls,
                dataset_root=dataset_root,
                input_files=input_files,
                input_fingerprint=fingerprint,
                judge_model=judge_model,
                range_guard=range_guard,
                max_tokens=arguments.judge_max_tokens,
            )
            snapshot["run"]["runtime_versions"] = expected[
                "runtime_versions"
            ]
            evaluator.atomic_write_json(output, snapshot)

        def checkpoint() -> None:
            refresh_summary(snapshot)
            evaluator.atomic_write_json(output, snapshot)

        already_complete = (
            snapshot.get("run", {}).get("status") == "complete"
            and snapshot.get("summary", {}).get(
                "quality_gate_passed"
            )
            is True
        )
        if arguments.rerun or not already_complete:
            judge_fn = evaluator.build_transformers_judge(
                judge_model_path,
                max_tokens=arguments.judge_max_tokens,
            )
            range_signal_fn = evaluator.build_local_range_signal_scorer(
                embedding_model_path=(
                    arguments.range_embedding_model_path.resolve()
                ),
                nli_model_path=(
                    arguments.range_nli_model_path.resolve()
                ),
            )
            coverage_signal_fn = (
                evaluator.build_local_atomic_coverage_signal_scorer(
                    nli_model_path=(
                        arguments.range_nli_model_path.resolve()
                    ),
                )
            )
            run_calibration(
                snapshot,
                judge_fn=judge_fn,
                range_signal_fn=range_signal_fn,
                coverage_signal_fn=coverage_signal_fn,
                range_nli_threshold=arguments.range_nli_threshold,
                range_embedding_delta_threshold=(
                    arguments.range_embedding_delta_threshold
                ),
                checkpoint=checkpoint,
                max_attempts=arguments.judge_max_attempts,
                rerun=arguments.rerun,
                selected_control_ids=selected_control_ids,
            )
        summary = snapshot["summary"]
        if (
            summary["completed_controls"] != summary["total_controls"]
            or summary["frame_errors"]
        ):
            snapshot["run"].update(
                status="failed_incomplete",
                failure_reason="judge_frame_incomplete_or_error",
                finished_at=utc_now(),
                updated_at=utc_now(),
            )
            evaluator.atomic_write_json(output, snapshot)
        elif not summary["quality_gate_passed"]:
            snapshot["run"].update(
                status="failed_quality_gate",
                failure_reason="calibration_control_mismatch",
                finished_at=utc_now(),
                updated_at=utc_now(),
            )
            evaluator.atomic_write_json(output, snapshot)
        else:
            snapshot["run"].pop("failure_reason", None)
            evaluator.atomic_write_json(output, snapshot)

    print(json.dumps(snapshot["summary"], ensure_ascii=False, indent=2))
    summary = snapshot["summary"]
    if (
        summary["completed_controls"] != summary["total_controls"]
        or summary["frame_errors"]
    ):
        print(
            "Calibration incomplete because one or more judge frames failed.",
            file=sys.stderr,
        )
        return 1
    if not summary["quality_gate_passed"]:
        print(
            "Judge calibration failed: "
            f"{summary['accuracy']:.1%} < {MINIMUM_ACCURACY:.1%}. "
            f"All {EXPECTED_CONTROL_COUNT} controls must match.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
