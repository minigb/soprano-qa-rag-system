"""Deterministic representative-range selection for evaluation cases."""

from __future__ import annotations

import hashlib
from typing import Sequence


REPRESENTATIVE_RANGE_POLICY = (
    "sha256_source_id_modulo_sorted_active_ranges_v1"
)
REPRESENTATIVE_RANGE_PROVENANCE = (
    "deterministic_representative_confirmed_knowledge_unit_range"
)
REVIEWED_HINT_CONTEXT_RANGE_PROVENANCE = (
    "explicit_reviewed_measure_hint_used_as_evaluation_selector_context"
)
REVIEWED_HINT_CONTEXT_RANGE_POLICY = (
    "explicit_reviewed_measure_hint_selector_context_v1"
)
REVIEWED_HINT_CONTEXT_CASE_KIND = "reviewed_hint_evaluation_context_range"
_POLICY_SALT = "soprano-qa-evaluation-representative-range-v1\0"


def select_representative_range(
    source_id: str,
    active_ranges: Sequence[Sequence[int]],
) -> list[int] | None:
    """Select one stable range from canonical, non-excluded candidates.

    Callers validate and sort the canonical ranges before invoking this
    helper. Hash-based selection avoids systematically testing only the
    beginning of each piece while remaining reproducible across runners and
    machines.
    """

    if not active_ranges:
        return None
    digest = hashlib.sha256(
        f"{_POLICY_SALT}{source_id}".encode("utf-8")
    ).digest()
    selected_index = int.from_bytes(digest[:8], "big") % len(active_ranges)
    start, end = active_ranges[selected_index]
    return [int(start), int(end)]
