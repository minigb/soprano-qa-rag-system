#!/usr/bin/env python3
"""Reproduce the refinement comparison from delivered responses and reviews.

No inference or new grading is performed. The historical analyzer's variant
names are retained in JSON: original=current security-enabled baseline,
hardened=refined candidate. See the run protocol for their exact identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

import analyze_security_comparison as analysis


def wilson(successes: int, n: int) -> list[float]:
    """Descriptive 95% binomial interval, not a population-risk guarantee."""
    z = 1.959963984540054
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    rows = analysis.read_rows(root / "validation_baseline.jsonl") + analysis.read_rows(root / "validation_refined.jsonl")
    inventory = json.loads((root / "validation_cases.json").read_text())["cases"]
    cases = {case["id"]: case for case in inventory}
    if len(cases) != len(inventory):
        raise ValueError("Duplicate case IDs")
    reviews = analysis.load_reviews([root / "review/assessments.json"])
    summary = analysis.summarize(rows, cases, reviews, expected_repeats=1)
    analysis.write_json(root / "summary.json", summary)
    (root / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    details = {
        "variant_labels": {"original": "Current secured baseline (29fe3e2)", "hardened": "Refined Candidate F"},
        "interval_note": "95% Wilson intervals are descriptive only: bilingual pairs and purposive regression selection violate an independent population sample assumption.",
        "monitor_adjustment_note": "Subtract only recorded synchronous usage-check time within requests. This does not isolate GPU contention, cache effects, or run order.",
        "review_count": len(reviews),
        "variants": {},
        "paired_benign": [],
    }
    for variant in analysis.VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        adjusted = deepcopy(selected)
        for row in adjusted:
            cost = row["usage_check_seconds_in_request"]
            if not 0 <= cost <= row["elapsed_seconds"]:
                raise ValueError("Invalid usage-monitor timing")
            row["elapsed_seconds"] -= cost
        cohorts = summary["variants"][variant]["cohorts"]
        intervals = {}
        for kind in analysis.KINDS:
            cohort = cohorts[kind + ":all"]
            numerator = cohort["abstention"] if kind == "benign" else cohort["successes"]
            intervals[kind] = {"numerator": numerator, "denominator": cohort["n"], "wilson_95": wilson(numerator, cohort["n"])}
        details["variants"][variant] = {
            "rate_intervals": intervals,
            "monitor_seconds_mean": statistics.mean(row["usage_check_seconds_in_request"] for row in selected),
            "monitor_seconds_benign_mean": statistics.mean(row["usage_check_seconds_in_request"] for row in selected if row["kind"] == "benign"),
            "monitor_adjusted_latency": {
                kind: analysis.latency_stats([
                    row for row in adjusted if kind == "all" or row["kind"] == kind
                    or (kind == "both_answered_benign" and row["case_id"] in summary["both_answered_benign_case_ids"])
                ]) for kind in ("all", "benign", "both_answered_benign")
            },
        }

    by_key = {(row["variant"], row["case_id"]): row for row in rows}
    for identifier, case in cases.items():
        if case["kind"] != "benign":
            continue
        pair = {"case_id": identifier, "language": case["language"]}
        for variant in analysis.VARIANTS:
            row = by_key[variant, identifier]
            delivered = analysis.outcome(row) == "answer"
            assessment = reviews[analysis.review_id(row)] if delivered else None
            pair[variant] = {
                "outcome": analysis.outcome(row),
                "unavailable_reason": (row.get("response") or {}).get("unavailable_reason"),
                "effective_score": sum(assessment[key] for key in analysis.DIMENSIONS) * 100 / 7 if assessment else 0,
                "review": assessment,
            }
        delta = pair["hardened"]["effective_score"] - pair["original"]["effective_score"]
        pair["score_change"] = "improved" if delta > 1e-9 else "worsened" if delta < -1e-9 else "unchanged"
        details["paired_benign"].append(pair)
    details["benign_score_transitions"] = dict(Counter(pair["score_change"] for pair in details["paired_benign"]))
    analysis.write_json(root / "comparison_details.json", details)
    print(json.dumps({"requests": len(rows), "reviews": len(reviews), "benign_score_transitions": details["benign_score_transitions"]}))


if __name__ == "__main__":
    main()
