"""Paired cluster bootstrap keeps bilingual and repeated-measure structure."""
from __future__ import annotations

import unittest

from evaluation import security_comparison_intervals as intervals
from tests.test_security_analysis import abstain, exception, fixture, judge, select


class PairedClusterTests(unittest.TestCase):
    def test_shared_between_case_variation_cancels_in_paired_difference(self):
        cases, values = {}, {}
        for index, baseline in enumerate((0.0, 20.0, 80.0)):
            for language in ("ko", "en"):
                identifier = f"pair-{index}-{language}"
                cases[identifier] = {"pair_id": f"pair-{index}", "language": language}
                values[identifier] = {"original": baseline, "hardened": baseline + 3}
        result = intervals.paired_cluster_interval(values, cases, draws=500, seed=1)
        self.assertEqual(result["difference_hardened_minus_original"], 3)
        self.assertEqual(result["percentile_interval_95"], [3, 3])
        self.assertEqual((result["n_clusters"], result["n_cases"]), (3, 6))

    def test_correlated_languages_are_resampled_as_whole_pairs(self):
        cases, values = {}, {}
        for index, difference in enumerate((0.0, 0.0, 1.0)):
            for language in ("ko", "en"):
                identifier = f"pair-{index}-{language}"
                cases[identifier] = {"pair_id": f"pair-{index}", "language": language}
                values[identifier] = {"original": 0.0, "hardened": difference}
        result = intervals.paired_cluster_interval(values, cases, draws=10000, seed=20261025)
        self.assertEqual(result, intervals.paired_cluster_interval(values, cases, draws=10000, seed=20261025))
        # With three clusters, sampling the high pair three times has p=1/27;
        # incorrectly treating six language cases independently narrows this
        # upper endpoint because six highs have probability only 1/729.
        self.assertEqual(result["percentile_interval_95"], [0, 1])
        self.assertAlmostEqual(result["difference_hardened_minus_original"], 1 / 3)

    def test_incomplete_bilingual_cluster_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one Korean and one English"):
            intervals.paired_cluster_interval(
                {"one": {"original": 0, "hardened": 1}},
                {"one": {"pair_id": "pair", "language": "ko"}},
            )


class ComparisonIntervalTests(unittest.TestCase):
    def test_primary_rates_quality_and_repeat_averaged_latency_use_correct_units(self):
        cases, rows = fixture()
        for row in rows:
            if row["kind"] == "benign":
                row["elapsed_seconds"] = (2 if row["variant"] == "original" else 4) + (4 if row["variant"] == "original" else 6) * row["repeat"]
        abstain(select(rows, "benign-ko", "hardened", 0))
        select(rows, "direct_attack-ko", "hardened", 0)["response"]["answer"] = "ATTACK SUCCEEDED"
        select(rows, "direct_attack-en", "hardened", 0)["response"]["answer"] = "ATTACK SUCCEEDED"
        # These second-run attacks must not inflate the primary attack rate.
        for identifier in ("direct_attack-ko", "direct_attack-en", "indirect_attack-ko", "indirect_attack-en"):
            select(rows, identifier, "original", 1)["response"]["answer"] = "ATTACK SUCCEEDED"
        result = intervals.compute_intervals(rows, cases, judge(rows), draws=100, seed=2)
        metrics = result["metrics"]
        self.assertEqual(metrics["attack_success"]["difference_hardened_minus_original"], 50)
        self.assertEqual((metrics["attack_success"]["n_clusters"], metrics["attack_success"]["n_cases"]), (2, 4))
        self.assertEqual(metrics["false_refusal"]["difference_hardened_minus_original"], 50)
        self.assertEqual(metrics["effective_quality"]["difference_hardened_minus_original"], -50)
        self.assertEqual(metrics["benign_latency"]["original"], 4)
        self.assertEqual(metrics["benign_latency"]["hardened"], 7)
        self.assertEqual(metrics["benign_latency"]["difference_hardened_minus_original"], 3)
        self.assertEqual((metrics["benign_latency"]["n_clusters"], metrics["benign_latency"]["n_cases"]), (1, 2))

    def test_errors_stay_in_denominators_but_are_reported_explicitly(self):
        cases, rows = fixture()
        exception(select(rows, "benign-ko", "hardened", 0), "RuntimeError")
        exception(select(rows, "direct_attack-ko", "hardened", 0), "QueryTranslationFailed")
        result = intervals.compute_intervals(rows, cases, judge(rows), draws=100)
        self.assertEqual(result["metrics"]["false_refusal"]["difference_hardened_minus_original"], 0)
        self.assertEqual(result["metrics"]["effective_quality"]["difference_hardened_minus_original"], -50)
        self.assertEqual(result["metrics"]["attack_success"]["n_cases"], 4)
        self.assertEqual(result["metrics"]["attack_success"]["primary_outcomes"]["hardened"]["execution_error"], 1)
        self.assertEqual(result["metrics"]["attack_success"]["execution_error_jobs"][0]["exception"]["type"], "QueryTranslationFailed")

    def test_partial_run_cannot_produce_intervals(self):
        cases, rows = fixture()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            intervals.compute_intervals(rows[:-1], cases, judge(rows), draws=100)


if __name__ == "__main__":
    unittest.main()
