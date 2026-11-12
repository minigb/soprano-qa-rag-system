"""Compare a complete generated cohort to the unchanged pre-security baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_joint_expanded_judge import frozen_write, validate_coverage
from summarize_expanded_comparison_bounded import latency, mean, paired_cluster_interval, read, rows, sha, summarize


def load_comparison(cases_path, result_paths, metric_paths, judge_path):
    cases = {c["id"]: c for c in read(cases_path)["cases"]}
    if len(cases) != 460:
        raise ValueError("Expected all 460 fixed requests")
    packets = read(judge_path / "packets.json")
    assessed = rows(judge_path / "assessments.jsonl")
    validate_coverage(packets, assessed, "claude-opus-5-5")
    judgments = {r["id"]: r["assessment"] for r in assessed}
    mapping = {r["key"]: r for r in read(judge_path / "mapping.json")}
    if len(mapping) != 920:
        raise ValueError("Expected two complete result mappings")
    variants, metric_files = {}, {}
    for name in ("original", "current"):
        path = result_paths[name]
        data = rows(path)
        summaries = read(metric_paths[name] / "summary.json")
        if (len(data) != 460 or {r["case_id"] for r in data} != set(cases)
                or len(summaries) != 1 or summaries[0]["count"] != 460 or summaries[0]["truncated_texts"]):
            raise ValueError("Incomplete cohort or changed metric protocol")
        metric_file = metric_paths[name] / (summaries[0]["config_id"] + ".json")
        bert = {r["id"]: r for r in read(metric_file)["cases"]}
        if set(bert) != set(cases):
            raise ValueError("Metric cohort differs")
        metric_files[name] = metric_file
        indexed = {}
        for row in data:
            cid = row["case_id"]
            item = mapping[path.stem + ":" + cid]
            judge = judgments.get(item["review_id"])
            response = row.get("response") or {}
            if (judge is not None) != (row["status"] == "response"):
                raise ValueError("Response/judge coverage differs")
            delivered = (row["status"] == "response" and response.get("generation_mode") != "unavailable"
                         and bool(response.get("answer", "").strip()))
            indexed[cid] = {"case": cases[cid], "row": row, "judge": judge, "bert": bert[cid],
                            "delivered": delivered, "useful": bool(delivered and judge and not judge["refusal"])}
        variants[name] = indexed
    return cases, variants, metric_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--current-results", type=Path, required=True)
    parser.add_argument("--original-metrics", type=Path, required=True)
    parser.add_argument("--current-metrics", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_paths = {"original": args.original_results, "current": args.current_results}
    metric_paths = {"original": args.original_metrics, "current": args.current_metrics}
    cases, variants, metric_files = load_comparison(args.cases, result_paths, metric_paths, args.judge)
    stats = {}
    for name, indexed in variants.items():
        items = list(indexed.values())
        groups = {"all": items}
        for family in ("original", "paraphrase", "ku_derived"):
            groups[family] = [i for i in items if i["case"]["family"] == family]
        for split in ("development", "validation"):
            groups["original_" + split] = [i for i in groups["original"] if i["case"]["split"] == split]
        stats[name] = {key: summarize(value) for key, value in groups.items()}
    pairs = [(variants["original"][cid], variants["current"][cid]) for cid in cases]
    both = [(a, b) for a, b in pairs if a["useful"] and b["useful"]]
    material = lambda i: bool(i["judge"] and i["judge"]["source_fidelity"] == "material_drift")
    paired = {
        "both_useful": len(both),
        "new_material_drift_ids": [b["case"]["id"] for a, b in pairs if material(b) and not material(a)],
        "resolved_material_drift_ids": [b["case"]["id"] for a, b in pairs if material(a) and not material(b) and b["useful"]],
        "current_no_useful_answer_ids": [b["case"]["id"] for a, b in pairs if not b["useful"]],
        "naturalness_both_useful": {
            "original_mean": mean([a["judge"]["naturalness"] for a, b in both]),
            "current_mean": mean([b["judge"]["naturalness"] for a, b in both]),
            "wins": sum(b["judge"]["naturalness"] > a["judge"]["naturalness"] for a, b in both),
            "ties": sum(b["judge"]["naturalness"] == a["judge"]["naturalness"] for a, b in both),
            "losses": sum(b["judge"]["naturalness"] < a["judge"]["naturalness"] for a, b in both),
        },
        "naturalness_effective_cluster_bootstrap": paired_cluster_interval(pairs, "naturalness"),
        "factual_effective_cluster_bootstrap": paired_cluster_interval(pairs, "factual_correctness"),
        "latency_difference_both_useful": latency([b["row"]["elapsed_seconds"] - a["row"]["elapsed_seconds"] for a, b in both]),
    }
    sources = {"cases": args.cases, "judge_mapping": args.judge / "mapping.json",
               "judge_assessments": args.judge / "assessments.jsonl", "provenance": args.provenance,
               **{"results_" + key: value for key, value in result_paths.items()},
               **{"scores_" + key: value for key, value in metric_files.items()}}
    report = {"contract": {
        "sources": {key: {"path": str(path.resolve()), "sha256": sha(path)} for key, path in sources.items()},
        "summary_script_sha256": sha(Path(__file__)),
        "failure_policy": "All 460 requests retained. Effective quality includes errors/refusals as zero. Unscorable overlength BERTScore is explicitly flagged and contributes zero; no truncation or reference changes.",
        "comparison_scope": "Full-system comparison with native corpora (946 original, 953 candidate; shared 946 and all expert KUs unchanged). Generator/model/settings changes are recorded in provenance. Serial single-run latency is descriptive; judge and model variability are not estimated. Reused internal evaluation, not a new holdout.",
    }, "variants": stats, "paired": paired}
    frozen_write(args.output, report)
    print(json.dumps({name: stat["all"] for name, stat in stats.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
