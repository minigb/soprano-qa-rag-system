"""Condition-masked review packets and paired, cluster-aware summaries."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import statistics

from evaluation.source_priority_matched.run import ROOT, digest, rank, write


def rows():
    return [json.loads(x) for x in (ROOT / "responses.jsonl").read_text().splitlines()]


def status(row):
    # Validation rejection is an abstention, not a runtime/backend failure.
    if (row.get("error") or {}).get("type") in {"SecurityViolation", "GeneratedAnswerRejected"}:
        return "rejected"
    return row["status"]


def packet(row, case):
    source_map = {x["record"]["id"]: x["record"] for x in case["pool"]}
    selected = [source_map[i] for i in row["selected_ids"]]
    def canonical(match):
        n = int(match.group(1)) - 1
        return row["selected_ids"][n] if 0 <= n < len(selected) else "INVALID_CITATION"
    cited_answer = re.sub(r"\bE(\d+)\b", canonical, row.get("raw_answer", "")) if status(row)=="answer" else ""
    value = {"case_id":case["id"], "question":case["question"], "piece":case["piece"],
        "kind":case["kind"], "reference_requirements":case["reference_requirements"],
        "reference_answer":case["reference_answer"], "answer":row["answer"],
        "cited_answer":cited_answer, "status":status(row),
        "conflict_exposed": bool(case.get("observed_conflict")) or any(i.startswith("synthetic-") for i in row["selected_ids"]),
        "evidence": sorted([{"id":r["id"], "type":r["evidence_type"], "answer":r["answer"],
                              "scope":next(x["scope"] for x in case["pool"] if x["record"]["id"]==r["id"])}
                            for r in selected], key=lambda x:x["id"])}
    value["review_id"] = "review-" + digest(value)[:16]
    return value


def make_packets():
    manifest = json.loads((ROOT / "frozen.json").read_text())
    cases = {c["id"]:c for c in manifest["cases"]}
    unique = {}
    mapping = []
    for row in rows():
        p = packet(row, cases[row["case_id"]]); unique[p["review_id"]] = p
        mapping.append({"case_id":row["case_id"], "budget":row["budget"],
                        "condition":row["condition"], "review_id":p["review_id"]})
    ordered = sorted(unique.values(), key=lambda x:digest(x["review_id"]))
    write(ROOT / "review_packets.json", ordered)
    write(ROOT / "review_key.json", mapping)
    print(len(ordered), "unique review packets from",len(mapping),"outputs")


def bootstrap(pairs, metric, draws=10000):
    if not pairs:
        return None
    clusters = defaultdict(list)
    for pair in pairs:
        clusters[pair["cluster"]].append(pair["priority"][metric] - pair["relevance"][metric])
    names = sorted(clusters)
    if len(names) < 2:
        values = [v for group in clusters.values() for v in group]
        return {"priority_minus_relevance":statistics.mean(values),
                "cluster_bootstrap_95_percentile_interval":None,
                "clusters":len(names), "pairs":len(pairs), "draws":0,
                "note":"Insufficient independent clusters for an uncertainty interval"}
    rng = random.Random(20261025)
    samples = []
    for _ in range(draws):
        values = [v for name in rng.choices(names, k=len(names)) for v in clusters[name]]
        samples.append(statistics.mean(values))
    samples.sort()
    values = [v for group in clusters.values() for v in group]
    return {"priority_minus_relevance":statistics.mean(values),
            "cluster_bootstrap_95_percentile_interval":[samples[int(draws*.025)], samples[int(draws*.975)]],
            "clusters":len(clusters),"pairs":len(pairs),"draws":draws}


def summarize():
    manifest = json.loads((ROOT / "frozen.json").read_text())
    if digest({k:v for k,v in manifest.items() if k != "contract_id"}) != manifest["contract_id"]:
        raise ValueError("Frozen manifest has changed")
    cases = {c["id"]:c for c in manifest["cases"]}
    if any(digest(c["pool"]) != c["pool_hash"] for c in cases.values()):
        raise ValueError("Frozen candidate contents have changed")
    output = rows()
    expected = {(c["id"],k,a) for c in cases.values() for k in c["budgets"] for a in ("priority","relevance")}
    actual = {(r["case_id"],r["budget"],r["condition"]) for r in output}
    if actual != expected or len(actual) != len(output):
        raise ValueError("Incomplete or duplicate evaluation outputs")
    if any(r["contract_id"] != manifest["contract_id"] or r["pool_hash"] != cases[r["case_id"]]["pool_hash"] for r in output):
        raise ValueError("Experiment contract or pool mismatch")
    reviews_doc = json.loads((ROOT / "reviews.json").read_text())
    reviews = {r["review_id"]:r for r in reviews_doc["reviews"]}
    pairs = {}
    for row in output:
        case=cases[row["case_id"]]
        if row["selected_ids"] != [x["record"]["id"] for x in rank(case["pool"],row["condition"],row["budget"])]:
            raise ValueError("Selected records disagree with frozen ranking")
        if row["selected_scores"] != [x["relevance"] for x in rank(case["pool"],row["condition"],row["budget"])]:
            raise ValueError("Selected relevance scores differ between freezing and inference")
        key=(row["case_id"],row["budget"])
        p = packet(row,case)
        review=reviews[p["review_id"]]
        for dimension in ("grounding","relevance","completeness"):
            if review[dimension] not in (0,1,2): raise ValueError("Invalid review score")
        if not review["notes"]: raise ValueError("Every review requires rationale")
        pair=pairs.setdefault(key,{"case_id":case["id"],"cluster":case["cluster"],"kind":case["kind"],"budget":row["budget"]})
        pair[row["condition"]]={**review,
            "quality_sum":sum(review[d] for d in ("grounding","relevance","completeness")),
            "fully_satisfactory":int(all(review[d]==2 for d in ("grounding","relevance","completeness"))),
            "accepted":int(status(row)=="answer"), "status":status(row),
            "target_retained":int(case["target_id"] in row["selected_ids"]) if case["target_id"] else 0,
            "selected_ids":row["selected_ids"], "conflict_exposed":p["conflict_exposed"],
            "mean_selected_cosine":statistics.mean(row["selected_scores"]) if row["selected_scores"] else 0}
    values=list(pairs.values())
    groups={"natural":[p for p in values if p["kind"]=="natural"],
            "negative_controls":[p for p in values if p["kind"]=="negative"],
            "conflict_selection_top2":[p for p in values if p["kind"]=="synthetic_conflict" and p["budget"]==2],
            "conflict_order_only_all3":[p for p in values if p["kind"]=="synthetic_conflict" and p["budget"]==3],
            "observed_disagreement":[p for p in values if p["kind"]=="observed_disagreement"]}
    summary={}
    for group, ps in groups.items():
        if not ps:
            continue
        entry={"pairs":len(ps), "different_selected_sets":sum(set(p['priority']['selected_ids'])!=set(p['relevance']['selected_ids']) for p in ps),
               "different_order_or_set":sum(p['priority']['selected_ids']!=p['relevance']['selected_ids'] for p in ps)}
        for arm in ("priority","relevance"):
            rs=[p[arm] for p in ps]
            exposed=[r for r in rs if r["conflict_exposed"]]
            entry[arm]={"accepted":sum(r["accepted"] for r in rs),
                "fully_satisfactory":sum(r["fully_satisfactory"] for r in rs),
                "mean_quality_0_to_6":statistics.mean(r["quality_sum"] for r in rs),
                "mean_grounding_0_to_2":statistics.mean(r["grounding"] for r in rs),
                "mean_relevance_0_to_2":statistics.mean(r["relevance"] for r in rs),
                "mean_completeness_0_to_2":statistics.mean(r["completeness"] for r in rs),
                "target_retained":sum(r["target_retained"] for r in rs),
                "mean_selected_cosine":statistics.mean(r["mean_selected_cosine"] for r in rs),
                "contradiction_adopted":sum(r["false_counterclaim_adopted"] is True for r in rs),
                "conflict_exposed":len(exposed),
                "conflict_acknowledged_when_exposed":sum(r["conflict_acknowledged"] is True for r in exposed),
                "attribution_errors":sum(r["attribution_error"] is True for r in rs)}
        entry["paired_quality_difference"]=bootstrap(ps,"quality_sum")
        entry["paired_satisfactory_difference"]=bootstrap(ps,"fully_satisfactory")
        summary[group]=entry
    report={"contract_id":manifest["contract_id"], "planned_outputs":len(expected),"completed_outputs":len(output),
        "unique_prompts":len({r['prompt_hash'] for r in output}),
        "reused_identical_prompts":sum('reused_identical_prompt' in r for r in output),
        "reviewer":reviews_doc['reviewer'],"summary":summary,"pairs":values}
    write(ROOT / "summary.json",report)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    import sys
    make_packets() if sys.argv[1]=="packets" else summarize()
