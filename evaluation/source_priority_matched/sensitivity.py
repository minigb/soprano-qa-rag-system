"""Post-hoc synthetic-marker sensitivity; main frozen experiment is unchanged."""
import copy
import json
from pathlib import Path
from evaluation.source_priority_matched import run


def prepare():
    main = json.loads((run.ROOT / "frozen.json").read_text())
    root = run.ROOT / "neutral_marker_sensitivity"
    root.mkdir(exist_ok=True)
    data = copy.deepcopy(main)
    data["parent_contract_id"] = main["contract_id"]
    data["sensitivity"] = (
        "Post-hoc exploratory control: all claims and relevance scores unchanged; "
        "only the fabricated web record's displayed provenance becomes neutral. "
        "All contradictions remain synthetic evaluation fixtures, not real sources. "
        "Uses all three records to isolate ordering and metadata sensitivity."
    )
    data["cases"] = [c for c in data["cases"] if c["kind"] == "synthetic_conflict"]
    for case in data["cases"]:
        case["budgets"] = [3]
        for entry in case["pool"]:
            if entry["record"]["id"].startswith("synthetic-"):
                entry["record"]["web_coaching_source"] = {
                    "title":"Source C", "author":"Source C", "kind":"text"}
        case["pool_hash"] = run.digest(case["pool"])
    data.pop("contract_id")
    data["contract_id"] = run.digest(data)
    run.write(root / "frozen.json", data)
    return root


if __name__ == "__main__":
    run.ROOT = prepare()
    run.generate_all()
