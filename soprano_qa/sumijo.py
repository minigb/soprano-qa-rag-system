"""Load Sumi Jo's direct feedback as its own provenance-bound lineage."""
from __future__ import annotations

import importlib.util
from pathlib import Path


EVIDENCE_TYPE = "sumijo_feedback"


def enabled(settings):
    return bool((settings.get("sumijo") or {}).get("enabled", False))


def input_paths(dataset_root):
    root = Path(dataset_root) / "sumijo-feedback"
    paths = [root / "knowledge_units.json", root / "tools" / "validate.py"]
    for pattern in ("text/*/*.txt", "audio/*/*.m4a"):
        paths.extend(sorted(root.glob(pattern)))
    return [(str(p.relative_to(dataset_root)), str(p)) for p in paths]


def build_records(dataset_root):
    from soprano_qa.corpus import PIECES, semantic_relevance_text, retrieval_text

    root = Path(dataset_root) / "sumijo-feedback"
    spec = importlib.util.spec_from_file_location("sumijo_validator", root / "tools" / "validate.py")
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load Sumi Jo lineage validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources, units = module.collect(root)
    records = []
    for unit in units:
        source = sources[unit["source_id"]]
        piece = PIECES[unit["piece"]]
        record = {
            "id": unit["knowledge_unit_id"], "evidence_type": EVIDENCE_TYPE,
            "piece": unit["piece"], "piece_title": piece["title"],
            "work": piece["work"], "composer": piece["composer"],
            "genre": piece["genre"], "language": piece["language"],
            "measure_range": [], "measure_scope": "unspecified",
            "measure_status": "unspecified",
            "retrieval_aliases": [source["question"]] if unit["knowledge_unit_id"].endswith("-01") else [],
            "answer": unit["answer"],
            "sumijo_source": {
                "source_id": source["source_id"], "speaker": source["speaker"],
                "files": source["files"], "sha256": source["sha256"],
                "attribution_basis": source["attribution_basis"],
            },
        }
        record["relevance_text"] = semantic_relevance_text(record)
        record["retrieval_text"] = retrieval_text(record)
        validate_record(record)
        records.append(record)
    return records


def validate_record(record):
    from soprano_qa.corpus import EXPERT_RELEASE_RECORD_KEYS, expect_exact_keys

    keys = tuple("sumijo_source" if k == "source_ids" else k for k in EXPERT_RELEASE_RECORD_KEYS)
    expect_exact_keys(record, keys, label="Sumi Jo release record")
    source = record["sumijo_source"]
    if (record["evidence_type"] != EVIDENCE_TYPE
            or not record["id"].startswith(source["source_id"] + "-ku-")
            or not source["source_id"].startswith("sumijo-" + record["piece"] + "-")
            or source["speaker"] != "Sumi Jo"
            or not source.get("attribution_basis")
            or set(source["files"]) != {"raw", "reviewed", "audio"}
            or set(source["sha256"]) != set(source["files"])
            or record["measure_status"] != "unspecified"
            or record["measure_scope"] != "unspecified"
            or record["measure_range"] or not record["answer"].strip()):
        raise ValueError("Invalid Sumi Jo release provenance or scope")
