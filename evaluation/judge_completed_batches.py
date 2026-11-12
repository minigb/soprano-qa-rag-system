"""Judge immutable, completed batches while a generation cohort is running.

Every result is retained in source order. Batches are selected by completion
count, never answer quality. Each batch is a complete ordinary joint review
whose exact packets can later be reused under the unchanged rubric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_joint_expanded_judge import frozen_write


def complete_lines(path):
    if not path.exists():
        return []
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines(keepends=True)
            if line.endswith("\n")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    ids = {case["id"] for case in cases}
    if len(ids) != len(cases) or args.batch_size < 1:
        raise ValueError("Invalid cohort or batch size")
    scripts = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    offset = 0
    while offset < len(cases):
        rows = complete_lines(args.results)
        if len({r["case_id"] for r in rows}) != len(rows) or any(r["case_id"] not in ids for r in rows):
            raise ValueError("Result cohort identity changed")
        count = min(args.batch_size, len(cases) - offset)
        if len(rows) < offset + count:
            time.sleep(10)
            continue
        selected = rows[offset:offset + count]
        batch = args.output / f"batch-{offset // args.batch_size:03d}"
        batch.mkdir(exist_ok=True)
        path = batch / args.results.name
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        if path.exists() and path.read_text() != text:
            raise ValueError("Completed result batch changed")
        path.write_text(text)
        selected_ids = {r["case_id"] for r in selected}
        frozen_write(batch / "cases.json", {"cases": [c for c in cases if c["id"] in selected_ids]})
        subprocess.run([sys.executable, str(scripts / "prepare_expanded_judge_packets.py"),
                        "--cases", str(batch / "cases.json"), "--results", str(path),
                        "--output", str(batch / "judge")], check=True)
        subprocess.run([sys.executable, str(scripts / "run_joint_expanded_judge.py"),
                        "--directory", str(batch / "judge"), "--workers", str(args.workers),
                        "--model", "claude-opus-5-5"], check=True)
        offset += count
        print(json.dumps({"judged_results": offset, "total": len(cases)}), flush=True)


if __name__ == "__main__":
    main()
