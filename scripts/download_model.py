#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the configured local GGUF checkpoint."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from soprano_qa.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None, help="Path to settings JSON")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.settings) if args.settings else load_settings()
    model_path = settings["model_path"]
    if os.path.exists(model_path) and not args.force:
        print("Model already exists: %s" % model_path)
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run: "
            "conda run -n soprano-qa python -m pip install -r requirements.txt"
        ) from exc

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("Downloading %s/%s" % (settings["model_repo_id"], settings["model_filename"]))
    downloaded = hf_hub_download(
        repo_id=settings["model_repo_id"],
        filename=settings["model_filename"],
        local_dir=os.path.dirname(model_path),
        force_download=args.force,
    )
    if os.path.abspath(downloaded) != os.path.abspath(model_path):
        shutil.copy2(downloaded, model_path)
    print("Model ready: %s" % model_path)


if __name__ == "__main__":
    main()
