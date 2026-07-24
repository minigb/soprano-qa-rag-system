#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration loading for the standalone Soprano QA system."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SETTINGS = os.path.join(PROJECT_ROOT, "config", "settings.json")
PATH_KEYS = {
    "dataset_root",
    "corpus_path",
    "stats_path",
    "feature_overrides_path",
    "model_path",
}


def load_settings(path: str = DEFAULT_SETTINGS) -> Dict[str, Any]:
    settings_path = os.path.abspath(os.path.expanduser(path))
    with open(settings_path, encoding="utf-8") as f:
        settings = json.load(f)
    base_dir = os.path.dirname(settings_path)
    for key in PATH_KEYS:
        value = settings.get(key)
        if not value:
            continue
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        if not os.path.isabs(expanded):
            expanded = os.path.join(base_dir, expanded)
        settings[key] = os.path.abspath(expanded)

    dataset_override = os.environ.get("SOPRANO_QA_DATASET_ROOT")
    if dataset_override:
        settings["dataset_root"] = os.path.abspath(
            os.path.expandvars(os.path.expanduser(dataset_override))
        )
    return settings


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)
