"""Per-measure score context (sung lyrics) for selected-range questions.

The dataset's ``sheet_music/lyrics/<piece>.json`` files record, for every
measure, the text printed under the solo voice staff. They are model
transcriptions that a person has not yet verified, so they are supplied to the
generator only as labelled context for locating a passage, never as musical
evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from typing import Any, Dict, List

MAX_CONTEXT_MEASURES = 24
_CACHE: Dict[str, Any] = {}


def _lyrics_dir(dataset_root: str) -> str:
    return os.path.join(os.path.realpath(dataset_root), "sheet_music", "lyrics")


def score_context_input_paths(dataset_root: str) -> List[str]:
    root = os.path.join(os.path.realpath(dataset_root), "sheet_music")
    paths = [os.path.join(root, "validate_lyrics.py")]
    lyrics_dir = _lyrics_dir(dataset_root)
    if os.path.isdir(lyrics_dir):
        paths.extend(
            sorted(
                os.path.join(lyrics_dir, name)
                for name in os.listdir(lyrics_dir)
                if name.endswith(".json")
            )
        )
    return paths


def _fingerprint(paths: List[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        with open(path, "rb") as file:
            digest.update(hashlib.sha256(file.read()).digest())
    return digest.hexdigest()


def load_measure_lyrics(dataset_root: str) -> Dict[str, Dict[str, Any]]:
    """Load lyric files after the dataset validator accepts them; fail closed."""

    paths = score_context_input_paths(dataset_root)
    key = _fingerprint(paths)
    if _CACHE.get("key") == key:
        return _CACHE["value"]
    validator_path = paths[0]
    spec = importlib.util.spec_from_file_location(
        "soprano_dataset_validate_lyrics", validator_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load lyric validator at %s" % validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    value: Dict[str, Dict[str, Any]] = {}
    for piece in module.PIECES:
        errors = module.check(piece)
        if errors:
            raise ValueError(
                "Measure lyrics for %s failed validation: %s"
                % (piece, "; ".join(errors[:3]))
            )
        with open(
            os.path.join(_lyrics_dir(dataset_root), piece + ".json"),
            encoding="utf-8",
        ) as file:
            document = json.load(file)
        value[piece] = {
            "status": document["status"],
            "line_labels": list(document["line_labels"]),
            "measures": {
                measure["measure"]: measure for measure in document["measures"]
            },
        }
    _CACHE.update(key=key, value=value)
    return value


def selected_measure_lyrics(
    lyrics: Dict[str, Dict[str, Any]],
    piece: str,
    measure_ranges: List[List[int]],
) -> Dict[str, Any] | None:
    """Return the lyric entries for the selected measures, or None."""

    document = lyrics.get(piece)
    if not document or not measure_ranges:
        return None
    numbers = sorted(
        {
            number
            for start, end in measure_ranges
            for number in range(int(start), int(end) + 1)
            if number in document["measures"]
        }
    )
    if not numbers:
        return None
    truncated = len(numbers) > MAX_CONTEXT_MEASURES
    numbers = numbers[:MAX_CONTEXT_MEASURES]
    return {
        "source": "sheet_music/lyrics",
        "status": document["status"],
        "line_labels": document["line_labels"],
        "truncated": truncated,
        "measures": [
            {
                "measure": number,
                "voice": document["measures"][number]["voice"],
                "lyric_lines": list(document["measures"][number]["lyric_lines"]),
            }
            for number in numbers
        ],
    }


def format_score_context(context: Dict[str, Any] | None) -> str:
    """Render the prompt block for the selected measures' sung text."""

    if not context:
        return ""
    lines = [
        "Score context (sung text in the selected bars)",
        "source_status: %s" % context["status"],
        "note: transcribed from the score image and not yet verified. It shows "
        "only which words are printed under the voice staff in each bar; it has "
        "no pitches, rhythms, dynamics, or performance advice. Use it only to "
        "identify or name the words sung in the selected bars. Never infer "
        "musical or performance facts from it, and do not cite it with an "
        "[E#] label.",
    ]
    for index, label in enumerate(context["line_labels"], start=1):
        lines.append("line %d = %s" % (index, label))
    for measure in context["measures"]:
        if measure["voice"] != "sung":
            text = "(voice %s)" % measure["voice"].replace("_", " ")
        elif not measure["lyric_lines"]:
            text = "(voice sings; no new text printed)"
        else:
            text = " | ".join(
                "line %d: %s" % (index, line)
                for index, line in enumerate(measure["lyric_lines"], start=1)
            )
        lines.append("bar %d: %s" % (measure["measure"], text))
    if context["truncated"]:
        lines.append(
            "(only the first %d selected bars are listed)" % MAX_CONTEXT_MEASURES
        )
    return "\n".join(lines)
