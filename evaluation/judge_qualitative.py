#!/usr/bin/env python3
"""Judge answers for qualitative expert-answer fidelity.

This is a lightweight LLM-as-a-judge pass over ``qualitative.json``. Its
pass/review/fail labels are conservative qualitative triage for later human
inspection, not an automatic production-acceptance gate.

The judge sees the generated answer, evaluation question, selected measure
range, and the case's finalized, range-applicable knowledge-unit authority.
Raw annotator answers, editorial notes, retrieval evidence, retrieval IDs, and
retrieval-hit diagnostics are intentionally omitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_ARTIFACT_TYPE = "five_piece_qualitative_rag_llm_evaluation"
SOURCE_SCHEMA_VERSION = "1.2"
ARTIFACT_TYPE = "five_piece_qualitative_answer_fidelity_judgment"
SCHEMA_VERSION = "2.1"
PROTOCOL_VERSION = "qualitative-answer-fidelity-v9-codex-cli"
PIECE_IDS = (
    "die-forelle",
    "in-flowery-clouds",
    "la-capinera",
    "nella-fantasia",
    "una-voce-poco-fa",
)
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "qualitative.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evaluation" / "qualitative_judged.json"
)
VERDICTS = ("pass", "review", "fail")
DRIFT_TYPES = (
    "agent_or_role",
    "predicate",
    "causal",
    "modality",
    "negation",
    "scope_or_quantifier",
    "entity_or_object",
    "temporal_or_condition",
    "unsupported_conflation",
    "unsupported_addition",
    "missing_core_meaning",
    "question_mismatch",
    "presentation_scope",
    "ambiguity",
)
FAIL_ONLY_DRIFT_TYPES = frozenset(
    {
        "agent_or_role",
        "predicate",
        "causal",
        "modality",
        "negation",
        "scope_or_quantifier",
        "entity_or_object",
        "temporal_or_condition",
        "unsupported_conflation",
        "unsupported_addition",
        "missing_core_meaning",
        "question_mismatch",
        "presentation_scope",
    }
)
REVIEW_ELIGIBLE_DRIFT_TYPES = frozenset({"ambiguity"})
JUDGE_RESPONSE_KEYS = {
    "verdict",
    "direct_answer",
    "faithful_to_authority",
    "drift_types",
    "material_differences",
    "reason",
}
MATERIAL_DIFFERENCE_KEYS = {
    "candidate_span",
    "reference_id",
    "reference_span",
    "drift_type",
    "explanation",
}
JUDGMENT_KEYS = {
    "status",
    *JUDGE_RESPONSE_KEYS,
    "attempt_count",
    "attempts",
    "started_at",
    "completed_at",
    "error",
}
ATTEMPT_KEYS = {
    "attempt",
    "started_at",
    "completed_at",
    "raw_response",
    "valid",
    "error",
}
MAX_REASON_CHARACTERS = 500
MAX_DIFFERENCE_CHARACTERS = 500
MAX_CODEX_DIAGNOSTIC_CHARACTERS = 500
DEFAULT_CODEX_TIMEOUT_SECONDS = 600
REASONING_EFFORT_ORDER = (
    "ultra",
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "none",
)
CODEX_JUDGE_PROVIDER_ID = "soprano_qa_judge"
CODEX_STATIC_CONFIG_OVERRIDES = (
    f'model_provider="{CODEX_JUDGE_PROVIDER_ID}"',
    (
        f'model_providers.{CODEX_JUDGE_PROVIDER_ID}.name='
        '"OpenAI authenticated no-retry judge"'
    ),
    f"model_providers.{CODEX_JUDGE_PROVIDER_ID}.requires_openai_auth=true",
    f'model_providers.{CODEX_JUDGE_PROVIDER_ID}.wire_api="responses"',
    f"model_providers.{CODEX_JUDGE_PROVIDER_ID}.supports_websockets=false",
    f"model_providers.{CODEX_JUDGE_PROVIDER_ID}.request_max_retries=0",
    f"model_providers.{CODEX_JUDGE_PROVIDER_ID}.stream_max_retries=0",
    'approval_policy="never"',
    'web_search="disabled"',
    'shell_environment_policy.inherit="none"',
)
CODEX_REQUIRED_DISABLED_FEATURES = (
    "apps",
    "apply_patch_streaming_events",
    "artifact",
    "auth_elicitation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode",
    "code_mode_host",
    "code_mode_only",
    "computer_use",
    "default_mode_request_user_input",
    "deferred_executor",
    "enable_fanout",
    "enable_mcp_apps",
    "exec_permission_approvals",
    "goals",
    "guardian_approval",
    "hooks",
    "image_generation",
    "in_app_browser",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "network_proxy",
    "non_prefixed_mcp_tool_names",
    "plugin_sharing",
    "plugins",
    "remote_plugin",
    "request_permissions_tool",
    "respect_system_proxy",
    "shell_snapshot",
    "shell_tool",
    "shell_zsh_fork",
    "skill_mcp_dependency_install",
    "standalone_web_search",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "unified_exec",
    "unified_exec_zsh_fork",
    "workspace_dependencies",
)
CODEX_KNOWN_DEPRECATED_INERT_FEATURES = (
    "web_search_cached",
    "web_search_request",
)
CODEX_TOOL_FEATURE_MARKERS = (
    "apps",
    "apply_patch",
    "artifact",
    "browser",
    "code_mode",
    "computer",
    "elicitation",
    "exec",
    "fanout",
    "goal",
    "hook",
    "image_generation",
    "repl",
    "mcp",
    "memor",
    "multi_agent",
    "permission",
    "plugin",
    "proxy",
    "remote_control",
    "shell",
    "tool",
    "web_search",
    "workspace_dependencies",
)
CODEX_MINIMAL_ENVIRONMENT = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
CODEX_ENVIRONMENT_POLICY = {
    "inherit_parent_environment": False,
    "fixed_keys": sorted(CODEX_MINIMAL_ENVIRONMENT),
    "temporary_keys": ["CODEX_HOME", "HOME", "TMPDIR"],
    "credential_source": "copied_auth_json_only",
    "provider_or_proxy_overrides_inherited": False,
}
CODEX_ROUTING_CONFIG_KEYS = (
    "model",
    "model_reasoning_effort",
    "service_tier",
    "model_provider",
    "profile",
    "profiles",
    "forced_login_method",
    "forced_chatgpt_workspace_id",
    "chatgpt_base_url",
    "openai_base_url",
    "model_providers",
)
CODEX_EXEC_FLAGS = (
    "--sandbox",
    "read-only",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "--cd",
    "<temporary-directory>",
    "--output-schema",
    "<temporary-schema-path>",
    "--output-last-message",
    "<temporary-result-path>",
    "--json",
    "--color",
    "never",
    "-",
)
FINAL_MEASURE_STATUSES = {"specific", "whole_piece", "unspecified"}
REFERENCE_ROLES = {"required_any", "supporting"}
MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT = (
    "measure_scoped_question_no_range"
)
SUPPORTED_COHORTS = (MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT,)
SOURCE_GENERATED_ANSWER_PATHS = {
    ("expert_verbatim", "authoritative_expert"),
    ("llm", "retrieved_evidence"),
    ("unavailable", "no_corpus_evidence"),
}
SOURCE_REMOVED_RESPONSE_FIELDS = {
    "allow_internal_knowledge",
    "context_limited",
    "generation_fallback_reason",
    "internal_knowledge_used",
}
SOURCE_REMOVED_RETRIEVAL_FIELDS = {
    "fallback_reason",
    "fallback_used",
    "lexical_fallback",
}
SOURCE_HYBRID_SEARCH_MODES = {
    "hybrid": "hybrid",
    "hybrid_no_dense_match": "hybrid",
    "lexical_route": "lexical",
}


JudgeFunction = Callable[[list[dict[str, str]]], str]
CheckpointFunction = Callable[[], None]


class QualitativeJudgeInputError(ValueError):
    """Raised when a source or resume artifact violates this contract."""


class QualitativeJudgeResponseError(ValueError):
    """Raised when the judge does not return the requested JSON."""


class QualitativeJudgeBackendError(RuntimeError):
    """Raised when the Codex CLI backend cannot judge safely."""


def _validated_source_model_record(
    value: Any,
    *,
    label: str,
    backend: str,
) -> dict[str, Any]:
    """Validate one immutable model checkpoint record from the source run."""

    if not isinstance(value, Mapping):
        raise QualitativeJudgeInputError(f"Input run.{label} is missing")
    path = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(path, str) or not path.strip():
        raise QualitativeJudgeInputError(
            f"Input run.{label}.path must be nonempty text"
        )
    if value.get("checkpoint_exists") is not True or value.get("kind") != "file":
        raise QualitativeJudgeInputError(
            f"Input run.{label} must identify an available model file"
        )
    if value.get("backend") != backend:
        raise QualitativeJudgeInputError(
            f"Input run.{label}.backend must be {backend}"
        )
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise QualitativeJudgeInputError(
            f"Input run.{label}.sha256 must be a SHA-256 digest"
        )
    return dict(value)


def _same_resolved_path(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not left.strip():
        return False
    if not isinstance(right, str) or not right.strip():
        return False
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _removed_alternate_field_paths(
    value: Any,
    *,
    prefix: str = "response",
) -> list[str]:
    """Find obsolete alternate-path metadata anywhere in a response."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if isinstance(key, str) and "fallback" in key.casefold():
                found.append(child_path)
            found.extend(
                _removed_alternate_field_paths(child, prefix=child_path)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _removed_alternate_field_paths(
                    child,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return found


def _validate_source_retrieval(
    value: Any,
    *,
    case_id: str,
    expected_embedding_model_path: str,
) -> None:
    if not isinstance(value, Mapping):
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline retrieval diagnostics are missing"
        )
    removed = SOURCE_REMOVED_RETRIEVAL_FIELDS & set(value)
    if removed:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline retrieval diagnostics use removed fields: "
            + ", ".join(sorted(removed))
        )
    if value.get("configured_mode") != "hybrid":
        raise QualitativeJudgeInputError(
            f"{case_id}: configured retrieval mode must be hybrid"
        )
    if value.get("active_mode") != "hybrid":
        raise QualitativeJudgeInputError(
            f"{case_id}: active retrieval mode must be hybrid"
        )
    if value.get("dense_available") is not True:
        raise QualitativeJudgeInputError(
            f"{case_id}: dense retrieval must be available"
        )
    if value.get("last_dense_error") is not None:
        raise QualitativeJudgeInputError(
            f"{case_id}: dense retrieval reported an error"
        )
    if not _same_resolved_path(
        value.get("embedding_model"),
        expected_embedding_model_path,
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: retrieval embedding model path does not match run metadata"
        )
    dimension = value.get("embedding_dimension")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 1
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: retrieval embedding dimension must be positive"
        )
    last_search_mode = value.get("last_search_mode")
    expected_route = SOURCE_HYBRID_SEARCH_MODES.get(last_search_mode)
    if expected_route is None:
        raise QualitativeJudgeInputError(
            f"{case_id}: invalid hybrid search mode {last_search_mode!r}"
        )
    if value.get("query_route") != expected_route:
        raise QualitativeJudgeInputError(
            f"{case_id}: query route is inconsistent with hybrid search mode"
        )
    for field in ("dense_attempted", "dense_contributed"):
        if not isinstance(value.get(field), bool):
            raise QualitativeJudgeInputError(
                f"{case_id}: retrieval {field} must be boolean"
            )
    dense_attempted = value["dense_attempted"]
    dense_contributed = value["dense_contributed"]
    if expected_route == "hybrid" and dense_attempted is not True:
        raise QualitativeJudgeInputError(
            f"{case_id}: hybrid query route must attempt dense retrieval"
        )
    if last_search_mode == "hybrid" and dense_contributed is not True:
        raise QualitativeJudgeInputError(
            f"{case_id}: hybrid retrieval result must include dense contribution"
        )
    if expected_route == "lexical" and (
        dense_attempted is not False or dense_contributed is not False
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: lexical query route cannot report dense work"
        )
    if last_search_mode == "hybrid_no_dense_match" and dense_contributed:
        raise QualitativeJudgeInputError(
            f"{case_id}: no-dense-match route cannot report dense contribution"
        )
    route_reason = value.get("route_reason")
    if not isinstance(route_reason, str) or not route_reason.strip():
        raise QualitativeJudgeInputError(
            f"{case_id}: retrieval route_reason must be nonempty text"
        )
    if not isinstance(value.get("ambiguous_candidates"), list):
        raise QualitativeJudgeInputError(
            f"{case_id}: retrieval ambiguous_candidates must be a list"
        )


def _validate_completed_source_response(
    result: Any,
    *,
    case_id: str,
    inference_input: Mapping[str, Any],
    generation_model_path: str,
    embedding_model_path: str,
) -> None:
    """Require the current production RAG+LLM response contract."""

    if not isinstance(result, Mapping):
        raise QualitativeJudgeInputError(
            f"{case_id}: completed case has no pipeline_result"
        )
    removed = SOURCE_REMOVED_RESPONSE_FIELDS & set(result)
    if removed:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline response uses removed fields: "
            + ", ".join(sorted(removed))
        )
    alternate_fields = _removed_alternate_field_paths(result)
    if alternate_fields:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline response contains removed alternate-path fields: "
            + ", ".join(alternate_fields)
        )
    if result.get("pipeline") != "soprano_qa":
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline response must come from soprano_qa"
        )
    expected_piece = inference_input.get("piece_id")
    expected_range = inference_input.get("measure_range")
    expected_scope = "range" if expected_range is not None else "whole_piece"
    if result.get("piece_id") != expected_piece:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline response piece_id does not match inference input"
        )
    if result.get("measure_range") != expected_range:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline response measure_range does not match inference input"
        )
    if result.get("scope") != expected_scope:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline response scope does not match inference input"
        )
    mode = result.get("generation_mode")
    basis = result.get("answer_basis")
    if (mode, basis) not in SOURCE_GENERATED_ANSWER_PATHS:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline response used an unsupported answer path"
        )
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline answer must be nonempty text"
        )
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(item, Mapping) for item in evidence
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline evidence must be a list of objects"
        )
    unavailable_reason = result.get("unavailable_reason")
    if mode == "unavailable":
        if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
            raise QualitativeJudgeInputError(
                f"{case_id}: unavailable answer requires unavailable_reason"
            )
    elif unavailable_reason is not None:
        raise QualitativeJudgeInputError(
            f"{case_id}: grounded answer cannot have unavailable_reason"
        )
    if mode == "expert_verbatim" and not any(
        item.get("kind") == "expert"
        or item.get("evidence_type") == "expert_annotation"
        for item in evidence
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: authoritative expert answer requires expert evidence"
        )
    if mode == "llm" and not evidence:
        raise QualitativeJudgeInputError(
            f"{case_id}: retrieved-evidence LLM answer requires evidence"
        )
    for field in ("has_primary_grounding", "has_confirmed_local_examples"):
        if not isinstance(result.get(field), bool):
            raise QualitativeJudgeInputError(
                f"{case_id}: pipeline response {field} must be boolean"
            )
    if mode != "unavailable" and result["has_primary_grounding"] is not True:
        raise QualitativeJudgeInputError(
            f"{case_id}: grounded answer must report primary grounding"
        )
    selected_grounding = result.get("has_selected_range_grounding")
    if expected_range is None:
        if selected_grounding is not None:
            raise QualitativeJudgeInputError(
                f"{case_id}: whole-piece response cannot report selected-range grounding"
            )
    elif not isinstance(selected_grounding, bool):
        raise QualitativeJudgeInputError(
            f"{case_id}: ranged response must report selected-range grounding"
        )

    model = result.get("model")
    if not isinstance(model, Mapping):
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline generation model status is missing"
        )
    if "llama_cpp_available" in model:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline model status uses removed llama_cpp_available"
        )
    if model.get("backend") != "llama-cpp-python":
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline generation backend must be llama-cpp-python"
        )
    if model.get("checkpoint_exists") is not True:
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline generation model must be available"
        )
    if not _same_resolved_path(model.get("path"), generation_model_path):
        raise QualitativeJudgeInputError(
            f"{case_id}: pipeline generation model path does not match run metadata"
        )
    _validate_source_retrieval(
        result.get("retrieval"),
        case_id=case_id,
        expected_embedding_model_path=embedding_model_path,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def soprano_qa_environment_record() -> dict[str, str]:
    """Require the project's supported Conda environment."""

    environment_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    environment_prefix = os.environ.get("CONDA_PREFIX", "")
    if environment_name != "soprano-qa":
        raise RuntimeError(
            "The qualitative Codex judge must run in the soprano-qa Conda "
            "environment. Use: conda run -n soprano-qa python "
            "evaluation/judge_qualitative.py"
        )
    if not environment_prefix or Path(environment_prefix).resolve() != Path(
        sys.prefix
    ).resolve():
        raise RuntimeError(
            "The active Conda prefix does not match the current Python "
            "interpreter; run the judge with conda run -n soprano-qa."
        )
    return {
        "conda_environment": environment_name,
        "conda_prefix": str(Path(environment_prefix).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
    }


def _bounded_safe_codex_diagnostic(value: Any) -> str:
    """Return one bounded line with common credentials and URL queries hidden."""

    if value is None:
        return ""
    text = " ".join(str(value).split())
    if not text:
        return ""
    text = re.sub(
        r"(?i)\bBearer\s+[^\s,;]+",
        "Bearer <redacted>",
        text,
    )
    text = re.sub(
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
        "<redacted-openai-key>",
        text,
    )
    text = re.sub(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        r"\.[A-Za-z0-9_-]{8,}\b",
        "<redacted-token>",
        text,
    )
    text = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|"
        r"authorization|cookie)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2<redacted>",
        text,
    )
    text = re.sub(
        r"(https?://[^\s?#]+)\?[^\s#]*(?:#[^\s]*)?",
        r"\1?<redacted-query>",
        text,
    )
    if len(text) > MAX_CODEX_DIAGNOSTIC_CHARACTERS:
        text = text[: MAX_CODEX_DIAGNOSTIC_CHARACTERS - 3] + "..."
    return text


def _completed_command(
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
    input_text: str | None = None,
    expected_returncode: int = 0,
    required_output: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one Codex CLI command and enforce its exact expected outcome."""

    try:
        completed = subprocess.run(
            list(arguments),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=dict(environment) if environment is not None else None,
        )
    except subprocess.TimeoutExpired as error:
        raise QualitativeJudgeBackendError(
            f"Codex CLI timed out after {timeout_seconds} seconds"
        ) from error
    except OSError as error:
        raise QualitativeJudgeBackendError(
            f"Cannot execute Codex CLI: {error}"
        ) from error
    combined_output = "\n".join((completed.stdout, completed.stderr))
    if completed.returncode != expected_returncode:
        diagnostic = _bounded_safe_codex_diagnostic(combined_output)
        raise QualitativeJudgeBackendError(
            "Codex CLI exited with status "
            f"{completed.returncode}: {diagnostic or 'no diagnostic'}"
        )
    if required_output is not None and required_output not in combined_output:
        diagnostic = _bounded_safe_codex_diagnostic(combined_output)
        raise QualitativeJudgeBackendError(
            "Codex CLI did not produce the required preflight result: "
            f"{diagnostic or 'no diagnostic'}"
        )
    return completed


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".codex"
    )


def _configured_codex_selection(config_path: Path) -> dict[str, Any]:
    """Read and fingerprint the model/routing settings that exec will ignore."""

    if not config_path.is_file():
        return {
            "model": None,
            "reasoning_effort": None,
            "service_tier": None,
            "routing_state": {
                "present_keys": [],
                "sha256": hashlib.sha256(canonical_json_bytes({})).hexdigest(),
            },
        }
    try:
        with config_path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise QualitativeJudgeBackendError(
            f"Cannot read Codex configuration {config_path}: {error}"
        ) from error

    def optional_text(key: str) -> str | None:
        value = config.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise QualitativeJudgeBackendError(
                f"Codex configuration {key} must be nonempty text"
            )
        return value.strip()

    routing_state = {
        key: deepcopy(config[key])
        for key in CODEX_ROUTING_CONFIG_KEYS
        if key in config
    }
    return {
        "model": optional_text("model"),
        "reasoning_effort": optional_text("model_reasoning_effort"),
        "service_tier": optional_text("service_tier"),
        "routing_state": {
            "present_keys": sorted(routing_state),
            "sha256": hashlib.sha256(
                canonical_json_bytes(routing_state)
            ).hexdigest(),
        },
    }


def _catalog_models(raw: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise QualitativeJudgeBackendError(
            "Codex model catalog is not valid JSON"
        ) from error
    models = value.get("models") if isinstance(value, Mapping) else None
    if not isinstance(models, list) or not models:
        raise QualitativeJudgeBackendError("Codex model catalog is empty")
    if any(not isinstance(model, Mapping) for model in models):
        raise QualitativeJudgeBackendError(
            "Codex model catalog contains an invalid entry"
        )
    return [dict(model) for model in models]


def _select_codex_model(
    models: Sequence[Mapping[str, Any]],
    configured: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str | None, str]:
    """Require the configured model and effort to be the catalog's strongest."""

    visible = [
        model
        for model in models
        if model.get("visibility") == "list"
        and isinstance(model.get("priority"), int)
        and not isinstance(model.get("priority"), bool)
        and isinstance(model.get("slug"), str)
        and model.get("slug")
    ]
    if not visible:
        raise QualitativeJudgeBackendError(
            "Codex model catalog has no selectable model"
        )
    best_priority = min(int(model["priority"]) for model in visible)
    highest_priority = [
        model for model in visible if int(model["priority"]) == best_priority
    ]
    if len(highest_priority) != 1:
        slugs = sorted(str(model["slug"]) for model in highest_priority)
        raise QualitativeJudgeBackendError(
            "Codex model catalog has an ambiguous highest priority: "
            f"{slugs}"
        )
    expected = dict(highest_priority[0])
    expected_slug = str(expected["slug"])
    configured_slug = configured.get("model")
    if not isinstance(configured_slug, str) or not configured_slug:
        raise QualitativeJudgeBackendError(
            "Codex model must be explicitly configured"
        )
    matching = [model for model in models if model.get("slug") == configured_slug]
    if len(matching) != 1:
        raise QualitativeJudgeBackendError(
            f"Configured Codex model is unavailable: {configured_slug}"
        )
    if configured_slug != expected_slug:
        raise QualitativeJudgeBackendError(
            f"Configured Codex model {configured_slug!r} is not the unique "
            f"highest-priority visible model {expected_slug!r}"
        )
    selected = expected
    selection_source = "verified_user_config_highest_priority"

    supported_efforts = [
        item.get("effort")
        for item in selected.get("supported_reasoning_levels") or []
        if isinstance(item, Mapping) and isinstance(item.get("effort"), str)
    ]
    unknown_efforts = sorted(
        set(supported_efforts) - set(REASONING_EFFORT_ORDER)
    )
    if unknown_efforts:
        raise QualitativeJudgeBackendError(
            "Cannot rank Codex reasoning efforts: " f"{unknown_efforts}"
        )
    strongest_effort = next(
        (
            candidate
            for candidate in REASONING_EFFORT_ORDER
            if candidate in supported_efforts
        ),
        None,
    )
    if strongest_effort is None:
        raise QualitativeJudgeBackendError(
            "Selected Codex model has no supported reasoning effort"
        )
    configured_effort = configured.get("reasoning_effort")
    if not isinstance(configured_effort, str) or not configured_effort:
        raise QualitativeJudgeBackendError(
            "Codex reasoning effort must be explicitly configured"
        )
    if configured_effort not in supported_efforts:
        raise QualitativeJudgeBackendError(
            f"Configured reasoning effort {configured_effort!r} is not "
            f"supported by {selected.get('slug')}"
        )
    if configured_effort != strongest_effort:
        raise QualitativeJudgeBackendError(
            f"Configured reasoning effort {configured_effort!r} is weaker "
            f"than strongest supported effort {strongest_effort!r}"
        )
    effort = configured_effort

    service_tier = configured.get("service_tier")
    if not isinstance(service_tier, str) or not service_tier:
        raise QualitativeJudgeBackendError(
            "Codex service tier must be explicitly configured"
        )
    supported_tiers = {
        item.get("id")
        for item in selected.get("service_tiers") or []
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and item.get("id")
    }
    if service_tier not in supported_tiers:
        raise QualitativeJudgeBackendError(
            f"Configured Codex service tier {service_tier!r} is not "
            f"supported by {selected.get('slug')}"
        )
    return selected, effort, service_tier, selection_source


def _parse_codex_feature_inventory(raw: str) -> dict[str, dict[str, Any]]:
    """Parse ``codex features list`` without accepting partial output."""

    if not isinstance(raw, str) or not raw.strip():
        raise QualitativeJudgeBackendError("Codex feature inventory is empty")
    inventory: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"(\S+)\s+(.+?)\s+(true|false)", line.strip())
        if match is None:
            raise QualitativeJudgeBackendError(
                f"Codex feature inventory line {line_number} is malformed"
            )
        name, stage, enabled_text = match.groups()
        if name in inventory:
            raise QualitativeJudgeBackendError(
                f"Codex feature inventory repeats {name!r}"
            )
        inventory[name] = {
            "stage": stage,
            "enabled": enabled_text == "true",
        }
    if not inventory:
        raise QualitativeJudgeBackendError("Codex feature inventory is empty")
    return inventory


def _validate_codex_tool_features(
    inventory: Mapping[str, Mapping[str, Any]],
    effective_inventory: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail unless every available tool-capable feature is known and off."""

    required = set(CODEX_REQUIRED_DISABLED_FEATURES)
    deprecated_inert = set(CODEX_KNOWN_DEPRECATED_INERT_FEATURES)
    classified = required | deprecated_inert
    missing = sorted(
        feature
        for feature in required
        if feature not in inventory
        or inventory[feature].get("stage") in {"deprecated", "removed"}
    )
    if missing:
        raise QualitativeJudgeBackendError(
            "Required Codex tool features are unavailable for preventive "
            f"disablement: {missing}"
        )
    invalid_deprecated = sorted(
        feature
        for feature in deprecated_inert
        if feature not in inventory
        or inventory[feature].get("stage") != "deprecated"
        or inventory[feature].get("enabled") is not False
        or feature not in effective_inventory
        or effective_inventory[feature].get("stage") != "deprecated"
        or effective_inventory[feature].get("enabled") is not False
    )
    if invalid_deprecated:
        raise QualitativeJudgeBackendError(
            "Known deprecated Codex tool features are not proven inert in "
            f"both feature inventories: {invalid_deprecated}"
        )
    unclassified = sorted(
        name
        for available_inventory in (inventory, effective_inventory)
        for name, record in available_inventory.items()
        if record.get("stage") != "removed"
        and any(marker in name for marker in CODEX_TOOL_FEATURE_MARKERS)
        and name not in classified
    )
    if unclassified:
        raise QualitativeJudgeBackendError(
            "Codex exposes new tool-capable features that are not covered by "
            f"the judge feature policy: {sorted(set(unclassified))}"
        )
    ineffective = sorted(
        feature
        for feature in required
        if feature not in effective_inventory
        or effective_inventory[feature].get("enabled") is not False
    )
    if ineffective:
        raise QualitativeJudgeBackendError(
            "Codex tool features could not be disabled: " f"{ineffective}"
        )


def _minimal_codex_environment(
    *, home: Path, codex_home: Path, temporary_directory: Path
) -> dict[str, str]:
    return {
        **CODEX_MINIMAL_ENVIRONMENT,
        "HOME": str(home),
        "CODEX_HOME": str(codex_home),
        "TMPDIR": str(temporary_directory),
    }


def _copy_codex_authentication(source_auth: Path, isolated_home: Path) -> Path:
    if not source_auth.is_file():
        raise QualitativeJudgeBackendError(
            f"Codex authentication file is missing: {source_auth}"
        )
    destination = isolated_home / "auth.json"
    try:
        shutil.copyfile(source_auth, destination)
        destination.chmod(0o600)
    except OSError as error:
        raise QualitativeJudgeBackendError(
            "Cannot isolate Codex authentication for judgment"
        ) from error
    return destination


def _tool_disabled_global_arguments(executable: str) -> list[str]:
    arguments = [executable]
    for feature in CODEX_REQUIRED_DISABLED_FEATURES:
        arguments.extend(["--disable", feature])
    return arguments


def codex_runtime_record(*, timeout_seconds: int) -> dict[str, Any]:
    """Fail fast and fingerprint the configured Codex CLI judge."""

    environment = soprano_qa_environment_record()
    executable_name = shutil.which("codex")
    if not executable_name:
        raise QualitativeJudgeBackendError(
            "Codex CLI is required for qualitative judgment"
        )
    executable = Path(executable_name).resolve()
    if not executable.is_file():
        raise QualitativeJudgeBackendError(
            f"Codex CLI path is not a file: {executable}"
        )

    source_codex_home = _codex_home().resolve()
    source_auth = source_codex_home / "auth.json"
    config_path = source_codex_home / "config.toml"
    configured = _configured_codex_selection(config_path)
    with tempfile.TemporaryDirectory(
        prefix="soprano-qa-codex-preflight-"
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        temporary_root.chmod(0o700)
        isolated_home = temporary_root / "home"
        isolated_codex_home = temporary_root / "codex-home"
        isolated_tmp = temporary_root / "tmp"
        for directory in (isolated_home, isolated_codex_home, isolated_tmp):
            directory.mkdir(mode=0o700)
        _copy_codex_authentication(source_auth, isolated_codex_home)
        codex_environment = _minimal_codex_environment(
            home=isolated_home,
            codex_home=isolated_codex_home,
            temporary_directory=isolated_tmp,
        )
        version = _completed_command(
            [str(executable), "--version"],
            timeout_seconds=30,
            environment=codex_environment,
        ).stdout.strip()
        if not version:
            raise QualitativeJudgeBackendError("Codex CLI version is empty")
        login_completed = _completed_command(
            [str(executable), "login", "status"],
            timeout_seconds=30,
            environment=codex_environment,
        )
        login_status = (
            login_completed.stdout.strip() or login_completed.stderr.strip()
        )
        if not login_status:
            raise QualitativeJudgeBackendError(
                "Codex CLI authentication status is empty"
            )
        catalog_raw = _completed_command(
            [str(executable), "debug", "models"],
            timeout_seconds=60,
            environment=codex_environment,
        ).stdout
        models = _catalog_models(catalog_raw)
        selected, effort, service_tier, selection_source = (
            _select_codex_model(models, configured)
        )
        selected_slug = selected.get("slug")
        if not isinstance(selected_slug, str) or not selected_slug:
            raise QualitativeJudgeBackendError(
                "Selected Codex model has no slug"
            )
        feature_raw = _completed_command(
            [str(executable), "features", "list"],
            timeout_seconds=30,
            environment=codex_environment,
        ).stdout
        feature_inventory = _parse_codex_feature_inventory(feature_raw)
        disabled_arguments = _tool_disabled_global_arguments(str(executable))
        effective_feature_raw = _completed_command(
            [*disabled_arguments, "features", "list"],
            timeout_seconds=30,
            environment=codex_environment,
        ).stdout
        effective_feature_inventory = _parse_codex_feature_inventory(
            effective_feature_raw
        )
        _validate_codex_tool_features(
            feature_inventory,
            effective_feature_inventory,
        )
        strict_preflight_arguments = [*disabled_arguments, "--strict-config"]
        preflight_overrides = [
            *CODEX_STATIC_CONFIG_OVERRIDES,
            f'model_reasoning_effort="{effort}"',
            f'service_tier="{service_tier}"',
        ]
        for override in preflight_overrides:
            strict_preflight_arguments.extend(["-c", override])
        _completed_command(
            [
                *strict_preflight_arguments,
                "exec",
                "--model",
                selected_slug,
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
            ],
            timeout_seconds=30,
            environment=codex_environment,
            input_text="",
            expected_returncode=1,
            required_output="No prompt provided via stdin.",
        )

    catalog_record = {
        "slug": selected_slug,
        "display_name": selected.get("display_name"),
        "description": selected.get("description"),
        "priority": selected.get("priority"),
        "visibility": selected.get("visibility"),
        "supported_reasoning_efforts": [
            item.get("effort")
            for item in selected.get("supported_reasoning_levels") or []
            if isinstance(item, Mapping)
        ],
        "service_tiers": [
            item.get("id")
            for item in selected.get("service_tiers") or []
            if isinstance(item, Mapping)
        ],
    }
    selection_record = {
        "model": selected_slug,
        "reasoning_effort": effort,
        "service_tier": service_tier,
    }
    return {
        "backend": "codex-cli",
        "cli": {
            "path": str(executable),
            "version": version,
            "sha256": file_sha256(executable),
            "login_status": login_status,
        },
        "model": {
            "slug": selected_slug,
            "selection_source": selection_source,
            "catalog": catalog_record,
            "catalog_entry_sha256": hashlib.sha256(
                canonical_json_bytes(selected)
            ).hexdigest(),
        },
        "settings": {
            "reasoning_effort": effort,
            "service_tier": service_tier,
            "timeout_seconds": timeout_seconds,
        },
        "invocation": {
            "strict_config": True,
            "config_overrides": list(CODEX_STATIC_CONFIG_OVERRIDES),
            "disabled_features": list(CODEX_REQUIRED_DISABLED_FEATURES),
            "known_deprecated_inert_features": list(
                CODEX_KNOWN_DEPRECATED_INERT_FEATURES
            ),
            "exec_flags": list(CODEX_EXEC_FLAGS),
        },
        "feature_policy": {
            "inventory_sha256": hashlib.sha256(
                canonical_json_bytes(feature_inventory)
            ).hexdigest(),
            "tool_feature_inventory": {
                feature: deepcopy(feature_inventory[feature])
                for feature in CODEX_REQUIRED_DISABLED_FEATURES
            },
            "deprecated_inert_feature_inventory": {
                feature: {
                    "base": deepcopy(feature_inventory[feature]),
                    "effective": deepcopy(effective_feature_inventory[feature]),
                }
                for feature in CODEX_KNOWN_DEPRECATED_INERT_FEATURES
            },
            "all_required_features_effectively_disabled": True,
            "all_known_deprecated_features_inert": True,
        },
        "configuration": {
            "path": str(config_path),
            "routing_state": deepcopy(configured["routing_state"]),
            "exec_ignores_user_config": True,
        },
        "authentication": {
            "source_path": str(source_auth),
            "isolated_copy_only": True,
        },
        "selection_sha256": hashlib.sha256(
            canonical_json_bytes(selection_record)
        ).hexdigest(),
        "environment": {
            "python": environment,
            "codex_subprocess_policy": deepcopy(CODEX_ENVIRONMENT_POLICY),
        },
        "tool_policy": "disabled_and_event_audited",
        "session_persistence": "ephemeral",
    }


def validate_source_artifact(source: Mapping[str, Any]) -> None:
    if source.get("artifact_type") != SOURCE_ARTIFACT_TYPE:
        raise QualitativeJudgeInputError(
            "Input is not a five-piece qualitative RAG+LLM artifact"
        )
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise QualitativeJudgeInputError(
            "Input qualitative artifact schema is not current; expected "
            f"{SOURCE_SCHEMA_VERSION}"
        )
    if not isinstance(source.get("questions"), list):
        raise QualitativeJudgeInputError("Input questions must be a list")
    source_run = source.get("run")
    if not isinstance(source_run, Mapping):
        raise QualitativeJudgeInputError("Input run metadata is missing")
    if source_run.get("generate") is not True:
        raise QualitativeJudgeInputError(
            "Input must contain mandatory RAG+LLM generation results"
        )
    if source_run.get("target_pieces") != list(PIECE_IDS):
        raise QualitativeJudgeInputError(
            "Input run must target the canonical five-piece set"
        )
    if source_run.get("status") not in {
        "not_started",
        "in_progress",
        "completed",
        "complete_with_errors",
    }:
        raise QualitativeJudgeInputError("Input run status is invalid")
    if "allow_internal_knowledge" in source_run:
        raise QualitativeJudgeInputError(
            "Input run uses removed allow_internal_knowledge metadata"
        )
    embedding_model = _validated_source_model_record(
        source_run.get("retrieval_embedding_model"),
        label="retrieval_embedding_model",
        backend="llama-cpp-embedding",
    )
    generation_model = _validated_source_model_record(
        source_run.get("generation_model"),
        label="generation_model",
        backend="llama-cpp-python",
    )

    seen_case_ids: set[str] = set()
    for question in source["questions"]:
        if not isinstance(question, Mapping):
            raise QualitativeJudgeInputError("Input question must be an object")
        for key in (
            "source_id",
            "piece_id",
            "paraphrased_question",
            "reference_material",
            "inference_runs",
        ):
            if key not in question:
                raise QualitativeJudgeInputError(
                    f"Input question is missing {key}"
                )
        if question["piece_id"] not in PIECE_IDS:
            raise QualitativeJudgeInputError(
                f"Unsupported piece_id {question['piece_id']}"
            )
        linked_units = _linked_unit_index(
            question.get("reference_material"),
            source_id=question["source_id"],
        )
        if not isinstance(question["inference_runs"], list):
            raise QualitativeJudgeInputError(
                f"{question['source_id']}: inference_runs must be a list"
            )
        for case in question["inference_runs"]:
            case_id = case.get("case_id") if isinstance(case, Mapping) else None
            if not isinstance(case_id, str) or not case_id:
                raise QualitativeJudgeInputError("Input case_id is missing")
            if case_id in seen_case_ids:
                raise QualitativeJudgeInputError(
                    f"Duplicate input case_id {case_id}"
                )
            seen_case_ids.add(case_id)
            if not isinstance(case.get("inference_input"), Mapping):
                raise QualitativeJudgeInputError(
                    f"{case_id}: inference_input is missing"
                )
            inference_input = case["inference_input"]
            if inference_input.get("piece_id") != question["piece_id"]:
                raise QualitativeJudgeInputError(
                    f"{case_id}: inference piece_id does not match question"
                )
            status = case.get("status")
            if status not in {"pending", "completed", "error"}:
                raise QualitativeJudgeInputError(
                    f"{case_id}: inference status is invalid"
                )
            if status == "completed":
                _validate_completed_source_response(
                    case.get("pipeline_result"),
                    case_id=case_id,
                    inference_input=inference_input,
                    generation_model_path=str(generation_model["path"]),
                    embedding_model_path=str(embedding_model["path"]),
                )
            elif case.get("pipeline_result") is not None:
                raise QualitativeJudgeInputError(
                    f"{case_id}: unfinished case cannot retain a pipeline_result"
                )
            applicable_ids = _case_applicable_ids(case, case_id=case_id)
            expected_roles = _case_reference_roles(
                case,
                case_id=case_id,
                applicable_ids=applicable_ids,
            )
            _validated_case_reference_authority(
                inference_input.get("case_reference_authority"),
                case_id=case_id,
                linked_units=linked_units,
                applicable_ids=applicable_ids,
                expected_roles=expected_roles,
            )


def _linked_unit_index(
    reference: Any,
    *,
    source_id: str,
) -> dict[str, Mapping[str, Any]]:
    """Index linked KUs for validation without copying them to judge packets."""

    if not isinstance(reference, Mapping) or not isinstance(
        reference.get("linked_knowledge_units"), list
    ):
        raise QualitativeJudgeInputError(
            f"{source_id}: linked knowledge units are missing"
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, unit in enumerate(reference["linked_knowledge_units"]):
        if not isinstance(unit, Mapping):
            raise QualitativeJudgeInputError(
                f"{source_id}: linked knowledge unit {index} is not an object"
            )
        unit_id = unit.get("knowledge_unit_id")
        if not isinstance(unit_id, str) or not unit_id:
            raise QualitativeJudgeInputError(
                f"{source_id}: linked knowledge unit {index} has no ID"
            )
        if unit_id in indexed:
            raise QualitativeJudgeInputError(
                f"{source_id}: duplicate linked knowledge unit {unit_id}"
            )
        indexed[unit_id] = unit
    return indexed


def _case_applicable_ids(case: Mapping[str, Any], *, case_id: str) -> list[str]:
    expected = case.get("expected_retrieval")
    applicable_ids = (
        expected.get("range_applicable_knowledge_unit_ids")
        if isinstance(expected, Mapping)
        else None
    )
    if (
        not isinstance(applicable_ids, list)
        or any(
            not isinstance(unit_id, str) or not unit_id
            for unit_id in applicable_ids
        )
        or len(set(applicable_ids)) != len(applicable_ids)
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: applicable knowledge-unit IDs are missing or invalid"
        )
    return list(applicable_ids)


def _case_reference_roles(
    case: Mapping[str, Any],
    *,
    case_id: str,
    applicable_ids: Sequence[str],
) -> dict[str, str]:
    """Validate the runner's semantic target/support partition."""

    inference_input = case.get("inference_input")
    expected = case.get("expected_retrieval")
    if not isinstance(inference_input, Mapping) or not isinstance(
        expected,
        Mapping,
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: semantic reference role metadata is missing"
        )

    values: dict[str, list[str]] = {}
    for field in (
        "semantic_required_any_knowledge_unit_ids",
        "semantic_supporting_knowledge_unit_ids",
    ):
        inference_value = inference_input.get(field)
        expected_value = expected.get(field)
        if (
            not isinstance(inference_value, list)
            or any(
                not isinstance(unit_id, str) or not unit_id
                for unit_id in inference_value
            )
            or len(set(inference_value)) != len(inference_value)
            or expected_value != inference_value
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: {field} is missing, invalid, or inconsistent"
            )
        values[field] = list(inference_value)

    required = values["semantic_required_any_knowledge_unit_ids"]
    supporting = values["semantic_supporting_knowledge_unit_ids"]
    required_set = set(required)
    supporting_set = set(supporting)
    applicable_set = set(applicable_ids)
    is_semantic_cohort = (
        inference_input.get("evaluation_cohort")
        == MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
    )
    if is_semantic_cohort:
        if (
            not required
            or required_set & supporting_set
            or required_set | supporting_set != applicable_set
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: semantic required-any and supporting IDs must "
                "partition the applicable knowledge units"
            )
    elif required or supporting:
        raise QualitativeJudgeInputError(
            f"{case_id}: semantic reference roles require the no-range "
            "semantic cohort"
        )
    else:
        required = list(applicable_ids)
        required_set = set(required)

    return {
        unit_id: (
            "required_any" if unit_id in required_set else "supporting"
        )
        for unit_id in applicable_ids
    }


def _validated_measure_ranges(value: Any, *, label: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise QualitativeJudgeInputError(f"{label}: expected a list")
    ranges: list[list[int]] = []
    for index, measure_range in enumerate(value):
        if (
            not isinstance(measure_range, list)
            or len(measure_range) != 2
            or any(
                isinstance(bound, bool) or not isinstance(bound, int)
                for bound in measure_range
            )
            or measure_range[0] < 1
            or measure_range[1] < measure_range[0]
        ):
            raise QualitativeJudgeInputError(
                f"{label}[{index}]: expected a positive inclusive range"
            )
        ranges.append(list(measure_range))
    if len({tuple(measure_range) for measure_range in ranges}) != len(ranges):
        raise QualitativeJudgeInputError(
            f"{label}: duplicate canonical measure ranges"
        )
    return ranges


def _validated_case_reference_authority(
    value: Any,
    *,
    case_id: str,
    linked_units: Mapping[str, Mapping[str, Any]],
    applicable_ids: Sequence[str],
    expected_roles: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and whitelist the only authority visible to the judge."""

    if not isinstance(value, Mapping):
        raise QualitativeJudgeInputError(
            f"{case_id}: case_reference_authority is missing"
        )
    if value.get("source") != "applicable_finalized_knowledge_units":
        raise QualitativeJudgeInputError(
            f"{case_id}: case reference source is not finalized knowledge units"
        )
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise QualitativeJudgeInputError(
            f"{case_id}: case reference authority has no items"
        )

    validated_items: list[dict[str, Any]] = []
    seen_unit_ids: set[str] = set()
    seen_reference_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise QualitativeJudgeInputError(
                f"{case_id}: authority item {index} is not an object"
            )
        unit_id = item.get("knowledge_unit_id")
        reference_id = item.get("reference_id")
        text = item.get("text")
        reference_role = item.get("reference_role")
        item_measure_status = item.get("measure_status")
        if not isinstance(unit_id, str) or not unit_id:
            raise QualitativeJudgeInputError(
                f"{case_id}: authority item {index} has no knowledge-unit ID"
            )
        if unit_id in seen_unit_ids:
            raise QualitativeJudgeInputError(
                f"{case_id}: duplicate authority knowledge unit {unit_id}"
            )
        if not isinstance(reference_id, str) or not reference_id:
            raise QualitativeJudgeInputError(
                f"{case_id}: authority item {index} has no reference ID"
            )
        if reference_id in seen_reference_ids:
            raise QualitativeJudgeInputError(
                f"{case_id}: duplicate authority reference ID {reference_id}"
            )
        if not isinstance(text, str) or not text.strip():
            raise QualitativeJudgeInputError(
                f"{case_id}: authority knowledge unit {unit_id} has no answer"
            )
        if (
            item.get("scope") != "curated_knowledge_unit_answer"
            or item.get("scope_authority") != "finalized_knowledge_unit"
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: authority knowledge unit {unit_id} is not finalized"
            )
        if (
            reference_role not in REFERENCE_ROLES
            or expected_roles.get(unit_id) != reference_role
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: authority knowledge unit {unit_id} has an "
                "invalid or inconsistent reference role"
            )
        item_measure_ranges = _validated_measure_ranges(
            item.get("measure_ranges"),
            label=(
                f"{case_id}.case_reference_authority.{unit_id}.measure_ranges"
            ),
        )
        linked_unit = linked_units.get(unit_id)
        if linked_unit is None:
            raise QualitativeJudgeInputError(
                f"{case_id}: authority knowledge unit {unit_id} is not linked"
            )
        rewrite_status = linked_unit.get("rewrite_status")
        linked_measure_status = linked_unit.get("measure_status")
        if (
            rewrite_status != "ready"
            or linked_measure_status not in FINAL_MEASURE_STATUSES
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: linked knowledge unit {unit_id} is not finalized "
                f"(rewrite_status={rewrite_status!r}, "
                f"measure_status={linked_measure_status!r})"
            )
        linked_answer = linked_unit.get("answer")
        if not isinstance(linked_answer, str) or not linked_answer.strip():
            raise QualitativeJudgeInputError(
                f"{case_id}: linked knowledge unit {unit_id} has no answer"
            )
        if text.strip() != linked_answer.strip():
            raise QualitativeJudgeInputError(
                f"{case_id}: authority answer does not match linked "
                f"knowledge unit {unit_id}"
            )
        linked_measure_ranges = _validated_measure_ranges(
            linked_unit.get("measure_ranges"),
            label=f"{case_id}.linked_knowledge_unit.{unit_id}.measure_ranges",
        )
        if (
            item_measure_status != linked_measure_status
            or item_measure_ranges != linked_measure_ranges
            or (
                item_measure_status == "specific"
                and not item_measure_ranges
            )
            or (
                item_measure_status != "specific"
                and item_measure_ranges
            )
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: authority scope metadata does not match linked "
                f"knowledge unit {unit_id}"
            )
        seen_unit_ids.add(unit_id)
        seen_reference_ids.add(reference_id)
        validated_items.append(
            {
                "reference_id": reference_id,
                "knowledge_unit_id": unit_id,
                "text": text.strip(),
                "scope": "curated_knowledge_unit_answer",
                "scope_authority": "finalized_knowledge_unit",
                "reference_role": reference_role,
                "measure_status": item_measure_status,
                "measure_ranges": item_measure_ranges,
            }
        )
    authority_ids = [item["knowledge_unit_id"] for item in validated_items]
    if authority_ids != list(applicable_ids):
        raise QualitativeJudgeInputError(
            f"{case_id}: authority knowledge units do not match the "
            "range-applicable knowledge units"
        )
    return {
        "source": "applicable_finalized_knowledge_units",
        "items": validated_items,
    }


def build_judgment_cases(
    source: Mapping[str, Any],
    *,
    cohort: str | None = None,
) -> list[dict[str, Any]]:
    """Build immutable judge packets from the qualitative inference artifact."""

    if cohort is not None and cohort not in SUPPORTED_COHORTS:
        raise QualitativeJudgeInputError(
            f"Unsupported qualitative judgment cohort {cohort!r}"
        )

    validate_source_artifact(source)
    judgments: list[dict[str, Any]] = []
    for question in source["questions"]:
        linked_units = _linked_unit_index(
            question.get("reference_material"),
            source_id=question["source_id"],
        )
        for inference_case in question["inference_runs"]:
            case_id = inference_case["case_id"]
            inference_input = inference_case["inference_input"]
            case_cohort = inference_input.get("evaluation_cohort")
            if case_cohort is not None and (
                not isinstance(case_cohort, str) or not case_cohort
            ):
                raise QualitativeJudgeInputError(
                    f"{case_id}: evaluation_cohort must be nonempty text"
                )
            if cohort is not None and case_cohort != cohort:
                continue
            if (
                cohort == MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
                and inference_input.get("measure_range") is not None
            ):
                raise QualitativeJudgeInputError(
                    f"{case_id}: {cohort} cases must not select a measure "
                    "range"
                )
            inference_status = inference_case.get("status")
            pipeline_result = inference_case.get("pipeline_result")
            candidate_answer: str | None = None
            if inference_status == "completed":
                if not isinstance(pipeline_result, Mapping):
                    raise QualitativeJudgeInputError(
                        f"{case_id}: completed case has no pipeline_result"
                    )
                answer = pipeline_result.get("answer")
                if not isinstance(answer, str):
                    raise QualitativeJudgeInputError(
                        f"{case_id}: pipeline answer must be text"
                    )
                candidate_answer = answer
            applicable_ids = _case_applicable_ids(
                inference_case,
                case_id=case_id,
            )
            expected_roles = _case_reference_roles(
                inference_case,
                case_id=case_id,
                applicable_ids=applicable_ids,
            )
            authority = _validated_case_reference_authority(
                inference_input.get("case_reference_authority"),
                case_id=case_id,
                linked_units=linked_units,
                applicable_ids=applicable_ids,
                expected_roles=expected_roles,
            )

            judgments.append(
                {
                    "case_id": case_id,
                    "source_id": question["source_id"],
                    "piece_id": question["piece_id"],
                    "question": inference_input.get(
                        "question", question["paraphrased_question"]
                    ),
                    "original_question": question.get("original_question", ""),
                    "measure_range": deepcopy(
                        inference_input.get("measure_range")
                    ),
                    "evaluation_cohort": case_cohort,
                    "candidate_answer": candidate_answer,
                    "case_reference_authority": authority,
                    "source_inference_status": inference_status,
                    "judgment": {
                        "status": (
                            "pending"
                            if inference_status == "completed"
                            else "unavailable"
                        ),
                        "verdict": None,
                        "direct_answer": None,
                        "faithful_to_authority": None,
                        "drift_types": [],
                        "material_differences": [],
                        "reason": None,
                        "attempt_count": 0,
                        "attempts": [],
                        "started_at": None,
                        "completed_at": None,
                        "error": None,
                    },
                }
            )

    if cohort is not None:
        if not judgments:
            raise QualitativeJudgeInputError(
                f"Judgment cohort {cohort!r} has no cases"
            )
        selected_pieces = {case["piece_id"] for case in judgments}
        if cohort == MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT and (
            selected_pieces != set(PIECE_IDS)
        ):
            missing = [
                piece_id
                for piece_id in PIECE_IDS
                if piece_id not in selected_pieces
            ]
            raise QualitativeJudgeInputError(
                f"Judgment cohort {cohort!r} must cover all five pieces; "
                f"missing {missing}"
            )
    return judgments


def judgment_selection_record(
    source: Mapping[str, Any],
    *,
    cohort: str | None,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe the immutable source-to-judge case selection."""

    source_case_count = sum(
        len(question.get("inference_runs") or [])
        for question in source.get("questions") or []
        if isinstance(question, Mapping)
    )
    selected_piece_ids = [
        piece_id
        for piece_id in PIECE_IDS
        if any(case.get("piece_id") == piece_id for case in cases)
    ]
    return {
        "mode": "cohort" if cohort is not None else "all_cases",
        "cohort": cohort,
        "source_case_count": source_case_count,
        "selected_case_count": len(cases),
        "selected_piece_ids": selected_piece_ids,
        "required_piece_ids": (
            list(PIECE_IDS)
            if cohort == MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
            else []
        ),
        "all_required_pieces_present": (
            cohort != MEASURE_SCOPED_QUESTION_NO_RANGE_COHORT
            or selected_piece_ids == list(PIECE_IDS)
        ),
    }


def build_judge_messages(case: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a retrieval-independent qualitative answer-fidelity prompt."""

    authority = case["case_reference_authority"]
    judge_visible_authority = {
        "source": authority["source"],
        "items": [
            {
                "reference_id": item["reference_id"],
                "text": item["text"],
                "reference_role": item["reference_role"],
                "measure_status": item["measure_status"],
                "measure_ranges": deepcopy(item["measure_ranges"]),
            }
            for item in authority["items"]
        ],
    }
    required_reference_ids = [
        item["reference_id"]
        for item in judge_visible_authority["items"]
        if item["reference_role"] == "required_any"
    ]
    supporting_reference_ids = [
        item["reference_id"]
        for item in judge_visible_authority["items"]
        if item["reference_role"] == "supporting"
    ]
    specific_ranges = {
        item["reference_id"]: deepcopy(item["measure_ranges"])
        for item in judge_visible_authority["items"]
        if item["measure_status"] == "specific"
    }
    payload = {
        "question": case["question"],
        "selected_measure_range": case["measure_range"],
        "case_reference_authority": judge_visible_authority,
        "judgment_contract": {
            "required_any_reference_ids": required_reference_ids,
            "supporting_reference_ids": supporting_reference_ids,
            "supporting_only_without_required": "fail",
            "no_selected_range_local_scope": {
                "active": (
                    case["measure_range"] is None and bool(specific_ranges)
                ),
                "canonical_ranges_by_reference_id": specific_ranges,
                "canonical_range_visibility": "judge_only_hidden_from_user",
                "candidate_must_name_canonical_range": False,
                "naturally_localized_direct_required_answer": "pass_eligible",
                "unrequested_measure_locator": "fail",
                "unrequested_measure_locator_failure_type": (
                    "presentation_scope"
                ),
                "correct_hidden_canonical_range_is_not_an_exception": True,
                "wrong_or_fabricated_range": "fail",
                "whole_piece_or_frequency_overgeneralization": "fail",
            },
        },
        "generated_answer": case["candidate_answer"],
    }
    system = """\
너는 최종 확정된 성악 전문가 지식 단위에 대한 답변 충실도를 보수적으로
판정하는 독립 심사자다. 파일, 검색, 셸, 웹 또는 다른 도구를 사용하지 말고,
오직 사용자 메시지에 포함된 질문, 범위, 최종 지식 단위 전문과 생성 답변만
판정 근거로 사용한다.

평가 기준:
1. 먼저 각 최종 지식 단위와 생성 답변을 원자 명제로 나누어 비교한다. 각
   명제의 행위자ㆍ발화자ㆍ역할, 술어ㆍ행동, 대상, 원인ㆍ목적, 가능성ㆍ당위ㆍ
   확실성, 부정, 시간ㆍ조건, 적용 범위ㆍ빈도ㆍ수량을 따로 맞춘다. 단어가 많이
   겹쳐도 이 관계 중 하나가 바뀌어 뜻이 달라지면 충실한 답변이 아니다.
2. 다음 drift_type을 명시적으로 검사하고 가장 구체적인 유형을 모두 기록한다.
   - agent_or_role: 작곡가, 가창자, 화자 등의 행위ㆍ발화 주체나 역할 변경
   - predicate: 무엇을 한다ㆍ나타낸다ㆍ바꾼다 등의 핵심 술어 변경
   - causal: 원인, 결과, 목적 또는 그 방향 변경
   - modality: 사실, 가능성, 권고, 의무, 추정 사이의 변경
   - negation: 긍정과 부정 또는 예외 조건의 변경
   - scope_or_quantifier: 곡 전체ㆍ국소 구간, 항상ㆍ때때로, 모두ㆍ일부 변경
   - entity_or_object: 행위ㆍ설명의 대상 또는 핵심 개체 변경
   - temporal_or_condition: 시점, 순서, 전제 또는 적용 조건 변경
   - unsupported_conflation: 서로 다른 근거 명제를 섞어 새 관계 생성
   - unsupported_addition: 최종 지식 단위가 뒷받침하지 않는 중요한 주장 추가
   - missing_core_meaning: 질문에 답하는 필수 의미 누락
   - question_mismatch: 질문과 다른 문제를 답함
   - presentation_scope: 요청하지 않은 내부 범위ㆍ마디 정보 노출
   - ambiguity: 실제 언어적 모호성 때문에 의미 보존을 확정할 수 없음
3. case_reference_authority.items의 text는 축약문이 아니라 최종 확정된 지식
   단위의 전문이다. reference_role=required_any 중 적어도 하나의 질문 핵심을
   생성 답변이 직접 전달해야 한다. supporting은 이를 보완할 뿐이며 supporting
   내용만 맞고 required_any 핵심이 없으면 fail이다.
4. measure_status와 measure_ranges는 범위 적합성만 판단한다. 사용자가 범위를
   선택하지 않았다는 이유로 specific 지식 단위의 신뢰도나 우선순위를 낮추지
   않는다. 우선순위는 오직 질문과 reference_role의 의미 관계로 정한다.
5. selected_measure_range가 null일 때 measure_status=specific인 항목의
   canonical measure_ranges는 심사자만 보는 내부 범위 메타데이터이며,
   사용자에게 보여 줄 후보 답변에 노출하거나 반복할 정보가 아니다.
   후보는 숫자 마디를 말할 필요가 없다. 대신 질문이 지칭한 프레이즈·표시·
   음악적 상황을 그대로 받거나, '해당 부분', '이 대목'처럼 자연스럽게
   국소화하면서 required_any의 직접 답을 충실히 전달하면 pass가 가능하다.
   질문이 마디 위치를 요청하지 않았는데 후보가 숫자 마디나 범위를 자발적으로
   제시하면 presentation_scope 실패로 fail이다. 내부 확정 범위와 정확히
   같아도 예외가 아니며, 다른 범위나 존재하지 않는 범위를 쓰면 역시 fail이다.
   국소 주장을 곡 전체·전 구간·항상·자주·많이 적용되는 사실이나 빈도로
   늘린 답변도 fail이다. measure_status=whole_piece인 항목에는 이 숨은 국소
   범위 규칙을 적용하지 않는다.
6. 원문 주석자 답변, 편집 메모, 다른 범위의 지식 단위, 검색 결과는 제공되지
   않는다. 제공되지 않은 내용을 추측하거나 누락으로 판정하지 않는다.
   다른 구간 주장을 선택 범위의 사실처럼 옮기면 fail이다.
7. 핵심 뜻을 정확히 전달하면 문장 구조나 표현이 달라도 되고, 모든 부수 예시를
   반복할 필요도 없다. 그러나 행위자와 술어를 합치거나 서로 다른 문장의 원인ㆍ
   목적을 섞어 새 뜻을 만들면 unsupported_conflation 또는 더 구체적인 유형으로
   fail이다. 예를 들어 'A가 X를 바꾸어 Y를 만든다'를 'A가 B에게 Z를 표현하라고
   말했다'로 바꾸는 것은 단순한 바꿔쓰기가 아니라 역할ㆍ술어ㆍ인과 변경이다.
8. pass는 질문에 직접 답하면서 모든 중요한 생성 명제가 최종 지식 단위에 의해
   함의되고 핵심 의미가 보존될 때만 가능하다. 중요한 의미 변경ㆍ모순ㆍ추가ㆍ
   누락은 fail이다. 행위자ㆍ역할, 술어ㆍ행동, 대상, 인과, 부정ㆍ양태, 범위ㆍ
   빈도, 시간ㆍ조건, 문장 혼합, 근거 없는 추가, 핵심 누락, 질문 불일치 또는
   presentation_scope 위반 중 하나라도 중요한 차이가 있으면 반드시
   faithful_to_authority=false 및 fail이다. review는 오직 실제 언어적 모호성
   (ambiguity)을 전문가가 확인해야 할 때만 쓰고, 이때도
   faithful_to_authority=false이다. 명백한 의미 변경을 review로 완화하지 않는다.
9. 검색 성공 여부, 예상 지식 단위 검색 여부, 검색 순위, 인용 형식은 평가하지
   않는다. 오직 최종 생성 답변의 내용만 평가한다.

출력 스키마의 모든 필드를 채운 JSON 객체 하나만 출력한다. drift_types에는
발견한 유형을 중복 없이 넣고, material_differences에는 각 차이의 생성 답변
구절, 대응 reference_id와 근거 구절, 유형, 구체적 설명을 쓴다. 모든 구절은
제공된 생성 답변 또는 해당 reference_id의 text에서 그대로 복사한다.
unsupported_addition은 candidate_span만 쓰고 reference_id/reference_span은 null,
missing_core_meaning은 candidate_span을 null로 하고 reference_id/reference_span을
쓴다. 그 밖에는 양쪽 구절을 모두 쓰되 presentation_scope만 근거 구절이 없으면
reference_id/reference_span을 함께 null로 할 수 있다.
pass이면 drift_types와 material_differences는 빈 배열이어야 한다. fail 또는
review이면 적어도 하나의 구체적 차이를 기록한다. reason은 한국어로 가장 중요한
판정 근거를 간결하게 적고, JSON 밖에 코드 펜스ㆍ설명ㆍ머리말을 쓰지 않는다."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def codex_judge_output_schema() -> dict[str, Any]:
    """Return the strict Structured Outputs schema supplied to Codex."""

    difference_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(MATERIAL_DIFFERENCE_KEYS),
        "properties": {
            "candidate_span": {"type": ["string", "null"]},
            "reference_id": {"type": ["string", "null"]},
            "reference_span": {"type": ["string", "null"]},
            "drift_type": {"type": "string", "enum": list(DRIFT_TYPES)},
            "explanation": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(JUDGE_RESPONSE_KEYS),
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "direct_answer": {"type": "boolean"},
            "faithful_to_authority": {"type": "boolean"},
            "drift_types": {
                "type": "array",
                "items": {"type": "string", "enum": list(DRIFT_TYPES)},
            },
            "material_differences": {
                "type": "array",
                "items": difference_schema,
            },
            "reason": {"type": "string"},
        },
    }


def _normalized_short_text(value: Any, *, label: str, nullable: bool = False):
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise QualitativeJudgeResponseError(f"{label} must be text")
    normalized = " ".join(value.split())
    if not normalized:
        raise QualitativeJudgeResponseError(f"{label} is empty")
    if len(normalized) > MAX_DIFFERENCE_CHARACTERS:
        raise QualitativeJudgeResponseError(
            f"{label} exceeds {MAX_DIFFERENCE_CHARACTERS} characters"
        )
    return normalized


def _normalized_source_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise QualitativeJudgeResponseError(f"{label} must be text")
    normalized = " ".join(value.split())
    if not normalized:
        raise QualitativeJudgeResponseError(f"{label} is empty")
    return normalized


def validate_judge_response(
    raw: str,
    *,
    candidate_answer: str,
    reference_texts_by_id: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise QualitativeJudgeResponseError("Judge response is empty")
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        raise QualitativeJudgeResponseError(
            f"Judge response is not one JSON object: {error.msg}"
        ) from error
    if not isinstance(value, dict) or set(value) != JUDGE_RESPONSE_KEYS:
        raise QualitativeJudgeResponseError(
            "Judge response has missing or unexpected fields"
        )
    verdict = value["verdict"]
    if verdict not in VERDICTS:
        raise QualitativeJudgeResponseError(
            f"Unsupported qualitative verdict {verdict!r}"
        )
    direct_answer = value["direct_answer"]
    faithful = value["faithful_to_authority"]
    if not isinstance(direct_answer, bool) or not isinstance(faithful, bool):
        raise QualitativeJudgeResponseError(
            "direct_answer and faithful_to_authority must be booleans"
        )
    drift_types = value["drift_types"]
    if (
        not isinstance(drift_types, list)
        or any(drift_type not in DRIFT_TYPES for drift_type in drift_types)
        or len(set(drift_types)) != len(drift_types)
    ):
        raise QualitativeJudgeResponseError(
            "drift_types must be unique supported failure types"
        )
    differences = value["material_differences"]
    if not isinstance(differences, list):
        raise QualitativeJudgeResponseError(
            "material_differences must be a list"
        )
    normalized_candidate = _normalized_source_text(
        candidate_answer,
        label="candidate_answer",
    )
    if (
        not isinstance(reference_texts_by_id, Mapping)
        or not reference_texts_by_id
    ):
        raise QualitativeJudgeResponseError(
            "reference_texts_by_id must be a nonempty mapping"
        )
    normalized_references: dict[str, str] = {}
    for reference_id, reference_text in reference_texts_by_id.items():
        if not isinstance(reference_id, str) or not reference_id:
            raise QualitativeJudgeResponseError(
                "Reference authority contains an invalid reference_id"
            )
        normalized_references[reference_id] = _normalized_source_text(
            reference_text,
            label=f"reference_texts_by_id[{reference_id!r}]",
        )
    normalized_differences: list[dict[str, Any]] = []
    for index, difference in enumerate(differences):
        label = f"material_differences[{index}]"
        if not isinstance(difference, Mapping) or set(difference) != (
            MATERIAL_DIFFERENCE_KEYS
        ):
            raise QualitativeJudgeResponseError(
                f"{label} has missing or unexpected fields"
            )
        drift_type = difference["drift_type"]
        if drift_type not in DRIFT_TYPES:
            raise QualitativeJudgeResponseError(
                f"{label}.drift_type is unsupported"
            )
        candidate_span = _normalized_short_text(
            difference["candidate_span"],
            label=f"{label}.candidate_span",
            nullable=True,
        )
        if (
            candidate_span is not None
            and candidate_span not in normalized_candidate
        ):
            raise QualitativeJudgeResponseError(
                f"{label}.candidate_span is not present in the candidate answer"
            )
        reference_id = difference["reference_id"]
        if reference_id is not None and (
            not isinstance(reference_id, str)
            or reference_id not in normalized_references
        ):
            raise QualitativeJudgeResponseError(
                f"{label}.reference_id is not in the supplied authority"
            )
        reference_span = _normalized_short_text(
            difference["reference_span"],
            label=f"{label}.reference_span",
            nullable=True,
        )
        if (reference_id is None) != (reference_span is None):
            raise QualitativeJudgeResponseError(
                f"{label}.reference_id and reference_span must both be null "
                "or both be supplied"
            )
        if (
            reference_id is not None
            and reference_span not in normalized_references[reference_id]
        ):
            raise QualitativeJudgeResponseError(
                f"{label}.reference_span is not present in {reference_id}"
            )
        if drift_type == "unsupported_addition":
            if candidate_span is None or reference_id is not None:
                raise QualitativeJudgeResponseError(
                    f"{label}: unsupported_addition requires a candidate span "
                    "and null reference fields"
                )
        elif drift_type == "missing_core_meaning":
            if candidate_span is not None or reference_id is None:
                raise QualitativeJudgeResponseError(
                    f"{label}: missing_core_meaning requires a null candidate "
                    "span and supplied reference fields"
                )
        elif drift_type == "presentation_scope":
            if candidate_span is None:
                raise QualitativeJudgeResponseError(
                    f"{label}: presentation_scope requires a candidate span"
                )
        elif candidate_span is None or reference_id is None:
            raise QualitativeJudgeResponseError(
                f"{label}: {drift_type} requires candidate and reference spans"
            )
        normalized_differences.append(
            {
                "candidate_span": candidate_span,
                "reference_id": reference_id,
                "reference_span": reference_span,
                "drift_type": drift_type,
                "explanation": _normalized_short_text(
                    difference["explanation"],
                    label=f"{label}.explanation",
                ),
            }
        )
    difference_types = {item["drift_type"] for item in normalized_differences}
    if difference_types != set(drift_types):
        raise QualitativeJudgeResponseError(
            "drift_types must exactly summarize material_differences"
        )
    if verdict == "pass":
        if (
            not direct_answer
            or not faithful
            or drift_types
            or normalized_differences
        ):
            raise QualitativeJudgeResponseError(
                "Pass requires a direct faithful answer with no material drift"
            )
    elif not normalized_differences:
        raise QualitativeJudgeResponseError(
            "Review and fail require at least one material difference"
        )
    if not direct_answer and not {
        "missing_core_meaning",
        "question_mismatch",
    }.intersection(drift_types):
        raise QualitativeJudgeResponseError(
            "A non-direct answer must identify missing_core_meaning or "
            "question_mismatch"
        )
    fail_only_drift = FAIL_ONLY_DRIFT_TYPES.intersection(drift_types)
    if fail_only_drift and (verdict != "fail" or faithful):
        raise QualitativeJudgeResponseError(
            "Material semantic drift requires fail and "
            "faithful_to_authority=false"
        )
    if verdict == "review" and (
        faithful
        or not set(drift_types).issubset(REVIEW_ELIGIBLE_DRIFT_TYPES)
    ):
        raise QualitativeJudgeResponseError(
            "Review is limited to unfaithful ambiguity"
        )
    if verdict == "fail" and faithful:
        raise QualitativeJudgeResponseError(
            "A failed answer cannot be marked faithful_to_authority"
        )
    if verdict == "fail" and not fail_only_drift:
        raise QualitativeJudgeResponseError(
            "Fail requires at least one material failure drift type"
        )
    reason = _normalized_short_text(value["reason"], label="reason")
    if len(reason) > MAX_REASON_CHARACTERS:
        raise QualitativeJudgeResponseError(
            f"Judge reason exceeds {MAX_REASON_CHARACTERS} characters"
        )
    return {
        "verdict": verdict,
        "direct_answer": direct_answer,
        "faithful_to_authority": faithful,
        "drift_types": list(drift_types),
        "material_differences": normalized_differences,
        "reason": reason,
    }


def build_input_fingerprint(
    *,
    source_judgment_input_sha256: str,
    selection: Mapping[str, Any],
    judge_runtime: Mapping[str, Any],
    implementation_sha256: str,
) -> str:
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_judgment_input_sha256": source_judgment_input_sha256,
        "selection": dict(selection),
        "judge_runtime": dict(judge_runtime),
        "implementation_sha256": implementation_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def immutable_judgment_case_payloads(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return exactly the case material that can affect a judgment.

    Checkpoint timestamps and verdicts are deliberately excluded.  The
    qualitative protocol version remains responsible for invalidating saved
    judgments when the judge instructions themselves change.
    """

    return [
        {
            key: deepcopy(value)
            for key, value in case.items()
            if key != "judgment"
        }
        for case in cases
    ]


def source_judgment_input_sha256(
    source: Mapping[str, Any],
    *,
    cohort: str | None = None,
) -> str:
    """Hash judge-visible source content, not volatile run timestamps."""

    cases = build_judgment_cases(source, cohort=cohort)
    selection = judgment_selection_record(
        source,
        cohort=cohort,
        cases=cases,
    )
    payload = {
        "artifact_type": source.get("artifact_type"),
        "schema_version": source.get("schema_version"),
        "source_input_fingerprint": source.get("run", {}).get(
            "input_fingerprint"
        ),
        "selection": selection,
        "cases": immutable_judgment_case_payloads(cases),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def new_snapshot(
    *,
    source: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    source_judgment_sha256: str | None = None,
    judge_runtime: Mapping[str, Any],
    input_fingerprint: str,
    implementation_sha256: str,
    cohort: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    cases = build_judgment_cases(source, cohort=cohort)
    selection = judgment_selection_record(
        source,
        cohort=cohort,
        cases=cases,
    )
    snapshot = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": "qualitative answer-fidelity Codex CLI judge",
            "version": PROTOCOL_VERSION,
            "purpose": (
                "Conservative pass/review/fail triage of generated answers "
                "against finalized, range-applicable knowledge-unit "
                "references with an isolated Codex CLI invocation."
            ),
            "strict_schema_2_0_automatic_pass_protocol": False,
            "issues_strict_automatic_pass_verdicts": False,
            "retrieval_exactness_is_answer_quality": False,
            "judge_independence_from_answer_generator": (
                "different_backend_established"
            ),
            "production_runtime_dependency": False,
            "human_review_remains_authoritative": True,
            "codex_invocations_per_case": "exactly_one",
            "failed_or_interrupted_case_policy": "new_output_required",
            "no_range_canonical_measure_scope_visibility": "judge_only",
            "no_range_candidate_measure_locator_required": False,
        },
        "run": {
            "status": "not_started",
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "source_artifact": {
                "path": str(source_path.resolve()),
                "sha256": source_sha256,
                "sha256_scope": "full_file_at_last_judgment",
                "judgment_input_sha256": source_judgment_sha256,
                "artifact_type": source.get("artifact_type"),
                "schema_version": source.get("schema_version"),
                "source_run_status": source.get("run", {}).get("status"),
            },
            "judge_runtime": dict(judge_runtime),
            "selection": selection,
            "input_fingerprint": input_fingerprint,
            "implementation_sha256": implementation_sha256,
            "last_minimum_pass_rate": None,
        },
        "cases": cases,
        "summary": {},
    }
    refresh_summary(snapshot)
    return snapshot


def _judgment_reference_texts(case: Mapping[str, Any]) -> dict[str, str]:
    authority = case.get("case_reference_authority")
    items = authority.get("items") if isinstance(authority, Mapping) else None
    if not isinstance(items, list) or not items:
        raise QualitativeJudgeInputError(
            f"{case.get('case_id', '<unknown>')}: persisted authority is invalid"
        )
    references: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise QualitativeJudgeInputError(
                f"{case.get('case_id', '<unknown>')}: persisted authority item "
                "is invalid"
            )
        reference_id = item.get("reference_id")
        text = item.get("text")
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in references
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise QualitativeJudgeInputError(
                f"{case.get('case_id', '<unknown>')}: persisted authority item "
                "has invalid reference text"
            )
        references[reference_id] = text
    return references


def _validate_persisted_error(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"type", "message"}:
        raise QualitativeJudgeInputError(f"{label} is invalid")
    if any(
        not isinstance(value[key], str) or not value[key].strip()
        for key in ("type", "message")
    ):
        raise QualitativeJudgeInputError(f"{label} is invalid")


def _has_empty_persisted_result(judgment: Mapping[str, Any]) -> bool:
    return all(
        (
            judgment.get("verdict") is None,
            judgment.get("direct_answer") is None,
            judgment.get("faithful_to_authority") is None,
            judgment.get("drift_types") == [],
            judgment.get("material_differences") == [],
            judgment.get("reason") is None,
        )
    )


def _validate_persisted_case_judgment(case: Mapping[str, Any]) -> None:
    """Reject forged or internally inconsistent checkpoint state."""

    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise QualitativeJudgeInputError("Persisted case_id is invalid")
    candidate_answer = case.get("candidate_answer")
    references = _judgment_reference_texts(case)
    judgment = case.get("judgment")
    if not isinstance(judgment, Mapping) or set(judgment) != JUDGMENT_KEYS:
        raise QualitativeJudgeInputError(
            f"{case_id}: persisted judgment fields are invalid"
        )
    status = judgment.get("status")
    if status not in {"pending", "unavailable", "running", "completed", "error"}:
        raise QualitativeJudgeInputError(
            f"{case_id}: persisted judgment status is invalid"
        )
    if status == "unavailable":
        if candidate_answer is not None:
            raise QualitativeJudgeInputError(
                f"{case_id}: unavailable case has a candidate answer"
            )
    elif not isinstance(candidate_answer, str) or not candidate_answer.strip():
        raise QualitativeJudgeInputError(
            f"{case_id}: persisted candidate answer is invalid"
        )
    attempt_count = judgment.get("attempt_count")
    attempts = judgment.get("attempts")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 0
        or attempt_count > 1
        or not isinstance(attempts, list)
        or attempt_count != len(attempts)
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: persisted attempt_count is inconsistent"
        )
    for index, attempt in enumerate(attempts, start=1):
        label = f"{case_id}: persisted attempt {index}"
        if not isinstance(attempt, Mapping) or set(attempt) != ATTEMPT_KEYS:
            raise QualitativeJudgeInputError(f"{label} fields are invalid")
        if attempt.get("attempt") != index:
            raise QualitativeJudgeInputError(f"{label} number is inconsistent")
        if not isinstance(attempt.get("started_at"), str) or not attempt[
            "started_at"
        ].strip():
            raise QualitativeJudgeInputError(f"{label} started_at is invalid")
        completed_at = attempt.get("completed_at")
        if completed_at is not None and (
            not isinstance(completed_at, str) or not completed_at.strip()
        ):
            raise QualitativeJudgeInputError(f"{label} completed_at is invalid")
        raw_response = attempt.get("raw_response")
        if raw_response is not None and not isinstance(raw_response, str):
            raise QualitativeJudgeInputError(f"{label} raw_response is invalid")
        if not isinstance(attempt.get("valid"), bool):
            raise QualitativeJudgeInputError(f"{label} valid flag is invalid")
        error = attempt.get("error")
        if error is not None:
            _validate_persisted_error(error, label=f"{label} error")
        if attempt["valid"] and (
            not isinstance(raw_response, str)
            or not raw_response.strip()
            or completed_at is None
            or error is not None
        ):
            raise QualitativeJudgeInputError(f"{label} valid state is invalid")
        is_running_tail = status == "running" and index == len(attempts)
        if is_running_tail:
            if attempt["valid"] or completed_at is not None or error is not None:
                raise QualitativeJudgeInputError(
                    f"{label} running state is invalid"
                )
        elif completed_at is None or (
            not attempt["valid"] and error is None
        ):
            raise QualitativeJudgeInputError(
                f"{label} completed state is invalid"
            )
        if attempt["valid"] and not (
            status == "completed" and index == len(attempts)
        ):
            raise QualitativeJudgeInputError(
                f"{label} cannot contain an earlier valid result"
            )

    empty_result = _has_empty_persisted_result(judgment)
    if status in {"pending", "unavailable"}:
        if (
            attempt_count != 0
            or not empty_result
            or judgment.get("started_at") is not None
            or judgment.get("completed_at") is not None
            or judgment.get("error") is not None
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: {status} judgment contains attempted results"
            )
        return

    if not isinstance(judgment.get("started_at"), str) or not judgment[
        "started_at"
    ].strip():
        raise QualitativeJudgeInputError(
            f"{case_id}: attempted judgment started_at is invalid"
        )
    if attempt_count < 1:
        raise QualitativeJudgeInputError(
            f"{case_id}: attempted judgment has zero attempts"
        )
    last_attempt = attempts[-1]
    if status == "running":
        if (
            not empty_result
            or judgment.get("completed_at") is not None
            or judgment.get("error") is not None
            or last_attempt.get("completed_at") is not None
            or last_attempt.get("raw_response") is not None
            or last_attempt.get("valid")
            or last_attempt.get("error") is not None
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: running judgment checkpoint is inconsistent"
            )
        return

    if status == "error":
        if (
            not empty_result
            or judgment.get("completed_at") is not None
            or judgment.get("error") is None
            or last_attempt.get("completed_at") is None
            or last_attempt.get("valid")
            or last_attempt.get("error") != judgment.get("error")
        ):
            raise QualitativeJudgeInputError(
                f"{case_id}: errored judgment checkpoint is inconsistent"
            )
        _validate_persisted_error(
            judgment["error"], label=f"{case_id}: judgment error"
        )
        return

    if (
        not isinstance(judgment.get("completed_at"), str)
        or not judgment["completed_at"].strip()
        or judgment.get("error") is not None
        or not last_attempt.get("valid")
    ):
        raise QualitativeJudgeInputError(
            f"{case_id}: completed judgment checkpoint is inconsistent"
        )
    try:
        assessment = validate_judge_response(
            last_attempt["raw_response"],
            candidate_answer=candidate_answer,
            reference_texts_by_id=references,
        )
    except QualitativeJudgeResponseError as error:
        raise QualitativeJudgeInputError(
            f"{case_id}: completed judgment raw response is invalid: {error}"
        ) from error
    if any(judgment.get(key) != assessment[key] for key in JUDGE_RESPONSE_KEYS):
        raise QualitativeJudgeInputError(
            f"{case_id}: completed judgment fields do not match raw response"
        )


def _validate_persisted_cases(snapshot: Mapping[str, Any]) -> None:
    cases = snapshot.get("cases")
    if not isinstance(cases, list):
        raise QualitativeJudgeInputError("Persisted cases must be a list")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise QualitativeJudgeInputError("Persisted case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise QualitativeJudgeInputError(
                "Persisted case IDs must be nonempty and unique"
            )
        seen.add(case_id)
        _validate_persisted_case_judgment(case)
    run = snapshot.get("run")
    if not isinstance(run, Mapping):
        raise QualitativeJudgeInputError("Persisted run record is invalid")
    expected = {"cases": deepcopy(cases), "run": {}}
    refresh_summary(expected)
    if run.get("status") != expected["run"]["status"]:
        raise QualitativeJudgeInputError(
            "Persisted run status is inconsistent with case judgments"
        )
    if snapshot.get("summary") != expected["summary"]:
        raise QualitativeJudgeInputError(
            "Persisted summary is inconsistent with case judgments"
        )


def validate_resume_snapshot(
    snapshot: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    judge_runtime: Mapping[str, Any],
    implementation_sha256: str,
    cohort: str | None = None,
) -> None:
    """Bind persisted judgments to freshly derived immutable inputs."""

    if snapshot.get("artifact_type") != ARTIFACT_TYPE:
        raise QualitativeJudgeInputError(
            "Output is not a five-piece qualitative judgment artifact"
        )
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise QualitativeJudgeInputError(
            "Output schema version is not resumable by this judge"
        )
    if snapshot.get("protocol", {}).get("version") != PROTOCOL_VERSION:
        raise QualitativeJudgeInputError(
            "Output protocol version is not resumable by this judge"
        )
    run = snapshot.get("run")
    if not isinstance(run, Mapping):
        raise QualitativeJudgeInputError("Output run record is invalid")
    expected_cases = build_judgment_cases(source, cohort=cohort)
    expected_selection = judgment_selection_record(
        source,
        cohort=cohort,
        cases=expected_cases,
    )
    expected_source_judgment_sha256 = source_judgment_input_sha256(
        source,
        cohort=cohort,
    )
    expected_fingerprint = build_input_fingerprint(
        source_judgment_input_sha256=expected_source_judgment_sha256,
        selection=expected_selection,
        judge_runtime=judge_runtime,
        implementation_sha256=implementation_sha256,
    )
    if run.get("input_fingerprint") != expected_fingerprint:
        raise QualitativeJudgeInputError(
            "Inference artifact or Codex judge configuration changed; "
            "choose a new --output path rather than mixing judgments"
        )
    _validate_persisted_cases(snapshot)
    if run.get("judge_runtime") != dict(judge_runtime):
        raise QualitativeJudgeInputError(
            "Persisted Codex runtime does not match fresh preflight"
        )
    if run.get("implementation_sha256") != implementation_sha256:
        raise QualitativeJudgeInputError(
            "Persisted judge implementation hash does not match"
        )
    if run.get("selection") != expected_selection:
        raise QualitativeJudgeInputError(
            "Persisted case selection does not match fresh source selection"
        )
    source_record = run.get("source_artifact")
    if not isinstance(source_record, Mapping) or source_record.get(
        "judgment_input_sha256"
    ) != expected_source_judgment_sha256:
        raise QualitativeJudgeInputError(
            "Persisted source judgment hash does not match fresh source"
        )
    if immutable_judgment_case_payloads(
        snapshot.get("cases") or []
    ) != immutable_judgment_case_payloads(expected_cases):
        raise QualitativeJudgeInputError(
            "Persisted immutable case payloads do not match fresh source"
        )


def can_migrate_semantically_identical_resume(
    snapshot: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    judge_runtime: Mapping[str, Any],
    implementation_sha256: str,
    cohort: str | None = None,
) -> bool:
    """Allow old full-file hashes to migrate after timestamp-only changes."""

    if snapshot.get("artifact_type") != ARTIFACT_TYPE:
        return False
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        return False
    if snapshot.get("protocol", {}).get("version") != PROTOCOL_VERSION:
        return False
    try:
        _validate_persisted_cases(snapshot)
    except QualitativeJudgeInputError:
        return False
    run = snapshot.get("run", {})
    if run.get("judge_runtime") != dict(judge_runtime):
        return False
    if run.get("implementation_sha256") != implementation_sha256:
        return False
    try:
        expected_cases = build_judgment_cases(source, cohort=cohort)
    except QualitativeJudgeInputError:
        return False
    expected_selection = judgment_selection_record(
        source,
        cohort=cohort,
        cases=expected_cases,
    )
    if run.get("selection") != expected_selection:
        return False
    expected_source_judgment_sha256 = source_judgment_input_sha256(
        source,
        cohort=cohort,
    )
    if run.get("source_artifact", {}).get(
        "judgment_input_sha256"
    ) != expected_source_judgment_sha256:
        return False
    return immutable_judgment_case_payloads(
        snapshot.get("cases") or []
    ) == immutable_judgment_case_payloads(expected_cases)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _piece_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(case["judgment"]["status"] for case in cases)
    verdicts = Counter(
        case["judgment"]["verdict"]
        for case in cases
        if case["judgment"]["status"] == "completed"
    )
    drift_types = Counter(
        drift_type
        for case in cases
        if case["judgment"]["status"] == "completed"
        for drift_type in case["judgment"].get("drift_types") or []
    )
    cases_with_drift = sum(
        1
        for case in cases
        if case["judgment"]["status"] == "completed"
        and bool(case["judgment"].get("drift_types"))
    )
    judgeable = len(cases) - statuses["unavailable"]
    completed = statuses["completed"]
    piece_complete = (
        bool(judgeable)
        and completed == judgeable
        and not statuses["unavailable"]
    )
    return {
        "case_count": len(cases),
        "judgeable_case_count": judgeable,
        "judgment_statuses": dict(sorted(statuses.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "drift_types": dict(sorted(drift_types.items())),
        "cases_with_drift": cases_with_drift,
        "completed_case_pass_rate": _rate(verdicts["pass"], completed),
        "final_case_pass_rate": (
            _rate(verdicts["pass"], judgeable) if piece_complete else None
        ),
        "complete": piece_complete,
    }


def refresh_summary(snapshot: dict[str, Any]) -> None:
    cases = snapshot.get("cases") or []
    statuses = Counter(case["judgment"]["status"] for case in cases)
    verdicts = Counter(
        case["judgment"]["verdict"]
        for case in cases
        if case["judgment"]["status"] == "completed"
    )
    drift_types = Counter(
        drift_type
        for case in cases
        if case["judgment"]["status"] == "completed"
        for drift_type in case["judgment"].get("drift_types") or []
    )
    cases_with_drift = sum(
        1
        for case in cases
        if case["judgment"]["status"] == "completed"
        and bool(case["judgment"].get("drift_types"))
    )
    judgeable = len(cases) - statuses["unavailable"]
    completed = statuses["completed"]

    if statuses["pending"] or statuses["running"]:
        attempted = completed + statuses["error"] + statuses["running"]
        run_status = "in_progress" if attempted else "not_started"
    elif statuses["error"]:
        run_status = "complete_with_errors"
    elif statuses["unavailable"]:
        run_status = "input_incomplete"
    elif judgeable and completed == judgeable:
        run_status = "completed"
    else:
        run_status = "input_incomplete"

    by_piece = {
        piece_id: _piece_summary(
            [case for case in cases if case["piece_id"] == piece_id]
        )
        for piece_id in PIECE_IDS
        if any(case["piece_id"] == piece_id for case in cases)
    }
    final_pass_rate = (
        _rate(verdicts["pass"], judgeable)
        if run_status == "completed"
        else None
    )
    snapshot["run"]["status"] = run_status
    snapshot["run"]["updated_at"] = utc_now()
    snapshot["run"]["finished_at"] = (
        snapshot["run"]["updated_at"]
        if run_status in {
            "completed",
            "complete_with_errors",
            "input_incomplete",
        }
        else None
    )
    snapshot["summary"] = {
        "case_count": len(cases),
        "judgeable_case_count": judgeable,
        "unavailable_case_count": statuses["unavailable"],
        "judgment_statuses": dict(sorted(statuses.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "drift_types": dict(sorted(drift_types.items())),
        "cases_with_drift": cases_with_drift,
        "completed_case_pass_rate": _rate(verdicts["pass"], completed),
        "final_case_pass_rate": final_pass_rate,
        "minimum_pass_rate_gate_eligible": run_status == "completed",
        "by_piece": by_piece,
        "metric_note": (
            "Pass rate measures qualitative generated-answer fidelity to "
            "expert references. It does not use exact retrieval recall and "
            "is not an automatic production-acceptance metric."
        ),
    }


def selected_incomplete_cases(
    snapshot: Mapping[str, Any],
    *,
    piece_ids: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    cases = list(snapshot.get("cases") or [])
    known_ids = {case["case_id"] for case in cases}
    requested_ids = set(case_ids or [])
    unknown = requested_ids - known_ids
    if unknown:
        raise QualitativeJudgeInputError(
            f"Unknown --case-id values: {sorted(unknown)}"
        )
    terminal_failures = [
        case["case_id"]
        for case in cases
        if case["judgment"]["status"] in {"running", "error"}
    ]
    if terminal_failures:
        raise QualitativeJudgeInputError(
            "A Codex judgment was interrupted or failed and cannot be "
            "resumed; choose a new --output path: "
            f"{terminal_failures}"
        )
    selected_pieces = set(piece_ids or PIECE_IDS)
    selected = [
        case
        for case in cases
        if case["piece_id"] in selected_pieces
        and (not requested_ids or case["case_id"] in requested_ids)
        and case["judgment"]["status"] == "pending"
    ]
    return selected[:limit] if limit is not None else selected


def run_judgments(
    snapshot: dict[str, Any],
    *,
    judge_fn: JudgeFunction,
    checkpoint: CheckpointFunction,
    piece_ids: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> int:
    """Judge each selected case once; any failure aborts immediately."""

    selected = selected_incomplete_cases(
        snapshot,
        piece_ids=piece_ids,
        case_ids=case_ids,
        limit=limit,
    )
    for case in selected:
        judgment = case["judgment"]
        if (
            judgment["status"] != "pending"
            or judgment["attempt_count"] != 0
            or judgment["attempts"]
        ):
            raise QualitativeJudgeInputError(
                f"{case['case_id']}: each case permits exactly one fresh "
                "Codex judgment"
            )
        if judgment["started_at"] is None:
            judgment["started_at"] = utc_now()
        judgment["status"] = "running"
        judgment["attempt_count"] = 1
        judgment["error"] = None
        judgment["verdict"] = None
        judgment["direct_answer"] = None
        judgment["faithful_to_authority"] = None
        judgment["drift_types"] = []
        judgment["material_differences"] = []
        judgment["reason"] = None
        judgment["completed_at"] = None
        attempt = {
            "attempt": judgment["attempt_count"],
            "started_at": utc_now(),
            "completed_at": None,
            "raw_response": None,
            "valid": False,
            "error": None,
        }
        judgment["attempts"].append(attempt)
        checkpoint()
        try:
            raw = judge_fn(build_judge_messages(case))
            attempt["raw_response"] = raw
            assessment = validate_judge_response(
                raw,
                candidate_answer=case["candidate_answer"],
                reference_texts_by_id=_judgment_reference_texts(case),
            )
            attempt["valid"] = True
            for key in JUDGE_RESPONSE_KEYS:
                judgment[key] = deepcopy(assessment[key])
            judgment["status"] = "completed"
            judgment["completed_at"] = utc_now()
        except Exception as error:
            attempt["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            judgment["status"] = "error"
            judgment["error"] = deepcopy(attempt["error"])
            raise
        finally:
            attempt["completed_at"] = utc_now()
            checkpoint()
    return len(selected)


def quality_gate_exit_code(
    snapshot: Mapping[str, Any],
    *,
    minimum_pass_rate: float,
) -> int:
    """Return 2 only for a complete run below the qualitative pass target."""

    if snapshot.get("run", {}).get("status") != "completed":
        return 0
    pass_rate = snapshot.get("summary", {}).get("final_case_pass_rate")
    if pass_rate is None:
        return 0
    return 2 if pass_rate < minimum_pass_rate else 0


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(canonical_json_bytes(value))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


@contextmanager
def exclusive_output_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another qualitative judge is writing {path}"
            ) from error
        yield


def _codex_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise QualitativeJudgeInputError(
            "Codex judge prompt must contain one system and one user message"
        )
    system = messages[0].get("content")
    user = messages[1].get("content")
    if not isinstance(system, str) or not isinstance(user, str):
        raise QualitativeJudgeInputError("Codex judge prompt content is invalid")
    return (
        "<judge_instructions>\n"
        f"{system}\n"
        "</judge_instructions>\n\n"
        "<judge_case>\n"
        f"{user}\n"
        "</judge_case>\n"
    )


def _codex_error_event_diagnostic(event: Mapping[str, Any]) -> str:
    """Extract only bounded, non-structural diagnostics from an error event."""

    parts: list[str] = []

    def add(label: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return
        diagnostic = _bounded_safe_codex_diagnostic(value)
        if diagnostic and diagnostic not in {part.split("=", 1)[-1] for part in parts}:
            parts.append(f"{label}={diagnostic}")

    def inspect(container: Mapping[str, Any], prefix: str = "") -> None:
        add(f"{prefix}code", container.get("code"))
        add(f"{prefix}message", container.get("message"))
        add(f"{prefix}reason", container.get("reason"))
        nested = container.get("error")
        if isinstance(nested, Mapping):
            add(f"{prefix}error.code", nested.get("code"))
            add(f"{prefix}error.type", nested.get("type"))
            add(f"{prefix}error.message", nested.get("message"))
            add(f"{prefix}error.reason", nested.get("reason"))
        else:
            add(f"{prefix}error", nested)

    inspect(event)
    item = event.get("item")
    if isinstance(item, Mapping):
        inspect(item, "item.")
    diagnostic = "; ".join(parts) or "no safe diagnostic fields"
    return _bounded_safe_codex_diagnostic(diagnostic)


def validate_codex_event_stream(raw: str) -> None:
    """Reject malformed runs, failed turns, and all tool activity."""

    if not isinstance(raw, str) or not raw.strip():
        raise QualitativeJudgeBackendError("Codex CLI emitted no JSONL events")
    completed_turns = 0
    allowed_item_types = {"reasoning", "agent_message"}
    allowed_event_types = {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
    }
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise QualitativeJudgeBackendError(
                f"Codex JSONL event {line_number} is malformed"
            ) from error
        if not isinstance(event, Mapping):
            raise QualitativeJudgeBackendError(
                f"Codex JSONL event {line_number} is not an object"
            )
        event_type = event.get("type")
        if event_type in {"error", "turn.failed"}:
            raise QualitativeJudgeBackendError(
                f"Codex CLI reported a backend failure at JSONL event "
                f"{line_number}: {_codex_error_event_diagnostic(event)}"
            )
        if event_type not in allowed_event_types:
            raise QualitativeJudgeBackendError(
                f"Codex CLI emitted an unexpected event {event_type!r}"
            )
        if event_type == "turn.completed":
            completed_turns += 1
        item = event.get("item")
        if (
            isinstance(event_type, str)
            and event_type.startswith("item.")
            and not isinstance(item, Mapping)
        ):
            raise QualitativeJudgeBackendError(
                f"Codex CLI event {event_type!r} has no item"
            )
        if isinstance(item, Mapping):
            item_type = item.get("type")
            if item_type == "error":
                raise QualitativeJudgeBackendError(
                    f"Codex CLI reported a backend failure at JSONL event "
                    f"{line_number}: {_codex_error_event_diagnostic(event)}"
                )
            if item_type not in allowed_item_types:
                raise QualitativeJudgeBackendError(
                    "Codex judge attempted a forbidden tool or item: "
                    f"{item_type!r}"
                )
    if completed_turns != 1:
        raise QualitativeJudgeBackendError(
            "Codex CLI event stream must contain exactly one completed turn"
        )


def _codex_exec_arguments(
    runtime: Mapping[str, Any],
    *,
    working_directory: Path,
    schema_path: Path,
    result_path: Path,
) -> list[str]:
    cli_path = runtime.get("cli", {}).get("path")
    model_slug = runtime.get("model", {}).get("slug")
    settings = runtime.get("settings", {})
    reasoning_effort = settings.get("reasoning_effort")
    service_tier = settings.get("service_tier")
    if not all(
        isinstance(value, str) and value
        for value in (
            cli_path,
            model_slug,
            reasoning_effort,
            service_tier,
        )
    ):
        raise QualitativeJudgeBackendError(
            "Codex runtime record is missing CLI, model, reasoning effort, "
            "or service tier"
        )

    arguments = _tool_disabled_global_arguments(str(cli_path))
    arguments.append("--strict-config")
    overrides = list(CODEX_STATIC_CONFIG_OVERRIDES)
    overrides.append(f'model_reasoning_effort="{reasoning_effort}"')
    overrides.append(f'service_tier="{service_tier}"')
    for override in overrides:
        arguments.extend(["-c", override])
    arguments.extend(
        [
            "exec",
            "--model",
            str(model_slug),
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--cd",
            str(working_directory),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(result_path),
            "--json",
            "--color",
            "never",
            "-",
        ]
    )
    return arguments


def build_codex_judge(runtime: Mapping[str, Any]) -> JudgeFunction:
    """Build the sole fail-closed, isolated Codex CLI judge backend."""

    immutable_runtime = deepcopy(dict(runtime))
    if immutable_runtime.get("backend") != "codex-cli":
        raise QualitativeJudgeBackendError(
            "Codex judge runtime must declare codex-cli as its backend"
        )
    timeout_seconds = immutable_runtime.get("settings", {}).get(
        "timeout_seconds"
    )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise QualitativeJudgeBackendError(
            "Codex runtime timeout_seconds must be positive"
        )
    if (
        immutable_runtime.get("invocation", {}).get("disabled_features")
        != list(CODEX_REQUIRED_DISABLED_FEATURES)
        or immutable_runtime.get("invocation", {}).get(
            "known_deprecated_inert_features"
        )
        != list(CODEX_KNOWN_DEPRECATED_INERT_FEATURES)
        or immutable_runtime.get("feature_policy", {}).get(
            "all_required_features_effectively_disabled"
        )
        is not True
        or immutable_runtime.get("feature_policy", {}).get(
            "all_known_deprecated_features_inert"
        )
        is not True
    ):
        raise QualitativeJudgeBackendError(
            "Codex runtime does not prove complete tool-feature disablement"
        )
    source_auth_value = immutable_runtime.get("authentication", {}).get(
        "source_path"
    )
    if not isinstance(source_auth_value, str) or not source_auth_value:
        raise QualitativeJudgeBackendError(
            "Codex runtime authentication source is missing"
        )
    source_auth = Path(source_auth_value)
    if not source_auth.is_file():
        raise QualitativeJudgeBackendError(
            f"Codex authentication file is missing: {source_auth}"
        )

    def generate(messages: list[dict[str, str]]) -> str:
        prompt = _codex_prompt(messages)
        with tempfile.TemporaryDirectory(
            prefix="soprano-qa-codex-judge-"
        ) as temporary_name:
            temporary_root = Path(temporary_name)
            temporary_root.chmod(0o700)
            working_directory = temporary_root / "empty-workspace"
            isolated_home = temporary_root / "home"
            isolated_codex_home = temporary_root / "codex-home"
            isolated_tmp = temporary_root / "tmp"
            working_directory.mkdir(mode=0o700)
            isolated_home.mkdir(mode=0o700)
            isolated_codex_home.mkdir(mode=0o700)
            isolated_tmp.mkdir(mode=0o700)
            _copy_codex_authentication(source_auth, isolated_codex_home)
            schema_path = temporary_root / "judge-output.schema.json"
            result_path = temporary_root / "judge-result.json"
            schema_path.write_bytes(
                canonical_json_bytes(codex_judge_output_schema())
            )
            environment = _minimal_codex_environment(
                home=isolated_home,
                codex_home=isolated_codex_home,
                temporary_directory=isolated_tmp,
            )
            completed = _completed_command(
                _codex_exec_arguments(
                    immutable_runtime,
                    working_directory=working_directory,
                    schema_path=schema_path,
                    result_path=result_path,
                ),
                timeout_seconds=timeout_seconds,
                environment=environment,
                input_text=prompt,
            )
            validate_codex_event_stream(completed.stdout)
            if not result_path.is_file():
                raise QualitativeJudgeBackendError(
                    "Codex CLI did not write its final structured response"
                )
            try:
                return result_path.read_text(encoding="utf-8")
            except OSError as error:
                raise QualitativeJudgeBackendError(
                    "Cannot read Codex CLI structured response"
                ) from error

    return generate


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cohort",
        choices=SUPPORTED_COHORTS,
        help=(
            "Build the judgment artifact from only this tagged source "
            "cohort. The selection is fingerprinted and resumable."
        ),
    )
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=DEFAULT_CODEX_TIMEOUT_SECONDS,
        help="Hard timeout for each isolated Codex CLI judgment.",
    )
    parser.add_argument(
        "--piece",
        action="append",
        choices=PIECE_IDS,
        help="Judge only this piece in this invocation; repeat as needed.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Judge only this stable inference case ID; repeat as needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Judge at most N incomplete selected cases in this invocation.",
    )
    parser.add_argument(
        "--minimum-pass-rate",
        type=float,
        default=0.90,
        help=(
            "Exit 2 only after a complete qualitative judgment run below "
            "this generated-answer fidelity rate."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.codex_timeout_seconds < 1:
        parser.error("--codex-timeout-seconds must be positive")
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= arguments.minimum_pass_rate <= 1:
        parser.error("--minimum-pass-rate must be between 0 and 1")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    source_path = arguments.input.resolve()
    output_path = arguments.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Qualitative inference artifact not found: {source_path}"
        )

    source = load_json(source_path)
    validate_source_artifact(source)
    source_sha256 = file_sha256(source_path)
    selected_source_cases = build_judgment_cases(
        source,
        cohort=arguments.cohort,
    )
    selection = judgment_selection_record(
        source,
        cohort=arguments.cohort,
        cases=selected_source_cases,
    )
    source_judgment_sha256 = source_judgment_input_sha256(
        source,
        cohort=arguments.cohort,
    )

    judge_runtime = codex_runtime_record(
        timeout_seconds=arguments.codex_timeout_seconds,
    )

    implementation_sha256 = file_sha256(Path(__file__).resolve())
    input_fingerprint = build_input_fingerprint(
        source_judgment_input_sha256=source_judgment_sha256,
        selection=selection,
        judge_runtime=judge_runtime,
        implementation_sha256=implementation_sha256,
    )
    with exclusive_output_lock(output_path):
        migrated_resume = False
        if output_path.exists():
            snapshot = load_json(output_path)
            try:
                validate_resume_snapshot(
                    snapshot,
                    source=source,
                    judge_runtime=judge_runtime,
                    implementation_sha256=implementation_sha256,
                    cohort=arguments.cohort,
                )
            except QualitativeJudgeInputError:
                if not can_migrate_semantically_identical_resume(
                    snapshot,
                    source=source,
                    judge_runtime=judge_runtime,
                    implementation_sha256=implementation_sha256,
                    cohort=arguments.cohort,
                ):
                    raise
                snapshot["run"]["input_fingerprint"] = input_fingerprint
                snapshot["run"]["implementation_sha256"] = (
                    implementation_sha256
                )
                snapshot["run"]["source_artifact"].update(
                    {
                        "sha256": source_sha256,
                        "sha256_scope": "full_file_at_last_judgment",
                        "judgment_input_sha256": source_judgment_sha256,
                    }
                )
                migrated_resume = True
        else:
            snapshot = new_snapshot(
                source=source,
                source_path=source_path,
                source_sha256=source_sha256,
                source_judgment_sha256=source_judgment_sha256,
                judge_runtime=judge_runtime,
                input_fingerprint=input_fingerprint,
                implementation_sha256=implementation_sha256,
                cohort=arguments.cohort,
            )
            atomic_write_json(output_path, snapshot)

        minimum_rate_changed = snapshot["run"].get(
            "last_minimum_pass_rate"
        ) != arguments.minimum_pass_rate
        snapshot["run"]["last_minimum_pass_rate"] = arguments.minimum_pass_rate

        def checkpoint() -> None:
            refresh_summary(snapshot)
            atomic_write_json(output_path, snapshot)

        judge_fn = build_codex_judge(judge_runtime)
        attempted = run_judgments(
            snapshot,
            judge_fn=judge_fn,
            checkpoint=checkpoint,
            piece_ids=arguments.piece,
            case_ids=arguments.case_id,
            limit=arguments.limit,
        )
        if attempted or migrated_resume or minimum_rate_changed:
            checkpoint()

    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "attempted_cases": attempted,
                "run_status": snapshot["run"]["status"],
                "summary": snapshot["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    gate_code = quality_gate_exit_code(
        snapshot,
        minimum_pass_rate=arguments.minimum_pass_rate,
    )
    if gate_code:
        rate = snapshot["summary"]["final_case_pass_rate"]
        print(
            "Qualitative answer-fidelity gate failed: "
            f"{rate:.1%} < {arguments.minimum_pass_rate:.1%}",
            file=sys.stderr,
        )
        return gate_code
    if snapshot["summary"]["judgment_statuses"].get("error", 0):
        return 1
    if snapshot["run"]["status"] == "input_incomplete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
