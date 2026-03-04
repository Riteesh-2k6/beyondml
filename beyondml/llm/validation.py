"""
LLM output validation — lightweight schema checking for agent JSON responses.

Validates that LLM JSON outputs contain expected keys and value types
without requiring heavy dependencies like Pydantic.
"""

import json
from typing import Dict, Any, List, Optional, Type


def validate_llm_json(
    raw_response: str,
    required_keys: Optional[List[str]] = None,
    optional_keys: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Parse and validate an LLM JSON response.

    Args:
        raw_response: Raw string from LLM (may have markdown fences).
        required_keys: Keys that MUST be present.
        optional_keys: Keys with default values to fill if missing.
        strict: If True, raise on missing required keys. If False,
                return partial result with defaults filled in.

    Returns:
        Parsed and validated dict.

    Raises:
        ValueError: If response is not valid JSON.
        KeyError: If strict=True and required keys are missing.
    """
    # Strip markdown code fences if present
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"LLM output is not a JSON object, got {type(parsed).__name__}")

    # Check required keys
    if required_keys:
        missing = [k for k in required_keys if k not in parsed]
        if missing and strict:
            raise KeyError(f"Missing required keys in LLM response: {missing}")

    # Fill optional keys with defaults
    if optional_keys:
        for key, default in optional_keys.items():
            if key not in parsed:
                parsed[key] = default

    return parsed


# ── Pre-built schemas for each agent ──

ORCHESTRATOR_SCHEMA = {
    "required": ["path", "suggested_target", "model_recommendations", "reasoning"],
    "defaults": {
        "path": "supervised",
        "suggested_target": None,
        "model_recommendations": ["RandomForest"],
        "reasoning": "",
    },
}

OUTLIER_SCHEMA = {
    "required": ["recommended_strategy"],
    "defaults": {
        "recommended_strategy": "cap",
        "reasoning": "Default: cap outliers at IQR bounds",
        "per_column_notes": {},
    },
}

REFLECTION_SCHEMA = {
    "required": ["reasoning"],
    "defaults": {
        "reasoning": "",
        "features_to_drop": [],
        "new_features": [],
        "next_model": None,
        "next_ga_generations": 5,
        "next_ga_pop_size": 10,
    },
}

FEATURE_SCHEMA = {
    "required": [],
    "defaults": {
        "features": [],
    },
}
