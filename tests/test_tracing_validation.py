"""Tests for AgentTrace and LLM validation modules."""

import pytest
import time
from beyondml.engine.tracing import AgentTrace, TraceStep
from beyondml.llm.validation import validate_llm_json


class TestAgentTrace:
    def test_start_and_finish(self):
        trace = AgentTrace()
        trace.start("EDA Agent", "shape=(200, 5)")
        time.sleep(0.01)
        step = trace.finish("target=x, insights=3")

        assert step is not None
        assert step.agent_name == "EDA Agent"
        assert step.status == "success"
        assert step.duration_ms > 0
        assert step.output_summary == "target=x, insights=3"

    def test_finish_error(self):
        trace = AgentTrace()
        trace.start("Orchestrator")
        step = trace.finish_error("LLM timed out")

        assert step.status == "error"
        assert step.error == "LLM timed out"

    def test_multiple_steps(self):
        trace = AgentTrace()
        trace.start("A")
        trace.finish("done_a")
        trace.start("B")
        trace.finish("done_b")
        trace.start("C")
        trace.finish_error("failed")

        assert len(trace.steps) == 3
        assert trace.steps[2].status == "error"
        assert trace.total_duration_ms() >= 0

    def test_summary_returns_dicts(self):
        trace = AgentTrace()
        trace.start("X")
        trace.finish("ok")

        s = trace.summary()
        assert isinstance(s, list)
        assert len(s) == 1
        assert s[0]["agent_name"] == "X"

    def test_to_json(self):
        trace = AgentTrace()
        trace.start("Y")
        trace.finish("ok")

        j = trace.to_json()
        assert '"total_steps": 1' in j

    def test_print_summary_format(self):
        trace = AgentTrace()
        trace.start("Agent1")
        trace.finish("done")

        text = trace.print_summary()
        assert "Agent1" in text
        assert "success" in text


class TestValidateLLMJson:
    def test_valid_json(self):
        raw = '{"path": "supervised", "target": "y"}'
        result = validate_llm_json(raw)
        assert result["path"] == "supervised"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        result = validate_llm_json(raw)
        assert result["key"] == "value"

    def test_required_keys_strict(self):
        raw = '{"a": 1}'
        with pytest.raises(KeyError, match="Missing required"):
            validate_llm_json(raw, required_keys=["b"], strict=True)

    def test_required_keys_lenient(self):
        raw = '{"a": 1}'
        result = validate_llm_json(raw, required_keys=["b"], strict=False)
        assert result == {"a": 1}

    def test_optional_defaults(self):
        raw = '{"a": 1}'
        result = validate_llm_json(
            raw, optional_keys={"b": "default_value", "c": 42}
        )
        assert result["b"] == "default_value"
        assert result["c"] == 42
        assert result["a"] == 1

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            validate_llm_json("not json at all")

    def test_non_object_raises(self):
        with pytest.raises(ValueError, match="not a JSON object"):
            validate_llm_json("[1, 2, 3]")
