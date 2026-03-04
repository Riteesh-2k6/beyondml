"""
AgentTrace — structured tracing for agent pipeline execution.

Records timing, inputs, outputs, and errors for every agent step,
enabling post-hoc analysis and debugging.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class TraceStep:
    """A single agent execution trace entry."""

    agent_name: str
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    status: str = "pending"  # pending, running, success, error
    input_summary: str = ""
    output_summary: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentTrace:
    """Collects structured trace steps for the pipeline execution."""

    def __init__(self):
        self.steps: List[TraceStep] = []
        self._current: Optional[TraceStep] = None

    def start(self, agent_name: str, input_summary: str = "") -> TraceStep:
        """Mark the start of an agent execution."""
        step = TraceStep(
            agent_name=agent_name,
            started_at=time.time(),
            status="running",
            input_summary=input_summary[:500],
        )
        self._current = step
        self.steps.append(step)
        return step

    def finish(
        self,
        output_summary: str = "",
        status: str = "success",
        error: Optional[str] = None,
        **metadata: Any,
    ) -> Optional[TraceStep]:
        """Mark the end of the current agent execution."""
        if self._current is None:
            return None

        self._current.finished_at = time.time()
        self._current.duration_ms = round(
            (self._current.finished_at - self._current.started_at) * 1000, 1
        )
        self._current.status = status
        self._current.output_summary = output_summary[:500]
        self._current.error = error
        self._current.metadata.update(metadata)

        finished = self._current
        self._current = None
        return finished

    def finish_error(self, error: str) -> Optional[TraceStep]:
        """Shorthand for finishing with an error."""
        return self.finish(status="error", error=error)

    def summary(self) -> List[Dict[str, Any]]:
        """Return all trace steps as a list of dicts."""
        return [asdict(s) for s in self.steps]

    def total_duration_ms(self) -> float:
        """Total duration of all completed steps."""
        return sum(s.duration_ms for s in self.steps if s.status != "pending")

    def to_json(self, indent: int = 2) -> str:
        """Serialize trace to JSON string."""
        return json.dumps(
            {
                "total_steps": len(self.steps),
                "total_duration_ms": self.total_duration_ms(),
                "steps": self.summary(),
            },
            indent=indent,
            default=str,
        )

    def print_summary(self) -> str:
        """Return a readable formatted summary of the trace."""
        lines = ["Agent Trace Summary", "=" * 50]
        for s in self.steps:
            icon = {"success": "✅", "error": "❌", "running": "🔄"}.get(s.status, "⬜")
            lines.append(
                f"  {icon} {s.agent_name:20s} {s.duration_ms:8.1f}ms  [{s.status}]"
            )
            if s.error:
                lines.append(f"     └─ Error: {s.error}")
        lines.append("=" * 50)
        lines.append(f"  Total: {self.total_duration_ms():.1f}ms")
        return "\n".join(lines)
