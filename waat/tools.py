"""Workflow-as-a-Tool implementations exposed to the super-agent."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .workflow import Workflow

try:
    from strands import tool
except ImportError:  # pragma: no cover - only used when Strands is unavailable.
    def tool(func: Any) -> Any:
        return func


MIN_REASONING_WORDS = 20
_ACTIVE_SESSION: ContextVar["WorkflowSession | None"] = ContextVar("waat_active_session", default=None)


@dataclass
class WorkflowSession:
    """Stateful workflow session used by check_workflow and update_workflow."""

    workflow: Workflow
    current_state_id: str | None = None
    updates: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.current_state_id is None:
            self.current_state_id = self.workflow.initial_state

    def check_workflow(self, state_id: str) -> dict[str, Any]:
        """Returns the current state spec only: action, action_params, transitions."""
        if state_id != self.current_state_id:
            return {
                "error": "state_mismatch",
                "message": f"Expected current state {self.current_state_id}, received {state_id}",
            }
        if self.workflow.is_terminal(state_id):
            return {"state_id": state_id, "terminal": True, "action": None, "action_params": {}, "transitions": []}
        return {"state_id": state_id, **self.workflow.get_state(state_id)}

    def update_workflow(self, current_state_id: str, next_state_id: str, reasoning: str) -> dict[str, Any]:
        """Advance the workflow with mandatory, non-trivial reasoning."""
        word_count = len([word for word in reasoning.split() if word.strip()])
        if word_count < MIN_REASONING_WORDS:
            return {
                "error": "reasoning_too_short",
                "message": (
                    f"Reasoning must contain at least {MIN_REASONING_WORDS} words; "
                    f"received {word_count}."
                ),
            }
        if current_state_id != self.current_state_id:
            return {
                "error": "state_mismatch",
                "message": f"Current state is {self.current_state_id}, not {current_state_id}.",
            }
        if self.workflow.is_terminal(current_state_id):
            return {"error": "terminal_state", "message": f"{current_state_id} is already terminal."}

        allowed_targets = self.workflow.transition_targets(current_state_id)
        if next_state_id not in allowed_targets:
            return {
                "error": "invalid_transition",
                "message": f"Cannot transition from {current_state_id} to {next_state_id}.",
                "allowed_targets": sorted(allowed_targets),
            }

        self.current_state_id = next_state_id
        update = {
            "from_state": current_state_id,
            "to_state": next_state_id,
            "reasoning": reasoning,
            "reasoning_word_count": word_count,
        }
        self.updates.append(update)
        return {"status": "ok", "current_state_id": self.current_state_id, **update}


def set_active_session(session: WorkflowSession) -> None:
    """Bind a WorkflowSession for exact-signature tool calls."""
    _ACTIVE_SESSION.set(session)


@tool
def check_workflow(state_id: str) -> dict[str, Any]:
    """Returns the current state spec: action, action_params, and transitions only."""
    session = _require_session()
    return session.check_workflow(state_id)


@tool
def update_workflow(current_state_id: str, next_state_id: str, reasoning: str) -> dict[str, Any]:
    """Advance the workflow, requiring non-trivial reasoning."""
    session = _require_session()
    return session.update_workflow(current_state_id, next_state_id, reasoning)


def _require_session() -> WorkflowSession:
    session = _ACTIVE_SESSION.get()
    if session is None:
        raise RuntimeError("No active WorkflowSession is bound for WaaT tools")
    return session
