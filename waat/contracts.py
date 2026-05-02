"""Shared runtime contracts for WaaT agents and evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AgentActionStatus = Literal["success", "failed", "partial", "requires_user_input", "requires_manual_review"]
WorkflowRunStatus = Literal["completed", "waiting_for_user", "waiting_for_system", "failed"]


@dataclass
class AgentActionResult:
    """Production-shaped result returned by a utility agent or interaction boundary."""

    agent: str
    action: str
    status: AgentActionStatus
    data: dict[str, Any] = field(default_factory=dict)
    external_refs: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    recommended_next_state: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PendingUserRequest:
    """User-facing request created when a workflow suspends at an interaction state."""

    type: str
    state_id: str
    prompt: str
    required_fields: list[str] = field(default_factory=list)
    confirmation_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
