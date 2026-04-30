"""Workflow loading and state graph primitives for WaaT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


@dataclass(frozen=True)
class Workflow:
    """Parsed workflow definition."""

    name: str
    initial_state: str
    terminal_states: set[str]
    states: dict[str, dict[str, Any]]

    def is_terminal(self, state_id: str) -> bool:
        return state_id in self.terminal_states

    def get_state(self, state_id: str) -> dict[str, Any]:
        if state_id not in self.states:
            raise KeyError(f"Unknown state: {state_id}")
        state = self.states[state_id]
        return {
            "action": state["action"],
            "action_params": dict(state.get("action_params", {})),
            "transitions": list(state.get("transitions", [])),
        }

    def transition_targets(self, state_id: str) -> set[str]:
        if self.is_terminal(state_id):
            return set()
        return {transition["to"] for transition in self.get_state(state_id)["transitions"]}


def load_workflow(path: str | Path) -> Workflow:
    """Load a workflow YAML file using the schema specified in the paper prompt."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load workflow YAML") from _YAML_IMPORT_ERROR

    workflow_path = Path(path)
    raw = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    required = {"workflow", "initial_state", "terminal_states", "states"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Workflow file is missing required keys: {sorted(missing)}")

    workflow = Workflow(
        name=raw["workflow"],
        initial_state=raw["initial_state"],
        terminal_states=set(raw["terminal_states"]),
        states=raw["states"],
    )
    _validate_workflow(workflow)
    return workflow


def _validate_workflow(workflow: Workflow) -> None:
    if workflow.initial_state not in workflow.states:
        raise ValueError(f"Initial state {workflow.initial_state!r} is not in states")

    for state_id, state in workflow.states.items():
        for key in ("action", "action_params", "transitions"):
            if key not in state:
                raise ValueError(f"State {state_id!r} is missing {key!r}")
        for transition in state["transitions"]:
            target = transition.get("to")
            if target not in workflow.states and target not in workflow.terminal_states:
                raise ValueError(f"State {state_id!r} has transition to unknown target {target!r}")
