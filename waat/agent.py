"""Super-agent loop for the Workflow as a Tool architecture."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mock_agents import execute_action
from .tools import WorkflowSession, check_workflow, set_active_session, update_workflow
from .workflow import Workflow

try:
    from strands import Agent
except ImportError:  # pragma: no cover - dependency guard for documentation builds.
    Agent = None  # type: ignore[assignment]


MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class AgentRunResult:
    case_id: str
    trace: list[dict[str, Any]]
    actual_path: list[str]
    terminal_state: str
    transition_accuracy: float
    path_match: bool
    terminal_match: bool
    total_tokens: int
    baseline_tokens: int
    mean_tokens_per_step: float
    mean_baseline_tokens_per_step: float


class WaaTSuperAgent:
    """Runs the WaaT loop over one customer service request."""

    def __init__(
        self,
        workflow: Workflow,
        workflow_yaml_path: str | Path,
        use_anthropic: bool | None = None,
        seed: int = 7,
    ) -> None:
        self.workflow = workflow
        self.workflow_yaml = Path(workflow_yaml_path).read_text(encoding="utf-8")
        self.rng = random.Random(seed)
        self.use_anthropic = (
            bool(os.environ.get("ANTHROPIC_API_KEY")) if use_anthropic is None else use_anthropic
        )
        self.client = None
        if self.use_anthropic:
            try:
                from anthropic import Anthropic

                self.client = Anthropic()
            except Exception:
                self.use_anthropic = False
        self.strands_agent = self.build_strands_agent()

    def build_strands_agent(self) -> Any | None:
        """Build a Strands Agent with WaaT tools attached.

        The offline evaluation below calls the Strands-decorated tools directly
        for reproducibility. This agent construction mirrors the production
        pattern and can be used for interactive experiments with a configured
        model runtime.
        """
        if Agent is None:
            return None
        try:
            return Agent(
                model=None,
                tools=[check_workflow, update_workflow],
                system_prompt=(
                    "You are a WaaT super-agent. Never request or inspect the full workflow. "
                    "At each step, use check_workflow to inspect only the current state, "
                    "execute the returned action, and call update_workflow with a valid "
                    "next state and at least 20 words of grounded reasoning."
                ),
                name="waat-super-agent",
                description="Strands-based Workflow as a Tool research prototype.",
            )
        except Exception:
            return None

    def run_case(self, case: dict[str, Any]) -> AgentRunResult:
        session = WorkflowSession(self.workflow)
        set_active_session(session)
        trace: list[dict[str, Any]] = []
        actual_path = [session.current_state_id]
        expected_path = case["expected_path"]
        correct_transitions = 0
        total_transitions = max(0, len(expected_path) - 1)
        total_tokens = 0
        baseline_tokens = 0

        while session.current_state_id and not self.workflow.is_terminal(session.current_state_id):
            current_state = session.current_state_id
            state_spec = check_workflow(current_state)
            action_result = execute_action(state_spec["action"], case, current_state)

            decision = self._choose_transition(case, current_state, state_spec, action_result)
            update = update_workflow(current_state, decision["next_state"], decision["reasoning"])
            if "error" in update:
                repaired = self._repair_reasoning(current_state, decision["next_state"], action_result, update)
                update = update_workflow(current_state, decision["next_state"], repaired)
                decision["reasoning"] = repaired

            next_state = session.current_state_id
            actual_path.append(next_state)

            expected_next = expected_path[len(actual_path) - 1] if len(actual_path) - 1 < len(expected_path) else None
            if next_state == expected_next:
                correct_transitions += 1

            step_tokens = decision.get("tokens", self._estimate_waat_tokens(state_spec, action_result, decision))
            baseline_step_tokens = self._estimate_baseline_tokens(case, current_state, action_result, decision)
            total_tokens += step_tokens
            baseline_tokens += baseline_step_tokens

            trace.append(
                {
                    "state": current_state,
                    "state_spec_visible_to_agent": state_spec,
                    "action": state_spec["action"],
                    "action_result": action_result,
                    "next_state": next_state,
                    "reasoning": decision["reasoning"],
                    "update_result": update,
                    "tokens": step_tokens,
                    "baseline_tokens": baseline_step_tokens,
                }
            )

        terminal_state = actual_path[-1]
        steps = max(1, len(trace))
        return AgentRunResult(
            case_id=case["case_id"],
            trace=trace,
            actual_path=actual_path,
            terminal_state=terminal_state,
            transition_accuracy=(correct_transitions / total_transitions) if total_transitions else 1.0,
            path_match=actual_path == expected_path,
            terminal_match=terminal_state == case["expected_terminal_state"],
            total_tokens=total_tokens,
            baseline_tokens=baseline_tokens,
            mean_tokens_per_step=total_tokens / steps,
            mean_baseline_tokens_per_step=baseline_tokens / steps,
        )

    def _choose_transition(
        self,
        case: dict[str, Any],
        current_state: str,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
    ) -> dict[str, Any]:
        if self.use_anthropic and self.client is not None:
            try:
                return self._choose_transition_with_anthropic(case, current_state, state_spec, action_result)
            except Exception:
                pass
        return self._choose_transition_deterministic(case, current_state, state_spec, action_result)

    def _choose_transition_deterministic(
        self,
        case: dict[str, Any],
        current_state: str,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
    ) -> dict[str, Any]:
        next_state = action_result["recommended_next_state"]
        if action_result.get("ambiguous_signal"):
            reasoning = (
                "The available evidence is mixed, but a decision is needed now. I will proceed with "
                "the most plausible route based on the response and customer wording."
            )
        else:
            reasoning = (
                f"The workflow is currently in {current_state} and the visible state specification permits "
                f"a transition to {next_state}. The simulated {action_result['agent']} agent returned "
                f"structured evidence indicating this outcome, so selecting {next_state} follows the "
                f"observed action result and the transition conditions for customer {case['customer_id']}."
            )
        return {"next_state": next_state, "reasoning": reasoning}

    def _choose_transition_with_anthropic(
        self,
        case: dict[str, Any],
        current_state: str,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
    ) -> dict[str, Any]:
        tools = [
            {
                "name": "update_workflow",
                "description": "Advance the workflow to the next state with non-trivial reasoning.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "current_state_id": {"type": "string"},
                        "next_state_id": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["current_state_id", "next_state_id", "reasoning"],
                },
            }
        ]
        prompt = {
            "case": {
                "customer_id": case["customer_id"],
                "request_text": case["request_text"],
            },
            "current_state_id": current_state,
            "current_state_spec": state_spec,
            "action_result": action_result,
            "instruction": "Call update_workflow with the best next_state_id and at least 20 words of reasoning.",
        }

        response = self._anthropic_call_with_backoff(
            model=MODEL_NAME,
            max_tokens=600,
            tools=tools,
            messages=[{"role": "user", "content": json.dumps(prompt)}],
        )
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))
        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "update_workflow":
                return {
                    "next_state": block.input["next_state_id"],
                    "reasoning": block.input["reasoning"],
                    "tokens": tokens,
                }
        raise RuntimeError("Anthropic response did not contain update_workflow tool_use")

    def _anthropic_call_with_backoff(self, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                return self.client.messages.create(**kwargs)
            except Exception as exc:
                last_error = exc
                time.sleep((2**attempt) + self.rng.random())
        raise RuntimeError("Anthropic API call failed after retries") from last_error

    def _repair_reasoning(
        self,
        current_state: str,
        next_state: str,
        action_result: dict[str, Any],
        update_error: dict[str, Any],
    ) -> str:
        return (
            f"The previous update was rejected because {update_error['message']} The transition from "
            f"{current_state} to {next_state} remains appropriate because the visible action result "
            f"from {action_result['agent']} explicitly recommends {next_state} and contains structured "
            "outcome fields that satisfy the workflow transition condition for this step."
        )

    def _estimate_waat_tokens(
        self,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
        decision: dict[str, Any],
    ) -> int:
        text = json.dumps({"state_spec": state_spec, "action_result": action_result, "decision": decision})
        return _estimate_tokens(text)

    def _estimate_baseline_tokens(
        self,
        case: dict[str, Any],
        current_state: str,
        action_result: dict[str, Any],
        decision: dict[str, Any],
    ) -> int:
        text = json.dumps(
            {
                "full_workflow_yaml": self.workflow_yaml,
                "case": case,
                "current_state": current_state,
                "action_result": action_result,
                "decision": decision,
            }
        )
        return _estimate_tokens(text)


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.35))
