"""Stepwise prompt-style workflow baseline for comparison against WaaT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bedrock_client import BedrockClaudeClient
from .contracts import AgentActionResult
from .mock_agents import execute_action
from .workflow import Workflow


@dataclass
class PromptBaselineResult:
    """Result from the prompt-guided stepwise workflow baseline."""

    actual_path: list[str]
    terminal_state: str
    transition_accuracy: float
    path_match: bool
    terminal_match: bool
    tokens: int
    trace: list[dict[str, Any]]


class PromptWorkflowBaseline:
    """Prompt baseline with the full workflow definition in instruction context."""

    def __init__(
        self,
        workflow: Workflow,
        workflow_yaml_path: str | Path,
        bedrock_profile: str | None = None,
        bedrock_region: str | None = None,
        bedrock_model_id: str | None = None,
        bedrock_verify_ssl: bool | str = True,
    ) -> None:
        self.workflow = workflow
        self.workflow_yaml = Path(workflow_yaml_path).read_text(encoding="utf-8")
        self.bedrock_client = BedrockClaudeClient(
            profile_name=bedrock_profile,
            region_name=bedrock_region,
            model_id=bedrock_model_id,
            verify_ssl=bedrock_verify_ssl,
        )

    def run_case(self, case: dict[str, Any]) -> PromptBaselineResult:
        actual_path = [self.workflow.initial_state]
        current_state = self.workflow.initial_state
        context: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        total_tokens = 0
        max_steps = len(self.workflow.states) + len(self.workflow.terminal_states) + 5

        for _ in range(max_steps):
            if self.workflow.is_terminal(current_state):
                break
            if current_state not in self.workflow.states:
                break

            state_spec = {"state_id": current_state, **self.workflow.get_state(current_state)}
            action_result = self._execute_or_interact(state_spec, case, context)
            context["last_action_result"] = action_result
            decision = self._choose_transition(case, current_state, state_spec, action_result, context)
            next_state = decision["next_state"]
            actual_path.append(next_state)
            total_tokens += decision["tokens"]
            trace.append(
                {
                    "state": current_state,
                    "state_spec_visible_to_baseline": state_spec,
                    "action": state_spec["action"],
                    "action_result": action_result,
                    "next_state": next_state,
                    "reasoning": decision["reasoning"],
                    "tokens": decision["tokens"],
                }
            )
            current_state = next_state

        terminal_state = actual_path[-1]
        expected_path = case["expected_path"]
        return PromptBaselineResult(
            actual_path=actual_path,
            terminal_state=terminal_state,
            transition_accuracy=_transition_accuracy(actual_path, expected_path),
            path_match=actual_path == expected_path,
            terminal_match=terminal_state == case["expected_terminal_state"],
            tokens=total_tokens,
            trace=trace,
        )

    def _choose_transition(
        self,
        case: dict[str, Any],
        current_state: str,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_targets = [transition["to"] for transition in state_spec.get("transitions", [])]
        system_prompt = (
            "You are a prompt-guided workflow controller baseline. The complete workflow definition is "
            "included below as static instruction context. Proceed one step at a time, using the current "
            "state, case context, and utility-agent output to choose the next state. You do not have access "
            "to check_workflow or update_workflow, so no external runtime will validate your transition. "
            "Return only JSON with keys next_state and reasoning.\n\n"
            f"Workflow definition:\n{self.workflow_yaml}"
        )
        payload = {
            "case": {
                "customer_id": case["customer_id"],
                "request_text": case["request_text"],
            },
            "current_state_id": current_state,
            "current_state_spec": state_spec,
            "allowed_targets_for_scoring": allowed_targets,
            "action_result": action_result,
            "workflow_context": context,
        }
        parsed, response = self.bedrock_client.converse_json(system_prompt, payload)
        next_state = str(parsed.get("next_state", ""))
        if not next_state:
            recommended = action_result.get("recommended_next_state")
            next_state = recommended if recommended in allowed_targets else (allowed_targets[0] if allowed_targets else "")
        reasoning = str(parsed.get("reasoning", "")).strip()
        return {"next_state": next_state, "reasoning": reasoning, "tokens": response.total_tokens}

    def _execute_or_interact(
        self,
        state_spec: dict[str, Any],
        case: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        action = state_spec["action"]
        if action == "request_user_confirmation":
            user_input = _next_user_turn(case, context)
            confirmation_field = state_spec.get("action_params", {}).get("confirmation_field", "confirmed")
            confirmed = bool(user_input.get(confirmation_field))
            return AgentActionResult(
                agent="user_interaction",
                action=action,
                status="success",
                data={
                    "user_input": user_input,
                    "confirmed": confirmed,
                    "confirmation_field": confirmation_field,
                },
                recommended_next_state=_recommended_interaction_target(state_spec, confirmed),
                rationale="The baseline received the user's response to the confirmation prompt.",
            ).to_dict()
        return execute_action(action, case, str(state_spec["state_id"]), state_spec)


def _next_user_turn(case: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    turns = list(case.get("user_turns", []))
    index = int(context.get("user_turn_index", 0))
    context["user_turn_index"] = index + 1
    if index < len(turns):
        return dict(turns[index])
    return {}


def _recommended_interaction_target(state_spec: dict[str, Any], confirmed: bool) -> str:
    transitions = state_spec.get("transitions", [])
    if not transitions:
        return "REQUEST_ESCALATED"
    if confirmed:
        return transitions[0]["to"]
    if len(transitions) > 1:
        return transitions[1]["to"]
    return transitions[0]["to"]


def _transition_accuracy(actual_path: list[str], expected_path: list[str]) -> float:
    expected_transitions = list(zip(expected_path, expected_path[1:]))
    actual_transitions = list(zip(actual_path, actual_path[1:]))
    if not expected_transitions:
        return 1.0
    correct = 0
    for index, expected_transition in enumerate(expected_transitions):
        if index < len(actual_transitions) and actual_transitions[index] == expected_transition:
            correct += 1
    return correct / len(expected_transitions)
