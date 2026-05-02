"""Super-agent loop for the Workflow as a Tool architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bedrock_client import BedrockClaudeClient
from .mock_agents import execute_action
from .tools import WorkflowSession, check_workflow, set_active_session, update_workflow
from .workflow import Workflow

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
    mean_tokens_per_step: float


class WaaTSuperAgent:
    """Runs the WaaT loop over one customer service request using Claude on Bedrock."""

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

    def run_case(self, case: dict[str, Any]) -> AgentRunResult:
        session = WorkflowSession(self.workflow)
        set_active_session(session)
        trace: list[dict[str, Any]] = []
        actual_path = [session.current_state_id]
        expected_path = case["expected_path"]
        correct_transitions = 0
        total_transitions = max(0, len(expected_path) - 1)
        total_tokens = 0

        while session.current_state_id and not self.workflow.is_terminal(session.current_state_id):
            current_state = session.current_state_id
            state_spec = check_workflow(current_state)
            action_result = execute_action(state_spec["action"], case, current_state, state_spec)

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

            step_tokens = decision["tokens"]
            total_tokens += step_tokens

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
            mean_tokens_per_step=total_tokens / steps,
        )

    def _choose_transition(
        self,
        case: dict[str, Any],
        current_state: str,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_targets = [transition["to"] for transition in state_spec.get("transitions", [])]
        system_prompt = (
            "You are a WaaT super-agent choosing the next workflow state. "
            "Use only the current state specification and action result. "
            "Return only JSON with keys next_state and reasoning. "
            "next_state must be one of the allowed transition targets. "
            "reasoning must be at least 20 words and grounded in the action result."
        )
        payload = {
            "case": {
                "customer_id": case["customer_id"],
                "request_text": case["request_text"],
            },
            "current_state_id": current_state,
            "current_state_spec": state_spec,
            "allowed_targets": allowed_targets,
            "action_result": action_result,
        }
        parsed, response = self.bedrock_client.converse_json(system_prompt, payload)
        next_state = str(parsed.get("next_state", ""))
        if next_state not in allowed_targets:
            next_state = action_result["recommended_next_state"]
        reasoning = str(parsed.get("reasoning", "")).strip()
        return {"next_state": next_state, "reasoning": reasoning, "tokens": response.total_tokens}

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
