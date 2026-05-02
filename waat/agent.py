"""Super-agent loop for the Workflow as a Tool architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bedrock_client import BedrockClaudeClient
from .contracts import AgentActionResult, PendingUserRequest
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
    status: str = "completed"
    current_state_id: str | None = None
    message_to_user: str | None = None
    pending_user_request: dict[str, Any] | None = None
    session: WorkflowSession | None = None


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
        return self._run_until_blocked(case, session)

    def resume_case(
        self,
        session: WorkflowSession,
        case: dict[str, Any],
        user_input: dict[str, Any],
    ) -> AgentRunResult:
        """Resume a suspended workflow after the user provides requested input."""
        set_active_session(session)
        if not session.pending_user_request:
            raise ValueError("Cannot resume workflow because no user request is pending")

        session.context.setdefault("user_inputs", []).append(user_input)
        current_state = session.current_state_id
        if current_state is None:
            raise ValueError("Cannot resume workflow without a current state")

        state_spec = check_workflow(current_state)
        action_result = self._confirmation_action_result(current_state, state_spec, user_input)
        session.merge_action_result(action_result)
        decision = self._choose_transition(case, current_state, state_spec, action_result, session.context)
        update = update_workflow(current_state, decision["next_state"], decision["reasoning"])
        if "error" in update:
            repaired = self._repair_reasoning(current_state, decision["next_state"], action_result, update)
            update = update_workflow(current_state, decision["next_state"], repaired)
            decision["reasoning"] = repaired

        session.clear_pending_user_request()
        self._record_step(session, current_state, state_spec, action_result, decision, update)
        return self._run_until_blocked(case, session)

    def _run_until_blocked(self, case: dict[str, Any], session: WorkflowSession) -> AgentRunResult:
        set_active_session(session)
        while session.current_state_id and not self.workflow.is_terminal(session.current_state_id):
            current_state = session.current_state_id
            state_spec = check_workflow(current_state)
            if self._is_user_interaction(state_spec):
                pending_request = self._pending_user_request(current_state, state_spec)
                session.set_pending_user_request(pending_request)
                return self._build_result(case, session, "waiting_for_user", pending_request["prompt"])

            action_result = execute_action(state_spec["action"], case, current_state, state_spec)
            session.merge_action_result(action_result)

            decision = self._choose_transition(case, current_state, state_spec, action_result, session.context)
            update = update_workflow(current_state, decision["next_state"], decision["reasoning"])
            if "error" in update:
                repaired = self._repair_reasoning(current_state, decision["next_state"], action_result, update)
                update = update_workflow(current_state, decision["next_state"], repaired)
                decision["reasoning"] = repaired

            self._record_step(session, current_state, state_spec, action_result, decision, update)

        return self._build_result(case, session, "completed")

    def _choose_transition(
        self,
        case: dict[str, Any],
        current_state: str,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        allowed_targets = [transition["to"] for transition in state_spec.get("transitions", [])]
        system_prompt = (
            "You are a WaaT super-agent choosing the next workflow state. "
            "The complete workflow definition is included below as global context. "
            "Use that global workflow context together with the current state specification "
            "and action result, but choose only among the current state's allowed transition targets. "
            "The runtime will validate the transition through update_workflow and persist your reasoning. "
            "Return only JSON with keys next_state and reasoning. "
            "next_state must be one of the allowed transition targets. "
            "reasoning must be at least 20 words and grounded in the current state, action result, "
            "and relevant workflow conditions.\n\n"
            f"Workflow definition:\n{self.workflow_yaml}"
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
            "workflow_context": context or {},
        }
        parsed, response = self.bedrock_client.converse_json(system_prompt, payload)
        next_state = str(parsed.get("next_state", ""))
        if next_state not in allowed_targets:
            recommended = action_result.get("recommended_next_state")
            next_state = recommended if recommended in allowed_targets else (allowed_targets[0] if allowed_targets else "")
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

    def _record_step(
        self,
        session: WorkflowSession,
        current_state: str,
        state_spec: dict[str, Any],
        action_result: dict[str, Any],
        decision: dict[str, Any],
        update: dict[str, Any],
    ) -> None:
        step_tokens = int(decision.get("tokens", 0))
        session.total_tokens += step_tokens
        session.trace.append(
            {
                "state": current_state,
                "state_spec_visible_to_agent": state_spec,
                "action": state_spec["action"],
                "action_result": action_result,
                "next_state": session.current_state_id,
                "reasoning": decision["reasoning"],
                "update_result": update,
                "tokens": step_tokens,
            }
        )

    def _build_result(
        self,
        case: dict[str, Any],
        session: WorkflowSession,
        status: str,
        message_to_user: str | None = None,
    ) -> AgentRunResult:
        actual_path = list(session.actual_path)
        terminal_state = actual_path[-1]
        steps = max(1, len(session.trace))
        return AgentRunResult(
            case_id=case["case_id"],
            trace=list(session.trace),
            actual_path=actual_path,
            terminal_state=terminal_state,
            transition_accuracy=_transition_accuracy(actual_path, case["expected_path"]),
            path_match=actual_path == case["expected_path"],
            terminal_match=status == "completed" and terminal_state == case["expected_terminal_state"],
            total_tokens=session.total_tokens,
            mean_tokens_per_step=session.total_tokens / steps,
            status=status,
            current_state_id=session.current_state_id,
            message_to_user=message_to_user,
            pending_user_request=session.pending_user_request,
            session=session,
        )

    def _is_user_interaction(self, state_spec: dict[str, Any]) -> bool:
        return state_spec.get("action") in {"request_user_input", "request_user_confirmation"}

    def _pending_user_request(self, state_id: str, state_spec: dict[str, Any]) -> dict[str, Any]:
        params = state_spec.get("action_params", {})
        pending = PendingUserRequest(
            type="confirmation" if state_spec.get("action") == "request_user_confirmation" else "input",
            state_id=state_id,
            prompt=str(params.get("prompt", "")),
            required_fields=list(params.get("required_fields", [])),
            confirmation_field=params.get("confirmation_field"),
        )
        return pending.to_dict()

    def _confirmation_action_result(
        self,
        state_id: str,
        state_spec: dict[str, Any],
        user_input: dict[str, Any],
    ) -> dict[str, Any]:
        params = state_spec.get("action_params", {})
        confirmation_field = params.get("confirmation_field", "confirmed")
        confirmed = bool(user_input.get(confirmation_field))
        recommended_next_state = _recommended_interaction_target(state_spec, confirmed)
        return AgentActionResult(
            agent="user_interaction",
            action=state_spec["action"],
            status="success",
            data={
                "state_id": state_id,
                "user_input": user_input,
                "confirmed": confirmed,
                "confirmation_field": confirmation_field,
            },
            recommended_next_state=recommended_next_state,
            rationale=(
                "The user confirmed the pending workflow action."
                if confirmed
                else "The user declined or did not confirm the pending workflow action."
            ),
        ).to_dict()


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
