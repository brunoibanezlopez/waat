"""Pure prompt-style workflow baseline for comparison against WaaT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bedrock_client import BedrockClaudeClient
from .workflow import Workflow


@dataclass
class PromptBaselineResult:
    """Result from the pure workflow-as-prompt baseline."""

    actual_path: list[str]
    terminal_state: str
    transition_accuracy: float
    path_match: bool
    terminal_match: bool
    tokens: int


class PromptWorkflowBaseline:
    """Pure prompt baseline using Claude on Bedrock with the full workflow YAML."""

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
        actual_path, tokens = self._predict_path(case)
        expected_path = case["expected_path"]
        terminal_state = actual_path[-1]
        return PromptBaselineResult(
            actual_path=actual_path,
            terminal_state=terminal_state,
            transition_accuracy=_transition_accuracy(actual_path, expected_path),
            path_match=actual_path == expected_path,
            terminal_match=terminal_state == case["expected_terminal_state"],
            tokens=tokens,
        )

    def _predict_path(self, case: dict[str, Any]) -> tuple[list[str], int]:
        system_prompt = (
            "You are a workflow controller baseline. You receive the full workflow YAML "
            "as prompt context and must predict the full state path for the case. "
            "Return only JSON with key actual_path, whose value is a list of state IDs."
        )
        payload = {
            "full_workflow_yaml": self.workflow_yaml,
            "case": {
                "customer_id": case["customer_id"],
                "request_text": case["request_text"],
            },
        }
        parsed, response = self.bedrock_client.converse_json(system_prompt, payload)
        actual_path = parsed.get("actual_path")
        if not isinstance(actual_path, list) or not actual_path:
            raise ValueError(f"Invalid baseline path from Bedrock: {actual_path!r}")
        cleaned_path = [str(state_id) for state_id in actual_path]
        return cleaned_path, response.total_tokens


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
