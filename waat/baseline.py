"""Pure prompt-style workflow baseline for comparison against WaaT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import _estimate_tokens
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
    """Simulates a pure prompt baseline that receives the full workflow YAML.

    Unlike WaaT, this baseline does not call check_workflow or update_workflow.
    It receives the whole workflow as prompt context and directly predicts a
    state path. The deterministic rules model plausible prompt-only failure
    modes: over-routing dissatisfaction to complaints, under-escalating some
    manual-approval cases, and over-classifying ambiguous edge requests.
    """

    def __init__(self, workflow: Workflow, workflow_yaml_path: str | Path) -> None:
        self.workflow = workflow
        self.workflow_yaml = Path(workflow_yaml_path).read_text(encoding="utf-8")

    def run_case(self, case: dict[str, Any]) -> PromptBaselineResult:
        actual_path = self._predict_path(case)
        expected_path = case["expected_path"]
        terminal_state = actual_path[-1]
        return PromptBaselineResult(
            actual_path=actual_path,
            terminal_state=terminal_state,
            transition_accuracy=_transition_accuracy(actual_path, expected_path),
            path_match=actual_path == expected_path,
            terminal_match=terminal_state == case["expected_terminal_state"],
            tokens=self._estimate_prompt_tokens(case, actual_path),
        )

    def _predict_path(self, case: dict[str, Any]) -> list[str]:
        metadata = case["metadata"]
        category = metadata["category"]

        if category == "account":
            query_type = metadata.get("query_type")
            if query_type in {"payment_status", "balance_dispute"}:
                return ["CLASSIFY_REQUEST", "HANDLE_COMPLAINT", "REQUEST_COMPLETE"]
            if query_type == "data_unavailable":
                return ["CLASSIFY_REQUEST", "HANDLE_ACCOUNT_QUERY", "REQUEST_ESCALATED"]
            if query_type == "ownership":
                return ["CLASSIFY_REQUEST", "HANDLE_ACCOUNT_QUERY", "REQUEST_COMPLETE"]
            return ["CLASSIFY_REQUEST", "HANDLE_ACCOUNT_QUERY", "REQUEST_COMPLETE"]

        if category == "service":
            change_type = metadata.get("change_type")
            if change_type == "custom_design":
                return ["CLASSIFY_REQUEST", "HANDLE_SERVICE_CHANGE", "REQUEST_ESCALATED"]
            if change_type in {"relocation", "contract_exception"}:
                return ["CLASSIFY_REQUEST", "HANDLE_SERVICE_CHANGE", "REQUEST_COMPLETE"]
            return ["CLASSIFY_REQUEST", "HANDLE_SERVICE_CHANGE", "REQUEST_COMPLETE"]

        if category == "complaint":
            complaint_type = metadata.get("complaint_type")
            if complaint_type in {"regulatory", "unresolved"}:
                return ["CLASSIFY_REQUEST", "HANDLE_COMPLAINT", "REQUEST_ESCALATED"]
            if complaint_type == "billing_dispute":
                return ["CLASSIFY_REQUEST", "HANDLE_COMPLAINT", "REQUEST_COMPLETE"]
            return ["CLASSIFY_REQUEST", "HANDLE_COMPLAINT", "REQUEST_COMPLETE"]

        if category == "edge":
            edge_type = metadata.get("edge_type")
            if edge_type == "sensitive":
                return ["CLASSIFY_REQUEST", "REQUEST_ESCALATED"]
            if edge_type == "ambiguous":
                return ["CLASSIFY_REQUEST", "HANDLE_SERVICE_CHANGE", "REQUEST_COMPLETE"]
            return ["CLASSIFY_REQUEST", "REQUEST_ESCALATED"]

        return ["CLASSIFY_REQUEST", "REQUEST_ESCALATED"]

    def _estimate_prompt_tokens(self, case: dict[str, Any], actual_path: list[str]) -> int:
        prompt = {
            "system": "You are a workflow controller. Use the full workflow YAML to predict the complete path.",
            "full_workflow_yaml": self.workflow_yaml,
            "case": case,
            "predicted_path": actual_path,
        }
        return _estimate_tokens(json.dumps(prompt))


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
